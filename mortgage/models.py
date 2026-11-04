"""Small transformer training and inference."""

import copy
import math
import time
from pathlib import Path
from typing import Required, TypedDict

import numpy as np
import torch
from torch import nn

from .data import FloatArray, IntArray, LoanData
from .evaluate import macro_f1
from .schema import LABELS


class TransformerArchitecture(TypedDict, total=False):
    dynamic_size: Required[int]
    static_size: Required[int]
    history: int
    hidden: int
    heads: int
    layers: int
    dropout: float
    static_context: bool
    transition_residual: bool
    state_embedding: bool


def transition_probabilities(
    current: IntArray, labels: IntArray, absorbing: bool = False
) -> FloatArray:
    counts = np.full((len(LABELS), len(LABELS)), 0.5, dtype=np.float64)
    np.add.at(counts, (current, labels), 1)
    probabilities = (counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)
    if absorbing:
        probabilities[4:] = np.eye(len(LABELS), dtype=np.float32)[4:]
    return probabilities


class MortgageTransformer(nn.Module):
    transition_log_probs: torch.Tensor | None

    def __init__(
        self,
        dynamic_size: int,
        static_size: int,
        history: int = 12,
        hidden: int = 64,
        heads: int = 4,
        layers: int = 2,
        dropout: float = 0.1,
        static_context: bool = False,
        transition_residual: bool = False,
        state_embedding: bool = False,
    ):
        super().__init__()
        self.projection = nn.Linear(dynamic_size, hidden)
        self.position = nn.Parameter(torch.empty(1, history, hidden))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            hidden, heads, hidden * 2, dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden + static_size, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(LABELS)),
        )
        # Construct after the existing layers so shared weights retain their
        # initialization for a given seed. Disabled models keep legacy state keys.
        self.static_projection = (
            nn.Linear(static_size, hidden) if static_context else None
        )
        self.state_embedding = nn.Embedding(4, hidden) if state_embedding else None
        self.register_buffer(
            "transition_log_probs",
            torch.zeros(len(LABELS), len(LABELS)) if transition_residual else None,
        )
        if transition_residual:
            # The initial prediction is exactly the training-only transition prior.
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def forward(
        self,
        dynamic: torch.Tensor,
        static: torch.Tensor,
        states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # All tokens precede the target month; no future tokens or target token exist.
        tokens = self.projection(dynamic) + self.position[:, : dynamic.shape[1]]
        if self.state_embedding is not None or self.transition_log_probs is not None:
            if states is None:
                raise ValueError("This architecture requires observed history states")
        if self.state_embedding is not None:
            assert states is not None
            tokens = tokens + self.state_embedding(states)
        if self.static_projection is not None:
            tokens = torch.cat(
                [self.static_projection(static).unsqueeze(1), tokens], dim=1
            )
        encoded = self.encoder(tokens)
        logits = self.classifier(torch.cat([encoded[:, -1], static], dim=-1))
        if self.transition_log_probs is not None:
            assert states is not None
            logits = logits + self.transition_log_probs[states[:, -1]]
        return logits


def learning_rate_multiplier(step: int, total_steps: int, warmup_steps: int) -> float:
    """One-based update schedule: linear warmup, then cosine decay to 5%."""
    if not 0 <= warmup_steps < total_steps or not 1 <= step <= total_steps:
        raise ValueError("Invalid warmup, update, or total-step count")
    if step <= warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.05 + 0.95 * (1 + math.cos(math.pi * progress)) / 2


def predict_transformer(
    model: MortgageTransformer,
    data: LoanData,
    indices: IntArray,
    batch_size: int,
    device: torch.device,
    history: int,
    state_only: bool = False,
) -> FloatArray:
    model.eval()
    probabilities = np.empty((len(indices), len(LABELS)), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            dynamic, static, _ = data.batch(batch, history, state_only)
            states = (
                torch.from_numpy(data.history_states(batch, history)).to(device)
                if model.state_embedding is not None
                or model.transition_log_probs is not None
                else None
            )
            logits = model(dynamic.to(device), static.to(device), states)
            probabilities[start : start + len(batch)] = logits.softmax(-1).cpu().numpy()
    return probabilities


def train_transformer(
    data: LoanData,
    train: IntArray,
    validation: IntArray,
    output: Path,
    seed: int,
    epochs: int = 10,
    patience: int = 3,
    batch_size: int = 512,
    hidden: int = 64,
    heads: int = 4,
    layers: int = 2,
    learning_rate: float = 1e-3,
    weighted: bool = False,
    history: int = 12,
    device_name: str = "cpu",
    state_only: bool = False,
    static_context: bool = False,
    max_steps: int = 0,
    validate_every: int = 400,
    dual_checkpoints: bool = False,
    lr_schedule: str = "constant",
    warmup_steps: int = 300,
    transition_residual: bool = False,
    state_embedding: bool = False,
) -> tuple[MortgageTransformer, list[dict[str, float]]]:
    if max_steps < 0 or validate_every < 1:
        raise ValueError(
            "Step budget must be nonnegative and validation cadence positive"
        )
    if max_steps and len(train) < batch_size:
        raise ValueError("Fixed-step training requires at least one full minibatch")
    if lr_schedule not in ["constant", "warmup_cosine"]:
        raise ValueError("Unknown learning-rate schedule")
    if lr_schedule != "constant" and (
        max_steps < 1 or not 0 <= warmup_steps < max_steps
    ):
        raise ValueError(
            "Warmup/cosine requires a fixed step budget greater than warmup"
        )
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    architecture: TransformerArchitecture = {
        "dynamic_size": 4 if state_only else data.dynamic.shape[1],
        "static_size": data.static.shape[1],
        "history": history,
        "hidden": hidden,
        "heads": heads,
        "layers": layers,
        "static_context": static_context,
        "transition_residual": transition_residual,
        "state_embedding": state_embedding,
    }
    model = MortgageTransformer(**architecture).to(device)
    if model.transition_log_probs is not None:
        prior = transition_probabilities(data.current[train], data.labels[train])
        model.transition_log_probs.copy_(torch.from_numpy(np.log(prior)).to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    counts = np.bincount(data.labels[train], minlength=len(LABELS))
    weights = np.zeros(len(LABELS), dtype=np.float32)
    present = counts > 0
    weights[present] = np.sqrt(counts.sum() / (present.sum() * counts[present]))
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, device=device) if weighted else None
    )
    best_loss = float("inf")
    best_f1 = -1.0
    best_state = copy.deepcopy(model.state_dict())
    stale = 0
    records = []
    generator = np.random.default_rng(seed)
    epoch = step = examples_seen = interval_examples = 0
    loss_sum = 0.0
    started = time.perf_counter()
    stop = False
    while not stop and (step < max_steps if max_steps else epoch < epochs):
        epoch += 1
        model.train()
        order = generator.permutation(train)
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            if max_steps and len(batch) < batch_size:
                break  # Equal update budgets also see exactly equal example counts.
            dynamic, static, labels = data.batch(batch, history, state_only)
            if lr_schedule == "warmup_cosine":
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate * learning_rate_multiplier(
                        step + 1, max_steps, warmup_steps
                    )
            states = (
                torch.from_numpy(data.history_states(batch, history)).to(device)
                if state_embedding or transition_residual
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(dynamic.to(device), static.to(device), states), labels.to(device)
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
            step += 1
            examples_seen += len(batch)
            interval_examples += len(batch)
            due = (
                (step % validate_every == 0 or step == max_steps)
                if max_steps
                else start + batch_size >= len(order)
            )
            if not due:
                continue
            probabilities = predict_transformer(
                model, data, validation, batch_size, device, history, state_only
            )
            validation_loss = float(
                -np.log(
                    np.clip(
                        probabilities[
                            np.arange(len(validation)), data.labels[validation]
                        ],
                        1e-15,
                        1,
                    )
                ).mean()
            )
            validation_f1 = macro_f1(data.labels[validation], probabilities)
            record = {
                "epoch": epoch,
                "step": step,
                "examples_seen": examples_seen,
                "train_loss": loss_sum / interval_examples,
                "validation_log_loss": validation_loss,
                "validation_macro_f1": validation_f1,
                "seconds": time.perf_counter() - started,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            records.append(record)
            print(
                f"step {step}, epoch {epoch}: train={record['train_loss']:.5f}, validation_loss={validation_loss:.5f}, validation_F1={validation_f1:.4f}, {record['seconds']:.1f}s",
                flush=True,
            )
            improved_loss = validation_loss < best_loss - 1e-5
            improved_f1 = validation_f1 > best_f1 + 1e-8
            if improved_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
            if improved_f1:
                best_f1 = validation_f1
            stale = (
                0 if improved_loss or (dual_checkpoints and improved_f1) else stale + 1
            )
            checkpoint = {
                "architecture": architecture,
                "state_dict": model.state_dict(),
                "seed": seed,
                "weighted": weighted,
                "class_weights": weights.tolist(),
                "epoch": epoch,
                "step": step,
                "examples_seen": examples_seen,
                "validation_log_loss": validation_loss,
                "validation_macro_f1": validation_f1,
                "feature_mode": "state_only" if state_only else "monthly_features",
                "lr_schedule": lr_schedule,
                "initial_learning_rate": learning_rate,
                "warmup_steps": warmup_steps if lr_schedule != "constant" else 0,
            }
            for improved, metric, filename in [
                (improved_loss, "validation_log_loss", "transformer.pt"),
                (
                    dual_checkpoints and improved_f1,
                    "validation_macro_f1",
                    "transformer_macro_f1.pt",
                ),
            ]:
                if improved:
                    torch.save(
                        {**checkpoint, "selection_metric": metric}, output / filename
                    )
            stop = step >= max_steps if max_steps else stale >= patience
            loss_sum = 0.0
            interval_examples = 0
            started = time.perf_counter()
            model.train()
            if stop:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, records
