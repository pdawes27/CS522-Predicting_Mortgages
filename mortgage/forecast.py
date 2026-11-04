"""Autoregressive state simulation and horizon backtests."""

import argparse
import json
from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch import nn

from .data import FloatArray, IntArray, LoanData, Preprocessing
from .evaluate import metrics
from .models import MortgageTransformer, transition_probabilities
from .prepare import write_json
from .schema import LABELS, month_numbers, year_month


class ForecastConfig(TypedDict):
    years: list[int]
    train_end: str | None
    eligibility_history: int | None


def generate_paths(
    model: nn.Module,
    histories: IntArray,
    static: FloatArray,
    horizon: int = 12,
    paths: int = 1000,
    seed: int = 522,
    batch_size: int = 2048,
    greedy: bool = False,
) -> tuple[NDArray[np.int8], FloatArray]:
    """Generate state tokens until exit. No future measurements are accepted.

    Raw trajectories use -1 after a terminal token. Marginals carry terminal states
    forward as absorbing outcomes, which is a probability accounting convention,
    not additional generated loan records. Shapes: [loans, paths, horizon] and
    [loans, horizon, classes].
    """
    if horizon < 1 or paths < 1 or batch_size < 1:
        raise ValueError("Horizon, paths, and batch size must be positive")
    if histories.ndim != 2 or not len(histories) or histories.shape[1] < 1:
        raise ValueError("Supply a nonempty matrix of active state histories")
    if np.any((histories < 0) | (histories >= 4)):
        raise ValueError("Histories must contain known active states 0 through 3")
    if (
        static.ndim != 2
        or len(static) != len(histories)
        or not np.isfinite(static).all()
    ):
        raise ValueError("Static attributes must match histories and be finite")
    if greedy and paths != 1:
        raise ValueError("Greedy generation uses one path per loan")
    generator = np.random.default_rng(seed)
    contexts = np.repeat(histories, paths, axis=0).copy()
    attributes = np.repeat(static, paths, axis=0)
    generated = np.full((len(contexts), horizon), -1, dtype=np.int8)
    occupied = np.empty_like(generated)
    active = np.arange(len(contexts))
    last = contexts[:, -1].copy()
    model.eval()
    with torch.inference_mode():
        for step in range(horizon):
            for start in range(0, len(active), batch_size):
                selected = active[start : start + batch_size]
                dynamic = torch.from_numpy(
                    np.eye(4, dtype=np.float32)[contexts[selected]]
                )
                logits = model(dynamic, torch.from_numpy(attributes[selected]))
                probabilities = logits.softmax(-1).cpu().numpy()
                if not np.isfinite(probabilities).all():
                    raise ValueError("Model produced nonfinite probabilities")
                if greedy:
                    next_state = probabilities.argmax(axis=1)
                else:
                    cumulative = np.cumsum(probabilities, axis=1)
                    cumulative[:, -1] = 1.0
                    next_state = (
                        generator.random(len(selected))[:, None] > cumulative
                    ).sum(axis=1)
                generated[selected, step] = next_state
                last[selected] = next_state
                contexts[selected, :-1] = contexts[selected, 1:]
                contexts[selected, -1] = next_state
            occupied[:, step] = last
            active = active[last[active] < 4]
    generated = generated.reshape(len(histories), paths, horizon)
    occupied = occupied.reshape(len(histories), paths, horizon)
    probabilities = np.stack(
        [(occupied == state).mean(axis=1) for state in range(len(LABELS))], axis=-1
    )
    return generated, probabilities.astype(np.float32)


def observed_horizons(data: LoanData, indices: IntArray, horizon: int) -> IntArray:
    """Observed future labels, carrying known exits; unknown/censored months stay -1."""
    targets = data.targets[indices]
    loans = data.target_loans[indices]
    ends = np.searchsorted(data.row_loan, loans, side="right") - 1
    last_month = data.months[ends]
    last_state = data.states[ends]
    actual = np.full((len(indices), horizon), -1, dtype=np.int64)
    for step in range(horizon):
        desired = data.target_months[indices] + step
        rows = np.minimum(targets + step, len(data.states) - 1)
        observed = (data.row_loan[rows] == loans) & (data.months[rows] == desired)
        actual[observed, step] = data.states[rows[observed]]
        terminated = (last_state >= 4) & (last_month <= desired)
        actual[terminated, step] = last_state[terminated]
    return actual


def load_forecaster(
    run: Path,
) -> tuple[MortgageTransformer, Preprocessing, ForecastConfig]:
    if not (run / "resources.json").exists():
        raise ValueError("Choose a completed autoregressive training run")
    checkpoint = torch.load(
        run / "autoregressive" / "transformer.pt", map_location="cpu", weights_only=True
    )
    if checkpoint.get("feature_mode") != "state_only":
        raise ValueError(
            "Recursive generation requires the state-only autoregressive checkpoint"
        )
    model = MortgageTransformer(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    preprocessing: Preprocessing = json.loads((run / "preprocessing.json").read_text())
    config = json.loads((run / "config.json").read_text())
    return (
        model,
        preprocessing,
        {
            "years": config["years"],
            "train_end": config.get("train_end"),
            "eligibility_history": config.get("eligibility_history"),
        },
    )


def forecast(args: argparse.Namespace) -> None:
    torch.set_num_threads(args.threads)
    run, output = Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh forecast output directory: {output}")
    model, preprocessing, config = load_forecaster(run)
    history = model.position.shape[1]
    data = LoanData(
        Path(args.prepared), [args.cohort], history, config["eligibility_history"]
    )
    data.apply_preprocessing(preprocessing)
    if args.loan_index < 0 or args.loan_index >= len(data.loans):
        raise ValueError("Loan index is outside the selected cohort")
    rows = np.flatnonzero(data.row_loan == args.loan_index)
    if args.as_of:
        cutoff = month_numbers(pd.Series([args.as_of.replace("-", "")]))[0]
        rows = rows[data.months[rows] <= cutoff]
        if not len(rows) or data.months[rows[-1]] != cutoff:
            raise ValueError("Loan has no observation in the requested as-of month")
    if len(rows) < history or np.any(np.diff(data.months[rows[-history:]]) != 1):
        raise ValueError(
            "Loan needs a consecutive active history of the configured length"
        )
    context = data.states[rows[-history:]][None, :]
    attributes = data.static[[args.loan_index]]
    trajectories, probabilities = generate_paths(
        model, context, attributes, args.horizon, args.paths, args.seed
    )
    greedy, _ = generate_paths(
        model, context, attributes, args.horizon, 1, args.seed, greedy=True
    )
    as_of = int(data.months[rows[-1]])
    output.mkdir(parents=True)
    frame = pd.DataFrame(probabilities[0], columns=LABELS)
    frame.insert(
        0, "month", [year_month(as_of + step) for step in range(1, args.horizon + 1)]
    )
    frame.insert(0, "horizon", np.arange(1, args.horizon + 1))
    frame.to_csv(output / "state_probabilities.csv", index=False)
    np.save(output / "sampled_paths.npy", trajectories[0])
    greedy_path = [LABELS[state] for state in greedy[0, 0] if state >= 0]
    write_json(
        output / "forecast.json",
        {
            "run": str(run),
            "cohort": args.cohort,
            "loan_index": args.loan_index,
            "as_of": year_month(as_of),
            "history": [LABELS[state] for state in context[0]],
            "horizon": args.horizon,
            "paths": args.paths,
            "seed": args.seed,
            "greedy_path": greedy_path,
            "model_train_end": config.get("train_end"),
            "payoff_by_horizon": float(probabilities[0, -1, 4]),
            "adverse_exit_or_reo_by_horizon": float(probabilities[0, -1, 5]),
            "other_exit_by_horizon": float(probabilities[0, -1, 6]),
            "ever_90_plus_during_horizon": float(
                (trajectories[0] == 3).any(axis=1).mean()
            ),
            "semantics": "Paths stop at an exit. Probability summaries carry exits forward as absorbing outcomes. Future balances and interest rates are not inputs. These are model simulations, not guarantees.",
        },
    )
    figure, ax = plt.subplots(figsize=(10, 5))
    ax.stackplot(frame.horizon, *[frame[label] for label in LABELS], labels=LABELS)
    ax.set(
        xlabel="Months ahead",
        ylabel="Estimated state probability",
        ylim=(0, 1),
        title=f"Autoregressive simulation from {year_month(as_of)} ({args.paths:,} sampled paths)",
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "forecast.png", dpi=160)
    plt.close(figure)
    print("Greedy path: " + " → ".join(greedy_path), flush=True)
    print(f"Saved forecast to {output}", flush=True)


def evaluate_rollouts(args: argparse.Namespace) -> None:
    if args.origins < 1:
        raise ValueError("The number of forecast origins must be positive")
    torch.set_num_threads(args.threads)
    run, output = Path(args.run), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh evaluation output directory: {output}")
    model, preprocessing, config = load_forecaster(run)
    data = LoanData(
        Path(args.prepared),
        config["years"],
        model.position.shape[1],
        config["eligibility_history"],
    )
    data.apply_preprocessing(preprocessing)
    partitions = np.load(run / "partitions.npz")
    test = partitions["test"]
    # One random origin per uniformly selected test loan, avoiding repeated nearby windows.
    _, starts, counts = np.unique(
        data.target_loans[test], return_index=True, return_counts=True
    )
    generator = np.random.default_rng(args.seed)
    groups = generator.choice(
        len(starts), min(args.origins, len(starts)), replace=False
    )
    indices = test[starts[groups] + generator.integers(counts[groups])]
    target_rows = data.targets[indices]
    histories = data.states[target_rows[:, None] + np.arange(-data.history, 0)]
    trajectories, probabilities = generate_paths(
        model,
        histories,
        data.static[data.target_loans[indices]],
        args.horizon,
        args.paths,
        args.seed,
    )
    actual = observed_horizons(data, indices, args.horizon)
    train = partitions["train"]
    transition = transition_probabilities(
        data.current[train], data.labels[train], absorbing=True
    )
    transition_distribution = np.eye(len(LABELS), dtype=np.float32)[
        data.current[indices]
    ]
    persistence = transition_distribution.copy()
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "origins_and_paths.npz",
        indices=indices,
        trajectories=trajectories,
        actual=actual,
    )
    records = []
    for step in range(args.horizon):
        transition_distribution = transition_distribution @ transition
        known = actual[:, step] >= 0
        if not known.any():
            continue
        labels = actual[known, step]
        for name, prob in [
            ("autoregressive", probabilities[:, step]),
            ("transition", transition_distribution),
            ("persistence", persistence),
        ]:
            scores = metrics(labels, prob[known])
            # Monte Carlo zeros make clipped log loss unstable; Brier is well-defined.
            scores["log_loss"] = None
            brier = float(
                np.square(prob[known] - np.eye(len(LABELS))[labels]).sum(axis=1).mean()
            )
            records.append(
                {"horizon": step + 1, "model": name, "brier": brier, **scores}
            )
    write_json(
        output / "metrics.json",
        {
            "config": vars(args),
            "origins": len(indices),
            "sampling": "one random test origin per uniformly selected test loan",
            "censoring": "Unknown/unobserved outcomes excluded separately at each horizon; known terminal events carried forward",
            "probabilities": "Autoregressive probabilities estimated from sampled paths; no future actual states or monthly features fed back",
            "records": records,
        },
    )
    table = pd.DataFrame(
        [
            {
                key: row[key]
                for key in [
                    "horizon",
                    "model",
                    "examples",
                    "accuracy",
                    "macro_f1",
                    "brier",
                ]
            }
            for row in records
        ]
    )
    table.to_csv(output / "comparison.csv", index=False)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, group in table.groupby("model"):
        axes[0].plot(group.horizon, group.macro_f1, marker="o", label=name)
        axes[1].plot(group.horizon, group.brier, marker="o", label=name)
    axes[0].set(xlabel="Months ahead", ylabel="Macro-F1")
    axes[1].set(
        xlabel="Months ahead", ylabel="Multiclass Brier score (lower is better)"
    )
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "horizon_performance.png", dpi=160)
    plt.close(figure)
    print(f"Saved recursive horizon evaluation to {output}", flush=True)
