"""End-to-end experiments with fixed partitions."""

import argparse
import json
import os
import platform
import resource
import shutil
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from .data import FloatArray, IntArray, LoanData, nested_sample_indices, sample_indices
from .evaluate import ModelReport, macro_f1, score_model, write_comparison
from .models import predict_transformer, train_transformer, transition_probabilities
from .prepare import write_json
from .schema import LABELS


def xgboost_predictions(
    model: xgb.Booster,
    data: LoanData,
    indices: IntArray,
    batch_size: int = 8192,
    state_indicators: bool = False,
) -> FloatArray:
    probabilities = np.empty((len(indices), len(LABELS)), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        probabilities[start : start + len(batch)] = model.predict(
            xgb.DMatrix(data.flat(batch, state_indicators=state_indicators), nthread=1),
            iteration_range=(0, model.best_iteration + 1),
        )
    return probabilities


def xgboost_macro_f1(
    probabilities: FloatArray, matrix: xgb.DMatrix
) -> tuple[str, float]:
    labels = matrix.get_label().astype(np.int64)
    probabilities = probabilities.reshape(-1, len(LABELS)).astype(
        np.float32, copy=False
    )
    return "macro_f1", macro_f1(labels, probabilities)


def run(args: argparse.Namespace) -> None:
    if args.history < 1 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("History, epochs, batch size, and patience must be positive")
    if args.hidden % args.heads:
        raise ValueError("hidden must be divisible by heads")
    if args.max_steps < 0 or args.validate_every < 1:
        raise ValueError("Invalid optimizer-step budget or validation cadence")
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable. Use CUDA-enabled PyTorch and allow GPU access, or select --device cpu."
            )
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device_description = (
        torch.cuda.get_device_name(0)
        if args.device == "cuda"
        else f"{args.threads} CPU threads"
    )
    print(f"Training device: {args.device} ({device_description})", flush=True)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    started = time.perf_counter()
    data = LoanData(
        Path(args.prepared), args.years, args.history, args.eligibility_history
    )
    train_all, validation_all, test_all = data.partitions(
        args.train_end, args.validation_end, args.test_cohorts
    )
    train = (
        sample_indices(train_all, args.train_limit, args.sample_seed)
        if args.nested_anchor is None
        else nested_sample_indices(
            train_all, args.train_limit, args.sample_seed, args.nested_anchor
        )
    )
    validation = sample_indices(validation_all, args.validation_limit, args.sample_seed)
    test = sample_indices(test_all, args.test_limit, args.sample_seed)
    print(
        f"Examples: train={len(train):,}/{len(train_all):,}, validation={len(validation):,}/{len(validation_all):,}, test={len(test):,}/{len(test_all):,}",
        flush=True,
    )
    output.mkdir(parents=True)
    write_json(output / "config.json", vars(args))
    write_json(output / "data_sources.json", data.summaries)
    write_json(output / "window_selection.json", data.window_summaries)
    write_json(output / "preprocessing.json", data.fit_preprocessing(train))
    np.savez_compressed(
        output / "partitions.npz", train=train, validation=validation, test=test
    )
    write_json(
        output / "population_signatures.json",
        {
            name: data.population_signature(indices)
            for name, indices in [
                ("train", train),
                ("validation", validation),
                ("test", test),
            ]
        },
    )
    data.loans[["loan_id", "cohort", "split"]].to_parquet(
        output / "loan_splits.parquet", index=False
    )
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_cuda": torch.version.cuda,
            "gpu_name": (
                torch.cuda.get_device_name(0) if args.device == "cuda" else None
            ),
            "packages": {
                name: version(name)
                for name in [
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "torch",
                    "xgboost",
                    "scikit-learn",
                ]
            },
        },
    )
    counts = pd.DataFrame(
        {
            "label": LABELS,
            **{
                name: np.bincount(data.labels[indices], minlength=len(LABELS))
                for name, indices in [
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                ]
            },
        }
    )
    counts.to_csv(output / "class_counts.csv", index=False)
    reports: list[ModelReport] = []
    for name in args.models:
        model_start = time.perf_counter()
        alternate: tuple[FloatArray, FloatArray] | None = None
        directory = output / name
        directory.mkdir()
        if name == "persistence":
            val_prob = np.eye(len(LABELS), dtype=np.float32)[data.current[validation]]
            test_prob = np.eye(len(LABELS), dtype=np.float32)[data.current[test]]
        elif name == "transition":
            probabilities = transition_probabilities(
                data.current[train], data.labels[train]
            )
            np.save(directory / "probabilities.npy", probabilities)
            val_prob, test_prob = (
                probabilities[data.current[validation]],
                probabilities[data.current[test]],
            )
        elif name in ["xgboost", "xgboost_states"]:
            state_indicators = name == "xgboost_states"
            training = xgb.DMatrix(
                data.flat(train, state_indicators=state_indicators),
                label=data.labels[train],
                nthread=args.threads,
            )
            valid = xgb.DMatrix(
                data.flat(validation, state_indicators=state_indicators),
                label=data.labels[validation],
                nthread=args.threads,
            )
            eval_history: dict[str, dict[str, list[float]]] = {}
            model = xgb.train(
                {
                    "objective": "multi:softprob",
                    "num_class": len(LABELS),
                    "eval_metric": "mlogloss",
                    "tree_method": "hist",
                    "device": args.device,
                    "max_depth": 4,
                    "eta": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "seed": args.seed,
                    "nthread": args.threads,
                },
                training,
                num_boost_round=args.boost_rounds,
                evals=[(valid, "validation")],
                early_stopping_rounds=None if args.dual_checkpoints else 15,
                custom_metric=xgboost_macro_f1 if args.dual_checkpoints else None,
                evals_result=eval_history,
                verbose_eval=50,
            )
            write_json(
                directory / "training_config.json", json.loads(model.save_config())
            )
            write_json(directory / "evaluation_history.json", eval_history)
            del training, valid
            if args.dual_checkpoints:
                model.save_model(directory / "all_rounds.ubj")
                rounds = {
                    "log_loss": int(np.argmin(eval_history["validation"]["mlogloss"])),
                    "macro_f1": int(np.argmax(eval_history["validation"]["macro_f1"])),
                }
                f1_model = model[: rounds["macro_f1"] + 1]
                f1_model.set_attr(
                    best_iteration=str(rounds["macro_f1"]),
                    selection_metric="validation_macro_f1",
                )
                alternate_directory = output / f"{name}_macro_f1"
                alternate_directory.mkdir()
                f1_model.save_model(alternate_directory / "model.ubj")
                alternate = (
                    xgboost_predictions(
                        f1_model, data, validation, state_indicators=state_indicators
                    ),
                    xgboost_predictions(
                        f1_model, data, test, state_indicators=state_indicators
                    ),
                )
                del f1_model
                model = model[: rounds["log_loss"] + 1]
                model.set_attr(
                    best_iteration=str(rounds["log_loss"]),
                    selection_metric="validation_log_loss",
                )
                write_json(
                    directory / "checkpoint_selection.json",
                    {
                        "validation_only": True,
                        "rounds_one_based": {
                            key: value + 1 for key, value in rounds.items()
                        },
                        "boosting_rounds_fitted": args.boost_rounds,
                    },
                )
            model.save_model(directory / "model.ubj")
            val_prob = xgboost_predictions(
                model, data, validation, state_indicators=state_indicators
            )
            test_prob = xgboost_predictions(
                model, data, test, state_indicators=state_indicators
            )
            del model
        else:
            history = 1 if name == "transformer_current" else args.history
            model, curve = train_transformer(
                data,
                train,
                validation,
                directory,
                args.seed,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                hidden=args.hidden,
                heads=args.heads,
                layers=args.layers,
                learning_rate=args.learning_rate,
                weighted=name == "transformer_weighted",
                history=history,
                device_name=args.device,
                state_only=name == "autoregressive",
                static_context=name == "transformer_conditioned",
                max_steps=args.max_steps,
                validate_every=args.validate_every,
                dual_checkpoints=args.dual_checkpoints,
                lr_schedule=args.lr_schedule,
                warmup_steps=args.warmup_steps,
                transition_residual=name == "transformer_residual",
                state_embedding=name == "transformer_state_embedding",
            )
            write_json(directory / "learning_curve.json", curve)
            val_prob = predict_transformer(
                model,
                data,
                validation,
                args.batch_size,
                torch.device(args.device),
                history,
                name == "autoregressive",
            )
            test_prob = predict_transformer(
                model,
                data,
                test,
                args.batch_size,
                torch.device(args.device),
                history,
                name == "autoregressive",
            )
            if args.dual_checkpoints:
                saved = torch.load(
                    directory / "transformer_macro_f1.pt",
                    map_location=torch.device(args.device),
                    weights_only=True,
                )
                model.load_state_dict(saved["state_dict"], strict=True)
                alternate_directory = output / f"{name}_macro_f1"
                alternate_directory.mkdir()
                shutil.copyfile(
                    directory / "transformer_macro_f1.pt",
                    alternate_directory / "transformer.pt",
                )
                alternate = (
                    predict_transformer(
                        model,
                        data,
                        validation,
                        args.batch_size,
                        torch.device(args.device),
                        history,
                        name == "autoregressive",
                    ),
                    predict_transformer(
                        model,
                        data,
                        test,
                        args.batch_size,
                        torch.device(args.device),
                        history,
                        name == "autoregressive",
                    ),
                )
            del model
        report = score_model(
            name,
            data,
            validation,
            test,
            val_prob,
            test_prob,
            output,
            time.perf_counter() - model_start,
            persistence=name == "persistence",
        )
        reports.append(report)
        if alternate is not None:
            reports.append(
                score_model(
                    f"{name}_macro_f1",
                    data,
                    validation,
                    test,
                    alternate[0],
                    alternate[1],
                    output,
                    time.perf_counter() - model_start,
                )
            )
        write_comparison(output, reports)
    write_json(
        output / "resources.json",
        {
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            "device": args.device,
            "threads": args.threads,
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated() / 1024**3
                if args.device == "cuda"
                else None
            ),
        },
    )
    print(f"Saved experiment to {output}", flush=True)
