"""Verify and summarize the controlled data-size experiment."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mortgage.prepare import write_json


def main() -> None:
    output = Path("reports/data-size")
    output.mkdir(parents=True, exist_ok=True)
    rows, classes, years, checkpoints, resources, transitions = [], [], [], [], [], []
    signatures = {}
    training_signatures = {}
    preprocessing_by_size = {}
    reference_identity = None
    reference_persistence = None
    models = [
        "persistence",
        "xgboost",
        "xgboost_macro_f1",
        "transformer",
        "transformer_macro_f1",
    ]
    for seed in [522, 523, 524]:
        small_train = None
        for size in [200_000, 1_000_000]:
            run = Path(f"runs/data-size-{size}-{seed}")
            resource = json.loads((run / "resources.json").read_text())
            config = json.loads((run / "config.json").read_text())
            assert config["seed"] == seed and config["train_limit"] == size
            assert config["history"] == config["eligibility_history"] == 12
            assert config["max_steps"] == 6000 and config["validate_every"] == 400
            assert config["dual_checkpoints"] and config["sample_seed"] == 522
            current = json.loads((run / "population_signatures.json").read_text())
            if not signatures:
                signatures = current
            assert current["validation"] == signatures["validation"]
            assert current["test"] == signatures["test"]
            preprocessing = json.loads((run / "preprocessing.json").read_text())
            if size not in training_signatures:
                training_signatures[size] = current["train"]
                preprocessing_by_size[size] = preprocessing
            assert current["train"] == training_signatures[size]
            assert preprocessing == preprocessing_by_size[size]
            with np.load(run / "partitions.npz") as indices:
                training = indices["train"]
            assert len(training) == size
            if small_train is None:
                small_train = training
            else:
                assert np.isin(small_train, training).all()
            resources.append({"seed": seed, "train_windows": size, **resource})
            curve = json.loads((run / "transformer/learning_curve.json").read_text())
            assert [point["step"] for point in curve] == list(range(400, 6001, 400))
            assert curve[-1]["examples_seen"] == 3_072_000
            figure, axes = plt.subplots(1, 2, figsize=(10, 3))
            for ax, metric in zip(axes, ["validation_log_loss", "validation_macro_f1"]):
                ax.plot(
                    [point["step"] for point in curve],
                    [point[metric] for point in curve],
                    marker=".",
                )
                ax.set(xlabel="Optimizer updates", ylabel=metric)
            figure.suptitle(f"{size:,} training windows, seed {seed}")
            figure.tight_layout()
            figure.savefig(output / f"learning_curve_{size}_{seed}.png", dpi=160)
            plt.close(figure)
            xgb_config = json.loads((run / "xgboost/training_config.json").read_text())
            assert xgb_config["learner"]["generic_param"]["device"].startswith("cuda")
            selection = json.loads(
                (run / "xgboost/checkpoint_selection.json").read_text()
            )
            assert selection["boosting_rounds_fitted"] == 200
            for name in models:
                report = json.loads((run / name / "metrics.json").read_text())
                identity_columns = [
                    "example_index",
                    "cohort",
                    "target_month",
                    "actual",
                    "current",
                ]
                identity = pd.read_parquet(
                    run / name / "predictions.parquet",
                    columns=identity_columns + ["predicted"],
                )
                predicted = identity.pop("predicted")
                if reference_identity is None:
                    reference_identity = identity
                else:
                    pd.testing.assert_frame_equal(reference_identity, identity)
                if name == "persistence":
                    np.testing.assert_array_equal(predicted, identity.current)
                    if reference_persistence is None:
                        reference_persistence = predicted
                    else:
                        np.testing.assert_array_equal(reference_persistence, predicted)
                base = {"seed": seed, "train_windows": size, "model": name}
                rows.append(
                    {
                        **base,
                        **{
                            metric: report["test"][metric]
                            for metric in [
                                "macro_f1",
                                "log_loss",
                                "accuracy",
                                "examples",
                            ]
                        },
                        "validation_macro_f1": report["validation"]["macro_f1"],
                        "validation_log_loss": report["validation"]["log_loss"],
                        "fit_and_evaluation_seconds": report["seconds"],
                    }
                )
                for label, scores in report["test"]["per_class"].items():
                    classes.append({**base, "label": label, **scores})
                for transition, scores in report["transitions"].items():
                    transitions.append({**base, "transition": transition, **scores})
                for year, scores in report["groups"]["prediction_year"].items():
                    years.append(
                        {
                            **base,
                            "year": int(year),
                            **{
                                metric: scores[metric]
                                for metric in [
                                    "macro_f1",
                                    "log_loss",
                                    "accuracy",
                                    "examples",
                                ]
                            },
                        }
                    )
                if name.startswith("transformer"):
                    saved = torch.load(
                        run / name / "transformer.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    assert saved["architecture"]["history"] == 12
                    assert not saved["architecture"]["static_context"]
                    checkpoints.append(
                        {
                            **base,
                            "selected_step_or_round": saved["step"],
                            "unit": "optimizer update",
                            "selection_metric": saved["selection_metric"],
                            "selected_epoch": saved["epoch"],
                            "fitted_updates": 6000,
                            "example_exposures": curve[-1]["examples_seen"],
                            "fitted_epochs": curve[-1]["epoch"],
                        }
                    )
                elif name.startswith("xgboost"):
                    criterion = "macro_f1" if name.endswith("macro_f1") else "log_loss"
                    checkpoints.append(
                        {
                            **base,
                            "selected_step_or_round": selection["rounds_one_based"][
                                criterion
                            ],
                            "unit": "boosting round",
                            "selection_metric": "validation_" + criterion,
                        }
                    )
            print(f"Verified {size:,} windows, seed {seed}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(output / "per_class.csv", index=False)
    pd.DataFrame(years).to_csv(output / "by_year.csv", index=False)
    pd.DataFrame(transitions).to_csv(output / "transitions.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(output / "checkpoints.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "resources.csv", index=False)
    table.groupby(["model", "train_windows"])[["macro_f1", "log_loss", "accuracy"]].agg(
        ["mean", "std"]
    ).to_csv(output / "summary.csv")
    differences = []
    for name in models:
        for metric in ["macro_f1", "log_loss", "accuracy"]:
            paired = table[table.model == name].pivot(
                index="seed", columns="train_windows", values=metric
            )
            for seed, row in paired.iterrows():
                differences.append(
                    {
                        "model": name,
                        "metric": metric,
                        "seed": seed,
                        "small": row[200_000],
                        "large": row[1_000_000],
                        "difference": row[1_000_000] - row[200_000],
                    }
                )
    paired = pd.DataFrame(differences)
    paired.to_csv(output / "paired_differences.csv", index=False)
    lines = [
        "# Training-data size: 200,000 versus 1 million windows",
        "",
        "See the [recorded protocol](PROTOCOL.md).",
        "",
        "All comparisons use nested training subsets, identical 100,000-example validation and 1,009,313-example test populations, 12-month context, and three paired initialization seeds. Both model families trained on the RTX 5070. Each transformer fit used 6,000 full-minibatch updates (3,072,000 example exposures); each XGBoost fit used 200 rounds. Preprocessing was fitted on each training subset separately.",
        "",
        "Each training run produces two checkpoint variants, selected solely by validation log loss or macro-F1. The names without `_macro_f1` use log-loss selection. Both variants are reported; they share one fit. Persistence is unchanged across all runs.",
        "",
        "| Model / checkpoint | Training windows | Macro-F1 ↑ | Log loss ↓ | Accuracy ↑ |",
        "|---|---:|---:|---:|---:|",
    ]
    for (name, size), group in table.groupby(["model", "train_windows"], sort=False):
        loss = "—" if name == "persistence" else f"{group.log_loss.mean():.5f}"
        lines.append(
            f"| {name} | {size:,} | {group.macro_f1.mean():.4f} | {loss} | {group.accuracy.mean():.5f} |"
        )
    lines.extend(
        [
            "",
            "Macro-F1 averages all seven predefined classes. Persistence produces hard predictions; its log loss is undefined. Accuracy is dominated by current loans.",
            "",
            "## Paired larger-minus-smaller changes",
            "",
            "| Model / checkpoint | Macro-F1 mean change | F1 improves in seeds | Log-loss mean change | Loss improves in seeds |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in models[1:]:
        f1 = paired[(paired.model == name) & (paired.metric == "macro_f1")].difference
        loss = paired[(paired.model == name) & (paired.metric == "log_loss")].difference
        lines.append(
            f"| {name} | {f1.mean():+.5f} | {int((f1 > 0).sum())}/3 | {loss.mean():+.5f} | {int((loss < 0).sum())}/3 |"
        )
    lines.extend(
        [
            "",
            "## Per-class mean scores",
            "",
            "| Model / checkpoint | Windows | Class | Precision | Recall | F1 | Test support |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for (name, size, label), group in pd.DataFrame(classes).groupby(
        ["model", "train_windows", "label"], sort=False
    ):
        lines.append(
            f"| {name} | {size:,} | {label} | {group.precision.mean():.4f} | {group.recall.mean():.4f} | {group.f1.mean():.4f} | {int(group.support.iloc[0]):,} |"
        )
    lines.extend(
        [
            "",
            "## Limits and reproducibility",
            "",
            "This measures data diversity under a fixed training budget. It does not establish convergence or the best hyperparameters for one million examples. XGBoost receives more work per round at the larger size; runtime is not matched across sizes or model families. Runtime CSV values include fitting and evaluation, and checkpoint variants share a fit, so do not sum their times. Full-run wall time and peak memory are recorded separately.",
            "",
            "Three model seeds summarize optimization variability, not population confidence intervals. Months overlap within loans, and rare exit classes have little support. Test data have already been inspected during earlier exploratory experiments. A consistent improvement across both macro-F1 and log loss would support a broad improvement claim; opposing changes should be described as a tradeoff. This population differs from the earlier 24-month-eligible context comparison.",
            "",
            "All seeds, paired changes, class scores, year scores, selected checkpoints, and resources are saved alongside this report. Training curves and fitted checkpoints remain in `runs/data-size-*`.",
            "",
            "![Overall performance across data sizes](overall_performance.png)",
            "",
            "```bash",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/run_data_size_experiment.py",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python -m analysis.compare_data_sizes",
            "```",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "validation.json",
        {
            "nested_training_subsets": True,
            "validation_and_test_signatures_match": True,
            "prediction_identities_match": True,
            "persistence_unchanged": True,
            "fixed_optimizer_updates_and_exposures": True,
            "xgboost_cuda_verified": True,
            "seeds": [522, 523, 524],
        },
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["macro_f1", "log_loss"]):
        for name in models[1:]:
            subset = table[table.model == name]
            means = subset.groupby("train_windows")[metric].mean()
            (line,) = ax.plot(means.index, means.values, marker="o", label=name)
            for seed in [522, 523, 524]:
                points = subset[subset.seed == seed].sort_values("train_windows")
                ax.plot(
                    points.train_windows,
                    points[metric],
                    color=line.get_color(),
                    alpha=0.25,
                )
        ax.set(
            xticks=[200_000, 1_000_000],
            xticklabels=["200k", "1m"],
            xlabel="Training windows",
            ylabel=metric,
        )
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "overall_performance.png", dpi=180)
    plt.close(figure)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
