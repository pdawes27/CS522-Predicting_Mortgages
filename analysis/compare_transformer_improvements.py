"""Verify and report every declared transformer improvement arm."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis.run_transformer_improvements import ROOT, RECIPES, experiment_sources
from mortgage.models import MortgageTransformer
from mortgage.prepare import write_json


def main() -> None:
    output = ROOT / "reports/transformer-improvements"
    selection = json.loads((output / "recipe_selection.json").read_text())
    selected = selection["selected_recipe"]
    rows, classes, years, transitions, resources, checkpoints = [], [], [], [], [], []
    reference_signatures = reference_preprocessing = reference_identity = None
    checked_runs = set()
    for stage, recipe, seed, run, name in experiment_sources():
        if run not in checked_runs:
            resource = json.loads((run / "resources.json").read_text())
            config = json.loads((run / "config.json").read_text())
            signatures = json.loads((run / "population_signatures.json").read_text())
            preprocessing = json.loads((run / "preprocessing.json").read_text())
            if reference_signatures is None:
                reference_signatures, reference_preprocessing = (
                    signatures,
                    preprocessing,
                )
            assert signatures == reference_signatures
            assert preprocessing == reference_preprocessing
            assert config["max_steps"] == 6000 and config["batch_size"] == 512
            assert config["train_limit"] == 1_000_000 and config["sample_seed"] == 522
            assert config["device"] == "cuda"
            assert config["learning_rate"] == RECIPES[recipe]["learning_rate"]
            assert (
                config.get("lr_schedule", "constant") == RECIPES[recipe]["lr_schedule"]
            )
            resources.append({"run": run.name, **resource})
            checked_runs.add(run)
        directory = run / name
        report = json.loads((directory / "metrics.json").read_text())
        identity = pd.read_parquet(
            directory / "predictions.parquet",
            columns=["example_index", "cohort", "target_month", "actual", "current"],
        )
        if reference_identity is None:
            reference_identity = identity
        else:
            pd.testing.assert_frame_equal(reference_identity, identity)
        base = {"stage": stage, "recipe": recipe, "seed": seed, "model": name}
        rows.append(
            {
                **base,
                **{
                    metric: report["test"][metric]
                    for metric in ["macro_f1", "log_loss", "accuracy", "examples"]
                },
                "validation_log_loss": report["validation"]["log_loss"],
                "validation_macro_f1": report["validation"]["macro_f1"],
                "fit_and_evaluation_seconds": report["seconds"],
            }
        )
        for label, scores in report["test"]["per_class"].items():
            classes.append({**base, "label": label, **scores})
        for year, scores in report["groups"]["prediction_year"].items():
            years.append(
                {
                    **base,
                    "year": int(year),
                    **{
                        metric: scores[metric]
                        for metric in ["macro_f1", "log_loss", "accuracy", "examples"]
                    },
                }
            )
        for transition, scores in report["transitions"].items():
            transitions.append({**base, "transition": transition, **scores})
        if name.startswith("transformer"):
            saved = torch.load(
                directory / "transformer.pt", weights_only=True, map_location="cpu"
            )
            model = MortgageTransformer(**saved["architecture"])
            model.load_state_dict(saved["state_dict"], strict=True)
            curve_path = run / name.removesuffix("_macro_f1") / "learning_curve.json"
            curve = json.loads(curve_path.read_text())
            assert [point["step"] for point in curve] == list(range(400, 6001, 400))
            assert curve[-1]["examples_seen"] == 3_072_000
            if name.startswith("transformer_residual"):
                prior = np.load(run / "transition/probabilities.npy")
                np.testing.assert_allclose(
                    model.transition_log_probs.numpy(), np.log(prior), rtol=0, atol=0
                )
            checkpoints.append(
                {
                    **base,
                    "selected_step": saved["step"],
                    "selection_metric": saved["selection_metric"],
                    "selected_epoch": saved["epoch"],
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "fitted_updates": 6000,
                    "examples_seen": curve[-1]["examples_seen"],
                }
            )
        if name.startswith("xgboost"):
            configuration = json.loads(
                (
                    run / name.removesuffix("_macro_f1") / "training_config.json"
                ).read_text()
            )
            assert configuration["learner"]["generic_param"]["device"].startswith(
                "cuda"
            )
            features = int(
                configuration["learner"]["learner_model_param"]["num_feature"]
            )
            assert features == (164 if name.startswith("xgboost_states") else 116)
    table = pd.DataFrame(rows)
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(classes).to_csv(output / "per_class.csv", index=False)
    pd.DataFrame(years).to_csv(output / "by_year.csv", index=False)
    pd.DataFrame(transitions).to_csv(output / "transitions.csv", index=False)
    pd.DataFrame(resources).to_csv(output / "resources.csv", index=False)
    pd.DataFrame(checkpoints).to_csv(output / "checkpoints.csv", index=False)
    table.groupby(["stage", "recipe", "model"])[
        ["macro_f1", "log_loss", "accuracy", "validation_log_loss"]
    ].agg(["mean", "std"]).to_csv(output / "summary.csv")
    paired = []
    for _, row in table[table.model.str.startswith("transformer")].iterrows():
        criterion = (
            "transformer_macro_f1" if row.model.endswith("_macro_f1") else "transformer"
        )
        baseline_recipe = "constant" if row.stage == "optimization" else selected
        baseline = table[
            (table.stage == "optimization")
            & (table.recipe == baseline_recipe)
            & (table.seed == row.seed)
            & (table.model == criterion)
        ].iloc[0]
        for metric in ["macro_f1", "log_loss", "accuracy"]:
            paired.append(
                {
                    "stage": row.stage,
                    "recipe": row.recipe,
                    "seed": row.seed,
                    "model": row.model,
                    "metric": metric,
                    "reference_recipe": baseline_recipe,
                    "reference_model": criterion,
                    "difference": row[metric] - baseline[metric],
                }
            )
    differences = pd.DataFrame(paired)
    differences.to_csv(output / "paired_differences.csv", index=False)
    expected_choice = min(
        RECIPES,
        key=lambda recipe: table[
            (table.stage == "optimization")
            & (table.recipe == recipe)
            & (table.model == "transformer")
        ].validation_log_loss.mean(),
    )
    assert selected == expected_choice
    lines = [
        "# Transformer improvement experiments",
        "",
        "See the [interpretation](INTERPRETATION.md) and [predeclared protocol](PROTOCOL.md).",
        "",
        f"The validation-only training-recipe selection chose **{selected}**. Architecture changes use that same recipe. All arms share one million training targets, 100,000 checkpoint-selection validation targets, 1,009,313 test targets, identical preprocessing, and three paired seeds. Every transformer fit uses 6,000 updates and 3,072,000 example exposures. All fitted models use CUDA on the RTX 5070.",
        "",
        "Names ending in `_macro_f1` use validation macro-F1 checkpoint selection; other transformer/XGBoost names use validation log loss. Both are reported and share a fit. The transition baseline is estimated from training data only; persistence is unchanged.",
        "",
        "## Overall means over three seeds",
        "",
        "| Stage | Recipe | Model / checkpoint | Validation log loss | Test macro-F1 ↑ | Test log loss ↓ | Test accuracy |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (stage, recipe, name), group in table.groupby(
        ["stage", "recipe", "model"], sort=False
    ):
        loss = "—" if name == "persistence" else f"{group.log_loss.mean():.5f}"
        valid = (
            "—" if name == "persistence" else f"{group.validation_log_loss.mean():.5f}"
        )
        lines.append(
            f"| {stage} | {recipe} | {name} | {valid} | {group.macro_f1.mean():.4f} | {loss} | {group.accuracy.mean():.5f} |"
        )
    lines.extend(
        [
            "",
            "## Paired changes against the appropriate ordinary transformer",
            "",
            "Optimization arms compare with constant 0.001. Architecture arms compare with the selected recipe's ordinary transformer. Positive macro-F1 changes and negative log-loss changes are improvements.",
            "",
            "| Stage | Recipe | Variant | Mean F1 change | F1 improves in seeds | Mean loss change | Loss improves in seeds |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for (stage, recipe, name), group in differences.groupby(
        ["stage", "recipe", "model"], sort=False
    ):
        f1, loss = (
            group[group.metric == "macro_f1"].difference,
            group[group.metric == "log_loss"].difference,
        )
        lines.append(
            f"| {stage} | {recipe} | {name} | {f1.mean():+.5f} | {int((f1 > 0).sum())}/3 | {loss.mean():+.5f} | {int((loss < 0).sum())}/3 |"
        )
    lines.extend(["", "## Calibration diagnostic", ""])
    calibration_path = output / "calibration/metrics.csv"
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        assert len(calibration) == 42
        calibration.groupby(["stage", "recipe", "model"])[
            [
                "temperature",
                "macro_f1",
                "log_loss_before",
                "log_loss_after",
                "brier_before",
                "brier_after",
            ]
        ].agg(["mean", "std"]).to_csv(output / "calibration_summary.csv")
        lines.extend(
            [
                "Temperatures are fitted only on 196,496 validation targets excluded from checkpoint selection. Calibration targets may share validation loans/input history with selection targets; all test loans remain separate. Scalar temperature scaling preserves classification, so macro-F1 is unchanged. Bounds are 0.1–10, using probabilities clipped at 1e-15.",
                "",
                "| Stage | Recipe | Model | Mean temperature | Log loss before | Log loss after | Brier before | Brier after |",
                "|---|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for (stage, recipe, name), group in calibration.groupby(
            ["stage", "recipe", "model"], sort=False
        ):
            lines.append(
                f"| {stage} | {recipe} | {name} | {group.temperature.mean():.3f} | {group.log_loss_before.mean():.5f} | {group.log_loss_after.mean():.5f} | {group.brier_before.mean():.5f} | {group.brier_after.mean():.5f} |"
            )
    else:
        lines.append(
            "Calibration has not yet been run. Run the calibration command below, then rebuild this report."
        )
    lines.extend(
        [
            "",
            "## Per-class and year-level results",
            "",
            "[Per-class precision, recall, F1, and support](per_class.csv), [year scores](by_year.csv), [transition accuracy](transitions.csv), [every seed](metrics.csv), [paired differences](paired_differences.csv), [checkpoints](checkpoints.csv), and [resources](resources.csv) are saved alongside this report. Calibration JSON files contain before/after class scores and year breakdowns for every arm. Training curves remain with their fitted checkpoints in `runs/`.",
            "",
            "## Limits",
            "",
            "Test data have already been inspected, so this is exploratory. Three seeds measure optimization variability, not population confidence intervals. Rare exits remain poorly supported. Prepending a static token, adding longer history, and scaling data were earlier separate experiments; no test-selected combinations are fitted here. The residual method includes zero initialization of its final correction layer, so a gain would not isolate the prior from that initialization. Runtime includes preprocessing and evaluation; checkpoint variants share a fit. Calibration does not demonstrate robustness to future distribution shifts.",
            "",
            "![Overall comparison](overall_performance.png)",
            "",
            "## Reproduce",
            "",
            "```bash",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/run_transformer_improvements.py",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python -m analysis.calibrate_transformer_improvements",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python -m analysis.compare_transformer_improvements",
            "```",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "validation.json",
        {
            "population_signatures_and_preprocessing_match": True,
            "prediction_identities_match": True,
            "fixed_updates_and_exposures": True,
            "prior_matches_training_transition_baseline": True,
            "validation_only_recipe_selection_verified": True,
            "xgboost_cuda_and_features_verified": True,
            "calibration_complete": calibration_path.exists(),
            "seeds": [522, 523, 524],
        },
    )
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    groups = table.groupby(["stage", "recipe", "model"], sort=False)
    plotted = [
        (f"{recipe}/{name}", group)
        for (stage, recipe, name), group in groups
        if name != "persistence"
    ]
    for ax, metric in zip(axes, ["macro_f1", "log_loss"]):
        ax.barh(
            range(len(plotted)),
            [group[metric].mean() for _, group in plotted],
            color="#2563eb",
        )
        for position, (_, group) in enumerate(plotted):
            ax.scatter(
                group[metric], [position] * len(group), color="black", s=12, alpha=0.6
            )
        ax.set(
            yticks=range(len(plotted)),
            yticklabels=[name for name, _ in plotted],
            xlabel=metric,
        )
        ax.tick_params(axis="y", labelsize=7)
    figure.set_size_inches(14, 9)
    figure.suptitle("Bars: three-seed means; dots: individual seeds")
    figure.tight_layout()
    figure.savefig(output / "overall_performance.png", dpi=180)
    plt.close(figure)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
