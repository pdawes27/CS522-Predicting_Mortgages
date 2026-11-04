"""Summarize the predeclared paired architecture experiment.

Run after seeds 522, 523, and 524 finish:
MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv/bin/python analysis/compare_static_context.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mortgage.evaluate import payoff_scores
from mortgage.models import MortgageTransformer
from mortgage.prepare import write_json
from mortgage.schema import LABELS


def main() -> None:
    output = Path("reports/static-conditioning")
    output.mkdir(parents=True, exist_ok=True)
    rows, year_rows = [], []
    common_signatures = common_preprocessing = None
    common_settings = None
    feature_columns = [f"p_{label}" for label in LABELS]
    reproduced_original = False
    for seed in [522, 523, 524]:
        run = Path(f"runs/static-context-{seed}")
        if not (run / "resources.json").exists():
            raise ValueError(f"Run incomplete: {run}")
        config = json.loads((run / "config.json").read_text())
        settings = {
            key: value for key, value in config.items() if key not in {"seed", "output"}
        }
        if common_settings is None:
            common_settings = settings
        else:
            assert settings == common_settings, "Training settings differ between seeds"
        if config["seed"] != seed or config["models"] != [
            "transformer",
            "transformer_conditioned",
        ]:
            raise ValueError(f"Unexpected run configuration: {run}")
        signatures = json.loads((run / "population_signatures.json").read_text())
        preprocessing = json.loads((run / "preprocessing.json").read_text())
        if common_signatures is None:
            common_signatures, common_preprocessing = signatures, preprocessing
        else:
            assert (
                signatures == common_signatures
            ), "Example populations differ between seeds"
            assert (
                preprocessing == common_preprocessing
            ), "Preprocessing differs between seeds"
        reference = None
        for name in config["models"]:
            directory = run / name
            scores = json.loads((directory / "metrics.json").read_text())
            frame = pd.read_parquet(directory / "predictions.parquet")
            if reference is None:
                reference = frame[
                    ["example_index", "actual", "target_month", "current", "cohort"]
                ]
            else:
                pd.testing.assert_frame_equal(reference, frame[reference.columns])
            saved = torch.load(
                directory / "transformer.pt", map_location="cpu", weights_only=True
            )
            assert saved["architecture"]["static_context"] == (
                name == "transformer_conditioned"
            )
            model = MortgageTransformer(**saved["architecture"])
            model.load_state_dict(saved["state_dict"], strict=True)
            rows.append(
                {
                    "seed": seed,
                    "model": name,
                    **payoff_scores(frame),
                    "macro_f1": scores["test"]["macro_f1"],
                    "log_loss": scores["test"]["log_loss"],
                    "validation_log_loss": scores["validation"]["log_loss"],
                    "checkpoint_epoch": saved["epoch"],
                    "epochs_run": len(
                        json.loads((directory / "learning_curve.json").read_text())
                    ),
                    "parameters": sum(param.numel() for param in model.parameters()),
                    "seconds": scores["seconds"],
                }
            )
            for year, group in frame.groupby(frame.target_month // 12):
                year_rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "year": int(year),
                        **payoff_scores(group),
                    }
                )
            if seed == 522 and name == "transformer":
                original = pd.read_parquet(
                    "runs/future-period/transformer/predictions.parquet"
                )
                pd.testing.assert_frame_equal(
                    frame[reference.columns], original[reference.columns]
                )
                np.testing.assert_array_equal(
                    frame[feature_columns].to_numpy(),
                    original[feature_columns].to_numpy(),
                )
                reproduced_original = True
        print(f"Verified seed {seed}", flush=True)
    table, yearly = pd.DataFrame(rows), pd.DataFrame(year_rows)
    table.to_csv(output / "metrics.csv", index=False)
    yearly.to_csv(output / "metrics_by_year.csv", index=False)
    paired = table.pivot(index="seed", columns="model", values="payoff_ap")
    paired["difference"] = paired.transformer_conditioned - paired.transformer
    paired.to_csv(output / "paired_payoff_ap.csv")
    reference_path = Path("runs/future-period/xgboost/predictions.parquet")
    xgboost = payoff_scores(pd.read_parquet(reference_path))
    averages = table.groupby("model").agg(
        {
            "payoff_ap": ["mean", "std"],
            "precision_at_1_percent": ["mean", "std"],
            "payoff_brier": ["mean", "std"],
            "macro_f1": ["mean", "std"],
            "log_loss": ["mean", "std"],
        }
    )
    averages.to_csv(output / "summary.csv")
    write_json(
        output / "validation.json",
        {
            "population_signatures_match": True,
            "preprocessing_matches": True,
            "training_settings_match": True,
            "legacy_seed_522_predictions_exactly_reproduced": reproduced_original,
            "strict_checkpoint_loads": True,
            "seeds": [522, 523, 524],
            "status": "Controlled exploratory comparison; test data previously inspected, not a fresh confirmation",
            "xgboost_reference": xgboost,
        },
    )
    lines = [
        "# Static-conditioned attention: paired experiment",
        "",
        "The [protocol](PROTOCOL.md) was recorded before fitting this variant.",
        "",
        "## Change",
        "",
        "The optional `transformer_conditioned` model prepends a projected origination-profile token to the 12 monthly tokens. Both variants retain the direct static-feature classifier connection. Data preparation, features, splits, optimization, and validation checkpoint selection are matched. The new projection adds parameters; this is an architecture comparison, not a parameter-matched capacity experiment.",
        "",
        "## Data and controls",
        "",
        "2016–2025 sample cohorts; training targets through 2022, validation in 2023, test after 2023. Each run uses the same 200,000 sampled training windows, 100,000 validation windows, and all 1,009,313 test windows. Ordered loan/month/label fingerprints and saved preprocessing match across all runs. Both models use 12-month context, two layers, four heads, hidden size 64, and up to 15 epochs with patience three. Checkpoints minimize multiclass validation log loss. Only the model seed varies across paired runs.",
        "",
        "The unconditioned seed-522 run reproduces the original future-period probabilities exactly. Old checkpoints remain loadable with conditioning disabled by default.",
        "",
        "## Results by seed",
        "",
        "| Seed | Model | Payoff AP | Payoffs in top 1% | Precision at 1% | Payoff Brier | Macro-F1 | Log loss | Selected epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.seed} | {row.model} | {row.payoff_ap:.5f} | {row.top_1_percent_payoffs} | {row.precision_at_1_percent:.2%} | {row.payoff_brier:.6f} | {row.macro_f1:.4f} | {row.log_loss:.5f} | {row.checkpoint_epoch} |"
        )
    continuation = chr(92)
    lines.extend(
        [
            "",
            "## Paired primary endpoint",
            "",
            "| Seed | Conditioned minus original payoff AP |",
            "|---|---:|",
        ]
    )
    for seed, row in paired.iterrows():
        lines.append(f"| {seed} | {row['difference']:+.5f} |")
    mean_difference = float(paired.difference.mean())
    wins = int((paired.difference > 0).sum())
    lines.extend(
        [
            "",
            f"Mean paired AP change: **{mean_difference:+.5f}**. Conditioning improves payoff AP in **{wins} of 3 seeds**. Seed standard deviations describe optimization variability; three seeds do not establish a precise population confidence interval.",
            "",
            f"Existing XGBoost reference (seed 522): payoff AP {xgboost['payoff_ap']:.5f}, top-1% payoff count {xgboost['top_1_percent_payoffs']}, precision {xgboost['precision_at_1_percent']:.2%}. This reference has not been retuned as part of this experiment.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if wins == 3:
        lines.append(
            "The static-context token improves the prespecified focused endpoint consistently across the three tested seeds. The absolute effect, other outcome metrics, calibration, and baseline tuning still matter. These inspected test sets provide exploratory evidence, not independent confirmation."
        )
    elif wins == 0:
        lines.append(
            "The static-context token does not improve the prespecified payoff-ranking endpoint in these runs. Preserve the original model as the current reference; do not present the architecture addition as an established improvement."
        )
    else:
        lines.append(
            "Results vary by seed, so the extra context token has not established a consistent payoff-ranking advantage. Avoid selecting only the favorable seed or changing the endpoint after seeing these results."
        )
    lines.extend(
        [
            "",
            "The projected token uses existing origination inputs only, so it introduces no future measurements. It does not address the shortage of rare adverse-exit labels or supply future market rates. Better payoff ranking does not by itself imply calibrated probabilities or improved delinquency/default detection.",
            "",
            "![Paired payoff ranking](payoff_ap_by_seed.png)",
            "",
            "## Reproduction",
            "",
            "```bash",
            "# Repeat for --seed 522, 523, and 524, using a fresh output directory each time.",
            f"MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv/bin/python -m mortgage.cli run {continuation}",
            f"  --years 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 {continuation}",
            f"  --train-end 2022-12 --validation-end 2023-12 {continuation}",
            f"  --train-limit 200000 --validation-limit 100000 --epochs 15 --seed 522 {continuation}",
            "  --models transformer transformer_conditioned --output runs/static-context-522-new",
            "```",
            "",
            "The summarizer reads the three original `runs/static-context-SEED` directories:",
            "",
            "```bash",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv/bin/python analysis/compare_static_context.py",
            "```",
            "",
            "Full model checkpoints, probabilities, confusion matrices, learning curves, run configurations, and source fingerprints remain under the local ignored run directories.",
        ]
    )
    gpu_root = Path("runs/static-context-gpu-522")
    if (gpu_root / "resources.json").exists():
        gpu_resources = json.loads((gpu_root / "resources.json").read_text())
        gpu_environment = json.loads((gpu_root / "environment.json").read_text())
        assert (
            gpu_resources["device"] == "cuda"
            and gpu_resources["peak_cuda_allocated_gib"] > 0
        )
        assert (
            json.loads((gpu_root / "population_signatures.json").read_text())
            == common_signatures
        )
        assert (
            json.loads((gpu_root / "preprocessing.json").read_text())
            == common_preprocessing
        )
        lines.extend(
            [
                "",
                "## Separate GPU repeat",
                "",
                f"A seed-522 pair also ran on {gpu_environment['gpu_name']} using PyTorch {gpu_environment['packages']['torch']} / CUDA {gpu_environment['torch_cuda']}. All population fingerprints and preprocessing match the CPU comparison. This repeat is not pooled with the primary CPU results.",
                "",
                "| Model | GPU payoff AP | Payoffs in top 1% | Median GPU epoch (s) | Median CPU epoch (s) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        gpu_rows = []
        for name in ["transformer", "transformer_conditioned"]:
            frame = pd.read_parquet(gpu_root / name / "predictions.parquet")
            gpu_curve = json.loads(
                (gpu_root / name / "learning_curve.json").read_text()
            )
            cpu_curve = json.loads(
                (
                    Path("runs/static-context-522") / name / "learning_curve.json"
                ).read_text()
            )
            row = {
                "model": name,
                **payoff_scores(frame),
                "median_gpu_epoch_seconds": float(
                    np.median([point["seconds"] for point in gpu_curve])
                ),
                "median_cpu_epoch_seconds": float(
                    np.median([point["seconds"] for point in cpu_curve])
                ),
            }
            gpu_rows.append(row)
            lines.append(
                f"| {name} | {row['payoff_ap']:.5f} | {row['top_1_percent_payoffs']} | {row['median_gpu_epoch_seconds']:.1f} | {row['median_cpu_epoch_seconds']:.1f} |"
            )
        pd.DataFrame(gpu_rows).to_csv(output / "gpu_metrics.csv", index=False)
        lines.extend(
            [
                "",
                f"GPU run wall time: {gpu_resources['wall_seconds']:.1f} seconds; peak allocated CUDA tensor memory: {gpu_resources['peak_cuda_allocated_gib']:.3f} GiB. Epochs include validation. CPU fits overlapped other CPU runs, so timings are descriptive, not an isolated benchmark. CPU/GPU floating-point paths can select different checkpoints.",
                "",
                "See [GPU verification details](GPU.md) and the [GPU setup instructions](../../README.md#gpu-training).",
            ]
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    figure, ax = plt.subplots(figsize=(7, 4))
    for seed, row in paired.iterrows():
        ax.plot(
            [0, 1],
            [row.transformer, row.transformer_conditioned],
            marker="o",
            label=f"Seed {seed}",
        )
    ax.axhline(
        float(xgboost["payoff_ap"]),
        color="gray",
        linestyle="--",
        label="Existing XGBoost",
    )
    ax.set(
        xticks=[0, 1],
        xticklabels=["Original transformer", "Static-context token"],
        ylabel="Next-month payoff average precision",
        title="Matched architecture comparison",
    )
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "payoff_ap_by_seed.png", dpi=180)
    plt.close(figure)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
