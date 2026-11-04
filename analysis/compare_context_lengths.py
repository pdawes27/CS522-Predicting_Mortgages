"""Analyze matched 12/24-month experiments.

Run after the six GPU jobs finish:
MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/compare_context_lengths.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from mortgage.evaluate import payoff_scores
from mortgage.models import MortgageTransformer
from mortgage.prepare import write_json


def main() -> None:
    output = Path("reports/context-length")
    output.mkdir(parents=True, exist_ok=True)
    models = ["persistence", "xgboost", "transformer", "transformer_conditioned"]
    rows, year_rows, window_rows = [], [], []
    signatures = static_preprocessing = vocabularies = None
    reference_predictions = None
    for seed in [522, 523, 524]:
        short_params = None
        for history in [12, 24]:
            run = Path(f"runs/context-{history}-{seed}")
            if not (run / "resources.json").exists():
                raise ValueError(f"Run is incomplete: {run}")
            config = json.loads((run / "config.json").read_text())
            assert config["history"] == history and config["eligibility_history"] == 24
            assert config["seed"] == seed and config["sample_seed"] == 522
            assert config["models"] == models and config["device"] == "cuda"
            for key, value in {
                "train_limit": 200_000,
                "validation_limit": 100_000,
                "test_limit": 0,
                "train_end": "2022-12",
                "validation_end": "2023-12",
                "epochs": 15,
                "patience": 3,
                "batch_size": 512,
                "hidden": 64,
                "heads": 4,
                "layers": 2,
                "learning_rate": 0.001,
                "boost_rounds": 200,
                "threads": 4,
            }.items():
                assert config[key] == value, f"Unexpected {key}: {run}"
            current_signatures = json.loads(
                (run / "population_signatures.json").read_text()
            )
            preprocessing = json.loads((run / "preprocessing.json").read_text())
            if signatures is None:
                signatures = current_signatures
                static_preprocessing = preprocessing["static"]
                vocabularies = preprocessing["vocabularies"]
            assert (
                signatures == current_signatures
            ), "Targets or sampled populations changed"
            assert static_preprocessing == preprocessing["static"]
            assert vocabularies == preprocessing["vocabularies"]
            if history == 12:
                short_params = preprocessing
            else:
                assert short_params is not None
                assert (
                    preprocessing["training_input_rows"]
                    >= short_params["training_input_rows"]
                )
            if seed == 522:
                for item in json.loads((run / "window_selection.json").read_text()):
                    window_rows.append({"history": history, **item})
            config_path = run / "xgboost" / "training_config.json"
            xgb_config = json.loads(config_path.read_text())
            assert xgb_config["learner"]["generic_param"]["device"].startswith("cuda")
            xgb_features = int(
                xgb_config["learner"]["learner_model_param"]["num_feature"]
            )
            assert xgb_features == history * 8 + 20
            for name in models:
                directory = run / name
                frame = pd.read_parquet(directory / "predictions.parquet")
                identity = frame[
                    ["example_index", "cohort", "target_month", "actual", "current"]
                ]
                if reference_predictions is None:
                    reference_predictions = identity
                else:
                    pd.testing.assert_frame_equal(reference_predictions, identity)
                if name == "persistence":
                    np.testing.assert_array_equal(frame.predicted, frame.current)
                scores = json.loads((directory / "metrics.json").read_text())
                parameters = None
                if name.startswith("transformer"):
                    saved = torch.load(
                        directory / "transformer.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                    assert saved["architecture"]["history"] == history
                    model = MortgageTransformer(**saved["architecture"])
                    model.load_state_dict(saved["state_dict"], strict=True)
                    parameters = sum(param.numel() for param in model.parameters())
                selected = frame.current.to_numpy() == 0
                actual_delinquency = np.isin(
                    frame.actual.to_numpy()[selected], [1, 2, 3]
                )
                delinquency_probability = (
                    frame[["p_30_59_days", "p_60_89_days", "p_90_plus_days"]]
                    .sum(axis=1)
                    .to_numpy()[selected]
                )
                rows.append(
                    {
                        "seed": seed,
                        "history": history,
                        "model": name,
                        **payoff_scores(frame),
                        "macro_f1": scores["test"]["macro_f1"],
                        "log_loss": scores["test"]["log_loss"],
                        "validation_log_loss": scores["validation"]["log_loss"],
                        "new_delinquency_ap": float(
                            average_precision_score(
                                actual_delinquency, delinquency_probability
                            )
                        ),
                        "test_examples": len(frame),
                        "parameters": parameters,
                        "xgboost_features": xgb_features if name == "xgboost" else None,
                    }
                )
                for year, group in frame.groupby(frame.target_month // 12):
                    year_rows.append(
                        {
                            "seed": seed,
                            "history": history,
                            "model": name,
                            "year": int(year),
                            **payoff_scores(group),
                        }
                    )
            print(f"Verified context {history}, seed {seed}", flush=True)
    table = pd.DataFrame(rows)
    assert reference_predictions is not None
    table.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(year_rows).to_csv(output / "metrics_by_year.csv", index=False)
    write_json(output / "window_selection.json", window_rows)
    comparisons = []
    for name in models:
        for metric in ["payoff_ap", "macro_f1", "new_delinquency_ap", "payoff_brier"]:
            paired = table.loc[table.model == name].pivot(
                index="seed", columns="history", values=metric
            )
            for seed, row in paired.iterrows():
                comparisons.append(
                    {
                        "model": name,
                        "metric": metric,
                        "seed": int(seed),
                        "history_12": row[12],
                        "history_24": row[24],
                        "difference_24_minus_12": row[24] - row[12],
                    }
                )
    paired_table = pd.DataFrame(comparisons)
    paired_table.to_csv(output / "paired_differences.csv", index=False)
    summary = table.groupby(["model", "history"])[
        [
            "payoff_ap",
            "macro_f1",
            "new_delinquency_ap",
            "payoff_brier",
            "precision_at_1_percent",
        ]
    ].agg(["mean", "std"])
    summary.to_csv(output / "summary.csv")
    lines = [
        "# More past context: 12 versus 24 months",
        "",
        "See the [predeclared protocol](PROTOCOL.md). This is an exploratory comparison on inspected test data.",
        "",
        "## What changes for each model?",
        "",
        "The target remains the next reporting month's state. The transformer attends to either 12 or 24 monthly tokens. XGBoost receives the same monthly vectors flattened into features, plus origination attributes: 116 input columns at 12 months and 212 at 24 months. Both must be retrained. Persistence copies the last observed state and does not use the older history; its predictions are identical on the matched examples.",
        "",
        "## Controls",
        "",
        f"All six runs use the same {len(reference_predictions):,} test loan-months, 200,000 sampled training targets, and 100,000 sampled validation targets. Every example has 24 consecutive active input months and an observed next-month target, even for the 12-month arm. All prediction identities and population fingerprints match. Static preprocessing and categorical vocabularies match; dynamic scaling is fitted separately on each context's training inputs.",
        "",
        "Training targets end in 2022; validation is in 2023; test targets follow. Seeds 522, 523, and 524 are paired across contexts. XGBoost and both transformers train with CUDA on the RTX 5070. Persistence is a direct vectorized last-state rule. Checkpoint selection and all other model/training settings follow the protocol. Cached prepared files were reused; window indices were recalculated without modifying raw files.",
        "",
        "The common-history restriction changes the population from the earlier broad 12-month experiments. It excludes early exits and newer/short-history loans. These scores should be compared within this experiment, not directly against the earlier test population.",
        "",
        "## Mean scores over three seeds",
        "",
        "| Model | Context | Payoff AP | New delinquency AP | Macro-F1 | Payoff Brier | Precision at top 1% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for (name, history), frame in table.groupby(["model", "history"], sort=False):
        precision = (
            "—"
            if name == "persistence"
            else f"{frame.precision_at_1_percent.mean():.2%}"
        )
        lines.append(
            f"| {name} | {history} | {frame.payoff_ap.mean():.5f} | {frame.new_delinquency_ap.mean():.5f} | {frame.macro_f1.mean():.4f} | {frame.payoff_brier.mean():.6f} | {precision} |"
        )
    lines.extend(
        [
            "",
            "AP and macro-F1: higher is better. Brier: lower is better. New-delinquency AP is measured among currently current loans; persistence gives all these loans the same zero delinquency-risk score, so its AP equals event prevalence. Persistence also assigns zero payoff probability to every loan: it cannot prioritize payoffs, and a top-one-percent payoff selection would be arbitrary because all scores tie.",
            "",
            "## Paired payoff-ranking effect",
            "",
            "| Model | Seed 522 AP change | Seed 523 AP change | Seed 524 AP change | Mean change | Positive seeds |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in models:
        values = (
            paired_table[
                (paired_table.model == name) & (paired_table.metric == "payoff_ap")
            ]
            .set_index("seed")
            .difference_24_minus_12
        )
        lines.append(
            f"| {name} | {values[522]:+.5f} | {values[523]:+.5f} | {values[524]:+.5f} | {values.mean():+.5f} | {int((values > 0).sum())}/3 |"
        )
    original = paired_table[
        (paired_table.model == "transformer") & (paired_table.metric == "payoff_ap")
    ].difference_24_minus_12
    wins = int((original > 0).sum())
    lines.extend(
        [
            "",
            f"The ordinary transformer's payoff AP improves in **{wins} of 3 seeds**, with mean change **{original.mean():+.5f}**.",
            "",
        ]
    )
    if wins == 3:
        lines.append(
            "Longer context consistently helps the focused transformer endpoint across these three seeds. Check the absolute scores and XGBoost comparison before claiming an advantage over other models."
        )
    elif wins == 0:
        lines.append(
            "Longer context does not help the ordinary transformer's payoff-ranking endpoint in these runs. Retain the shorter context as the current reference rather than presenting the larger window as an improvement."
        )
    else:
        lines.append(
            "The longer-context effect varies by seed. This does not establish a reliable transformer improvement, even if the mean change is positive."
        )
    lines.extend(
        [
            "",
            "Persistence's predictions and metrics match exactly across context lengths. Changes in persistence scores on other experiments would reflect different evaluated loans/months or forecast horizons, not learning from additional past observations.",
            "",
            "Full per-seed metrics, yearly payoff ranking, paired differences, and effective window selection are saved alongside this report. Three seeds describe limited optimization variability; they do not establish a precise population confidence interval or independent confirmation. Longer learned positional embeddings add parameters and alter seeded initialization. Rare default-like exits remain poorly supported.",
            "",
            "![Past-context comparison](payoff_ap_by_context.png)",
            "",
            "## Reproduction",
            "",
            "```bash",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/run_context_experiment.py",
            "MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv-gpu/bin/python analysis/compare_context_lengths.py",
            "```",
            "",
            "The driver skips only completed runs whose configurations match its protocol. Existing incomplete run directories are not overwritten. GPU access is required outside the agent sandbox; XGBoost uses `device=cuda` with `tree_method=hist`.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    write_json(
        output / "validation.json",
        {
            "population_signatures_match": True,
            "prediction_identities_match": True,
            "static_preprocessing_matches": True,
            "persistence_predictions_identical": True,
            "xgboost_cuda_verified": True,
            "checkpoint_history_verified": True,
            "contexts": [12, 24],
            "seeds": [522, 523, 524],
        },
    )
    figure, ax = plt.subplots(figsize=(8, 5))
    for name in models[1:]:
        for seed in [522, 523, 524]:
            points = table[(table.model == name) & (table.seed == seed)].sort_values(
                "history"
            )
            ax.plot(
                points.history,
                points.payoff_ap,
                alpha=0.3,
                color={
                    "xgboost": "#6b7280",
                    "transformer": "#2563eb",
                    "transformer_conditioned": "#d97706",
                }[name],
            )
        means = table[table.model == name].groupby("history").payoff_ap.mean()
        ax.plot(
            means.index,
            means.values,
            marker="o",
            linewidth=2,
            label=name,
            color={
                "xgboost": "#6b7280",
                "transformer": "#2563eb",
                "transformer_conditioned": "#d97706",
            }[name],
        )
    ax.set(
        xticks=[12, 24],
        xlabel="Past months supplied to the model",
        ylabel="Next-month payoff average precision",
        title="Context length on identical eligible prediction targets",
    )
    ax.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "payoff_ap_by_context.png", dpi=180)
    plt.close(figure)
    print(f"Saved {output / 'RESULTS.md'}", flush=True)


if __name__ == "__main__":
    main()
