"""Build an aggregate report from completed runs."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .schema import LABELS


def build_report(prepared: Path, runs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    summaries = [
        json.loads(path.read_text()) for path in sorted(prepared.glob("*/summary.json"))
    ]
    if not summaries:
        raise ValueError("No prepared dataset summaries found")
    lines = [
        "# Mortgage outcome experiment report",
        "",
        "Prepared from local data-quality summaries and completed experiment artifacts.",
        "The project authors should review this draft and verify the related-work references before submission.",
        "",
        "## Research question",
        "",
        "Does a small transformer using 12 months of history improve next-month loan-state predictions over persistence and XGBoost?",
        "",
        "## Data inventory",
        "",
        "| Origination year | Loans | Raw monthly rows | Eligible windows | Reporting coverage | ZIP MB | Extracted MB |",
        "|---|---:|---:|---:|---|---:|---:|",
    ]
    counts = np.zeros(len(LABELS), dtype=np.int64)
    for summary in summaries:
        extracted = sum(source["bytes"] for source in summary["sources"])
        compressed = summary["compressed_bytes"]
        compressed_text = (
            f"{compressed / 1e6:.1f}" if compressed is not None else "unknown"
        )
        lines.append(
            f"| {summary['cohort']} | {summary['loans']:,} | {summary['performance_rows_raw']:,} | {summary['windows']:,} | {' to '.join(summary['date_range'])} | {compressed_text} | {extracted / 1e6:.1f} |"
        )
        counts += summary["window_class_counts"]
    total_loans = sum(summary["loans"] for summary in summaries)
    total_rows = sum(summary["performance_rows_raw"] for summary in summaries)
    gap_windows = sum(
        summary.get("window_exclusions", {}).get("reporting_gap", 0)
        for summary in summaries
    )
    after_exit = sum(summary["dropped_after_first_exit"] for summary in summaries)
    lines.extend(
        [
            "",
            f"Total: {total_loans:,} loans and {total_rows:,} raw monthly records. Cohort samples have equal loan counts, not equal monthly record counts.",
            "",
            f"Preparation removed {after_exit:,} rows after first exit/REO and excluded {gap_windows:,} windows spanning reporting gaps. Other exclusions and missingness are recorded per cohort in the quality manifests.",
            "",
            "The downloaded release label and exact download date are unconfirmed. File SHA-256 checksums and the observed column layout identify the local inputs.",
            "",
            "| Outcome | Eligible windows | Fraction |",
            "|---|---:|---:|",
        ]
    )
    for label, count in zip(LABELS, counts):
        lines.append(f"| {label} | {count:,} | {count / counts.sum():.6%} |")
    figure, ax = plt.subplots(figsize=(10, 5))
    ax.bar(LABELS, counts)
    ax.set_yscale("log")
    ax.set(
        ylabel="Eligible windows (log scale)",
        title="Outcome support across downloaded cohorts",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    figure.tight_layout()
    figure.savefig(output.parent / "class_distribution.png", dpi=160)
    plt.close(figure)
    lines.extend(
        [
            "",
            "![Outcome support](class_distribution.png)",
            "",
            "## Methods",
            "",
            "Loans are assigned reproducibly to approximately 70%/15%/15% training/validation/test partitions. All models in an experiment use the same examples. Numeric preprocessing and categorical vocabularies are fitted only on training inputs. Windows require 12 consecutive active observations and an observed next-month target; histories stop at the first exit/REO event. See [methodology](../docs/methodology.md) for exact label mappings and feature rules.",
            "",
            "Transformer: two encoder layers, four attention heads, hidden size 64 by default, with origination features joined at the classifier. Early stopping minimizes natural-distribution validation log loss. The one-month ablation retains the same target population. Class weighting uses training counts only.",
            "",
            "Macro-F1 below always uses the fixed seven classes. Persistence has no probabilistic log loss. Counts, rare-class precision/recall, supported-class macro-F1, transition errors, and group metrics are available in each run's local artifacts.",
            "",
            "## Completed experiments",
            "",
        ]
    )
    comparison_rows = []
    for run in runs:
        if not (run / "resources.json").exists():
            raise ValueError(f"Run is incomplete: {run}")
        config = json.loads((run / "config.json").read_text())
        resources = json.loads((run / "resources.json").read_text())
        table = pd.read_csv(run / "comparison.csv")
        class_counts = pd.read_csv(run / "class_counts.csv")
        training_count = int(class_counts.train.sum())
        validation_count = int(class_counts.validation.sum())
        test_count = int(class_counts.test.sum())
        description = (
            f"Cohorts: {', '.join(map(str, config['years']))}; model seed {config['seed']}; "
            f"{training_count:,} training, {validation_count:,} validation, and {test_count:,} test windows. "
            f"Runtime {resources['wall_seconds'] / 60:.1f} minutes; peak process RSS {resources['peak_rss_gib']:.2f} GiB."
        )
        if config.get("train_end"):
            description += f" Training targets through {config['train_end']}; validation through {config['validation_end']}; test targets thereafter."
        else:
            description += " Loan-held-out evaluation across available reporting years; this is not a future-only test."
        if config.get("test_cohorts"):
            description += f" Test cohorts restricted to {config['test_cohorts']}."
        lines.extend(
            [
                f"### {run.name}",
                "",
                description,
                "",
                "| Model | Validation log loss | Test macro-F1 | Test accuracy | Test log loss |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model in table.model.unique():
            valid = table[(table.model == model) & (table.split == "validation")].iloc[
                0
            ]
            test = table[(table.model == model) & (table.split == "test")].iloc[0]
            val_loss = "—" if pd.isna(valid.log_loss) else f"{valid.log_loss:.5f}"
            test_loss = "—" if pd.isna(test.log_loss) else f"{test.log_loss:.5f}"
            lines.append(
                f"| {model} | {val_loss} | {test.macro_f1:.4f} | {test.accuracy:.4f} | {test_loss} |"
            )
            comparison_rows.append(
                {
                    "run": run.name,
                    "model": model,
                    "macro_f1": test.macro_f1,
                    "accuracy": test.accuracy,
                    "log_loss": test.log_loss,
                    "test_examples": test_count,
                    "seed": config["seed"],
                }
            )
        rare = class_counts[
            class_counts.label.isin(["adverse_exit_or_reo", "other_exit"])
        ]
        support_text = "; ".join(
            f"{row.label}: train={row.train}, validation={row.validation}, test={row.test}"
            for row in rare.itertuples()
        )
        lines.extend(["", "Rare-event support: " + support_text + ".", ""])
        probabilistic = table[(table.split == "validation") & table.log_loss.notna()]
        best_name = (
            str(probabilistic.sort_values("log_loss").iloc[0].model)
            if len(probabilistic)
            else "persistence"
        )
        best = json.loads((run / best_name / "metrics.json").read_text())
        confusion_name = f"{run.name}_selected_confusion.png"
        shutil.copyfile(
            run / best_name / "confusion.png", output.parent / confusion_name
        )
        lines.extend(
            [
                f"Lowest validation log loss selects **{best_name}** among this run's probabilistic candidates.",
                "",
                "| Outcome | Test support | Precision | Recall | F1 |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in LABELS:
            score = best["test"]["per_class"][label]
            lines.append(
                f"| {label} | {score['support']:,} | {score['precision']:.4f} | {score['recall']:.4f} | {score['f1']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| Actual transition | Test examples | Selected model accuracy |",
                "|---|---:|---:|",
            ]
        )
        for name, score in best["transitions"].items():
            accuracy = "—" if score["accuracy"] is None else f"{score['accuracy']:.4f}"
            lines.append(f"| {name} | {score['examples']:,} | {accuracy} |")
        lines.append("")
        lines.extend(
            [
                f"![{run.name}: validation-selected model confusion matrix]({confusion_name})",
                "",
            ]
        )
        curves = sorted(run.glob("*/learning_curve.json"))
        if curves:
            figure, ax = plt.subplots(figsize=(8, 4))
            for path in curves:
                curve = pd.read_json(path)
                ax.plot(
                    curve.step if config.get("max_steps") else curve.epoch,
                    curve.validation_log_loss,
                    marker="o",
                    label=path.parent.name,
                )
            ax.set(
                xlabel="Optimizer update" if config.get("max_steps") else "Epoch",
                ylabel="Validation log loss",
                title=f"{run.name}: checkpoint selection",
            )
            ax.legend(fontsize=8)
            figure.tight_layout()
            curve_name = f"{run.name}_learning_curves.png"
            figure.savefig(output.parent / curve_name, dpi=160)
            plt.close(figure)
            lines.extend([f"![Validation learning curves]({curve_name})", ""])
        rollout = run / "rollout-evaluation"
        if (rollout / "comparison.csv").exists():
            table = pd.read_csv(rollout / "comparison.csv")
            horizons = sorted(set([1, 3, 6, int(table.horizon.max())]))
            lines.extend(
                [
                    "#### Recursive forecasts",
                    "",
                    "State-only histories and fixed origination attributes allow recursive generation without assuming future balances or rates. Each generated state is fed back; paths stop at the first terminal event. The backtest uses one random origin per uniformly selected held-out loan and 128 sampled paths per origin in this run.",
                    "",
                    "| Months ahead | Model | Observed outcomes | Macro-F1 | Brier score |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for row in table[table.horizon.isin(horizons)].itertuples():
                lines.append(
                    f"| {row.horizon} | {row.model} | {row.examples:,} | {row.macro_f1:.4f} | {row.brier:.5f} |"
                )
            name = f"{run.name}_horizon_performance.png"
            shutil.copyfile(rollout / "horizon_performance.png", output.parent / name)
            lines.extend(
                [
                    "",
                    "Brier score measures probability error; lower is better. Monte Carlo log loss is omitted because finite trajectory samples produce zero-probability estimates. Known exits are absorbing for probability accounting; unobserved/censored outcomes are excluded separately at each horizon. Changing outcome support means differences across horizons are not solely measures of accumulated model error.",
                    "",
                    f"![Recursive forecast performance]({name})",
                    "",
                ]
            )
    pd.DataFrame(comparison_rows).to_csv(
        output.parent / "experiment_comparison.csv", index=False
    )
    lines.extend(
        [
            "## Interpretation and limitations",
            "",
            "Compare expansion runs only when the test cohorts, eligible windows, file fingerprints, date boundaries, and test sampling are identical. A fixed training-window cap measures expansion under a fixed compute/data budget, rather than training on every available example. Validation populations may differ between single-year and expanded runs; they are used only to choose each run's checkpoint.",
            "",
            "High accuracy can follow from mostly unchanged current loans. Focus on macro-F1, class support, worsening/recovery transitions, and rare-event recall. Adverse-exit counts may be too small for reliable conclusions. Do not interpret seven classes as seven ordered delinquency levels or claim to predict all stages of foreclosure.",
            "",
            "The fixed-history requirement excludes early outcomes and short histories. Long loans contribute more correlated windows. Equal cohort samples do not reproduce origination-volume weights. Historical snapshots include reporting revisions and do not guarantee real-time availability of all fields. Calendar-held-out results answer a different question from random loan-held-out results. Multiple optimization seeds are not independent samples of the underlying loan population.",
            "",
            "Recursive state forecasting is implemented and evaluated separately from the full-feature one-month predictor. It omits evolving balances, interest rates, and economic conditions; its simulated paths should not be interpreted as calibrated long-term mortgage default probabilities. Longer histories and directly supervised fixed-horizon prediction remain optional. Crisis cohorts were not supplied. Related-work titles, authors, and narrow summaries were checked against primary sources, but full-paper reading and author review remain necessary before academic submission.",
            "",
            "## Reproduction",
            "",
            "See [README](../README.md) for setup, preparation, training, future-period evaluation, and tests. Each local run saves its exact command options in config.json, library versions, source fingerprints, selected examples, loan assignments, preprocessing, fitted models, predictions, metrics, and resources. Licensed data and per-loan artifacts remain excluded from version control.",
            "",
        ]
    )
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output}", flush=True)
