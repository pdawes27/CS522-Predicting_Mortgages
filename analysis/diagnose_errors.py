"""Exploratory error analysis of saved forecasts.

Run: MPLCONFIGDIR=/tmp/mortgage-matplotlib .venv/bin/python analysis/diagnose_errors.py
No models or thresholds are fitted here. These tests have already been inspected;
the resulting hypotheses require validation-led follow-up experiments.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score

from mortgage.prepare import select_windows, write_json


def rank_groups(
    scores: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    order = np.argsort(-scores, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(scores) - 1]
    return order, ends


def weighted_ap(
    target: NDArray[np.bool_],
    order: NDArray[np.int64],
    ends: NDArray[np.int64],
    weights: NDArray[np.float64],
) -> float:
    weighted_positive = np.cumsum(weights[order] * target[order])[ends]
    weighted_count = np.cumsum(weights[order])[ends]
    precision = np.divide(
        weighted_positive,
        weighted_count,
        out=np.zeros_like(weighted_positive),
        where=weighted_count > 0,
    )
    increments = np.diff(np.r_[0.0, weighted_positive]) / weighted_positive[-1]
    return float((increments * precision).sum())


def loan_clusters(run: Path, examples: NDArray[np.int64]) -> NDArray[np.int64]:
    """Map saved window indices to loans without loading all feature arrays."""
    config = json.loads((run / "config.json").read_text())
    clusters = np.full(len(examples), -1, dtype=np.int64)
    window_offset = loan_offset = 0
    for year in sorted(config["years"]):
        directory = Path(config["prepared"]) / str(year)
        summary = json.loads((directory / "summary.json").read_text())
        eligibility = config.get("eligibility_history") or config.get("history", 12)
        monthly = None
        if summary["history"] == eligibility:
            targets = np.load(directory / "targets.npy", mmap_mode="r")
        else:
            monthly = pd.read_parquet(
                directory / "monthly.parquet", columns=["loan_index", "month", "state"]
            )
            targets, _ = select_windows(
                monthly.loan_index.to_numpy(),
                monthly.month.to_numpy(),
                monthly.state.to_numpy(),
                eligibility,
            )
        selected = (examples >= window_offset) & (
            examples < window_offset + len(targets)
        )
        if selected.any():
            rows = targets[examples[selected] - window_offset]
            row_loans = (
                pd.read_parquet(directory / "monthly.parquet", columns=["loan_index"])
                if monthly is None
                else monthly
            ).loan_index.to_numpy()
            clusters[selected] = row_loans[rows] + loan_offset
        window_offset += len(targets)
        loan_offset += summary["loans"]
    if (clusters < 0).any():
        raise ValueError("Unmapped test windows")
    return np.unique(clusters, return_inverse=True)[1]


def main() -> None:
    output = Path("reports/diagnostics")
    output.mkdir(parents=True, exist_ok=True)
    # Check tie handling and zero bootstrap weights against the library definition.
    target = np.array([True, False, True, False, True])
    scores = np.array([0.8, 0.8, 0.6, 0.2, 0.1])
    weights = np.array([2.0, 0.0, 1.0, 3.0, 1.0])
    order, ends = rank_groups(scores)
    np.testing.assert_allclose(
        weighted_ap(target, order, ends, weights),
        average_precision_score(target, scores, sample_weight=weights),
    )
    ranking_rows, event_rows, history_rows, calibration_rows, intervals = (
        [],
        [],
        [],
        [],
        {},
    )
    for run_name in ["2018-main", "future-period"]:
        run = Path("runs") / run_name
        frames = {
            name: pd.read_parquet(run / name / "predictions.parquet")
            for name in ["xgboost", "transformer"]
        }
        reference = frames["xgboost"]
        actual = reference.actual.to_numpy()
        current = reference.current.to_numpy()
        payoff = actual == 4
        if run_name == "2018-main":
            for name in ["transformer_current", "transformer", "xgboost"]:
                frame = (
                    frames[name]
                    if name in frames
                    else pd.read_parquet(run / name / "predictions.parquet")
                )
                selected = frame.current.to_numpy() == 0
                event = np.isin(frame.actual.to_numpy()[selected], [1, 2, 3])
                score = (
                    frame[["p_30_59_days", "p_60_89_days", "p_90_plus_days"]]
                    .sum(axis=1)
                    .to_numpy()[selected]
                )
                history_rows.append(
                    {
                        "model": name,
                        "context_months": 1 if name == "transformer_current" else 12,
                        "examples": int(selected.sum()),
                        "events": int(event.sum()),
                        "new_delinquency_average_precision": float(
                            average_precision_score(event, score)
                        ),
                    }
                )
        for name, frame in frames.items():
            np.testing.assert_array_equal(reference.example_index, frame.example_index)
            np.testing.assert_array_equal(actual, frame.actual)
            predicted = frame.predicted.to_numpy()
            probability = frame.p_payoff.to_numpy()
            calibration_rows.append(
                {
                    "run": run_name,
                    "model": name,
                    "observed_payoff_rate": float(payoff.mean()),
                    "mean_predicted_payoff_probability": float(probability.mean()),
                    "payoff_brier": float(np.square(probability - payoff).mean()),
                    "always_zero_brier": float(payoff.mean()),
                }
            )
            for event, actual_event, predicted_event in [
                ("recovery", (actual < current) & (actual < 4), predicted < current),
                (
                    "worsening",
                    (actual > current) & (actual < 4),
                    (predicted > current) & (predicted < 4),
                ),
            ]:
                true_positive = int((actual_event & predicted_event).sum())
                event_rows.append(
                    {
                        "run": run_name,
                        "model": name,
                        "event": event,
                        "actual_events": int(actual_event.sum()),
                        "predicted_events": int(predicted_event.sum()),
                        "precision": true_positive / max(1, int(predicted_event.sum())),
                        "recall": true_positive / max(1, int(actual_event.sum())),
                    }
                )
            for year in [None, *sorted((frame.target_month // 12).unique())]:
                selected = (
                    np.ones(len(frame), dtype=bool)
                    if year is None
                    else frame.target_month.to_numpy() // 12 == year
                )
                y = payoff[selected]
                probability = frame.p_payoff.to_numpy()[selected]
                budget = max(1, int(len(y) * 0.01))
                chosen = np.argsort(-probability, kind="stable")[:budget]
                ranking_rows.append(
                    {
                        "run": run_name,
                        "year": "all" if year is None else str(year),
                        "model": name,
                        "examples": len(y),
                        "payoffs": int(y.sum()),
                        "average_precision": float(
                            average_precision_score(y, probability)
                        ),
                        "review_budget": budget,
                        "payoffs_in_top_1_percent": int(y[chosen].sum()),
                        "precision_at_1_percent": float(y[chosen].mean()),
                        "recall_at_1_percent": float(y[chosen].sum() / y.sum()),
                    }
                )
        clusters = loan_clusters(run, reference.example_index.to_numpy())
        count = int(clusters.max()) + 1
        ranked = {
            name: rank_groups(frame.p_payoff.to_numpy(dtype=np.float64))
            for name, frame in frames.items()
        }
        generator = np.random.default_rng(522)
        differences = []
        for _ in range(200):
            cluster_weights = np.bincount(
                generator.integers(count, size=count), minlength=count
            )
            row_weights = cluster_weights[clusters].astype(np.float64)
            aps = {
                name: weighted_ap(payoff, order, ends, row_weights)
                for name, (order, ends) in ranked.items()
            }
            differences.append(aps["transformer"] - aps["xgboost"])
        intervals[run_name] = {
            "loans": count,
            "bootstrap_replicates": 200,
            "ap_difference_95_percentile_interval": np.quantile(
                differences, [0.025, 0.975]
            ).tolist(),
        }
        print(run_name, intervals[run_name], flush=True)
    ranking = pd.DataFrame(ranking_rows)
    ranking.to_csv(output / "payoff_ranking.csv", index=False)
    pd.DataFrame(event_rows).to_csv(output / "transition_detection.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output / "history_ablation.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output / "payoff_calibration.csv", index=False
    )
    write_json(
        output / "uncertainty.json",
        {
            "method": "Paired loan-cluster bootstrap, resampling whole loans with replacement; fixed fitted models",
            "status": "Exploratory: these test results were inspected to select the hypothesis. Intervals exclude optimization/tuning uncertainty.",
            "intervals": intervals,
        },
    )
    future = ranking[(ranking.run == "future-period") & (ranking.year != "all")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, frame in future.groupby("model"):
        axes[0].plot(frame.year, frame.average_precision, marker="o", label=name)
    axes[0].set(
        xlabel="Prediction year",
        ylabel="Payoff average precision",
        title="Future-period payoff ranking",
    )
    axes[0].legend()
    pooled = ranking[(ranking.run == "future-period") & (ranking.year == "all")]
    bars = axes[1].bar(
        pooled.model, pooled.payoffs_in_top_1_percent, color=["#6b7280", "#2563eb"]
    )
    axes[1].bar_label(bars)
    axes[1].set(
        ylabel="Actual payoffs among top-ranked windows",
        title="Same top-1% selection budget",
    )
    figure.tight_layout()
    figure.savefig(output / "payoff_advantage.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
