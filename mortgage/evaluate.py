"""Comparable metrics and aggregate figures."""

from pathlib import Path
from typing import TypedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .data import FloatArray, IntArray, LoanData
from .prepare import write_json
from .schema import LABELS


def macro_f1(labels: IntArray, probabilities: FloatArray) -> float:
    """Fixed-seven-class macro-F1, including zero for absent classes."""
    predicted = probabilities.argmax(axis=1)
    matrix = np.bincount(
        labels * len(LABELS) + predicted, minlength=len(LABELS) ** 2
    ).reshape(len(LABELS), len(LABELS))
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    values = np.divide(
        2 * matrix.diagonal(),
        denominator,
        out=np.zeros(len(LABELS), dtype=float),
        where=denominator > 0,
    )
    return float(values.mean())


def payoff_scores(frame: pd.DataFrame) -> dict[str, float | int]:
    """Rank payoff/maturity risk at a fixed one-percent loan-month budget."""
    actual = frame.actual.to_numpy() == 4
    probability = frame.p_payoff.to_numpy()
    count = max(1, int(len(frame) * 0.01))
    selected = np.argsort(-probability, kind="stable")[:count]
    return {
        "payoff_ap": (
            float(average_precision_score(actual, probability)) if actual.any() else 0.0
        ),
        "payoff_brier": float(np.square(actual - probability).mean()),
        "payoff_mean_probability": float(probability.mean()),
        "observed_payoff_rate": float(actual.mean()),
        "top_1_percent_budget": count,
        "top_1_percent_payoffs": int(actual[selected].sum()),
        "precision_at_1_percent": float(actual[selected].mean()),
    }


class ClassScore(TypedDict):
    precision: float
    recall: float
    f1: float
    support: int


class Scores(TypedDict):
    examples: int
    accuracy: float
    macro_f1: float
    supported_macro_f1: float
    log_loss: float | None
    per_class: dict[str, ClassScore]
    confusion_matrix: list[list[int]]


class TransitionScore(TypedDict):
    examples: int
    accuracy: float | None


class ModelReport(TypedDict):
    model: str
    validation: Scores
    test: Scores
    groups: dict[str, dict[str, Scores]]
    transitions: dict[str, TransitionScore]
    seconds: float


class ComparisonRow(TypedDict):
    model: str
    split: str
    examples: int
    macro_f1: float
    supported_macro_f1: float
    accuracy: float
    log_loss: float | None
    seconds: float


def metrics(
    labels: IntArray, probabilities: FloatArray, predictions: IntArray | None = None
) -> Scores:
    if not len(labels):
        raise ValueError("Cannot score an empty partition")
    if probabilities.shape != (len(labels), len(LABELS)):
        raise ValueError("Probability shape does not match labels")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one")
    predicted = probabilities.argmax(axis=1) if predictions is None else predictions
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=np.arange(len(LABELS)), zero_division=0
    )
    per_class: dict[str, ClassScore] = {
        name: {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s),
        }
        for name, p, r, f, s in zip(LABELS, precision, recall, f1, support)
    }
    log_loss = float(
        -np.log(
            np.clip(
                probabilities[np.arange(len(labels)), labels].astype(np.float64),
                1e-15,
                1,
            )
        ).mean()
    )
    return {
        "examples": len(labels),
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1.mean()),
        "supported_macro_f1": float(f1[support > 0].mean()),
        "log_loss": log_loss,
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=np.arange(len(LABELS))
        ).tolist(),
    }


def score_model(
    name: str,
    data: LoanData,
    validation: IntArray,
    test: IntArray,
    validation_prob: FloatArray,
    test_prob: FloatArray,
    output: Path,
    seconds: float,
    persistence: bool = False,
) -> ModelReport:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    val_predictions = data.current[validation] if persistence else None
    predictions = data.current[test] if persistence else test_prob.argmax(axis=1)
    validation_metrics = metrics(
        data.labels[validation], validation_prob, val_predictions
    )
    test_metrics = metrics(data.labels[test], test_prob, predictions)
    # Persistence is a hard-label baseline, not a probabilistic model. The caller
    # supplies one-hot arrays only for common scoring; do not report their log loss.
    if persistence:
        validation_metrics["log_loss"] = None
        test_metrics["log_loss"] = None
    labels = data.labels[test]
    current = data.current[test]
    groups: dict[str, dict[str, Scores]] = {}
    for group_name, values in [
        ("prediction_year", data.target_months[test] // 12),
        ("origination_cohort", data.cohorts[test]),
    ]:
        groups[group_name] = {}
        for value in np.unique(values):
            mask = values == value
            result = metrics(labels[mask], test_prob[mask], predictions[mask])
            if persistence:
                result["log_loss"] = None
            groups[group_name][str(value)] = result
    transitions: dict[str, TransitionScore] = {}
    for transition, mask in {
        "unchanged": labels == current,
        "worsening": (labels < 4) & (labels > current),
        "recovery": (labels < current) & (labels < 4),
        "terminal": labels >= 4,
    }.items():
        transitions[transition] = {
            "examples": int(mask.sum()),
            "accuracy": (
                float((predictions[mask] == labels[mask]).mean())
                if mask.any()
                else None
            ),
        }
    report: ModelReport = {
        "model": name,
        "validation": validation_metrics,
        "test": test_metrics,
        "groups": groups,
        "transitions": transitions,
        "seconds": seconds,
    }
    write_json(directory / "metrics.json", report)
    predictions_frame = pd.DataFrame(
        {
            "example_index": test,
            "cohort": data.cohorts[test],
            "target_month": data.target_months[test],
            "current": current,
            "actual": labels,
            "predicted": predictions,
        }
    )
    for index, label in enumerate(LABELS):
        predictions_frame[f"p_{label}"] = test_prob[:, index]
    predictions_frame.to_parquet(directory / "predictions.parquet", index=False)
    matrix = np.asarray(test_metrics["confusion_matrix"])
    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(9, 7))
    plot = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted state",
        ylabel="Actual state",
        title=f"{name}: row-normalized confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for row in range(len(LABELS)):
        for column in range(len(LABELS)):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if normalized[row, column] > 0.5 else "black",
            )
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(directory / "confusion.png", dpi=160)
    plt.close(fig)
    print(
        f"{name}: test macro-F1={test_metrics['macro_f1']:.4f}, accuracy={test_metrics['accuracy']:.4f}",
        flush=True,
    )
    return report


def write_comparison(output: Path, reports: list[ModelReport]) -> None:
    # Read JSON into a frame to keep numeric report construction simple.
    rows: list[ComparisonRow] = []
    by_year = []
    for report in reports:
        for split, scores in [
            ("validation", report["validation"]),
            ("test", report["test"]),
        ]:
            rows.append(
                {
                    "model": report["model"],
                    "split": split,
                    "examples": scores["examples"],
                    "macro_f1": scores["macro_f1"],
                    "supported_macro_f1": scores["supported_macro_f1"],
                    "accuracy": scores["accuracy"],
                    "log_loss": scores["log_loss"],
                    "seconds": report["seconds"],
                }
            )
        for year, result in report["groups"]["prediction_year"].items():
            by_year.append(
                {
                    "model": report["model"],
                    "year": int(year),
                    "macro_f1": result["macro_f1"],
                    "examples": result["examples"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output / "comparison.csv", index=False)
    lines = [
        "# Model comparison",
        "",
        "Macro-F1 uses all seven predefined classes, assigning zero to absent classes.",
        "Supported macro-F1 averages only classes with actual examples; compare like-for-like test sets.",
        "Persistence has no probabilistic log loss. Runtime includes fitting and evaluation.",
        "",
        "| Model | Split | Examples | Macro-F1 | Supported macro-F1 | Accuracy | Log loss |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        loss = f"{row['log_loss']:.5f}" if row["log_loss"] is not None else "—"
        lines.append(
            f"| {row['model']} | {row['split']} | {row['examples']:,} | {row['macro_f1']:.4f} | {row['supported_macro_f1']:.4f} | {row['accuracy']:.4f} | {loss} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    years = pd.DataFrame(by_year)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, frame in years.groupby("model"):
        frame = frame.sort_values("year")
        ax.plot(frame.year, frame.macro_f1, marker="o", label=name)
    ax.set(
        xlabel="Prediction year",
        ylabel="Macro-F1 (fixed seven classes)",
        title="Held-out performance over time",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "performance_by_year.png", dpi=160)
    plt.close(fig)
