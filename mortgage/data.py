"""Indexed histories and training-only preprocessing."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray

from .prepare import select_windows
from .schema import DYNAMIC_NUMERIC, STATIC_CATEGORICAL, STATIC_NUMERIC, month_numbers

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


class NumericParameters(TypedDict):
    median: list[float]
    mean: list[float]
    scale: list[float]


class Preprocessing(TypedDict):
    dynamic: NumericParameters
    static: NumericParameters
    vocabularies: dict[str, list[str]]
    dynamic_features: list[str]
    static_numeric: list[str]
    static_categorical: list[str]
    balance_transform: str
    missing_indicators: bool
    training_input_rows: int
    training_loans: int


def encode_categories(
    loans: pd.DataFrame, vocabularies: dict[str, list[str]]
) -> FloatArray:
    columns = []
    for name in STATIC_CATEGORICAL:
        vocabulary = vocabularies[name]
        mapping = {value: index + 1 for index, value in enumerate(vocabulary)}
        codes = loans[name].map(mapping).fillna(0).to_numpy(dtype=np.int64)
        columns.append(np.eye(len(vocabulary) + 1, dtype=np.float32)[codes])
    return np.concatenate(columns, axis=1)


def fit_numeric(values: FloatArray) -> NumericParameters:
    # Explicit per-feature handling avoids NaN statistics for entirely absent fields.
    medians = np.array(
        [
            np.median(col[np.isfinite(col)]) if np.isfinite(col).any() else 0
            for col in values.T
        ],
        dtype=np.float32,
    )
    filled = np.where(np.isfinite(values), values, medians)
    return {
        "median": medians.tolist(),
        "mean": filled.mean(axis=0).tolist(),
        "scale": np.maximum(filled.std(axis=0), 1e-6).tolist(),
    }


def scale_numeric(values: FloatArray, params: NumericParameters) -> FloatArray:
    missing = ~np.isfinite(values)
    filled = np.where(missing, np.asarray(params["median"], dtype=np.float32), values)
    scaled = (filled - np.asarray(params["mean"], dtype=np.float32)) / np.asarray(
        params["scale"], dtype=np.float32
    )
    return np.concatenate([scaled, missing.astype(np.float32)], axis=1)


class LoanData:
    """Monthly arrays shared by all windows; static attributes stored once per loan."""

    def __init__(
        self,
        prepared: Path,
        years: list[int],
        history: int = 12,
        eligibility_history: int | None = None,
    ):
        if len(set(years)) != len(years):
            raise ValueError("Duplicate cohort years")
        self.history = history
        self.eligibility_history = (
            history if eligibility_history is None else eligibility_history
        )
        if history < 1 or self.eligibility_history < history:
            raise ValueError(
                "Eligibility history must be at least the positive model history length"
            )
        self.years = sorted(years)
        monthly_parts, loan_parts, target_parts = [], [], []
        self.summaries = []
        self.window_summaries = []
        row_offset = loan_offset = 0
        for year in self.years:
            directory = prepared / str(year)
            summary = json.loads((directory / "summary.json").read_text())
            self.summaries.append(summary)
            frame = pd.read_parquet(directory / "monthly.parquet")
            loans = pd.read_parquet(directory / "loans.parquet")
            if summary["history"] == self.eligibility_history:
                targets = np.load(directory / "targets.npy")
                exclusions = summary.get("window_exclusions", {})
            else:
                targets, exclusions = select_windows(
                    frame.loan_index.to_numpy(),
                    frame.month.to_numpy(),
                    frame.state.to_numpy(),
                    self.eligibility_history,
                )
            self.window_summaries.append(
                {
                    "cohort": year,
                    "model_history": history,
                    "eligibility_history": self.eligibility_history,
                    "windows": len(targets),
                    "exclusions": exclusions,
                }
            )
            frame["loan_index"] += loan_offset
            monthly_parts.append(frame)
            loan_parts.append(loans)
            target_parts.append(targets + row_offset)
            row_offset += len(frame)
            loan_offset += len(loans)
        if len({summary["seed"] for summary in self.summaries}) != 1:
            raise ValueError("Prepared years use inconsistent split seeds")
        monthly = pd.concat(monthly_parts, ignore_index=True)
        self.loans = pd.concat(loan_parts, ignore_index=True)
        if self.loans.loan_id.duplicated().any():
            raise ValueError("Loan IDs overlap across cohorts")
        self.targets = np.concatenate(target_parts)
        self.row_loan = monthly.loan_index.to_numpy(dtype=np.int64)
        self.months = monthly.month.to_numpy(dtype=np.int32)
        self.states = monthly.state.to_numpy(dtype=np.int64)
        self.dynamic_raw = monthly[DYNAMIC_NUMERIC].to_numpy(
            dtype=np.float32, copy=True
        )
        self.dynamic_raw[:, 0] = np.log1p(self.dynamic_raw[:, 0])
        self.static_raw = self.loans[STATIC_NUMERIC].to_numpy(
            dtype=np.float32, copy=True
        )
        self.static_raw[:, STATIC_NUMERIC.index("original_upb")] = np.log1p(
            self.static_raw[:, STATIC_NUMERIC.index("original_upb")]
        )
        self.labels = self.states[self.targets]
        self.current = self.states[self.targets - 1]
        self.target_loans = self.row_loan[self.targets]
        self.cohorts = self.loans.cohort.to_numpy(dtype=np.int32)[self.target_loans]
        self.target_months = self.months[self.targets]
        self.split = self.loans.split.to_numpy(dtype=np.int8)[self.target_loans]
        self.dynamic = np.empty((0, 0), dtype=np.float32)
        self.static = np.empty((0, 0), dtype=np.float32)

    def partitions(
        self,
        train_end: str | None = None,
        validation_end: str | None = None,
        test_cohorts: list[int] | None = None,
    ) -> tuple[IntArray, IntArray, IntArray]:
        if bool(train_end) != bool(validation_end):
            raise ValueError("Supply both --train-end and --validation-end")
        masks = [self.split == index for index in range(3)]
        if train_end and validation_end:
            cutoffs = month_numbers(
                pd.Series([train_end.replace("-", ""), validation_end.replace("-", "")])
            )
            if cutoffs[0] >= cutoffs[1]:
                raise ValueError("Training cutoff must precede validation cutoff")
            masks[0] &= self.target_months <= cutoffs[0]
            masks[1] &= (self.target_months > cutoffs[0]) & (
                self.target_months <= cutoffs[1]
            )
            masks[2] &= self.target_months > cutoffs[1]
        if test_cohorts:
            masks[2] &= np.isin(self.cohorts, test_cohorts)
        partitions = (
            np.flatnonzero(masks[0]),
            np.flatnonzero(masks[1]),
            np.flatnonzero(masks[2]),
        )
        if any(len(part) == 0 for part in partitions):
            raise ValueError("An evaluation partition has no eligible examples")
        return partitions

    def fit_preprocessing(self, train: IntArray) -> Preprocessing:
        # Union of selected training input intervals; target/future rows are excluded.
        targets = self.targets[train]
        difference = np.zeros(len(self.row_loan) + 1, dtype=np.int32)
        np.add.at(difference, targets - self.history, 1)
        np.add.at(difference, targets, -1)
        input_mask = np.cumsum(difference[:-1]) > 0
        train_loans = np.unique(self.target_loans[train])
        dynamic_params = fit_numeric(self.dynamic_raw[input_mask])
        static_params = fit_numeric(self.static_raw[train_loans])
        self.dynamic = scale_numeric(self.dynamic_raw, dynamic_params)
        numeric = scale_numeric(self.static_raw, static_params)
        vocabularies = {}
        for name in STATIC_CATEGORICAL:
            vocabulary = sorted(self.loans.iloc[train_loans][name].unique().tolist())
            vocabularies[name] = vocabulary
        self.static = np.concatenate(
            [numeric, encode_categories(self.loans, vocabularies)], axis=1
        )
        return {
            "dynamic": dynamic_params,
            "static": static_params,
            "vocabularies": vocabularies,
            "dynamic_features": DYNAMIC_NUMERIC,
            "static_numeric": STATIC_NUMERIC,
            "static_categorical": STATIC_CATEGORICAL,
            "balance_transform": "log1p",
            "missing_indicators": True,
            "training_input_rows": int(input_mask.sum()),
            "training_loans": len(train_loans),
        }

    def apply_preprocessing(self, params: Preprocessing) -> None:
        if (
            params["static_numeric"] != STATIC_NUMERIC
            or params["dynamic_features"] != DYNAMIC_NUMERIC
        ):
            raise ValueError(
                "Checkpoint features do not match the prepared data schema"
            )
        self.dynamic = scale_numeric(self.dynamic_raw, params["dynamic"])
        self.static = np.concatenate(
            [
                scale_numeric(self.static_raw, params["static"]),
                encode_categories(self.loans, params["vocabularies"]),
            ],
            axis=1,
        )

    def population_signature(self, indices: IntArray) -> str:
        """Fingerprint ordered loan/month/label identities, independent of row offsets."""
        records = pd.DataFrame(
            {
                "loan_id": self.loans.loan_id.to_numpy()[self.target_loans[indices]],
                "month": self.target_months[indices],
                "label": self.labels[indices],
            }
        )
        hashes = pd.util.hash_pandas_object(records, index=False).to_numpy()
        return hashlib.sha256(hashes.tobytes()).hexdigest()

    def batch(
        self, indices: IntArray, history: int | None = None, state_only: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = self.history if history is None else history
        if length < 1 or length > self.eligibility_history:
            raise ValueError("Requested input history exceeds the eligible context")
        targets = self.targets[indices]
        rows = targets[:, None] + np.arange(-length, 0)
        dynamic = (
            np.eye(4, dtype=np.float32)[self.states[rows]]
            if state_only
            else self.dynamic[rows]
        )
        return (
            torch.from_numpy(dynamic),
            torch.from_numpy(self.static[self.target_loans[indices]]),
            torch.from_numpy(self.labels[indices]),
        )

    def history_states(self, indices: IntArray, history: int | None = None) -> IntArray:
        length = self.history if history is None else history
        if length < 1 or length > self.eligibility_history:
            raise ValueError("Requested input history exceeds the eligible context")
        rows = self.targets[indices, None] + np.arange(-length, 0)
        return self.states[rows]

    def flat(
        self,
        indices: IntArray,
        history: int | None = None,
        state_indicators: bool = False,
    ) -> FloatArray:
        dynamic, static, _ = self.batch(indices, history)
        columns = [dynamic.numpy().reshape(len(indices), -1), static.numpy()]
        if state_indicators:
            columns.append(
                np.eye(4, dtype=np.float32)[
                    self.history_states(indices, history)
                ].reshape(len(indices), -1)
            )
        return np.concatenate(columns, axis=1)


def sample_indices(indices: IntArray, limit: int, seed: int) -> IntArray:
    if limit < 0:
        raise ValueError("Sample limits must be nonnegative; use 0 for all examples")
    if limit and len(indices) > limit:
        return np.sort(
            np.random.default_rng(seed).choice(indices, limit, replace=False)
        )
    return indices


def nested_sample_indices(
    indices: IntArray, limit: int, seed: int, anchor: int
) -> IntArray:
    """Preserve the original anchor sample and add a fixed ordering of unused targets."""
    if anchor < 1 or limit < 0 or (limit and limit < min(anchor, len(indices))):
        raise ValueError(
            "Nested sample limits must be zero/all or at least the positive anchor"
        )
    if not limit or limit >= len(indices):
        return indices
    base = sample_indices(indices, anchor, seed)
    if limit == len(base):
        return base
    remaining = np.setdiff1d(indices, base, assume_unique=True)
    tail = np.random.default_rng(seed + 1).permutation(remaining)
    return np.sort(np.concatenate([base, tail[: limit - len(base)]]))
