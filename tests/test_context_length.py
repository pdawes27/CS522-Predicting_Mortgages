"""Matched-history eligibility and baseline checks."""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from analysis.diagnose_errors import loan_clusters
from mortgage.data import LoanData, sample_indices
from mortgage.prepare import prepare_year
from test_pipeline import make_raw


def test_common_eligibility_keeps_targets_and_persistence_fixed(tmp_path: Path):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    make_raw(raw, 2018)
    # The fixture has 24 records per loan, enough for matched 6/12-month contexts.
    prepare_year(raw, prepared, 2018, 522, 6, "synthetic", "test")
    short = LoanData(prepared, [2018], history=6, eligibility_history=12)
    long = LoanData(prepared, [2018], history=12, eligibility_history=12)
    np.testing.assert_array_equal(short.targets, long.targets)
    for left, right in zip(short.partitions(), long.partitions()):
        np.testing.assert_array_equal(left, right)
        assert short.population_signature(left) == long.population_signature(right)
    train = sample_indices(short.partitions()[0], 5, 522)
    short.fit_preprocessing(train)
    long.fit_preprocessing(train)
    np.testing.assert_array_equal(short.current, long.current)
    # XGBoost gets six additional monthly vectors; origination columns are shared.
    assert long.flat(train).shape[1] - short.flat(train).shape[1] == 6 * 8
    np.testing.assert_array_equal(
        short.flat(train)[:, -short.static.shape[1] :],
        long.flat(train)[:, -long.static.shape[1] :],
    )
    assert short.batch(train)[0].shape[1] == 6
    assert long.batch(train)[0].shape[1] == 12
    target_rows = short.targets[train]
    history_rows = target_rows[:, None] + np.arange(-12, 0)
    np.testing.assert_array_equal(
        long.states[history_rows[:, -6:]],
        short.states[target_rows[:, None] + np.arange(-6, 0)],
    )
    # Context-specific scaling sees only the selected training input rows.
    assert (
        long.fit_preprocessing(train)["training_input_rows"]
        >= short.fit_preprocessing(train)["training_input_rows"]
    )


def test_invalid_eligibility_fails_before_reading_files(tmp_path):
    with pytest.raises(ValueError, match="Eligibility history"):
        LoanData(tmp_path, [2018], history=24, eligibility_history=12)


def test_cluster_mapping_uses_effective_window_indices(tmp_path):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    make_raw(raw, 2018)
    prepare_year(raw, prepared, 2018, 522, 6, "synthetic", "test")
    data = LoanData(prepared, [2018], history=6, eligibility_history=12)
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.json").write_text(
        json.dumps(
            {
                "prepared": str(prepared),
                "years": [2018],
                "history": 6,
                "eligibility_history": 12,
            }
        )
    )
    indices = np.arange(len(data.targets), dtype=np.int64)
    expected = np.unique(data.target_loans, return_inverse=True)[1]
    np.testing.assert_array_equal(loan_clusters(run, indices), expected)


def test_longer_eligibility_rejects_internal_reporting_gaps(tmp_path):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    make_raw(raw, 2018)
    prepare_year(raw, prepared, 2018, 522, 6, "synthetic", "test")
    path = prepared / "2018" / "monthly.parquet"
    frame = pd.read_parquet(path)
    # Remove a month outside the most recent six months but inside twelve.
    first_loan = frame.loan_index.iloc[0]
    missing_month = frame.loc[frame.loan_index == first_loan, "month"].iloc[8]
    frame = frame.loc[
        ~((frame.loan_index == first_loan) & (frame.month == missing_month))
    ]
    frame.to_parquet(path, index=False)
    data = LoanData(prepared, [2018], history=6, eligibility_history=12)
    assert data.window_summaries[0]["exclusions"]["reporting_gap"] > 0
    targets = data.targets[data.target_loans == first_loan]
    assert np.all(data.months[targets] - data.months[targets - 12] == 12)
