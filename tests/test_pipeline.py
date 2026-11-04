"""Leakage, labels, and boundary checks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from mortgage.data import LoanData, fit_numeric, scale_numeric
from mortgage.evaluate import metrics
from mortgage.models import MortgageTransformer
from mortgage.prepare import prepare_year, select_windows
from mortgage.schema import LABELS, month_numbers, outcome_states, split_for_loan


def test_outcomes_distinguish_missing_and_exits():
    frame = pd.DataFrame(
        {
            "delinquency": [
                "00",
                "01",
                "02",
                "12",
                "XX",
                "RA",
                "00",
                "99",
                "00",
                "-1",
                "1.5",
            ],
            "zero_code": ["", "", "", "", "", "", "01", "03", "96", "", ""],
        }
    )
    np.testing.assert_array_equal(
        outcome_states(frame), [0, 1, 2, 3, -1, 5, 4, 5, 6, -1, -1]
    )


def test_windows_reject_gaps_cross_loan_and_post_terminal_inputs():
    loans = np.array([0] * 8 + [1] * 5, dtype=np.int64)
    months = np.array([0, 1, 2, 3, 5, 6, 7, 8, 0, 1, 2, 3, 4], dtype=np.int32)
    states = np.array([0, 0, 1, 4, 0, 0, -1, 0, 0, 1, 2, 5, 0], dtype=np.int8)
    selected, exclusions = select_windows(loans, months, states, 2)
    np.testing.assert_array_equal(selected, [2, 3, 10, 11])
    assert sum(exclusions.values()) + len(selected) == len(loans)
    assert exclusions["reporting_gap"] == 2


def test_months_validate_and_cross_year():
    result = month_numbers(pd.Series(["202012", "202101"]))
    assert result[1] - result[0] == 1
    for invalid in ["202013", "202000", "2020-01", ""]:
        with pytest.raises(ValueError):
            month_numbers(pd.Series([invalid]))


def test_numeric_missing_columns_and_unseen_missing_values():
    train = np.array([[1, np.nan], [3, np.nan]], dtype=np.float32)
    params = fit_numeric(train)
    transformed = scale_numeric(np.array([[np.nan, 5]], dtype=np.float32), params)
    assert np.isfinite(transformed).all()
    assert transformed.shape == (1, 4)
    np.testing.assert_array_equal(transformed[:, 2:], [[1, 0]])


def make_raw(raw: Path, year: int) -> None:
    raw.mkdir(exist_ok=True)
    ids = []
    for split in range(3):
        ids.append(
            next(
                f"F{year % 100:02d}Q1{i:07d}"
                for i in range(1000)
                if split_for_loan(f"F{year % 100:02d}Q1{i:07d}") == split
            )
        )
    orig_rows, performance_rows = [], []
    for loan in ids:
        orig = [""] * 31
        for index, value in {
            0: "700",
            7: "P",
            9: "35",
            10: "200000",
            11: "80",
            12: "4",
            19: loan,
            20: "P",
            21: "360",
        }.items():
            orig[index] = value
        orig_rows.append("|".join(orig))
        for month in range(1, 25):
            period = (year + (month - 1) // 12) * 100 + (month - 1) % 12 + 1
            perf = [""] * 35
            for index, value in {
                0: loan,
                1: str(period),
                2: str(200000 - month * 100),
                3: "00",
                4: str(month),
                10: "4",
            }.items():
                perf[index] = value
            if month == 24:
                perf[8], perf[9] = "01", str(period)
            performance_rows.append("|".join(perf))
    (raw / f"sample_orig_{year}.txt").write_text("\n".join(orig_rows) + "\n")
    (raw / f"sample_perf_{year}.txt").write_text("\n".join(performance_rows) + "\n")


def test_prepare_splits_dates_and_train_only_preprocessing(tmp_path):
    raw, prepared = tmp_path / "raw", tmp_path / "processed"
    for year in [2018, 2019]:
        make_raw(raw, year)
        summary = prepare_year(raw, prepared, year, 522, 12, "synthetic", "test")
        assert summary["windows"] == 36
        assert summary["window_class_counts"][4] == 3
    single = LoanData(prepared, [2018])
    data = LoanData(prepared, [2018, 2019])
    np.testing.assert_array_equal(single.loans.split, data.loans.iloc[:3].split)
    single_test = single.partitions()[2]
    expanded_test = data.partitions(test_cohorts=[2018])[2]
    assert single.population_signature(single_test) == data.population_signature(
        expanded_test
    )
    train, validation, test = data.partitions()
    loan_sets = [set(data.target_loans[part]) for part in [train, validation, test]]
    assert not (
        loan_sets[0] & loan_sets[1]
        or loan_sets[0] & loan_sets[2]
        or loan_sets[1] & loan_sets[2]
    )
    params = data.fit_preprocessing(train)
    data.dynamic_raw[data.loans.split.to_numpy()[data.row_loan] != 0] = 1e8
    assert data.fit_preprocessing(train) == params
    held_out_loans = data.loans.split.to_numpy() != 0
    data.static_raw[held_out_loans] = 1e8
    data.loans.loc[held_out_loans, "purpose"] = "UNSEEN"
    assert data.fit_preprocessing(train) == params
    # The first category slot is reserved for values absent from training.
    np.testing.assert_array_equal(data.static[held_out_loans, 12], 1)
    # A target-only row must not influence its own input preprocessing.
    last_train = train[data.target_months[train] == data.target_months[train].max()]
    first_params = data.fit_preprocessing(last_train)
    data.dynamic_raw[data.targets[last_train]] = 1e9
    assert data.fit_preprocessing(last_train) == first_params
    train_time, val_time, test_time = data.partitions("2019-01", "2019-06")
    assert data.target_months[train_time].max() < data.target_months[val_time].min()
    assert data.target_months[val_time].max() < data.target_months[test_time].min()
    assert set(data.cohorts[data.partitions(test_cohorts=[2018])[2]]) == {2018}


def test_prepare_stops_at_first_reo_and_checks_duplicates(tmp_path):
    raw = tmp_path / "raw"
    make_raw(raw, 2018)
    perf_path = raw / "sample_perf_2018.txt"
    lines = perf_path.read_text().splitlines()
    row = lines[13].split("|")
    row[3] = "RA"
    lines[13] = "|".join(row)
    perf_path.write_text("\n".join(lines) + "\n")
    summary = prepare_year(
        raw, tmp_path / "prepared", 2018, 522, 12, "synthetic", "test"
    )
    assert summary["dropped_after_first_exit"] == 10
    assert summary["window_class_counts"][5] == 1
    perf_path.write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(ValueError, match="duplicate loan-month"):
        prepare_year(raw, tmp_path / "prepared", 2018, 522, 12, "synthetic", "test")


def test_metrics_keep_rare_classes_and_reject_bad_probabilities():
    actual = np.array([0, 4], dtype=np.int64)
    probabilities = np.eye(len(LABELS), dtype=np.float32)[actual]
    report = metrics(actual, probabilities)
    assert report["macro_f1"] == pytest.approx(2 / 7)
    assert report["supported_macro_f1"] == 1
    with pytest.raises(ValueError):
        metrics(actual, probabilities * 2)


def test_transformer_learns_tiny_problem():
    torch.set_num_threads(1)
    torch.manual_seed(522)
    model = MortgageTransformer(
        4, 3, history=12, hidden=16, heads=4, layers=1, dropout=0
    )
    dynamic = torch.randn(16, 12, 4)
    static = torch.randn(16, 3)
    labels = (static[:, 0] > 0).long()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    criterion = torch.nn.CrossEntropyLoss()
    first_loss = float(criterion(model(dynamic, static), labels).detach())
    for _ in range(50):
        optimizer.zero_grad()
        loss = criterion(model(dynamic, static), labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    assert float(loss.detach()) < first_loss * 0.1
