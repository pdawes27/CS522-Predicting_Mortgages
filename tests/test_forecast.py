"""Recursive generation and censoring checks."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from mortgage.data import LoanData
from mortgage.forecast import generate_paths, observed_horizons
from mortgage.models import transition_probabilities
from mortgage.prepare import prepare_year
from mortgage.schema import LABELS
from test_pipeline import make_raw


class Progression(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, dynamic, static):
        self.calls += 1
        next_state = dynamic[:, -1].argmax(-1) + 1
        return nn.functional.one_hot(next_state, len(LABELS)).float() * 100


class UniformStates(nn.Module):
    def forward(self, dynamic, static):
        return torch.zeros(len(dynamic), len(LABELS))


def test_transition_baseline_preserves_terminal_states():
    matrix = transition_probabilities(
        np.array([0, 0, 1]), np.array([0, 4, 2]), absorbing=True
    )
    np.testing.assert_array_equal(matrix[4:], np.eye(len(LABELS))[4:])
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-6)


def test_generation_feeds_back_tokens_and_stops_at_exit():
    model = Progression()
    histories = np.zeros((1, 12), dtype=np.int64)
    static = np.zeros((1, 2), dtype=np.float32)
    paths, probabilities = generate_paths(
        model, histories, static, horizon=8, paths=1, greedy=True
    )
    np.testing.assert_array_equal(paths[0, 0], [1, 2, 3, 4, -1, -1, -1, -1])
    assert model.calls == 4
    np.testing.assert_array_equal(probabilities[0, 3:, 4], 1)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1)


def test_sampling_reproducible_and_terminal_marginals_absorbing():
    histories = np.zeros((2, 12), dtype=np.int64)
    static = np.zeros((2, 2), dtype=np.float32)
    first, probabilities = generate_paths(
        UniformStates(), histories, static, horizon=8, paths=100, seed=7
    )
    second, _ = generate_paths(
        UniformStates(), histories, static, horizon=8, paths=100, seed=7
    )
    different, _ = generate_paths(
        UniformStates(), histories, static, horizon=8, paths=100, seed=8
    )
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1, atol=1e-6)
    assert (np.diff(probabilities[:, :, 4:].sum(axis=-1), axis=1) >= -1e-6).all()
    for path in first.reshape(-1, 8):
        exits = np.flatnonzero(path >= 4)
        if len(exits):
            assert (path[exits[0] + 1 :] == -1).all()


def test_generation_rejects_unknown_and_closed_histories():
    static = np.zeros((1, 2), dtype=np.float32)
    for state in [-1, 4, 5, 6]:
        with pytest.raises(ValueError, match="active states"):
            generate_paths(UniformStates(), np.full((1, 12), state), static)
    with pytest.raises(ValueError):
        generate_paths(
            UniformStates(), np.zeros((1, 12), dtype=np.int64), static, horizon=0
        )


def test_multihorizon_labels_keep_censoring_distinct_from_exit(tmp_path: Path):
    raw = tmp_path / "raw"
    make_raw(raw, 2018)
    perf_path = raw / "sample_perf_2018.txt"
    lines = perf_path.read_text().splitlines()
    # Leave the first loan active at the disclosure cutoff; other loans pay off.
    row = lines[23].split("|")
    row[8] = row[9] = ""
    lines[23] = "|".join(row)
    perf_path.write_text("\n".join(lines) + "\n")
    prepare_year(raw, tmp_path / "processed", 2018, 522, 12, "synthetic", "test")
    data = LoanData(tmp_path / "processed", [2018])
    last_origins = np.flatnonzero(data.target_months == data.target_months.max())
    actual = observed_horizons(data, last_origins, 4)
    for index, values in zip(last_origins, actual):
        if data.labels[index] == 4:
            np.testing.assert_array_equal(values, [4, 4, 4, 4])
        else:
            np.testing.assert_array_equal(values, [0, -1, -1, -1])
