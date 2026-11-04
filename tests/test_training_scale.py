"""Nested data, fixed training budgets, and validation-only checkpoints."""

from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.metrics import f1_score

from mortgage.data import LoanData, nested_sample_indices, sample_indices
from mortgage.evaluate import macro_f1
from mortgage.models import (
    predict_transformer,
    train_transformer,
    transition_probabilities,
)
from mortgage.prepare import prepare_year
from test_pipeline import make_raw


def test_nested_samples_preserve_anchor_and_all_larger_subsets():
    population = np.arange(200, dtype=np.int64)
    small = nested_sample_indices(population, 20, 522, 20)
    medium = nested_sample_indices(population, 70, 522, 20)
    large = nested_sample_indices(population, 130, 522, 20)
    np.testing.assert_array_equal(small, sample_indices(population, 20, 522))
    assert np.isin(small, medium).all() and np.isin(medium, large).all()
    assert len(np.unique(large)) == 130
    np.testing.assert_array_equal(
        nested_sample_indices(population, 0, 522, 20), population
    )
    np.testing.assert_array_equal(
        nested_sample_indices(population, 70, 522, 20), medium
    )
    with pytest.raises(ValueError):
        nested_sample_indices(population, 10, 522, 20)


def test_fast_macro_f1_matches_sklearn_with_absent_classes():
    labels = np.array([0, 0, 1, 2, 4], dtype=np.int64)
    predicted = np.array([0, 1, 1, 0, 4])
    probabilities = np.eye(7, dtype=np.float32)[predicted]
    assert macro_f1(labels, probabilities) == pytest.approx(
        f1_score(
            labels, predicted, labels=list(range(7)), average="macro", zero_division=0
        )
    )


@pytest.mark.parametrize("variant", ["ordinary", "residual", "embedding"])
def test_fixed_updates_save_both_validation_selected_checkpoints(
    tmp_path: Path, variant: str
):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    make_raw(raw, 2018)
    prepare_year(raw, prepared, 2018, 522, 12, "synthetic", "test")
    data = LoanData(prepared, [2018])
    train, validation, _ = data.partitions()
    data.fit_preprocessing(train)
    observed = data.history_states(train)
    np.testing.assert_array_equal(observed[:, -1], data.current[train])
    flat = data.flat(train, state_indicators=True)
    np.testing.assert_array_equal(flat[:, -48:].reshape(-1, 12, 4).argmax(-1), observed)
    output = tmp_path / "fit"
    output.mkdir()
    torch.set_num_threads(1)
    model, curve = train_transformer(
        data,
        train,
        validation,
        output,
        522,
        epochs=1,
        max_steps=7,
        validate_every=3,
        dual_checkpoints=True,
        batch_size=4,
        hidden=16,
        heads=4,
        layers=1,
        transition_residual=variant == "residual",
        state_embedding=variant == "embedding",
        lr_schedule="constant" if variant == "ordinary" else "warmup_cosine",
        warmup_steps=2,
    )
    assert [point["step"] for point in curve] == [3, 6, 7]
    assert curve[-1]["examples_seen"] == 28
    assert curve[-1]["learning_rate"] == pytest.approx(
        0.001 if variant == "ordinary" else 0.00005
    )
    assert curve[-1]["epoch"] > 1  # Fixed updates override the epoch cap.
    for filename, selection in [
        ("transformer.pt", "validation_log_loss"),
        ("transformer_macro_f1.pt", "validation_macro_f1"),
    ]:
        saved = torch.load(output / filename, weights_only=True)
        if variant == "residual":
            prior = transition_probabilities(data.current[train], data.labels[train])
            np.testing.assert_array_equal(
                saved["state_dict"]["transition_log_probs"].numpy(), np.log(prior)
            )
        assert saved["selection_metric"] == selection
        assert saved["step"] in [3, 6, 7]
        model.load_state_dict(saved["state_dict"], strict=True)
        probability = predict_transformer(
            model, data, validation, 4, torch.device("cpu"), 12
        )
        actual_loss = -np.log(
            probability[np.arange(len(validation)), data.labels[validation]]
        ).mean()
        assert actual_loss == pytest.approx(saved["validation_log_loss"])
        assert macro_f1(data.labels[validation], probability) == pytest.approx(
            saved["validation_macro_f1"]
        )
    loss_checkpoint = torch.load(output / "transformer.pt", weights_only=True)
    f1_checkpoint = torch.load(output / "transformer_macro_f1.pt", weights_only=True)
    assert (
        loss_checkpoint["validation_log_loss"]
        <= min(point["validation_log_loss"] for point in curve) + 1e-5
    )
    assert f1_checkpoint["validation_macro_f1"] == max(
        point["validation_macro_f1"] for point in curve
    )
