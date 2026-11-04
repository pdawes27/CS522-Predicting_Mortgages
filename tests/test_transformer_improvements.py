"""Learning schedules, transition priors, and categorical states."""

from pathlib import Path

import numpy as np
import pytest
import torch

from mortgage.models import (
    MortgageTransformer,
    learning_rate_multiplier,
    transition_probabilities,
)


def test_warmup_cosine_endpoints_and_decay():
    assert learning_rate_multiplier(1, 100, 10) == pytest.approx(0.1)
    assert learning_rate_multiplier(10, 100, 10) == 1
    assert learning_rate_multiplier(100, 100, 10) == pytest.approx(0.05)
    values = [learning_rate_multiplier(step, 100, 10) for step in range(10, 101)]
    assert np.all(np.diff(values) <= 0)
    with pytest.raises(ValueError):
        learning_rate_multiplier(1, 10, 10)


def test_residual_starts_at_training_prior_and_roundtrips(tmp_path: Path):
    architecture = dict(
        dynamic_size=8,
        static_size=3,
        hidden=16,
        heads=4,
        layers=1,
        dropout=0.0,
        transition_residual=True,
    )
    model = MortgageTransformer(**architecture).eval()
    prior = transition_probabilities(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 2]))
    model.transition_log_probs.copy_(torch.from_numpy(np.log(prior)))
    dynamic, static = torch.randn(2, 12, 8), torch.randn(2, 3)
    states = torch.tensor([[0] * 12, [1] * 12])
    expected = torch.from_numpy(prior[[0, 1]])
    torch.testing.assert_close(model(dynamic, static, states).softmax(-1), expected)
    with pytest.raises(ValueError, match="requires observed"):
        model(dynamic, static)
    loss = torch.nn.functional.cross_entropy(
        model(dynamic, static, states), torch.tensor([1, 2])
    )
    loss.backward()
    assert model.classifier[-1].weight.grad.abs().sum() > 0
    assert model.transition_log_probs.grad is None
    path = tmp_path / "model.pt"
    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = MortgageTransformer(**saved["architecture"]).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(restored(dynamic, static, states).softmax(-1), expected)


def test_state_embeddings_preserve_shared_weights_and_receive_gradients():
    options = dict(
        dynamic_size=8, static_size=3, hidden=16, heads=4, layers=1, dropout=0.0
    )
    torch.manual_seed(522)
    ordinary = MortgageTransformer(**options)
    torch.manual_seed(522)
    embedded = MortgageTransformer(**options, state_embedding=True)
    for key, value in ordinary.state_dict().items():
        torch.testing.assert_close(value, embedded.state_dict()[key], rtol=0, atol=0)
    dynamic, static = torch.randn(2, 12, 8), torch.randn(2, 3)
    states = torch.tensor([[0] * 6 + [1] * 6, [2] * 6 + [3] * 6])
    embedded(dynamic, static, states).square().sum().backward()
    assert embedded.state_embedding.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="requires observed"):
        embedded(dynamic, static)
