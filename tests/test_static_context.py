"""Static-context conditioning and checkpoint compatibility."""

from pathlib import Path

import pytest
import torch

from mortgage.models import MortgageTransformer, TransformerArchitecture


def test_static_context_changes_encoder_inputs_and_receives_gradients():
    torch.manual_seed(522)
    model = MortgageTransformer(
        4, 3, hidden=16, heads=4, layers=1, dropout=0, static_context=True
    )
    dynamic = torch.randn(2, 12, 4)
    static = torch.randn(2, 3, requires_grad=True)
    captured = []
    handle = model.encoder.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach().clone())
    )
    logits = model(dynamic, static)
    handle.remove()
    assert logits.shape == (2, 7)
    assert captured[0].shape == (2, 13, 16)
    torch.testing.assert_close(
        captured[0][:, 1:], model.projection(dynamic) + model.position
    )
    assert model.static_projection is not None
    torch.testing.assert_close(captured[0][:, 0], model.static_projection(static))
    # Exclude the direct static-feature classifier path: a gradient must still
    # travel through the context token and attention to the static projection.
    with torch.no_grad():
        model.classifier[0].weight[:, 16:] = 0
    model(dynamic, static).square().sum().backward()
    assert model.static_projection.weight.grad is not None
    assert model.static_projection.weight.grad.abs().sum() > 0
    assert static.grad is not None and static.grad.abs().sum() > 0


@pytest.mark.parametrize("conditioned", [False, True])
def test_context_flag_checkpoint_roundtrip_and_legacy_default(
    tmp_path: Path, conditioned: bool
):
    architecture: TransformerArchitecture = {
        "dynamic_size": 4,
        "static_size": 3,
        "hidden": 16,
        "heads": 4,
        "layers": 1,
        "dropout": 0.0,
    }
    if conditioned:
        architecture["static_context"] = True
    torch.manual_seed(522)
    model = MortgageTransformer(**architecture).eval()
    dynamic, static = torch.randn(2, 12, 4), torch.randn(2, 3)
    expected = model(dynamic, static)
    path = tmp_path / "checkpoint.pt"
    torch.save({"architecture": architecture, "state_dict": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = MortgageTransformer(**saved["architecture"]).eval()
    restored.load_state_dict(saved["state_dict"], strict=True)
    torch.testing.assert_close(restored(dynamic, static), expected)
    assert (restored.static_projection is not None) == conditioned
    if not conditioned:
        assert not any(
            key.startswith("static_projection") for key in restored.state_dict()
        )


def test_conditioning_preserves_shared_initialization():
    options = dict(dynamic_size=4, static_size=3, hidden=16, heads=4, layers=1)
    torch.manual_seed(522)
    baseline = MortgageTransformer(**options)
    torch.manual_seed(522)
    conditioned = MortgageTransformer(**options, static_context=True)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, conditioned.state_dict()[key], rtol=0, atol=0)
