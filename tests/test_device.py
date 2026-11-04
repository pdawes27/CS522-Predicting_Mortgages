"""Fail fast when a requested GPU is unavailable."""

from argparse import Namespace

import pytest
import torch

from mortgage.experiment import run


def test_cuda_unavailable_fails_before_loading_data(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    args = Namespace(
        history=12,
        epochs=1,
        batch_size=512,
        patience=3,
        hidden=64,
        heads=4,
        device="cuda",
        max_steps=0,
        validate_every=400,
    )
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        run(args)
