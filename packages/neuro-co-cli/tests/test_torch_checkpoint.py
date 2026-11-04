"""The Torch CLI adapter shares the core's checkpoint recovery contract."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from neuro_co.cli.training import TrainingConfig, batch, initial_weights, make_trainer


def _config():
    return TrainingConfig(
        problem="tsp",
        size=5,
        batch_size=3,
        n_starts=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        eval_batch_size=3,
    )


def _draw():
    return random.random(), float(np.random.rand()), torch.rand(3)


def test_torch_adapter_resumes_optimizer_sampler_and_rng(tmp_path):
    config = _config()
    weights = initial_weights(config)
    original = make_trainer(config, "torch", "cpu", weights)
    original.train_step(batch(config, "train", 0))
    path = tmp_path / "nested" / "latest.pt"
    original.save(path, 1)
    expected_rng = _draw()
    expected = original.train_step(batch(config, "train", 1))

    resumed = make_trainer(config, "torch", "cpu", weights)
    assert resumed.load(path) == 1
    actual_rng = _draw()
    assert actual_rng[:2] == expected_rng[:2]
    torch.testing.assert_close(actual_rng[2], expected_rng[2], rtol=0, atol=0)
    actual = resumed.train_step(batch(config, "train", 1))
    assert actual == expected
    assert resumed.algo._step == original.algo._step == 2
    for name, tensor in original.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], tensor, rtol=0, atol=0)
    assert (
        resumed.algo.opt.state_dict()["param_groups"]
        == original.algo.opt.state_dict()["param_groups"]
    )


def test_inference_does_not_restore_training_or_global_rng(tmp_path):
    config = _config()
    trainer = make_trainer(config, "torch", "cpu", initial_weights(config))
    path = tmp_path / "best.pt"
    trainer.save(path, 4)
    checkpoint = torch.load(path, weights_only=True)
    # Inference may use a different device and needs neither training component.
    checkpoint["global_rng_state"] = {"device": "cuda:99"}
    checkpoint["training_state"] = {"not": "needed for inference"}
    torch.save(checkpoint, path)
    cpu_rng = torch.get_rng_state().clone()
    assert trainer.load(path, resume=False) == 4
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert np.isfinite(trainer.evaluate(batch(config, "test"))).all()


def test_legacy_torch_adapter_checkpoint_still_loads(tmp_path):
    config = _config()
    weights = initial_weights(config)
    original = make_trainer(config, "torch", "cpu", weights)
    original.train_step(batch(config, "train", 0))
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": original.model.state_dict(),
            "optimizer": original.algo.opt.state_dict(),
            "sampler_state": original.rng.get_state(),
            "step": 1,
        },
        path,
    )
    resumed = make_trainer(config, "torch", "cpu", weights)
    assert resumed.load(path) == 1
    assert resumed.train_step(batch(config, "train", 1)) == original.train_step(
        batch(config, "train", 1)
    )


def test_adapter_checks_resume_compatibility_before_loading_weights(tmp_path):
    config = _config()
    weights = initial_weights(config)
    original = make_trainer(config, "torch", "cpu", weights)
    original.train_step(batch(config, "train", 0))
    path = tmp_path / "latest.pt"
    original.save(path, 1)
    config = replace(config, lr=config.lr * 2)
    resumed = make_trainer(config, "torch", "cpu", weights)
    before = {name: value.clone() for name, value in resumed.model.state_dict().items()}
    with pytest.raises(ValueError):
        resumed.load(path)
    for name, value in resumed.model.state_dict().items():
        assert torch.equal(value, before[name])
