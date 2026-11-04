"""Training recovery preserves updates, random streams, and existing checkpoints."""

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from torch import nn

from neuro_co.core.checkpoint import (
    atomic_save,
    capture_rng_state,
    restore_rng_state,
    training_state,
    validate_rng_state,
    validate_training_state,
)
from neuro_co.core.precision import Precision
from neuro_co.core.train import Trainer
from neuro_co.core.train_utils import WeightEMA


@dataclass
class _Config:
    lr: float = 0.01
    ema_decay: float = 0.8


class _Algo(nn.Module):
    def __init__(self, *, lr=0.01):
        super().__init__()
        self.cfg = _Config(lr=lr)
        self.model = nn.Sequential(nn.Linear(3, 5), nn.Dropout(0.3), nn.Linear(5, 1))
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, 2, gamma=0.7)
        self.precision = Precision()
        self.ema = WeightEMA(self.model, decay=self.cfg.ema_decay)
        self._step = 0

    def train_step(self, rng):
        self.model.train()
        x = torch.rand(4, 3, generator=rng)
        y = random.random() + np.random.random() + torch.rand(())
        loss = (self.model(x) - y).square().mean()
        self.opt.zero_grad(set_to_none=True)
        self.precision.backward(loss)
        self.precision.step(self.opt)
        self.precision.update()
        self.sched.step()
        self.ema.update()
        self._step += 1
        return {"loss": loss.item()}

    def eval_step(self, rng):
        return {"eval_reward": float(self._step)}

    @contextmanager
    def eval_weights(self):
        self.ema.apply()
        try:
            yield
        finally:
            self.ema.restore()


def _seed(value=7):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            _equal(x, y)
    else:
        assert a == b


def test_resume_matches_uninterrupted_training_and_rng(tmp_path):
    _seed()
    full = Trainer(_Algo(), steps=6, seed=19)
    full.fit()
    expected_random = (random.random(), np.random.random(), torch.rand(3))

    _seed()
    partial = Trainer(_Algo(), steps=3, seed=19, ckpt_dir=tmp_path)
    partial.fit()
    _seed(999)
    resumed = Trainer(_Algo(), steps=6, seed=999, resume_from=tmp_path / "latest.pt")
    resumed.fit()

    assert resumed.state.step == resumed.algo._step == 6
    _equal(full.algo.state_dict(), resumed.algo.state_dict())
    _equal(training_state(full.algo), training_state(resumed.algo))
    _equal(full.rng.get_state(), resumed.rng.get_state())
    _equal(expected_random, (random.random(), np.random.random(), torch.rand(3)))


def test_completed_target_does_not_train_or_rewrite_checkpoint(tmp_path):
    trainer = Trainer(_Algo(), steps=2, ckpt_dir=tmp_path)
    trainer.fit()
    path = tmp_path / "latest.pt"
    before = path.read_bytes()
    resumed = Trainer(_Algo(), steps=2, ckpt_dir=tmp_path, resume_from=path)
    resumed.fit()
    resumed.fit()
    assert resumed.state.step == resumed.algo._step == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_failed_atomic_write_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    path = tmp_path / "latest.pt"
    atomic_save({"step": 1}, path)
    before = path.read_bytes()

    def fail(payload, handle):
        handle.write(b"partial checkpoint")
        raise OSError("storage interrupted")

    if failure == "serialize":
        monkeypatch.setattr(torch, "save", fail)
    else:

        def fail_replace(source, target):
            raise OSError("storage interrupted")

        monkeypatch.setattr("neuro_co.core.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="storage interrupted"):
        atomic_save({"step": 2}, path)
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_corrupt_checkpoint_does_not_mutate_model(tmp_path):
    trainer = Trainer(_Algo(), steps=0)
    before = copy.deepcopy(trainer.algo.state_dict())
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"not a torch checkpoint")
    with pytest.raises(Exception):
        trainer.load(path)
    _equal(before, trainer.algo.state_dict())


@pytest.mark.parametrize(
    "change", ["config", "optimizer", "scheduler", "precision", "ema", "device", "model"]
)
def test_incompatible_checkpoint_is_rejected_before_model_load(tmp_path, change):
    trainer = Trainer(_Algo(), steps=1, ckpt_dir=tmp_path)
    trainer.fit()
    payload = torch.load(tmp_path / "latest.pt", weights_only=True)
    state = payload["training_state"]
    if change == "config":
        state["config"]["lr"] = 9.0
    elif change == "optimizer":
        state["components"]["opt"]["type"] = "torch.optim.SGD"
    elif change == "scheduler":
        del state["components"]["sched"]
    elif change == "precision":
        state["components"]["precision"]["state"]["precision"] = "bf16"
    elif change == "ema":
        state["components"]["ema"]["state"]["decay"] = 0.1
    elif change == "device":
        payload["global_rng_state"]["device"] = "cuda:1"
    else:
        payload["algo_state"]["model.0.weight"] = torch.zeros(9, 3)
    atomic_save(payload, tmp_path / "bad.pt")
    fresh = Trainer(_Algo(), steps=2)
    before = copy.deepcopy(fresh.algo.state_dict())
    with pytest.raises(ValueError):
        fresh.load(tmp_path / "bad.pt")
    _equal(before, fresh.algo.state_dict())
    assert fresh.state.step == 0


def test_legacy_checkpoint_restores_saved_weights_and_sampling_generator(tmp_path):
    original = _Algo()
    generator = torch.Generator().manual_seed(123)
    atomic_save(
        {"step": 4, "algo_state": original.state_dict(), "rng_state": generator.get_state()},
        tmp_path / "old.pt",
    )
    restored = Trainer(_Algo(), steps=4, resume_from=tmp_path / "old.pt")
    restored.fit()
    assert restored.state.step == restored.algo._step == 4
    _equal(restored.algo.state_dict(), original.state_dict())
    _equal(restored.rng.get_state(), generator.get_state())


def test_ema_best_checkpoint_is_for_inference_only(tmp_path):
    trainer = Trainer(_Algo(), steps=2, eval_every=1, ckpt_dir=tmp_path)
    trainer.fit()
    best = torch.load(tmp_path / "best.pt", weights_only=True)
    assert best["resume_training"] is False
    for name, shadow in trainer.algo.ema.shadow.items():
        assert torch.equal(best["algo_state"]["model." + name], shadow)
    with pytest.raises(ValueError, match="EMA evaluation checkpoint"):
        Trainer(_Algo(), steps=3, resume_from=tmp_path / "best.pt")
    Trainer(_Algo(), steps=3, resume_from=tmp_path / "latest.pt")


def test_fp16_gradient_scaler_round_trip():
    precision = Precision("fp16", device="cpu")
    parameter = nn.Parameter(torch.tensor(2.0))
    opt = torch.optim.SGD([parameter], lr=0.1)
    precision.backward(parameter.square())
    precision.step(opt)
    precision.update()
    saved = precision.state_dict()
    assert saved["scaler"]["_growth_tracker"] == 1
    restored = Precision("fp16", device="cpu")
    restored.load_state_dict(saved)
    _equal(restored.state_dict(), saved)
    with pytest.raises(ValueError, match="precision"):
        Precision("bf16", device="cpu").load_state_dict(saved)


def test_selected_cuda_rng_never_touches_other_devices(monkeypatch):
    state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    reads, writes = [], []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "get_rng_state", lambda device: (reads.append(str(device)), state.clone())[1]
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, device: writes.append((str(device), value.clone())),
    )
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("all-device read"))
    monkeypatch.setattr(
        torch.cuda, "set_rng_state_all", lambda value: pytest.fail("all-device write")
    )
    saved = capture_rng_state("cuda")
    assert saved["device"] == "cuda:1"
    assert saved["accelerator"].device.type == "cpu"
    restore_rng_state(saved, "cuda:1")
    assert reads == ["cuda:1"]
    assert writes[0][0] == "cuda:1"
    assert len(writes) == 1
    before = random.getstate()
    with pytest.raises(ValueError, match="RNG device"):
        restore_rng_state(saved, "cuda:0")
    assert random.getstate() == before
    assert len(writes) == 1


def test_mps_rng_round_trip_with_explicit_scope(monkeypatch):
    state = torch.tensor([2, 4, 6], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: state.clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda value: restored.append(value))
    saved = capture_rng_state("mps")
    restore_rng_state(saved, "mps")
    assert torch.equal(restored[0], state)


def test_rng_validation_does_not_consume_global_streams():
    saved = capture_rng_state("cpu")
    validate_rng_state(saved, "cpu")
    _equal(saved, capture_rng_state("cpu"))


def test_optimizer_group_shape_is_checked():
    algo = _Algo()
    saved = training_state(algo)
    saved["components"]["opt"]["state"]["param_groups"][0]["params"].pop()
    with pytest.raises(ValueError, match="parameter groups"):
        validate_training_state(algo, saved)
