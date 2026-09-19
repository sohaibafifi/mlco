"""Smoke tests for the Lightning per-epoch energy callback."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_aet_callback_lazy_import_via_aet_namespace(tmp_path: Path) -> None:
    """`from neuro_co.aet import AETCallback` triggers the lazy lightning import."""
    pytest.importorskip("lightning")
    from neuro_co.aet import AETCallback

    cb = AETCallback(output_dir=tmp_path, backend="tdp", hardware_id="generic-cpu")
    assert cb.output_dir == tmp_path
    assert cb.backend == "tdp"


def test_aet_callback_writes_per_epoch_json(tmp_path: Path) -> None:
    """End-to-end: callback opens + closes tracker, writes JSON per epoch."""
    pytest.importorskip("lightning")
    from neuro_co.aet import AETCallback

    cb = AETCallback(output_dir=tmp_path, backend="tdp", hardware_id="generic-cpu")

    class _DummyTrainer:
        current_epoch = 0

    cb.on_train_epoch_start(_DummyTrainer(), None)
    cb.on_train_epoch_end(_DummyTrainer(), None)

    _DummyTrainer.current_epoch = 1
    cb.on_train_epoch_start(_DummyTrainer(), None)
    cb.on_train_epoch_end(_DummyTrainer(), None)

    cb.on_train_end(_DummyTrainer(), None)

    epoch0 = tmp_path / "energy_train_epoch_0.json"
    epoch1 = tmp_path / "energy_train_epoch_1.json"
    rollup = tmp_path / "energy_train_epochs.json"
    assert epoch0.is_file()
    assert epoch1.is_file()
    assert rollup.is_file()

    r0 = json.loads(epoch0.read_text())
    assert r0["schema_version"] == "1.0"
    assert r0["extra"]["label"] == "train_epoch_0"
    assert r0["extra"]["epoch"] == 0
    assert "energy_j" in r0
    assert "co2_total_kg" in r0
    assert "energy_wh" not in r0

    roll = json.loads(rollup.read_text())
    assert roll["n_epochs"] == 2
    assert roll["schema_version"] == "aet-training-rollup/v1"
    assert roll["total_energy_j"] == r0["energy_j"] + json.loads(epoch1.read_text())["energy_j"]


def test_aet_callback_skips_writes_with_no_epochs(tmp_path: Path) -> None:
    """on_train_end is a no-op when no epoch ever fired."""
    pytest.importorskip("lightning")
    from neuro_co.aet import AETCallback

    cb = AETCallback(output_dir=tmp_path, backend="tdp")
    cb.on_train_end(None, None)
    assert not (tmp_path / "energy_train_epochs.json").exists()
