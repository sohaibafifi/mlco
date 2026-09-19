"""Smoke tests for the aet package."""

from __future__ import annotations

import time

import pytest

from neuro_co.aet import EnergyReading, EnergyTracker, amortize_embodied
from neuro_co.aet.analysis import (
    TrainAggregate,
    aggregate_training,
    asymptotic_ratio,
    build_aet_table,
    crossover_n,
)
from neuro_co.aet.energy.rapl import tdp_estimate_wh


def test_embodied_known_hw_amortizes_linearly() -> None:
    full = amortize_embodied("nvidia-a100", t_used_s=5 * 365 * 24 * 3600)
    half = amortize_embodied("nvidia-a100", t_used_s=int(0.5 * 5 * 365 * 24 * 3600))
    assert full == pytest.approx(150.0)
    assert half == pytest.approx(75.0, rel=0.01)


def test_embodied_unknown_falls_back_to_generic() -> None:
    val_gpu = amortize_embodied("some-unknown-gpu", t_used_s=3600)
    val_cpu = amortize_embodied("some-unknown-cpu", t_used_s=3600)
    assert val_gpu > 0 and val_cpu > 0
    assert val_gpu != val_cpu


def test_embodied_zero_lifetime() -> None:
    assert amortize_embodied("nvidia-a100", t_used_s=10, t_lifetime_s=0) == 0.0


def test_tdp_estimate() -> None:
    wh = tdp_estimate_wh("nvidia-a100", 3600.0)
    assert wh == pytest.approx(300.0)
    assert tdp_estimate_wh(None, 0.0) == 0.0
    assert tdp_estimate_wh("totally-unknown", 3600.0) > 0


def test_tracker_with_tdp_fallback_produces_reading() -> None:
    with EnergyTracker(
        "smoke",
        backend="tdp",
        hardware_id="generic-cpu",
        report_embodied=False,
        items=10,
    ) as t:
        time.sleep(0.05)
    r = t.reading
    assert r is not None
    assert r.duration_s >= 0.04
    assert r.energy_j >= 0.0
    assert r.items_processed == 10
    assert r.backend == "tdp"
    d = r.to_dict()
    assert d["throughput_items_per_s"] >= 0
    assert "backend" in d


def test_tracker_backend_chain_resolution() -> None:
    assert EnergyTracker._resolve_chain("tdp")[0] == "tdp"
    assert EnergyTracker._resolve_chain("hwcounters")[0] == "hwcounters"
    assert EnergyTracker._resolve_chain("unknown") == ["codecarbon", "hwcounters", "tdp"]


def test_aggregate_training_handles_empty() -> None:
    agg = aggregate_training([])
    assert agg.n_seeds == 0
    assert agg.hardware_id is None


def test_aggregate_training_simple() -> None:
    records = [
        {"energy_wh": 100.0, "co2_g_total": 50.0, "hardware_id": "nvidia-a100"},
        {"energy_wh": 200.0, "co2_g_total": 80.0, "hardware_id": "nvidia-a100"},
        {"energy_wh": 150.0, "co2_g_total": 65.0, "hardware_id": "nvidia-a100"},
    ]
    agg = aggregate_training(records)
    assert agg.n_seeds == 3
    assert agg.hardware_id == "nvidia-a100"
    assert agg.energy_wh_median == pytest.approx(150.0)
    assert agg.co2_g_median == pytest.approx(65.0)


def test_crossover_and_ratio() -> None:
    assert crossover_n(1000.0, 0.5, 1.5) == pytest.approx(1000.0)
    assert crossover_n(1000.0, 1.5, 1.0) is None
    assert asymptotic_ratio(0.5, 1.5) == pytest.approx(1.0 / 3.0)
    assert asymptotic_ratio(0.5, 0.0) is None


def test_build_aet_table_minimal() -> None:
    train_agg = TrainAggregate(
        energy_wh_median=1000.0,
        energy_wh_p25=900.0,
        energy_wh_p75=1100.0,
        co2_g_median=500.0,
        co2_g_p25=450.0,
        co2_g_p75=550.0,
        n_seeds=3,
        hardware_id="nvidia-a100",
    )
    inf = [
        {
            "variant": "AM",
            "batch_size": 1024,
            "n_instances": 10000,
            "energy_wh_per_item": 0.001,
            "co2_g_total": 5.0,
            "gap_to_bks": 1.0,
            "size_key": 50,
            "seed": 1,
        }
    ]
    base = [
        {
            "file": "test_50.npz",
            "solver": "PyVRP",
            "num_problems": 10,
            "energy_wh": 0.5,
            "co2_g_total": 0.25,
            "size": 50,
            "thread_mode": "mono",
            "baseline_gap": 0.0,
        }
    ]
    rows = build_aet_table(train_agg, inf, base, deltas=[0.5, 1.0, 2.0])
    assert len(rows) == 3
    feasible_rows = [r for r in rows if r["feasible"]]
    assert len(feasible_rows) >= 1
    for r in feasible_rows:
        assert r["aet_E_status"] == "unidentified"
        assert r["aet_C_status"] == "unidentified"
        assert r["aet_E_point_estimate"] > 0
        assert r["aet_C_point_estimate"] > 0


def test_energy_reading_aggregates() -> None:
    r = EnergyReading(
        duration_s=10.0,
        energy_j=100.0,
        co2_operational_kg=0.01,
        co2_embodied_kg=0.005,
        items_processed=20,
    )
    assert r.co2_total_kg == pytest.approx(0.015)
    assert r.throughput == pytest.approx(2.0)
    d = r.to_dict()
    assert d["co2_total_kg"] == pytest.approx(0.015)
    assert d["throughput_items_per_s"] == pytest.approx(2.0)
