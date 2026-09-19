"""Regression tests for AET input semantics and classifications."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from neuro_co.aet import EnergyReading
from neuro_co.aet.analysis import TrainAggregate, aggregate_training, build_aet_table, load_records

HOST_ID = "qualified-host"
CALIBRATION_SHA256 = "a" * 64
PROVENANCE_SHA256 = "b" * 64
ARTIFACT_SHA256 = "c" * 64
INSTANCE_MANIFEST_SHA256 = "d" * 64


def _train_aggregate() -> TrainAggregate:
    return TrainAggregate(
        energy_wh_median=1000.0,
        energy_wh_p25=900.0,
        energy_wh_p75=1100.0,
        co2_g_median=500.0,
        co2_g_p25=450.0,
        co2_g_p75=550.0,
        n_seeds=3,
        hardware_id="nvidia-a100",
        energy_wh_mean=1000.0,
        co2_g_mean=500.0,
        n_records=3,
        training_measurement_status="qualified",
        training_measurement_reason="training_measurements_qualified_confirmatory",
        training_measurement_qualified_confirmatory=True,
        training_host_id=HOST_ID,
        training_calibration_sha256=CALIBRATION_SHA256,
    )


def _canonical_record(
    *,
    energy_j: float,
    co2_operational_kg: float,
    co2_embodied_kg: float,
    items: int,
    hardware_id: str,
    **extra: Any,
) -> dict[str, Any]:
    qualified_extra: dict[str, Any] = {
        "measurement_qualified_confirmatory": True,
        "fallback": False,
        "host_id": HOST_ID,
        "calibration_sha256": CALIBRATION_SHA256,
        "provenance_sha256": PROVENANCE_SHA256,
        "artifact_sha256": ARTIFACT_SHA256,
        "instance_manifest_sha256": INSTANCE_MANIFEST_SHA256,
    }
    qualified_extra.update(extra)
    record = EnergyReading(
        duration_s=10.0,
        energy_j=energy_j,
        co2_operational_kg=co2_operational_kg,
        co2_embodied_kg=co2_embodied_kg,
        items_processed=items,
        backend="wall_meter",
        energy_domains=("whole_system_ac",),
        measurement_scope="whole_system_ac",
        hardware={"id": hardware_id, "pue": 1.0},
        extra=qualified_extra,
    ).to_dict()
    return record


def _canonical_pair() -> tuple[dict[str, Any], dict[str, Any]]:
    inference = _canonical_record(
        energy_j=3600.0,
        co2_operational_kg=0.004,
        co2_embodied_kg=0.001,
        items=1000,
        hardware_id="canonical-gpu",
        variant="AM",
        batch_size=128,
        size_key=50,
        gap_to_reference_pct=1.0,
        quality_feasible_confirmatory=True,
        delta_energy_j_per_item_ci_low=144.0,
        delta_energy_j_per_item_ci_high=216.0,
        seed=1,
    )
    baseline = _canonical_record(
        energy_j=18000.0,
        co2_operational_kg=0.002,
        co2_embodied_kg=0.0005,
        items=100,
        hardware_id="canonical-cpu",
        size=50,
        thread_mode="mono",
        gap_to_reference_pct=0.0,
    )
    return inference, baseline


def _one_row(
    inference: dict[str, Any],
    baseline: dict[str, Any],
    *,
    delta: float = 1.0,
    confirmatory_bundle_valid: bool = True,
) -> dict[str, Any]:
    rows = build_aet_table(
        _train_aggregate(),
        [inference],
        [baseline],
        deltas=[delta],
        confirmatory_bundle_valid=confirmatory_bundle_valid,
    )
    assert len(rows) == 1
    return rows[0]


def test_canonical_si_round_trip_is_finite_and_takes_precedence() -> None:
    inference, baseline = _canonical_pair()

    # Contradictory legacy aliases must not override canonical SI fields.
    inference.update(
        {
            "energy_wh": 999.0,
            "energy_wh_per_item": 999.0,
            "n_instances": 1,
            "co2_g_total": 999.0,
            "hardware_id": "legacy-gpu",
            "throughput": 1.0,
        }
    )
    baseline.update({"energy_wh": 1.0, "num_problems": 1, "co2_g_total": 999.0})

    row = _one_row(inference, baseline)

    assert row["feasible"] is True
    assert row["quality_status"] == "feasible"
    assert row["measurement_status"] == "qualified"
    assert row["measurement_qualified_confirmatory"] is True
    assert row["aet_E_status"] == "finite"
    assert row["aet_status"] == "finite"
    assert row["hardware_id"] == "canonical-gpu"
    assert row["throughput_items_per_s"] == pytest.approx(100.0)
    assert row["E_NN_wh_per_inst"] == pytest.approx(0.001)
    assert row["E_meta_wh_per_inst"] == pytest.approx(0.05)
    assert row["delta_E_wh_per_inst"] == pytest.approx(0.049)
    assert row["aet_E"] == pytest.approx(1000.0 / 0.049)
    assert row["C_NN_g_per_inst"] == pytest.approx(0.005)
    assert row["C_meta_g_per_inst"] == pytest.approx(0.025)


def test_aggregate_training_prefers_canonical_si_fields() -> None:
    records = [
        _canonical_record(
            energy_j=360000.0,
            co2_operational_kg=0.04,
            co2_embodied_kg=0.01,
            items=1,
            hardware_id="canonical-gpu",
        ),
        _canonical_record(
            energy_j=720000.0,
            co2_operational_kg=0.06,
            co2_embodied_kg=0.02,
            items=1,
            hardware_id="canonical-gpu",
        ),
    ]
    records[0].update({"energy_wh": 999.0, "co2_g_total": 999.0, "hardware_id": "legacy-gpu"})

    aggregate = aggregate_training(records)

    assert aggregate.energy_wh_median == pytest.approx(150.0)
    assert aggregate.energy_wh_mean == pytest.approx(150.0)
    assert aggregate.co2_g_median == pytest.approx(65.0)
    assert aggregate.co2_g_mean == pytest.approx(65.0)
    assert aggregate.n_seeds == 2
    assert aggregate.n_records == 2
    assert aggregate.training_measurement_status == "qualified"
    assert aggregate.training_measurement_qualified_confirmatory is True
    assert aggregate.hardware_id == "canonical-gpu"


def test_invalid_training_record_is_counted_and_blocks_qualification() -> None:
    valid = _canonical_record(
        energy_j=360000.0,
        co2_operational_kg=0.04,
        co2_embodied_kg=0.01,
        items=1,
        hardware_id="canonical-gpu",
    )
    invalid = _canonical_record(
        energy_j=720000.0,
        co2_operational_kg=0.06,
        co2_embodied_kg=0.02,
        items=1,
        hardware_id="canonical-gpu",
    )
    invalid["energy_j"] = float("nan")

    aggregate = aggregate_training([valid, invalid])

    assert aggregate.n_records == 2
    assert aggregate.n_seeds == 1
    assert aggregate.energy_wh_mean == pytest.approx(100.0)
    assert aggregate.training_measurement_status == "invalid_input"
    assert aggregate.training_measurement_reason == "invalid_training_energy_record"
    assert aggregate.training_measurement_qualified_confirmatory is False


def test_primary_aet_uses_training_mean_not_median() -> None:
    inference, baseline = _canonical_pair()
    aggregate = _train_aggregate()
    aggregate.energy_wh_mean = 200.0
    aggregate.energy_wh_median = 100.0

    rows = build_aet_table(
        aggregate,
        [inference],
        [baseline],
        deltas=[1.0],
        confirmatory_bundle_valid=True,
    )

    assert rows[0]["E_train_wh_mean"] == pytest.approx(200.0)
    assert rows[0]["E_train_wh_median"] == pytest.approx(100.0)
    assert rows[0]["aet_E"] == pytest.approx(200.0 / 0.049)


def test_tdp_measurement_can_never_be_promoted_to_finite() -> None:
    inference, baseline = _canonical_pair()
    inference["backend"] = "tdp"
    inference["extra"]["tdp_fallback"] = True

    row = _one_row(inference, baseline)

    assert row["inference_measurement_status"] == "invalid_input"
    assert row["measurement_status"] == "invalid_input"
    assert row["measurement_reason"] == "inference_tdp_backend_disallowed"
    assert row["aet_E_status"] == "invalid_input"
    assert math.isnan(row["aet_E"])
    assert math.isnan(row["aet_E_point_estimate"])


def test_fallback_measurement_can_never_be_promoted_to_finite() -> None:
    inference, baseline = _canonical_pair()
    baseline["extra"]["fallback"] = True

    row = _one_row(inference, baseline)

    assert row["baseline_measurement_status"] == "invalid_input"
    assert row["measurement_status"] == "invalid_input"
    assert row["measurement_reason"] == "baseline_fallback_measurement_disallowed"
    assert row["aet_E_status"] == "invalid_input"
    assert math.isnan(row["aet_E"])


@pytest.mark.parametrize(
    ("location", "key", "value", "reason"),
    [
        ("record", "backend", "hwcounters", "non_wall_meter_backend_claim"),
        ("record", "measurement_scope", "it_components", "invalid_measurement_scope"),
        ("record", "energy_domains", ["gpu"], "invalid_energy_domains"),
        ("hardware", "pue", 1.4, "pue_must_equal_one"),
    ],
)
def test_nonqualifying_measurement_contract_is_invalid(
    location: str, key: str, value: Any, reason: str
) -> None:
    inference, baseline = _canonical_pair()
    target = inference if location == "record" else inference["hardware"]
    target[key] = value

    row = _one_row(inference, baseline)

    assert row["inference_measurement_reason"] == reason
    assert row["measurement_status"] == "invalid_input"
    assert row["aet_E_status"] == "invalid_input"


def test_unvalidated_bundle_cannot_be_promoted_by_record_flags() -> None:
    inference, baseline = _canonical_pair()

    rows = build_aet_table(
        _train_aggregate(),
        [inference],
        [baseline],
        deltas=[1.0],
    )

    assert rows[0]["confirmatory_bundle_valid"] is False
    assert rows[0]["measurement_status"] == "unidentified"
    assert rows[0]["measurement_reason"] == "confirmatory_bundle_not_validated"
    assert rows[0]["aet_E_status"] == "unidentified"
    assert math.isnan(rows[0]["aet_E"])


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("host_id", "other-host", "measurement_host_id_mismatch"),
        ("calibration_sha256", "e" * 64, "measurement_calibration_sha256_mismatch"),
        ("instance_manifest_sha256", "f" * 64, "instance_manifest_sha256_mismatch"),
    ],
)
def test_comparison_identity_mismatch_is_invalid(key: str, value: str, reason: str) -> None:
    inference, baseline = _canonical_pair()
    baseline["extra"][key] = value

    row = _one_row(inference, baseline)

    assert row["measurement_status"] == "invalid_input"
    assert row["measurement_reason"] == reason
    assert row["aet_E_status"] == "invalid_input"


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        ("provenance_sha256", "invalid_provenance_sha256"),
        ("artifact_sha256", "invalid_artifact_sha256"),
    ],
)
def test_invalid_evidence_hash_is_invalid_input(key: str, reason: str) -> None:
    inference, baseline = _canonical_pair()
    inference["extra"][key] = "not-a-sha"

    row = _one_row(inference, baseline)

    assert row["inference_measurement_reason"] == reason
    assert row["aet_E_status"] == "invalid_input"


def test_load_records_is_strict_by_default_with_forensic_opt_out(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(FileNotFoundError, match="no JSON record matched"):
        load_records([str(missing)])
    assert load_records([str(missing)], strict=False) == []

    malformed = tmp_path / "malformed.json"
    malformed.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed JSON record"):
        load_records([str(malformed)])
    assert load_records([str(malformed)], strict=False) == []


@pytest.mark.parametrize("bad_gap", [None, float("nan"), float("inf"), float("-inf")])
def test_missing_or_nonfinite_nn_quality_is_invalid_not_zero(bad_gap: float | None) -> None:
    inference, baseline = _canonical_pair()
    inference["extra"]["gap_to_reference_pct"] = bad_gap

    row = _one_row(inference, baseline)

    assert row["feasible"] is None
    assert row["quality_status"] == "invalid_input"
    assert row["aet_E_status"] == "invalid_input"
    assert row["aet_E_reason"] == "missing_or_nonfinite_quality"
    assert math.isnan(row["nn_gap_pct"])
    assert math.isnan(row["aet_E"])


def test_absent_baseline_quality_is_invalid_not_zero() -> None:
    inference, baseline = _canonical_pair()
    del baseline["extra"]["gap_to_reference_pct"]

    row = _one_row(inference, baseline)

    assert row["feasible"] is None
    assert row["quality_status"] == "invalid_input"
    assert row["aet_E_status"] == "invalid_input"
    assert math.isnan(row["baseline_gap_pct"])
    assert math.isnan(row["aet_E"])


def test_measured_quality_failure_has_infinite_threshold() -> None:
    inference, baseline = _canonical_pair()
    inference["extra"]["quality_feasible_confirmatory"] = False

    row = _one_row(inference, baseline, delta=0.5)

    assert row["feasible"] is False
    assert row["quality_status"] == "infeasible"
    assert row["aet_E_status"] == "infinite"
    assert row["aet_E_reason"] == "quality_infeasible"
    assert math.isinf(row["aet_E"])


def test_quality_failure_remains_infinite_when_measurement_is_unqualified() -> None:
    inference, baseline = _canonical_pair()
    inference["extra"]["quality_feasible_confirmatory"] = False
    del inference["extra"]["measurement_qualified_confirmatory"]

    row = _one_row(inference, baseline, delta=0.5)

    assert row["measurement_status"] == "unidentified"
    assert row["quality_status"] == "infeasible"
    assert row["aet_E_status"] == "infinite"
    assert row["aet_E_reason"] == "quality_infeasible"
    assert math.isinf(row["aet_E"])


def test_non_positive_energy_saving_has_infinite_threshold() -> None:
    inference, baseline = _canonical_pair()
    baseline["energy_j"] = inference["energy_j"]
    baseline["items_processed"] = inference["items_processed"]
    inference["extra"]["delta_energy_j_per_item_ci_low"] = -72.0
    inference["extra"]["delta_energy_j_per_item_ci_high"] = 0.0

    row = _one_row(inference, baseline)

    assert row["feasible"] is True
    assert row["aet_E_status"] == "infinite"
    assert row["aet_E_reason"] == "saving_ci_non_positive"
    assert row["delta_E_wh_per_inst"] == pytest.approx(0.0)
    assert math.isinf(row["aet_E"])


def test_energy_saving_interval_crossing_zero_is_unidentified() -> None:
    inference, baseline = _canonical_pair()
    inference["extra"]["delta_energy_j_per_item_ci_low"] = -36.0
    inference["extra"]["delta_energy_j_per_item_ci_high"] = 216.0

    row = _one_row(inference, baseline)

    assert row["feasible"] is True
    assert row["aet_E_status"] == "unidentified"
    assert row["aet_E_reason"] == "saving_ci_includes_zero"
    assert row["delta_E_wh_per_inst_ci_low"] == pytest.approx(-0.01)
    assert row["delta_E_wh_per_inst_ci_high"] == pytest.approx(0.06)
    assert math.isnan(row["aet_E"])


def test_partial_energy_saving_interval_is_invalid() -> None:
    inference, baseline = _canonical_pair()
    del inference["extra"]["delta_energy_j_per_item_ci_high"]

    row = _one_row(inference, baseline)

    assert row["aet_E_status"] == "invalid_input"
    assert row["aet_E_reason"] == "malformed_saving_confidence_interval"
    assert math.isnan(row["aet_E"])


def test_missing_energy_is_invalid_not_zero() -> None:
    inference, baseline = _canonical_pair()
    inference["energy_j"] = None

    row = _one_row(inference, baseline)

    assert row["feasible"] is True
    assert row["aet_E_status"] == "invalid_input"
    assert row["aet_E_reason"] == "inference_invalid_energy_j"
    assert math.isnan(row["E_NN_wh_per_inst"])
    assert math.isnan(row["aet_E"])


def test_legacy_wh_and_grams_records_remain_supported() -> None:
    inference = {
        "variant": "AM",
        "batch_size": 32,
        "n_instances": 100,
        "energy_wh_per_item": 0.01,
        "co2_g_total": 1.0,
        "gap_to_bks": 1.0,
        "size_key": 50,
        "hardware_id": "legacy-gpu",
    }
    baseline = {
        "num_problems": 10,
        "energy_wh": 1.0,
        "co2_g_total": 0.5,
        "baseline_gap": 0.0,
        "size": 50,
    }

    row = _one_row(inference, baseline)

    assert row["feasible"] is True
    assert row["aet_E_status"] == "unidentified"
    assert row["quality_status"] == "unidentified"
    assert row["quality_evidence"] == "scalar_gap_point_estimate"
    assert row["measurement_status"] == "unidentified"
    assert row["aet_E_reason"] == "inference_missing_measurement_qualification"
    assert math.isnan(row["aet_E"])
    assert row["aet_E_point_estimate"] == pytest.approx(1000.0 / 0.09)
    assert row["E_NN_wh_per_inst"] == pytest.approx(0.01)
    assert row["E_meta_wh_per_inst"] == pytest.approx(0.1)
    assert row["hardware_id"] == "legacy-gpu"


def test_missing_energy_interval_cannot_be_confirmatory_finite() -> None:
    inference, baseline = _canonical_pair()
    del inference["extra"]["delta_energy_j_per_item_ci_low"]
    del inference["extra"]["delta_energy_j_per_item_ci_high"]

    row = _one_row(inference, baseline)

    assert row["feasible"] is True
    assert row["aet_E_status"] == "unidentified"
    assert row["aet_E_reason"] == "missing_saving_confidence_interval"
    assert math.isnan(row["aet_E"])
    assert row["aet_E_point_estimate"] == pytest.approx(1000.0 / 0.049)


def test_positive_energy_ci_without_confirmatory_quality_cannot_be_finite() -> None:
    inference, baseline = _canonical_pair()
    del inference["extra"]["quality_feasible_confirmatory"]

    row = _one_row(inference, baseline)

    assert row["feasible"] is True  # backward-compatible scalar-gap result
    assert row["quality_point_feasible"] is True
    assert row["quality_feasible_confirmatory"] is None
    assert row["quality_status"] == "unidentified"
    assert row["quality_evidence"] == "scalar_gap_point_estimate"
    assert row["aet_E_status"] == "unidentified"
    assert row["aet_E_reason"] == "quality_not_confirmatory"
    assert math.isnan(row["aet_E"])
    assert row["aet_E_point_estimate"] == pytest.approx(1000.0 / 0.049)


def test_size_mismatch_is_invalid_instead_of_crossing_all_baselines() -> None:
    inference, baseline = _canonical_pair()
    baseline["extra"]["size"] = 100

    row = _one_row(inference, baseline)

    assert row["baseline_match_status"] == "invalid_input"
    assert row["feasible"] is None
    assert row["quality_status"] == "invalid_input"
    assert row["aet_E_status"] == "invalid_input"
    assert math.isnan(row["aet_E"])
