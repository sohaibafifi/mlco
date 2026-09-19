"""Quality-constrained Amortized Efficiency Threshold computation.

For each combination of (variant, size, batch_size, thread_mode, delta)::

    feasible = gap_NN <= baseline_gap + delta
    AET_E = E_train_aggregate / (E_meta_per_inst - E_NN_per_inst)
    AET_C = C_train_aggregate / (C_meta_per_inst - C_NN_per_inst)

The output distinguishes four states instead of encoding every failure as
``+inf``:

``finite``
    Confirmatory quality and measurement pass, the complete input bundle is
    validated, and the saving interval is strictly positive.
``infinite``
    Confirmatory quality fails, or the saving interval is non-positive.
``unidentified``
    Confirmatory evidence is absent or the saving interval includes zero.
``invalid_input``
    A required value is missing, non-finite, negative where prohibited, or an
    optional confidence interval is malformed.

Missing quality is never imputed. In particular, ``None`` and non-finite gaps
cannot become zero. Canonical :class:`EnergyReading` SI fields are consumed
first. Legacy Wh and gCO2eq fields remain explicit fallbacks for old artifacts.
The primary training numerator is the arithmetic mean over valid seeds;
median and IQR are retained only as descriptive compatibility statistics.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

# Public legacy constant retained for import compatibility. The confirmatory
# estimator does not regularize its denominator.
EPS: float = 1e-12

STATUS_FINITE = "finite"
STATUS_INFINITE = "infinite"
STATUS_UNIDENTIFIED = "unidentified"
STATUS_INVALID = "invalid_input"

QUALITY_FEASIBLE = "feasible"
QUALITY_INFEASIBLE = "infeasible"
QUALITY_UNIDENTIFIED = STATUS_UNIDENTIFIED
QUALITY_INVALID = STATUS_INVALID

MEASUREMENT_QUALIFIED = "qualified"


@dataclass
class TrainAggregate:
    """Primary mean plus descriptive median/IQR across valid training seeds."""

    energy_wh_median: float
    energy_wh_p25: float
    energy_wh_p75: float
    co2_g_median: float
    co2_g_p25: float
    co2_g_p75: float
    n_seeds: int
    hardware_id: str | None
    raw: list[dict[str, Any]] = field(default_factory=list)
    energy_wh_mean: float = float("nan")
    co2_g_mean: float = float("nan")
    n_records: int = 0
    training_measurement_status: str = STATUS_UNIDENTIFIED
    training_measurement_reason: str = "missing_training_measurement_evidence"
    training_measurement_qualified_confirmatory: bool = False
    training_host_id: str | None = None
    training_calibration_sha256: str | None = None


@dataclass(frozen=True)
class _MeasurementQualification:
    status: str
    reason: str
    host_id: str | None = None
    calibration_sha256: str | None = None
    provenance_sha256: str | None = None
    artifact_sha256: str | None = None
    instance_manifest_sha256: str | None = None


def load_records(paths: Iterable[str], *, strict: bool = True) -> list[dict[str, Any]]:
    """Load JSON records from one or more glob patterns or paths.

    Each file may contain a single dict or a list of dicts. Strict mode is the
    default and raises on missing paths, unreadable files, malformed JSON, and
    non-object records. ``strict=False`` is an explicit forensic compatibility
    mode that skips such inputs.
    """
    records: list[dict[str, Any]] = []
    for pat in paths:
        matches = sorted(glob.glob(pat))
        if not matches:
            if strict:
                raise FileNotFoundError(f"no JSON record matched path or pattern: {pat}")
            continue
        for path in matches:
            try:
                with open(path) as f:
                    data = json.load(f)
            except OSError:
                if strict:
                    raise
                continue
            except json.JSONDecodeError as exc:
                if strict:
                    raise ValueError(f"malformed JSON record: {path}") from exc
                continue
            if isinstance(data, dict):
                records.append(data)
            elif isinstance(data, list):
                invalid_entries = [entry for entry in data if not isinstance(entry, dict)]
                if invalid_entries and strict:
                    raise TypeError(f"JSON record list contains non-object entries: {path}")
                records.extend(entry for entry in data if isinstance(entry, dict))
            elif strict:
                raise TypeError(f"JSON record must be an object or list of objects: {path}")
    return records


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for absent/invalid input."""
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _first_present(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, Any]:
    """Return the first present key without treating zero as missing."""
    for key in keys:
        if key in record:
            return True, record[key]
    return False, None


def _metadata_value(record: dict[str, Any], key: str) -> Any:
    """Read canonical nested metadata before a legacy top-level field."""
    extra = record.get("extra")
    if isinstance(extra, dict) and key in extra:
        return extra[key]
    return record.get(key)


def _metadata_present(record: dict[str, Any], key: str) -> bool:
    extra = record.get("extra")
    return (isinstance(extra, dict) and key in extra) or key in record


def _first_metadata_finite(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    """Read the first present nested/top-level metadata alias as finite."""
    for key in keys:
        if _metadata_present(record, key):
            return _finite_float(_metadata_value(record, key))
    return None


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _valid_sha256(value: Any) -> str | None:
    normalized = _nonempty_string(value)
    if normalized is None or len(normalized) != 64:
        return None
    lowered = normalized.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        return None
    return lowered


def _record_measurement_qualification(
    record: dict[str, Any],
    *,
    role: str,
) -> _MeasurementQualification:
    """Validate the evidence required for a confirmatory energy measurement."""
    backend = _nonempty_string(record.get("backend"))

    fallback_values: dict[str, bool] = {}
    for key in ("fallback", "allow_fallback", "tdp_fallback"):
        if not _metadata_present(record, key):
            continue
        value = _metadata_value(record, key)
        if not isinstance(value, bool):
            return _MeasurementQualification(STATUS_INVALID, f"invalid_{key}_flag")
        fallback_values[key] = value

    if backend is not None and backend.lower() == "tdp":
        return _MeasurementQualification(STATUS_INVALID, "tdp_backend_disallowed")
    if any(fallback_values.values()):
        return _MeasurementQualification(STATUS_INVALID, "fallback_measurement_disallowed")

    proof_key = "measurement_qualified_confirmatory"
    if not _metadata_present(record, proof_key):
        return _MeasurementQualification(STATUS_UNIDENTIFIED, "missing_measurement_qualification")
    explicit_proof = _metadata_value(record, proof_key)
    if not isinstance(explicit_proof, bool):
        return _MeasurementQualification(STATUS_INVALID, "invalid_measurement_qualification_flag")
    if not explicit_proof:
        return _MeasurementQualification(
            STATUS_UNIDENTIFIED, "measurement_not_qualified_confirmatory"
        )

    # A true proof flag is a claim that must agree with the canonical record.
    if record.get("schema_version") != "1.0":
        return _MeasurementQualification(STATUS_INVALID, "noncanonical_energy_schema")
    units = record.get("units")
    if not isinstance(units, dict) or units.get("energy_j") != "J":
        return _MeasurementQualification(STATUS_INVALID, "invalid_energy_units")
    energy_j = _finite_float(record.get("energy_j"))
    if energy_j is None or energy_j < 0:
        return _MeasurementQualification(STATUS_INVALID, "invalid_energy_j")
    co2_total_kg = _finite_float(record.get("co2_total_kg"))
    if co2_total_kg is None or co2_total_kg < 0:
        return _MeasurementQualification(STATUS_INVALID, "invalid_co2_total_kg")
    if role in {"inference", "baseline"}:
        items = _finite_float(record.get("items_processed"))
        if items is None or items <= 0:
            return _MeasurementQualification(STATUS_INVALID, "invalid_items_processed")

    if backend != "wall_meter":
        return _MeasurementQualification(STATUS_INVALID, "non_wall_meter_backend_claim")
    if record.get("measurement_scope") != "whole_system_ac":
        return _MeasurementQualification(STATUS_INVALID, "invalid_measurement_scope")
    domains = record.get("energy_domains")
    if (
        isinstance(domains, str)
        or not isinstance(domains, (list, tuple))
        or tuple(domains) != ("whole_system_ac",)
    ):
        return _MeasurementQualification(STATUS_INVALID, "invalid_energy_domains")

    hardware = record.get("hardware")
    pue = _finite_float(hardware.get("pue")) if isinstance(hardware, dict) else None
    if pue != 1.0:
        return _MeasurementQualification(STATUS_INVALID, "pue_must_equal_one")

    fallback_evidence = fallback_values.get("fallback") is False or (
        fallback_values.get("allow_fallback") is False
        and fallback_values.get("tdp_fallback") is False
    )
    if not fallback_evidence:
        return _MeasurementQualification(STATUS_INVALID, "missing_no_fallback_evidence")

    host_id = _nonempty_string(_metadata_value(record, "host_id"))
    calibration_sha = _valid_sha256(_metadata_value(record, "calibration_sha256"))
    provenance_sha = _valid_sha256(_metadata_value(record, "provenance_sha256"))
    artifact_sha = _valid_sha256(_metadata_value(record, "artifact_sha256"))
    if host_id is None:
        return _MeasurementQualification(STATUS_INVALID, "invalid_host_id")
    if calibration_sha is None:
        return _MeasurementQualification(STATUS_INVALID, "invalid_calibration_sha256")
    if provenance_sha is None:
        return _MeasurementQualification(STATUS_INVALID, "invalid_provenance_sha256")
    if artifact_sha is None:
        return _MeasurementQualification(STATUS_INVALID, "invalid_artifact_sha256")

    instance_sha = None
    if role in {"inference", "baseline"}:
        instance_sha = _valid_sha256(_metadata_value(record, "instance_manifest_sha256"))
        if instance_sha is None:
            return _MeasurementQualification(STATUS_INVALID, "invalid_instance_manifest_sha256")

    return _MeasurementQualification(
        MEASUREMENT_QUALIFIED,
        "measurement_qualified_confirmatory",
        host_id=host_id,
        calibration_sha256=calibration_sha,
        provenance_sha256=provenance_sha,
        artifact_sha256=artifact_sha,
        instance_manifest_sha256=instance_sha,
    )


def _hardware_id(record: dict[str, Any]) -> str | None:
    """Read canonical ``hardware.id`` before the legacy top-level alias."""
    hardware = record.get("hardware")
    if isinstance(hardware, dict):
        value = hardware.get("id")
        if value is not None and str(value).strip():
            return str(value)
    value = record.get("hardware_id")
    if value is not None and str(value).strip():
        return str(value)
    return None


def _item_count(record: dict[str, Any], legacy_key: str) -> float | None:
    """Read canonical item count, falling back only when it is absent."""
    if "items_processed" in record:
        count = _finite_float(record["items_processed"])
    else:
        count = _finite_float(record.get(legacy_key))
    return count if count is not None and count > 0 else None


def _total_energy_wh(record: dict[str, Any]) -> float | None:
    """Read total energy, preferring canonical joules over legacy Wh."""
    if "energy_j" in record:
        energy_j = _finite_float(record["energy_j"])
        if energy_j is None or energy_j < 0:
            return None
        return energy_j / 3600.0
    energy_wh = _finite_float(record.get("energy_wh"))
    if energy_wh is None or energy_wh < 0:
        return None
    return energy_wh


def _energy_wh_per_item(record: dict[str, Any], count: float | None) -> float | None:
    """Read per-item energy using SI totals first and legacy aliases second."""
    if "energy_j" in record:
        total_wh = _total_energy_wh(record)
        if total_wh is None or count is None:
            return None
        return total_wh / count

    if "energy_wh_per_item" in record:
        value = _finite_float(record["energy_wh_per_item"])
        return value if value is not None and value >= 0 else None

    total_wh = _total_energy_wh(record)
    if total_wh is None or count is None:
        return None
    return total_wh / count


def _total_co2_g(record: dict[str, Any], *, include_embodied: bool) -> float | None:
    """Read total carbon in grams, preferring canonical kilograms."""
    if include_embodied:
        canonical_key = "co2_total_kg"
        legacy_keys = ("co2_g_total", "co2_g_operational")
    else:
        canonical_key = "co2_operational_kg"
        legacy_keys = ("co2_g_operational",)

    if canonical_key in record:
        value_kg = _finite_float(record[canonical_key])
        if value_kg is None or value_kg < 0:
            return None
        return value_kg * 1000.0

    present, legacy_value = _first_present(record, legacy_keys)
    if present:
        value_g = _finite_float(legacy_value)
        return value_g if value_g is not None and value_g >= 0 else None

    # Old records sometimes stored only operational kilograms. When embodied
    # kilograms are also present, add them explicitly.
    if "co2_operational_kg" not in record:
        return None
    operational = _finite_float(record["co2_operational_kg"])
    if operational is None or operational < 0:
        return None
    embodied = 0.0
    if include_embodied and "co2_embodied_kg" in record:
        embodied_value = _finite_float(record["co2_embodied_kg"])
        if embodied_value is None or embodied_value < 0:
            return None
        embodied = embodied_value
    return (operational + embodied) * 1000.0


def _co2_g_per_item(
    record: dict[str, Any],
    count: float | None,
    *,
    include_embodied: bool,
) -> float | None:
    total_g = _total_co2_g(record, include_embodied=include_embodied)
    if total_g is None or count is None:
        return None
    return total_g / count


def _gap_pct(record: dict[str, Any], *, baseline: bool) -> float | None:
    """Read an independently referenced gap without ever imputing zero."""
    keys = (
        ("gap_to_reference_pct", "baseline_gap", "gap_to_bks")
        if baseline
        else ("gap_to_reference_pct", "gap_to_bks")
    )
    return _first_metadata_finite(record, keys)


def _delta_energy_ci_wh_per_item(
    inference: dict[str, Any], baseline: dict[str, Any]
) -> tuple[str, tuple[float, float] | None]:
    """Read an optional comparison-level saving interval.

    The canonical fields are ``delta_energy_j_per_item_ci_low`` and
    ``delta_energy_j_per_item_ci_high``. They describe
    ``baseline - inference`` and may be attached to either record. Legacy Wh
    aliases are accepted. A partial, conflicting, non-finite, or reversed
    interval is invalid. Absence is distinct from invalidity.
    """

    def from_record(record: dict[str, Any]) -> tuple[str, tuple[float, float] | None]:
        canonical = (
            "delta_energy_j_per_item_ci_low",
            "delta_energy_j_per_item_ci_high",
        )
        legacy = (
            "delta_energy_wh_per_item_ci_low",
            "delta_energy_wh_per_item_ci_high",
        )
        canonical_present = any(_metadata_present(record, key) for key in canonical)
        legacy_present = any(_metadata_present(record, key) for key in legacy)
        if canonical_present:
            if not all(_metadata_present(record, key) for key in canonical):
                return STATUS_INVALID, None
            lo = _finite_float(_metadata_value(record, canonical[0]))
            hi = _finite_float(_metadata_value(record, canonical[1]))
            scale = 1.0 / 3600.0
        elif legacy_present:
            if not all(_metadata_present(record, key) for key in legacy):
                return STATUS_INVALID, None
            lo = _finite_float(_metadata_value(record, legacy[0]))
            hi = _finite_float(_metadata_value(record, legacy[1]))
            scale = 1.0
        else:
            return "absent", None
        if lo is None or hi is None or lo > hi:
            return STATUS_INVALID, None
        return "present", (lo * scale, hi * scale)

    inf_state, inf_ci = from_record(inference)
    base_state, base_ci = from_record(baseline)
    if STATUS_INVALID in (inf_state, base_state):
        return STATUS_INVALID, None
    if inf_ci is not None and base_ci is not None:
        if inf_ci != base_ci:
            return STATUS_INVALID, None
        return "present", inf_ci
    if inf_ci is not None:
        return "present", inf_ci
    if base_ci is not None:
        return "present", base_ci
    return "absent", None


def _confirmatory_quality_flag(
    inference: dict[str, Any], baseline: dict[str, Any]
) -> tuple[str, bool | None]:
    """Read an external confirmatory quality decision without coercion."""
    key = "quality_feasible_confirmatory"

    def from_record(record: dict[str, Any]) -> tuple[str, bool | None]:
        if not _metadata_present(record, key):
            return "absent", None
        value = _metadata_value(record, key)
        if not isinstance(value, bool):
            return STATUS_INVALID, None
        return "present", value

    inf_state, inf_value = from_record(inference)
    base_state, base_value = from_record(baseline)
    if STATUS_INVALID in (inf_state, base_state):
        return STATUS_INVALID, None
    if inf_value is not None and base_value is not None and inf_value != base_value:
        return STATUS_INVALID, None
    if inf_value is not None:
        return "present", inf_value
    if base_value is not None:
        return "present", base_value
    return "absent", None


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    """Return (median, p25, p75); NaN if empty."""
    cleaned = sorted(v for v in values if math.isfinite(v))
    if not cleaned:
        return float("nan"), float("nan"), float("nan")
    n = len(cleaned)

    def q(p: float) -> float:
        idx = (n - 1) * p
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return cleaned[lo]
        frac = idx - lo
        return cleaned[lo] * (1 - frac) + cleaned[hi] * frac

    return q(0.5), q(0.25), q(0.75)


def _mean(values: list[float]) -> float:
    cleaned = [value for value in values if math.isfinite(value)]
    return sum(cleaned) / len(cleaned) if cleaned else float("nan")


def aggregate_training(records: list[dict[str, Any]]) -> TrainAggregate:
    """Aggregate valid training measurements with explicit qualification.

    Canonical SI fields from :meth:`EnergyReading.to_dict` take precedence:
    ``energy_j``, ``co2_total_kg``, and ``hardware.id``. The returned public
    columns remain Wh and gCO2eq for compatibility with existing AET tables.
    The arithmetic mean is the primary deployment training cost. Median and
    IQR remain descriptive compatibility fields.

    ``n_seeds`` counts records with valid energy only. A malformed record is
    never silently promoted away: it makes the aggregate confirmatory status
    ``invalid_input`` even though descriptive statistics over valid records
    remain available.
    """
    energies: list[float] = []
    co2s: list[float] = []
    qualifications: list[_MeasurementQualification] = []
    invalid_energy_record = False
    for r in records:
        e = _total_energy_wh(r)
        if e is not None:
            energies.append(e)
        else:
            invalid_energy_record = True
        c = _total_co2_g(r, include_embodied=True)
        if c is not None:
            co2s.append(c)
        qualifications.append(_record_measurement_qualification(r, role="training"))
    e_med, e_25, e_75 = _percentiles(energies)
    c_med, c_25, c_75 = _percentiles(co2s)
    e_mean = _mean(energies)
    c_mean = _mean(co2s)

    training_status = STATUS_UNIDENTIFIED
    training_reason = "missing_training_measurement_evidence"
    training_host_id = None
    training_calibration_sha = None
    if invalid_energy_record:
        training_status = STATUS_INVALID
        training_reason = "invalid_training_energy_record"
    elif any(q.status == STATUS_INVALID for q in qualifications):
        invalid = next(q for q in qualifications if q.status == STATUS_INVALID)
        training_status = STATUS_INVALID
        training_reason = f"training_{invalid.reason}"
    elif records and all(q.status == MEASUREMENT_QUALIFIED for q in qualifications):
        hosts = {q.host_id for q in qualifications}
        calibrations = {q.calibration_sha256 for q in qualifications}
        if len(hosts) != 1:
            training_status = STATUS_INVALID
            training_reason = "training_host_id_mismatch"
        elif len(calibrations) != 1:
            training_status = STATUS_INVALID
            training_reason = "training_calibration_sha256_mismatch"
        else:
            training_status = MEASUREMENT_QUALIFIED
            training_reason = "training_measurements_qualified_confirmatory"
            training_host_id = next(iter(hosts))
            training_calibration_sha = next(iter(calibrations))
    elif records:
        training_reason = "training_measurement_evidence_incomplete"

    hw = None
    for record in records:
        hw = _hardware_id(record)
        if hw is not None:
            break
    return TrainAggregate(
        energy_wh_median=e_med,
        energy_wh_p25=e_25,
        energy_wh_p75=e_75,
        co2_g_median=c_med,
        co2_g_p25=c_25,
        co2_g_p75=c_75,
        n_seeds=len(energies),
        hardware_id=hw,
        raw=records,
        energy_wh_mean=e_mean,
        co2_g_mean=c_mean,
        n_records=len(records),
        training_measurement_status=training_status,
        training_measurement_reason=training_reason,
        training_measurement_qualified_confirmatory=(training_status == MEASUREMENT_QUALIFIED),
        training_host_id=training_host_id,
        training_calibration_sha256=training_calibration_sha,
    )


def _comparison_measurement_qualification(
    train_agg: TrainAggregate,
    inference: _MeasurementQualification,
    baseline: _MeasurementQualification,
    *,
    confirmatory_bundle_valid: bool,
) -> _MeasurementQualification:
    """Validate measurement evidence across the complete comparison bundle."""
    if not isinstance(confirmatory_bundle_valid, bool):
        return _MeasurementQualification(STATUS_INVALID, "invalid_bundle_validation_flag")

    if train_agg.training_measurement_status == STATUS_INVALID:
        return _MeasurementQualification(STATUS_INVALID, train_agg.training_measurement_reason)
    if inference.status == STATUS_INVALID:
        return _MeasurementQualification(STATUS_INVALID, f"inference_{inference.reason}")
    if baseline.status == STATUS_INVALID:
        return _MeasurementQualification(STATUS_INVALID, f"baseline_{baseline.reason}")

    if not confirmatory_bundle_valid:
        return _MeasurementQualification(STATUS_UNIDENTIFIED, "confirmatory_bundle_not_validated")
    if train_agg.training_measurement_status != MEASUREMENT_QUALIFIED:
        return _MeasurementQualification(STATUS_UNIDENTIFIED, train_agg.training_measurement_reason)
    if not train_agg.training_measurement_qualified_confirmatory:
        return _MeasurementQualification(
            STATUS_INVALID, "training_qualification_flag_contradiction"
        )
    if inference.status != MEASUREMENT_QUALIFIED:
        return _MeasurementQualification(STATUS_UNIDENTIFIED, f"inference_{inference.reason}")
    if baseline.status != MEASUREMENT_QUALIFIED:
        return _MeasurementQualification(STATUS_UNIDENTIFIED, f"baseline_{baseline.reason}")

    training_mean = _finite_float(train_agg.energy_wh_mean)
    if training_mean is None or training_mean < 0:
        return _MeasurementQualification(STATUS_INVALID, "invalid_training_energy_mean")
    training_host = _nonempty_string(train_agg.training_host_id)
    training_calibration = _valid_sha256(train_agg.training_calibration_sha256)
    if training_host is None or training_calibration is None:
        return _MeasurementQualification(STATUS_INVALID, "invalid_training_measurement_identity")

    hosts = {training_host, inference.host_id, baseline.host_id}
    if len(hosts) != 1:
        return _MeasurementQualification(STATUS_INVALID, "measurement_host_id_mismatch")
    calibrations = {
        training_calibration,
        inference.calibration_sha256,
        baseline.calibration_sha256,
    }
    if len(calibrations) != 1:
        return _MeasurementQualification(STATUS_INVALID, "measurement_calibration_sha256_mismatch")
    if inference.instance_manifest_sha256 != baseline.instance_manifest_sha256:
        return _MeasurementQualification(STATUS_INVALID, "instance_manifest_sha256_mismatch")

    return _MeasurementQualification(
        MEASUREMENT_QUALIFIED,
        "comparison_measurements_qualified_confirmatory",
        host_id=training_host,
        calibration_sha256=training_calibration,
        instance_manifest_sha256=inference.instance_manifest_sha256,
    )


def _safe_div(num: float, den: float) -> float:
    """Divide valid non-negative cost by a strictly positive saving.

    Invalid input returns NaN. A non-positive denominator returns +inf. No
    epsilon regularizer is used in the estimator.
    """
    if not math.isfinite(num) or not math.isfinite(den) or num < 0:
        return float("nan")
    if den <= 0:
        return float("inf")
    return num / den


def _size_from_filename(path: str) -> int | None:
    name = os.path.basename(path)
    base = name.split(".")[0]
    for prefix in ("test_", "val_"):
        if base.startswith(prefix):
            try:
                return int(base[len(prefix) :])
            except ValueError:
                return None
    return None


def _quality_classification(
    nn_gap: float | None,
    baseline_gap: float | None,
    delta: Any,
    confirmatory_state: str,
    confirmatory_value: bool | None,
) -> tuple[bool | None, bool | None, str, float | None, str]:
    """Return reported/point feasibility, status, delta, and reason."""
    delta_value = _finite_float(delta)
    if nn_gap is None or baseline_gap is None or delta_value is None or delta_value < 0:
        return None, None, QUALITY_INVALID, delta_value, "missing_or_nonfinite_quality"
    point_feasible = nn_gap <= baseline_gap + delta_value
    if confirmatory_state == STATUS_INVALID:
        return (
            None,
            point_feasible,
            QUALITY_INVALID,
            delta_value,
            "invalid_confirmatory_quality_flag",
        )
    if confirmatory_state == "absent":
        return (
            point_feasible,
            point_feasible,
            QUALITY_UNIDENTIFIED,
            delta_value,
            "missing_confirmatory_quality_decision",
        )
    if confirmatory_value is True and not point_feasible:
        return (
            None,
            point_feasible,
            QUALITY_INVALID,
            delta_value,
            "confirmatory_quality_conflicts_with_scalar_gap",
        )
    if confirmatory_value is True:
        return (
            True,
            point_feasible,
            QUALITY_FEASIBLE,
            delta_value,
            "confirmatory_quality_feasible",
        )
    return (
        False,
        point_feasible,
        QUALITY_INFEASIBLE,
        delta_value,
        "confirmatory_quality_infeasible",
    )


def _classify_aet(
    train_cost: float,
    saving: float | None,
    quality_status: str,
    *,
    measurement_status: str,
    measurement_reason: str,
    ci_state: str = "absent",
    saving_ci: tuple[float, float] | None = None,
) -> tuple[float, str, str]:
    """Compute an AET value and its authoritative classification."""
    if measurement_status == STATUS_INVALID:
        return float("nan"), STATUS_INVALID, measurement_reason
    if quality_status == QUALITY_INVALID:
        return float("nan"), STATUS_INVALID, "missing_or_nonfinite_quality"
    if quality_status == QUALITY_INFEASIBLE:
        return float("inf"), STATUS_INFINITE, "quality_infeasible"
    if measurement_status != MEASUREMENT_QUALIFIED:
        return float("nan"), STATUS_UNIDENTIFIED, measurement_reason
    if quality_status == QUALITY_UNIDENTIFIED:
        return float("nan"), STATUS_UNIDENTIFIED, "quality_not_confirmatory"
    if not math.isfinite(train_cost) or train_cost < 0:
        return float("nan"), STATUS_INVALID, "missing_or_nonfinite_training_cost"
    if saving is None or not math.isfinite(saving):
        return float("nan"), STATUS_INVALID, "missing_or_nonfinite_deployment_cost"
    if ci_state == STATUS_INVALID:
        return float("nan"), STATUS_INVALID, "malformed_saving_confidence_interval"
    if saving_ci is not None:
        ci_low, ci_high = saving_ci
        if ci_low > 0:
            if saving <= 0:
                return float("nan"), STATUS_INVALID, "point_estimate_outside_positive_interval"
            return train_cost / saving, STATUS_FINITE, "saving_ci_strictly_positive"
        if ci_high <= 0:
            return float("inf"), STATUS_INFINITE, "saving_ci_non_positive"
        return float("nan"), STATUS_UNIDENTIFIED, "saving_ci_includes_zero"
    return float("nan"), STATUS_UNIDENTIFIED, "missing_saving_confidence_interval"


def _point_aet(train_cost: float, saving: float | None, feasible: bool | None) -> float:
    """Return the backward-compatible, explicitly non-confirmatory point ratio."""
    if feasible is None:
        return float("nan")
    if not feasible:
        return float("inf")
    if saving is None:
        return float("nan")
    return _safe_div(train_cost, saving)


def build_aet_table(
    train_agg: TrainAggregate,
    inference_records: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
    deltas: list[float],
    *,
    use_embodied_for_co2: bool = True,
    confirmatory_bundle_valid: bool = False,
) -> list[dict[str, Any]]:
    """Cross every inference record with matching baseline records per size.

    Canonical input comes directly from :meth:`EnergyReading.to_dict`:
    ``items_processed``, ``energy_j``, ``co2_*_kg``,
    ``throughput_items_per_s``, and ``hardware.id``. Historical Wh/g aliases
    are retained as fallbacks. Public output names stay in Wh and grams for
    backward compatibility.

    ``feasible`` is tri-state. It is ``None`` when either gap is missing or
    non-finite, never ``False``. In that case the AET values are NaN and their
    status is ``invalid_input``. The ``aet_E_status`` and ``aet_C_status``
    fields, not numeric finiteness alone, are authoritative.

    An optional comparison-level 95 percent interval for the per-item energy
    saving may be supplied as ``delta_energy_j_per_item_ci_low/high``. A range
    crossing zero yields ``unidentified``. Without such a range, the
    confirmatory classification is also ``unidentified``; the exploratory
    ratio is retained separately as ``aet_E_point_estimate``.

    Likewise, ``finite`` requires an externally computed
    ``quality_feasible_confirmatory=True`` decision for the one pre-registered
    delta passed to this call. Scalar gaps alone remain legacy point evidence
    and cannot promote a row beyond ``unidentified``.

    Confirmatory measurement additionally requires qualified canonical wall
    meter records for training, inference, and baseline, plus
    ``confirmatory_bundle_valid=True`` from a caller that has verified the
    complete expected manifest. Qualified records carry
    ``measurement_qualified_confirmatory``, ``host_id``,
    ``calibration_sha256``, ``provenance_sha256``, and ``artifact_sha256``;
    inference and baseline also carry the same ``instance_manifest_sha256``.
    The default is intentionally non-promoting.

    Returns a list of row dicts; the caller may build a `pandas.DataFrame`
    if desired (the analysis package keeps pandas as an optional dependency).
    """
    if not inference_records or not baseline_records:
        return []

    rows: list[dict[str, Any]] = []
    for inf in inference_records:
        inference_measurement = _record_measurement_qualification(inf, role="inference")
        n_inst = _item_count(inf, "n_instances")
        size_key = _metadata_value(inf, "size_key")
        if size_key is None:
            size_key = _metadata_value(inf, "size")
        if size_key is None:
            tf = _metadata_value(inf, "test_file")
            size_key = _size_from_filename(str(tf)) if tf else None

        e_nn = _energy_wh_per_item(inf, n_inst)
        c_nn = _co2_g_per_item(
            inf,
            n_inst,
            include_embodied=use_embodied_for_co2,
        )
        nn_gap = _gap_pct(inf, baseline=False)

        # Filter baselines by size when possible.
        matching = []
        for b in baseline_records:
            b_size = _metadata_value(b, "size")
            baseline_file = _metadata_value(b, "file")
            if b_size is None and baseline_file is not None:
                b_size = _size_from_filename(str(baseline_file))
            if size_key is None or b_size is None or b_size != size_key:
                continue
            matching.append((b, b_size))
        if not matching:
            # Preserve an explicit invalid row instead of silently crossing an
            # inference record with a baseline from another or unknown size.
            matching = [({}, None)]

        for base, b_size in matching:
            match_status = "matched" if base else STATUS_INVALID
            baseline_measurement = (
                _record_measurement_qualification(base, role="baseline")
                if base
                else _MeasurementQualification(STATUS_INVALID, "baseline_size_match_missing")
            )
            comparison_measurement = _comparison_measurement_qualification(
                train_agg,
                inference_measurement,
                baseline_measurement,
                confirmatory_bundle_valid=confirmatory_bundle_valid,
            )
            n_b = _item_count(base, "num_problems")
            e_meta = _energy_wh_per_item(base, n_b)
            c_meta = _co2_g_per_item(
                base,
                n_b,
                include_embodied=use_embodied_for_co2,
            )
            baseline_gap = _gap_pct(base, baseline=True)
            e_ci_state, e_saving_ci = _delta_energy_ci_wh_per_item(inf, base)
            quality_flag_state, quality_flag_value = _confirmatory_quality_flag(inf, base)
            if quality_flag_state == "present" and len(deltas) != 1:
                quality_flag_state = STATUS_INVALID
                quality_flag_value = None

            e_saving = e_meta - e_nn if e_meta is not None and e_nn is not None else None
            c_saving = c_meta - c_nn if c_meta is not None and c_nn is not None else None
            train_energy_primary = (
                train_agg.energy_wh_mean
                if math.isfinite(train_agg.energy_wh_mean)
                else train_agg.energy_wh_median
            )
            train_co2_primary = (
                train_agg.co2_g_mean
                if math.isfinite(train_agg.co2_g_mean)
                else train_agg.co2_g_median
            )

            for delta in deltas:
                (
                    feasible,
                    quality_point_feasible,
                    quality_status,
                    delta_value,
                    quality_reason,
                ) = _quality_classification(
                    nn_gap,
                    baseline_gap,
                    delta,
                    quality_flag_state,
                    quality_flag_value,
                )
                aet_E, aet_E_status, aet_E_reason = _classify_aet(
                    train_energy_primary,
                    e_saving,
                    quality_status,
                    measurement_status=comparison_measurement.status,
                    measurement_reason=comparison_measurement.reason,
                    ci_state=e_ci_state,
                    saving_ci=e_saving_ci,
                )
                aet_C, aet_C_status, aet_C_reason = _classify_aet(
                    train_co2_primary,
                    c_saving,
                    quality_status,
                    measurement_status=comparison_measurement.status,
                    measurement_reason=comparison_measurement.reason,
                )
                if comparison_measurement.status == STATUS_INVALID:
                    aet_E_point = aet_C_point = float("nan")
                else:
                    aet_E_point = _point_aet(train_energy_primary, e_saving, quality_point_feasible)
                    aet_C_point = _point_aet(train_co2_primary, c_saving, quality_point_feasible)

                if aet_E_status == STATUS_FINITE:
                    assert e_saving is not None
                    aet_E_p25 = _safe_div(train_agg.energy_wh_p25, e_saving)
                    aet_E_p75 = _safe_div(train_agg.energy_wh_p75, e_saving)
                elif aet_E_status == STATUS_INFINITE:
                    aet_E_p25 = aet_E_p75 = float("inf")
                else:
                    aet_E_p25 = aet_E_p75 = float("nan")

                max_runtime = _finite_float(_metadata_value(base, "max_runtime_s"))
                throughput = (
                    _finite_float(inf.get("throughput_items_per_s"))
                    if "throughput_items_per_s" in inf
                    else _finite_float(inf.get("throughput"))
                )
                rows.append(
                    {
                        "variant": _metadata_value(inf, "variant"),
                        "size": b_size if b_size is not None else size_key,
                        "batch_size": _metadata_value(inf, "batch_size"),
                        "threads_mode": _metadata_value(base, "thread_mode"),
                        "num_procs": _metadata_value(base, "num_procs"),
                        "baseline_match_status": match_status,
                        "baseline_max_runtime_s": (
                            max_runtime if max_runtime is not None else float("nan")
                        ),
                        "delta_pct": (delta_value if delta_value is not None else float("nan")),
                        "nn_gap_pct": nn_gap if nn_gap is not None else float("nan"),
                        "baseline_gap_pct": (
                            baseline_gap if baseline_gap is not None else float("nan")
                        ),
                        "feasible": feasible,
                        "quality_point_feasible": quality_point_feasible,
                        "quality_status": quality_status,
                        "quality_reason": quality_reason,
                        "quality_feasible_confirmatory": (
                            quality_flag_value if quality_flag_state == "present" else None
                        ),
                        "quality_evidence": (
                            "confirmatory_external_decision"
                            if quality_flag_state == "present"
                            else (
                                "scalar_gap_point_estimate"
                                if quality_status == QUALITY_UNIDENTIFIED
                                else "missing_or_invalid_quality_evidence"
                            )
                        ),
                        "measurement_status": comparison_measurement.status,
                        "measurement_reason": comparison_measurement.reason,
                        "measurement_qualified_confirmatory": (
                            comparison_measurement.status == MEASUREMENT_QUALIFIED
                        ),
                        "confirmatory_bundle_valid": (
                            confirmatory_bundle_valid
                            if isinstance(confirmatory_bundle_valid, bool)
                            else None
                        ),
                        "inference_measurement_status": inference_measurement.status,
                        "inference_measurement_reason": inference_measurement.reason,
                        "baseline_measurement_status": baseline_measurement.status,
                        "baseline_measurement_reason": baseline_measurement.reason,
                        "training_measurement_status": (train_agg.training_measurement_status),
                        "training_measurement_reason": (train_agg.training_measurement_reason),
                        "measurement_host_id": comparison_measurement.host_id,
                        "measurement_calibration_sha256": (
                            comparison_measurement.calibration_sha256
                        ),
                        "instance_manifest_sha256": (
                            comparison_measurement.instance_manifest_sha256
                        ),
                        "hardware_id": _hardware_id(inf),
                        "seed": _metadata_value(inf, "seed"),
                        "throughput_items_per_s": throughput,
                        "E_train_wh_mean": train_agg.energy_wh_mean,
                        "E_train_wh_median": train_agg.energy_wh_median,
                        "E_NN_wh_per_inst": e_nn if e_nn is not None else float("nan"),
                        "E_meta_wh_per_inst": (e_meta if e_meta is not None else float("nan")),
                        "delta_E_wh_per_inst": (e_saving if e_saving is not None else float("nan")),
                        "delta_E_wh_per_inst_ci_low": (
                            e_saving_ci[0] if e_saving_ci is not None else float("nan")
                        ),
                        "delta_E_wh_per_inst_ci_high": (
                            e_saving_ci[1] if e_saving_ci is not None else float("nan")
                        ),
                        "delta_E_ci_status": e_ci_state,
                        "C_train_g_mean": train_agg.co2_g_mean,
                        "C_train_g_median": train_agg.co2_g_median,
                        "C_NN_g_per_inst": c_nn if c_nn is not None else float("nan"),
                        "C_meta_g_per_inst": (c_meta if c_meta is not None else float("nan")),
                        "delta_C_g_per_inst": (c_saving if c_saving is not None else float("nan")),
                        "aet_E": aet_E,
                        "aet_E_point_estimate": aet_E_point,
                        "aet_E_p25": aet_E_p25,
                        "aet_E_p75": aet_E_p75,
                        "aet_E_status": aet_E_status,
                        "aet_E_reason": aet_E_reason,
                        "aet_E_evidence": (
                            "saving_confidence_interval"
                            if e_saving_ci is not None
                            else "point_estimate_only"
                        ),
                        "aet_C": aet_C,
                        "aet_C_point_estimate": aet_C_point,
                        "aet_C_status": aet_C_status,
                        "aet_C_reason": aet_C_reason,
                        "aet_C_evidence": "point_estimate_only",
                        # Energy is the primary historical AET quantity.
                        "aet_status": aet_E_status,
                        "aet_reason": aet_E_reason,
                    }
                )
    return rows
