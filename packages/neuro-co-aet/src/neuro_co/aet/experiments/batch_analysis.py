"""Fail-closed AET analysis for the Windows batch-frontier campaign.

The primary numerator is the mean energy needed to execute one sealed training
recipe, estimated from five independent seeds. It is never the sum of the five
replications. Selection and wider study debt are reported separately and are
never silently imputed.

The denominator is paired on a fixed HGS-10 baseline within each measured
round and batch size. The resulting metric is explicitly scoped to the CPU and
GPU component counters observed by the native-Windows campaign. This module
does not produce carbon or whole-system energy claims.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ANALYSIS_SCHEMA: Final = "aet-batch-analysis/v1"
MANIFEST_SCHEMA: Final = "aet-batch-analysis-manifest/v1"
COMPLETION_SCHEMA: Final = "aet-batch-experiment-completion/v1"
TRAINING_SUMMARY_SCHEMA: Final = "aet-training-debt-summary/v1"
FRONTIER_SUMMARY_SCHEMA: Final = "aet-batch-frontier-summary/v1"

STATUS_FINITE: Final = "finite"
STATUS_INFINITE: Final = "infinite"
STATUS_UNIDENTIFIED: Final = "unidentified"

EXPECTED_ARCHITECTURES: Final = ("am", "gnn")
EXPECTED_SEEDS: Final = (2, 3, 4, 5, 6)
T_CRITICAL_95_DF4: Final = 2.7764451051977987


class AETBatchAnalysisError(RuntimeError):
    """Raised when an input or provenance contract cannot be established."""


@dataclass(frozen=True)
class TrainingModel:
    architecture: str
    seed: int
    energy_j: float
    checkpoint_sha256: str
    model_state_sha256: str
    model_identity_sha256: str


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AETBatchAnalysisError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AETBatchAnalysisError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AETBatchAnalysisError(f"{path} must contain one JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AETBatchAnalysisError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise AETBatchAnalysisError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AETBatchAnalysisError(f"{where} must be a non-empty string")
    return value.strip()


def _sha256(value: Any, where: str) -> str:
    result = _string(value, where).lower()
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise AETBatchAnalysisError(f"{where} must be a lowercase SHA-256 digest")
    return result


def _finite_float(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AETBatchAnalysisError(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AETBatchAnalysisError(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise AETBatchAnalysisError(f"{where} must be {qualifier}")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AETBatchAnalysisError(f"{where} must be a positive integer")
    return value


def _architecture_entries(value: Any, where: str) -> dict[str, dict[str, Any]]:
    """Normalize either a keyed object or an array with architecture fields."""
    if isinstance(value, dict):
        entries = {
            _string(key, f"{where} key").lower(): _mapping(item, f"{where}.{key}")
            for key, item in value.items()
        }
    elif isinstance(value, list):
        entries = {}
        for index, item_value in enumerate(value):
            item = _mapping(item_value, f"{where}[{index}]")
            architecture = _string(
                item.get("architecture"), f"{where}[{index}].architecture"
            ).lower()
            if architecture in entries:
                raise AETBatchAnalysisError(f"duplicate architecture {architecture!r}")
            entries[architecture] = item
    else:
        raise AETBatchAnalysisError(f"{where} must be an object or array")
    if set(entries) != set(EXPECTED_ARCHITECTURES):
        raise AETBatchAnalysisError(
            f"{where} must contain exactly {', '.join(EXPECTED_ARCHITECTURES)}"
        )
    return entries


def _extract_training_models(
    summary: dict[str, Any],
) -> tuple[dict[str, list[TrainingModel]], dict[str, dict[str, Any]]]:
    if summary.get("schema_version") != TRAINING_SUMMARY_SCHEMA:
        raise AETBatchAnalysisError("unsupported training debt summary schema")
    if summary.get("status") != "complete":
        raise AETBatchAnalysisError("training debt campaign is not complete")

    entries = _architecture_entries(summary.get("architectures"), "training.architectures")
    models: dict[str, list[TrainingModel]] = {}
    debt_reports: dict[str, dict[str, Any]] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        entry = entries[architecture]
        raw_seeds = entry.get("seeds", entry.get("per_seed"))
        seed_entries = _list(raw_seeds, f"training.architectures.{architecture}.seeds")
        architecture_models: list[TrainingModel] = []
        for index, raw_seed in enumerate(seed_entries):
            seed_entry = _mapping(raw_seed, f"training.architectures.{architecture}.seeds[{index}]")
            seed = _positive_int(seed_entry.get("seed"), f"training {architecture} seed")
            energy_value = seed_entry.get("energy_j")
            if energy_value is None:
                energy = _mapping(
                    seed_entry.get("energy"), f"training {architecture} seed {seed}.energy"
                )
                energy_value = energy.get("observed_component_energy_j", energy.get("energy_j"))
            architecture_models.append(
                TrainingModel(
                    architecture=architecture,
                    seed=seed,
                    energy_j=_finite_float(
                        energy_value,
                        f"training {architecture} seed {seed} energy_j",
                        positive=True,
                    ),
                    checkpoint_sha256=_sha256(
                        seed_entry.get("checkpoint_sha256"),
                        f"training {architecture} seed {seed} checkpoint_sha256",
                    ),
                    model_state_sha256=_sha256(
                        seed_entry.get("model_state_sha256"),
                        f"training {architecture} seed {seed} model_state_sha256",
                    ),
                    model_identity_sha256=_sha256(
                        seed_entry.get("model_identity_sha256"),
                        f"training {architecture} seed {seed} model_identity_sha256",
                    ),
                )
            )
        architecture_models.sort(key=lambda model: model.seed)
        if tuple(model.seed for model in architecture_models) != EXPECTED_SEEDS:
            raise AETBatchAnalysisError(
                f"training {architecture} must contain exactly seeds {EXPECTED_SEEDS}"
            )
        if len({model.checkpoint_sha256 for model in architecture_models}) != len(EXPECTED_SEEDS):
            raise AETBatchAnalysisError(f"training {architecture} checkpoint hashes are not unique")

        computed_mean = statistics.fmean(model.energy_j for model in architecture_models)
        declared_mean = _finite_float(
            entry.get("recipe_training_energy_j_mean"),
            f"training {architecture} recipe_training_energy_j_mean",
            positive=True,
        )
        if not math.isclose(computed_mean, declared_mean, rel_tol=1e-12, abs_tol=1e-9):
            raise AETBatchAnalysisError(
                f"training {architecture} declared mean does not match its five seeds"
            )

        selection = _mapping(entry.get("selection_debt"), f"training {architecture}.selection_debt")
        study = _mapping(entry.get("study_debt"), f"training {architecture}.study_debt")
        for label, debt in (("selection", selection), ("study", study)):
            status = debt.get("status")
            if status not in {"unknown", "diagnostic"}:
                raise AETBatchAnalysisError(
                    f"training {architecture} {label} debt must be unknown or diagnostic"
                )
            if debt.get("included_in_primary") is not False:
                raise AETBatchAnalysisError(
                    f"training {architecture} {label} debt may not enter the primary numerator"
                )
        frontier_eligible = entry.get("frontier_eligible")
        if not isinstance(frontier_eligible, bool):
            raise AETBatchAnalysisError(
                f"training {architecture} frontier_eligible must be boolean"
            )
        quality_status = _string(
            entry.get("quality_status"), f"training {architecture}.quality_status"
        ).lower()
        expected_quality_status = "feasible" if frontier_eligible else "infeasible"
        if quality_status != expected_quality_status:
            raise AETBatchAnalysisError(f"training {architecture} quality status is inconsistent")
        quality_gate = _mapping(entry.get("quality_gate"), f"training {architecture}.quality_gate")
        expected_gate_status = "quality_passed" if frontier_eligible else "quality_nonpass"
        if (
            quality_gate.get("passed") is not frontier_eligible
            or quality_gate.get("status") != expected_gate_status
            or entry.get("energy_frontier_measurement_required") is not True
        ):
            raise AETBatchAnalysisError(f"training {architecture} quality gate is inconsistent")
        normalized_quality_gate = {
            "passed": frontier_eligible,
            "status": expected_gate_status,
            "path": _string(quality_gate.get("path"), f"training {architecture}.quality_gate.path"),
            "sha256": _sha256(
                quality_gate.get("sha256"), f"training {architecture}.quality_gate.sha256"
            ),
        }
        models[architecture] = architecture_models
        debt_reports[architecture] = {
            "recipe_training_energy_j_mean": computed_mean,
            "recipe_training_energy_j_median": statistics.median(
                model.energy_j for model in architecture_models
            ),
            "recipe_training_energy_j_min": min(model.energy_j for model in architecture_models),
            "recipe_training_energy_j_max": max(model.energy_j for model in architecture_models),
            "replicate_seed_count": len(architecture_models),
            "deployment_training_runs_represented": 1,
            "selection_debt": selection,
            "study_debt": study,
            "quality_status": quality_status,
            "frontier_eligible": frontier_eligible,
            "quality_gate": normalized_quality_gate,
            "energy_frontier_measurement_required": True,
        }
    return models, debt_reports


def _model_binding_key(model: TrainingModel) -> tuple[str, int]:
    return model.architecture, model.seed


def _verify_frontier_binding(
    frontier: dict[str, Any],
    training_models: dict[str, list[TrainingModel]],
    training_reports: dict[str, dict[str, Any]],
    *,
    training_manifest_sha256: str,
) -> None:
    binding = _mapping(frontier.get("training_source"), "frontier.training_source")
    declared_manifest_sha = _sha256(
        binding.get("training_manifest_sha256"),
        "frontier.training_source.training_manifest_sha256",
    )
    if declared_manifest_sha != training_manifest_sha256:
        raise AETBatchAnalysisError("frontier is not bound to this training manifest")

    raw_models = _list(binding.get("models"), "frontier.training_source.models")
    bound: dict[tuple[str, int], tuple[str, str, str]] = {}
    for index, raw_model in enumerate(raw_models):
        model = _mapping(raw_model, f"frontier.training_source.models[{index}]")
        architecture = _string(
            model.get("architecture"),
            f"frontier.training_source.models[{index}].architecture",
        ).lower()
        seed = _positive_int(
            model.get("seed", model.get("training_seed")),
            f"frontier.training_source.models[{index}].seed",
        )
        key = (architecture, seed)
        if key in bound:
            raise AETBatchAnalysisError(f"duplicate frontier model binding for {key}")
        bound[key] = (
            _sha256(model.get("checkpoint_sha256"), f"frontier binding {key} checkpoint"),
            _sha256(model.get("model_state_sha256"), f"frontier binding {key} model state"),
            _sha256(model.get("model_identity_sha256"), f"frontier binding {key} identity"),
        )

    expected: dict[tuple[str, int], tuple[str, str, str]] = {}
    for models in training_models.values():
        for model in models:
            expected[_model_binding_key(model)] = (
                model.checkpoint_sha256,
                model.model_state_sha256,
                model.model_identity_sha256,
            )
    if bound != expected:
        missing = sorted(set(expected) - set(bound))
        extra = sorted(set(bound) - set(expected))
        mismatched = sorted(
            key for key in set(bound) & set(expected) if bound[key] != expected[key]
        )
        raise AETBatchAnalysisError(
            "frontier model binding mismatch: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )

    raw_quality = _mapping(
        binding.get("architecture_quality"),
        "frontier.training_source.architecture_quality",
    )
    if set(raw_quality) != set(EXPECTED_ARCHITECTURES):
        raise AETBatchAnalysisError(
            "frontier training source quality must contain exactly AM and GNN"
        )
    for architecture in EXPECTED_ARCHITECTURES:
        receipt = _mapping(
            raw_quality[architecture],
            f"frontier.training_source.architecture_quality.{architecture}",
        )
        expected_report = training_reports[architecture]
        normalized_receipt = {
            "quality_status": receipt.get("quality_status"),
            "frontier_eligible": receipt.get("frontier_eligible"),
            "quality_gate": receipt.get("quality_gate"),
            "energy_frontier_measurement_required": receipt.get(
                "energy_frontier_measurement_required"
            ),
        }
        expected_receipt = {
            key: expected_report[key]
            for key in (
                "quality_status",
                "frontier_eligible",
                "quality_gate",
                "energy_frontier_measurement_required",
            )
        }
        if normalized_receipt != expected_receipt:
            raise AETBatchAnalysisError(
                f"frontier training source quality mismatch for {architecture}"
            )


def descriptive_t_interval_95(values: list[float]) -> tuple[float, float]:
    """Return the predeclared descriptive 95 percent t interval for five pairs."""
    if len(values) != 5:
        raise AETBatchAnalysisError("each batch cell must contain exactly five paired deltas")
    if any(not math.isfinite(value) for value in values):
        raise AETBatchAnalysisError("paired deltas must be finite")
    mean = statistics.fmean(values)
    half_width = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return mean - half_width, mean + half_width


def classify_batch(
    *,
    recipe_training_energy_j_mean: float,
    paired_delta_j_per_instance: list[float],
    quality_status: Any,
) -> dict[str, Any]:
    """Classify one batch with quality and descriptive paired uncertainty."""
    training_mean = _finite_float(
        recipe_training_energy_j_mean,
        "recipe_training_energy_j_mean",
        positive=True,
    )
    normalized_quality = _string(quality_status, "quality_status").lower()
    if normalized_quality not in {"feasible", "infeasible", "unidentified"}:
        raise AETBatchAnalysisError(f"unknown quality status {quality_status!r}")

    values = [
        _finite_float(value, f"paired_delta_j_per_instance[{index}]")
        for index, value in enumerate(paired_delta_j_per_instance)
    ]
    if values:
        low, high = descriptive_t_interval_95(values)
        mean = statistics.fmean(values)
        median = statistics.median(values)
        minimum = min(values)
        maximum = max(values)
        interval: list[float] | None = [low, high]
        interval_crosses_zero: bool | None = low <= 0.0 <= high
        if low > 0.0:
            evidence_status = "descriptive_interval_strictly_positive"
        elif high < 0.0:
            evidence_status = "descriptive_interval_strictly_negative"
        else:
            evidence_status = "descriptive_interval_includes_zero"
    else:
        low = high = mean = None
        median = minimum = maximum = None
        interval = None
        interval_crosses_zero = None
        evidence_status = "not_available"

    status: str
    reason: str
    aet_instances: float | None = None
    sensitivity: list[float] | None = None
    hypothetical_training_debt_sensitivity: dict[str, Any] | None = None
    if normalized_quality == "unidentified":
        status = STATUS_UNIDENTIFIED
        reason = "quality_not_established"
    elif normalized_quality == "infeasible":
        status = STATUS_INFINITE
        reason = "quality_constraint_failed"
    elif not values:
        status = STATUS_UNIDENTIFIED
        reason = "paired_energy_measurement_absent"
    elif mean is not None and mean > 0.0:
        status = STATUS_FINITE
        reason = "quality_feasible_and_paired_delta_mean_strictly_positive"
        aet_instances = training_mean / mean
        if low is not None and high is not None and low > 0.0:
            sensitivity = [training_mean / high, training_mean / low]
        hypothetical_training_debt_sensitivity = {
            "status": "hypothetical_not_measured",
            "included_in_primary": False,
            "primary_multiplier": 1,
            "interpretation": (
                "counterfactual numerator scaling only; no selection or discarded-run energy "
                "was measured"
            ),
            "values": [
                {
                    "training_debt_multiplier": multiplier,
                    "aet_instances": aet_instances * multiplier,
                }
                for multiplier in (1, 10, 100)
            ],
        }
    else:
        status = STATUS_INFINITE
        reason = "paired_delta_mean_nonpositive"

    return {
        "aet_status": status,
        "aet_reason": reason,
        "aet_instances": aet_instances,
        "aet_descriptive_sensitivity_instances": sensitivity,
        "hypothetical_training_debt_multiplier_sensitivity": (
            hypothetical_training_debt_sensitivity
        ),
        "quality_status": normalized_quality,
        "recipe_training_energy_j_mean": training_mean,
        "paired_delta_hgs_minus_neural_j_per_instance_mean": mean,
        "paired_delta_hgs_minus_neural_j_per_instance_median": median,
        "paired_delta_hgs_minus_neural_j_per_instance_min": minimum,
        "paired_delta_hgs_minus_neural_j_per_instance_max": maximum,
        "paired_delta_descriptive_t_interval_95_j_per_instance": interval,
        "interval_crosses_zero": interval_crosses_zero,
        "evidence_status": evidence_status,
        "paired_rounds": len(values),
        "uncertainty_role": "descriptive_pilot_only",
        "inferential_use": False,
    }


def _extract_cell_deltas(cell: dict[str, Any], where: str) -> list[float]:
    raw_pairs = _list(cell.get("pairs", []), f"{where}.pairs")
    deltas: list[float] = []
    seen_rounds: set[int] = set()
    for index, raw_pair in enumerate(raw_pairs):
        pair = _mapping(raw_pair, f"{where}.pairs[{index}]")
        round_value = pair.get("round", index)
        if isinstance(round_value, bool) or not isinstance(round_value, int) or round_value < 0:
            raise AETBatchAnalysisError(f"{where}.pairs[{index}].round must be non-negative")
        if round_value in seen_rounds:
            raise AETBatchAnalysisError(f"{where} contains duplicate round {round_value}")
        seen_rounds.add(round_value)
        value = pair.get("delta_hgs_minus_neural_j_per_instance")
        if value is None:
            value = pair.get("conservative_delta_hgs_minus_neural_j_per_instance")
        deltas.append(_finite_float(value, f"{where}.pairs[{index}].delta"))
    return deltas


def _cell_quality_gate(cell: dict[str, Any], where: str) -> tuple[str, dict[str, Any]]:
    status = _string(cell.get("quality_status"), f"{where}.quality_status").lower()
    if status not in {"feasible", "infeasible"}:
        raise AETBatchAnalysisError(f"{where}.quality_status must be feasible or infeasible")
    eligible = cell.get("frontier_eligible")
    expected_eligible = status == "feasible"
    if eligible is not expected_eligible:
        raise AETBatchAnalysisError(f"{where}.frontier_eligible is inconsistent")
    gate = _mapping(cell.get("quality_gate"), f"{where}.quality_gate")
    expected_gate_status = "quality_passed" if expected_eligible else "quality_nonpass"
    summary = _mapping(gate.get("summary"), f"{where}.quality_gate.summary")
    if (
        gate.get("passed") is not expected_eligible
        or gate.get("status") != expected_gate_status
        or summary.get("passed") is not expected_eligible
    ):
        raise AETBatchAnalysisError(f"{where}.quality_gate is inconsistent")
    _finite_float(gate.get("threshold_pct"), f"{where}.quality_gate.threshold_pct", positive=True)
    return status, gate


def build_analysis(
    training_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    *,
    training_manifest_sha256: str,
    frontier_manifest_sha256: str,
) -> dict[str, Any]:
    """Validate both completed campaigns and build the architecture tables."""
    training_models, debt_reports = _extract_training_models(training_summary)
    if frontier_summary.get("schema_version") != FRONTIER_SUMMARY_SCHEMA:
        raise AETBatchAnalysisError("unsupported batch frontier summary schema")
    if frontier_summary.get("status") != "complete":
        raise AETBatchAnalysisError("batch frontier campaign is not complete")
    _verify_frontier_binding(
        frontier_summary,
        training_models,
        debt_reports,
        training_manifest_sha256=training_manifest_sha256,
    )

    frontier_entries = _architecture_entries(
        frontier_summary.get("architectures"), "frontier.architectures"
    )
    architecture_results: list[dict[str, Any]] = []
    all_batch_sizes: set[int] = set()
    for architecture in EXPECTED_ARCHITECTURES:
        entry = frontier_entries[architecture]
        expected_quality = debt_reports[architecture]
        entry_quality = {
            "quality_status": entry.get("quality_status"),
            "frontier_eligible": entry.get("frontier_eligible"),
            "quality_gate": entry.get("quality_gate"),
        }
        expected_entry_quality = {
            key: expected_quality[key]
            for key in ("quality_status", "frontier_eligible", "quality_gate")
        }
        if entry_quality != expected_entry_quality:
            raise AETBatchAnalysisError(
                f"frontier {architecture} quality receipt does not match training"
            )
        raw_cells = entry.get("batches", entry.get("cells"))
        cells = _list(raw_cells, f"frontier.architectures.{architecture}.batches")
        rows: list[dict[str, Any]] = []
        seen_batches: set[int] = set()
        for index, raw_cell in enumerate(cells):
            where = f"frontier.architectures.{architecture}.batches[{index}]"
            cell = _mapping(raw_cell, where)
            batch_size = _positive_int(cell.get("batch_size"), f"{where}.batch_size")
            if batch_size in seen_batches:
                raise AETBatchAnalysisError(
                    f"frontier {architecture} contains duplicate batch {batch_size}"
                )
            seen_batches.add(batch_size)
            all_batch_sizes.add(batch_size)
            cell_quality_status, cell_quality_gate = _cell_quality_gate(cell, where)
            classified = classify_batch(
                recipe_training_energy_j_mean=debt_reports[architecture][
                    "recipe_training_energy_j_mean"
                ],
                paired_delta_j_per_instance=_extract_cell_deltas(cell, where),
                quality_status=cell_quality_status,
            )
            rows.append(
                {
                    "architecture": architecture,
                    "batch_size": batch_size,
                    "quality_gate": cell_quality_gate,
                    "training_quality_status": expected_quality["quality_status"],
                    **classified,
                }
            )
        if not rows:
            raise AETBatchAnalysisError(f"frontier {architecture} contains no measured batches")
        rows.sort(key=lambda row: row["batch_size"])
        architecture_results.append(
            {
                "architecture": architecture,
                "training_debt": debt_reports[architecture],
                "batches": rows,
            }
        )

    comparison: list[dict[str, Any]] = []
    indexed = {
        result["architecture"]: {row["batch_size"]: row for row in result["batches"]}
        for result in architecture_results
    }
    for batch_size in sorted(all_batch_sizes):
        row: dict[str, Any] = {"batch_size": batch_size}
        finite_values: dict[str, float] = {}
        for architecture in EXPECTED_ARCHITECTURES:
            item = indexed[architecture].get(batch_size)
            row[f"{architecture}_measured"] = item is not None
            row[f"{architecture}_aet_status"] = item["aet_status"] if item else None
            row[f"{architecture}_aet_instances"] = item["aet_instances"] if item else None
            if item is not None and item["aet_status"] == STATUS_FINITE:
                finite_values[architecture] = item["aet_instances"]
        if len(finite_values) == len(EXPECTED_ARCHITECTURES):
            best = min(finite_values, key=finite_values.__getitem__)
            row["lower_finite_aet_architecture"] = best
            row["finite_aet_ratio_gnn_over_am"] = finite_values["gnn"] / finite_values["am"]
        else:
            row["lower_finite_aet_architecture"] = None
            row["finite_aet_ratio_gnn_over_am"] = None
        comparison.append(row)

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "metric": {
            "name": "batch_indexed_amortized_efficiency_threshold",
            "formula": "mean_recipe_training_energy_j / mean_paired_(hgs10_minus_neural)_j_per_instance",
            "unit": "instances",
            "baseline": "fixed_hgs10",
            "training_numerator": "mean_one_recipe_run_across_five_seeds",
            "replicate_energies_are_not_summed": True,
        },
        "scope": {
            "energy": "native_windows_observed_cpu_package_plus_gpu_component_counters",
            "whole_system_energy": False,
            "carbon_accounting": "none",
            "selection_debt_imputed": False,
            "study_debt_imputed": False,
            "uncertainty": "five_paired_rounds_descriptive_only",
        },
        "inputs": {
            "training_manifest_sha256": training_manifest_sha256,
            "frontier_manifest_sha256": frontier_manifest_sha256,
        },
        "architectures": architecture_results,
        "architecture_comparison": comparison,
    }


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_csv(path: Path, analysis: dict[str, Any]) -> None:
    rows = [row for architecture in analysis["architectures"] for row in architecture["batches"]]
    fields = [
        "architecture",
        "batch_size",
        "quality_status",
        "aet_status",
        "aet_reason",
        "aet_instances",
        "evidence_status",
        "interval_crosses_zero",
        "recipe_training_energy_j_mean",
        "paired_delta_hgs_minus_neural_j_per_instance_mean",
        "paired_rounds",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _completion(analysis: dict[str, Any], output_root: Path) -> dict[str, Any]:
    architectures: list[dict[str, Any]] = []
    for entry in analysis["architectures"]:
        finite = [
            row["aet_instances"] for row in entry["batches"] if row["aet_status"] == STATUS_FINITE
        ]
        architectures.append(
            {
                "architecture": entry["architecture"],
                "training_seed_count": entry["training_debt"]["replicate_seed_count"],
                "measured_batch_count": len(entry["batches"]),
                "finite_aet_count": len(finite),
                "infinite_aet_count": sum(
                    row["aet_status"] == STATUS_INFINITE for row in entry["batches"]
                ),
                "unidentified_aet_count": sum(
                    row["aet_status"] == STATUS_UNIDENTIFIED for row in entry["batches"]
                ),
                "minimum_finite_aet_instances": min(finite) if finite else None,
            }
        )
    return {
        "schema_version": COMPLETION_SCHEMA,
        "status": "complete",
        "baseline": "fixed_hgs10",
        "architectures": architectures,
        "output_root": _display_path(output_root),
        "whole_system_energy": False,
        "carbon_accounting": "none",
    }


def _verify_input_manifest(root: Path, *, required_paths: tuple[str, ...]) -> tuple[Path, str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise AETBatchAnalysisError(f"missing completed manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    if not str(manifest.get("status", "")).startswith("complete"):
        raise AETBatchAnalysisError(f"input manifest is not complete: {manifest_path}")
    checksums_path = root / "SHA256SUMS"
    try:
        lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AETBatchAnalysisError(f"cannot read input checksums {checksums_path}: {exc}") from exc
    checked: set[str] = set()
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise AETBatchAnalysisError(f"malformed input checksum line in {checksums_path}")
        expected, relative_name = parts
        _sha256(expected, f"input checksum for {relative_name}")
        if (
            not relative_name
            or "\\" in relative_name
            or relative_name in checked
            or Path(relative_name).is_absolute()
            or any(part in {"", ".", ".."} for part in Path(relative_name).parts)
        ):
            raise AETBatchAnalysisError(f"unsafe input checksum path {relative_name!r}")
        path = (root / relative_name).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AETBatchAnalysisError(
                f"input checksum path escapes its bundle: {relative_name!r}"
            ) from exc
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise AETBatchAnalysisError(f"input checksum mismatch for {relative_name}")
        checked.add(relative_name)
    if "manifest.json" not in checked:
        raise AETBatchAnalysisError(f"input checksums do not bind manifest.json: {root}")
    missing_required = sorted(set(required_paths) - checked)
    if missing_required:
        raise AETBatchAnalysisError(
            f"input checksums do not bind required artifacts: {missing_required}"
        )
    return manifest_path, _sha256_file(manifest_path)


def _verify_output_checksums(root: Path) -> None:
    checksums_path = root / "SHA256SUMS"
    try:
        lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AETBatchAnalysisError(f"cannot read {checksums_path}: {exc}") from exc
    if not lines:
        raise AETBatchAnalysisError("analysis SHA256SUMS is empty")
    seen: set[str] = set()
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise AETBatchAnalysisError("analysis SHA256SUMS has a malformed line")
        expected, name = parts
        _sha256(expected, f"SHA256SUMS digest for {name}")
        if not name or name in seen or Path(name).name != name:
            raise AETBatchAnalysisError("analysis SHA256SUMS contains an unsafe or duplicate name")
        seen.add(name)
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise AETBatchAnalysisError(f"analysis checksum mismatch for {name}")


def execute_analysis(training_root: Path, frontier_root: Path, output_root: Path) -> dict[str, Any]:
    """Analyze completed inputs, then emit a checksummed immutable result bundle."""
    training_root = training_root.resolve()
    frontier_root = frontier_root.resolve()
    output_root = output_root.resolve()
    training_manifest_path, training_manifest_sha = _verify_input_manifest(
        training_root,
        required_paths=("training-energy-summary.json",),
    )
    frontier_manifest_path, frontier_manifest_sha = _verify_input_manifest(
        frontier_root,
        required_paths=("batch-frontier-summary.json",),
    )
    training_summary_path = training_root / "training-energy-summary.json"
    frontier_summary_path = frontier_root / "batch-frontier-summary.json"
    training_summary = _read_json(training_summary_path)
    frontier_summary = _read_json(frontier_summary_path)

    analysis = build_analysis(
        training_summary,
        frontier_summary,
        training_manifest_sha256=training_manifest_sha,
        frontier_manifest_sha256=frontier_manifest_sha,
    )
    input_identity = _sha256_json(
        {
            "training_manifest_sha256": training_manifest_sha,
            "training_summary_sha256": _sha256_file(training_summary_path),
            "frontier_manifest_sha256": frontier_manifest_sha,
            "frontier_summary_sha256": _sha256_file(frontier_summary_path),
        }
    )

    if output_root.exists():
        existing_manifest_path = output_root / "manifest.json"
        existing_analysis_path = output_root / "aet-batch-analysis.json"
        existing_checksums_path = output_root / "SHA256SUMS"
        existing_completion_path = output_root / "completion.json"
        if not all(
            path.is_file()
            for path in (
                existing_manifest_path,
                existing_analysis_path,
                existing_completion_path,
                existing_checksums_path,
            )
        ):
            raise AETBatchAnalysisError(
                f"refusing incomplete pre-existing analysis directory: {output_root}"
            )
        existing_manifest = _read_json(existing_manifest_path)
        if existing_manifest.get("input_identity_sha256") != input_identity:
            raise AETBatchAnalysisError(
                "analysis output already exists for different input manifests"
            )
        _verify_output_checksums(output_root)
        return _read_json(existing_analysis_path)

    output_root.mkdir(parents=True)
    analysis_path = output_root / "aet-batch-analysis.json"
    table_path = output_root / "aet-batch-table.csv"
    comparison_path = output_root / "architecture-comparison.json"
    completion_path = output_root / "completion.json"
    _write_json(analysis_path, analysis)
    _write_csv(table_path, analysis)
    _write_json(comparison_path, analysis["architecture_comparison"])
    _write_json(completion_path, _completion(analysis, output_root))

    output_hashes = {
        path.name: _sha256_file(path)
        for path in (analysis_path, table_path, comparison_path, completion_path)
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "input_identity_sha256": input_identity,
        "inputs": {
            "training_manifest": _display_path(training_manifest_path),
            "training_manifest_sha256": training_manifest_sha,
            "training_summary_sha256": _sha256_file(training_summary_path),
            "frontier_manifest": _display_path(frontier_manifest_path),
            "frontier_manifest_sha256": frontier_manifest_sha,
            "frontier_summary_sha256": _sha256_file(frontier_summary_path),
        },
        "outputs": output_hashes,
        "classification": {
            "whole_system_energy": False,
            "carbon_accounting": "none",
            "selection_debt_imputed": False,
            "study_debt_imputed": False,
        },
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    checksummed = [analysis_path, table_path, comparison_path, completion_path, manifest_path]
    checksum_lines = [f"{_sha256_file(path)}  {path.name}" for path in checksummed]
    (output_root / "SHA256SUMS").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8", newline="\n"
    )
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--frontier-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis = execute_analysis(args.training_root, args.frontier_root, args.output_root)
    except (AETBatchAnalysisError, OSError) as exc:
        print(f"AET batch analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_completion(analysis, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
