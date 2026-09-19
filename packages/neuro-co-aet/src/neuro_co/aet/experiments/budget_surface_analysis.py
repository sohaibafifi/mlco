"""Build the AM/GNN batch by HGS-budget AET sensitivity surface.

This module performs no solver execution and no energy measurement. It joins
three completed, checksummed bundles: measured training debt, the neural batch
frontier, and the HGS budget-sensitivity campaign. Training and neural
inference are resampled together by model seed. HGS round bundles are
resampled independently, with one HGS draw shared by every budget.
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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from neuro_co.aet.experiments import batch_analysis as batch

ANALYSIS_SCHEMA = "aet-budget-surface-analysis/v1"
MANIFEST_SCHEMA = "aet-budget-surface-manifest/v1"
COMPLETION_SCHEMA = "aet-budget-surface-completion/v1"
FRONTIER_SCHEMA = "aet-batch-frontier-summary/v1"
EXPECTED_ARCHITECTURES = ("am", "gnn")
EXPECTED_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
EXPECTED_BUDGETS = (10, 30, 100, 300)
EXPECTED_SEEDS = (2, 3, 4, 5, 6)
EXPECTED_ROUNDS = (0, 1, 2, 3, 4)
EXPECTED_HGS_SEEDS = (50_101, 51_101, 52_101, 53_101, 54_101)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 3497
BOOTSTRAP_QUANTILES = (0.025, 0.5, 0.975)
TRAINING_DEBT_MULTIPLIERS = (1, 10, 100)


class BudgetSurfaceAnalysisError(RuntimeError):
    """Raised when an input bundle cannot support the sensitivity surface."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BudgetSurfaceAnalysisError(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise BudgetSurfaceAnalysisError(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise BudgetSurfaceAnalysisError(f"{where} must be a non-empty string")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise BudgetSurfaceAnalysisError(f"{where} must be boolean")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BudgetSurfaceAnalysisError(f"{where} must be an integer >= {minimum}")
    return value


def _finite(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BudgetSurfaceAnalysisError(f"{where} must be finite")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise BudgetSurfaceAnalysisError(f"{where} must be {qualifier}")
    return result


def _sha(value: Any, where: str) -> str:
    text = _string(value, where)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise BudgetSurfaceAnalysisError(f"{where} must be a lowercase SHA-256")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BudgetSurfaceAnalysisError(f"cannot read JSON {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(_read_json_value(path), str(path))


def _write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _verify_bundle(root: Path, required_paths: tuple[str, ...]) -> tuple[Path, str]:
    try:
        return batch._verify_input_manifest(root, required_paths=required_paths)
    except batch.AETBatchAnalysisError as exc:
        raise BudgetSurfaceAnalysisError(str(exc)) from exc


def _architecture_entries(value: Any, where: str) -> dict[str, dict[str, Any]]:
    try:
        return batch._architecture_entries(value, where)
    except batch.AETBatchAnalysisError as exc:
        raise BudgetSurfaceAnalysisError(str(exc)) from exc


def _training_evidence(
    training_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    *,
    training_manifest_sha256: str,
) -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    try:
        models, reports = batch._extract_training_models(training_summary)
        batch._verify_frontier_binding(
            frontier_summary,
            models,
            reports,
            training_manifest_sha256=training_manifest_sha256,
        )
    except batch.AETBatchAnalysisError as exc:
        raise BudgetSurfaceAnalysisError(str(exc)) from exc
    energies = {
        architecture: [model.energy_j for model in models[architecture]]
        for architecture in EXPECTED_ARCHITECTURES
    }
    return energies, reports


def _sensitivity_source_binding(
    summary: dict[str, Any],
    *,
    frontier_manifest_sha256: str,
    frontier_summary_sha256: str,
    frontier_checksums_sha256: str,
) -> dict[str, Any]:
    binding = _mapping(summary.get("source_binding"), "sensitivity.source_binding")
    if (
        _sha(
            binding.get("batch_frontier_manifest_sha256"),
            "sensitivity source frontier manifest",
        )
        != frontier_manifest_sha256
    ):
        raise BudgetSurfaceAnalysisError(
            "HGS sensitivity summary is not bound to this batch-frontier manifest"
        )
    if (
        _sha(
            binding.get("batch_frontier_summary_sha256"),
            "sensitivity source frontier summary",
        )
        != frontier_summary_sha256
    ):
        raise BudgetSurfaceAnalysisError(
            "HGS sensitivity summary is not bound to this batch-frontier summary"
        )
    if (
        _sha(
            binding.get("batch_frontier_checksums_sha256"),
            "sensitivity source frontier checksums",
        )
        != frontier_checksums_sha256
    ):
        raise BudgetSurfaceAnalysisError(
            "HGS sensitivity summary is not bound to this batch-frontier checksum inventory"
        )
    return binding


def _verify_summary_manifest(
    manifest: dict[str, Any],
    *,
    schema: str,
    summary_field: str,
    summary_sha256: str,
    where: str,
) -> None:
    if manifest.get("schema_version") != schema or manifest.get("status") != "complete":
        raise BudgetSurfaceAnalysisError(f"unsupported or incomplete {where} manifest")
    if _sha(manifest.get(summary_field), f"{where}.{summary_field}") != summary_sha256:
        raise BudgetSurfaceAnalysisError(f"{where} manifest does not bind its summary")


def _quality_status(value: Any, where: str) -> tuple[str, dict[str, Any]]:
    record = _mapping(value, where)
    status = _string(record.get("quality_status"), f"{where}.quality_status").lower()
    gate = _mapping(record.get("quality_gate"), f"{where}.quality_gate")
    if status not in {"feasible", "infeasible"}:
        raise BudgetSurfaceAnalysisError(f"{where}.quality_status must be feasible or infeasible")
    expected = status == "feasible"
    eligible = record.get("frontier_eligible", record.get("eligible", expected))
    if eligible is not expected:
        raise BudgetSurfaceAnalysisError(f"{where} quality eligibility is inconsistent")
    expected_gate_status = "quality_passed" if expected else "quality_nonpass"
    if gate.get("passed") is not expected or gate.get("status") != expected_gate_status:
        raise BudgetSurfaceAnalysisError(f"{where}.quality_gate.passed is inconsistent")
    return status, gate


def _round_energy(round_value: Any, where: str) -> tuple[int, int, float, float, float, float]:
    record = _mapping(round_value, where)
    round_index = _integer(record.get("round", record.get("round_index")), f"{where}.round")
    hgs_seed = _integer(record.get("hgs_seed"), f"{where}.hgs_seed", minimum=1)
    cpu = _finite(
        record.get("cpu_package_energy_j_per_instance"),
        f"{where}.cpu_package_energy_j_per_instance",
        positive=True,
    )
    observed = _finite(
        record.get("observed_component_energy_j_per_instance"),
        f"{where}.observed_component_energy_j_per_instance",
        positive=True,
    )
    gpu = _finite(
        record.get("gpu_energy_j_per_instance"),
        f"{where}.gpu_energy_j_per_instance",
        positive=True,
    )
    if not math.isclose(observed, cpu + gpu, rel_tol=1e-12, abs_tol=1e-12):
        raise BudgetSurfaceAnalysisError(f"{where} observed energy does not equal CPU plus GPU")
    throughput = _finite(
        record.get("throughput_instances_per_s"),
        f"{where}.throughput_instances_per_s",
        positive=True,
    )
    return round_index, hgs_seed, cpu, gpu, observed, throughput


def _hgs_evidence(
    sensitivity_summary: dict[str, Any],
) -> tuple[
    dict[int, list[float]],
    dict[int, list[float]],
    dict[int, dict[str, Any]],
    dict[int, list[dict[str, Any]]],
]:
    schema = _string(sensitivity_summary.get("schema_version"), "sensitivity.schema_version")
    if schema != "aet-hgs-budget-sensitivity-summary/v1":
        raise BudgetSurfaceAnalysisError("unsupported HGS budget-sensitivity summary schema")
    if sensitivity_summary.get("status") != "complete":
        raise BudgetSurfaceAnalysisError("HGS budget-sensitivity campaign is not complete")
    classification = _mapping(
        sensitivity_summary.get("classification"), "sensitivity.classification"
    )
    if (
        classification.get("purpose") != "software_exploratory"
        or classification.get("scientific_use") is not False
        or classification.get("confirmatory_eligible") is not False
        or classification.get("whole_system_energy") is not False
        or classification.get("carbon_accounting") != "none"
    ):
        raise BudgetSurfaceAnalysisError("HGS sensitivity classification changed")
    for flag in (
        "training_was_run",
        "checkpoint_was_loaded",
        "neural_inference_was_run",
        "capacity_probe_was_run",
        "reference_was_generated",
        "source_bundle_was_mutated",
    ):
        if sensitivity_summary.get(flag) is not False:
            raise BudgetSurfaceAnalysisError(f"sensitivity.{flag} must be false")
    raw_budgets = _mapping(sensitivity_summary.get("budgets"), "sensitivity.budgets")
    try:
        normalized = {
            int(key): _mapping(value, f"sensitivity.budgets.{key}")
            for key, value in raw_budgets.items()
        }
    except (TypeError, ValueError) as exc:
        raise BudgetSurfaceAnalysisError("sensitivity budget keys must be integers") from exc
    if set(normalized) != set(EXPECTED_BUDGETS):
        raise BudgetSurfaceAnalysisError(
            f"sensitivity budgets must be exactly {list(EXPECTED_BUDGETS)}"
        )
    cpu_energies: dict[int, list[float]] = {}
    observed_energies: dict[int, list[float]] = {}
    qualities: dict[int, dict[str, Any]] = {}
    rounds_by_budget: dict[int, list[dict[str, Any]]] = {}
    for budget in EXPECTED_BUDGETS:
        entry = normalized[budget]
        status, gate = _quality_status(entry, f"sensitivity.budgets.{budget}")
        rounds = _list(entry.get("rounds"), f"sensitivity.budgets.{budget}.rounds")
        indexed: dict[int, dict[str, Any]] = {}
        cpu_by_round: dict[int, float] = {}
        observed_by_round: dict[int, float] = {}
        for index, raw_round in enumerate(rounds):
            where = f"sensitivity.budgets.{budget}.rounds[{index}]"
            round_index, hgs_seed, cpu, gpu, observed, throughput = _round_energy(raw_round, where)
            if round_index in indexed:
                raise BudgetSurfaceAnalysisError(
                    f"sensitivity budget {budget} contains duplicate round {round_index}"
                )
            if round_index not in EXPECTED_ROUNDS or hgs_seed != EXPECTED_HGS_SEEDS[round_index]:
                raise BudgetSurfaceAnalysisError(
                    f"sensitivity budget {budget} round {round_index} has the wrong HGS seed"
                )
            indexed[round_index] = {
                "round": round_index,
                "hgs_seed": hgs_seed,
                "cpu_package_energy_j_per_instance": cpu,
                "gpu_energy_j_per_instance": gpu,
                "observed_component_energy_j_per_instance": observed,
                "throughput_instances_per_s": throughput,
            }
            cpu_by_round[round_index] = cpu
            observed_by_round[round_index] = observed
        if tuple(sorted(indexed)) != EXPECTED_ROUNDS:
            raise BudgetSurfaceAnalysisError(
                f"sensitivity budget {budget} must contain rounds {EXPECTED_ROUNDS}"
            )
        cpu_energies[budget] = [cpu_by_round[index] for index in EXPECTED_ROUNDS]
        observed_energies[budget] = [observed_by_round[index] for index in EXPECTED_ROUNDS]
        qualities[budget] = {
            "quality_status": status,
            "frontier_eligible": status == "feasible",
            "quality_gate": gate,
        }
        rounds_by_budget[budget] = [indexed[index] for index in EXPECTED_ROUNDS]
    return cpu_energies, observed_energies, qualities, rounds_by_budget


def _neural_cells(
    frontier_summary: dict[str, Any],
    training_reports: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[int, list[float]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[int, dict[str, Any]]],
]:
    if frontier_summary.get("schema_version") != FRONTIER_SCHEMA:
        raise BudgetSurfaceAnalysisError("unsupported batch-frontier summary schema")
    if frontier_summary.get("status") != "complete":
        raise BudgetSurfaceAnalysisError("batch-frontier campaign is not complete")
    architectures = _architecture_entries(
        frontier_summary.get("architectures"), "frontier.architectures"
    )
    energies: dict[str, dict[int, list[float]]] = {}
    qualities: dict[str, dict[int, dict[str, Any]]] = {}
    memory: dict[str, dict[int, dict[str, Any]]] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        entry = architectures[architecture]
        receipt = {
            key: entry.get(key) for key in ("quality_status", "frontier_eligible", "quality_gate")
        }
        expected_receipt = {
            key: training_reports[architecture][key]
            for key in ("quality_status", "frontier_eligible", "quality_gate")
        }
        if receipt != expected_receipt:
            raise BudgetSurfaceAnalysisError(
                f"frontier {architecture} quality receipt does not match training"
            )
        capacity = _mapping(entry.get("capacity"), f"frontier.{architecture}.capacity")
        records = _list(capacity.get("records"), f"frontier.{architecture}.capacity.records")
        memory_records: dict[int, dict[str, Any]] = {}
        for index, raw_record in enumerate(records):
            record = _mapping(raw_record, f"frontier.{architecture}.capacity.records[{index}]")
            batch_size = _integer(record.get("batch_size"), "capacity batch", minimum=1)
            peak = _integer(record.get("peak_memory_allocated_b"), "peak memory", minimum=1)
            total = _integer(record.get("device_total_memory_b"), "device memory", minimum=1)
            memory_records[batch_size] = {
                "memory_oversubscribed": peak > total,
                "peak_memory_allocated_b": peak,
                "device_total_memory_b": total,
            }
        raw_cells = _list(entry.get("batches"), f"frontier.{architecture}.batches")
        energy_cells: dict[int, list[float]] = {}
        quality_cells: dict[int, dict[str, Any]] = {}
        for cell_index, raw_cell in enumerate(raw_cells):
            where = f"frontier.{architecture}.batches[{cell_index}]"
            cell = _mapping(raw_cell, where)
            batch_size = _integer(cell.get("batch_size"), f"{where}.batch_size", minimum=1)
            if batch_size in energy_cells:
                raise BudgetSurfaceAnalysisError(
                    f"frontier {architecture} contains duplicate batch {batch_size}"
                )
            try:
                status, gate = batch._cell_quality_gate(cell, where)
            except batch.AETBatchAnalysisError as exc:
                raise BudgetSurfaceAnalysisError(str(exc)) from exc
            pairs = _list(cell.get("pairs"), f"{where}.pairs")
            indexed: dict[int, tuple[int, float]] = {}
            for pair_index, raw_pair in enumerate(pairs):
                pair = _mapping(raw_pair, f"{where}.pairs[{pair_index}]")
                round_index = _integer(pair.get("round"), f"{where}.pairs[{pair_index}].round")
                seed = _integer(
                    pair.get("training_seed"),
                    f"{where}.pairs[{pair_index}].training_seed",
                    minimum=1,
                )
                energy = _finite(
                    pair.get("neural_observed_components_j_per_instance"),
                    f"{where}.pairs[{pair_index}].neural energy",
                    positive=True,
                )
                if round_index in indexed:
                    raise BudgetSurfaceAnalysisError(f"{where} contains duplicate round")
                indexed[round_index] = (seed, energy)
            if tuple(sorted(indexed)) != EXPECTED_ROUNDS:
                raise BudgetSurfaceAnalysisError(f"{where} must contain five rounds")
            seeds = tuple(indexed[index][0] for index in EXPECTED_ROUNDS)
            if seeds != EXPECTED_SEEDS:
                raise BudgetSurfaceAnalysisError(
                    f"{where} must align rounds with training seeds {EXPECTED_SEEDS}"
                )
            energy_cells[batch_size] = [indexed[index][1] for index in EXPECTED_ROUNDS]
            quality_cells[batch_size] = {
                "quality_status": status,
                "frontier_eligible": status == "feasible",
                "quality_gate": gate,
            }
        if tuple(sorted(energy_cells)) != EXPECTED_BATCHES:
            raise BudgetSurfaceAnalysisError(
                f"frontier {architecture} must contain batches {EXPECTED_BATCHES}"
            )
        if set(memory_records) != set(EXPECTED_BATCHES):
            raise BudgetSurfaceAnalysisError(
                f"frontier {architecture} capacity records do not match the batch grid"
            )
        oversubscribed = {
            batch_size
            for batch_size, record in memory_records.items()
            if record["memory_oversubscribed"]
        }
        if oversubscribed != {512}:
            raise BudgetSurfaceAnalysisError(
                f"frontier {architecture} memory-oversubscribed batches must be exactly [512]"
            )
        energies[architecture] = energy_cells
        qualities[architecture] = quality_cells
        memory[architecture] = memory_records
    return energies, qualities, memory


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    result = np.quantile(values, BOOTSTRAP_QUANTILES)
    return {
        "q025": float(result[0]),
        "median": float(result[1]),
        "q975": float(result[2]),
    }


def _bootstrap_cell(
    training: Sequence[float],
    neural: Sequence[float],
    hgs: Sequence[float],
    *,
    model_draws: np.ndarray,
    hgs_draws: np.ndarray,
) -> dict[str, Any]:
    training_array = np.asarray(training, dtype=np.float64)
    neural_array = np.asarray(neural, dtype=np.float64)
    hgs_array = np.asarray(hgs, dtype=np.float64)
    if training_array.shape != (5,) or neural_array.shape != (5,) or hgs_array.shape != (5,):
        raise BudgetSurfaceAnalysisError("bootstrap inputs must each contain five observations")
    sampled_training = training_array[model_draws].mean(axis=1)
    sampled_neural = neural_array[model_draws].mean(axis=1)
    sampled_hgs = hgs_array[hgs_draws].mean(axis=1)
    denominators = sampled_hgs - sampled_neural
    positive = denominators > 0.0
    nonpositive_count = int((~positive).sum())
    finite_count = int(positive.sum())
    sensitivity: list[dict[str, Any]] = []
    for multiplier in TRAINING_DEBT_MULTIPLIERS:
        finite_aet = multiplier * sampled_training[positive] / denominators[positive]
        sensitivity.append(
            {
                "training_debt_multiplier": multiplier,
                "evidence_status": (
                    "primary_measured_recipe"
                    if multiplier == 1
                    else "hypothetical_unmeasured_multiplier"
                ),
                "included_in_primary": multiplier == 1,
                "conditional_finite_aet_quantiles_instances": _quantiles(finite_aet),
            }
        )
    return {
        "replicates": BOOTSTRAP_REPLICATES,
        "generator": "numpy-pcg64",
        "seed": BOOTSTRAP_SEED,
        "training_inference_model_seed_paired": True,
        "hgs_round_resampling_independent": True,
        "same_hgs_round_draw_retained_across_budgets": True,
        "cross_solver_measurements_paired_or_contemporaneous": False,
        "inferential_claim": False,
        "nonpositive_denominator_draws_preserved": True,
        "nonpositive_denominator_draw_count": nonpositive_count,
        "finite_denominator_draw_count": finite_count,
        "infinite_fraction": nonpositive_count / BOOTSTRAP_REPLICATES,
        "conditional_finite_fraction": finite_count / BOOTSTRAP_REPLICATES,
        "training_debt_sensitivity": sensitivity,
    }


def _cell_classification(
    training: Sequence[float],
    neural: Sequence[float],
    hgs_cpu: Sequence[float],
    hgs_observed: Sequence[float],
    *,
    source_gates_passed: bool,
    drift_passed: bool,
    neural_quality: dict[str, Any],
    hgs_quality: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    training_mean = statistics.fmean(training)
    neural_mean = statistics.fmean(neural)
    hgs_cpu_mean = statistics.fmean(hgs_cpu)
    hgs_observed_mean = statistics.fmean(hgs_observed)
    primary_denominator = hgs_cpu_mean - neural_mean
    diagnostic_same_host_delta = hgs_observed_mean - neural_mean
    quality_passed = (
        neural_quality.get("quality_status") == "feasible"
        and hgs_quality.get("quality_status") == "feasible"
    )
    cell_evaluable = source_gates_passed and drift_passed and quality_passed
    reasons: list[str] = []
    if not source_gates_passed:
        reasons.append("source_provenance_gate_failed")
    if not drift_passed:
        reasons.append("hgs10_drift_bridge_failed")
    if not quality_passed:
        reasons.append("combined_quality_failed")
    if reasons:
        status = "not_evaluable"
        reason = "+".join(reasons)
        aet = None
    elif primary_denominator <= 0.0:
        status = "infinite"
        reason = "mean_hgs_minus_neural_energy_nonpositive"
        aet = None
    else:
        status = "finite"
        reason = "drift_and_quality_passed_with_positive_mean_denominator"
        aet = training_mean / primary_denominator
    debt = []
    bootstrap_by_multiplier = {
        item["training_debt_multiplier"]: item for item in bootstrap["training_debt_sensitivity"]
    }
    for multiplier in TRAINING_DEBT_MULTIPLIERS:
        debt.append(
            {
                "training_debt_multiplier": multiplier,
                "evidence_status": (
                    "primary_measured_recipe"
                    if multiplier == 1
                    else "hypothetical_unmeasured_multiplier"
                ),
                "included_in_primary": multiplier == 1,
                "aet_instances": aet * multiplier if aet is not None else None,
                "conditional_finite_aet_quantiles_instances": (
                    bootstrap_by_multiplier[multiplier][
                        "conditional_finite_aet_quantiles_instances"
                    ]
                    if status != "not_evaluable"
                    else None
                ),
            }
        )
    return {
        "aet_status": status,
        "aet_reason": reason,
        "aet_instances": aet,
        "recipe_training_energy_j_mean": training_mean,
        "neural_observed_components_j_per_instance_mean": neural_mean,
        "hgs_cpu_package_j_per_instance_mean": hgs_cpu_mean,
        "hgs_observed_components_j_per_instance_mean": hgs_observed_mean,
        "primary_delta_hgs_cpu_minus_neural_observed_components_j_per_instance_mean": (
            primary_denominator
        ),
        "diagnostic_same_host_delta_hgs_observed_minus_neural_observed_components_j_per_instance_mean": (
            diagnostic_same_host_delta
        ),
        "aet_uses_primary_delta_only": True,
        "cell_evaluable": cell_evaluable,
        "combined_quality_passed": quality_passed,
        "training_debt_sensitivity": debt,
    }


def build_surface(
    training_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    sensitivity_summary: dict[str, Any],
    *,
    training_manifest_sha256: str,
    training_checksums_sha256: str,
    frontier_manifest_sha256: str,
    frontier_summary_sha256: str,
    frontier_checksums_sha256: str,
    sensitivity_manifest_sha256: str,
    sensitivity_checksums_sha256: str,
) -> dict[str, Any]:
    """Validate completed inputs and build the 80-cell sensitivity surface."""

    training, training_reports = _training_evidence(
        training_summary,
        frontier_summary,
        training_manifest_sha256=training_manifest_sha256,
    )
    source_binding = _sensitivity_source_binding(
        sensitivity_summary,
        frontier_manifest_sha256=frontier_manifest_sha256,
        frontier_summary_sha256=frontier_summary_sha256,
        frontier_checksums_sha256=frontier_checksums_sha256,
    )
    neural, neural_quality, memory = _neural_cells(frontier_summary, training_reports)
    hgs_cpu, hgs_observed, hgs_quality, hgs_rounds = _hgs_evidence(sensitivity_summary)
    combined_eligible = _boolean(
        sensitivity_summary.get("combined_surface_eligible"),
        "sensitivity.combined_surface_eligible",
    )
    drift = _mapping(
        sensitivity_summary.get("hgs10_drift_bridge"),
        "sensitivity.hgs10_drift_bridge",
    )
    drift_passed = drift.get("passed")
    if not isinstance(drift_passed, bool):
        raise BudgetSurfaceAnalysisError("HGS-10 drift bridge lacks a boolean result")
    source_gates_passed = _boolean(
        source_binding.get("source_gates_passed"),
        "sensitivity.source_binding.source_gates_passed",
    )
    all_hgs_quality_passed = all(
        hgs_quality[budget]["quality_status"] == "feasible" for budget in EXPECTED_BUDGETS
    )
    declared_all_quality = _boolean(
        sensitivity_summary.get("all_budget_quality_gates_passed"),
        "sensitivity.all_budget_quality_gates_passed",
    )
    if declared_all_quality is not all_hgs_quality_passed:
        raise BudgetSurfaceAnalysisError(
            "all-budget quality eligibility disagrees with individual HGS gates"
        )
    expected_combined = source_gates_passed and drift_passed and all_hgs_quality_passed
    if combined_eligible is not expected_combined:
        raise BudgetSurfaceAnalysisError(
            "combined-surface eligibility disagrees with its source, drift, and quality gates"
        )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    model_draws = rng.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5))
    hgs_draws = rng.integers(0, 5, size=(BOOTSTRAP_REPLICATES, 5))

    rows: list[dict[str, Any]] = []
    for architecture in EXPECTED_ARCHITECTURES:
        for batch_size in EXPECTED_BATCHES:
            for budget in EXPECTED_BUDGETS:
                bootstrap = _bootstrap_cell(
                    training[architecture],
                    neural[architecture][batch_size],
                    hgs_cpu[budget],
                    model_draws=model_draws,
                    hgs_draws=hgs_draws,
                )
                classification = _cell_classification(
                    training[architecture],
                    neural[architecture][batch_size],
                    hgs_cpu[budget],
                    hgs_observed[budget],
                    source_gates_passed=source_gates_passed,
                    drift_passed=drift_passed,
                    neural_quality=neural_quality[architecture][batch_size],
                    hgs_quality=hgs_quality[budget],
                    bootstrap=bootstrap,
                )
                bootstrap["aet_interpretation_status"] = (
                    "descriptive_energy_bootstrap"
                    if classification["aet_status"] != "not_evaluable"
                    else "energy_diagnostic_only_quality_or_drift_failed"
                )
                rows.append(
                    {
                        "architecture": architecture,
                        "batch_size": batch_size,
                        "hgs_budget": budget,
                        **memory[architecture][batch_size],
                        "neural_quality": neural_quality[architecture][batch_size],
                        "hgs_quality": hgs_quality[budget],
                        "combined_surface_eligible": combined_eligible,
                        "bootstrap": bootstrap,
                        **classification,
                    }
                )

    comparison: list[dict[str, Any]] = []
    indexed = {(row["architecture"], row["batch_size"], row["hgs_budget"]): row for row in rows}
    for batch_size in EXPECTED_BATCHES:
        for budget in EXPECTED_BUDGETS:
            am = indexed[("am", batch_size, budget)]
            gnn = indexed[("gnn", batch_size, budget)]
            finite = {
                architecture: row["aet_instances"]
                for architecture, row in (("am", am), ("gnn", gnn))
                if row["aet_status"] == "finite"
            }
            comparison.append(
                {
                    "batch_size": batch_size,
                    "hgs_budget": budget,
                    "am_aet_status": am["aet_status"],
                    "am_aet_instances": am["aet_instances"],
                    "gnn_aet_status": gnn["aet_status"],
                    "gnn_aet_instances": gnn["aet_instances"],
                    "lower_finite_aet_architecture": (
                        min(finite, key=finite.__getitem__) if len(finite) == 2 else None
                    ),
                    "finite_aet_ratio_gnn_over_am": (
                        finite["gnn"] / finite["am"] if len(finite) == 2 else None
                    ),
                }
            )

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "metric": {
            "name": "batch_and_hgs_budget_indexed_amortized_efficiency_threshold",
            "formula": "mean_recipe_training_energy_j / mean_(hgs_cpu_minus_neural_cpu_gpu)_j_per_instance",
            "unit": "instances",
            "training_numerator": "one_recipe_run_resampled_by_model_seed",
            "replicate_energies_are_not_summed": True,
            "primary_denominator": (
                "HGS CPU-package energy minus neural observed CPU+GPU component energy"
            ),
            "same_host_diagnostic_not_used_for_aet": (
                "HGS observed CPU+idle-GPU component energy minus neural observed "
                "CPU+GPU component energy"
            ),
        },
        "scope": {
            "purpose": "software_exploratory",
            "scientific_use": False,
            "confirmatory_eligible": False,
            "whole_system_energy": False,
            "carbon_accounting": "none",
            "cross_solver_measurements_contemporaneous": False,
            "inferential_claim": False,
            "training_inference_model_seed_pairing": True,
            "hgs_round_resampling": "independent_round_bundles_shared_across_budgets",
            "hgs10_drift_bridge": drift,
            "source_gates_passed": source_gates_passed,
            "all_budget_quality_gates_passed": all_hgs_quality_passed,
            "combined_surface_eligible": combined_eligible,
        },
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "generator": "numpy-pcg64",
            "seed": BOOTSTRAP_SEED,
            "conditional_quantiles": list(BOOTSTRAP_QUANTILES),
            "nonpositive_denominator_mass_preserved": True,
        },
        "inputs": {
            "training_manifest_sha256": training_manifest_sha256,
            "training_checksums_sha256": training_checksums_sha256,
            "frontier_manifest_sha256": frontier_manifest_sha256,
            "frontier_summary_sha256": frontier_summary_sha256,
            "frontier_checksums_sha256": frontier_checksums_sha256,
            "sensitivity_manifest_sha256": sensitivity_manifest_sha256,
            "sensitivity_checksums_sha256": sensitivity_checksums_sha256,
        },
        "training_debt": training_reports,
        "hgs_budgets": {
            str(budget): {
                **hgs_quality[budget],
                "rounds": hgs_rounds[budget],
            }
            for budget in EXPECTED_BUDGETS
        },
        "rows": rows,
        "architecture_comparison": comparison,
    }


def _write_csv(path: Path, analysis: dict[str, Any]) -> None:
    fields = [
        "architecture",
        "batch_size",
        "hgs_budget",
        "memory_oversubscribed",
        "cell_evaluable",
        "combined_surface_eligible",
        "neural_quality_status",
        "hgs_quality_status",
        "aet_status",
        "aet_reason",
        "aet_instances",
        "aet_x10_training_debt_instances",
        "aet_x100_training_debt_instances",
        "recipe_training_energy_j_mean",
        "neural_observed_components_j_per_instance_mean",
        "hgs_cpu_package_j_per_instance_mean",
        "hgs_observed_components_j_per_instance_mean",
        "primary_delta_hgs_cpu_minus_neural_observed_components_j_per_instance_mean",
        "diagnostic_same_host_delta_hgs_observed_minus_neural_observed_components_j_per_instance_mean",
        "aet_uses_primary_delta_only",
        "bootstrap_interpretation_status",
        "bootstrap_inferential_claim",
        "bootstrap_infinite_fraction",
        "bootstrap_conditional_q025_instances",
        "bootstrap_conditional_median_instances",
        "bootstrap_conditional_q975_instances",
        "bootstrap_x10_conditional_q025_instances",
        "bootstrap_x10_conditional_median_instances",
        "bootstrap_x10_conditional_q975_instances",
        "bootstrap_x100_conditional_q025_instances",
        "bootstrap_x100_conditional_median_instances",
        "bootstrap_x100_conditional_q975_instances",
    ]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for source in analysis["rows"]:
            point_debt = {
                item["training_debt_multiplier"]: item
                for item in source["training_debt_sensitivity"]
            }
            bootstrap_debt = {
                item["training_debt_multiplier"]: item["conditional_finite_aet_quantiles_instances"]
                for item in source["bootstrap"]["training_debt_sensitivity"]
            }
            x1 = bootstrap_debt[1]
            x10 = bootstrap_debt[10]
            x100 = bootstrap_debt[100]
            writer.writerow(
                {
                    **{key: source.get(key) for key in fields},
                    "aet_x10_training_debt_instances": point_debt[10]["aet_instances"],
                    "aet_x100_training_debt_instances": point_debt[100]["aet_instances"],
                    "neural_quality_status": source["neural_quality"]["quality_status"],
                    "hgs_quality_status": source["hgs_quality"]["quality_status"],
                    "bootstrap_interpretation_status": source["bootstrap"][
                        "aet_interpretation_status"
                    ],
                    "bootstrap_inferential_claim": source["bootstrap"]["inferential_claim"],
                    "bootstrap_infinite_fraction": source["bootstrap"]["infinite_fraction"],
                    "bootstrap_conditional_q025_instances": x1["q025"] if x1 else None,
                    "bootstrap_conditional_median_instances": x1["median"] if x1 else None,
                    "bootstrap_conditional_q975_instances": x1["q975"] if x1 else None,
                    "bootstrap_x10_conditional_q025_instances": x10["q025"] if x10 else None,
                    "bootstrap_x10_conditional_median_instances": (x10["median"] if x10 else None),
                    "bootstrap_x10_conditional_q975_instances": x10["q975"] if x10 else None,
                    "bootstrap_x100_conditional_q025_instances": x100["q025"] if x100 else None,
                    "bootstrap_x100_conditional_median_instances": (
                        x100["median"] if x100 else None
                    ),
                    "bootstrap_x100_conditional_q975_instances": x100["q975"] if x100 else None,
                }
            )
    temporary.replace(path)


def _completion(analysis: dict[str, Any], output_root: Path) -> dict[str, Any]:
    counts = {status: 0 for status in ("finite", "infinite", "not_evaluable")}
    for row in analysis["rows"]:
        counts[row["aet_status"]] += 1
    return {
        "schema_version": COMPLETION_SCHEMA,
        "status": "complete",
        "row_count": len(analysis["rows"]),
        "architectures": list(EXPECTED_ARCHITECTURES),
        "batch_sizes": list(EXPECTED_BATCHES),
        "hgs_budgets": list(EXPECTED_BUDGETS),
        "classification_counts": counts,
        "combined_surface_eligible": analysis["scope"]["combined_surface_eligible"],
        "output_root": _display_path(output_root),
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "scientific_use": False,
        "confirmatory_eligible": False,
    }


def _verify_output_checksums(root: Path) -> None:
    try:
        batch._verify_output_checksums(root)
    except batch.AETBatchAnalysisError as exc:
        raise BudgetSurfaceAnalysisError(str(exc)) from exc
    expected = {
        "aet-budget-surface.json",
        "aet-budget-table.csv",
        "architecture-comparison.json",
        "completion.json",
        "manifest.json",
    }
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != expected | {"SHA256SUMS"} or any(
        entry.is_symlink() or not entry.is_file() for entry in entries
    ):
        raise BudgetSurfaceAnalysisError("analysis output inventory is not exact")
    listed = {
        line.split("  ", 1)[1]
        for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    if listed != expected:
        raise BudgetSurfaceAnalysisError("analysis SHA256SUMS inventory is not exhaustive")


def _display_path(path: Path) -> str:
    return batch._display_path(path)


def execute_analysis(
    training_root: Path,
    frontier_root: Path,
    sensitivity_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Verify three immutable inputs and emit an immutable surface bundle."""

    training_root = training_root.resolve()
    frontier_root = frontier_root.resolve()
    sensitivity_root = sensitivity_root.resolve()
    output_root = output_root.resolve()
    training_manifest_path, training_manifest_sha = _verify_bundle(
        training_root, ("training-energy-summary.json",)
    )
    frontier_manifest_path, frontier_manifest_sha = _verify_bundle(
        frontier_root, ("batch-frontier-summary.json",)
    )
    sensitivity_manifest_path, sensitivity_manifest_sha = _verify_bundle(
        sensitivity_root, ("hgs-budget-sensitivity-summary.json",)
    )
    training_summary_path = training_root / "training-energy-summary.json"
    frontier_summary_path = frontier_root / "batch-frontier-summary.json"
    sensitivity_summary_path = sensitivity_root / "hgs-budget-sensitivity-summary.json"
    training_summary_sha = _sha256_file(training_summary_path)
    training_checksums_sha = _sha256_file(training_root / "SHA256SUMS")
    frontier_summary_sha = _sha256_file(frontier_summary_path)
    frontier_checksums_sha = _sha256_file(frontier_root / "SHA256SUMS")
    sensitivity_summary_sha = _sha256_file(sensitivity_summary_path)
    sensitivity_checksums_sha = _sha256_file(sensitivity_root / "SHA256SUMS")
    _verify_summary_manifest(
        _read_json(training_manifest_path),
        schema="aet-training-debt-manifest/v1",
        summary_field="training_energy_summary_sha256",
        summary_sha256=training_summary_sha,
        where="training",
    )
    _verify_summary_manifest(
        _read_json(frontier_manifest_path),
        schema="aet-batch-frontier-manifest/v1",
        summary_field="batch_frontier_summary_sha256",
        summary_sha256=frontier_summary_sha,
        where="frontier",
    )
    sensitivity_manifest = _read_json(sensitivity_manifest_path)
    _verify_summary_manifest(
        sensitivity_manifest,
        schema="aet-hgs-budget-sensitivity-manifest/v1",
        summary_field="summary_sha256",
        summary_sha256=sensitivity_summary_sha,
        where="sensitivity",
    )
    sensitivity_summary = _read_json(sensitivity_summary_path)
    if sensitivity_manifest.get("source_binding") != sensitivity_summary.get(
        "source_binding"
    ) or sensitivity_manifest.get("combined_surface_eligible") is not sensitivity_summary.get(
        "combined_surface_eligible"
    ):
        raise BudgetSurfaceAnalysisError(
            "sensitivity manifest disagrees with the sensitivity summary"
        )
    analysis = build_surface(
        _read_json(training_summary_path),
        _read_json(frontier_summary_path),
        sensitivity_summary,
        training_manifest_sha256=training_manifest_sha,
        training_checksums_sha256=training_checksums_sha,
        frontier_manifest_sha256=frontier_manifest_sha,
        frontier_summary_sha256=frontier_summary_sha,
        frontier_checksums_sha256=frontier_checksums_sha,
        sensitivity_manifest_sha256=sensitivity_manifest_sha,
        sensitivity_checksums_sha256=sensitivity_checksums_sha,
    )
    analysis_source_sha = _sha256_file(Path(__file__).resolve())
    analysis["inputs"]["analysis_source_sha256"] = analysis_source_sha
    input_identity = _sha256_json(
        {
            "analysis_source_sha256": analysis_source_sha,
            "training_manifest_sha256": training_manifest_sha,
            "training_checksums_sha256": training_checksums_sha,
            "training_summary_sha256": training_summary_sha,
            "frontier_manifest_sha256": frontier_manifest_sha,
            "frontier_checksums_sha256": frontier_checksums_sha,
            "frontier_summary_sha256": frontier_summary_sha,
            "sensitivity_manifest_sha256": sensitivity_manifest_sha,
            "sensitivity_checksums_sha256": sensitivity_checksums_sha,
            "sensitivity_summary_sha256": sensitivity_summary_sha,
        }
    )

    if output_root.exists():
        required = (
            output_root / "aet-budget-surface.json",
            output_root / "aet-budget-table.csv",
            output_root / "architecture-comparison.json",
            output_root / "completion.json",
            output_root / "manifest.json",
            output_root / "SHA256SUMS",
        )
        if not all(path.is_file() for path in required):
            raise BudgetSurfaceAnalysisError(
                f"refusing incomplete pre-existing analysis directory: {output_root}"
            )
        manifest = _read_json(output_root / "manifest.json")
        if (
            manifest.get("schema_version") != MANIFEST_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("analysis_source_sha256") != analysis_source_sha
            or manifest.get("input_identity_sha256") != input_identity
        ):
            raise BudgetSurfaceAnalysisError(
                "analysis output manifest is stale or belongs to different inputs"
            )
        _verify_output_checksums(output_root)
        expected_output_hashes = {
            path.name: _sha256_file(path)
            for path in (
                output_root / "aet-budget-surface.json",
                output_root / "aet-budget-table.csv",
                output_root / "architecture-comparison.json",
                output_root / "completion.json",
            )
        }
        if manifest.get("outputs") != expected_output_hashes:
            raise BudgetSurfaceAnalysisError(
                "analysis manifest output hashes do not match the cached artifacts"
            )
        stored_analysis = _read_json(output_root / "aet-budget-surface.json")
        stored_comparison = _read_json_value(output_root / "architecture-comparison.json")
        stored_completion = _read_json(output_root / "completion.json")
        expected_completion = _completion(analysis, output_root)
        temporary_table = output_root / f".expected-table.{os.getpid()}.tmp"
        try:
            _write_csv(temporary_table, analysis)
            table_matches = (
                temporary_table.read_bytes() == (output_root / "aet-budget-table.csv").read_bytes()
            )
        finally:
            temporary_table.unlink(missing_ok=True)
        if (
            stored_analysis != analysis
            or stored_comparison != analysis["architecture_comparison"]
            or stored_completion != expected_completion
            or not table_matches
        ):
            raise BudgetSurfaceAnalysisError(
                "cached analysis outputs differ from the current analysis implementation"
            )
        return stored_analysis

    output_root.mkdir(parents=True)
    analysis_path = output_root / "aet-budget-surface.json"
    table_path = output_root / "aet-budget-table.csv"
    comparison_path = output_root / "architecture-comparison.json"
    completion_path = output_root / "completion.json"
    _write_json(analysis_path, analysis)
    _write_csv(table_path, analysis)
    _write_json(comparison_path, analysis["architecture_comparison"])
    _write_json(completion_path, _completion(analysis, output_root))
    outputs = {
        path.name: _sha256_file(path)
        for path in (analysis_path, table_path, comparison_path, completion_path)
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "analysis_source_sha256": analysis_source_sha,
        "input_identity_sha256": input_identity,
        "inputs": {
            "training_manifest": _display_path(training_manifest_path),
            "training_manifest_sha256": training_manifest_sha,
            "training_checksums_sha256": training_checksums_sha,
            "training_summary_sha256": training_summary_sha,
            "frontier_manifest": _display_path(frontier_manifest_path),
            "frontier_manifest_sha256": frontier_manifest_sha,
            "frontier_checksums_sha256": frontier_checksums_sha,
            "frontier_summary_sha256": frontier_summary_sha,
            "sensitivity_manifest": _display_path(sensitivity_manifest_path),
            "sensitivity_manifest_sha256": sensitivity_manifest_sha,
            "sensitivity_checksums_sha256": sensitivity_checksums_sha,
            "sensitivity_summary_sha256": sensitivity_summary_sha,
        },
        "outputs": outputs,
        "classification": {
            "purpose": "software_exploratory",
            "scientific_use": False,
            "confirmatory_eligible": False,
            "whole_system_energy": False,
            "carbon_accounting": "none",
            "cross_solver_measurements_contemporaneous": False,
        },
    }
    manifest_path = output_root / "manifest.json"
    _write_json(manifest_path, manifest)
    checksummed = [analysis_path, table_path, comparison_path, completion_path, manifest_path]
    (output_root / "SHA256SUMS").write_text(
        "\n".join(f"{_sha256_file(path)}  {path.name}" for path in checksummed) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return analysis


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--frontier-root", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        analysis = execute_analysis(
            args.training_root,
            args.frontier_root,
            args.sensitivity_root,
            args.output_root,
        )
    except (BudgetSurfaceAnalysisError, OSError, ValueError) as exc:
        print(f"AET budget-surface analysis failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(_completion(analysis, args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
