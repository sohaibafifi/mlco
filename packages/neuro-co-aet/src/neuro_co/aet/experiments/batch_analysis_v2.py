"""Analyze the 100-epoch greedy-1x1 campaign against 16-worker HGS-10s.

The primary AET numerator is the mean measured predeployment debt for one
100-epoch model, including prospective checkpoint selection.  Five training
replications estimate that quantity and are not summed in the primary metric.
When the earlier 40-epoch campaign is supplied, its measured energy is reported
separately as study/development debt and never substituted into primary AET.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any, Final

ANALYSIS_SCHEMA: Final = "aet-batch-analysis-v2/v1"
MANIFEST_SCHEMA: Final = "aet-batch-analysis-v2-manifest/v1"
COMPLETION_SCHEMA: Final = "aet-batch-analysis-v2-completion/v1"
TRAINING_SCHEMA: Final = "aet-training-debt-v2-summary/v1"
FRONTIER_SCHEMA: Final = "aet-batch-frontier-v2-summary/v1"
PRIOR_TRAINING_SCHEMA: Final = "aet-training-debt-summary/v1"
EXPECTED_ARCHITECTURES: Final = ("am", "gnn")
EXPECTED_SEEDS: Final = (2, 3, 4, 5, 6)
T_CRITICAL_95_DF4: Final = 2.7764451051977987

STATUS_FINITE: Final = "finite"
STATUS_INFINITE: Final = "infinite"
STATUS_INFEASIBLE: Final = "infeasible"
OUTPUT_FILENAMES: Final = (
    "aet-batch-analysis-v2.json",
    "aet-batch-table-v2.csv",
    "architecture-comparison-v2.json",
    "completion.json",
    "manifest.json",
)


class AETBatchAnalysisV2Error(RuntimeError):
    """Raised when v2 results or provenance are incomplete or inconsistent."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AETBatchAnalysisV2Error(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AETBatchAnalysisV2Error(f"{path} must contain one JSON object")
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AETBatchAnalysisV2Error(f"{where} must be an object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise AETBatchAnalysisV2Error(f"{where} must be an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AETBatchAnalysisV2Error(f"{where} must be a non-empty string")
    return value.strip()


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise AETBatchAnalysisV2Error(f"{where} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AETBatchAnalysisV2Error(f"{where} must be a finite number") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive and finite" if positive else "finite"
        raise AETBatchAnalysisV2Error(f"{where} must be {qualifier}")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AETBatchAnalysisV2Error(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AETBatchAnalysisV2Error(f"{where} must be a non-negative integer")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _architectures(value: Any, where: str) -> dict[str, dict[str, Any]]:
    raw = _mapping(value, where)
    if set(raw) != set(EXPECTED_ARCHITECTURES):
        raise AETBatchAnalysisV2Error(f"{where} must contain exactly AM and GNN")
    return {name: _mapping(raw[name], f"{where}.{name}") for name in EXPECTED_ARCHITECTURES}


def _training_debt(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if summary.get("schema_version") != TRAINING_SCHEMA or summary.get("status") != "complete":
        raise AETBatchAnalysisV2Error("100-epoch training summary is not complete v2 data")
    checkpoint_entries = _list(summary.get("checkpoint_entries"), "training.checkpoint_entries")
    if len(checkpoint_entries) != len(EXPECTED_ARCHITECTURES) * len(EXPECTED_SEEDS):
        raise AETBatchAnalysisV2Error(
            "training summary must contain ten prospectively selected checkpoint entries"
        )
    attempt_audit = _mapping(summary.get("attempt_audit"), "training.attempt_audit")
    raw_known_excluded = _mapping(
        attempt_audit.get("known_excluded_component_energy_j_by_architecture"),
        "training.attempt_audit.known_excluded_component_energy_j_by_architecture",
    )
    if set(raw_known_excluded) != set(EXPECTED_ARCHITECTURES):
        raise AETBatchAnalysisV2Error(
            "training attempt audit must report known excluded energy for AM and GNN"
        )
    known_excluded = {
        architecture: _number(
            raw_known_excluded[architecture],
            f"training attempt audit known excluded {architecture} energy",
        )
        for architecture in EXPECTED_ARCHITECTURES
    }
    raw_known_counts = _mapping(
        attempt_audit.get("known_excluded_attempt_count_by_architecture"),
        "training.attempt_audit.known_excluded_attempt_count_by_architecture",
    )
    raw_unknown_counts = _mapping(
        attempt_audit.get("unknown_excluded_attempt_count_by_architecture"),
        "training.attempt_audit.unknown_excluded_attempt_count_by_architecture",
    )
    if set(raw_known_counts) != set(EXPECTED_ARCHITECTURES) or set(raw_unknown_counts) != set(
        EXPECTED_ARCHITECTURES
    ):
        raise AETBatchAnalysisV2Error(
            "training attempt audit counts must report exactly AM and GNN"
        )
    architectures = _architectures(summary.get("architectures"), "training.architectures")
    result: dict[str, dict[str, Any]] = {}
    for architecture, entry in architectures.items():
        raw_seeds = _list(entry.get("per_seed"), f"training.{architecture}.per_seed")
        debts: list[float] = []
        training_energies: list[float] = []
        selection_energies: list[float] = []
        selected_epochs: list[int] = []
        seeds: list[int] = []
        for index, raw_seed in enumerate(raw_seeds):
            row = _mapping(raw_seed, f"training.{architecture}.per_seed[{index}]")
            seed = _positive_int(
                row.get("training_seed", row.get("seed")),
                f"training.{architecture}.per_seed[{index}].training_seed",
            )
            training = _number(
                row.get("training_energy_j"),
                f"training.{architecture}.seed{seed}.training_energy_j",
                positive=True,
            )
            selection = _number(
                row.get("checkpoint_selection_energy_j"),
                f"training.{architecture}.seed{seed}.checkpoint_selection_energy_j",
            )
            debt = _number(
                row.get("predeployment_energy_debt_j"),
                f"training.{architecture}.seed{seed}.predeployment_energy_debt_j",
                positive=True,
            )
            if not math.isclose(debt, training + selection, rel_tol=1e-10, abs_tol=1e-6):
                raise AETBatchAnalysisV2Error(
                    f"training {architecture} seed {seed} debt does not equal training plus selection"
                )
            selected_epoch = _positive_int(
                row.get("selected_epoch"), f"training.{architecture}.seed{seed}.selected_epoch"
            )
            if not 90 <= selected_epoch <= 100:
                raise AETBatchAnalysisV2Error(
                    f"training {architecture} seed {seed} selected epoch is outside 90..100"
                )
            _number(
                row.get("selected_mean_gap_pct"),
                f"training.{architecture}.seed{seed}.selected_mean_gap_pct",
            )
            seeds.append(seed)
            training_energies.append(training)
            selection_energies.append(selection)
            debts.append(debt)
            selected_epochs.append(selected_epoch)
        if tuple(sorted(seeds)) != EXPECTED_SEEDS or len(set(seeds)) != len(EXPECTED_SEEDS):
            raise AETBatchAnalysisV2Error(
                f"training {architecture} must contain exactly seeds {EXPECTED_SEEDS}"
            )
        declared = _number(
            entry.get("recipe_training_energy_j_mean"),
            f"training.{architecture}.recipe_training_energy_j_mean",
            positive=True,
        )
        if not math.isclose(declared, statistics.fmean(debts), rel_tol=1e-10, abs_tol=1e-6):
            raise AETBatchAnalysisV2Error(
                f"training {architecture} primary debt mean does not match per-seed debt"
            )
        for field, values in (
            ("training_only_energy_j_mean", training_energies),
            ("checkpoint_selection_energy_j_mean", selection_energies),
        ):
            value = _number(entry.get(field), f"training.{architecture}.{field}")
            if not math.isclose(value, statistics.fmean(values), rel_tol=1e-10, abs_tol=1e-6):
                raise AETBatchAnalysisV2Error(
                    f"training {architecture} {field} does not match per-seed data"
                )
        if (
            entry.get("quality_status") != "not_evaluated_on_final_holdout"
            or entry.get("frontier_eligible") is not False
            or entry.get("deployment_holdout_required") is not True
        ):
            raise AETBatchAnalysisV2Error(
                f"training {architecture} must defer final quality to deployment"
            )
        completed_campaign_energy = sum(debts)
        result[architecture] = {
            "primary_one_model_recipe_debt_j": declared,
            "definition": "mean_100_epoch_training_plus_prospective_checkpoint_selection",
            "replication_count": 5,
            "replication_energies_are_not_summed": True,
            "training_only_energy_j_mean": statistics.fmean(training_energies),
            "checkpoint_selection_energy_j_mean": statistics.fmean(selection_energies),
            "current_v2_completed_replications_energy_j_sum": completed_campaign_energy,
            "current_v2_known_excluded_component_energy_j": known_excluded[architecture],
            "current_v2_known_excluded_attempt_count": _nonnegative_int(
                raw_known_counts[architecture],
                f"training attempt audit known excluded {architecture} count",
            ),
            "current_v2_unknown_excluded_attempt_count": _nonnegative_int(
                raw_unknown_counts[architecture],
                f"training attempt audit unknown excluded {architecture} count",
            ),
            "current_v2_measured_known_energy_j": (
                completed_campaign_energy + known_excluded[architecture]
            ),
            "selected_epochs": selected_epochs,
        }
    return result


def _prior_energy(summary: dict[str, Any]) -> dict[str, float]:
    if (
        summary.get("schema_version") != PRIOR_TRAINING_SCHEMA
        or summary.get("status") != "complete"
    ):
        raise AETBatchAnalysisV2Error("prior training summary is not complete v1 data")
    architectures = _architectures(summary.get("architectures"), "prior.architectures")
    result: dict[str, float] = {}
    for architecture, entry in architectures.items():
        rows = _list(entry.get("seeds", entry.get("per_seed")), f"prior.{architecture}.seeds")
        values: list[float] = []
        seeds: list[int] = []
        for index, raw in enumerate(rows):
            row = _mapping(raw, f"prior.{architecture}.seeds[{index}]")
            seeds.append(
                _positive_int(row.get("seed", row.get("training_seed")), "prior training seed")
            )
            value = row.get("energy_j")
            if value is None:
                energy = _mapping(row.get("energy"), "prior seed energy")
                value = energy.get("observed_component_energy_j", energy.get("energy_j"))
            values.append(_number(value, "prior seed energy_j", positive=True))
        if tuple(sorted(seeds)) != EXPECTED_SEEDS or len(set(seeds)) != 5:
            raise AETBatchAnalysisV2Error(
                f"prior {architecture} must contain exactly seeds {EXPECTED_SEEDS}"
            )
        result[architecture] = sum(values)
    return result


def _t_interval(values: list[float]) -> list[float]:
    if len(values) != 5:
        raise AETBatchAnalysisV2Error("each AET cell must contain exactly five paired rounds")
    mean = statistics.fmean(values)
    half = T_CRITICAL_95_DF4 * statistics.stdev(values) / math.sqrt(len(values))
    return [mean - half, mean + half]


def classify_aet_cell(
    *,
    training_debt_j: float,
    batch_size: int,
    pairs: list[dict[str, Any]],
    quality_passed: bool,
) -> dict[str, Any]:
    """Classify one fixed architecture and batch against fixed HGS-10s."""
    debt = _number(training_debt_j, "training_debt_j", positive=True)
    batch = _positive_int(batch_size, "batch_size")
    if len(pairs) != 5:
        raise AETBatchAnalysisV2Error("each AET cell must contain exactly five paired rounds")
    hgs_values: list[float] = []
    neural_values: list[float] = []
    deltas: list[float] = []
    seen_rounds: set[int] = set()
    for index, raw_pair in enumerate(pairs):
        pair = _mapping(raw_pair, f"pairs[{index}]")
        round_index = pair.get("round", index)
        if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
            raise AETBatchAnalysisV2Error(f"pairs[{index}].round must be non-negative")
        if round_index in seen_rounds:
            raise AETBatchAnalysisV2Error(f"duplicate paired round {round_index}")
        seen_rounds.add(round_index)
        hgs = _number(
            pair.get("hgs_cpu_package_j_per_instance"),
            f"pairs[{index}].hgs_cpu_package_j_per_instance",
            positive=True,
        )
        neural = _number(
            pair.get("neural_observed_components_j_per_instance"),
            f"pairs[{index}].neural_observed_components_j_per_instance",
            positive=True,
        )
        delta = _number(pair.get("delta_hgs_minus_neural_j_per_instance"), f"pairs[{index}].delta")
        if not math.isclose(delta, hgs - neural, rel_tol=1e-9, abs_tol=1e-8):
            raise AETBatchAnalysisV2Error(
                f"pairs[{index}] delta does not equal HGS minus neural energy"
            )
        hgs_values.append(hgs)
        neural_values.append(neural)
        deltas.append(delta)

    hgs_mean = statistics.fmean(hgs_values)
    neural_mean = statistics.fmean(neural_values)
    delta_mean = statistics.fmean(deltas)
    interval = _t_interval(deltas)
    if not quality_passed:
        status = STATUS_INFEASIBLE
        reason = "quality_constraint_failed"
        aet_instances = None
        minimum_instances_abstract = None
        minimum_full_batches = None
        minimum_instances_full_batches = None
        eligible = False
    elif delta_mean <= 0.0:
        status = STATUS_INFINITE
        reason = "hgs_marginal_energy_not_greater_than_neural"
        aet_instances = None
        minimum_instances_abstract = None
        minimum_full_batches = None
        minimum_instances_full_batches = None
        eligible = False
    else:
        status = STATUS_FINITE
        reason = "quality_passed_and_hgs_marginal_energy_greater_than_neural"
        aet_instances = debt / delta_mean
        minimum_instances_abstract = math.ceil(aet_instances)
        minimum_full_batches = math.ceil(debt / (delta_mean * batch))
        minimum_instances_full_batches = minimum_full_batches * batch
        eligible = True
    return {
        "aet_status": status,
        "aet_reason": reason,
        "aet_eligible": eligible,
        "aet_instances": aet_instances,
        "aet_instances_continuous": aet_instances,
        "minimum_instances_abstract_ceiling": minimum_instances_abstract,
        "minimum_full_batches_to_amortize": minimum_full_batches,
        "minimum_instances_at_full_batches": minimum_instances_full_batches,
        "quality_status": "feasible" if quality_passed else "infeasible",
        "batch_size": batch,
        "training_debt_j": debt,
        "hgs_j_per_instance_mean": hgs_mean,
        "neural_j_per_instance_mean": neural_mean,
        "hgs_j_per_batch_equivalent_mean": hgs_mean * batch,
        "neural_j_per_batch_mean": neural_mean * batch,
        "hgs_minus_neural_j_per_instance_mean": delta_mean,
        "paired_delta_descriptive_t_interval_95_j_per_instance": interval,
        "paired_rounds": len(deltas),
        "uncertainty_role": "descriptive_only",
    }


def _quality_passed(cell: dict[str, Any], where: str) -> tuple[bool, dict[str, Any]]:
    status = _string(cell.get("quality_status"), f"{where}.quality_status").lower()
    if status not in {"feasible", "infeasible"}:
        raise AETBatchAnalysisV2Error(f"{where}.quality_status must be feasible or infeasible")
    passed = status == "feasible"
    if cell.get("frontier_eligible") is not passed:
        raise AETBatchAnalysisV2Error(f"{where}.frontier_eligible is inconsistent")
    gate = _mapping(cell.get("quality_gate"), f"{where}.quality_gate")
    if gate.get("passed") is not passed:
        raise AETBatchAnalysisV2Error(f"{where}.quality_gate is inconsistent")
    return passed, gate


def build_analysis(
    training_summary: dict[str, Any],
    frontier_summary: dict[str, Any],
    *,
    training_manifest_sha256: str,
    frontier_manifest_sha256: str,
    prior_campaign_energy_j: dict[str, float] | None = None,
    prior_training_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the batch-indexed AET table after validating fixed policies."""
    debts = _training_debt(training_summary)
    if frontier_summary.get("schema_version") != FRONTIER_SCHEMA:
        raise AETBatchAnalysisV2Error("unsupported v2 frontier summary schema")
    if frontier_summary.get("status") != "complete":
        raise AETBatchAnalysisV2Error("v2 frontier campaign is not complete")
    source = _mapping(frontier_summary.get("training_source"), "frontier.training_source")
    if source.get("training_manifest_sha256") != training_manifest_sha256:
        raise AETBatchAnalysisV2Error("frontier is not bound to this v2 training manifest")

    neural_policy = _mapping(frontier_summary.get("neural_policy"), "frontier.neural_policy")
    if (
        neural_policy.get("mode_id") != "greedy-1x1"
        or neural_policy.get("n_starts") != 1
        or neural_policy.get("augmentations") != 1
    ):
        raise AETBatchAnalysisV2Error("frontier neural policy is not greedy 1x1")
    hgs = _mapping(frontier_summary.get("hgs_reference"), "frontier.hgs_reference")
    if (
        not math.isclose(_number(hgs.get("max_runtime_s"), "HGS max_runtime_s"), 10.0)
        or hgs.get("parallel_workers") != 16
        or hgs.get("cpu_threads_per_worker") != 1
        or hgs.get("parallel") is not True
        or hgs.get("per_instance_independent_limit") is not True
        or hgs.get("executor") != "windows_process_pool"
        or hgs.get("energy_normalization")
        != "aggregate_cpu_package_energy_for_whole_pool_divided_by_512_original_instances"
        or hgs.get("measurement_boundary")
        != "warm_process_pool_ready_before_tracker_through_all_512_results_returned"
        or hgs.get("pool_creation_included_in_energy") is not False
        or hgs.get("worker_imports_included_in_energy") is not False
        or hgs.get("warmup_included_in_energy") is not False
        or hgs.get("pool_shutdown_included_in_energy") is not False
        or hgs.get("warmup_real_solve_count_per_block") != 16
    ):
        raise AETBatchAnalysisV2Error(
            "frontier baseline is not the aggregate-energy 16-worker HGS-10s policy"
        )
    hgs_quality_gate = _mapping(hgs.get("quality_gate"), "frontier.hgs_reference.quality_gate")
    hgs_quality_passed = (
        hgs.get("quality_status") == "feasible"
        and hgs.get("frontier_eligible") is True
        and hgs_quality_gate.get("passed") is True
    )

    frontier_architectures = _architectures(
        frontier_summary.get("architectures"), "frontier.architectures"
    )
    results: list[dict[str, Any]] = []
    all_batches: set[int] = set()
    for architecture in EXPECTED_ARCHITECTURES:
        rows: list[dict[str, Any]] = []
        raw_cells = _list(
            frontier_architectures[architecture].get("batches"),
            f"frontier.{architecture}.batches",
        )
        seen: set[int] = set()
        for index, raw_cell in enumerate(raw_cells):
            where = f"frontier.{architecture}.batches[{index}]"
            cell = _mapping(raw_cell, where)
            batch = _positive_int(cell.get("batch_size"), f"{where}.batch_size")
            if batch in seen:
                raise AETBatchAnalysisV2Error(f"duplicate {architecture} batch {batch}")
            seen.add(batch)
            all_batches.add(batch)
            neural_passed, neural_gate = _quality_passed(cell, where)
            paired_quality_passed = neural_passed and hgs_quality_passed
            classified = classify_aet_cell(
                training_debt_j=debts[architecture]["primary_one_model_recipe_debt_j"],
                batch_size=batch,
                pairs=_list(cell.get("pairs"), f"{where}.pairs"),
                quality_passed=paired_quality_passed,
            )
            rows.append(
                {
                    "architecture": architecture,
                    "quality_gate": {
                        "passed": paired_quality_passed,
                        "neural": neural_gate,
                        "hgs": hgs_quality_gate,
                    },
                    "neural_quality_status": "feasible" if neural_passed else "infeasible",
                    "hgs_quality_status": "feasible" if hgs_quality_passed else "infeasible",
                    **classified,
                }
            )
        if not rows:
            raise AETBatchAnalysisV2Error(f"frontier {architecture} contains no batches")
        rows.sort(key=lambda row: row["batch_size"])
        variants: dict[str, Any] = {
            "primary_one_model_recipe": {
                "energy_j": debts[architecture]["primary_one_model_recipe_debt_j"],
                "included_in_primary_aet": True,
                "definition": debts[architecture]["definition"],
            },
            "current_v2_campaign": {
                "status": "measured_known_lower_bound",
                "energy_j": debts[architecture]["current_v2_measured_known_energy_j"],
                "completed_replications_energy_j": debts[architecture][
                    "current_v2_completed_replications_energy_j_sum"
                ],
                "known_excluded_component_energy_j": debts[architecture][
                    "current_v2_known_excluded_component_energy_j"
                ],
                "known_excluded_attempt_count": debts[architecture][
                    "current_v2_known_excluded_attempt_count"
                ],
                "unknown_excluded_attempt_count": debts[architecture][
                    "current_v2_unknown_excluded_attempt_count"
                ],
                "included_in_primary_aet": False,
                "definition": (
                    "sum_of_five_completed_100_epoch_training_and_selection_replications_"
                    "plus_known_measured_discarded_or_interrupted_v2_energy"
                ),
                "caveat": (
                    "lower bound; excludes failures or prototypes whose energy was not measured"
                ),
            },
        }
        if prior_campaign_energy_j is None:
            variants["measured_study_development"] = {
                "status": "prior_40_epoch_campaign_not_supplied",
                "energy_j": None,
                "included_in_primary_aet": False,
            }
        else:
            prior = prior_campaign_energy_j[architecture]
            variants["measured_study_development"] = {
                "status": "measured_known_lower_bound",
                "energy_j": prior + debts[architecture]["current_v2_measured_known_energy_j"],
                "prior_40_epoch_campaign_energy_j": prior,
                "current_v2_completed_replications_energy_j": debts[architecture][
                    "current_v2_completed_replications_energy_j_sum"
                ],
                "current_v2_known_excluded_component_energy_j": debts[architecture][
                    "current_v2_known_excluded_component_energy_j"
                ],
                "current_v2_known_excluded_attempt_count": debts[architecture][
                    "current_v2_known_excluded_attempt_count"
                ],
                "current_v2_unknown_excluded_attempt_count": debts[architecture][
                    "current_v2_unknown_excluded_attempt_count"
                ],
                "included_in_primary_aet": False,
                "definition": (
                    "completed_v1_runs_plus_completed_v2_replications_plus_known_measured_"
                    "discarded_or_interrupted_v2_energy"
                ),
                "caveat": (
                    "lower bound; unmeasured old prototypes and failures with unknown energy "
                    "cannot be reconstructed"
                ),
            }
        for row in rows:
            row["current_v2_campaign_measured_known_lower_bound_j"] = variants[
                "current_v2_campaign"
            ]["energy_j"]
            row["current_v2_completed_replications_energy_j"] = variants["current_v2_campaign"][
                "completed_replications_energy_j"
            ]
            row["current_v2_known_excluded_component_energy_j"] = variants["current_v2_campaign"][
                "known_excluded_component_energy_j"
            ]
            row["current_v2_known_excluded_attempt_count"] = variants["current_v2_campaign"][
                "known_excluded_attempt_count"
            ]
            row["current_v2_unknown_excluded_attempt_count"] = variants["current_v2_campaign"][
                "unknown_excluded_attempt_count"
            ]
            row["measured_study_development_status"] = variants["measured_study_development"][
                "status"
            ]
            row["measured_study_development_energy_j"] = variants["measured_study_development"][
                "energy_j"
            ]
        results.append(
            {
                "architecture": architecture,
                "training_debt": debts[architecture],
                "training_debt_variants": variants,
                "batches": rows,
            }
        )

    indexed = {
        result["architecture"]: {row["batch_size"]: row for row in result["batches"]}
        for result in results
    }
    comparison: list[dict[str, Any]] = []
    for batch in sorted(all_batches):
        row: dict[str, Any] = {"batch_size": batch}
        finite: dict[str, float] = {}
        for architecture in EXPECTED_ARCHITECTURES:
            item = indexed[architecture].get(batch)
            row[f"{architecture}_aet_status"] = item["aet_status"] if item else None
            row[f"{architecture}_aet_instances"] = item["aet_instances"] if item else None
            if item and item["aet_status"] == STATUS_FINITE:
                finite[architecture] = item["aet_instances"]
        row["lower_finite_aet_architecture"] = (
            min(finite, key=finite.__getitem__) if len(finite) == 2 else None
        )
        comparison.append(row)

    return {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "complete",
        "metric": {
            "formula": "mean_100_epoch_training_plus_selection_debt_j / mean_paired_(aggregate_hgs10s_pool_package_minus_neural)_j_per_instance",
            "unit": "instances",
            "baseline": "parallel_16_worker_hgs_10_seconds_per_instance",
            "baseline_energy_normalization": (
                "aggregate_cpu_package_energy_for_whole_pool_divided_by_512_original_instances"
            ),
            "baseline_measurement_protocol": {
                key: hgs[key]
                for key in (
                    "measurement_boundary",
                    "pool_creation_included_in_energy",
                    "worker_imports_included_in_energy",
                    "warmup_included_in_energy",
                    "pool_shutdown_included_in_energy",
                    "warmup_real_solve_count_per_block",
                )
            },
            "neural_policy": "greedy_1x1",
            "finite_requires_quality_pass": True,
            "finite_requires_hgs_marginal_energy_greater_than_neural": True,
            "realizable_threshold": (
                "ceil(training_debt_j / (mean_delta_j_per_instance * batch_size)) full batches"
            ),
        },
        "status_semantics": {
            "finite": "quality passes and HGS marginal energy is greater than neural",
            "infinite": "quality passes but HGS marginal energy is not greater than neural",
            "infeasible": "quality constraint fails regardless of energy",
        },
        "scope": {
            "energy": "native_windows_observed_cpu_package_plus_gpu_component_counters",
            "whole_system_energy": False,
            "carbon_accounting": "none",
            "study_development_debt": "measured_known_lower_bound",
            "unknown_excluded_energy_imputed": False,
            "historical_unmeasured_prototypes_reconstructed": False,
        },
        "inputs": {
            "training_manifest_sha256": training_manifest_sha256,
            "frontier_manifest_sha256": frontier_manifest_sha256,
            "prior_training_manifest_sha256": prior_training_manifest_sha256,
        },
        "architectures": results,
        "architecture_comparison": comparison,
    }


def _verify_bundle(root: Path, required: str) -> tuple[dict[str, Any], str]:
    manifest_path = root / "manifest.json"
    checksums_path = root / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise AETBatchAnalysisV2Error(f"incomplete input bundle: {root}")
    manifest = _read_json(manifest_path)
    if not str(manifest.get("status", "")).startswith("complete"):
        raise AETBatchAnalysisV2Error(f"input bundle is not complete: {root}")
    expected: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or Path(parts[1]).is_absolute() or ".." in Path(parts[1]).parts:
            raise AETBatchAnalysisV2Error(f"unsafe checksum line in {checksums_path}")
        expected[parts[1]] = parts[0]
    for name in ("manifest.json", required):
        path = root / name
        if name not in expected or not path.is_file() or _sha256_file(path) != expected[name]:
            raise AETBatchAnalysisV2Error(f"input checksum mismatch for {path}")
    return manifest, _sha256_file(manifest_path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_csv(path: Path, analysis: dict[str, Any]) -> None:
    fields = [
        "architecture",
        "batch_size",
        "quality_status",
        "aet_status",
        "aet_reason",
        "aet_eligible",
        "aet_instances",
        "aet_instances_continuous",
        "minimum_instances_abstract_ceiling",
        "minimum_full_batches_to_amortize",
        "minimum_instances_at_full_batches",
        "training_debt_j",
        "hgs_j_per_instance_mean",
        "neural_j_per_instance_mean",
        "hgs_j_per_batch_equivalent_mean",
        "neural_j_per_batch_mean",
        "hgs_minus_neural_j_per_instance_mean",
        "current_v2_campaign_measured_known_lower_bound_j",
        "current_v2_completed_replications_energy_j",
        "current_v2_known_excluded_component_energy_j",
        "current_v2_known_excluded_attempt_count",
        "current_v2_unknown_excluded_attempt_count",
        "measured_study_development_status",
        "measured_study_development_energy_j",
    ]
    rows = [row for architecture in analysis["architectures"] for row in architecture["batches"]]
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _staging_path(output_root: Path) -> Path:
    return output_root.parent / f".{output_root.name}.aet-batch-analysis-v2-staging"


def _recover_owned_staging(output_root: Path) -> Path:
    """Remove only the fixed sibling staging directory owned by this analysis."""
    staging = _staging_path(output_root)
    if staging.parent != output_root.parent or staging.name != (
        f".{output_root.name}.aet-batch-analysis-v2-staging"
    ):
        raise AETBatchAnalysisV2Error("unsafe analysis staging path")
    if staging.is_symlink():
        raise AETBatchAnalysisV2Error(f"refusing symlinked analysis staging path: {staging}")
    if staging.exists():
        if not staging.is_dir():
            raise AETBatchAnalysisV2Error(
                f"refusing non-directory analysis staging path: {staging}"
            )
        shutil.rmtree(staging)
    return staging


def _read_output_checksums(output_root: Path) -> dict[str, str]:
    checksums_path = output_root / "SHA256SUMS"
    try:
        lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AETBatchAnalysisV2Error(f"cannot read {checksums_path}: {exc}") from exc
    checksums: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise AETBatchAnalysisV2Error("analysis SHA256SUMS has a malformed line")
        digest, name = parts
        if (
            len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or Path(name).name != name
            or name in checksums
        ):
            raise AETBatchAnalysisV2Error("analysis SHA256SUMS has an unsafe entry")
        checksums[name] = digest
    if set(checksums) != set(OUTPUT_FILENAMES):
        raise AETBatchAnalysisV2Error("analysis SHA256SUMS does not bind the exact output set")
    return checksums


def _validate_completed_output(
    output_root: Path,
    *,
    input_identity: str,
    expected_inputs: dict[str, str | None],
) -> dict[str, Any]:
    if output_root.is_symlink() or not output_root.is_dir():
        raise AETBatchAnalysisV2Error(f"analysis output is not a safe directory: {output_root}")
    checksums = _read_output_checksums(output_root)
    for name, expected in checksums.items():
        path = output_root / name
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise AETBatchAnalysisV2Error(f"analysis checksum mismatch for {name}")
    manifest = _read_json(output_root / "manifest.json")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("input_identity_sha256") != input_identity
        or manifest.get("inputs") != expected_inputs
    ):
        raise AETBatchAnalysisV2Error("analysis manifest does not match the current inputs")
    expected_outputs = {
        name: checksums[name] for name in OUTPUT_FILENAMES if name != "manifest.json"
    }
    if manifest.get("outputs") != expected_outputs:
        raise AETBatchAnalysisV2Error("analysis manifest output bindings are inconsistent")
    analysis = _read_json(output_root / "aet-batch-analysis-v2.json")
    if analysis.get("schema_version") != ANALYSIS_SCHEMA or analysis.get("status") != "complete":
        raise AETBatchAnalysisV2Error("completed analysis payload is invalid")
    return analysis


def execute_analysis(
    training_root: Path,
    frontier_root: Path,
    output_root: Path,
    *,
    prior_training_root: Path | None = None,
) -> dict[str, Any]:
    training_root = training_root.resolve()
    frontier_root = frontier_root.resolve()
    output_root = output_root.resolve()
    _, training_manifest_sha = _verify_bundle(training_root, "training-energy-summary.json")
    _, frontier_manifest_sha = _verify_bundle(frontier_root, "batch-frontier-v2-summary.json")
    training_path = training_root / "training-energy-summary.json"
    frontier_path = frontier_root / "batch-frontier-v2-summary.json"
    prior_energy = None
    prior_manifest_sha = None
    prior_summary_sha = None
    if prior_training_root is not None:
        prior_root = prior_training_root.resolve()
        _, prior_manifest_sha = _verify_bundle(prior_root, "training-energy-summary.json")
        prior_path = prior_root / "training-energy-summary.json"
        prior_energy = _prior_energy(_read_json(prior_path))
        prior_summary_sha = _sha256_file(prior_path)

    analysis = build_analysis(
        _read_json(training_path),
        _read_json(frontier_path),
        training_manifest_sha256=training_manifest_sha,
        frontier_manifest_sha256=frontier_manifest_sha,
        prior_campaign_energy_j=prior_energy,
        prior_training_manifest_sha256=prior_manifest_sha,
    )
    input_bindings = {
        "training_manifest_sha256": training_manifest_sha,
        "training_summary_sha256": _sha256_file(training_path),
        "frontier_manifest_sha256": frontier_manifest_sha,
        "frontier_summary_sha256": _sha256_file(frontier_path),
        "prior_training_manifest_sha256": prior_manifest_sha,
        "prior_training_summary_sha256": prior_summary_sha,
    }
    input_identity = _sha256_json(input_bindings)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = _recover_owned_staging(output_root)
    if output_root.exists():
        return _validate_completed_output(
            output_root,
            input_identity=input_identity,
            expected_inputs=input_bindings,
        )

    staging_root.mkdir()
    analysis_path = staging_root / "aet-batch-analysis-v2.json"
    table_path = staging_root / "aet-batch-table-v2.csv"
    comparison_path = staging_root / "architecture-comparison-v2.json"
    completion_path = staging_root / "completion.json"
    _write_json(analysis_path, analysis)
    _write_csv(table_path, analysis)
    _write_json(comparison_path, analysis["architecture_comparison"])
    completion = {
        "schema_version": COMPLETION_SCHEMA,
        "status": "complete",
        "baseline": "parallel_16_worker_hgs_10_seconds_per_instance",
        "neural_policy": "greedy_1x1",
        "study_development_debt": {
            architecture["architecture"]: architecture["training_debt_variants"][
                "measured_study_development"
            ]
            for architecture in analysis["architectures"]
        },
        "study_development_debt_caveat": (
            "measured known lower bound; unmeasured old prototypes and failures with "
            "unknown energy cannot be reconstructed"
        ),
        "finite_cells": sum(
            row["aet_status"] == STATUS_FINITE
            for architecture in analysis["architectures"]
            for row in architecture["batches"]
        ),
        "infinite_cells": sum(
            row["aet_status"] == STATUS_INFINITE
            for architecture in analysis["architectures"]
            for row in architecture["batches"]
        ),
        "infeasible_cells": sum(
            row["aet_status"] == STATUS_INFEASIBLE
            for architecture in analysis["architectures"]
            for row in architecture["batches"]
        ),
    }
    _write_json(completion_path, completion)
    outputs = [analysis_path, table_path, comparison_path, completion_path]
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": "complete",
        "input_identity_sha256": input_identity,
        "inputs": input_bindings,
        "outputs": {path.name: _sha256_file(path) for path in outputs},
    }
    manifest_path = staging_root / "manifest.json"
    _write_json(manifest_path, manifest)
    outputs.append(manifest_path)
    (staging_root / "SHA256SUMS").write_text(
        "".join(f"{_sha256_file(path)}  {path.name}\n" for path in outputs),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(staging_root, output_root)
    return _validate_completed_output(
        output_root,
        input_identity=input_identity,
        expected_inputs=input_bindings,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--frontier-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-training-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = execute_analysis(
            args.training_root,
            args.frontier_root,
            args.output_root,
            prior_training_root=args.prior_training_root,
        )
    except (AETBatchAnalysisV2Error, OSError) as exc:
        print(f"AET batch analysis v2 failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "schema_version": result["schema_version"],
                "output_root": str(args.output_root),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
