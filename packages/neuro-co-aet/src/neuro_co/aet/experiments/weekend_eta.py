"""Build a pre-attestation ETA for the native-Windows AET weekend campaign.

The estimate combines the timed AM/GNN training probe with the closed
deployment schedule.  HGS is budgeted as waves of a fixed 16-process pool,
with ten seconds available to each instance. Neural blocks are budgeted from
the minimum measured block duration. No energy measurement is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import yaml

ETA_SCHEMA: Final = "aet-weekend-campaign-eta/v1"
EXPECTED_ARCHITECTURES: Final = ("am", "gnn")
DEFAULT_FIXED_OVERHEAD_S: Final = 1_800.0
MARGIN_FRACTIONS: Final = {"lower": 0.08, "point": 0.12, "upper": 0.16}


class WeekendETAError(RuntimeError):
    """Raised when the dry-run evidence cannot support a campaign ETA."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WeekendETAError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WeekendETAError(f"{path} must contain one JSON object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise WeekendETAError(f"cannot read YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WeekendETAError(f"{path} must contain one YAML mapping")
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WeekendETAError(f"{where} must be an object")
    return value


def _number(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise WeekendETAError(f"{where} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WeekendETAError(f"{where} must be a finite number >= {minimum}") from exc
    if not math.isfinite(result) or result < minimum:
        raise WeekendETAError(f"{where} must be a finite number >= {minimum}")
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise WeekendETAError(f"{where} must be an integer >= {minimum}")
    return value


def _training_seconds(estimate: dict[str, Any]) -> dict[str, float]:
    if estimate.get("status") != "ready":
        raise WeekendETAError("training estimate is not ready")
    classification = _mapping(estimate.get("classification"), "training.classification")
    if classification.get("energy_measurement") != "none":
        raise WeekendETAError("training ETA probe must not measure energy")
    cuda_probe = _mapping(estimate.get("cuda_probe"), "training.cuda_probe")
    if cuda_probe.get("real_forward_backward") is not True:
        raise WeekendETAError("training ETA requires a real CUDA forward/backward probe")

    architectures = _mapping(estimate.get("architectures"), "training.architectures")
    if set(architectures) != set(EXPECTED_ARCHITECTURES):
        raise WeekendETAError("training estimate must contain exactly AM and GNN")
    totals = {"lower": 0.0, "point": 0.0, "upper": 0.0}
    for architecture in EXPECTED_ARCHITECTURES:
        row = _mapping(architectures[architecture], f"training.architectures.{architecture}")
        remaining = _integer(
            row.get("remaining_seed_count"),
            f"training.architectures.{architecture}.remaining_seed_count",
        )
        completed = _integer(
            row.get("completed_seed_count"),
            f"training.architectures.{architecture}.completed_seed_count",
        )
        if remaining + completed != 5:
            raise WeekendETAError(
                f"training {architecture} completed and remaining seed counts must sum to five"
            )
        for bound in totals:
            estimate_field = (
                f"seconds_per_complete_cell_{bound}"
                if f"seconds_per_complete_cell_{bound}" in row
                else f"seconds_per_training_seed_{bound}"
            )
            per_seed = _number(
                row.get(estimate_field),
                f"training {architecture} {estimate_field}",
            )
            totals[bound] += remaining * per_seed
    if not totals["lower"] <= totals["point"] <= totals["upper"]:
        raise WeekendETAError("training estimate must satisfy lower <= point <= upper")
    return totals


def _capacity_batches(estimate: dict[str, Any], recipe_batches: list[int]) -> dict[str, list[int]]:
    capacity = _mapping(estimate.get("capacity_probe"), "frontier.capacity_probe")
    if capacity.get("measured_energy") is not False:
        raise WeekendETAError("frontier capacity probe must not measure energy")
    raw_architectures = _mapping(capacity.get("architectures"), "frontier.capacity.architectures")
    if set(raw_architectures) != set(EXPECTED_ARCHITECTURES):
        raise WeekendETAError("frontier capacity estimate must contain exactly AM and GNN")
    result: dict[str, list[int]] = {}
    for architecture in EXPECTED_ARCHITECTURES:
        row = _mapping(raw_architectures[architecture], f"frontier.capacity.{architecture}")
        raw_prefix = row.get("feasible_prefix")
        if not isinstance(raw_prefix, list) or not raw_prefix:
            raise WeekendETAError(f"frontier capacity {architecture} needs a feasible prefix")
        prefix = [
            _integer(value, f"frontier capacity {architecture} batch", minimum=1)
            for value in raw_prefix
        ]
        if prefix != recipe_batches[: len(prefix)]:
            raise WeekendETAError(
                f"frontier capacity {architecture} is not a prefix of the recipe grid"
            )
        result[architecture] = prefix
    return result


def _frontier_seconds(
    estimate: dict[str, Any], recipe: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    if estimate.get("status") not in {"complete", "ready"}:
        raise WeekendETAError("frontier estimate is not complete")
    if (
        estimate.get("energy_measured") is not False
        or estimate.get("writes_performed") is not False
    ):
        raise WeekendETAError("frontier estimate must be dry and unmeasured")

    dataset = _mapping(recipe.get("dataset"), "frontier recipe dataset")
    neural_policy = _mapping(recipe.get("neural_policy"), "frontier recipe neural_policy")
    hgs_policy = _mapping(recipe.get("hgs_policy"), "frontier recipe hgs_policy")
    schedule = _mapping(recipe.get("schedule"), "frontier recipe schedule")
    measurement = _mapping(recipe.get("measurement"), "frontier recipe measurement")

    num_instances = _integer(dataset.get("num_instances"), "dataset.num_instances", minimum=1)
    rounds = _integer(schedule.get("paired_rounds"), "schedule.paired_rounds", minimum=1)
    hgs_blocks = _integer(
        schedule.get("hgs_blocks_per_round"),
        "schedule.hgs_blocks_per_round",
        minimum=1,
    )
    max_runtime_s = _number(
        hgs_policy.get("max_runtime_s"), "hgs_policy.max_runtime_s", minimum=0.001
    )
    worker_count = _integer(
        hgs_policy.get("parallel_workers"), "hgs_policy.parallel_workers", minimum=1
    )
    if (
        worker_count != 16
        or hgs_policy.get("parallel") is not True
        or hgs_policy.get("per_instance_independent_limit") is not True
        or hgs_policy.get("cpu_threads_per_worker") != 1
        or hgs_policy.get("executor") != "windows_process_pool"
    ):
        raise WeekendETAError("HGS ETA requires the fixed 16-process one-thread-per-worker pool")
    if not math.isclose(max_runtime_s, 10.0, rel_tol=0.0, abs_tol=1e-12):
        raise WeekendETAError("weekend AET baseline must use ten seconds per HGS instance")

    raw_batches = neural_policy.get("batch_candidates")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise WeekendETAError("neural_policy.batch_candidates must be non-empty")
    batches = [_integer(value, "neural_policy batch candidate", minimum=1) for value in raw_batches]
    if batches != sorted(set(batches)):
        raise WeekendETAError("neural batch candidates must be sorted and unique")
    if neural_policy.get("n_starts") != 1 or neural_policy.get("augmentations") != 1:
        raise WeekendETAError("weekend deployment must use greedy 1x1 decoding")
    feasible = _capacity_batches(estimate, batches)
    minimum_block_value = measurement.get(
        "minimum_neural_block_duration_s", measurement.get("minimum_block_duration_s")
    )
    minimum_block_s = _number(
        minimum_block_value,
        "measurement.minimum_neural_block_duration_s",
        minimum=1.0,
    )

    expected_hgs_blocks = rounds * hgs_blocks
    expected_neural_blocks = rounds * sum(len(feasible[item]) for item in EXPECTED_ARCHITECTURES)
    total_hgs_blocks = _integer(
        estimate.get("total_hgs_block_count"), "frontier.total_hgs_block_count"
    )
    completed_hgs_blocks = _integer(
        estimate.get("completed_hgs_block_count"), "frontier.completed_hgs_block_count"
    )
    remaining_hgs_blocks = _integer(
        estimate.get("remaining_hgs_block_count"), "frontier.remaining_hgs_block_count"
    )
    total_neural_blocks = _integer(
        estimate.get("total_neural_block_count"), "frontier.total_neural_block_count"
    )
    completed_neural_blocks = _integer(
        estimate.get("completed_neural_block_count"),
        "frontier.completed_neural_block_count",
    )
    remaining_neural_blocks = _integer(
        estimate.get("remaining_neural_block_count"),
        "frontier.remaining_neural_block_count",
    )
    total_blocks = _integer(estimate.get("total_block_count"), "frontier.total_block_count")
    completed_blocks = _integer(
        estimate.get("completed_block_count"), "frontier.completed_block_count"
    )
    remaining_blocks = _integer(
        estimate.get("remaining_block_count"), "frontier.remaining_block_count"
    )
    if (
        total_hgs_blocks != expected_hgs_blocks
        or completed_hgs_blocks + remaining_hgs_blocks != total_hgs_blocks
        or total_neural_blocks != expected_neural_blocks
        or completed_neural_blocks + remaining_neural_blocks != total_neural_blocks
        or total_blocks != total_hgs_blocks + total_neural_blocks
        or completed_blocks != completed_hgs_blocks + completed_neural_blocks
        or remaining_blocks != remaining_hgs_blocks + remaining_neural_blocks
        or completed_blocks + remaining_blocks != total_blocks
    ):
        raise WeekendETAError("frontier completed and remaining block identities are inconsistent")

    hgs_waves_per_block = math.ceil(num_instances / worker_count)
    hgs_s = remaining_hgs_blocks * hgs_waves_per_block * max_runtime_s
    neural_s = remaining_neural_blocks * minimum_block_s
    raw_s = hgs_s + neural_s
    seconds = {bound: raw_s for bound in ("lower", "point", "upper")}
    return seconds, {
        "paired_rounds": rounds,
        "instances_per_hgs_block": num_instances,
        "hgs_blocks_total": total_hgs_blocks,
        "hgs_blocks_completed": completed_hgs_blocks,
        "hgs_blocks_remaining": remaining_hgs_blocks,
        "hgs_parallel_workers": worker_count,
        "hgs_cpu_threads_per_worker": 1,
        "hgs_waves_per_block": hgs_waves_per_block,
        "hgs_max_runtime_s_per_instance": max_runtime_s,
        "hgs_budget_walltime_s": hgs_s,
        "neural_minimum_block_duration_s": minimum_block_s,
        "neural_blocks_total": total_neural_blocks,
        "neural_blocks_completed": completed_neural_blocks,
        "neural_blocks_remaining": remaining_neural_blocks,
        "neural_minimum_walltime_s": neural_s,
        "feasible_batches": feasible,
        "greedy_1x1": True,
    }


def build_eta(
    training_estimate: dict[str, Any],
    frontier_estimate: dict[str, Any],
    frontier_recipe: dict[str, Any],
    *,
    now: datetime | None = None,
    fixed_overhead_s: float = DEFAULT_FIXED_OVERHEAD_S,
) -> dict[str, Any]:
    """Combine dry training and deployment estimates into one return window."""
    training_s = _training_seconds(training_estimate)
    frontier_s, schedule = _frontier_seconds(frontier_estimate, frontier_recipe)
    overhead = _number(fixed_overhead_s, "fixed_overhead_s")
    total_s = {
        bound: (training_s[bound] + frontier_s[bound] + overhead) * (1.0 + MARGIN_FRACTIONS[bound])
        for bound in ("lower", "point", "upper")
    }
    if not total_s["lower"] <= total_s["point"] <= total_s["upper"]:
        raise WeekendETAError("combined estimate must satisfy lower <= point <= upper")
    generated_at = now or datetime.now(UTC)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    return {
        "schema_version": ETA_SCHEMA,
        "status": "ready",
        "generated_at": generated_at.isoformat(),
        "energy_measured": False,
        "writes_outside_eta_report": False,
        "remaining_training_walltime_s": training_s,
        "remaining_frontier_walltime_s": frontier_s,
        "fixed_overhead_s": overhead,
        "operational_margin_fraction": MARGIN_FRACTIONS,
        "remaining_total_walltime_s": total_s,
        "remaining_total_hours": {bound: total_s[bound] / 3_600.0 for bound in total_s},
        "recommended_return_at": (generated_at + timedelta(seconds=total_s["upper"])).isoformat(),
        "schedule_basis": schedule,
        "interpretation": (
            "pre-attestation walltime estimate; HGS uses waves of its fixed 16-process "
            "pool and greedy neural cells use their minimum measured block duration"
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-estimate", type=Path, required=True)
    parser.add_argument("--frontier-estimate", type=Path, required=True)
    parser.add_argument("--frontier-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build_eta(
            _read_json(args.training_estimate),
            _read_json(args.frontier_estimate),
            _read_yaml(args.frontier_recipe),
        )
        _write_json(args.output, result)
    except (OSError, WeekendETAError) as exc:
        print(f"weekend ETA failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
