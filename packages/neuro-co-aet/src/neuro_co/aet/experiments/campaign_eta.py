"""Build a resume-aware return-time estimate for the complete AET campaign."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final

import yaml

ETA_SCHEMA: Final = "aet-complete-campaign-eta/v1"
TRAINING_ESTIMATE_SCHEMA: Final = "aet-training-debt-estimate/v1"
FRONTIER_ESTIMATE_SCHEMA: Final = "aet-batch-frontier-estimate/v1"
EXPECTED_ARCHITECTURES: Final = ("am", "gnn")
DEFAULT_FIXED_OVERHEAD_S: Final = 900.0
DEFAULT_MARGIN_FRACTIONS: Final = {"lower": 0.10, "point": 0.25, "upper": 0.40}
QUALITY_CYCLE_FACTORS: Final = {"lower": 1.0, "point": 1.5, "upper": 3.0}
NEURAL_OVERSHOOT_FACTORS: Final = {"lower": 0.0, "point": 1.0, "upper": 2.0}


class AETCampaignETAError(RuntimeError):
    """Raised when the timed probe cannot support a credible estimate."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AETCampaignETAError(f"cannot read estimate {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AETCampaignETAError("training estimate must contain one JSON object")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AETCampaignETAError(f"cannot read frontier recipe {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AETCampaignETAError("frontier recipe must contain one mapping")
    return value


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AETCampaignETAError(f"{where} must be an object")
    return value


def _finite(value: Any, where: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool):
        raise AETCampaignETAError(f"{where} must be a finite number >= {minimum}")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AETCampaignETAError(f"{where} must be a finite number >= {minimum}") from exc
    if not math.isfinite(result) or result < minimum:
        raise AETCampaignETAError(f"{where} must be a finite number >= {minimum}")
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AETCampaignETAError(f"{where} must be an integer >= {minimum}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def build_eta(
    estimate: dict[str, Any],
    frontier_recipe: dict[str, Any],
    *,
    frontier_estimate: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Combine real CUDA timing with the exact frontier schedule implied by capacity."""
    if estimate.get("schema_version") != TRAINING_ESTIMATE_SCHEMA:
        raise AETCampaignETAError("unsupported training estimate schema")
    if estimate.get("status") != "ready":
        raise AETCampaignETAError("training estimate is not ready")
    cuda_probe = _mapping(estimate.get("cuda_probe"), "estimate.cuda_probe")
    if cuda_probe.get("real_forward_backward") is not True:
        raise AETCampaignETAError("ETA requires a real CUDA forward/backward probe")
    architecture_qualification = _mapping(
        cuda_probe.get("architecture_qualification"),
        "estimate.cuda_probe.architecture_qualification",
    )
    if set(architecture_qualification) != set(EXPECTED_ARCHITECTURES) or not all(
        _mapping(architecture_qualification[item], f"CUDA qualification {item}").get("ready")
        is True
        for item in EXPECTED_ARCHITECTURES
    ):
        raise AETCampaignETAError("CUDA probe must qualify exactly AM and GNN")

    architecture_estimates = _mapping(estimate.get("architectures"), "estimate.architectures")
    if set(architecture_estimates) != set(EXPECTED_ARCHITECTURES):
        raise AETCampaignETAError("training estimate must cover exactly AM and GNN")
    remaining_training = {"lower": 0.0, "point": 0.0, "upper": 0.0}
    remaining_seed_counts: dict[str, int] = {}
    architecture_rows: list[dict[str, Any]] = []
    for architecture in EXPECTED_ARCHITECTURES:
        entry = _mapping(architecture_estimates[architecture], f"estimate.{architecture}")
        remaining_count = _integer(
            entry.get("remaining_seed_count"), f"estimate.{architecture}.remaining_seed_count"
        )
        completed_count = _integer(
            entry.get("completed_seed_count"), f"estimate.{architecture}.completed_seed_count"
        )
        if remaining_count + completed_count != 5:
            raise AETCampaignETAError(f"estimate.{architecture} seed counts must sum to five")
        remaining_seed_counts[architecture] = remaining_count
        per_seed_lower = _finite(
            entry.get("seconds_per_training_seed_lower"),
            f"estimate.{architecture}.seconds_per_training_seed_lower",
        )
        per_seed_point = _finite(
            entry.get("seconds_per_training_seed_point"),
            f"estimate.{architecture}.seconds_per_training_seed_point",
        )
        per_seed_upper = _finite(
            entry.get("seconds_per_training_seed_upper"),
            f"estimate.{architecture}.seconds_per_training_seed_upper",
        )
        lower = remaining_count * per_seed_lower
        point = remaining_count * per_seed_point
        upper = remaining_count * per_seed_upper
        if not lower <= point <= upper:
            raise AETCampaignETAError(
                f"estimate.{architecture} walltime must satisfy lower <= point <= upper"
            )
        for key, value in (("lower", lower), ("point", point), ("upper", upper)):
            remaining_training[key] += value

        architecture_rows.append(
            {
                "architecture": architecture,
                "completed_seed_count": completed_count,
                "remaining_seed_count": remaining_count,
                "remaining_training_walltime_s": {
                    "lower": lower,
                    "point": point,
                    "upper": upper,
                },
                "seconds_per_training_seed": {
                    "lower": per_seed_lower,
                    "point": per_seed_point,
                    "upper": per_seed_upper,
                },
            }
        )

    schedule = _mapping(frontier_recipe.get("schedule"), "frontier.schedule")
    measurement = _mapping(frontier_recipe.get("measurement"), "frontier.measurement")
    dataset = _mapping(frontier_recipe.get("dataset"), "frontier.dataset")
    limits = _mapping(frontier_recipe.get("limits"), "frontier.limits")
    rounds = _integer(schedule.get("paired_rounds"), "frontier.schedule.paired_rounds", minimum=1)
    hgs_per_round = _integer(
        schedule.get("hgs_blocks_per_round"),
        "frontier.schedule.hgs_blocks_per_round",
        minimum=1,
    )
    block_duration = _finite(
        measurement.get("minimum_block_duration_s"),
        "frontier.measurement.minimum_block_duration_s",
        minimum=1.0,
    )
    corpus_size = _integer(
        dataset.get("num_instances"), "frontier.dataset.num_instances", minimum=1
    )
    maximum_block_walltime_s = _finite(
        limits.get("maximum_block_walltime_s"),
        "frontier.limits.maximum_block_walltime_s",
        minimum=block_duration,
    )
    campaign_attestation_max_age_s = _finite(
        limits.get("campaign_attestation_max_age_s"),
        "frontier.limits.campaign_attestation_max_age_s",
        minimum=1.0,
    )
    frontier_campaign_maximum_s = _finite(
        limits.get("maximum_campaign_walltime_s"),
        "frontier.limits.maximum_campaign_walltime_s",
        minimum=1.0,
    )
    training_campaign_maximum_s = _finite(
        estimate.get("maximum_campaign_walltime_s"),
        "estimate.maximum_campaign_walltime_s",
        minimum=1.0,
    )
    allowed_campaign_walltime_s = min(
        campaign_attestation_max_age_s,
        frontier_campaign_maximum_s,
        training_campaign_maximum_s,
    )
    neural_policy = _mapping(frontier_recipe.get("neural_policy"), "frontier.neural_policy")
    candidates = neural_policy.get("batch_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AETCampaignETAError("frontier.neural_policy.batch_candidates must be non-empty")
    candidate_values = [
        _integer(value, "frontier batch candidate", minimum=1) for value in candidates
    ]
    if candidate_values != sorted(set(candidate_values)):
        raise AETCampaignETAError("frontier batch candidates must be sorted and unique")
    candidate_count = len(candidate_values)
    minimum_frontier_blocks = rounds * (hgs_per_round + len(EXPECTED_ARCHITECTURES))
    maximum_frontier_blocks = rounds * (
        hgs_per_round + len(EXPECTED_ARCHITECTURES) * candidate_count
    )
    minimum_frontier_s = minimum_frontier_blocks * block_duration
    maximum_frontier_s = maximum_frontier_blocks * block_duration
    feasible_batches: dict[str, list[int]] | None = None
    quality_evaluation_walltime = {"lower": 0.0, "point": 0.0, "upper": 0.0}
    neural_cycle_estimates: dict[str, dict[int, float]] | None = None
    overshoot_walltime = {"lower": 0.0, "point": 0.0, "upper": 0.0}
    if frontier_estimate is not None:
        if (
            frontier_estimate.get("schema_version") != FRONTIER_ESTIMATE_SCHEMA
            or frontier_estimate.get("status") != "complete"
            or frontier_estimate.get("energy_measured") is not False
            or frontier_estimate.get("writes_performed") is not False
        ):
            raise AETCampaignETAError("frontier estimate is not a completed unmeasured probe")
        probe = _mapping(frontier_estimate.get("capacity_probe"), "frontier capacity probe")
        if probe.get("checkpoint_backed") is not False or probe.get("measured_energy") is not False:
            raise AETCampaignETAError(
                "frontier ETA capacity probe must be unmeasured and untrained"
            )
        probe_architectures = _mapping(
            probe.get("architectures"), "frontier capacity architectures"
        )
        if set(probe_architectures) != set(EXPECTED_ARCHITECTURES):
            raise AETCampaignETAError("frontier estimate must cover exactly AM and GNN")
        feasible_batches = {}
        neural_cycle_estimates = {}
        for architecture in EXPECTED_ARCHITECTURES:
            record = _mapping(
                probe_architectures[architecture], f"frontier capacity {architecture}"
            )
            raw_prefix = record.get("feasible_prefix")
            if not isinstance(raw_prefix, list) or not raw_prefix:
                raise AETCampaignETAError(
                    f"frontier capacity {architecture} needs a non-empty feasible prefix"
                )
            prefix = [
                _integer(value, f"frontier capacity {architecture} batch", minimum=1)
                for value in raw_prefix
            ]
            if prefix != candidate_values[: len(prefix)]:
                raise AETCampaignETAError(
                    f"frontier capacity {architecture} is not a candidate prefix"
                )
            feasible_batches[architecture] = prefix
            raw_records = record.get("records")
            if not isinstance(raw_records, list):
                raise AETCampaignETAError(
                    f"frontier capacity {architecture} records must be an array"
                )
            elapsed_by_batch: dict[int, float] = {}
            for item_index, raw_item in enumerate(raw_records):
                item = _mapping(
                    raw_item,
                    f"frontier capacity {architecture}.records[{item_index}]",
                )
                if item.get("status") != "feasible":
                    continue
                batch = _integer(
                    item.get("batch_size"),
                    f"frontier capacity {architecture} record batch",
                    minimum=1,
                )
                elapsed_by_batch[batch] = _finite(
                    item.get("elapsed_s"),
                    f"frontier capacity {architecture} batch {batch} elapsed_s",
                    minimum=1e-9,
                )
            if any(batch not in elapsed_by_batch for batch in prefix):
                raise AETCampaignETAError(
                    f"frontier capacity {architecture} lacks timed feasible records"
                )
            cycle_estimates = {
                batch: elapsed_by_batch[batch] * math.ceil(corpus_size / batch) for batch in prefix
            }
            neural_cycle_estimates[architecture] = cycle_estimates
            quality_batch_size = 4
            if quality_batch_size not in elapsed_by_batch:
                raise AETCampaignETAError(
                    f"frontier capacity {architecture} did not qualify quality batch 4"
                )
            quality_cycle_s = elapsed_by_batch[quality_batch_size] * math.ceil(
                corpus_size / quality_batch_size
            )
            for bound, factor in QUALITY_CYCLE_FACTORS.items():
                quality_evaluation_walltime[bound] += (
                    remaining_seed_counts[architecture] * quality_cycle_s * factor
                )
        resolved_blocks = rounds * (
            hgs_per_round + sum(len(values) for values in feasible_batches.values())
        )
        if frontier_estimate.get("resolved_block_count") != resolved_blocks:
            raise AETCampaignETAError("frontier estimate block count does not match capacity")
        resolved_frontier_s = resolved_blocks * block_duration
        declared_frontier_s = _finite(
            frontier_estimate.get("minimum_measured_walltime_s"),
            "frontier estimate minimum measured walltime",
        )
        if not math.isclose(resolved_frontier_s, declared_frontier_s, abs_tol=1e-9):
            raise AETCampaignETAError("frontier estimate walltime does not match its schedule")
        neural_cycle_sum_s = rounds * sum(
            sum(cycles.values()) for cycles in neural_cycle_estimates.values()
        )
        successful_block_headroom_s = maximum_block_walltime_s - block_duration
        overshoot_walltime = {
            "lower": 0.0,
            "point": neural_cycle_sum_s
            + rounds * hgs_per_round * min(block_duration, successful_block_headroom_s),
            "upper": rounds
            * sum(
                sum(
                    min(
                        NEURAL_OVERSHOOT_FACTORS["upper"] * cycle_s,
                        successful_block_headroom_s,
                    )
                    for cycle_s in cycles.values()
                )
                for cycles in neural_cycle_estimates.values()
            )
            + rounds * hgs_per_round * successful_block_headroom_s,
        }
        frontier_walltime = {
            bound: resolved_frontier_s + overshoot_walltime[bound]
            for bound in ("lower", "point", "upper")
        }
    else:
        resolved_blocks = None
        quality_evaluation_walltime = {
            "lower": 0.0,
            "point": 1800.0,
            "upper": 3600.0,
        }
        frontier_walltime = {
            "lower": minimum_frontier_s,
            "point": maximum_frontier_s,
            "upper": maximum_frontier_blocks * maximum_block_walltime_s,
        }

    fixed_overhead_s = DEFAULT_FIXED_OVERHEAD_S
    lower_margin = DEFAULT_MARGIN_FRACTIONS["lower"]
    point_margin = DEFAULT_MARGIN_FRACTIONS["point"]
    upper_margin = DEFAULT_MARGIN_FRACTIONS["upper"]

    raw_totals = {
        key: remaining_training[key]
        + quality_evaluation_walltime[key]
        + frontier_walltime[key]
        + fixed_overhead_s
        for key in ("lower", "point", "upper")
    }
    total_s = {
        "lower": raw_totals["lower"] * (1.0 + lower_margin),
        "point": raw_totals["point"] * (1.0 + point_margin),
        "upper": raw_totals["upper"] * (1.0 + upper_margin),
    }
    if not total_s["lower"] <= total_s["point"] <= total_s["upper"]:
        raise AETCampaignETAError("combined ETA must be ordered")
    if total_s["upper"] > allowed_campaign_walltime_s:
        raise AETCampaignETAError(
            "conservative campaign ETA exceeds the attestation or campaign walltime limit"
        )

    generated_at = now or datetime.now().astimezone()
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise AETCampaignETAError("ETA timestamp must include a timezone")
    completion = {
        key: (generated_at + timedelta(seconds=value)).isoformat() for key, value in total_s.items()
    }
    return {
        "schema_version": ETA_SCHEMA,
        "status": "ready",
        "generated_at": generated_at.isoformat(),
        "architectures": architecture_rows,
        "remaining_training_walltime_s": remaining_training,
        "remaining_quality_evaluation_walltime_s": quality_evaluation_walltime,
        "frontier": {
            "paired_rounds": rounds,
            "hgs_blocks_per_round": hgs_per_round,
            "capacity_known_pre_attestation": feasible_batches is not None,
            "feasible_batch_sizes": feasible_batches,
            "resolved_measured_block_count": resolved_blocks,
            "exact_minimum_measured_block_count": minimum_frontier_blocks,
            "maximum_measured_block_count": maximum_frontier_blocks,
            "minimum_block_duration_s": block_duration,
            "exact_minimum_measured_walltime_s": minimum_frontier_s,
            "maximum_minimum_measured_walltime_s": maximum_frontier_s,
            "neural_full_corpus_cycle_estimates_s": neural_cycle_estimates,
            "block_completion_overshoot_budget_s": overshoot_walltime,
            "hgs_upper_bound_uses_maximum_block_walltime": True,
            "eta_walltime_s": frontier_walltime,
        },
        "margin": {
            "fixed_overhead_s": fixed_overhead_s,
            "lower_fraction": lower_margin,
            "point_fraction": point_margin,
            "upper_fraction": upper_margin,
        },
        "remaining_total_walltime_s": total_s,
        "remaining_total_hours": {key: value / 3600.0 for key, value in total_s.items()},
        "estimated_completion_at": completion,
        "recommended_return_at": completion["upper"],
        "resume_aware": True,
        "resume_awareness": {
            "training_completed_seeds": True,
            "frontier_completed_blocks": False,
            "frontier_assumption": "full_resolved_schedule",
        },
        "limit_check": {
            "allowed_campaign_walltime_s": allowed_campaign_walltime_s,
            "upper_eta_within_limit": True,
        },
        "estimate_scope": {
            "quality_evaluations_budgeted_from_unmeasured_batch4_cuda_probe": (
                frontier_estimate is not None
            ),
            "whole_corpus_block_overshoot_budgeted": True,
            "energy_measurement": "none",
        },
        "measurement_started": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-estimate", type=Path, required=True)
    parser.add_argument("--frontier-estimate", type=Path)
    parser.add_argument("--frontier-recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        eta = build_eta(
            _read_json(args.training_estimate),
            _read_yaml(args.frontier_recipe),
            frontier_estimate=(
                _read_json(args.frontier_estimate) if args.frontier_estimate is not None else None
            ),
        )
        _write_json(args.output, eta)
    except (AETCampaignETAError, OSError) as exc:
        print(f"AET campaign ETA failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(eta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
