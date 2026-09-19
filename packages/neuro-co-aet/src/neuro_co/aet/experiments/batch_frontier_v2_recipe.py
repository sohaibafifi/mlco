"""Closed recipe for the corrected native-Windows CVRP50 deployment study.

Version 2 deliberately keeps the two quantities that were conflated in the
pilot separate: neural deployment is one greedy rollout per original problem
instance, while each HGS process receives ten seconds for its assigned
problem instance.  The old POMO 50x8 and HGS iteration-budget recipes remain
unchanged and cannot be supplied to this loader.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import yaml

from neuro_co.aet.experiments import software_recipe as software
from neuro_co.aet.experiments.deployment_energy_recipe import (
    GPU_DEVICE_ID_SHA256,
    HGS_SEEDS,
    QUALITY_SOURCE_ARTIFACTS,
    QUALITY_SOURCE_CORPUS_CONTENT_SHA256,
    QUALITY_SOURCE_GIT_SHA,
    QUALITY_SOURCE_ROOT,
    QUALITY_SOURCE_SNAPSHOT_SHA256,
    Artifact,
    Dataset,
    SourceBundle,
)

SCHEMA_VERSION = "aet-journal-batch-frontier-v2/v1"
KIND = "aet-journal-batch-frontier-v2"
OUTPUT_ROOT = "experiments/aet-journal/raw/batch-frontier-v2/cvrp50-seed2723"
TRAINING_SOURCE_ROOT = "experiments/aet-journal/raw/training-debt-v2/cvrp50-epoch100-seeds2-6"
TRAINING_MANIFEST_SCHEMA = "aet-training-debt-v2-manifest/v1"
ARCHITECTURES = ("am", "gnn")
TRAINING_SEEDS = (2, 3, 4, 5, 6)
EVALUATION_SEEDS = (102_000, 103_000, 104_000, 105_000, 106_000)
BATCH_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
HGS_MAX_RUNTIME_S = 10.0
QUALITY_THRESHOLD_PCT = 5.0

CLASSIFICATION: dict[str, Any] = {
    "purpose": "software_exploratory_batch_frontier_v2",
    "scientific_use": False,
    "confirmatory_eligible": False,
    "whole_system_energy": False,
    "aet_ready_components_only": True,
    "carbon_accounting": "none",
}

_TOP_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "qualification",
    "quality_source",
    "training_source",
    "dataset",
    "neural_policy",
    "hgs_policy",
    "capacity_probe",
    "quality_gate",
    "schedule",
    "measurement",
    "limits",
}


class BatchFrontierV2RecipeValidationError(ValueError):
    """Raised when a recipe differs from the corrected closed protocol."""


@dataclass(frozen=True, slots=True)
class TrainingSource:
    root: str
    manifest_path: str
    manifest_schema: str
    expected_status: str
    architectures: tuple[str, ...]
    training_seeds: tuple[int, ...]
    selected_epoch_min: int
    selected_epoch_max: int


@dataclass(frozen=True, slots=True)
class NeuralPolicy:
    mode_id: str
    n_starts: int
    augmentations: int
    forced_first_actions: bool
    inference_precision: str
    batch_candidates: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HGSPolicy:
    solver: str
    max_runtime_s: float
    scaling_factor: int
    collect_stats: bool
    parallel_workers: int
    cpu_threads_per_worker: int
    parallel: bool
    per_instance_independent_limit: bool
    executor: str
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CapacityProbePolicy:
    measured: bool
    checkpoint_seed: int
    stop_after_first_infeasible: bool
    require_contiguous_prefix: bool
    record_peak_cuda_memory: bool
    validate_routes: bool


@dataclass(frozen=True, slots=True)
class FrontierBlock:
    round_index: int
    order_index: int
    policy: str
    architecture: str | None
    batch_size: int | None
    training_seed: int
    evaluation_seed: int
    hgs_seed: int
    relative_path: str


@dataclass(frozen=True, slots=True)
class BatchFrontierV2Recipe:
    name: str
    host_id: str
    gpu_index: int
    gpu_device_id_sha256: str
    output_root: str
    preflight_report: str
    quality_source: SourceBundle
    training_source: TrainingSource
    dataset: Dataset
    neural_policy: NeuralPolicy
    hgs_policy: HGSPolicy
    capacity_probe: CapacityProbePolicy
    quality_threshold_pct: float
    paired_rounds: int
    schedule_design: str
    minimum_block_duration_s: float
    maximum_block_walltime_s: float
    campaign_attestation_max_age_s: float
    maximum_campaign_walltime_s: float


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchFrontierV2RecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise BatchFrontierV2RecipeValidationError(
            f"{where} keys differ; missing={missing!r}, unknown={unknown!r}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if type(value) is not type(expected) or value != expected:
        raise BatchFrontierV2RecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BatchFrontierV2RecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchFrontierV2RecipeValidationError(f"{where} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BatchFrontierV2RecipeValidationError(f"{where} must be a positive number")
    return result


def _relative(value: Any, where: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BatchFrontierV2RecipeValidationError(f"{where} must be a relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise BatchFrontierV2RecipeValidationError(f"{where} must be a safe relative path")
    if suffix is not None and posix.suffix != suffix:
        raise BatchFrontierV2RecipeValidationError(f"{where} must end with {suffix}")
    return posix.as_posix()


def _artifact(value: Any) -> Artifact:
    raw = _mapping(value, "quality_source.artifacts[]")
    _closed(raw, {"path", "sha256"}, "quality_source.artifacts[]")
    sha = raw["sha256"]
    if (
        not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
    ):
        raise BatchFrontierV2RecipeValidationError("artifact sha256 is invalid")
    return Artifact(_relative(raw["path"], "artifact.path"), sha)


def load_recipe(path: Path) -> BatchFrontierV2Recipe:
    """Load and strictly validate the corrected deployment protocol."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BatchFrontierV2RecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    raw = _mapping(raw, "recipe")
    _closed(raw, _TOP_KEYS, "recipe")
    _exact(raw["schema_version"], SCHEMA_VERSION, "recipe.schema_version")
    _exact(raw["kind"], KIND, "recipe.kind")
    _exact(raw["classification"], CLASSIFICATION, "classification")

    platform = _mapping(raw["platform"], "platform")
    _closed(
        platform,
        {"execution_layer", "host_id", "accelerator_label", "gpu_index", "gpu_device_id_sha256"},
        "platform",
    )
    _exact(platform["execution_layer"], "windows-native", "platform.execution_layer")
    _exact(platform["accelerator_label"], "NVIDIA RTX A4500", "platform.accelerator_label")
    _exact(platform["gpu_index"], 0, "platform.gpu_index")
    _exact(
        platform["gpu_device_id_sha256"],
        GPU_DEVICE_ID_SHA256,
        "platform.gpu_device_id_sha256",
    )

    qualification = _mapping(raw["qualification"], "qualification")
    _closed(qualification, {"preflight_report"}, "qualification")

    quality = _mapping(raw["quality_source"], "quality_source")
    _closed(
        quality,
        {"root", "expected_status", "source_git_sha", "source_snapshot_sha256", "artifacts"},
        "quality_source",
    )
    artifacts_raw = quality["artifacts"]
    if not isinstance(artifacts_raw, list):
        raise BatchFrontierV2RecipeValidationError("quality_source.artifacts must be a list")
    artifacts = tuple(_artifact(item) for item in artifacts_raw)
    _exact(
        tuple((item.path, item.sha256) for item in artifacts),
        QUALITY_SOURCE_ARTIFACTS,
        "quality_source.artifacts",
    )
    quality_source = SourceBundle(
        root=_exact(quality["root"], QUALITY_SOURCE_ROOT, "quality_source.root"),
        expected_status=_exact(
            quality["expected_status"], "complete_confirmatory_passed", "quality_source.status"
        ),
        source_git_sha=_exact(
            quality["source_git_sha"], QUALITY_SOURCE_GIT_SHA, "quality_source.source_git_sha"
        ),
        source_snapshot_sha256=_exact(
            quality["source_snapshot_sha256"],
            QUALITY_SOURCE_SNAPSHOT_SHA256,
            "quality_source.source_snapshot_sha256",
        ),
        artifacts=artifacts,
    )

    training = _mapping(raw["training_source"], "training_source")
    _closed(
        training,
        {
            "root",
            "manifest_path",
            "manifest_schema",
            "expected_status",
            "architectures",
            "training_seeds",
            "selected_epoch_min",
            "selected_epoch_max",
        },
        "training_source",
    )
    _exact(tuple(training["architectures"]), ARCHITECTURES, "training_source.architectures")
    _exact(tuple(training["training_seeds"]), TRAINING_SEEDS, "training_source.training_seeds")
    training_source = TrainingSource(
        root=_exact(training["root"], TRAINING_SOURCE_ROOT, "training_source.root"),
        manifest_path=_exact(
            training["manifest_path"], "manifest.json", "training_source.manifest"
        ),
        manifest_schema=_exact(
            training["manifest_schema"], TRAINING_MANIFEST_SCHEMA, "training_source.schema"
        ),
        expected_status=_exact(training["expected_status"], "complete", "training_source.status"),
        architectures=ARCHITECTURES,
        training_seeds=TRAINING_SEEDS,
        selected_epoch_min=_exact(
            _integer(training["selected_epoch_min"], "training_source.selected_epoch_min"),
            90,
            "training_source.selected_epoch_min",
        ),
        selected_epoch_max=_exact(
            _integer(training["selected_epoch_max"], "training_source.selected_epoch_max"),
            100,
            "training_source.selected_epoch_max",
        ),
    )

    dataset_raw = _mapping(raw["dataset"], "dataset")
    _closed(
        dataset_raw,
        {
            "id",
            "problem",
            "size",
            "capacity",
            "max_demand",
            "num_instances",
            "seed",
            "source_artifact",
            "content_sha256",
        },
        "dataset",
    )
    _exact(dataset_raw["content_sha256"], QUALITY_SOURCE_CORPUS_CONTENT_SHA256, "dataset.hash")
    dataset = Dataset(
        dataset_id=_exact(dataset_raw["id"], "cvrp50-hgs10-confirmatory-seed2723", "dataset.id"),
        problem=_exact(dataset_raw["problem"], "cvrp", "dataset.problem"),
        size=_exact(dataset_raw["size"], 50, "dataset.size"),
        capacity=_exact(dataset_raw["capacity"], 40.0, "dataset.capacity"),
        max_demand=_exact(dataset_raw["max_demand"], 9, "dataset.max_demand"),
        num_instances=_exact(dataset_raw["num_instances"], 512, "dataset.num_instances"),
        seed=_exact(dataset_raw["seed"], 2723, "dataset.seed"),
        artifact=_exact(
            dataset_raw["source_artifact"],
            "shared/cvrp50-hgs10-confirmatory-seed2723.npz",
            "dataset.source_artifact",
        ),
        forbidden_content_sha256=(),
    )

    neural = _mapping(raw["neural_policy"], "neural_policy")
    _closed(
        neural,
        {
            "mode_id",
            "n_starts",
            "augmentations",
            "forced_first_actions",
            "inference_precision",
            "batch_candidates",
            "evaluation_seeds",
        },
        "neural_policy",
    )
    neural_policy = NeuralPolicy(
        mode_id=_exact(neural["mode_id"], "greedy-1x1", "neural_policy.mode_id"),
        n_starts=_exact(neural["n_starts"], 1, "neural_policy.n_starts"),
        augmentations=_exact(neural["augmentations"], 1, "neural_policy.augmentations"),
        forced_first_actions=_exact(
            neural["forced_first_actions"], False, "neural_policy.forced_first_actions"
        ),
        inference_precision=_exact(
            neural["inference_precision"], "fp32", "neural_policy.inference_precision"
        ),
        batch_candidates=_exact(
            tuple(neural["batch_candidates"]), BATCH_CANDIDATES, "neural_policy.batch_candidates"
        ),
        evaluation_seeds=_exact(
            tuple(neural["evaluation_seeds"]), EVALUATION_SEEDS, "neural_policy.evaluation_seeds"
        ),
    )

    hgs = _mapping(raw["hgs_policy"], "hgs_policy")
    _closed(
        hgs,
        {
            "solver",
            "max_runtime_s",
            "scaling_factor",
            "collect_stats",
            "parallel_workers",
            "cpu_threads_per_worker",
            "parallel",
            "per_instance_independent_limit",
            "executor",
            "seeds",
        },
        "hgs_policy",
    )
    hgs_policy = HGSPolicy(
        solver=_exact(hgs["solver"], "pyvrp-hgs", "hgs_policy.solver"),
        max_runtime_s=_exact(hgs["max_runtime_s"], HGS_MAX_RUNTIME_S, "hgs_policy.max_runtime_s"),
        scaling_factor=_exact(hgs["scaling_factor"], 1_000_000, "hgs_policy.scaling_factor"),
        collect_stats=_exact(hgs["collect_stats"], False, "hgs_policy.collect_stats"),
        parallel_workers=_exact(hgs["parallel_workers"], 16, "hgs_policy.parallel_workers"),
        cpu_threads_per_worker=_exact(
            hgs["cpu_threads_per_worker"], 1, "hgs_policy.cpu_threads_per_worker"
        ),
        parallel=_exact(hgs["parallel"], True, "hgs_policy.parallel"),
        per_instance_independent_limit=_exact(
            hgs["per_instance_independent_limit"],
            True,
            "hgs_policy.per_instance_independent_limit",
        ),
        executor=_exact(hgs["executor"], "windows_process_pool", "hgs_policy.executor"),
        seeds=_exact(tuple(hgs["seeds"]), HGS_SEEDS, "hgs_policy.seeds"),
    )

    capacity = _mapping(raw["capacity_probe"], "capacity_probe")
    _closed(
        capacity,
        {
            "measured",
            "checkpoint_seed",
            "stop_after_first_infeasible",
            "require_contiguous_prefix",
            "record_peak_cuda_memory",
            "validate_routes",
        },
        "capacity_probe",
    )
    capacity_policy = CapacityProbePolicy(
        measured=_exact(capacity["measured"], False, "capacity_probe.measured"),
        checkpoint_seed=_exact(capacity["checkpoint_seed"], 2, "capacity_probe.checkpoint_seed"),
        stop_after_first_infeasible=_exact(
            capacity["stop_after_first_infeasible"],
            True,
            "capacity_probe.stop_after_first_infeasible",
        ),
        require_contiguous_prefix=_exact(
            capacity["require_contiguous_prefix"], True, "capacity_probe.require_contiguous_prefix"
        ),
        record_peak_cuda_memory=_exact(
            capacity["record_peak_cuda_memory"], True, "capacity_probe.record_peak_cuda_memory"
        ),
        validate_routes=_exact(capacity["validate_routes"], True, "capacity_probe.validate_routes"),
    )

    gate = _mapping(raw["quality_gate"], "quality_gate")
    _closed(gate, {"maximum_mean_gap_pct"}, "quality_gate")
    threshold = _exact(
        gate["maximum_mean_gap_pct"], QUALITY_THRESHOLD_PCT, "quality_gate.maximum_mean_gap_pct"
    )

    schedule = _mapping(raw["schedule"], "schedule")
    _closed(schedule, {"paired_rounds", "hgs_blocks_per_round", "design"}, "schedule")
    _exact(schedule["paired_rounds"], 5, "schedule.paired_rounds")
    _exact(schedule["hgs_blocks_per_round"], 1, "schedule.hgs_blocks_per_round")

    measurement = _mapping(raw["measurement"], "measurement")
    _closed(
        measurement,
        {
            "backend",
            "fallback",
            "domains",
            "minimum_block_duration_s",
            "neural_duration_policy",
            "hgs_duration_policy",
            "normalization_unit",
            "pue",
            "report_embodied",
            "whole_system",
            "exclusive_attestation",
            "gpu_process_lists",
        },
        "measurement",
    )
    _exact(
        measurement["backend"],
        "windows_emi_plus_nvml_total_energy_counter",
        "measurement.backend",
    )
    _exact(measurement["fallback"], False, "measurement.fallback")
    _exact(tuple(measurement["domains"]), ("cpu_package", "gpu"), "measurement.domains")
    _exact(measurement["neural_duration_policy"], "repeat_complete_corpus", "measurement.neural")
    _exact(
        measurement["hgs_duration_policy"],
        "one_complete_corpus_parallel_pool",
        "measurement.hgs",
    )
    _exact(measurement["normalization_unit"], "original_problem_instance", "measurement.unit")
    _exact(measurement["pue"], 1.0, "measurement.pue")
    _exact(measurement["report_embodied"], False, "measurement.report_embodied")
    _exact(measurement["whole_system"], False, "measurement.whole_system")
    _exact(
        measurement["exclusive_attestation"],
        "operator_authoritative_campaign",
        "measurement.exclusive_attestation",
    )
    _exact(measurement["gpu_process_lists"], "diagnostic_only", "measurement.gpu_process_lists")

    limits = _mapping(raw["limits"], "limits")
    _closed(
        limits,
        {
            "maximum_block_walltime_s",
            "campaign_attestation_max_age_s",
            "maximum_campaign_walltime_s",
        },
        "limits",
    )
    max_block = _positive_number(limits["maximum_block_walltime_s"], "limits.max_block")
    parallel_hgs_floor = (
        math.ceil(dataset.num_instances / hgs_policy.parallel_workers) * hgs_policy.max_runtime_s
    )
    if max_block <= parallel_hgs_floor:
        raise BatchFrontierV2RecipeValidationError(
            "maximum_block_walltime_s must exceed the parallel HGS block floor"
        )

    return BatchFrontierV2Recipe(
        name=str(raw["name"]),
        host_id=str(platform["host_id"]),
        gpu_index=0,
        gpu_device_id_sha256=GPU_DEVICE_ID_SHA256,
        output_root=_exact(raw["output_root"], OUTPUT_ROOT, "output_root"),
        preflight_report=_relative(
            qualification["preflight_report"], "qualification.preflight_report", ".json"
        ),
        quality_source=quality_source,
        training_source=training_source,
        dataset=dataset,
        neural_policy=neural_policy,
        hgs_policy=hgs_policy,
        capacity_probe=capacity_policy,
        quality_threshold_pct=float(threshold),
        paired_rounds=5,
        schedule_design=str(schedule["design"]),
        minimum_block_duration_s=_exact(
            measurement["minimum_block_duration_s"],
            120,
            "measurement.minimum_block_duration_s",
        ),
        maximum_block_walltime_s=max_block,
        campaign_attestation_max_age_s=_positive_number(
            limits["campaign_attestation_max_age_s"], "limits.attestation"
        ),
        maximum_campaign_walltime_s=_positive_number(
            limits["maximum_campaign_walltime_s"], "limits.campaign"
        ),
    )


def expected_blocks(
    recipe: BatchFrontierV2Recipe,
    feasible_batches: dict[str, tuple[int, ...]],
) -> tuple[FrontierBlock, ...]:
    """Return the deterministic five-round schedule after the capacity probe."""

    if set(feasible_batches) != set(ARCHITECTURES):
        raise BatchFrontierV2RecipeValidationError("capacity result must cover AM and GNN")
    cells = [
        (architecture, batch)
        for batch in recipe.neural_policy.batch_candidates
        for architecture in ARCHITECTURES
        if batch in feasible_batches[architecture]
    ]
    if any(not values for values in feasible_batches.values()):
        raise BatchFrontierV2RecipeValidationError("every architecture needs a feasible batch")
    count = len(cells)
    positions = (0, count, count // 2, count // 4, (3 * count) // 4)
    blocks: list[FrontierBlock] = []
    for round_index in range(recipe.paired_rounds):
        offset = (round_index // 2) * max(1, count // 3)
        ordered = cells[offset:] + cells[:offset]
        if round_index % 2:
            ordered = list(reversed(ordered))
        tokens: list[tuple[str | None, int | None]] = list(ordered)
        tokens.insert(positions[round_index], (None, None))
        for order_index, (architecture, batch_size) in enumerate(tokens):
            policy = "hgs" if architecture is None else "neural"
            suffix = "hgs10s" if architecture is None else f"{architecture}-b{batch_size:03d}"
            blocks.append(
                FrontierBlock(
                    round_index,
                    order_index,
                    policy,
                    architecture,
                    batch_size,
                    TRAINING_SEEDS[round_index],
                    EVALUATION_SEEDS[round_index],
                    HGS_SEEDS[round_index],
                    f"blocks/round-{round_index:02d}/order-{order_index:02d}-{suffix}.json",
                )
            )
    return tuple(blocks)


def minimum_measured_walltime_s(
    recipe: BatchFrontierV2Recipe,
    feasible_batches: dict[str, tuple[int, ...]] | None = None,
) -> float:
    """Return the protocol floor with 16 independent HGS jobs per parallel wave."""

    feasible = feasible_batches or {
        architecture: recipe.neural_policy.batch_candidates for architecture in ARCHITECTURES
    }
    neural_cells = sum(len(values) for values in feasible.values())
    neural = recipe.paired_rounds * neural_cells * recipe.minimum_block_duration_s
    hgs = (
        recipe.paired_rounds
        * math.ceil(recipe.dataset.num_instances / recipe.hgs_policy.parallel_workers)
        * recipe.hgs_policy.max_runtime_s
    )
    return neural + hgs


def qualify_for_execution(
    recipe: BatchFrontierV2Recipe,
    root: Path,
    training_source: Path | None = None,
) -> dict[str, Any]:
    """Perform a read-only host and source availability check."""

    source = training_source or root.joinpath(*PurePosixPath(recipe.training_source.root).parts)
    manifest_path = source / recipe.training_source.manifest_path
    training_valid = False
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        training_valid = bool(
            isinstance(manifest, dict)
            and manifest.get("schema_version") == recipe.training_source.manifest_schema
            and manifest.get("status") == recipe.training_source.expected_status
            and isinstance(manifest.get("checkpoint_entries"), list)
            and len(manifest["checkpoint_entries"]) == 10
        )
    quality_root = root.joinpath(*PurePosixPath(recipe.quality_source.root).parts)
    quality_valid = quality_root.is_dir() and all(
        (quality_root.joinpath(*PurePosixPath(artifact.path).parts)).is_file()
        for artifact in recipe.quality_source.artifacts
    )
    probe = SimpleNamespace(
        preflight_report=recipe.preflight_report,
        host_id=recipe.host_id,
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    native = software._qualification_report(probe)  # type: ignore[arg-type]
    ready = native["ready_to_execute"] is True and training_valid and quality_valid
    return {
        "native": native,
        "training_source_valid": training_valid,
        "quality_source_available": quality_valid,
        "ready_to_execute": ready,
    }


def dry_run(
    path: Path,
    workspace_root: Path | None = None,
    training_source: Path | None = None,
) -> dict[str, Any]:
    recipe = load_recipe(path)
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    qualification = qualify_for_execution(recipe, root, training_source)
    full_cells = len(ARCHITECTURES) * len(BATCH_CANDIDATES)
    floor = minimum_measured_walltime_s(recipe)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "valid_ready" if qualification["ready_to_execute"] else "valid_not_qualified",
        "classification": CLASSIFICATION,
        "name": recipe.name,
        "output_root": recipe.output_root,
        "neural_policy": "greedy-1x1",
        "hgs_max_runtime_s_per_instance": recipe.hgs_policy.max_runtime_s,
        "hgs_parallel_workers": recipe.hgs_policy.parallel_workers,
        "hgs_per_instance_independent_limit": True,
        "batch_candidates": list(BATCH_CANDIDATES),
        "maximum_block_count": recipe.paired_rounds * (1 + full_cells),
        "minimum_measured_walltime_s_at_full_capacity": floor,
        "minimum_measured_walltime_hours_at_full_capacity": floor / 3600,
        "qualification": qualification,
        "ready_to_execute": qualification["ready_to_execute"],
        "writes_performed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    report = dry_run(args.recipe, training_source=args.training_source)
    return 2 if args.require_ready and not report["ready_to_execute"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
