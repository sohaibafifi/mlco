"""Closed recipe for the native-Windows CVRP50 neural batch frontier.

The campaign compares two preregistered neural architecture identifiers over
the same powers-of-two batch grid and five training seeds.  One HGS-10 block
per round is shared by every neural cell in that round.  The measured training
bundle is resolved at run time and then content-addressed in the durable run
state, so inference weights and the training-energy numerator cannot drift.
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
    HGSPolicy,
    SourceBundle,
)

SCHEMA_VERSION = "aet-journal-batch-frontier/v1"
KIND = "aet-journal-batch-frontier"
OUTPUT_ROOT = "experiments/aet-journal/raw/batch-frontier/cvrp50-seed2723"
TRAINING_SOURCE_ROOT = "experiments/aet-journal/raw/training-debt/cvrp50-epoch40-seeds2-6"
TRAINING_MANIFEST_SCHEMA = "aet-training-debt-manifest/v1"
TRAINING_COMPLETE_STATUS = "complete"
ARCHITECTURES = ("am", "gnn")
TRAINING_SEEDS = (2, 3, 4, 5, 6)
EVALUATION_SEEDS = (102_000, 103_000, 104_000, 105_000, 106_000)
BATCH_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

CLASSIFICATION: dict[str, Any] = {
    "purpose": "software_exploratory_batch_frontier",
    "scientific_use": False,
    "confirmatory_eligible": False,
    "whole_system_energy": False,
    "aet_ready_components_only": True,
    "carbon_accounting": "none",
}

_SAFE = __import__("re").compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
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
    "schedule",
    "measurement",
    "limits",
}


class BatchFrontierRecipeValidationError(ValueError):
    """Raised when the batch-frontier recipe differs from the closed design."""


@dataclass(frozen=True, slots=True)
class TrainingSource:
    root: str
    manifest_path: str
    manifest_schema: str
    expected_status: str
    architectures: tuple[str, ...]
    training_seeds: tuple[int, ...]
    checkpoint_epoch: int


@dataclass(frozen=True, slots=True)
class ArchitecturePolicy:
    architecture: str
    encoder_class: str
    same_pomo_objective: bool
    same_pointer_decoder: bool


@dataclass(frozen=True, slots=True)
class NeuralPolicy:
    mode_id: str
    checkpoint_epoch: int
    n_starts: int
    augmentations: int
    forced_first_actions: bool
    inference_precision: str
    batch_candidates: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    architectures: tuple[ArchitecturePolicy, ...]


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
class BatchFrontierRecipe:
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
    paired_rounds: int
    schedule_design: str
    minimum_block_duration_s: float
    maximum_block_walltime_s: float
    campaign_attestation_max_age_s: float
    maximum_campaign_walltime_s: float


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchFrontierRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise BatchFrontierRecipeValidationError(
            f"{where} keys differ; missing={missing!r}, unknown={unknown!r}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if value != expected or type(value) is not type(expected):
        raise BatchFrontierRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BatchFrontierRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchFrontierRecipeValidationError(f"{where} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise BatchFrontierRecipeValidationError(f"{where} must be a positive number")
    return result


def _relative(value: Any, where: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BatchFrontierRecipeValidationError(f"{where} must be a relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise BatchFrontierRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} or not _SAFE.fullmatch(part) for part in posix.parts):
        raise BatchFrontierRecipeValidationError(f"{where} is not a safe relative path")
    if suffix is not None and posix.suffix != suffix:
        raise BatchFrontierRecipeValidationError(f"{where} must end with {suffix}")
    return posix.as_posix()


def _artifact(value: Any, where: str) -> Artifact:
    raw = _mapping(value, where)
    _closed(raw, {"path", "sha256"}, where)
    path = _relative(raw["path"], f"{where}.path")
    sha = raw["sha256"]
    if not isinstance(sha, str) or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise BatchFrontierRecipeValidationError(f"{where}.sha256 must be a lowercase SHA-256")
    return Artifact(path, sha)


def load_recipe(path: Path) -> BatchFrontierRecipe:
    """Load and strictly validate the one authorized batch-frontier recipe."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BatchFrontierRecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
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
    _exact(_integer(platform["gpu_index"], "platform.gpu_index"), 0, "platform.gpu_index")
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
    quality_artifacts_raw = quality["artifacts"]
    if not isinstance(quality_artifacts_raw, list):
        raise BatchFrontierRecipeValidationError("quality_source.artifacts must be a list")
    quality_artifacts = tuple(
        _artifact(item, "quality_source.artifacts[]") for item in quality_artifacts_raw
    )
    _exact(
        tuple((item.path, item.sha256) for item in quality_artifacts),
        QUALITY_SOURCE_ARTIFACTS,
        "quality_source.artifacts",
    )
    quality_source = SourceBundle(
        root=_exact(
            _relative(quality["root"], "quality_source.root"),
            QUALITY_SOURCE_ROOT,
            "quality_source.root",
        ),
        expected_status=_exact(
            quality["expected_status"],
            "complete_confirmatory_passed",
            "quality_source.expected_status",
        ),
        source_git_sha=_exact(
            quality["source_git_sha"], QUALITY_SOURCE_GIT_SHA, "quality_source.source_git_sha"
        ),
        source_snapshot_sha256=_exact(
            quality["source_snapshot_sha256"],
            QUALITY_SOURCE_SNAPSHOT_SHA256,
            "quality_source.source_snapshot_sha256",
        ),
        artifacts=quality_artifacts,
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
            "checkpoint_epoch",
        },
        "training_source",
    )
    _exact(tuple(training["architectures"]), ARCHITECTURES, "training_source.architectures")
    _exact(tuple(training["training_seeds"]), TRAINING_SEEDS, "training_source.training_seeds")
    training_source = TrainingSource(
        root=_exact(
            _relative(training["root"], "training_source.root"),
            TRAINING_SOURCE_ROOT,
            "training_source.root",
        ),
        manifest_path=_exact(
            _relative(training["manifest_path"], "training_source.manifest_path", ".json"),
            "manifest.json",
            "training_source.manifest_path",
        ),
        manifest_schema=_exact(
            training["manifest_schema"], TRAINING_MANIFEST_SCHEMA, "training_source.manifest_schema"
        ),
        expected_status=_exact(
            training["expected_status"], TRAINING_COMPLETE_STATUS, "training_source.expected_status"
        ),
        architectures=ARCHITECTURES,
        training_seeds=TRAINING_SEEDS,
        checkpoint_epoch=_exact(
            _integer(training["checkpoint_epoch"], "training_source.checkpoint_epoch"),
            40,
            "training_source.checkpoint_epoch",
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
    _exact(
        dataset_raw["content_sha256"],
        QUALITY_SOURCE_CORPUS_CONTENT_SHA256,
        "dataset.content_sha256",
    )
    dataset = Dataset(
        dataset_id=_exact(dataset_raw["id"], "cvrp50-hgs10-confirmatory-seed2723", "dataset.id"),
        problem=_exact(dataset_raw["problem"], "cvrp", "dataset.problem"),
        size=_exact(_integer(dataset_raw["size"], "dataset.size", 1), 50, "dataset.size"),
        capacity=_exact(
            _number(dataset_raw["capacity"], "dataset.capacity"), 40.0, "dataset.capacity"
        ),
        max_demand=_exact(
            _integer(dataset_raw["max_demand"], "dataset.max_demand", 1), 9, "dataset.max_demand"
        ),
        num_instances=_exact(
            _integer(dataset_raw["num_instances"], "dataset.num_instances", 1),
            512,
            "dataset.num_instances",
        ),
        seed=_exact(_integer(dataset_raw["seed"], "dataset.seed"), 2723, "dataset.seed"),
        artifact=_exact(
            _relative(dataset_raw["source_artifact"], "dataset.source_artifact", ".npz"),
            "shared/cvrp50-hgs10-confirmatory-seed2723.npz",
            "dataset.source_artifact",
        ),
        forbidden_content_sha256=(),
    )

    neural_raw = _mapping(raw["neural_policy"], "neural_policy")
    _closed(
        neural_raw,
        {
            "mode_id",
            "checkpoint_epoch",
            "n_starts",
            "augmentations",
            "forced_first_actions",
            "inference_precision",
            "batch_candidates",
            "evaluation_seeds",
            "architectures",
        },
        "neural_policy",
    )
    _exact(
        tuple(neural_raw["batch_candidates"]), BATCH_CANDIDATES, "neural_policy.batch_candidates"
    )
    _exact(
        tuple(neural_raw["evaluation_seeds"]), EVALUATION_SEEDS, "neural_policy.evaluation_seeds"
    )
    arch_raw = neural_raw["architectures"]
    if not isinstance(arch_raw, list):
        raise BatchFrontierRecipeValidationError("neural_policy.architectures must be a list")
    architectures: list[ArchitecturePolicy] = []
    expected_architectures = (
        ("am", "neuro_co.core.models.encoders.am.AMEncoder", True, True),
        ("gnn", "neuro_co.core.models.encoders.gnn.GNNEncoder", True, True),
    )
    for item in arch_raw:
        value = _mapping(item, "neural_policy.architectures[]")
        _closed(
            value,
            {"id", "encoder_class", "same_pomo_objective", "same_pointer_decoder"},
            "neural_policy.architectures[]",
        )
        architectures.append(
            ArchitecturePolicy(
                str(value["id"]),
                str(value["encoder_class"]),
                _exact(
                    value["same_pomo_objective"],
                    True,
                    "neural_policy.architectures[].same_pomo_objective",
                ),
                _exact(
                    value["same_pointer_decoder"],
                    True,
                    "neural_policy.architectures[].same_pointer_decoder",
                ),
            )
        )
    _exact(
        tuple(
            (
                item.architecture,
                item.encoder_class,
                item.same_pomo_objective,
                item.same_pointer_decoder,
            )
            for item in architectures
        ),
        expected_architectures,
        "neural_policy.architectures",
    )
    neural = NeuralPolicy(
        mode_id=_exact(neural_raw["mode_id"], "pomo-50x8", "neural_policy.mode_id"),
        checkpoint_epoch=_exact(
            _integer(neural_raw["checkpoint_epoch"], "neural_policy.checkpoint_epoch"),
            40,
            "neural_policy.checkpoint_epoch",
        ),
        n_starts=_exact(
            _integer(neural_raw["n_starts"], "neural_policy.n_starts", 1),
            50,
            "neural_policy.n_starts",
        ),
        augmentations=_exact(
            _integer(neural_raw["augmentations"], "neural_policy.augmentations", 1),
            8,
            "neural_policy.augmentations",
        ),
        forced_first_actions=_exact(
            neural_raw["forced_first_actions"], True, "neural_policy.forced_first_actions"
        ),
        inference_precision=_exact(
            neural_raw["inference_precision"], "fp32", "neural_policy.inference_precision"
        ),
        batch_candidates=BATCH_CANDIDATES,
        evaluation_seeds=EVALUATION_SEEDS,
        architectures=tuple(architectures),
    )

    hgs_raw = _mapping(raw["hgs_policy"], "hgs_policy")
    _closed(
        hgs_raw,
        {"solver", "max_iterations", "scaling_factor", "collect_stats", "cpu_threads", "seeds"},
        "hgs_policy",
    )
    _exact(tuple(hgs_raw["seeds"]), HGS_SEEDS, "hgs_policy.seeds")
    hgs = HGSPolicy(
        solver=_exact(hgs_raw["solver"], "pyvrp-hgs", "hgs_policy.solver"),
        max_iterations=_exact(
            _integer(hgs_raw["max_iterations"], "hgs_policy.max_iterations", 1),
            10,
            "hgs_policy.max_iterations",
        ),
        scaling_factor=_exact(
            _integer(hgs_raw["scaling_factor"], "hgs_policy.scaling_factor", 1),
            1_000_000,
            "hgs_policy.scaling_factor",
        ),
        collect_stats=_exact(hgs_raw["collect_stats"], False, "hgs_policy.collect_stats"),
        cpu_threads=_exact(
            _integer(hgs_raw["cpu_threads"], "hgs_policy.cpu_threads", 1),
            1,
            "hgs_policy.cpu_threads",
        ),
        seeds=HGS_SEEDS,
    )

    probe_raw = _mapping(raw["capacity_probe"], "capacity_probe")
    expected_probe = {
        "measured": False,
        "checkpoint_seed": 2,
        "stop_after_first_infeasible": True,
        "require_contiguous_prefix": True,
        "record_peak_cuda_memory": True,
        "validate_routes": True,
    }
    _exact(probe_raw, expected_probe, "capacity_probe")
    probe = CapacityProbePolicy(**probe_raw)

    schedule = _mapping(raw["schedule"], "schedule")
    expected_schedule = {
        "paired_rounds": 5,
        "hgs_blocks_per_round": 1,
        "design": "balanced_deterministic_rotations_after_capacity_probe",
    }
    _exact(schedule, expected_schedule, "schedule")

    measurement = _mapping(raw["measurement"], "measurement")
    expected_measurement = {
        "backend": "windows_emi_plus_nvml_total_energy_counter",
        "fallback": False,
        "domains": ["cpu_package", "gpu"],
        "minimum_block_duration_s": 120,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "pue": 1.0,
        "report_embodied": False,
        "whole_system": False,
        "exclusive_attestation": "operator_authoritative_campaign",
        "gpu_process_lists": "diagnostic_only",
    }
    _exact(measurement, expected_measurement, "measurement")

    limits = _mapping(raw["limits"], "limits")
    expected_limits = {
        "maximum_block_walltime_s": 900,
        "campaign_attestation_max_age_s": 129600,
        "maximum_campaign_walltime_s": 129600,
    }
    _exact(limits, expected_limits, "limits")
    return BatchFrontierRecipe(
        name=_exact(raw["name"], "aet-batch-frontier-cvrp50-windows", "recipe.name"),
        host_id=_exact(platform["host_id"], "win-a4500-01", "platform.host_id"),
        gpu_index=0,
        gpu_device_id_sha256=GPU_DEVICE_ID_SHA256,
        output_root=_exact(
            _relative(raw["output_root"], "output_root"), OUTPUT_ROOT, "output_root"
        ),
        preflight_report=_exact(
            _relative(qualification["preflight_report"], "qualification.preflight_report", ".json"),
            "experiments/aet-journal/qualification/windows-batch-frontier-preflight.json",
            "qualification.preflight_report",
        ),
        quality_source=quality_source,
        training_source=training_source,
        dataset=dataset,
        neural_policy=neural,
        hgs_policy=hgs,
        capacity_probe=probe,
        paired_rounds=5,
        schedule_design=str(schedule["design"]),
        minimum_block_duration_s=120.0,
        maximum_block_walltime_s=float(limits["maximum_block_walltime_s"]),
        campaign_attestation_max_age_s=float(limits["campaign_attestation_max_age_s"]),
        maximum_campaign_walltime_s=float(limits["maximum_campaign_walltime_s"]),
    )


def expected_blocks(
    recipe: BatchFrontierRecipe,
    feasible_batches: dict[str, tuple[int, ...]],
) -> tuple[FrontierBlock, ...]:
    """Build the deterministic balanced schedule after capacity resolution."""

    if set(feasible_batches) != set(ARCHITECTURES):
        raise BatchFrontierRecipeValidationError("capacity result must cover every architecture")
    cells: list[tuple[str, int]] = []
    for batch in recipe.neural_policy.batch_candidates:
        for architecture in ARCHITECTURES:
            if batch in feasible_batches[architecture]:
                cells.append((architecture, batch))
    if any(not values for values in feasible_batches.values()):
        raise BatchFrontierRecipeValidationError("every architecture needs a feasible batch")

    blocks: list[FrontierBlock] = []
    count = len(cells)
    hgs_positions = (0, count, count // 2, count // 4, (3 * count) // 4)
    for round_index in range(recipe.paired_rounds):
        offset = (round_index // 2) * max(1, count // 3)
        ordered = cells[offset:] + cells[:offset]
        if round_index % 2:
            ordered = list(reversed(ordered))
        tokens: list[tuple[str | None, int | None]] = list(ordered)
        tokens.insert(hgs_positions[round_index], (None, None))
        for order_index, (architecture, batch_size) in enumerate(tokens):
            policy = "hgs" if architecture is None else "neural"
            suffix = "hgs" if architecture is None else f"{architecture}-b{batch_size:03d}"
            blocks.append(
                FrontierBlock(
                    round_index=round_index,
                    order_index=order_index,
                    policy=policy,
                    architecture=architecture,
                    batch_size=batch_size,
                    training_seed=TRAINING_SEEDS[round_index],
                    evaluation_seed=EVALUATION_SEEDS[round_index],
                    hgs_seed=HGS_SEEDS[round_index],
                    relative_path=f"blocks/round-{round_index:02d}/order-{order_index:02d}-{suffix}.json",
                )
            )
    return tuple(blocks)


def _source_report(source: SourceBundle, root: Path) -> dict[str, Any]:
    import hashlib

    source_root = root.joinpath(*PurePosixPath(source.root).parts)
    valid = source_root.is_dir() and not source_root.is_symlink()
    records: list[dict[str, Any]] = []
    for artifact in source.artifacts:
        path = source_root.joinpath(*PurePosixPath(artifact.path).parts)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        matches = not path.is_symlink() and digest == artifact.sha256
        records.append({"path": artifact.path, "sha256_matches": matches})
        valid = valid and matches
    return {"root": source.root, "valid": valid, "artifacts": records}


def qualify_for_execution(
    recipe: BatchFrontierRecipe,
    root: Path,
    training_source: Path | None = None,
) -> dict[str, Any]:
    """Perform read-only host, quality-source, and training-bundle checks."""

    quality = _source_report(recipe.quality_source, root)
    source_root = training_source or root.joinpath(
        *PurePosixPath(recipe.training_source.root).parts
    )
    manifest_path = source_root / recipe.training_source.manifest_path
    training_report: dict[str, Any] = {"root": str(source_root), "valid": False}
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if isinstance(manifest, dict):
            present_architectures = manifest.get("architectures")
            architecture_keys = (
                set(present_architectures) if isinstance(present_architectures, dict) else set()
            )
            training_report.update(
                {
                    "manifest_schema": manifest.get("schema_version"),
                    "status": manifest.get("status"),
                    "architectures": sorted(architecture_keys),
                    "valid": (
                        manifest.get("schema_version") == recipe.training_source.manifest_schema
                        and manifest.get("status") == recipe.training_source.expected_status
                        and architecture_keys == set(recipe.training_source.architectures)
                    ),
                }
            )
    probe = SimpleNamespace(
        preflight_report=recipe.preflight_report,
        host_id=recipe.host_id,
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    native = software._qualification_report(probe)  # type: ignore[arg-type]
    ready = (
        native["ready_to_execute"] is True
        and quality["valid"] is True
        and training_report["valid"] is True
    )
    return {
        "native_windows_counters_ready": native["ready_to_execute"],
        "native": native,
        "quality_source": quality,
        "training_source": training_report,
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
    maximum_cells = len(ARCHITECTURES) * len(BATCH_CANDIDATES)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "valid_ready_software_exploratory"
        if qualification["ready_to_execute"]
        else "valid_not_qualified",
        "classification": CLASSIFICATION,
        "name": recipe.name,
        "output_root": recipe.output_root,
        "architectures": list(ARCHITECTURES),
        "batch_candidates": list(BATCH_CANDIDATES),
        "capacity_probe_measured": False,
        "maximum_block_count": recipe.paired_rounds * (1 + maximum_cells),
        "minimum_measured_walltime_s_at_full_capacity": recipe.minimum_block_duration_s
        * recipe.paired_rounds
        * (1 + maximum_cells),
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
