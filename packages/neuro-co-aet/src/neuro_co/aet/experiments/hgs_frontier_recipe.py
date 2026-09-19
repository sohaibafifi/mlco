"""Closed selection recipe for the exploratory native-Windows HGS frontier.

This recipe selects the smallest HGS iteration budget that satisfies the
predeclared quality gate on one validation corpus. Energy is measured only as
exploratory descriptive evidence and cannot affect selection. No holdout is
defined or executed, and no result from this study is AET eligible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any

from neuro_co.aet.experiments import software_recipe as software
from neuro_co.aet.experiments.quality_recipe import HGSReference, ORToolsReference

SCHEMA_VERSION = "aet-journal-hgs-frontier/v1"
RECIPE_KIND = "aet-journal-hgs-frontier"

REPLICATION_ROOT = "experiments/aet-journal/raw/quality-replication/cvrp50-epoch40-seeds2-6"
REPLICATION_MANIFEST_SHA256 = "df47d74fe890f35103e1aa761b925ca7c082f741af8f732c1b689129e971f49f"
REPLICATION_CHECKSUMS_SHA256 = "e59d5fb1449b75f0439a06ce587e7129ba0f6e068f0a920dc7a3021e45f0202f"
REPLICATION_ASSESSMENT_SHA256 = "7cc4b1b2e257c6bc5380684776f39347105c2fd647d5445d7c33a12e3774f320"
REPLICATION_RECIPE_SHA256 = "e0617bd5fd43057763128d62005b9f12ef1e26308191a6bc077dd803d6b4160c"
REPLICATION_RUN_STATE_SHA256 = "afa50d5b27f7655e49e9fa3f35604a51ef83098969f777c11c4233db0d12a4bb"
REPLICATION_REFERENCE_LOCK_SHA256 = (
    "90a418104cc98b55405ea19413c0ac85f990f3bb7117d710afd77e9ea2a9e57d"
)
REPLICATION_GIT_SHA = "bf8d5c5dfb1a566019cd56135efa91f2e460ee6b"
REPLICATION_SOURCE_SNAPSHOT_SHA256 = (
    "3dcc187f5e875f6427e92c4c1a707461a101c8d37f1abaca029d2c03a9cd8e19"
)

GPU_DEVICE_ID_SHA256 = "686de95a427cb6c0734302fcaf694ddb9c37ba897579d9a66b75c838efeb04cf"
EXPECTED_BUDGETS = (1, 3, 10, 30, 100, 300)
EXPECTED_CANDIDATE_SEEDS = (
    20101,
    21101,
    22101,
    23101,
    24101,
    25101,
    26101,
    27101,
    28101,
    29101,
)
EXPECTED_REFERENCE_SEEDS = (11101, 12101, 13101)
EXPECTED_BUDGET_ORDERS = (
    (100, 3, 1, 300, 30, 10),
    (1, 300, 30, 10, 100, 3),
    (3, 1, 300, 30, 10, 100),
    (10, 100, 3, 1, 300, 30),
    (30, 10, 100, 3, 1, 300),
    (300, 30, 10, 100, 3, 1),
    (100, 3, 1, 300, 30, 10),
    (1, 300, 30, 10, 100, 3),
    (300, 30, 10, 100, 3, 1),
    (30, 10, 100, 3, 1, 300),
)
EXPECTED_CHECKPOINTS = (
    (
        2,
        "seeds/seed-002/training/checkpoints/epoch-040.pt",
        "ff4a912038df1d9d21fecf26c5c07f8fef4cd431ac5f6ac90ad77bfa00bdb3b0",
    ),
    (
        3,
        "seeds/seed-003/training/checkpoints/epoch-040.pt",
        "56cd106399750ce163d1aa822b1425dbc915f0d879a90f21c731c80401370bb3",
    ),
    (
        4,
        "seeds/seed-004/training/checkpoints/epoch-040.pt",
        "5c3e7e68dc26bf0deb4bb72b23351525a6c4a50739a95c6c5cf4ac25587d89e8",
    ),
    (
        5,
        "seeds/seed-005/training/checkpoints/epoch-040.pt",
        "2cd08a62e580119ad15045f65c4245d29b8e4e9e321d0efa3eee81b6c6efb7b1",
    ),
    (
        6,
        "seeds/seed-006/training/checkpoints/epoch-040.pt",
        "b9f88cd0c364d3d315f30e032fa14db04f441561f99666ab631c258209a9f3b4",
    ),
)
EXPECTED_SOURCE_PATTERNS = (
    "pyproject.toml",
    "uv.lock",
    "packages/neuro-co-core/pyproject.toml",
    "packages/neuro-co-core/src/**/*.py",
    "packages/neuro-co-problems/pyproject.toml",
    "packages/neuro-co-problems/src/**/*.py",
    "packages/neuro-co-aet/pyproject.toml",
    "packages/neuro-co-aet/src/**/*.py",
    "scripts/aet_journal/run_hgs_frontier_windows.ps1",
    "recipes/aet_journal/hgs_frontier_cvrp50_windows.yaml",
)

_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "qualification",
    "replication_source",
    "neural_prerequisite",
    "source_snapshot",
    "dataset",
    "reference",
    "candidate_frontier",
    "round_schedule",
    "measurement",
    "bootstrap",
    "quality_gate",
    "selection",
}
_CLASSIFICATION_KEYS = {
    "purpose",
    "stage",
    "scientific_use",
    "confirmatory_eligible",
    "aet_eligible",
    "cross_solver_energy_comparable",
    "holdout_executed",
}
_PLATFORM_KEYS = {
    "execution_layer",
    "host_id",
    "accelerator_label",
    "cpu_label",
    "cpu_label_source",
    "cpu_identity_independently_verified",
    "gpu_index",
    "gpu_device_id_sha256",
}
_QUALIFICATION_KEYS = {
    "preflight_report",
    "required_preflight_schema",
    "required_codecarbon_version",
    "require_windows_emi",
    "require_nvml_total_energy_counter",
}
_REPLICATION_SOURCE_KEYS = {
    "root",
    "expected_status",
    "source_git_sha",
    "source_snapshot_sha256",
    "manifest",
    "checksums",
    "assessment",
    "recipe",
    "run_state",
    "reference_lock",
    "checkpoint_epoch",
    "mode_id",
    "checkpoints",
}
_ARTIFACT_KEYS = {"path", "sha256"}
_CHECKPOINT_KEYS = {"training_seed", "path", "sha256"}
_NEURAL_PREREQUISITE_KEYS = {
    "mode_id",
    "checkpoint_epoch",
    "training_seeds",
    "evaluation_seeds",
    "metric",
    "maximum_mean_gap_pct",
    "comparison",
    "maximum_invalid_instances",
    "require_finite",
    "require_each_seed_below_threshold",
    "require_t_ucb_below_threshold",
    "require_bootstrap_ucb_below_threshold",
    "seed_t_critical_value",
    "seed_t_degrees_of_freedom",
    "bootstrap_method",
    "bootstrap_replicates",
    "bootstrap_generator",
    "bootstrap_seed",
    "bootstrap_confidence_level",
    "require_pass_before_hgs_measurement",
    "stop_without_hgs_on_nonpass",
}
_SOURCE_SNAPSHOT_KEYS = {
    "schema_version",
    "freeze_at_initialization",
    "require_match_on_resume",
    "papers_and_experiment_outputs_excluded",
    "include_patterns",
}
_DATASET_KEYS = {"problem", "size", "capacity", "max_demand", "selection"}
_SPLIT_KEYS = {"id", "num_instances", "seed", "artifact"}
_REFERENCE_KEYS = {
    "policy",
    "locked_before_candidate_evaluation",
    "hgs",
    "ortools",
}
_HGS_KEYS = {"solver", "seeds", "max_iterations"}
_ORTOOLS_KEYS = {"solver", "seed", "solution_limit", "scaling_factor"}
_CANDIDATE_KEYS = {
    "solver",
    "budget_kind",
    "budgets",
    "seeds",
    "scaling_factor",
    "collect_stats",
    "warmup_instances",
    "max_newly_completed_rounds_per_invocation",
}
_ROUND_SCHEDULE_KEYS = {"strategy", "generator", "seed", "unit", "budget_orders"}
_MEASUREMENT_KEYS = {
    "tracker_backend",
    "required_domains",
    "cpu_primary_backend",
    "gpu_backend",
    "primary_estimand",
    "energy_j_role",
    "gpu_energy_role",
    "pue",
    "fallback",
    "minimum_block_duration_s",
    "repeat_full_corpus",
    "exclusive_host_access_required",
    "exclusive_device_access_required",
    "positive_component_energy_required",
}
_BOOTSTRAP_KEYS = {
    "method",
    "replicates",
    "generator",
    "seed",
    "confidence_level",
    "seed_t_critical_value",
    "seed_t_degrees_of_freedom",
}
_QUALITY_GATE_KEYS = {
    "metric",
    "maximum_mean_gap_pct",
    "comparison",
    "maximum_invalid_instances",
    "require_finite",
    "require_each_seed_below_threshold",
    "require_t_ucb_below_threshold",
    "require_bootstrap_ucb_below_threshold",
}
_SELECTION_KEYS = {
    "rule",
    "energy_used",
    "holdout_executed",
    "require_all_budgets_evaluated",
}
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class HGSFrontierRecipeValidationError(ValueError):
    """Raised when the HGS frontier recipe violates its closed contract."""


@dataclass(frozen=True, slots=True)
class ArtifactEvidence:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class CheckpointEvidence:
    training_seed: int
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ReplicationSource:
    root: str
    expected_status: str
    source_git_sha: str
    source_snapshot_sha256: str
    manifest: ArtifactEvidence
    checksums: ArtifactEvidence
    assessment: ArtifactEvidence
    recipe: ArtifactEvidence
    run_state: ArtifactEvidence
    reference_lock: ArtifactEvidence
    checkpoint_epoch: int
    mode_id: str
    checkpoints: tuple[CheckpointEvidence, ...]


@dataclass(frozen=True, slots=True)
class NeuralPrerequisite:
    mode_id: str
    checkpoint_epoch: int
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    metric: str
    maximum_mean_gap_pct: float
    comparison: str
    maximum_invalid_instances: int
    require_finite: bool
    require_each_seed_below_threshold: bool
    require_t_ucb_below_threshold: bool
    require_bootstrap_ucb_below_threshold: bool
    seed_t_critical_value: float
    seed_t_degrees_of_freedom: int
    bootstrap_method: str
    bootstrap_replicates: int
    bootstrap_generator: str
    bootstrap_seed: int
    bootstrap_confidence_level: float
    require_pass_before_hgs_measurement: bool
    stop_without_hgs_on_nonpass: bool


@dataclass(frozen=True, slots=True)
class SourceSnapshotConfig:
    schema_version: str
    freeze_at_initialization: bool
    require_match_on_resume: bool
    papers_and_experiment_outputs_excluded: bool
    include_patterns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    split_id: str
    num_instances: int
    seed: int
    artifact: str


@dataclass(frozen=True, slots=True)
class FrontierDataset:
    problem: str
    size: int
    capacity: float
    max_demand: int
    selection: DatasetSplit


@dataclass(frozen=True, slots=True)
class FrontierReference:
    policy: str
    locked_before_candidate_evaluation: bool
    hgs: HGSReference
    ortools: ORToolsReference


@dataclass(frozen=True, slots=True)
class CandidateFrontier:
    solver: str
    budget_kind: str
    budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    scaling_factor: int
    collect_stats: bool
    warmup_instances: int
    max_newly_completed_rounds_per_invocation: int


@dataclass(frozen=True, slots=True)
class RoundSchedule:
    strategy: str
    generator: str
    seed: int
    unit: str
    budget_orders: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    tracker_backend: str
    required_domains: tuple[str, ...]
    cpu_primary_backend: str
    gpu_backend: str
    primary_estimand: str
    energy_j_role: str
    gpu_energy_role: str
    pue: float
    fallback: bool
    minimum_block_duration_s: float
    repeat_full_corpus: bool
    exclusive_host_access_required: bool
    exclusive_device_access_required: bool
    positive_component_energy_required: bool


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    method: str
    replicates: int
    generator: str
    seed: int
    confidence_level: float
    seed_t_critical_value: float
    seed_t_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class QualityGate:
    metric: str
    maximum_mean_gap_pct: float
    comparison: str
    maximum_invalid_instances: int
    require_finite: bool
    require_each_seed_below_threshold: bool
    require_t_ucb_below_threshold: bool
    require_bootstrap_ucb_below_threshold: bool


@dataclass(frozen=True, slots=True)
class SelectionRule:
    rule: str
    energy_used: bool
    holdout_executed: bool
    require_all_budgets_evaluated: bool


@dataclass(frozen=True, slots=True)
class AETHGSFrontierRecipe:
    name: str
    output_root: str
    host_id: str
    execution_layer: str
    cpu_label: str
    cpu_label_source: str
    cpu_identity_independently_verified: bool
    gpu_index: int
    expected_accelerator_label: str
    gpu_device_id_sha256: str
    preflight_report: str
    replication_source: ReplicationSource
    neural_prerequisite: NeuralPrerequisite
    source_snapshot: SourceSnapshotConfig
    dataset: FrontierDataset
    reference: FrontierReference
    candidates: CandidateFrontier
    round_schedule: RoundSchedule
    measurement: MeasurementConfig
    bootstrap: BootstrapConfig
    gate: QualityGate
    selection: SelectionRule


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HGSFrontierRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = sorted(repr(key) for key in set(value) - keys)
    if unknown:
        raise HGSFrontierRecipeValidationError(f"{where} has unknown keys: {', '.join(unknown)}")
    missing = sorted(keys - set(value))
    if missing:
        raise HGSFrontierRecipeValidationError(
            f"{where} is missing required keys: {', '.join(missing)}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if isinstance(expected, bool) and not isinstance(value, bool):
        raise HGSFrontierRecipeValidationError(f"{where} must be exactly {expected!r}")
    if value != expected:
        raise HGSFrontierRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HGSFrontierRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HGSFrontierRecipeValidationError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive finite" if positive else "non-negative finite"
        raise HGSFrontierRecipeValidationError(f"{where} must be a {qualifier} number")
    return result


def _slug(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise HGSFrontierRecipeValidationError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise HGSFrontierRecipeValidationError(f"{where} uses a Windows-reserved name")
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise HGSFrontierRecipeValidationError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, where: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HGSFrontierRecipeValidationError(
            f"{where} must be a non-empty relative path using '/' separators"
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise HGSFrontierRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise HGSFrontierRecipeValidationError(f"{where} may not contain '.' or '..'")
    for part in posix_path.parts:
        if not _SAFE_PATH_COMPONENT.fullmatch(part):
            raise HGSFrontierRecipeValidationError(
                f"{where} has an unsafe path component: {part!r}"
            )
        if part.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
            raise HGSFrontierRecipeValidationError(
                f"{where} uses a Windows-reserved path component"
            )
    if suffix is not None and posix_path.suffix != suffix:
        raise HGSFrontierRecipeValidationError(f"{where} must identify a {suffix} file")
    return posix_path.as_posix()


def _tuple_of_ints(value: Any, where: str, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise HGSFrontierRecipeValidationError(f"{where} must be a list")
    return tuple(_integer(item, f"{where}[]", minimum) for item in value)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("AET HGS frontier validation needs PyYAML") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HGSFrontierRecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise HGSFrontierRecipeValidationError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(raw, "recipe")


def _parse_artifact(value: Any, where: str, expected: tuple[str, str]) -> ArtifactEvidence:
    raw = _mapping(value, where)
    _closed(raw, _ARTIFACT_KEYS, where)
    return ArtifactEvidence(
        path=_exact(_relative_path(raw["path"], f"{where}.path"), expected[0], f"{where}.path"),
        sha256=_exact(_sha256(raw["sha256"], f"{where}.sha256"), expected[1], f"{where}.sha256"),
    )


def _parse_replication_source(value: Any) -> ReplicationSource:
    raw = _mapping(value, "recipe.replication_source")
    _closed(raw, _REPLICATION_SOURCE_KEYS, "recipe.replication_source")
    checkpoints_raw = raw["checkpoints"]
    if not isinstance(checkpoints_raw, list):
        raise HGSFrontierRecipeValidationError("replication_source.checkpoints must be a list")
    checkpoints: list[CheckpointEvidence] = []
    for index, checkpoint_value in enumerate(checkpoints_raw):
        where = f"replication_source.checkpoints[{index}]"
        checkpoint_raw = _mapping(checkpoint_value, where)
        _closed(checkpoint_raw, _CHECKPOINT_KEYS, where)
        checkpoints.append(
            CheckpointEvidence(
                training_seed=_integer(checkpoint_raw["training_seed"], f"{where}.training_seed"),
                path=_relative_path(checkpoint_raw["path"], f"{where}.path", suffix=".pt"),
                sha256=_sha256(checkpoint_raw["sha256"], f"{where}.sha256"),
            )
        )
    actual_checkpoints = tuple((item.training_seed, item.path, item.sha256) for item in checkpoints)
    _exact(actual_checkpoints, EXPECTED_CHECKPOINTS, "replication_source.checkpoints")
    return ReplicationSource(
        root=_exact(
            _relative_path(raw["root"], "replication_source.root"),
            REPLICATION_ROOT,
            "replication_source.root",
        ),
        expected_status=_exact(
            raw["expected_status"],
            "complete_replication_passed",
            "replication_source.expected_status",
        ),
        source_git_sha=_exact(
            raw["source_git_sha"],
            REPLICATION_GIT_SHA,
            "replication_source.source_git_sha",
        ),
        source_snapshot_sha256=_exact(
            _sha256(
                raw["source_snapshot_sha256"],
                "replication_source.source_snapshot_sha256",
            ),
            REPLICATION_SOURCE_SNAPSHOT_SHA256,
            "replication_source.source_snapshot_sha256",
        ),
        manifest=_parse_artifact(
            raw["manifest"],
            "replication_source.manifest",
            ("manifest.json", REPLICATION_MANIFEST_SHA256),
        ),
        checksums=_parse_artifact(
            raw["checksums"],
            "replication_source.checksums",
            ("SHA256SUMS", REPLICATION_CHECKSUMS_SHA256),
        ),
        assessment=_parse_artifact(
            raw["assessment"],
            "replication_source.assessment",
            ("replication-assessment.json", REPLICATION_ASSESSMENT_SHA256),
        ),
        recipe=_parse_artifact(
            raw["recipe"],
            "replication_source.recipe",
            ("recipe.yaml", REPLICATION_RECIPE_SHA256),
        ),
        run_state=_parse_artifact(
            raw["run_state"],
            "replication_source.run_state",
            ("run-state.json", REPLICATION_RUN_STATE_SHA256),
        ),
        reference_lock=_parse_artifact(
            raw["reference_lock"],
            "replication_source.reference_lock",
            ("reference/reference-lock.json", REPLICATION_REFERENCE_LOCK_SHA256),
        ),
        checkpoint_epoch=_exact(
            _integer(raw["checkpoint_epoch"], "replication_source.checkpoint_epoch"),
            40,
            "replication_source.checkpoint_epoch",
        ),
        mode_id=_exact(raw["mode_id"], "pomo-50x8", "replication_source.mode_id"),
        checkpoints=tuple(checkpoints),
    )


def _parse_neural_prerequisite(value: Any) -> NeuralPrerequisite:
    raw = _mapping(value, "recipe.neural_prerequisite")
    _closed(raw, _NEURAL_PREREQUISITE_KEYS, "recipe.neural_prerequisite")
    training_seeds = _tuple_of_ints(raw["training_seeds"], "neural_prerequisite.training_seeds")
    evaluation_seeds = _tuple_of_ints(
        raw["evaluation_seeds"], "neural_prerequisite.evaluation_seeds"
    )
    _exact(training_seeds, (2, 3, 4, 5, 6), "neural_prerequisite.training_seeds")
    _exact(
        evaluation_seeds,
        (82000, 83000, 84000, 85000, 86000),
        "neural_prerequisite.evaluation_seeds",
    )
    if len(training_seeds) != len(evaluation_seeds):
        raise HGSFrontierRecipeValidationError(
            "neural_prerequisite training and evaluation seeds must be paired"
        )
    return NeuralPrerequisite(
        mode_id=_exact(raw["mode_id"], "pomo-50x8", "neural_prerequisite.mode_id"),
        checkpoint_epoch=_exact(
            _integer(raw["checkpoint_epoch"], "neural_prerequisite.checkpoint_epoch"),
            40,
            "neural_prerequisite.checkpoint_epoch",
        ),
        training_seeds=training_seeds,
        evaluation_seeds=evaluation_seeds,
        metric=_exact(
            raw["metric"],
            "mean_gap_to_locked_reference_pct",
            "neural_prerequisite.metric",
        ),
        maximum_mean_gap_pct=_exact(
            _number(
                raw["maximum_mean_gap_pct"],
                "neural_prerequisite.maximum_mean_gap_pct",
            ),
            5.0,
            "neural_prerequisite.maximum_mean_gap_pct",
        ),
        comparison=_exact(
            raw["comparison"],
            "strict_less_than",
            "neural_prerequisite.comparison",
        ),
        maximum_invalid_instances=_exact(
            _integer(
                raw["maximum_invalid_instances"],
                "neural_prerequisite.maximum_invalid_instances",
            ),
            0,
            "neural_prerequisite.maximum_invalid_instances",
        ),
        require_finite=_exact(raw["require_finite"], True, "neural_prerequisite.require_finite"),
        require_each_seed_below_threshold=_exact(
            raw["require_each_seed_below_threshold"],
            True,
            "neural_prerequisite.require_each_seed_below_threshold",
        ),
        require_t_ucb_below_threshold=_exact(
            raw["require_t_ucb_below_threshold"],
            True,
            "neural_prerequisite.require_t_ucb_below_threshold",
        ),
        require_bootstrap_ucb_below_threshold=_exact(
            raw["require_bootstrap_ucb_below_threshold"],
            True,
            "neural_prerequisite.require_bootstrap_ucb_below_threshold",
        ),
        seed_t_critical_value=_exact(
            _number(
                raw["seed_t_critical_value"],
                "neural_prerequisite.seed_t_critical_value",
                positive=True,
            ),
            2.13184678632665,
            "neural_prerequisite.seed_t_critical_value",
        ),
        seed_t_degrees_of_freedom=_exact(
            _integer(
                raw["seed_t_degrees_of_freedom"],
                "neural_prerequisite.seed_t_degrees_of_freedom",
                1,
            ),
            4,
            "neural_prerequisite.seed_t_degrees_of_freedom",
        ),
        bootstrap_method=_exact(
            raw["bootstrap_method"],
            "crossed_seed_by_instance_percentile",
            "neural_prerequisite.bootstrap_method",
        ),
        bootstrap_replicates=_exact(
            _integer(
                raw["bootstrap_replicates"],
                "neural_prerequisite.bootstrap_replicates",
                1,
            ),
            10_000,
            "neural_prerequisite.bootstrap_replicates",
        ),
        bootstrap_generator=_exact(
            raw["bootstrap_generator"],
            "numpy-pcg64",
            "neural_prerequisite.bootstrap_generator",
        ),
        bootstrap_seed=_exact(
            _integer(raw["bootstrap_seed"], "neural_prerequisite.bootstrap_seed"),
            3492,
            "neural_prerequisite.bootstrap_seed",
        ),
        bootstrap_confidence_level=_exact(
            _number(
                raw["bootstrap_confidence_level"],
                "neural_prerequisite.bootstrap_confidence_level",
                positive=True,
            ),
            0.95,
            "neural_prerequisite.bootstrap_confidence_level",
        ),
        require_pass_before_hgs_measurement=_exact(
            raw["require_pass_before_hgs_measurement"],
            True,
            "neural_prerequisite.require_pass_before_hgs_measurement",
        ),
        stop_without_hgs_on_nonpass=_exact(
            raw["stop_without_hgs_on_nonpass"],
            True,
            "neural_prerequisite.stop_without_hgs_on_nonpass",
        ),
    )


def _parse_source_snapshot(value: Any) -> SourceSnapshotConfig:
    raw = _mapping(value, "recipe.source_snapshot")
    _closed(raw, _SOURCE_SNAPSHOT_KEYS, "recipe.source_snapshot")
    patterns = raw["include_patterns"]
    if not isinstance(patterns, list) or not all(isinstance(item, str) for item in patterns):
        raise HGSFrontierRecipeValidationError("source_snapshot.include_patterns must be a list")
    exact_patterns = _exact(
        tuple(patterns),
        EXPECTED_SOURCE_PATTERNS,
        "source_snapshot.include_patterns",
    )
    return SourceSnapshotConfig(
        schema_version=_exact(
            raw["schema_version"],
            "aet-hgs-frontier-source-snapshot/v1",
            "source_snapshot.schema_version",
        ),
        freeze_at_initialization=_exact(
            raw["freeze_at_initialization"], True, "source_snapshot.freeze_at_initialization"
        ),
        require_match_on_resume=_exact(
            raw["require_match_on_resume"], True, "source_snapshot.require_match_on_resume"
        ),
        papers_and_experiment_outputs_excluded=_exact(
            raw["papers_and_experiment_outputs_excluded"],
            True,
            "source_snapshot.papers_and_experiment_outputs_excluded",
        ),
        include_patterns=exact_patterns,
    )


def _parse_dataset(value: Any) -> FrontierDataset:
    raw = _mapping(value, "recipe.dataset")
    _closed(raw, _DATASET_KEYS, "recipe.dataset")
    split_raw = _mapping(raw["selection"], "dataset.selection")
    _closed(split_raw, _SPLIT_KEYS, "dataset.selection")
    split = DatasetSplit(
        split_id=_exact(
            split_raw["id"],
            "cvrp50-hgs-frontier-selection-seed2721",
            "dataset.selection.id",
        ),
        num_instances=_exact(
            _integer(split_raw["num_instances"], "dataset.selection.num_instances", 1),
            512,
            "dataset.selection.num_instances",
        ),
        seed=_exact(
            _integer(split_raw["seed"], "dataset.selection.seed"),
            2721,
            "dataset.selection.seed",
        ),
        artifact=_exact(
            _relative_path(split_raw["artifact"], "dataset.selection.artifact", suffix=".npz"),
            "shared/cvrp50-hgs-frontier-selection-seed2721.npz",
            "dataset.selection.artifact",
        ),
    )
    return FrontierDataset(
        problem=_exact(raw["problem"], "cvrp", "dataset.problem"),
        size=_exact(_integer(raw["size"], "dataset.size", 2), 50, "dataset.size"),
        capacity=_exact(
            _number(raw["capacity"], "dataset.capacity", positive=True),
            40.0,
            "dataset.capacity",
        ),
        max_demand=_exact(
            _integer(raw["max_demand"], "dataset.max_demand", 1),
            9,
            "dataset.max_demand",
        ),
        selection=split,
    )


def _parse_reference(value: Any) -> FrontierReference:
    raw = _mapping(value, "recipe.reference")
    _closed(raw, _REFERENCE_KEYS, "recipe.reference")
    hgs_raw = _mapping(raw["hgs"], "reference.hgs")
    _closed(hgs_raw, _HGS_KEYS, "reference.hgs")
    hgs_seeds = _tuple_of_ints(hgs_raw["seeds"], "reference.hgs.seeds")
    _exact(hgs_seeds, EXPECTED_REFERENCE_SEEDS, "reference.hgs.seeds")
    ortools_raw = _mapping(raw["ortools"], "reference.ortools")
    _closed(ortools_raw, _ORTOOLS_KEYS, "reference.ortools")
    return FrontierReference(
        policy=_exact(raw["policy"], "best_validated_per_instance", "reference.policy"),
        locked_before_candidate_evaluation=_exact(
            raw["locked_before_candidate_evaluation"],
            True,
            "reference.locked_before_candidate_evaluation",
        ),
        hgs=HGSReference(
            solver=_exact(hgs_raw["solver"], "pyvrp-hgs", "reference.hgs.solver"),
            seeds=hgs_seeds,
            max_iterations=_exact(
                _integer(hgs_raw["max_iterations"], "reference.hgs.max_iterations", 1),
                1000,
                "reference.hgs.max_iterations",
            ),
        ),
        ortools=ORToolsReference(
            solver=_exact(ortools_raw["solver"], "ortools-routing-gls", "reference.ortools.solver"),
            seed=_exact(
                _integer(ortools_raw["seed"], "reference.ortools.seed"),
                14101,
                "reference.ortools.seed",
            ),
            solution_limit=_exact(
                _integer(ortools_raw["solution_limit"], "reference.ortools.solution_limit", 1),
                200,
                "reference.ortools.solution_limit",
            ),
            scaling_factor=_exact(
                _integer(ortools_raw["scaling_factor"], "reference.ortools.scaling_factor", 1),
                1_000_000,
                "reference.ortools.scaling_factor",
            ),
        ),
    )


def _parse_candidates(value: Any) -> CandidateFrontier:
    raw = _mapping(value, "recipe.candidate_frontier")
    _closed(raw, _CANDIDATE_KEYS, "recipe.candidate_frontier")
    budgets = _tuple_of_ints(raw["budgets"], "candidate_frontier.budgets", 1)
    seeds = _tuple_of_ints(raw["seeds"], "candidate_frontier.seeds")
    _exact(budgets, EXPECTED_BUDGETS, "candidate_frontier.budgets")
    _exact(seeds, EXPECTED_CANDIDATE_SEEDS, "candidate_frontier.seeds")
    if set(seeds) & set(EXPECTED_REFERENCE_SEEDS):
        raise HGSFrontierRecipeValidationError(
            "candidate_frontier.seeds must be disjoint from reference.hgs.seeds"
        )
    return CandidateFrontier(
        solver=_exact(raw["solver"], "pyvrp-hgs", "candidate_frontier.solver"),
        budget_kind=_exact(raw["budget_kind"], "max_iterations", "candidate_frontier.budget_kind"),
        budgets=budgets,
        seeds=seeds,
        scaling_factor=_exact(
            _integer(raw["scaling_factor"], "candidate_frontier.scaling_factor", 1),
            1_000_000,
            "candidate_frontier.scaling_factor",
        ),
        collect_stats=_exact(raw["collect_stats"], False, "candidate_frontier.collect_stats"),
        warmup_instances=_exact(
            _integer(raw["warmup_instances"], "candidate_frontier.warmup_instances", 1),
            1,
            "candidate_frontier.warmup_instances",
        ),
        max_newly_completed_rounds_per_invocation=_exact(
            _integer(
                raw["max_newly_completed_rounds_per_invocation"],
                "candidate_frontier.max_newly_completed_rounds_per_invocation",
                1,
            ),
            1,
            "candidate_frontier.max_newly_completed_rounds_per_invocation",
        ),
    )


def _balanced_cyclic_latin_square(
    budgets: tuple[int, ...], *, seed: int, rounds: int
) -> tuple[tuple[int, ...], ...]:
    import numpy as np

    rng = np.random.default_rng(seed)
    base = tuple(int(item) for item in rng.permutation(budgets))
    rows: list[tuple[int, ...]] = []
    while len(rows) < rounds:
        shifts = tuple(int(item) for item in rng.permutation(len(base)))
        for shift in shifts:
            rows.append(base[shift:] + base[:shift])
            if len(rows) == rounds:
                break
    return tuple(rows)


def _parse_round_schedule(value: Any) -> RoundSchedule:
    raw = _mapping(value, "recipe.round_schedule")
    _closed(raw, _ROUND_SCHEDULE_KEYS, "recipe.round_schedule")
    budget_orders_raw = raw["budget_orders"]
    if not isinstance(budget_orders_raw, list):
        raise HGSFrontierRecipeValidationError("round_schedule.budget_orders must be a list")
    budget_orders = tuple(
        _tuple_of_ints(row, f"round_schedule.budget_orders[{index}]", 1)
        for index, row in enumerate(budget_orders_raw)
    )
    algorithmic_orders = _balanced_cyclic_latin_square(
        EXPECTED_BUDGETS,
        seed=3494,
        rounds=len(EXPECTED_CANDIDATE_SEEDS),
    )
    _exact(
        algorithmic_orders,
        EXPECTED_BUDGET_ORDERS,
        "internal balanced schedule algorithm",
    )
    _exact(
        budget_orders,
        algorithmic_orders,
        "round_schedule.budget_orders",
    )
    for position in range(len(EXPECTED_BUDGETS)):
        counts = {
            budget: sum(row[position] == budget for row in budget_orders)
            for budget in EXPECTED_BUDGETS
        }
        if any(count not in {1, 2} for count in counts.values()):
            raise HGSFrontierRecipeValidationError(
                "round_schedule.budget_orders must place every budget once or twice "
                "at every position"
            )
    return RoundSchedule(
        strategy=_exact(
            raw["strategy"],
            "randomized_balanced_cyclic_latin_square",
            "round_schedule.strategy",
        ),
        generator=_exact(raw["generator"], "numpy-pcg64", "round_schedule.generator"),
        seed=_exact(_integer(raw["seed"], "round_schedule.seed"), 3494, "round_schedule.seed"),
        unit=_exact(raw["unit"], "paired_seed_round", "round_schedule.unit"),
        budget_orders=budget_orders,
    )


def _parse_measurement(value: Any) -> MeasurementConfig:
    raw = _mapping(value, "recipe.measurement")
    _closed(raw, _MEASUREMENT_KEYS, "recipe.measurement")
    domains = raw["required_domains"]
    if not isinstance(domains, list):
        raise HGSFrontierRecipeValidationError("measurement.required_domains must be a list")
    exact_domains = _exact(tuple(domains), ("cpu", "gpu"), "measurement.required_domains")
    minimum_duration = _exact(
        _number(
            raw["minimum_block_duration_s"],
            "measurement.minimum_block_duration_s",
            positive=True,
        ),
        120.0,
        "measurement.minimum_block_duration_s",
    )
    return MeasurementConfig(
        tracker_backend=_exact(raw["tracker_backend"], "hwcounters", "measurement.tracker_backend"),
        required_domains=exact_domains,
        cpu_primary_backend=_exact(
            raw["cpu_primary_backend"], "windows_emi", "measurement.cpu_primary_backend"
        ),
        gpu_backend=_exact(
            raw["gpu_backend"], "nvml_total_energy_counter", "measurement.gpu_backend"
        ),
        primary_estimand=_exact(
            raw["primary_estimand"],
            "cpu_package_energy_j_per_valid_instance",
            "measurement.primary_estimand",
        ),
        energy_j_role=_exact(raw["energy_j_role"], "diagnostic_only", "measurement.energy_j_role"),
        gpu_energy_role=_exact(
            raw["gpu_energy_role"], "diagnostic_only", "measurement.gpu_energy_role"
        ),
        pue=_exact(_number(raw["pue"], "measurement.pue", positive=True), 1.0, "measurement.pue"),
        fallback=_exact(raw["fallback"], False, "measurement.fallback"),
        minimum_block_duration_s=minimum_duration,
        repeat_full_corpus=_exact(
            raw["repeat_full_corpus"], True, "measurement.repeat_full_corpus"
        ),
        exclusive_host_access_required=_exact(
            raw["exclusive_host_access_required"],
            True,
            "measurement.exclusive_host_access_required",
        ),
        exclusive_device_access_required=_exact(
            raw["exclusive_device_access_required"],
            True,
            "measurement.exclusive_device_access_required",
        ),
        positive_component_energy_required=_exact(
            raw["positive_component_energy_required"],
            True,
            "measurement.positive_component_energy_required",
        ),
    )


def _parse_bootstrap(value: Any) -> BootstrapConfig:
    raw = _mapping(value, "recipe.bootstrap")
    _closed(raw, _BOOTSTRAP_KEYS, "recipe.bootstrap")
    return BootstrapConfig(
        method=_exact(raw["method"], "crossed_seed_by_instance_percentile", "bootstrap.method"),
        replicates=_exact(
            _integer(raw["replicates"], "bootstrap.replicates", 1),
            10_000,
            "bootstrap.replicates",
        ),
        generator=_exact(raw["generator"], "numpy-pcg64", "bootstrap.generator"),
        seed=_exact(_integer(raw["seed"], "bootstrap.seed"), 3492, "bootstrap.seed"),
        confidence_level=_exact(
            _number(raw["confidence_level"], "bootstrap.confidence_level", positive=True),
            0.95,
            "bootstrap.confidence_level",
        ),
        seed_t_critical_value=_exact(
            _number(
                raw["seed_t_critical_value"],
                "bootstrap.seed_t_critical_value",
                positive=True,
            ),
            1.8331129326536335,
            "bootstrap.seed_t_critical_value",
        ),
        seed_t_degrees_of_freedom=_exact(
            _integer(
                raw["seed_t_degrees_of_freedom"],
                "bootstrap.seed_t_degrees_of_freedom",
                1,
            ),
            9,
            "bootstrap.seed_t_degrees_of_freedom",
        ),
    )


def _parse_gate(value: Any) -> QualityGate:
    raw = _mapping(value, "recipe.quality_gate")
    _closed(raw, _QUALITY_GATE_KEYS, "recipe.quality_gate")
    return QualityGate(
        metric=_exact(raw["metric"], "mean_gap_to_locked_reference_pct", "quality_gate.metric"),
        maximum_mean_gap_pct=_exact(
            _number(raw["maximum_mean_gap_pct"], "quality_gate.maximum_mean_gap_pct"),
            5.0,
            "quality_gate.maximum_mean_gap_pct",
        ),
        comparison=_exact(raw["comparison"], "strict_less_than", "quality_gate.comparison"),
        maximum_invalid_instances=_exact(
            _integer(
                raw["maximum_invalid_instances"],
                "quality_gate.maximum_invalid_instances",
            ),
            0,
            "quality_gate.maximum_invalid_instances",
        ),
        require_finite=_exact(raw["require_finite"], True, "quality_gate.require_finite"),
        require_each_seed_below_threshold=_exact(
            raw["require_each_seed_below_threshold"],
            True,
            "quality_gate.require_each_seed_below_threshold",
        ),
        require_t_ucb_below_threshold=_exact(
            raw["require_t_ucb_below_threshold"],
            True,
            "quality_gate.require_t_ucb_below_threshold",
        ),
        require_bootstrap_ucb_below_threshold=_exact(
            raw["require_bootstrap_ucb_below_threshold"],
            True,
            "quality_gate.require_bootstrap_ucb_below_threshold",
        ),
    )


def _parse_selection(value: Any) -> SelectionRule:
    raw = _mapping(value, "recipe.selection")
    _closed(raw, _SELECTION_KEYS, "recipe.selection")
    return SelectionRule(
        rule=_exact(raw["rule"], "smallest_passing_budget", "selection.rule"),
        energy_used=_exact(raw["energy_used"], False, "selection.energy_used"),
        holdout_executed=_exact(raw["holdout_executed"], False, "selection.holdout_executed"),
        require_all_budgets_evaluated=_exact(
            raw["require_all_budgets_evaluated"],
            True,
            "selection.require_all_budgets_evaluated",
        ),
    )


def load_aet_hgs_frontier_recipe(path: Path) -> AETHGSFrontierRecipe:
    """Load the immutable selection-only HGS frontier recipe."""

    raw = _load_yaml(Path(path))
    _closed(raw, _TOP_LEVEL_KEYS, "recipe")
    _exact(raw["schema_version"], SCHEMA_VERSION, "recipe.schema_version")
    _exact(raw["kind"], RECIPE_KIND, "recipe.kind")

    classification = _mapping(raw["classification"], "recipe.classification")
    _closed(classification, _CLASSIFICATION_KEYS, "recipe.classification")
    for key, expected in {
        "purpose": "software_exploratory",
        "stage": "selection",
        "scientific_use": False,
        "confirmatory_eligible": False,
        "aet_eligible": False,
        "cross_solver_energy_comparable": False,
        "holdout_executed": False,
    }.items():
        _exact(classification[key], expected, f"classification.{key}")

    platform = _mapping(raw["platform"], "recipe.platform")
    _closed(platform, _PLATFORM_KEYS, "recipe.platform")
    execution_layer = _exact(
        platform["execution_layer"], "windows-native", "platform.execution_layer"
    )
    host_id = _exact(platform["host_id"], "win-a4500-01", "platform.host_id")
    accelerator = _exact(
        platform["accelerator_label"], "NVIDIA RTX A4500", "platform.accelerator_label"
    )
    cpu_label = _exact(platform["cpu_label"], "Intel Core i7-13700", "platform.cpu_label")
    cpu_label_source = _exact(
        platform["cpu_label_source"],
        "operator_declared_launcher_argument",
        "platform.cpu_label_source",
    )
    cpu_identity_independently_verified = _exact(
        platform["cpu_identity_independently_verified"],
        False,
        "platform.cpu_identity_independently_verified",
    )
    gpu_index = _exact(
        _integer(platform["gpu_index"], "platform.gpu_index"), 0, "platform.gpu_index"
    )
    gpu_device_id = _exact(
        _sha256(platform["gpu_device_id_sha256"], "platform.gpu_device_id_sha256"),
        GPU_DEVICE_ID_SHA256,
        "platform.gpu_device_id_sha256",
    )

    qualification = _mapping(raw["qualification"], "recipe.qualification")
    _closed(qualification, _QUALIFICATION_KEYS, "recipe.qualification")
    preflight_report = _exact(
        _relative_path(
            qualification["preflight_report"],
            "qualification.preflight_report",
            suffix=".json",
        ),
        "experiments/aet-journal/qualification/windows-hgs-frontier-preflight.json",
        "qualification.preflight_report",
    )
    _exact(
        qualification["required_preflight_schema"],
        "aet-preflight/v1",
        "qualification.required_preflight_schema",
    )
    _exact(
        qualification["required_codecarbon_version"],
        "3.3.1",
        "qualification.required_codecarbon_version",
    )
    _exact(qualification["require_windows_emi"], True, "qualification.require_windows_emi")
    _exact(
        qualification["require_nvml_total_energy_counter"],
        True,
        "qualification.require_nvml_total_energy_counter",
    )

    output_root = _exact(
        _relative_path(raw["output_root"], "recipe.output_root"),
        "experiments/aet-journal/raw/hgs-frontier/cvrp50-selection-seed2721",
        "recipe.output_root",
    )
    replication_source = _parse_replication_source(raw["replication_source"])
    neural_prerequisite = _parse_neural_prerequisite(raw["neural_prerequisite"])
    source_snapshot = _parse_source_snapshot(raw["source_snapshot"])
    dataset = _parse_dataset(raw["dataset"])
    reference = _parse_reference(raw["reference"])
    candidates = _parse_candidates(raw["candidate_frontier"])
    if max(candidates.budgets) >= reference.hgs.max_iterations:
        raise HGSFrontierRecipeValidationError(
            "candidate budgets must remain below the locked HGS reference budget"
        )
    round_schedule = _parse_round_schedule(raw["round_schedule"])
    measurement = _parse_measurement(raw["measurement"])
    bootstrap = _parse_bootstrap(raw["bootstrap"])
    gate = _parse_gate(raw["quality_gate"])
    selection = _parse_selection(raw["selection"])

    return AETHGSFrontierRecipe(
        name=_exact(
            raw["name"],
            "aet-hgs-frontier-cvrp50-selection-windows",
            "recipe.name",
        ),
        output_root=output_root,
        host_id=host_id,
        execution_layer=execution_layer,
        cpu_label=cpu_label,
        cpu_label_source=cpu_label_source,
        cpu_identity_independently_verified=cpu_identity_independently_verified,
        gpu_index=gpu_index,
        expected_accelerator_label=accelerator,
        gpu_device_id_sha256=gpu_device_id,
        preflight_report=preflight_report,
        replication_source=replication_source,
        neural_prerequisite=neural_prerequisite,
        source_snapshot=source_snapshot,
        dataset=dataset,
        reference=reference,
        candidates=candidates,
        round_schedule=round_schedule,
        measurement=measurement,
        bootstrap=bootstrap,
        gate=gate,
        selection=selection,
    )


def _repository_root() -> Path:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return Path.cwd().resolve()
    return Path(value).resolve()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def _replication_source_qualification(
    recipe: AETHGSFrontierRecipe, repository_root: Path
) -> dict[str, Any]:
    source = recipe.replication_source
    source_root = repository_root / source.root
    artifacts = {
        "manifest": source.manifest,
        "checksums": source.checksums,
        "assessment": source.assessment,
        "recipe": source.recipe,
        "run_state": source.run_state,
        "reference_lock": source.reference_lock,
    }
    artifact_checks: dict[str, dict[str, Any]] = {}
    for name, artifact in artifacts.items():
        path = source_root / artifact.path
        exists = path.is_file()
        observed = _hash_file(path) if exists else None
        artifact_checks[name] = {
            "path": path.as_posix(),
            "exists": exists,
            "expected_sha256": artifact.sha256,
            "observed_sha256": observed,
            "sha256_matches": observed == artifact.sha256,
        }

    checkpoint_checks: list[dict[str, Any]] = []
    for checkpoint in source.checkpoints:
        path = source_root / checkpoint.path
        exists = path.is_file()
        observed = _hash_file(path) if exists else None
        checkpoint_checks.append(
            {
                "training_seed": checkpoint.training_seed,
                "path": path.as_posix(),
                "exists": exists,
                "expected_sha256": checkpoint.sha256,
                "observed_sha256": observed,
                "sha256_matches": observed == checkpoint.sha256,
            }
        )

    semantic_checks: dict[str, bool] = {}
    semantic_error: str | None = None
    if artifact_checks["manifest"]["sha256_matches"]:
        try:
            manifest = _load_json_mapping(source_root / source.manifest.path)
            manifest_source = manifest.get("source")
            if not isinstance(manifest_source, dict):
                manifest_source = {}
            snapshot = manifest_source.get("scoped_source_snapshot")
            if not isinstance(snapshot, dict):
                snapshot = {}
            semantic_checks = {
                "manifest_status_matches": manifest.get("status") == source.expected_status,
                "manifest_git_sha_matches": (
                    manifest_source.get("git_sha") == source.source_git_sha
                ),
                "manifest_source_snapshot_matches": (
                    snapshot.get("sha256") == source.source_snapshot_sha256
                ),
            }
            if artifact_checks["assessment"]["sha256_matches"]:
                assessment = _load_json_mapping(source_root / source.assessment.path)
                semantic_checks["assessment_status_matches"] = (
                    assessment.get("status") == source.expected_status
                )
            if artifact_checks["run_state"]["sha256_matches"]:
                run_state = _load_json_mapping(source_root / source.run_state.path)
                run_snapshot = run_state.get("source_snapshot")
                if not isinstance(run_snapshot, dict):
                    run_snapshot = {}
                semantic_checks.update(
                    {
                        "run_state_status_matches": (
                            run_state.get("status") == source.expected_status
                        ),
                        "run_state_git_sha_matches": (
                            run_state.get("git_sha") == source.source_git_sha
                        ),
                        "run_state_source_snapshot_matches": (
                            run_snapshot.get("sha256") == source.source_snapshot_sha256
                        ),
                        "run_state_manifest_hash_matches": (
                            run_state.get("manifest_sha256") == source.manifest.sha256
                        ),
                        "run_state_checksums_hash_matches": (
                            run_state.get("checksums_sha256") == source.checksums.sha256
                        ),
                        "run_state_recipe_hash_matches": (
                            run_state.get("recipe_sha256") == source.recipe.sha256
                        ),
                        "run_state_reference_lock_hash_matches": (
                            run_state.get("reference_lock_sha256") == source.reference_lock.sha256
                        ),
                        "run_state_reference_was_locked_first": (
                            run_state.get(
                                "reference_locked_before_fresh_seed_training_and_evaluation"
                            )
                            is True
                        ),
                        "run_state_completed_seeds_match": (
                            run_state.get("completed_seeds") == [2, 3, 4, 5, 6]
                        ),
                    }
                )
            if artifact_checks["reference_lock"]["sha256_matches"]:
                reference_lock = _load_json_mapping(source_root / source.reference_lock.path)
                semantic_checks.update(
                    {
                        "reference_lock_status_matches": (reference_lock.get("status") == "locked"),
                        "reference_lock_policy_matches": (
                            reference_lock.get("policy") == "best_validated_per_instance"
                        ),
                        "reference_lock_precedes_fresh_evaluation": (
                            reference_lock.get("locked_before_fresh_seed_training_and_evaluation")
                            is True
                        ),
                    }
                )
            checksum_path = source_root / source.checksums.path
            if artifact_checks["checksums"]["sha256_matches"]:
                inventory: dict[str, str] = {}
                for line in checksum_path.read_text(encoding="utf-8").splitlines():
                    digest, separator, member = line.partition("  ")
                    if separator:
                        inventory[member.replace("\\", "/")] = digest
                semantic_checks["checkpoint_inventory_matches"] = all(
                    inventory.get(checkpoint.path) == checkpoint.sha256
                    for checkpoint in source.checkpoints
                )
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"

    ready = bool(
        all(item["sha256_matches"] for item in artifact_checks.values())
        and checkpoint_checks
        and all(item["sha256_matches"] for item in checkpoint_checks)
        and semantic_checks
        and all(semantic_checks.values())
        and semantic_error is None
    )
    return {
        "root": source_root.as_posix(),
        "artifact_checks": artifact_checks,
        "checkpoint_checks": checkpoint_checks,
        "semantic_checks": semantic_checks,
        "error": semantic_error,
        "ready": ready,
    }


def _software_qualification_report(recipe: Any) -> dict[str, Any]:
    return software._qualification_report(recipe)


def _preflight_cpu_qualification(path: Path, expected_cpu_label: str) -> dict[str, Any]:
    preflight_cpu_label: Any = None
    error: str | None = None
    try:
        report = _load_json_mapping(path)
        system = report.get("system")
        if isinstance(system, dict):
            preflight_cpu_label = system.get("cpu_label")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    matches = bool(preflight_cpu_label == expected_cpu_label and error is None)
    return {
        "expected_cpu_label": expected_cpu_label,
        "preflight_cpu_label": preflight_cpu_label,
        "declared_cpu_label_matches": matches,
        "cpu_label_source": "operator_declared_launcher_argument",
        "cpu_identity_independently_verified": False,
        "error": error,
    }


def runtime_qualification(
    recipe: AETHGSFrontierRecipe,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Qualify the dedicated preflight and frozen replication source read-only."""

    root = (repository_root or _repository_root()).resolve()
    preflight_path = (root / recipe.preflight_report).resolve()
    preflight = _software_qualification_report(
        SimpleNamespace(
            preflight_report=str(preflight_path),
            host_id=recipe.host_id,
            cpu_label=recipe.cpu_label,
            gpu_index=recipe.gpu_index,
            gpu_device_id_sha256=recipe.gpu_device_id_sha256,
        )
    )
    cpu_identity = _preflight_cpu_qualification(preflight_path, recipe.cpu_label)
    preflight = {**preflight, **cpu_identity}
    replication = _replication_source_qualification(recipe, root)
    source_definition_valid = bool(
        recipe.source_snapshot.freeze_at_initialization
        and recipe.source_snapshot.require_match_on_resume
        and recipe.source_snapshot.papers_and_experiment_outputs_excluded
        and "packages/neuro-co-aet/src/**/*.py" in recipe.source_snapshot.include_patterns
    )
    ready = bool(
        preflight.get("ready_to_execute") is True
        and cpu_identity["declared_cpu_label_matches"]
        and replication["ready"]
        and source_definition_valid
    )
    return {
        "preflight": preflight,
        "replication_source": replication,
        "source_snapshot_definition_valid": source_definition_valid,
        "ready_to_execute": ready,
    }


def dry_run(path: Path) -> dict[str, Any]:
    """Validate and report the complete selection plan without any writes."""

    recipe = load_aet_hgs_frontier_recipe(path)
    qualification = runtime_qualification(recipe)
    candidate_runs = len(recipe.candidates.budgets) * len(recipe.candidates.seeds)
    report = {
        "status": (
            "valid_ready_hgs_frontier_selection"
            if qualification["ready_to_execute"]
            else "valid_not_qualified"
        ),
        "schema_version": SCHEMA_VERSION,
        "name": recipe.name,
        "purpose": "software_exploratory",
        "stage": "selection",
        "scientific_use": False,
        "confirmatory_eligible": False,
        "aet_eligible": False,
        "cross_solver_energy_comparable": False,
        "holdout_executed": False,
        "host_id": recipe.host_id,
        "execution_layer": recipe.execution_layer,
        "cpu_label": recipe.cpu_label,
        "cpu_label_source": recipe.cpu_label_source,
        "cpu_identity_independently_verified": (recipe.cpu_identity_independently_verified),
        "gpu_index": recipe.gpu_index,
        "selection_instances": recipe.dataset.selection.num_instances,
        "candidate_budgets": list(recipe.candidates.budgets),
        "candidate_seeds": list(recipe.candidates.seeds),
        "candidate_runs": candidate_runs,
        "neural_prerequisite_mode": recipe.neural_prerequisite.mode_id,
        "neural_prerequisite_training_seeds": list(recipe.neural_prerequisite.training_seeds),
        "neural_prerequisite_evaluation_seeds": list(recipe.neural_prerequisite.evaluation_seeds),
        "neural_prerequisite_required_before_hgs_measurement": (
            recipe.neural_prerequisite.require_pass_before_hgs_measurement
        ),
        "reference_hgs_solves": (
            recipe.dataset.selection.num_instances * len(recipe.reference.hgs.seeds)
        ),
        "reference_ortools_solves": recipe.dataset.selection.num_instances,
        "round_schedule_seed": recipe.round_schedule.seed,
        "minimum_block_duration_s": recipe.measurement.minimum_block_duration_s,
        "primary_estimand": recipe.measurement.primary_estimand,
        "energy_used_for_selection": recipe.selection.energy_used,
        "bootstrap_replicates": recipe.bootstrap.replicates,
        "bootstrap_seed": recipe.bootstrap.seed,
        "quality_gate_comparison": recipe.gate.comparison,
        "quality_gate_maximum_mean_gap_pct": recipe.gate.maximum_mean_gap_pct,
        "selection_rule": recipe.selection.rule,
        "max_newly_completed_rounds_per_invocation": (
            recipe.candidates.max_newly_completed_rounds_per_invocation
        ),
        "output_root": recipe.output_root,
        "qualification": qualification,
        "ready_to_execute": qualification["ready_to_execute"],
        "writes_performed": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    report = dry_run(args.recipe)
    return 2 if args.require_ready and not report["ready_to_execute"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AETHGSFrontierRecipe",
    "ArtifactEvidence",
    "BootstrapConfig",
    "CandidateFrontier",
    "CheckpointEvidence",
    "DatasetSplit",
    "FrontierDataset",
    "FrontierReference",
    "HGSFrontierRecipeValidationError",
    "MeasurementConfig",
    "NeuralPrerequisite",
    "QualityGate",
    "ReplicationSource",
    "RoundSchedule",
    "SelectionRule",
    "SourceSnapshotConfig",
    "dry_run",
    "load_aet_hgs_frontier_recipe",
    "runtime_qualification",
]
