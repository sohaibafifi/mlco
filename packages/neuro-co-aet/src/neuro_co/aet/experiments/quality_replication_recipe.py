"""Closed recipe and preflight for the AET journal quality replication.

The replication records no energy and computes no AET. It evaluates five
fresh training seeds on one common exploratory development corpus while the
seed-1 discovery bundle is retained only as provenance.
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
from typing import Any, cast
from zipfile import BadZipFile, ZipFile

from neuro_co.aet.experiments.quality_recipe import (
    AETQualityRecipe,
    EpochScheduler,
    EvaluationMode,
    HGSReference,
    ORToolsReference,
    QualityModel,
    ReferenceConfig,
)
from neuro_co.aet.experiments.quality_recipe import (
    runtime_qualification as quality_runtime_qualification,
)

SCHEMA_VERSION = "aet-journal-quality-replication/v1"
RECIPE_KIND = "aet-journal-quality-replication"

BASE_RECIPE_PATH = "recipes/aet_journal/quality_pilot_cvrp50_windows.yaml"
BASE_RECIPE_SEMANTIC_SHA256 = "e79cdb04e533b81f0efe330dd787b2a78a62207862c4be5c1bd0f2d9a37b433d"
DISCOVERY_ZIP_PATH = "windows-a4500-quality-pilot.zip"
DISCOVERY_ZIP_SHA256 = "b2f258d702fdd642c47f570ea45c20b1d20610cb0899757a24821f4f0f6d1e94"
DISCOVERY_MANIFEST_SHA256 = "35f38234cd79decaca37afba45d9f09a3c57388fc844a8fa949f43776078ddd6"
DISCOVERY_CHECKSUMS_SHA256 = "4dabc799422d8bab995b1d46ea3efe410975a960d98f1337a23e9d03bb8e9c3b"
DISCOVERY_CHECKPOINT_SHA256 = "116dae69aa8e17e95a6a74a11a134eae4717063ee872cf5746b6ef5678e0ea66"
DISCOVERY_GIT_SHA = "746e339e51ef02fe50ec4e15a673888a32b950b7"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "base_recipe",
    "discovery_zip",
    "dataset",
    "reference",
    "model",
    "training",
    "evaluation",
    "bootstrap",
    "quality_gate",
}
_CLASSIFICATION_KEYS = {
    "purpose",
    "scientific_use",
    "aet_eligible",
    "energy_measurement",
    "exclusive_access_required",
}
_PLATFORM_KEYS = {"execution_layer", "host_id", "accelerator_label", "gpu_index"}
_BASE_RECIPE_KEYS = {"path", "semantic_sha256"}
_DISCOVERY_KEYS = {
    "path",
    "sha256",
    "manifest_sha256",
    "checksums_sha256",
    "expected_status",
    "git_sha",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "expected_primary_mode",
    "use",
    "evaluated",
    "included_in_aggregation",
}
_DATASET_KEYS = {"problem", "size", "capacity", "max_demand", "development"}
_SPLIT_KEYS = {"id", "num_instances", "seed", "artifact"}
_REFERENCE_KEYS = {"policy", "locked_before_neural_evaluation", "hgs", "ortools"}
_HGS_KEYS = {"solver", "seeds", "max_iterations"}
_ORTOOLS_KEYS = {"solver", "seed", "solution_limit", "scaling_factor"}
_MODEL_KEYS = {
    "architecture",
    "expected_parameter_count",
    "hidden_dim",
    "num_layers",
    "num_heads",
    "dynamic_context",
}
_TRAINING_KEYS = {
    "algorithm",
    "seeds",
    "epochs",
    "fixed_epoch",
    "instances_per_epoch",
    "batch_size",
    "n_starts",
    "precision",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "gradient_clip",
    "scheduler",
    "checkpoint_epochs",
    "max_walltime_per_invocation_s",
    "max_newly_completed_seeds_per_invocation",
}
_SCHEDULER_KEYS = {"name", "milestones", "gamma"}
_EVALUATION_KEYS = {
    "primary_mode_id",
    "secondary_mode_id",
    "diagnostic_mode_id",
    "modes",
}
_MODE_KEYS = {
    "id",
    "role",
    "n_starts",
    "augmentations",
    "forced_first_actions",
    "batch_size",
    "inference_precision",
}
_BOOTSTRAP_KEYS = {
    "method",
    "replicates",
    "generator",
    "seed",
    "confidence_level",
    "shared_resamples_across_pomo_modes",
    "seed_t_critical_value",
    "seed_t_degrees_of_freedom",
}
_GATE_KEYS = {
    "metric",
    "primary_mode_id",
    "maximum_mean_gap_pct",
    "maximum_invalid_instances",
    "require_finite",
    "require_each_seed_below_threshold",
    "require_t_ucb_below_threshold",
    "require_bootstrap_ucb_below_threshold",
    "ordered_secondary_only_if_primary_passes",
}
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class QualityReplicationRecipeValidationError(ValueError):
    """Raised when a replication recipe violates its closed schema."""


@dataclass(frozen=True, slots=True)
class BaseRecipeProvenance:
    path: str
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class DiscoveryZipProvenance:
    path: str
    sha256: str
    manifest_sha256: str
    checksums_sha256: str
    expected_status: str
    git_sha: str
    checkpoint_epoch: int
    checkpoint_sha256: str
    expected_primary_mode: str
    use: str
    evaluated: bool
    included_in_aggregation: bool


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    split_id: str
    num_instances: int
    seed: int
    artifact: str


@dataclass(frozen=True, slots=True)
class ReplicationDataset:
    problem: str
    size: int
    capacity: float
    max_demand: int
    development: DatasetSplit


@dataclass(frozen=True, slots=True)
class ReplicationTraining:
    algorithm: str
    seeds: tuple[int, ...]
    epochs: int
    fixed_epoch: int
    instances_per_epoch: int
    batch_size: int
    n_starts: int
    precision: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    scheduler: EpochScheduler
    checkpoint_epochs: tuple[int, ...]
    max_walltime_per_invocation_s: float
    max_newly_completed_seeds_per_invocation: int


@dataclass(frozen=True, slots=True)
class ReplicationEvaluation:
    modes: tuple[EvaluationMode, ...]
    primary_mode_id: str
    secondary_mode_id: str
    diagnostic_mode_id: str


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    method: str
    replicates: int
    generator: str
    seed: int
    confidence_level: float
    shared_resamples_across_pomo_modes: bool
    seed_t_critical_value: float
    seed_t_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class ReplicationQualityGate:
    metric: str
    primary_mode_id: str
    maximum_mean_gap_pct: float
    maximum_invalid_instances: int
    require_finite: bool
    require_each_seed_below_threshold: bool
    require_t_ucb_below_threshold: bool
    require_bootstrap_ucb_below_threshold: bool
    ordered_secondary_only_if_primary_passes: bool


@dataclass(frozen=True, slots=True)
class AETQualityReplicationRecipe:
    name: str
    output_root: str
    host_id: str
    execution_layer: str
    gpu_index: int
    expected_accelerator_label: str
    base_recipe: BaseRecipeProvenance
    discovery_zip: DiscoveryZipProvenance
    dataset: ReplicationDataset
    reference: ReferenceConfig
    model: QualityModel
    training: ReplicationTraining
    evaluation: ReplicationEvaluation
    bootstrap: BootstrapConfig
    gate: ReplicationQualityGate

    @property
    def seeds(self) -> tuple[int, ...]:
        return self.training.seeds

    @property
    def fixed_epoch(self) -> int:
        return self.training.fixed_epoch

    @property
    def primary_mode_id(self) -> str:
        return self.evaluation.primary_mode_id

    @property
    def secondary_mode_id(self) -> str:
        return self.evaluation.secondary_mode_id

    @property
    def diagnostic_mode_id(self) -> str:
        return self.evaluation.diagnostic_mode_id


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualityReplicationRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = sorted(repr(key) for key in set(value) - keys)
    if unknown:
        raise QualityReplicationRecipeValidationError(
            f"{where} has unknown keys: {', '.join(unknown)}"
        )
    missing = sorted(keys - set(value))
    if missing:
        raise QualityReplicationRecipeValidationError(
            f"{where} is missing required keys: {', '.join(missing)}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if isinstance(expected, bool) and not isinstance(value, bool):
        raise QualityReplicationRecipeValidationError(f"{where} must be exactly {expected!r}")
    if value != expected:
        raise QualityReplicationRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualityReplicationRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityReplicationRecipeValidationError(f"{where} must be a finite number")
    result = float(value)
    invalid = not math.isfinite(result) or (result <= 0 if positive else result < 0)
    if invalid:
        qualifier = "positive finite" if positive else "non-negative finite"
        raise QualityReplicationRecipeValidationError(f"{where} must be a {qualifier} number")
    return result


def _slug(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise QualityReplicationRecipeValidationError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise QualityReplicationRecipeValidationError(f"{where} uses a Windows-reserved name")
    return value


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise QualityReplicationRecipeValidationError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _relative_path(value: Any, where: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise QualityReplicationRecipeValidationError(
            f"{where} must be a non-empty relative path using '/' separators"
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise QualityReplicationRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise QualityReplicationRecipeValidationError(f"{where} may not contain '.' or '..'")
    for part in posix_path.parts:
        _slug(part, f"{where} component")
    if suffix is not None and posix_path.suffix != suffix:
        raise QualityReplicationRecipeValidationError(f"{where} must identify a {suffix} file")
    return posix_path.as_posix()


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("AET quality replication validation needs PyYAML") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QualityReplicationRecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise QualityReplicationRecipeValidationError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(raw, "recipe")


def canonical_yaml_sha256(path: Path) -> str:
    """Hash a YAML document after deterministic semantic JSON normalization."""

    serialized = json.dumps(
        _load_yaml(path),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _parse_split(value: Any) -> DatasetSplit:
    raw = _mapping(value, "dataset.development")
    _closed(raw, _SPLIT_KEYS, "dataset.development")
    artifact = _relative_path(raw["artifact"], "dataset.development.artifact", suffix=".npz")
    if not artifact.startswith("shared/"):
        raise QualityReplicationRecipeValidationError(
            "dataset.development.artifact must stay under shared/"
        )
    return DatasetSplit(
        split_id=_exact(raw["id"], "cvrp50-development-seed2711", "dataset.development.id"),
        num_instances=_exact(
            _integer(raw["num_instances"], "dataset.development.num_instances", 1),
            512,
            "dataset.development.num_instances",
        ),
        seed=_exact(
            _integer(raw["seed"], "dataset.development.seed"),
            2711,
            "dataset.development.seed",
        ),
        artifact=_exact(
            artifact,
            "shared/cvrp50-development-seed2711.npz",
            "dataset.development.artifact",
        ),
    )


def _parse_reference(value: Any) -> ReferenceConfig:
    raw = _mapping(value, "recipe.reference")
    _closed(raw, _REFERENCE_KEYS, "recipe.reference")
    hgs_raw = _mapping(raw["hgs"], "reference.hgs")
    _closed(hgs_raw, _HGS_KEYS, "reference.hgs")
    seeds_raw = hgs_raw["seeds"]
    if not isinstance(seeds_raw, list):
        raise QualityReplicationRecipeValidationError("reference.hgs.seeds must be a list")
    hgs_seeds = tuple(_integer(item, "reference.hgs.seeds[]") for item in seeds_raw)
    _exact(hgs_seeds, (10101, 10201, 10301), "reference.hgs.seeds")
    ortools_raw = _mapping(raw["ortools"], "reference.ortools")
    _closed(ortools_raw, _ORTOOLS_KEYS, "reference.ortools")
    return ReferenceConfig(
        policy=_exact(raw["policy"], "best_validated_per_instance", "reference.policy"),
        locked_before_neural_evaluation=_exact(
            raw["locked_before_neural_evaluation"],
            True,
            "reference.locked_before_neural_evaluation",
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
            solver=_exact(
                ortools_raw["solver"],
                "ortools-routing-gls",
                "reference.ortools.solver",
            ),
            seed=_exact(
                _integer(ortools_raw["seed"], "reference.ortools.seed"),
                10401,
                "reference.ortools.seed",
            ),
            solution_limit=_exact(
                _integer(
                    ortools_raw["solution_limit"],
                    "reference.ortools.solution_limit",
                    1,
                ),
                200,
                "reference.ortools.solution_limit",
            ),
            scaling_factor=_exact(
                _integer(
                    ortools_raw["scaling_factor"],
                    "reference.ortools.scaling_factor",
                    1,
                ),
                1_000_000,
                "reference.ortools.scaling_factor",
            ),
        ),
    )


def _parse_model(value: Any) -> QualityModel:
    raw = _mapping(value, "recipe.model")
    _closed(raw, _MODEL_KEYS, "recipe.model")
    return QualityModel(
        architecture=_exact(raw["architecture"], "mlco-am", "model.architecture"),
        expected_parameter_count=_exact(
            _integer(raw["expected_parameter_count"], "model.expected_parameter_count", 1),
            741_248,
            "model.expected_parameter_count",
        ),
        hidden_dim=_exact(
            _integer(raw["hidden_dim"], "model.hidden_dim", 1), 128, "model.hidden_dim"
        ),
        num_layers=_exact(
            _integer(raw["num_layers"], "model.num_layers", 1), 3, "model.num_layers"
        ),
        num_heads=_exact(_integer(raw["num_heads"], "model.num_heads", 1), 8, "model.num_heads"),
        dynamic_context=_exact(
            raw["dynamic_context"],
            "remaining_capacity_normalized",
            "model.dynamic_context",
        ),
    )


def _tuple_of_ints(value: Any, where: str, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise QualityReplicationRecipeValidationError(f"{where} must be a list")
    return tuple(_integer(item, f"{where}[]", minimum) for item in value)


def _parse_training(value: Any) -> ReplicationTraining:
    raw = _mapping(value, "recipe.training")
    _closed(raw, _TRAINING_KEYS, "recipe.training")
    seeds = _tuple_of_ints(raw["seeds"], "training.seeds")
    _exact(seeds, (2, 3, 4, 5, 6), "training.seeds")
    scheduler_raw = _mapping(raw["scheduler"], "training.scheduler")
    _closed(scheduler_raw, _SCHEDULER_KEYS, "training.scheduler")
    milestones = _tuple_of_ints(scheduler_raw["milestones"], "training.scheduler.milestones", 1)
    _exact(milestones, (36, 38), "training.scheduler.milestones")
    checkpoints = _tuple_of_ints(raw["checkpoint_epochs"], "training.checkpoint_epochs")
    _exact(
        checkpoints,
        (0, 10, 20, 30, 36, 37, 38, 39, 40),
        "training.checkpoint_epochs",
    )
    return ReplicationTraining(
        algorithm=_exact(raw["algorithm"], "pomo", "training.algorithm"),
        seeds=seeds,
        epochs=_exact(_integer(raw["epochs"], "training.epochs", 1), 40, "training.epochs"),
        fixed_epoch=_exact(
            _integer(raw["fixed_epoch"], "training.fixed_epoch", 1),
            40,
            "training.fixed_epoch",
        ),
        instances_per_epoch=_exact(
            _integer(raw["instances_per_epoch"], "training.instances_per_epoch", 1),
            20_000,
            "training.instances_per_epoch",
        ),
        batch_size=_exact(
            _integer(raw["batch_size"], "training.batch_size", 1),
            256,
            "training.batch_size",
        ),
        n_starts=_exact(
            _integer(raw["n_starts"], "training.n_starts", 1),
            50,
            "training.n_starts",
        ),
        precision=_exact(raw["precision"], "bf16", "training.precision"),
        optimizer=_exact(raw["optimizer"], "adam", "training.optimizer"),
        learning_rate=_exact(
            _number(raw["learning_rate"], "training.learning_rate", positive=True),
            0.0001,
            "training.learning_rate",
        ),
        weight_decay=_exact(
            _number(raw["weight_decay"], "training.weight_decay"),
            0.000001,
            "training.weight_decay",
        ),
        gradient_clip=_exact(
            _number(raw["gradient_clip"], "training.gradient_clip"),
            1.0,
            "training.gradient_clip",
        ),
        scheduler=EpochScheduler(
            name=_exact(scheduler_raw["name"], "multistep-epoch", "training.scheduler.name"),
            milestones=milestones,
            gamma=_exact(
                _number(
                    scheduler_raw["gamma"],
                    "training.scheduler.gamma",
                    positive=True,
                ),
                0.1,
                "training.scheduler.gamma",
            ),
        ),
        checkpoint_epochs=checkpoints,
        max_walltime_per_invocation_s=_exact(
            _number(
                raw["max_walltime_per_invocation_s"],
                "training.max_walltime_per_invocation_s",
                positive=True,
            ),
            3600.0,
            "training.max_walltime_per_invocation_s",
        ),
        max_newly_completed_seeds_per_invocation=_exact(
            _integer(
                raw["max_newly_completed_seeds_per_invocation"],
                "training.max_newly_completed_seeds_per_invocation",
                1,
            ),
            1,
            "training.max_newly_completed_seeds_per_invocation",
        ),
    )


def _parse_evaluation(value: Any) -> ReplicationEvaluation:
    raw = _mapping(value, "recipe.evaluation")
    _closed(raw, _EVALUATION_KEYS, "recipe.evaluation")
    expected_ids = ("pomo-50x8", "pomo-50x1", "greedy-1x1")
    expected_roles = ("primary", "secondary", "diagnostic")
    expected_configs = (
        (50, 8, True, 4),
        (50, 1, True, 16),
        (1, 1, False, 128),
    )
    modes_raw = raw["modes"]
    if not isinstance(modes_raw, list):
        raise QualityReplicationRecipeValidationError("evaluation.modes must be a list")
    modes: list[EvaluationMode] = []
    roles: list[str] = []
    for index, mode_value in enumerate(modes_raw):
        where = f"evaluation.modes[{index}]"
        mode_raw = _mapping(mode_value, where)
        _closed(mode_raw, _MODE_KEYS, where)
        n_starts = _integer(mode_raw["n_starts"], f"{where}.n_starts", 1)
        forced = mode_raw["forced_first_actions"]
        if not isinstance(forced, bool):
            raise QualityReplicationRecipeValidationError(
                f"{where}.forced_first_actions must be boolean"
            )
        if (n_starts > 1) != forced:
            raise QualityReplicationRecipeValidationError(
                f"{where} must force first actions exactly when n_starts > 1"
            )
        augmentations = _integer(mode_raw["augmentations"], f"{where}.augmentations", 1)
        if augmentations not in {1, 8}:
            raise QualityReplicationRecipeValidationError(f"{where}.augmentations must be 1 or 8")
        mode = EvaluationMode(
            mode_id=_slug(mode_raw["id"], f"{where}.id"),
            n_starts=n_starts,
            augmentations=augmentations,
            forced_first_actions=forced,
            batch_size=_integer(mode_raw["batch_size"], f"{where}.batch_size", 1),
            inference_precision=_exact(
                mode_raw["inference_precision"],
                "fp32",
                f"{where}.inference_precision",
            ),
        )
        if index >= len(expected_ids) or mode.mode_id != expected_ids[index]:
            raise QualityReplicationRecipeValidationError(
                "evaluation mode order must be exactly " + repr(expected_ids)
            )
        role = _slug(mode_raw["role"], f"{where}.role")
        if role != expected_roles[index]:
            raise QualityReplicationRecipeValidationError(
                "evaluation mode roles must be exactly " + repr(expected_roles)
            )
        expected_config = expected_configs[index]
        actual_config = (
            mode.n_starts,
            mode.augmentations,
            mode.forced_first_actions,
            mode.batch_size,
        )
        if expected_config is None or actual_config != expected_config:
            raise QualityReplicationRecipeValidationError(
                f"{where} configuration must be exactly {expected_config!r}"
            )
        modes.append(mode)
        roles.append(role)
    _exact(tuple(mode.mode_id for mode in modes), expected_ids, "evaluation mode order")
    _exact(tuple(roles), expected_roles, "evaluation mode roles")
    return ReplicationEvaluation(
        modes=tuple(modes),
        primary_mode_id=_exact(
            raw["primary_mode_id"], expected_ids[0], "evaluation.primary_mode_id"
        ),
        secondary_mode_id=_exact(
            raw["secondary_mode_id"], expected_ids[1], "evaluation.secondary_mode_id"
        ),
        diagnostic_mode_id=_exact(
            raw["diagnostic_mode_id"], expected_ids[2], "evaluation.diagnostic_mode_id"
        ),
    )


def load_aet_quality_replication_recipe(path: Path) -> AETQualityReplicationRecipe:
    """Load and strictly validate the frozen multi-seed replication recipe."""

    raw = _load_yaml(Path(path))
    _closed(raw, _TOP_LEVEL_KEYS, "recipe")
    _exact(raw["schema_version"], SCHEMA_VERSION, "recipe.schema_version")
    _exact(raw["kind"], RECIPE_KIND, "recipe.kind")

    classification = _mapping(raw["classification"], "recipe.classification")
    _closed(classification, _CLASSIFICATION_KEYS, "recipe.classification")
    for key, expected in {
        "purpose": "quality_exploratory",
        "scientific_use": False,
        "aet_eligible": False,
        "energy_measurement": "none",
        "exclusive_access_required": False,
    }.items():
        _exact(classification[key], expected, f"classification.{key}")

    platform_raw = _mapping(raw["platform"], "recipe.platform")
    _closed(platform_raw, _PLATFORM_KEYS, "recipe.platform")
    execution_layer = _exact(
        platform_raw["execution_layer"], "windows-native", "platform.execution_layer"
    )
    host_id = _exact(platform_raw["host_id"], "win-a4500-01", "platform.host_id")
    accelerator = _exact(
        platform_raw["accelerator_label"],
        "NVIDIA RTX A4500",
        "platform.accelerator_label",
    )
    gpu_index = _exact(
        _integer(platform_raw["gpu_index"], "platform.gpu_index"),
        0,
        "platform.gpu_index",
    )

    base_raw = _mapping(raw["base_recipe"], "recipe.base_recipe")
    _closed(base_raw, _BASE_RECIPE_KEYS, "recipe.base_recipe")
    base_recipe = BaseRecipeProvenance(
        path=_exact(
            _relative_path(base_raw["path"], "base_recipe.path", suffix=".yaml"),
            BASE_RECIPE_PATH,
            "base_recipe.path",
        ),
        semantic_sha256=_exact(
            _sha256(base_raw["semantic_sha256"], "base_recipe.semantic_sha256"),
            BASE_RECIPE_SEMANTIC_SHA256,
            "base_recipe.semantic_sha256",
        ),
    )

    discovery_raw = _mapping(raw["discovery_zip"], "recipe.discovery_zip")
    _closed(discovery_raw, _DISCOVERY_KEYS, "recipe.discovery_zip")
    discovery = DiscoveryZipProvenance(
        path=_exact(
            _relative_path(discovery_raw["path"], "discovery_zip.path", suffix=".zip"),
            DISCOVERY_ZIP_PATH,
            "discovery_zip.path",
        ),
        sha256=_exact(
            _sha256(discovery_raw["sha256"], "discovery_zip.sha256"),
            DISCOVERY_ZIP_SHA256,
            "discovery_zip.sha256",
        ),
        manifest_sha256=_exact(
            _sha256(discovery_raw["manifest_sha256"], "discovery_zip.manifest_sha256"),
            DISCOVERY_MANIFEST_SHA256,
            "discovery_zip.manifest_sha256",
        ),
        checksums_sha256=_exact(
            _sha256(discovery_raw["checksums_sha256"], "discovery_zip.checksums_sha256"),
            DISCOVERY_CHECKSUMS_SHA256,
            "discovery_zip.checksums_sha256",
        ),
        expected_status=_exact(
            discovery_raw["expected_status"],
            "complete_quality_gate_passed",
            "discovery_zip.expected_status",
        ),
        git_sha=_exact(discovery_raw["git_sha"], DISCOVERY_GIT_SHA, "discovery_zip.git_sha"),
        checkpoint_epoch=_exact(
            _integer(discovery_raw["checkpoint_epoch"], "discovery_zip.checkpoint_epoch"),
            40,
            "discovery_zip.checkpoint_epoch",
        ),
        checkpoint_sha256=_exact(
            _sha256(
                discovery_raw["checkpoint_sha256"],
                "discovery_zip.checkpoint_sha256",
            ),
            DISCOVERY_CHECKPOINT_SHA256,
            "discovery_zip.checkpoint_sha256",
        ),
        expected_primary_mode=_exact(
            discovery_raw["expected_primary_mode"],
            "pomo-50x8",
            "discovery_zip.expected_primary_mode",
        ),
        use=_exact(discovery_raw["use"], "provenance_only", "discovery_zip.use"),
        evaluated=_exact(discovery_raw["evaluated"], False, "discovery_zip.evaluated"),
        included_in_aggregation=_exact(
            discovery_raw["included_in_aggregation"],
            False,
            "discovery_zip.included_in_aggregation",
        ),
    )

    dataset_raw = _mapping(raw["dataset"], "recipe.dataset")
    _closed(dataset_raw, _DATASET_KEYS, "recipe.dataset")
    dataset = ReplicationDataset(
        problem=_exact(dataset_raw["problem"], "cvrp", "dataset.problem"),
        size=_exact(_integer(dataset_raw["size"], "dataset.size", 2), 50, "dataset.size"),
        capacity=_exact(
            _number(dataset_raw["capacity"], "dataset.capacity", positive=True),
            40.0,
            "dataset.capacity",
        ),
        max_demand=_exact(
            _integer(dataset_raw["max_demand"], "dataset.max_demand", 1),
            9,
            "dataset.max_demand",
        ),
        development=_parse_split(dataset_raw["development"]),
    )
    reference = _parse_reference(raw["reference"])
    model = _parse_model(raw["model"])
    training = _parse_training(raw["training"])
    evaluation = _parse_evaluation(raw["evaluation"])

    bootstrap_raw = _mapping(raw["bootstrap"], "recipe.bootstrap")
    _closed(bootstrap_raw, _BOOTSTRAP_KEYS, "recipe.bootstrap")
    bootstrap = BootstrapConfig(
        method=_exact(
            bootstrap_raw["method"],
            "crossed_seed_by_instance_percentile",
            "bootstrap.method",
        ),
        replicates=_exact(
            _integer(bootstrap_raw["replicates"], "bootstrap.replicates", 1),
            10_000,
            "bootstrap.replicates",
        ),
        generator=_exact(bootstrap_raw["generator"], "numpy-pcg64", "bootstrap.generator"),
        seed=_exact(_integer(bootstrap_raw["seed"], "bootstrap.seed"), 3491, "bootstrap.seed"),
        confidence_level=_exact(
            _number(
                bootstrap_raw["confidence_level"],
                "bootstrap.confidence_level",
                positive=True,
            ),
            0.95,
            "bootstrap.confidence_level",
        ),
        shared_resamples_across_pomo_modes=_exact(
            bootstrap_raw["shared_resamples_across_pomo_modes"],
            True,
            "bootstrap.shared_resamples_across_pomo_modes",
        ),
        seed_t_critical_value=_exact(
            _number(
                bootstrap_raw["seed_t_critical_value"],
                "bootstrap.seed_t_critical_value",
                positive=True,
            ),
            2.13184678632665,
            "bootstrap.seed_t_critical_value",
        ),
        seed_t_degrees_of_freedom=_exact(
            _integer(
                bootstrap_raw["seed_t_degrees_of_freedom"],
                "bootstrap.seed_t_degrees_of_freedom",
                1,
            ),
            4,
            "bootstrap.seed_t_degrees_of_freedom",
        ),
    )

    gate_raw = _mapping(raw["quality_gate"], "recipe.quality_gate")
    _closed(gate_raw, _GATE_KEYS, "recipe.quality_gate")
    gate = ReplicationQualityGate(
        metric=_exact(
            gate_raw["metric"],
            "mean_gap_to_locked_reference_pct",
            "quality_gate.metric",
        ),
        primary_mode_id=_exact(
            gate_raw["primary_mode_id"],
            evaluation.primary_mode_id,
            "quality_gate.primary_mode_id",
        ),
        maximum_mean_gap_pct=_exact(
            _number(
                gate_raw["maximum_mean_gap_pct"],
                "quality_gate.maximum_mean_gap_pct",
            ),
            5.0,
            "quality_gate.maximum_mean_gap_pct",
        ),
        maximum_invalid_instances=_exact(
            _integer(
                gate_raw["maximum_invalid_instances"],
                "quality_gate.maximum_invalid_instances",
            ),
            0,
            "quality_gate.maximum_invalid_instances",
        ),
        require_finite=_exact(gate_raw["require_finite"], True, "quality_gate.require_finite"),
        require_each_seed_below_threshold=_exact(
            gate_raw["require_each_seed_below_threshold"],
            True,
            "quality_gate.require_each_seed_below_threshold",
        ),
        require_t_ucb_below_threshold=_exact(
            gate_raw["require_t_ucb_below_threshold"],
            True,
            "quality_gate.require_t_ucb_below_threshold",
        ),
        require_bootstrap_ucb_below_threshold=_exact(
            gate_raw["require_bootstrap_ucb_below_threshold"],
            True,
            "quality_gate.require_bootstrap_ucb_below_threshold",
        ),
        ordered_secondary_only_if_primary_passes=_exact(
            gate_raw["ordered_secondary_only_if_primary_passes"],
            True,
            "quality_gate.ordered_secondary_only_if_primary_passes",
        ),
    )

    return AETQualityReplicationRecipe(
        name=_exact(
            raw["name"],
            "aet-quality-replication-cvrp50-epoch40-seeds2-6",
            "recipe.name",
        ),
        output_root=_exact(
            _relative_path(raw["output_root"], "recipe.output_root"),
            "experiments/aet-journal/raw/quality-replication/cvrp50-epoch40-seeds2-6",
            "recipe.output_root",
        ),
        host_id=host_id,
        execution_layer=execution_layer,
        gpu_index=gpu_index,
        expected_accelerator_label=accelerator,
        base_recipe=base_recipe,
        discovery_zip=discovery,
        dataset=dataset,
        reference=reference,
        model=model,
        training=training,
        evaluation=evaluation,
        bootstrap=bootstrap,
        gate=gate,
    )


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_members(archive: ZipFile) -> dict[str, Any]:
    """Index ZIP members by canonical separators and reject ambiguous names."""

    members: dict[str, Any] = {}
    casefolded: dict[str, str] = {}
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        path = PurePosixPath(name)
        if (
            not name
            or name.startswith("/")
            or PureWindowsPath(name).drive
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise QualityReplicationRecipeValidationError(
                f"discovery ZIP contains an unsafe member path: {info.filename!r}"
            )
        folded = name.casefold()
        if name in members or (folded in casefolded and casefolded[folded] != name):
            raise QualityReplicationRecipeValidationError(
                f"discovery ZIP contains an ambiguous member path: {info.filename!r}"
            )
        members[name] = info
        casefolded[folded] = name
    return members


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


def _input_qualification(
    recipe: AETQualityReplicationRecipe, repository_root: Path
) -> dict[str, Any]:
    base_path = repository_root / recipe.base_recipe.path
    base_exists = base_path.is_file()
    base_semantic_sha256: str | None = None
    base_semantic_matches = False
    base_error: str | None = None
    if base_exists:
        try:
            base_semantic_sha256 = canonical_yaml_sha256(base_path)
            base_semantic_matches = base_semantic_sha256 == recipe.base_recipe.semantic_sha256
        except Exception as exc:
            base_error = f"{type(exc).__name__}: {exc}"

    external_zip_path = repository_root / recipe.discovery_zip.path
    internal_zip_path = (
        repository_root
        / recipe.output_root
        / "provenance"
        / "discovery"
        / recipe.discovery_zip.path
    )
    zip_path = external_zip_path if external_zip_path.is_file() else internal_zip_path
    zip_exists = zip_path.is_file()
    archive_sha256: str | None = None
    archive_sha256_matches = False
    member_checks: dict[str, bool] = {}
    archive_error: str | None = None
    if zip_exists:
        try:
            archive_sha256 = _hash_file(zip_path)
            archive_sha256_matches = archive_sha256 == recipe.discovery_zip.sha256
            if archive_sha256_matches:
                with ZipFile(zip_path) as archive:
                    members = _zip_members(archive)
                    manifest_bytes = archive.read(members["manifest.json"])
                    checksums_bytes = archive.read(members["SHA256SUMS"])
                    checkpoint_bytes = archive.read(members["training/checkpoints/epoch-040.pt"])
                    manifest = json.loads(manifest_bytes)
                    member_checks = {
                        "manifest_sha256_matches": (
                            _hash_bytes(manifest_bytes) == recipe.discovery_zip.manifest_sha256
                        ),
                        "checksums_sha256_matches": (
                            _hash_bytes(checksums_bytes) == recipe.discovery_zip.checksums_sha256
                        ),
                        "checkpoint_sha256_matches": (
                            _hash_bytes(checkpoint_bytes) == recipe.discovery_zip.checkpoint_sha256
                        ),
                        "status_matches": (
                            manifest.get("status") == recipe.discovery_zip.expected_status
                        ),
                        "source_commit_matches": (
                            manifest.get("source", {}).get("git_sha")
                            == recipe.discovery_zip.git_sha
                        ),
                        "checkpoint_is_primary_epoch40": (
                            manifest.get("checkpoint_selection", {})
                            .get("selections", {})
                            .get(recipe.discovery_zip.expected_primary_mode, {})
                            .get("epoch")
                            == recipe.discovery_zip.checkpoint_epoch
                        ),
                        "seed1_excluded_from_replication": (
                            not recipe.discovery_zip.evaluated
                            and not recipe.discovery_zip.included_in_aggregation
                        ),
                    }
        except (BadZipFile, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            archive_error = f"{type(exc).__name__}: {exc}"
    discovery_ready = bool(
        zip_exists
        and archive_sha256_matches
        and member_checks
        and all(member_checks.values())
        and archive_error is None
    )
    return {
        "base_recipe_exists": base_exists,
        "base_recipe_semantic_sha256": base_semantic_sha256,
        "base_recipe_semantic_sha256_matches": base_semantic_matches,
        "base_recipe_error": base_error,
        "discovery_zip_exists": zip_exists,
        "discovery_zip_source": (
            "repository_root"
            if external_zip_path.is_file()
            else "imported_replication_bundle"
            if internal_zip_path.is_file()
            else None
        ),
        "discovery_zip_sha256": archive_sha256,
        "discovery_zip_sha256_matches": archive_sha256_matches,
        "discovery_member_checks": member_checks,
        "discovery_zip_error": archive_error,
        "inputs_ready": bool(base_semantic_matches and discovery_ready),
    }


def runtime_qualification(
    recipe: AETQualityReplicationRecipe,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Qualify the Windows runtime and the two frozen input artifacts."""

    runtime = quality_runtime_qualification(
        cast(
            AETQualityRecipe,
            SimpleNamespace(
                gpu_index=recipe.gpu_index,
                accelerator_label=recipe.expected_accelerator_label,
                host_id=recipe.host_id,
            ),
        )
    )
    inputs = _input_qualification(recipe, repository_root or _repository_root())
    return {
        **runtime,
        "inputs": inputs,
        "ready_to_execute": bool(runtime["ready_to_execute"] and inputs["inputs_ready"]),
    }


def dry_run(path: Path) -> dict[str, Any]:
    """Validate the recipe and report all planned work without writes."""

    recipe = load_aet_quality_replication_recipe(path)
    qualification = runtime_qualification(recipe)
    seed_count = len(recipe.training.seeds)
    report = {
        "status": (
            "valid_ready_quality_replication"
            if qualification["ready_to_execute"]
            else "valid_not_qualified"
        ),
        "schema_version": SCHEMA_VERSION,
        "name": recipe.name,
        "purpose": "quality_exploratory",
        "scientific_use": False,
        "aet_eligible": False,
        "energy_measurement": "none",
        "exclusive_access_required": False,
        "seed1_use": "provenance_only",
        "seed1_evaluated": False,
        "seed1_included_in_aggregation": False,
        "training_seeds": list(recipe.training.seeds),
        "training_seed_count": seed_count,
        "fixed_epoch": recipe.training.fixed_epoch,
        "training_instances": (
            seed_count * recipe.training.epochs * recipe.training.instances_per_epoch
        ),
        "development_instances": recipe.dataset.development.num_instances,
        "planned_hgs_solves": (
            recipe.dataset.development.num_instances * len(recipe.reference.hgs.seeds)
        ),
        "planned_ortools_solves": recipe.dataset.development.num_instances,
        "evaluation_mode_order": [mode.mode_id for mode in recipe.evaluation.modes],
        "planned_neural_evaluations": seed_count * len(recipe.evaluation.modes),
        "maximum_newly_completed_seeds_per_invocation": (
            recipe.training.max_newly_completed_seeds_per_invocation
        ),
        "maximum_training_walltime_per_invocation_s": (
            recipe.training.max_walltime_per_invocation_s
        ),
        "bootstrap_replicates": recipe.bootstrap.replicates,
        "bootstrap_seed": recipe.bootstrap.seed,
        "quality_gate_primary_mode": recipe.gate.primary_mode_id,
        "quality_gate_maximum_mean_gap_pct": recipe.gate.maximum_mean_gap_pct,
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
    "AETQualityReplicationRecipe",
    "BootstrapConfig",
    "DatasetSplit",
    "QualityReplicationRecipeValidationError",
    "ReplicationQualityGate",
    "canonical_yaml_sha256",
    "dry_run",
    "load_aet_quality_replication_recipe",
    "runtime_qualification",
]
