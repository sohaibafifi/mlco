"""Closed recipe for the first AET journal quality pilot.

The quality pilot is deliberately separate from energy measurement.  It is a
screening experiment that decides whether a trained policy is accurate enough
to justify a later measured AET campaign.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "aet-journal-quality-pilot/v1"
RECIPE_KIND = "aet-journal-quality-pilot"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "dataset",
    "reference",
    "model",
    "training",
    "evaluation",
    "quality_gate",
}
_CLASSIFICATION_KEYS = {
    "purpose",
    "scientific_use",
    "aet_eligible",
    "energy_measurement",
}
_PLATFORM_KEYS = {"execution_layer", "host_id", "accelerator_label", "gpu_index"}
_DATASET_KEYS = {"problem", "size", "capacity", "max_demand", "selection", "holdout"}
_SPLIT_KEYS = {"id", "num_instances", "seed", "artifact"}
_REFERENCE_KEYS = {
    "policy",
    "locked_before_neural_evaluation",
    "hgs",
    "ortools",
}
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
    "seed",
    "epochs",
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
}
_SCHEDULER_KEYS = {"name", "milestones", "gamma"}
_EVALUATION_KEYS = {"selection_rule", "modes"}
_MODE_KEYS = {
    "id",
    "n_starts",
    "augmentations",
    "forced_first_actions",
    "batch_size",
    "inference_precision",
}
_GATE_KEYS = {
    "metric",
    "maximum_mean_gap_pct",
    "maximum_invalid_instances",
    "require_finite",
    "split",
}
_SAFE_COMPONENT = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_WINDOWS_RESERVED_NAMES = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class QualityRecipeValidationError(ValueError):
    """Raised when the quality-pilot recipe violates its closed schema."""


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    split_id: str
    num_instances: int
    seed: int
    artifact: str


@dataclass(frozen=True, slots=True)
class QualityDataset:
    problem: str
    size: int
    capacity: float
    max_demand: int
    selection: DatasetSplit
    holdout: DatasetSplit


@dataclass(frozen=True, slots=True)
class HGSReference:
    solver: str
    seeds: tuple[int, ...]
    max_iterations: int


@dataclass(frozen=True, slots=True)
class ORToolsReference:
    solver: str
    seed: int
    solution_limit: int
    scaling_factor: int


@dataclass(frozen=True, slots=True)
class ReferenceConfig:
    policy: str
    locked_before_neural_evaluation: bool
    hgs: HGSReference
    ortools: ORToolsReference


@dataclass(frozen=True, slots=True)
class QualityModel:
    architecture: str
    expected_parameter_count: int
    hidden_dim: int
    num_layers: int
    num_heads: int
    dynamic_context: str


@dataclass(frozen=True, slots=True)
class EpochScheduler:
    name: str
    milestones: tuple[int, ...]
    gamma: float


@dataclass(frozen=True, slots=True)
class QualityTraining:
    algorithm: str
    seed: int
    epochs: int
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


@dataclass(frozen=True, slots=True)
class EvaluationMode:
    mode_id: str
    n_starts: int
    augmentations: int
    forced_first_actions: bool
    batch_size: int
    inference_precision: str


@dataclass(frozen=True, slots=True)
class QualityEvaluation:
    selection_rule: str
    modes: tuple[EvaluationMode, ...]


@dataclass(frozen=True, slots=True)
class QualityGate:
    metric: str
    maximum_mean_gap_pct: float
    maximum_invalid_instances: int
    require_finite: bool
    split: str


@dataclass(frozen=True, slots=True)
class AETQualityRecipe:
    name: str
    host_id: str
    accelerator_label: str
    gpu_index: int
    output_root: str
    dataset: QualityDataset
    reference: ReferenceConfig
    model: QualityModel
    training: QualityTraining
    evaluation: QualityEvaluation
    quality_gate: QualityGate


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualityRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = sorted(repr(key) for key in set(value) - keys)
    if unknown:
        raise QualityRecipeValidationError(f"{where} has unknown keys: {', '.join(unknown)}")
    missing = sorted(keys - set(value))
    if missing:
        raise QualityRecipeValidationError(
            f"{where} is missing required keys: {', '.join(missing)}"
        )


def _slug(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise QualityRecipeValidationError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise QualityRecipeValidationError(f"{where} uses a Windows-reserved name")
    return value


def _relative_path(value: Any, where: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise QualityRecipeValidationError(
            f"{where} must be a non-empty relative path using '/' separators"
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise QualityRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise QualityRecipeValidationError(f"{where} may not contain '.' or '..'")
    for part in posix_path.parts:
        _slug(part, f"{where} component")
    if suffix is not None and posix_path.suffix != suffix:
        raise QualityRecipeValidationError(f"{where} must identify a {suffix} file")
    return posix_path.as_posix()


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise QualityRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityRecipeValidationError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        qualifier = "positive finite" if positive else "non-negative finite"
        raise QualityRecipeValidationError(f"{where} must be a {qualifier} number")
    return result


def _exact(value: Any, expected: Any, where: str) -> Any:
    if value != expected:
        raise QualityRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _split(raw: Any, where: str) -> DatasetSplit:
    value = _mapping(raw, where)
    _closed(value, _SPLIT_KEYS, where)
    artifact = _relative_path(value["artifact"], f"{where}.artifact", suffix=".npz")
    if not artifact.startswith("shared/"):
        raise QualityRecipeValidationError(f"{where}.artifact must stay under shared/")
    return DatasetSplit(
        split_id=_slug(value["id"], f"{where}.id"),
        num_instances=_integer(value["num_instances"], f"{where}.num_instances", 1),
        seed=_integer(value["seed"], f"{where}.seed"),
        artifact=artifact,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("AET quality recipe validation needs PyYAML") from exc
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QualityRecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise QualityRecipeValidationError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(raw, "recipe")


def load_aet_quality_recipe(path: Path) -> AETQualityRecipe:
    """Load and strictly validate one quality-only pilot recipe."""

    raw = _load_yaml(Path(path))
    _closed(raw, _TOP_LEVEL_KEYS, "recipe")
    _exact(raw["schema_version"], SCHEMA_VERSION, "recipe.schema_version")
    _exact(raw["kind"], RECIPE_KIND, "recipe.kind")

    classification = _mapping(raw["classification"], "recipe.classification")
    _closed(classification, _CLASSIFICATION_KEYS, "recipe.classification")
    _exact(classification["purpose"], "quality_exploratory", "classification.purpose")
    _exact(classification["scientific_use"], False, "classification.scientific_use")
    _exact(classification["aet_eligible"], False, "classification.aet_eligible")
    _exact(classification["energy_measurement"], "none", "classification.energy_measurement")

    platform_cfg = _mapping(raw["platform"], "recipe.platform")
    _closed(platform_cfg, _PLATFORM_KEYS, "recipe.platform")
    _exact(platform_cfg["execution_layer"], "windows-native", "platform.execution_layer")
    host_id = _slug(platform_cfg["host_id"], "platform.host_id")
    accelerator_label = platform_cfg["accelerator_label"]
    if not isinstance(accelerator_label, str) or not accelerator_label.strip():
        raise QualityRecipeValidationError("platform.accelerator_label must be non-empty text")
    gpu_index = _integer(platform_cfg["gpu_index"], "platform.gpu_index")

    dataset_raw = _mapping(raw["dataset"], "recipe.dataset")
    _closed(dataset_raw, _DATASET_KEYS, "recipe.dataset")
    dataset = QualityDataset(
        problem=_exact(dataset_raw["problem"], "cvrp", "dataset.problem"),
        size=_integer(dataset_raw["size"], "dataset.size", 2),
        capacity=_number(dataset_raw["capacity"], "dataset.capacity", positive=True),
        max_demand=_integer(dataset_raw["max_demand"], "dataset.max_demand", 1),
        selection=_split(dataset_raw["selection"], "dataset.selection"),
        holdout=_split(dataset_raw["holdout"], "dataset.holdout"),
    )
    if dataset.selection.seed == dataset.holdout.seed:
        raise QualityRecipeValidationError("selection and holdout seeds must differ")
    if dataset.selection.artifact == dataset.holdout.artifact:
        raise QualityRecipeValidationError("selection and holdout artifacts must differ")

    reference_raw = _mapping(raw["reference"], "recipe.reference")
    _closed(reference_raw, _REFERENCE_KEYS, "recipe.reference")
    hgs_raw = _mapping(reference_raw["hgs"], "reference.hgs")
    _closed(hgs_raw, _HGS_KEYS, "reference.hgs")
    seeds_raw = hgs_raw["seeds"]
    if not isinstance(seeds_raw, list) or not seeds_raw:
        raise QualityRecipeValidationError("reference.hgs.seeds must be a non-empty list")
    hgs_seeds = tuple(_integer(value, "reference.hgs.seeds[]") for value in seeds_raw)
    if len(set(hgs_seeds)) != len(hgs_seeds):
        raise QualityRecipeValidationError("reference.hgs.seeds must be unique")
    ortools_raw = _mapping(reference_raw["ortools"], "reference.ortools")
    _closed(ortools_raw, _ORTOOLS_KEYS, "reference.ortools")
    reference = ReferenceConfig(
        policy=_exact(
            reference_raw["policy"],
            "best_validated_per_instance",
            "reference.policy",
        ),
        locked_before_neural_evaluation=_exact(
            reference_raw["locked_before_neural_evaluation"],
            True,
            "reference.locked_before_neural_evaluation",
        ),
        hgs=HGSReference(
            solver=_exact(hgs_raw["solver"], "pyvrp-hgs", "reference.hgs.solver"),
            seeds=hgs_seeds,
            max_iterations=_integer(hgs_raw["max_iterations"], "reference.hgs.max_iterations", 1),
        ),
        ortools=ORToolsReference(
            solver=_exact(
                ortools_raw["solver"],
                "ortools-routing-gls",
                "reference.ortools.solver",
            ),
            seed=_integer(ortools_raw["seed"], "reference.ortools.seed"),
            solution_limit=_integer(
                ortools_raw["solution_limit"],
                "reference.ortools.solution_limit",
                1,
            ),
            scaling_factor=_integer(
                ortools_raw["scaling_factor"], "reference.ortools.scaling_factor", 1
            ),
        ),
    )

    model_raw = _mapping(raw["model"], "recipe.model")
    _closed(model_raw, _MODEL_KEYS, "recipe.model")
    model = QualityModel(
        architecture=_exact(model_raw["architecture"], "mlco-am", "model.architecture"),
        expected_parameter_count=_integer(
            model_raw["expected_parameter_count"], "model.expected_parameter_count", 1
        ),
        hidden_dim=_integer(model_raw["hidden_dim"], "model.hidden_dim", 1),
        num_layers=_integer(model_raw["num_layers"], "model.num_layers", 1),
        num_heads=_integer(model_raw["num_heads"], "model.num_heads", 1),
        dynamic_context=_exact(
            model_raw["dynamic_context"],
            "remaining_capacity_normalized",
            "model.dynamic_context",
        ),
    )
    if model.hidden_dim % model.num_heads != 0:
        raise QualityRecipeValidationError("model.hidden_dim must be divisible by num_heads")

    training_raw = _mapping(raw["training"], "recipe.training")
    _closed(training_raw, _TRAINING_KEYS, "recipe.training")
    scheduler_raw = _mapping(training_raw["scheduler"], "training.scheduler")
    _closed(scheduler_raw, _SCHEDULER_KEYS, "training.scheduler")
    milestones_raw = scheduler_raw["milestones"]
    if not isinstance(milestones_raw, list):
        raise QualityRecipeValidationError("training.scheduler.milestones must be a list")
    milestones = tuple(
        _integer(value, "training.scheduler.milestones[]", 1) for value in milestones_raw
    )
    if milestones != tuple(sorted(set(milestones))):
        raise QualityRecipeValidationError("training.scheduler.milestones must be increasing")
    checkpoint_raw = training_raw["checkpoint_epochs"]
    if not isinstance(checkpoint_raw, list) or not checkpoint_raw:
        raise QualityRecipeValidationError("training.checkpoint_epochs must be non-empty")
    checkpoint_epochs = tuple(
        _integer(value, "training.checkpoint_epochs[]") for value in checkpoint_raw
    )
    if checkpoint_epochs != tuple(sorted(set(checkpoint_epochs))):
        raise QualityRecipeValidationError("training.checkpoint_epochs must be increasing")
    training = QualityTraining(
        algorithm=_exact(training_raw["algorithm"], "pomo", "training.algorithm"),
        seed=_integer(training_raw["seed"], "training.seed"),
        epochs=_integer(training_raw["epochs"], "training.epochs", 1),
        instances_per_epoch=_integer(
            training_raw["instances_per_epoch"], "training.instances_per_epoch", 1
        ),
        batch_size=_integer(training_raw["batch_size"], "training.batch_size", 1),
        n_starts=_integer(training_raw["n_starts"], "training.n_starts", 1),
        precision=_exact(training_raw["precision"], "bf16", "training.precision"),
        optimizer=_exact(training_raw["optimizer"], "adam", "training.optimizer"),
        learning_rate=_number(
            training_raw["learning_rate"], "training.learning_rate", positive=True
        ),
        weight_decay=_number(training_raw["weight_decay"], "training.weight_decay"),
        gradient_clip=_number(training_raw["gradient_clip"], "training.gradient_clip"),
        scheduler=EpochScheduler(
            name=_exact(scheduler_raw["name"], "multistep-epoch", "scheduler.name"),
            milestones=milestones,
            gamma=_number(scheduler_raw["gamma"], "scheduler.gamma", positive=True),
        ),
        checkpoint_epochs=checkpoint_epochs,
        max_walltime_per_invocation_s=_number(
            training_raw["max_walltime_per_invocation_s"],
            "training.max_walltime_per_invocation_s",
            positive=True,
        ),
    )
    if training.checkpoint_epochs[0] != 0 or training.checkpoint_epochs[-1] != training.epochs:
        raise QualityRecipeValidationError("checkpoint_epochs must start at 0 and end at epochs")
    if any(value >= training.epochs for value in training.scheduler.milestones):
        raise QualityRecipeValidationError("scheduler milestones must be below training.epochs")
    if training.n_starts > dataset.size:
        raise QualityRecipeValidationError("training.n_starts may not exceed dataset.size")

    evaluation_raw = _mapping(raw["evaluation"], "recipe.evaluation")
    _closed(evaluation_raw, _EVALUATION_KEYS, "recipe.evaluation")
    modes_raw = evaluation_raw["modes"]
    if not isinstance(modes_raw, list) or not modes_raw:
        raise QualityRecipeValidationError("evaluation.modes must be a non-empty list")
    modes: list[EvaluationMode] = []
    for index, mode_value in enumerate(modes_raw):
        where = f"evaluation.modes[{index}]"
        mode_raw = _mapping(mode_value, where)
        _closed(mode_raw, _MODE_KEYS, where)
        augmentations = _integer(mode_raw["augmentations"], f"{where}.augmentations", 1)
        if augmentations not in {1, 8}:
            raise QualityRecipeValidationError(f"{where}.augmentations must be 1 or 8")
        forced = mode_raw["forced_first_actions"]
        if not isinstance(forced, bool):
            raise QualityRecipeValidationError(f"{where}.forced_first_actions must be boolean")
        n_starts = _integer(mode_raw["n_starts"], f"{where}.n_starts", 1)
        if (n_starts > 1) != forced:
            raise QualityRecipeValidationError(
                f"{where} must force first actions exactly when n_starts > 1"
            )
        modes.append(
            EvaluationMode(
                mode_id=_slug(mode_raw["id"], f"{where}.id"),
                n_starts=n_starts,
                augmentations=augmentations,
                forced_first_actions=forced,
                batch_size=_integer(mode_raw["batch_size"], f"{where}.batch_size", 1),
                inference_precision=_exact(
                    mode_raw["inference_precision"], "fp32", f"{where}.inference_precision"
                ),
            )
        )
    if len({mode.mode_id for mode in modes}) != len(modes):
        raise QualityRecipeValidationError("evaluation mode ids must be unique")
    evaluation = QualityEvaluation(
        selection_rule=_exact(
            evaluation_raw["selection_rule"],
            "feasible_then_lowest_mean_cost_then_earliest_epoch",
            "evaluation.selection_rule",
        ),
        modes=tuple(modes),
    )

    gate_raw = _mapping(raw["quality_gate"], "recipe.quality_gate")
    _closed(gate_raw, _GATE_KEYS, "recipe.quality_gate")
    require_finite = gate_raw["require_finite"]
    if not isinstance(require_finite, bool):
        raise QualityRecipeValidationError("quality_gate.require_finite must be boolean")
    gate = QualityGate(
        metric=_exact(
            gate_raw["metric"],
            "mean_gap_to_locked_reference_pct",
            "quality_gate.metric",
        ),
        maximum_mean_gap_pct=_number(
            gate_raw["maximum_mean_gap_pct"], "quality_gate.maximum_mean_gap_pct"
        ),
        maximum_invalid_instances=_integer(
            gate_raw["maximum_invalid_instances"], "quality_gate.maximum_invalid_instances"
        ),
        require_finite=require_finite,
        split=_exact(gate_raw["split"], "holdout", "quality_gate.split"),
    )

    return AETQualityRecipe(
        name=_slug(raw["name"], "recipe.name"),
        host_id=host_id,
        accelerator_label=accelerator_label,
        gpu_index=gpu_index,
        output_root=_relative_path(raw["output_root"], "recipe.output_root"),
        dataset=dataset,
        reference=reference,
        model=model,
        training=training,
        evaluation=evaluation,
        quality_gate=gate,
    )


def _git_snapshot() -> dict[str, Any]:
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain=v1"], text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "root": None, "sha": None, "dirty": None}
    return {"available": True, "root": root, "sha": sha, "dirty": bool(status)}


def runtime_qualification(recipe: AETQualityRecipe) -> dict[str, Any]:
    """Check only what this quality-only pilot actually needs."""

    windows_native = sys.platform == "win32" and platform.system() == "Windows"
    import_targets = {
        "torch": "torch",
        "pyvrp": "pyvrp",
        "ortools": "ortools.constraint_solver.pywrapcp",
    }
    imports: dict[str, bool] = {}
    import_errors: dict[str, str] = {}
    for name, target in import_targets.items():
        try:
            module = importlib.import_module(target)
            imports[name] = module is not None
        except Exception as exc:
            imports[name] = False
            import_errors[name] = f"{type(exc).__name__}: {exc}"
    cuda_available = False
    cuda_device_count = 0
    selected_gpu_available = False
    bf16_supported = False
    bf16_active_probe_passed = False
    bf16_active_probe_error: str | None = None
    selected_gpu_name: str | None = None
    selected_gpu_memory_total_b: int | None = None
    selected_gpu_memory_free_b: int | None = None
    gpu_identity_matches = False
    if imports["torch"]:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        selected_gpu_available = 0 <= recipe.gpu_index < cuda_device_count
        if selected_gpu_available:
            try:
                torch.cuda.set_device(recipe.gpu_index)
                selected_gpu_name = torch.cuda.get_device_name(recipe.gpu_index)
                gpu_identity_matches = selected_gpu_name == recipe.accelerator_label
                selected_gpu_memory_free_b, selected_gpu_memory_total_b = (
                    int(value) for value in torch.cuda.mem_get_info(recipe.gpu_index)
                )
                bf16_supported = bool(torch.cuda.is_bf16_supported())
                if bf16_supported:
                    left = torch.ones(
                        (8, 8),
                        dtype=torch.bfloat16,
                        device=torch.device(f"cuda:{recipe.gpu_index}"),
                    )
                    product = left @ left
                    torch.cuda.synchronize(recipe.gpu_index)
                    bf16_active_probe_passed = bool(
                        product.dtype == torch.bfloat16
                        and torch.isfinite(product.float()).all().item()
                    )
            except Exception as exc:
                bf16_active_probe_error = f"{type(exc).__name__}: {exc}"
    git = _git_snapshot()
    ready = bool(
        windows_native
        and all(imports.values())
        and cuda_available
        and selected_gpu_available
        and gpu_identity_matches
        and bf16_supported
        and bf16_active_probe_passed
        and git["available"]
    )
    return {
        "windows_native": windows_native,
        "imports": imports,
        "import_errors": import_errors,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "selected_gpu_index": recipe.gpu_index,
        "selected_gpu_available": selected_gpu_available,
        "selected_gpu_name": selected_gpu_name,
        "declared_host_id": recipe.host_id,
        "expected_gpu_name": recipe.accelerator_label,
        "gpu_identity_matches": gpu_identity_matches,
        "selected_gpu_memory_total_b": selected_gpu_memory_total_b,
        "selected_gpu_memory_free_b": selected_gpu_memory_free_b,
        "gpu_free_memory_is_a_gate": False,
        "bf16_supported": bf16_supported,
        "bf16_active_probe_passed": bf16_active_probe_passed,
        "bf16_active_probe_error": bf16_active_probe_error,
        "git": git,
        "git_clean_required": False,
        "ready_to_execute": ready,
    }


def dry_run(path: Path) -> dict[str, Any]:
    """Validate the recipe and report bounds without creating output."""

    recipe = load_aet_quality_recipe(path)
    qualification = runtime_qualification(recipe)
    total_instances = recipe.dataset.selection.num_instances + recipe.dataset.holdout.num_instances
    hgs_solves = total_instances * len(recipe.reference.hgs.seeds)
    ortools_solves = total_instances
    checkpoint_evaluations = len(recipe.training.checkpoint_epochs) * len(recipe.evaluation.modes)
    report = {
        "status": "valid_ready_quality_exploratory"
        if qualification["ready_to_execute"]
        else "valid_not_qualified",
        "schema_version": SCHEMA_VERSION,
        "name": recipe.name,
        "purpose": "quality_exploratory",
        "scientific_use": False,
        "aet_eligible": False,
        "energy_measurement": "none",
        "exclusive_access_required": False,
        "output_root": recipe.output_root,
        "reference_instances": total_instances,
        "planned_hgs_solves": hgs_solves,
        "planned_ortools_solves": ortools_solves,
        "training_seed_count": 1,
        "training_epochs": recipe.training.epochs,
        "training_instances": recipe.training.epochs * recipe.training.instances_per_epoch,
        "checkpoint_count": len(recipe.training.checkpoint_epochs),
        "checkpoint_selection_evaluations": checkpoint_evaluations,
        "holdout_evaluations": len(recipe.evaluation.modes),
        "maximum_training_walltime_per_invocation_s": (
            recipe.training.max_walltime_per_invocation_s
        ),
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
    "AETQualityRecipe",
    "DatasetSplit",
    "EvaluationMode",
    "QualityRecipeValidationError",
    "dry_run",
    "load_aet_quality_recipe",
    "runtime_qualification",
]
