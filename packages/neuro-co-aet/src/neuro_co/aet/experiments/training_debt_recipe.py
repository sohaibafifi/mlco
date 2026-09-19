"""Closed AM/GNN training-debt recipe for the AET journal campaign.

This recipe validates configuration. The runner measures
CPU-package and GPU component counters on native Windows.  Those counters are
not whole-system energy and this recipe never turns them into carbon or an AET
claim by itself.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

from neuro_co.aet.experiments import software_recipe

SCHEMA_VERSION = "aet-journal-training-debt/v1"
KIND = "aet-journal-training-debt"
OUTPUT_ROOT = "experiments/aet-journal/raw/training-debt/cvrp50-epoch40-seeds2-6"
GPU_DEVICE_ID_SHA256 = "686de95a427cb6c0734302fcaf694ddb9c37ba897579d9a66b75c838efeb04cf"
SUPPORTED_ARCHITECTURES = ("am", "gnn")

CLASSIFICATION: dict[str, Any] = {
    "purpose": "software_exploratory_training_debt",
    "scientific_use": False,
    "aet_eligible": False,
    "whole_system_energy": False,
    "cross_solver_energy_comparable": False,
    "carbon_accounting": "none",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_TOP_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "qualification",
    "base_recipe",
    "quality_source",
    "architectures",
    "training",
    "evaluation",
    "quality_gate",
    "measurement",
    "limits",
}


class TrainingDebtRecipeValidationError(ValueError):
    """Raised when a training-debt recipe differs from the closed protocol."""


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Architecture:
    architecture_id: str
    backbone: str
    hidden_dim: int
    num_layers: int
    num_heads: int
    model_configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Training:
    algorithm: str
    seeds: tuple[int, ...]
    epochs: int
    instances_per_epoch: int
    batch_size: int
    n_starts: int
    precision: str
    optimizer: str
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    scheduler_name: str
    scheduler_milestones: tuple[int, ...]
    scheduler_gamma: float
    checkpoint_epochs: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class Evaluation:
    mode_id: str
    n_starts: int
    augmentations: int
    forced_first_actions: bool
    batch_size: int
    inference_precision: str


@dataclass(frozen=True, slots=True)
class QualityGate:
    threshold_pct: float
    maximum_invalid_instances: int
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_quantile: float
    t_critical_value: float
    t_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class TrainingDebtRecipe:
    name: str
    host_id: str
    gpu_index: int
    gpu_device_id_sha256: str
    output_root: str
    preflight_report: str
    base_recipe: Artifact
    quality_source_root: str
    quality_source_status: str
    quality_source_artifacts: tuple[Artifact, ...]
    corpus: Artifact
    corpus_content_sha256: str
    reference: Artifact
    reference_lock: Artifact
    architectures: tuple[Architecture, ...]
    training: Training
    evaluation: Evaluation
    quality_gate: QualityGate
    minimum_measured_duration_s: float
    maximum_seed_walltime_s: float
    maximum_campaign_walltime_s: float
    attestation_max_age_s: float
    eta_calibration_steps: int


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingDebtRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        raise TrainingDebtRecipeValidationError(
            f"{where} keys differ; missing={missing}, unknown={unknown}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if type(value) is not type(expected) or value != expected:
        raise TrainingDebtRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingDebtRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TrainingDebtRecipeValidationError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise TrainingDebtRecipeValidationError(f"{where} must be finite and positive")
    return result


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingDebtRecipeValidationError(f"{where} must be a non-empty string")
    return value


def _sha(value: Any, where: str) -> str:
    value = _string(value, where)
    if _SHA256.fullmatch(value) is None:
        raise TrainingDebtRecipeValidationError(f"{where} must be a lowercase SHA-256")
    return value


def _path(value: Any, where: str) -> str:
    value = _string(value, where)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or "\\" in value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise TrainingDebtRecipeValidationError(f"{where} must be a safe relative POSIX path")
    return value


def _artifact(raw: Any, where: str) -> Artifact:
    value = _mapping(raw, where)
    _closed(value, {"path", "sha256"}, where)
    return Artifact(_path(value["path"], f"{where}.path"), _sha(value["sha256"], f"{where}.sha256"))


def _architecture(raw: Any, where: str) -> Architecture:
    value = _mapping(raw, where)
    _closed(
        value,
        {
            "id",
            "backbone",
            "hidden_dim",
            "num_layers",
            "num_heads",
            "model_configuration",
        },
        where,
    )
    architecture_id = _string(value["id"], f"{where}.id")
    backbone = _string(value["backbone"], f"{where}.backbone")
    if architecture_id not in SUPPORTED_ARCHITECTURES or backbone != architecture_id:
        raise TrainingDebtRecipeValidationError(f"{where} must bind a supported id to itself")
    configuration = _mapping(value["model_configuration"], f"{where}.model_configuration")
    expected_configuration = {
        "backbone": architecture_id,
        "hidden_dim": _integer(value["hidden_dim"], f"{where}.hidden_dim", minimum=1),
        "num_layers": _integer(value["num_layers"], f"{where}.num_layers", minimum=1),
        "num_heads": _integer(value["num_heads"], f"{where}.num_heads", minimum=1),
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise TrainingDebtRecipeValidationError(
                f"{where}.model_configuration.{key} must equal the shared model field"
            )
    if architecture_id == "am":
        _closed(configuration, set(expected_configuration), f"{where}.model_configuration")
    else:
        required = {
            *expected_configuration,
            "encoder",
            "normalization",
            "prenorm",
            "dropout",
            "sparsify",
            "k_sparse",
            "edge_features",
            "rbf_k",
            "fourier_feats",
            "residual",
            "decoder",
        }
        _closed(configuration, required, f"{where}.model_configuration")
        expected_gnn = {
            "encoder": "GNNEncoder",
            "normalization": "batch",
            "prenorm": True,
            "dropout": 0.1,
            "sparsify": True,
            "k_sparse": 10,
            "edge_features": True,
            "rbf_k": 16,
            "fourier_feats": 1,
            "residual": True,
            "decoder": "mlco-pointer-decoder",
        }
        for key, expected in expected_gnn.items():
            _exact(configuration[key], expected, f"{where}.model_configuration.{key}")
    return Architecture(
        architecture_id=architecture_id,
        backbone=backbone,
        hidden_dim=expected_configuration["hidden_dim"],
        num_layers=expected_configuration["num_layers"],
        num_heads=expected_configuration["num_heads"],
        model_configuration=dict(configuration),
    )


def load_aet_training_debt_recipe(path: Path) -> TrainingDebtRecipe:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingDebtRecipeValidationError(f"cannot load recipe: {exc}") from exc
    value = _mapping(raw, "recipe")
    _closed(value, _TOP_KEYS, "recipe")
    _exact(value["schema_version"], SCHEMA_VERSION, "recipe.schema_version")
    _exact(value["kind"], KIND, "recipe.kind")
    _exact(value["classification"], CLASSIFICATION, "recipe.classification")
    _exact(value["output_root"], OUTPUT_ROOT, "recipe.output_root")

    platform = _mapping(value["platform"], "recipe.platform")
    _closed(
        platform,
        {"execution_layer", "host_id", "accelerator_label", "gpu_index", "gpu_device_id_sha256"},
        "recipe.platform",
    )
    _exact(platform["execution_layer"], "windows-native", "recipe.platform.execution_layer")
    _exact(platform["host_id"], "win-a4500-01", "recipe.platform.host_id")
    _exact(platform["accelerator_label"], "NVIDIA RTX A4500", "recipe.platform.accelerator_label")
    _exact(platform["gpu_index"], 0, "recipe.platform.gpu_index")
    _exact(
        platform["gpu_device_id_sha256"],
        GPU_DEVICE_ID_SHA256,
        "recipe.platform.gpu_device_id_sha256",
    )

    qualification = _mapping(value["qualification"], "recipe.qualification")
    _closed(
        qualification,
        {
            "preflight_report",
            "require_git_clean",
            "require_windows_emi",
            "require_nvml_total_energy_counter",
            "allow_energy_fallback",
        },
        "recipe.qualification",
    )
    _exact(qualification["require_git_clean"], False, "recipe.qualification.require_git_clean")
    _exact(qualification["require_windows_emi"], True, "recipe.qualification.require_windows_emi")
    _exact(
        qualification["require_nvml_total_energy_counter"],
        True,
        "recipe.qualification.require_nvml_total_energy_counter",
    )
    _exact(
        qualification["allow_energy_fallback"], False, "recipe.qualification.allow_energy_fallback"
    )

    base_recipe = _artifact(value["base_recipe"], "recipe.base_recipe")
    source = _mapping(value["quality_source"], "recipe.quality_source")
    _closed(
        source,
        {
            "root",
            "expected_status",
            "artifacts",
            "corpus",
            "corpus_content_sha256",
            "reference",
            "reference_lock",
        },
        "recipe.quality_source",
    )
    artifacts_raw = source["artifacts"]
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise TrainingDebtRecipeValidationError("recipe.quality_source.artifacts must be non-empty")
    artifacts = tuple(
        _artifact(item, f"recipe.quality_source.artifacts[{index}]")
        for index, item in enumerate(artifacts_raw)
    )
    if len({item.path for item in artifacts}) != len(artifacts):
        raise TrainingDebtRecipeValidationError(
            "recipe.quality_source.artifacts contains duplicate paths"
        )

    architectures_raw = value["architectures"]
    if not isinstance(architectures_raw, list):
        raise TrainingDebtRecipeValidationError("recipe.architectures must be a list")
    architectures = tuple(
        _architecture(item, f"recipe.architectures[{index}]")
        for index, item in enumerate(architectures_raw)
    )
    if tuple(item.architecture_id for item in architectures) != SUPPORTED_ARCHITECTURES:
        raise TrainingDebtRecipeValidationError(
            "recipe.architectures must be ordered exactly [am, gnn]"
        )

    training_raw = _mapping(value["training"], "recipe.training")
    _closed(
        training_raw,
        {
            "algorithm",
            "seeds",
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
        },
        "recipe.training",
    )
    seeds = tuple(training_raw["seeds"]) if isinstance(training_raw["seeds"], list) else ()
    _exact(seeds, (2, 3, 4, 5, 6), "recipe.training.seeds")
    scheduler = _mapping(training_raw["scheduler"], "recipe.training.scheduler")
    _closed(scheduler, {"name", "milestones", "gamma"}, "recipe.training.scheduler")
    milestones = tuple(scheduler["milestones"]) if isinstance(scheduler["milestones"], list) else ()
    checkpoints = (
        tuple(training_raw["checkpoint_epochs"])
        if isinstance(training_raw["checkpoint_epochs"], list)
        else ()
    )
    expected_training = {
        "algorithm": "pomo",
        "epochs": 40,
        "instances_per_epoch": 20_000,
        "batch_size": 256,
        "n_starts": 50,
        "precision": "bf16",
        "optimizer": "adam",
        "learning_rate": 0.0001,
        "weight_decay": 0.000001,
        "gradient_clip": 1.0,
    }
    for key, expected in expected_training.items():
        _exact(training_raw[key], expected, f"recipe.training.{key}")
    _exact(scheduler["name"], "multistep-epoch", "recipe.training.scheduler.name")
    _exact(milestones, (36, 38), "recipe.training.scheduler.milestones")
    _exact(scheduler["gamma"], 0.1, "recipe.training.scheduler.gamma")
    _exact(checkpoints, (0, 10, 20, 30, 36, 37, 38, 39, 40), "recipe.training.checkpoint_epochs")
    training = Training(
        algorithm="pomo",
        seeds=seeds,
        epochs=40,
        instances_per_epoch=20_000,
        batch_size=256,
        n_starts=50,
        precision="bf16",
        optimizer="adam",
        learning_rate=0.0001,
        weight_decay=0.000001,
        gradient_clip=1.0,
        scheduler_name="multistep-epoch",
        scheduler_milestones=milestones,
        scheduler_gamma=0.1,
        checkpoint_epochs=checkpoints,
    )

    evaluation_raw = _mapping(value["evaluation"], "recipe.evaluation")
    _closed(
        evaluation_raw,
        {
            "mode_id",
            "n_starts",
            "augmentations",
            "forced_first_actions",
            "batch_size",
            "inference_precision",
        },
        "recipe.evaluation",
    )
    expected_evaluation = {
        "mode_id": "pomo-50x8",
        "n_starts": 50,
        "augmentations": 8,
        "forced_first_actions": True,
        "batch_size": 4,
        "inference_precision": "fp32",
    }
    for key, expected in expected_evaluation.items():
        _exact(evaluation_raw[key], expected, f"recipe.evaluation.{key}")
    evaluation = Evaluation(**expected_evaluation)

    gate_raw = _mapping(value["quality_gate"], "recipe.quality_gate")
    _closed(
        gate_raw,
        {
            "threshold_pct",
            "maximum_invalid_instances",
            "bootstrap_replicates",
            "bootstrap_seed",
            "bootstrap_quantile",
            "t_critical_value",
            "t_degrees_of_freedom",
        },
        "recipe.quality_gate",
    )
    expected_gate = {
        "threshold_pct": 5.0,
        "maximum_invalid_instances": 0,
        "bootstrap_replicates": 10_000,
        "bootstrap_seed": 3495,
        "bootstrap_quantile": 0.95,
        "t_critical_value": 2.13184678632665,
        "t_degrees_of_freedom": 4,
    }
    for key, expected in expected_gate.items():
        _exact(gate_raw[key], expected, f"recipe.quality_gate.{key}")
    gate = QualityGate(**expected_gate)

    measurement = _mapping(value["measurement"], "recipe.measurement")
    _closed(
        measurement,
        {
            "cpu_primary_backend",
            "gpu_primary_backend",
            "pue",
            "report_embodied",
            "grid_intensity_g_per_kwh",
            "allow_fallback",
            "required_domains",
            "whole_system_energy",
            "carbon_accounting",
        },
        "recipe.measurement",
    )
    expected_measurement = {
        "cpu_primary_backend": "windows_emi",
        "gpu_primary_backend": "nvml_total_energy_counter",
        "pue": 1.0,
        "report_embodied": False,
        "grid_intensity_g_per_kwh": 0.0,
        "allow_fallback": False,
        "required_domains": ["cpu", "gpu"],
        "whole_system_energy": False,
        "carbon_accounting": "none",
    }
    for key, expected in expected_measurement.items():
        _exact(measurement[key], expected, f"recipe.measurement.{key}")

    limits = _mapping(value["limits"], "recipe.limits")
    _closed(
        limits,
        {
            "minimum_measured_duration_s",
            "maximum_seed_walltime_s",
            "maximum_campaign_walltime_s",
            "attestation_max_age_s",
            "eta_calibration_steps",
        },
        "recipe.limits",
    )
    minimum = _number(
        limits["minimum_measured_duration_s"],
        "recipe.limits.minimum_measured_duration_s",
        positive=True,
    )
    seed_max = _number(
        limits["maximum_seed_walltime_s"], "recipe.limits.maximum_seed_walltime_s", positive=True
    )
    campaign_max = _number(
        limits["maximum_campaign_walltime_s"],
        "recipe.limits.maximum_campaign_walltime_s",
        positive=True,
    )
    attestation_max = _number(
        limits["attestation_max_age_s"], "recipe.limits.attestation_max_age_s", positive=True
    )
    calibration_steps = _integer(
        limits["eta_calibration_steps"], "recipe.limits.eta_calibration_steps", minimum=1
    )
    if (
        minimum != 60.0
        or seed_max != 14_400.0
        or campaign_max != 129_600.0
        or attestation_max != 129_600.0
        or calibration_steps != 3
    ):
        raise TrainingDebtRecipeValidationError("recipe.limits differs from the frozen campaign")

    name = _string(value["name"], "recipe.name")
    if _SAFE.fullmatch(name) is None:
        raise TrainingDebtRecipeValidationError("recipe.name is not a safe identifier")
    return TrainingDebtRecipe(
        name=name,
        host_id=platform["host_id"],
        gpu_index=platform["gpu_index"],
        gpu_device_id_sha256=platform["gpu_device_id_sha256"],
        output_root=OUTPUT_ROOT,
        preflight_report=_path(
            qualification["preflight_report"], "recipe.qualification.preflight_report"
        ),
        base_recipe=base_recipe,
        quality_source_root=_path(source["root"], "recipe.quality_source.root"),
        quality_source_status=_string(
            source["expected_status"], "recipe.quality_source.expected_status"
        ),
        quality_source_artifacts=artifacts,
        corpus=_artifact(source["corpus"], "recipe.quality_source.corpus"),
        corpus_content_sha256=_sha(
            source["corpus_content_sha256"], "recipe.quality_source.corpus_content_sha256"
        ),
        reference=_artifact(source["reference"], "recipe.quality_source.reference"),
        reference_lock=_artifact(source["reference_lock"], "recipe.quality_source.reference_lock"),
        architectures=architectures,
        training=training,
        evaluation=evaluation,
        quality_gate=gate,
        minimum_measured_duration_s=minimum,
        maximum_seed_walltime_s=seed_max,
        maximum_campaign_walltime_s=campaign_max,
        attestation_max_age_s=attestation_max,
        eta_calibration_steps=calibration_steps,
    )


def recipe_summary(recipe: TrainingDebtRecipe) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "valid_closed_recipe",
        "name": recipe.name,
        "execution_layer": "windows-native",
        "host_id": recipe.host_id,
        "gpu_index": recipe.gpu_index,
        "architectures": [item.architecture_id for item in recipe.architectures],
        "training_seeds_per_architecture": list(recipe.training.seeds),
        "measured_training_runs": len(recipe.architectures) * len(recipe.training.seeds),
        "epochs_per_run": recipe.training.epochs,
        "instances_per_epoch": recipe.training.instances_per_epoch,
        "training_batch_size": recipe.training.batch_size,
        "output_root": recipe.output_root,
        "maximum_seed_walltime_s": recipe.maximum_seed_walltime_s,
        "maximum_campaign_walltime_s": recipe.maximum_campaign_walltime_s,
        "classification": CLASSIFICATION,
    }


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _preflight_qualification_checks(
    recipe: TrainingDebtRecipe,
    preflight: dict[str, Any],
    current_git: dict[str, Any],
) -> dict[str, bool]:
    readiness = _object(preflight.get("readiness"))
    system = _object(preflight.get("system"))
    runtime = _object(preflight.get("runtime"))
    recorded_git = _object(preflight.get("git"))
    backends = _object(preflight.get("backends"))
    emi = _object(backends.get("windows_emi"))
    nvml = _object(backends.get("nvml"))
    device = _object(nvml.get("selected_device"))
    recorded_sha = recorded_git.get("sha")
    recorded_fingerprint = recorded_git.get("worktree_fingerprint_sha256")

    return {
        "host_id_matches": preflight.get("host_id") == recipe.host_id,
        "windows_native": system.get("execution_layer") == "windows-native",
        "codecarbon_3_3_1": runtime.get("codecarbon") == "3.3.1",
        "exploratory_software_ready": readiness.get("exploratory_software_ready") is True,
        "windows_emi_active": emi.get("active_probe") is True,
        "windows_emi_available": emi.get("available") is True,
        "windows_emi_no_fallback": emi.get("fallback_used") is False,
        "windows_emi_counter_positive": emi.get("counter_positive") is True,
        "nvml_selection_valid": nvml.get("selection_valid") is True,
        "nvml_index_matches": nvml.get("selected_index") == recipe.gpu_index,
        "gpu_identity_matches": device.get("device_id_sha256") == recipe.gpu_device_id_sha256,
        "nvml_total_energy_counter": device.get("measurement_mode") == "total_energy_counter",
        "nvml_counter_monotonic": device.get("counter_monotonic") is True,
        "nvml_counter_positive": device.get("counter_positive") is True,
        "git_snapshot_available": recorded_git.get("available") is True
        and current_git.get("available") is True,
        "git_commit_matches": isinstance(recorded_sha, str)
        and current_git.get("sha") == recorded_sha,
        "worktree_fingerprint_matches": isinstance(recorded_fingerprint, str)
        and current_git.get("worktree_fingerprint_sha256") == recorded_fingerprint,
    }


def runtime_qualification(
    recipe: TrainingDebtRecipe,
    *,
    repository_root: Path | None = None,
    active_architecture_probe: bool = True,
) -> dict[str, Any]:
    """Qualify sources, component counters, and both model backbones.

    The architecture probe is intentionally unmeasured.  It catches an absent
    or incompatible GNN adapter before the operator provides the EXCLUSIVE
    attestation and before any multi-hour training run starts.
    """

    root = (Path.cwd() if repository_root is None else repository_root).resolve()
    errors: list[str] = []
    source_checks: list[dict[str, Any]] = []
    architecture_checks: dict[str, dict[str, Any]] = {}

    def check_file(path: Path, expected: str, label: str) -> None:
        import hashlib

        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            errors.append(f"{label} unavailable: {exc}")
            source_checks.append({"label": label, "path": path.as_posix(), "valid": False})
            return
        valid = digest == expected
        source_checks.append(
            {
                "label": label,
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "expected_sha256": expected,
                "valid": valid,
            }
        )
        if not valid:
            errors.append(f"{label} SHA-256 changed")

    check_file(root / recipe.base_recipe.path, recipe.base_recipe.sha256, "base_recipe")
    quality_root = root.joinpath(*PurePosixPath(recipe.quality_source_root).parts)
    for artifact in recipe.quality_source_artifacts:
        check_file(quality_root / artifact.path, artifact.sha256, f"quality_source:{artifact.path}")
    for label, artifact in (
        ("quality_corpus", recipe.corpus),
        ("quality_reference", recipe.reference),
        ("quality_reference_lock", recipe.reference_lock),
    ):
        check_file(quality_root / artifact.path, artifact.sha256, label)

    preflight_path = root.joinpath(*PurePosixPath(recipe.preflight_report).parts)
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        preflight = {}
        errors.append(f"qualified preflight unavailable: {exc}")
    if not isinstance(preflight, dict):
        preflight = {}
        errors.append("qualified preflight is not a JSON object")
    current_git = software_recipe._current_git_snapshot((preflight_path,))
    preflight_checks = _preflight_qualification_checks(recipe, preflight, current_git)
    preflight_ready = all(preflight_checks.values())
    if not preflight_ready:
        errors.extend(
            f"preflight check failed: {name}"
            for name, passed in preflight_checks.items()
            if not passed
        )

    if sys.platform != "win32":
        errors.append("training debt must run from native Windows Python")
    elif active_architecture_probe:
        try:
            import torch
            from torch_geometric.nn import TransformerConv

            from neuro_co.core.factory import make_algo, make_env, make_model
            from neuro_co.core.models import GNNEncoder, PointerDecoder

            if not torch.cuda.is_available() or recipe.gpu_index >= torch.cuda.device_count():
                raise RuntimeError("selected CUDA device is unavailable")
            torch.cuda.set_device(recipe.gpu_index)
            for index, architecture in enumerate(recipe.architectures):
                seed = 34_191 + index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                env = make_env("cvrp", size=50, capacity=40.0, max_demand=9)
                model = make_model(
                    env,
                    backbone=architecture.backbone,
                    hidden_dim=architecture.hidden_dim,
                    num_layers=architecture.num_layers,
                    num_heads=architecture.num_heads,
                )
                encoder = getattr(model, "encoder", None)
                if architecture.architecture_id == "gnn":
                    if not isinstance(encoder, GNNEncoder):
                        raise RuntimeError("GNN encoder class changed")
                    if not encoder.blocks or not all(
                        isinstance(block.gnn, TransformerConv) for block in encoder.blocks
                    ):
                        raise RuntimeError("GNN message-passing backend changed")
                    if not isinstance(getattr(model, "decoder", None), PointerDecoder):
                        raise RuntimeError("GNN pointer decoder changed")
                    try:
                        import importlib.metadata

                        torch_geometric_version = importlib.metadata.version("torch-geometric")
                    except importlib.metadata.PackageNotFoundError as exc:
                        raise RuntimeError("torch-geometric is absent") from exc
                    backend_metadata = {
                        "encoder_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
                        "layer_backend": "torch_geometric.TransformerConv",
                        "edge_index_backend": "torch_cdist_topk",
                        "encoder_attributes": {
                            "in_dim": encoder.in_dim,
                            "hidden_dim": encoder.hidden_dim,
                            "num_layers": encoder.num_layers,
                            "num_heads": encoder.num_heads,
                            "k_sparse": encoder.k_sparse,
                            "dropout": encoder.dropout,
                            "rbf_k": encoder.rbf_k,
                            "fourier_feats": encoder.fourier_feats,
                            "residual": encoder.residual,
                        },
                    }
                else:
                    backend_metadata = {
                        "encoder_class": type(encoder).__qualname__,
                        "layer_backend": "torch",
                    }
                    torch_geometric_version = None
                algo = make_algo(
                    recipe.training.algorithm,
                    model,
                    env,
                    device=f"cuda:{recipe.gpu_index}",
                    batch_size=recipe.training.batch_size,
                    n_starts=recipe.training.n_starts,
                    lr=recipe.training.learning_rate,
                    optimizer=recipe.training.optimizer,
                    weight_decay=recipe.training.weight_decay,
                    grad_clip=recipe.training.gradient_clip,
                    precision=recipe.training.precision,
                    eval_batch_size=4,
                    eval_augment=1,
                    lr_warmup_steps=0,
                    lr_total_steps=0,
                )
                generator = torch.Generator(device=f"cuda:{recipe.gpu_index}").manual_seed(seed)
                algo.train_step(generator)
                torch.cuda.synchronize(recipe.gpu_index)
                durations: list[float] = []
                for _ in range(recipe.eta_calibration_steps):
                    started = time.perf_counter()
                    metrics = algo.train_step(generator)
                    torch.cuda.synchronize(recipe.gpu_index)
                    durations.append(time.perf_counter() - started)
                    if any(not math.isfinite(float(metric)) for metric in metrics.values()):
                        raise RuntimeError("architecture calibration produced a non-finite metric")
                durations.sort()
                median_step_s = durations[len(durations) // 2]
                parameter_count = sum(parameter.numel() for parameter in model.parameters())
                architecture_checks[architecture.architecture_id] = {
                    "ready": True,
                    "backbone": architecture.backbone,
                    "encoder_type": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
                    "backend_metadata": backend_metadata,
                    "torch_geometric_version": torch_geometric_version,
                    "parameter_count": parameter_count,
                    "calibration_steps": recipe.eta_calibration_steps,
                    "median_optimizer_step_s": median_step_s,
                }
                del algo, model, env
                torch.cuda.empty_cache()
        except Exception as exc:
            errors.append(
                f"AM/GNN CUDA architecture qualification failed: {type(exc).__name__}: {exc}"
            )

    if set(architecture_checks) == set(SUPPORTED_ARCHITECTURES):
        am_step = architecture_checks["am"]["median_optimizer_step_s"]
        optimizer_steps = recipe.training.epochs * math.ceil(
            recipe.training.instances_per_epoch / recipe.training.batch_size
        )
        for check in architecture_checks.values():
            ratio = check["median_optimizer_step_s"] / am_step
            check["estimated_seed_walltime_s_from_am_anchor"] = 3526.0 * ratio
            check["optimizer_steps_per_seed"] = optimizer_steps

    return {
        "schema_version": "aet-training-debt-qualification/v1",
        "ready_to_execute": not errors,
        "preflight_ready": preflight_ready,
        "preflight_checks": preflight_checks,
        "source_checks": source_checks,
        "architecture_checks": architecture_checks,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        recipe = load_aet_training_debt_recipe(args.recipe)
    except TrainingDebtRecipeValidationError as exc:
        print(f"training-debt recipe invalid: {exc}", file=sys.stderr)
        return 2
    summary = recipe_summary(recipe)
    qualification = runtime_qualification(recipe) if args.require_ready else None
    summary["qualification"] = qualification
    summary["ready_to_execute"] = (
        qualification["ready_to_execute"] if qualification is not None else sys.platform == "win32"
    )
    if args.require_ready and not summary["ready_to_execute"]:
        summary["status"] = "valid_not_qualified"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CLASSIFICATION",
    "KIND",
    "SCHEMA_VERSION",
    "Architecture",
    "Artifact",
    "QualityGate",
    "Training",
    "TrainingDebtRecipe",
    "TrainingDebtRecipeValidationError",
    "load_aet_training_debt_recipe",
    "recipe_summary",
    "runtime_qualification",
]
