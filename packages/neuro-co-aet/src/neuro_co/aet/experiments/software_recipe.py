"""Closed validation for the native-Windows exploratory AET software smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCHEMA_VERSION = "aet-journal-software-recipe/v1"
RECIPE_KIND = "aet-journal-software-smoke"
PREFLIGHT_SCHEMA_VERSION = "aet-preflight/v1"
SUPPORTED_CODECARBON_VERSION = "3.3.1"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "qualification",
    "measurement",
    "workload",
    "runs",
}
_CLASSIFICATION_KEYS = {"purpose", "scientific_use", "confirmatory_eligible"}
_PLATFORM_KEYS = {"host_id", "execution_layer", "gpu_index", "gpu_device_id_sha256"}
_QUALIFICATION_KEYS = {"preflight_report"}
_MEASUREMENT_KEYS = {
    "pue",
    "fallback",
    "cpu_primary_backend",
    "gpu_primary_backend",
    "scope",
    "minimum_block_duration_s",
    "cpu_attribution_scope",
    "gpu_attribution_scope",
    "exclusive_host_access_required",
    "exclusive_device_access_required",
    "positive_component_energy_required",
}
_WORKLOAD_KEYS = {
    "dataset",
    "model",
    "training",
    "inference",
    "baseline",
    "quality",
    "duration_policy",
}
_DATASET_KEYS = {
    "id",
    "problem",
    "size",
    "capacity",
    "max_demand",
    "num_instances",
    "seed",
    "artifact",
}
_MODEL_KEYS = {"architecture", "hidden_dim", "num_layers", "num_heads"}
_TRAINING_KEYS = {
    "run_id",
    "algorithm",
    "batch_size",
    "learning_rate",
    "minimum_steps",
    "checkpoint_artifact",
}
_INFERENCE_KEYS = {"run_id", "checkpoint_from", "batch_size"}
_BASELINE_KEYS = {"run_id", "solver", "max_iterations"}
_QUALITY_KEYS = {"metric", "maximum_gap_pct"}
_RUN_KEYS = {"id", "stage", "seed", "repetitions", "max_walltime_s", "output_subdir"}
_STAGES = {"training", "inference", "baseline"}
_EXACT_SCOPE = ["cpu_package", "gpu"]
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


class SoftwareRecipeValidationError(ValueError):
    """Raised when the software smoke recipe violates its closed schema."""


@dataclass(frozen=True)
class SoftwareRun:
    run_id: str
    stage: str
    seed: int
    repetitions: int
    max_walltime_s: float
    output_subdir: str


@dataclass(frozen=True)
class SharedDataset:
    dataset_id: str
    problem: str
    size: int
    capacity: float
    max_demand: int
    num_instances: int
    seed: int
    artifact: str


@dataclass(frozen=True)
class AttentionModelConfig:
    architecture: str
    hidden_dim: int
    num_layers: int
    num_heads: int


@dataclass(frozen=True)
class TrainingWorkload:
    run_id: str
    algorithm: str
    batch_size: int
    learning_rate: float
    minimum_steps: int
    checkpoint_artifact: str


@dataclass(frozen=True)
class InferenceWorkload:
    run_id: str
    checkpoint_from: str
    batch_size: int


@dataclass(frozen=True)
class BaselineWorkload:
    run_id: str
    solver: str
    max_iterations: int


@dataclass(frozen=True)
class QualityConstraint:
    metric: str
    maximum_gap_pct: float


@dataclass(frozen=True)
class SoftwareWorkload:
    dataset: SharedDataset
    model: AttentionModelConfig
    training: TrainingWorkload
    inference: InferenceWorkload
    baseline: BaselineWorkload
    quality: QualityConstraint
    duration_policy: str


@dataclass(frozen=True)
class AETSoftwareRecipe:
    name: str
    host_id: str
    gpu_index: int
    gpu_device_id_sha256: str
    output_root: str
    preflight_report: str
    measurement: dict[str, Any]
    workload: SoftwareWorkload
    runs: tuple[SoftwareRun, ...]


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SoftwareRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = sorted(repr(key) for key in set(value) - keys)
    if unknown:
        raise SoftwareRecipeValidationError(f"{where} has unknown keys: {', '.join(unknown)}")
    missing = sorted(keys - set(value))
    if missing:
        raise SoftwareRecipeValidationError(
            f"{where} is missing required keys: {', '.join(missing)}"
        )


def _slug(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise SoftwareRecipeValidationError(
            f"{where} must contain only lowercase letters, digits, '.', '_' and '-'"
        )
    if value.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
        raise SoftwareRecipeValidationError(f"{where} uses a Windows-reserved name")
    return value


def _safe_relative_path(value: Any, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SoftwareRecipeValidationError(
            f"{where} must be a non-empty relative path using '/' separators"
        )
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise SoftwareRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise SoftwareRecipeValidationError(f"{where} may not contain '.' or '..'")
    for part in posix_path.parts:
        _slug(part, f"{where} component")
    return posix_path


def _artifact_path(value: Any, where: str, suffix: str) -> str:
    path = _safe_relative_path(value, where)
    if path.suffix != suffix:
        raise SoftwareRecipeValidationError(f"{where} must identify a {suffix} file")
    return path.as_posix()


def _integer(value: Any, where: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SoftwareRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _positive_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SoftwareRecipeValidationError(f"{where} must be a positive finite number")
    return float(value)


def _nonnegative_number(value: Any, where: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise SoftwareRecipeValidationError(f"{where} must be a non-negative finite number")
    return float(value)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("AET software recipe validation needs PyYAML") from exc
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SoftwareRecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise SoftwareRecipeValidationError(f"invalid YAML in {path}: {exc}") from exc
    return _mapping(payload, "recipe")


def load_aet_software_recipe(path: Path) -> AETSoftwareRecipe:
    """Load the dedicated software-only recipe and reject every open field."""
    raw = _load_yaml(Path(path))
    _closed(raw, _TOP_LEVEL_KEYS, "recipe")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise SoftwareRecipeValidationError(
            f"recipe.schema_version must be exactly {SCHEMA_VERSION!r}"
        )
    if raw["kind"] != RECIPE_KIND:
        raise SoftwareRecipeValidationError(f"recipe.kind must be exactly {RECIPE_KIND!r}")
    name = _slug(raw["name"], "recipe.name")

    classification = _mapping(raw["classification"], "recipe.classification")
    _closed(classification, _CLASSIFICATION_KEYS, "recipe.classification")
    if classification["purpose"] != "software_exploratory":
        raise SoftwareRecipeValidationError(
            "recipe.classification.purpose must be exactly 'software_exploratory'"
        )
    if classification["scientific_use"] is not False:
        raise SoftwareRecipeValidationError("recipe.classification.scientific_use must be false")
    if classification["confirmatory_eligible"] is not False:
        raise SoftwareRecipeValidationError(
            "recipe.classification.confirmatory_eligible must be false"
        )

    platform = _mapping(raw["platform"], "recipe.platform")
    _closed(platform, _PLATFORM_KEYS, "recipe.platform")
    host_id = _slug(platform["host_id"], "recipe.platform.host_id")
    if platform["execution_layer"] != "windows-native":
        raise SoftwareRecipeValidationError(
            "recipe.platform.execution_layer must be exactly 'windows-native'"
        )
    gpu_index = _integer(platform["gpu_index"], "recipe.platform.gpu_index", 0)
    gpu_device_id = platform["gpu_device_id_sha256"]
    if not isinstance(gpu_device_id, str) or not _SHA256.fullmatch(gpu_device_id):
        raise SoftwareRecipeValidationError(
            "recipe.platform.gpu_device_id_sha256 must be a lowercase SHA-256 digest"
        )

    output_root = _safe_relative_path(raw["output_root"], "recipe.output_root")
    required_prefix = ("experiments", "aet-journal", "raw", "software-smoke")
    if output_root.parts[: len(required_prefix)] != required_prefix:
        raise SoftwareRecipeValidationError(
            "recipe.output_root must stay under experiments/aet-journal/raw/software-smoke/"
        )
    if len(output_root.parts) == len(required_prefix):
        raise SoftwareRecipeValidationError("recipe.output_root must name a run collection")

    qualification = _mapping(raw["qualification"], "recipe.qualification")
    _closed(qualification, _QUALIFICATION_KEYS, "recipe.qualification")
    preflight_path = _safe_relative_path(
        qualification["preflight_report"],
        "recipe.qualification.preflight_report",
    )
    if preflight_path.suffix != ".json":
        raise SoftwareRecipeValidationError(
            "recipe.qualification.preflight_report must identify a .json file"
        )

    measurement = _mapping(raw["measurement"], "recipe.measurement")
    _closed(measurement, _MEASUREMENT_KEYS, "recipe.measurement")
    if measurement["pue"] != 1.0 or isinstance(measurement["pue"], bool):
        raise SoftwareRecipeValidationError("recipe.measurement.pue must be exactly 1.0")
    if measurement["fallback"] is not False:
        raise SoftwareRecipeValidationError("recipe.measurement.fallback must be false")
    if measurement["cpu_primary_backend"] != "windows_emi":
        raise SoftwareRecipeValidationError(
            "recipe.measurement.cpu_primary_backend must be exactly 'windows_emi'"
        )
    if measurement["gpu_primary_backend"] != "nvml_total_energy_counter":
        raise SoftwareRecipeValidationError(
            "recipe.measurement.gpu_primary_backend must be exactly 'nvml_total_energy_counter'"
        )
    if measurement["scope"] != _EXACT_SCOPE:
        raise SoftwareRecipeValidationError(
            "recipe.measurement.scope must be exactly ['cpu_package', 'gpu']; "
            "whole_system_ac is forbidden"
        )
    if measurement["cpu_attribution_scope"] != "machine_wide":
        raise SoftwareRecipeValidationError(
            "recipe.measurement.cpu_attribution_scope must be exactly 'machine_wide'"
        )
    if measurement["gpu_attribution_scope"] != "device_wide":
        raise SoftwareRecipeValidationError(
            "recipe.measurement.gpu_attribution_scope must be exactly 'device_wide'"
        )
    if measurement["exclusive_host_access_required"] is not True:
        raise SoftwareRecipeValidationError(
            "recipe.measurement.exclusive_host_access_required must be true"
        )
    if measurement["exclusive_device_access_required"] is not True:
        raise SoftwareRecipeValidationError(
            "recipe.measurement.exclusive_device_access_required must be true"
        )
    if measurement["positive_component_energy_required"] is not True:
        raise SoftwareRecipeValidationError(
            "recipe.measurement.positive_component_energy_required must be true"
        )
    minimum_duration = _positive_number(
        measurement["minimum_block_duration_s"],
        "recipe.measurement.minimum_block_duration_s",
    )
    if minimum_duration < 120.0:
        raise SoftwareRecipeValidationError(
            "recipe.measurement.minimum_block_duration_s must be >= 120"
        )

    workload_raw = _mapping(raw["workload"], "recipe.workload")
    _closed(workload_raw, _WORKLOAD_KEYS, "recipe.workload")
    if workload_raw["duration_policy"] != "repeat_full_workload":
        raise SoftwareRecipeValidationError(
            "recipe.workload.duration_policy must be exactly 'repeat_full_workload'"
        )

    dataset_raw = _mapping(workload_raw["dataset"], "recipe.workload.dataset")
    _closed(dataset_raw, _DATASET_KEYS, "recipe.workload.dataset")
    dataset_problem = _slug(dataset_raw["problem"], "recipe.workload.dataset.problem")
    if dataset_problem != "cvrp":
        raise SoftwareRecipeValidationError(
            "recipe.workload.dataset.problem must be exactly 'cvrp'"
        )
    dataset_size = _integer(dataset_raw["size"], "recipe.workload.dataset.size", 1)
    if dataset_size != 20:
        raise SoftwareRecipeValidationError("recipe.workload.dataset.size must be exactly 20")
    dataset_capacity = _positive_number(dataset_raw["capacity"], "recipe.workload.dataset.capacity")
    if dataset_capacity != 30.0:
        raise SoftwareRecipeValidationError("recipe.workload.dataset.capacity must be exactly 30.0")
    dataset_max_demand = _integer(
        dataset_raw["max_demand"], "recipe.workload.dataset.max_demand", 1
    )
    if dataset_max_demand != 9:
        raise SoftwareRecipeValidationError("recipe.workload.dataset.max_demand must be exactly 9")
    dataset_num_instances = _integer(
        dataset_raw["num_instances"], "recipe.workload.dataset.num_instances", 1
    )
    if dataset_num_instances != 128:
        raise SoftwareRecipeValidationError(
            "recipe.workload.dataset.num_instances must be exactly 128"
        )
    dataset = SharedDataset(
        dataset_id=_slug(dataset_raw["id"], "recipe.workload.dataset.id"),
        problem=dataset_problem,
        size=dataset_size,
        capacity=dataset_capacity,
        max_demand=dataset_max_demand,
        num_instances=dataset_num_instances,
        seed=_integer(dataset_raw["seed"], "recipe.workload.dataset.seed", 0),
        artifact=_artifact_path(
            dataset_raw["artifact"], "recipe.workload.dataset.artifact", ".npz"
        ),
    )

    model_raw = _mapping(workload_raw["model"], "recipe.workload.model")
    _closed(model_raw, _MODEL_KEYS, "recipe.workload.model")
    if model_raw["architecture"] != "am":
        raise SoftwareRecipeValidationError(
            "recipe.workload.model.architecture must be exactly 'am'"
        )
    hidden_dim = _integer(model_raw["hidden_dim"], "recipe.workload.model.hidden_dim", 1)
    num_layers = _integer(model_raw["num_layers"], "recipe.workload.model.num_layers", 1)
    num_heads = _integer(model_raw["num_heads"], "recipe.workload.model.num_heads", 1)
    if (hidden_dim, num_layers, num_heads) != (32, 1, 4):
        raise SoftwareRecipeValidationError(
            "recipe.workload.model must use hidden_dim=32, num_layers=1 and num_heads=4"
        )
    model = AttentionModelConfig(
        architecture="am",
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
    )

    training_raw = _mapping(workload_raw["training"], "recipe.workload.training")
    _closed(training_raw, _TRAINING_KEYS, "recipe.workload.training")
    if training_raw["algorithm"] != "reinforce":
        raise SoftwareRecipeValidationError(
            "recipe.workload.training.algorithm must be exactly 'reinforce'"
        )
    training_batch_size = _integer(
        training_raw["batch_size"], "recipe.workload.training.batch_size", 1
    )
    if training_batch_size != 64:
        raise SoftwareRecipeValidationError(
            "recipe.workload.training.batch_size must be exactly 64"
        )
    minimum_steps = _integer(
        training_raw["minimum_steps"], "recipe.workload.training.minimum_steps", 1
    )
    if minimum_steps != 1:
        raise SoftwareRecipeValidationError(
            "recipe.workload.training.minimum_steps must be exactly 1"
        )
    training = TrainingWorkload(
        run_id=_slug(training_raw["run_id"], "recipe.workload.training.run_id"),
        algorithm="reinforce",
        batch_size=training_batch_size,
        learning_rate=_positive_number(
            training_raw["learning_rate"], "recipe.workload.training.learning_rate"
        ),
        minimum_steps=minimum_steps,
        checkpoint_artifact=_artifact_path(
            training_raw["checkpoint_artifact"],
            "recipe.workload.training.checkpoint_artifact",
            ".pt",
        ),
    )

    inference_raw = _mapping(workload_raw["inference"], "recipe.workload.inference")
    _closed(inference_raw, _INFERENCE_KEYS, "recipe.workload.inference")
    inference_batch_size = _integer(
        inference_raw["batch_size"], "recipe.workload.inference.batch_size", 1
    )
    if inference_batch_size != 128:
        raise SoftwareRecipeValidationError(
            "recipe.workload.inference.batch_size must be exactly 128"
        )
    inference = InferenceWorkload(
        run_id=_slug(inference_raw["run_id"], "recipe.workload.inference.run_id"),
        checkpoint_from=_slug(
            inference_raw["checkpoint_from"],
            "recipe.workload.inference.checkpoint_from",
        ),
        batch_size=inference_batch_size,
    )

    baseline_raw = _mapping(workload_raw["baseline"], "recipe.workload.baseline")
    _closed(baseline_raw, _BASELINE_KEYS, "recipe.workload.baseline")
    if baseline_raw["solver"] != "pyvrp-hgs":
        raise SoftwareRecipeValidationError(
            "recipe.workload.baseline.solver must be exactly 'pyvrp-hgs'"
        )
    baseline = BaselineWorkload(
        run_id=_slug(baseline_raw["run_id"], "recipe.workload.baseline.run_id"),
        solver="pyvrp-hgs",
        max_iterations=_integer(
            baseline_raw["max_iterations"],
            "recipe.workload.baseline.max_iterations",
            1,
        ),
    )

    quality_raw = _mapping(workload_raw["quality"], "recipe.workload.quality")
    _closed(quality_raw, _QUALITY_KEYS, "recipe.workload.quality")
    if quality_raw["metric"] != "mean_gap_to_baseline_pct":
        raise SoftwareRecipeValidationError(
            "recipe.workload.quality.metric must be exactly 'mean_gap_to_baseline_pct'"
        )
    maximum_gap_pct = _nonnegative_number(
        quality_raw["maximum_gap_pct"],
        "recipe.workload.quality.maximum_gap_pct",
    )
    if maximum_gap_pct != 10.0:
        raise SoftwareRecipeValidationError(
            "recipe.workload.quality.maximum_gap_pct must be exactly 10.0"
        )
    quality = QualityConstraint(
        metric="mean_gap_to_baseline_pct",
        maximum_gap_pct=maximum_gap_pct,
    )
    workload = SoftwareWorkload(
        dataset=dataset,
        model=model,
        training=training,
        inference=inference,
        baseline=baseline,
        quality=quality,
        duration_policy="repeat_full_workload",
    )

    raw_runs = raw["runs"]
    if not isinstance(raw_runs, list) or len(raw_runs) != 3:
        raise SoftwareRecipeValidationError("recipe.runs must contain exactly three runs")
    runs: list[SoftwareRun] = []
    seen_ids: set[str] = set()
    seen_outputs: set[str] = set()
    for index, raw_run in enumerate(raw_runs):
        where = f"recipe.runs[{index}]"
        run = _mapping(raw_run, where)
        _closed(run, _RUN_KEYS, where)
        run_id = _slug(run["id"], f"{where}.id")
        if run_id in seen_ids:
            raise SoftwareRecipeValidationError(f"duplicate run id: {run_id}")
        seen_ids.add(run_id)
        stage = run["stage"]
        if not isinstance(stage, str) or stage not in _STAGES:
            raise SoftwareRecipeValidationError(
                f"{where}.stage must be one of: {', '.join(sorted(_STAGES))}"
            )
        output_subdir = _safe_relative_path(run["output_subdir"], f"{where}.output_subdir")
        if output_subdir.parts[0] != stage:
            raise SoftwareRecipeValidationError(
                f"{where}.output_subdir must stay under the {stage}/ directory"
            )
        output_key = output_subdir.as_posix()
        if output_key in seen_outputs:
            raise SoftwareRecipeValidationError(f"duplicate run output_subdir: {output_key}")
        seen_outputs.add(output_key)
        max_walltime_s = _positive_number(run["max_walltime_s"], f"{where}.max_walltime_s")
        if max_walltime_s < minimum_duration:
            raise SoftwareRecipeValidationError(
                f"{where}.max_walltime_s must be >= minimum_block_duration_s"
            )
        repetitions = _integer(run["repetitions"], f"{where}.repetitions", 1)
        if repetitions != 1:
            raise SoftwareRecipeValidationError(f"{where}.repetitions must be exactly 1")
        runs.append(
            SoftwareRun(
                run_id=run_id,
                stage=stage,
                seed=_integer(run["seed"], f"{where}.seed", 0),
                repetitions=repetitions,
                max_walltime_s=max_walltime_s,
                output_subdir=output_key,
            )
        )

    runs_by_stage = {run.stage: run for run in runs}
    if set(runs_by_stage) != _STAGES:
        raise SoftwareRecipeValidationError(
            "recipe.runs must contain exactly one training, one inference and one baseline run"
        )
    expected_references = {
        "training": training.run_id,
        "inference": inference.run_id,
        "baseline": baseline.run_id,
    }
    for stage, expected_run_id in expected_references.items():
        if runs_by_stage[stage].run_id != expected_run_id:
            raise SoftwareRecipeValidationError(
                f"recipe.workload.{stage}.run_id must reference the {stage} run"
            )
    if inference.checkpoint_from != training.run_id:
        raise SoftwareRecipeValidationError(
            "recipe.workload.inference.checkpoint_from must reference the training run"
        )
    dataset_artifact = PurePosixPath(dataset.artifact)
    if dataset_artifact.parts[0] != "shared":
        raise SoftwareRecipeValidationError(
            "recipe.workload.dataset.artifact must stay under the shared/ directory"
        )
    checkpoint_artifact = PurePosixPath(training.checkpoint_artifact)
    training_output = PurePosixPath(runs_by_stage["training"].output_subdir)
    if checkpoint_artifact.parts[: len(training_output.parts)] != training_output.parts:
        raise SoftwareRecipeValidationError(
            "recipe.workload.training.checkpoint_artifact must stay under the training run output"
        )
    if dataset.seed == runs_by_stage["training"].seed:
        raise SoftwareRecipeValidationError(
            "recipe.workload.dataset.seed must differ from the training run seed"
        )

    return AETSoftwareRecipe(
        name=name,
        host_id=host_id,
        gpu_index=gpu_index,
        gpu_device_id_sha256=gpu_device_id,
        output_root=output_root.as_posix(),
        preflight_report=preflight_path.as_posix(),
        measurement=dict(measurement),
        workload=workload,
        runs=tuple(runs),
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _command(*args: str, cwd: Path, binary: bool = False) -> str | bytes:
    return subprocess.check_output(
        list(args),
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=not binary,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _current_git_snapshot(excluded: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Reproduce the preflight source fingerprint, allowing an unchanged dirty tree."""
    try:
        root = Path(
            str(_command("git", "rev-parse", "--show-toplevel", cwd=Path.cwd())).strip()
        ).resolve()
        sha = str(_command("git", "rev-parse", "HEAD", cwd=root)).strip()
        diff = _command("git", "diff", "--binary", "HEAD", cwd=root, binary=True)
        untracked = _command(
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            cwd=root,
            binary=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return {"available": False}

    excluded_paths = {path.resolve() for path in excluded}
    diff_bytes = diff if isinstance(diff, bytes) else diff.encode()
    untracked_bytes = untracked if isinstance(untracked, bytes) else untracked.encode()
    digest = hashlib.sha256()
    digest.update(diff_bytes)
    included_untracked: list[str] = []
    for raw_path in sorted(part for part in untracked_bytes.split(b"\0") if part):
        relative = raw_path.decode("utf-8", errors="surrogateescape")
        candidate = (root / relative).resolve()
        if candidate in excluded_paths or not candidate.is_file():
            continue
        included_untracked.append(relative)
        digest.update(relative.encode("utf-8", errors="surrogateescape"))
        digest.update(bytes.fromhex(_sha256_file(candidate)))
    return {
        "available": True,
        "sha": sha,
        "dirty": bool(diff_bytes) or bool(included_untracked),
        "worktree_fingerprint_sha256": digest.hexdigest(),
    }


def _qualification_report(recipe: AETSoftwareRecipe) -> dict[str, Any]:
    preflight_path = Path(recipe.preflight_report)
    preflight = _load_json(preflight_path)
    report = preflight or {}
    system = _dict(report.get("system"))
    runtime = _dict(report.get("runtime"))
    runtime_imports = _dict(runtime.get("imports"))
    git = _dict(report.get("git"))
    strict_tracker = _dict(report.get("strict_tracker"))
    readiness = _dict(report.get("readiness"))
    backends = _dict(report.get("backends"))
    emi = _dict(backends.get("windows_emi"))
    nvml = _dict(backends.get("nvml"))
    device = _dict(nvml.get("selected_device"))
    torch_cuda = _dict(backends.get("torch_cuda"))

    current_git = _current_git_snapshot((preflight_path,))
    codecarbon_version = runtime.get("codecarbon")
    codecarbon_ready = bool(
        codecarbon_version == SUPPORTED_CODECARBON_VERSION
        and emi.get("codecarbon_version") == codecarbon_version
        and emi.get("version_supported") is True
    )
    required_runtime_imports = (
        "neuro_co.aet",
        "neuro_co.aet.experiments.software_smoke_runner",
        "neuro_co.problems.cvrp.pyvrp",
    )
    runtime_ready = bool(
        runtime.get("python_supported") is True
        and all(
            _dict(runtime_imports.get(module)).get("importable") is True
            for module in required_runtime_imports
        )
    )
    strict_tracker_ready = strict_tracker.get("component_contract_supported") is True
    windows_ready = bool(
        system.get("execution_layer") == "windows-native"
        and isinstance(system.get("windows_build"), int)
        and not isinstance(system.get("windows_build"), bool)
        and system["windows_build"] >= 22000
    )
    emi_ready = bool(
        emi.get("active_probe") is True
        and emi.get("available") is True
        and emi.get("platform_windows") is True
        and emi.get("backend_mode") == "windows_emi"
        and emi.get("interface_class") == "WindowsEMI"
        and emi.get("fallback_used") is False
        and emi.get("measurement_scope") == "cpu_package"
        and emi.get("ram_included") is False
        and emi.get("counter_positive") is True
        and isinstance(emi.get("sample_energy_j"), (int, float))
        and not isinstance(emi.get("sample_energy_j"), bool)
        and math.isfinite(emi["sample_energy_j"])
        and emi["sample_energy_j"] > 0
    )
    gpu_identity_matches = bool(
        nvml.get("selected_index") == recipe.gpu_index
        and device.get("index") == recipe.gpu_index
        and device.get("device_id_sha256") == recipe.gpu_device_id_sha256
    )
    nvml_ready = bool(
        nvml.get("active_probe") is True
        and nvml.get("available") is True
        and nvml.get("selection_valid") is True
        and device.get("selected") is True
        and device.get("measurement_mode") == "total_energy_counter"
        and device.get("measurement_mode_supported") is True
        and device.get("total_energy_counter_supported") is True
        and device.get("counter_monotonic") is True
        and device.get("counter_positive") is True
        and isinstance(device.get("sample_energy_delta_mj"), (int, float))
        and not isinstance(device.get("sample_energy_delta_mj"), bool)
        and math.isfinite(device["sample_energy_delta_mj"])
        and device["sample_energy_delta_mj"] > 0
        and gpu_identity_matches
    )
    torch_cuda_ready = bool(
        torch_cuda.get("available") is True
        and torch_cuda.get("active_probe") is True
        and torch_cuda.get("selection_valid") is True
        and torch_cuda.get("identity_verified") is True
        and torch_cuda.get("visibility_remapped") is False
        and torch_cuda.get("requested_nvml_index") == recipe.gpu_index
        and torch_cuda.get("selected_cuda_index") == recipe.gpu_index
        and torch_cuda.get("compute_probe_passed") is True
    )
    source_snapshot_matches = bool(
        current_git.get("available") is True
        and git.get("available") is True
        and git.get("sha") == current_git.get("sha")
        and isinstance(git.get("worktree_fingerprint_sha256"), str)
        and _SHA256.fullmatch(git["worktree_fingerprint_sha256"])
        and git.get("worktree_fingerprint_sha256") == current_git.get("worktree_fingerprint_sha256")
    )
    host_linked = report.get("host_id") == recipe.host_id
    preflight_readiness_consistent = bool(
        readiness.get("exploratory_software_ready") is True
        and host_linked
        and windows_ready
        and runtime_ready
        and codecarbon_ready
        and strict_tracker_ready
        and emi_ready
        and nvml_ready
        and torch_cuda_ready
    )
    preflight_valid = bool(
        preflight
        and report.get("schema_version") == PREFLIGHT_SCHEMA_VERSION
        and host_linked
        and windows_ready
        and runtime_ready
        and codecarbon_ready
        and strict_tracker_ready
        and emi_ready
        and nvml_ready
        and torch_cuda_ready
        and preflight_readiness_consistent
        and source_snapshot_matches
    )
    return {
        "preflight_report": preflight_path.as_posix(),
        "preflight_valid": preflight_valid,
        "host_linked": host_linked,
        "gpu_identity_matches": gpu_identity_matches,
        "windows_11_native": windows_ready,
        "codecarbon_version": codecarbon_version,
        "runtime_ready": runtime_ready,
        "codecarbon_exactly_3_3_1": codecarbon_ready,
        "strict_tracker_component_contract_ready": strict_tracker_ready,
        "windows_emi_ready": emi_ready,
        "nvml_total_energy_counter_ready": nvml_ready,
        "torch_cuda_compute_ready": torch_cuda_ready,
        "preflight_readiness_consistent": preflight_readiness_consistent,
        "current_git_commit": current_git.get("sha"),
        "current_worktree_dirty": current_git.get("dirty"),
        "current_worktree_fingerprint_sha256": current_git.get("worktree_fingerprint_sha256"),
        "source_snapshot_matches": source_snapshot_matches,
        "ready_to_execute": preflight_valid,
    }


def qualify_for_execution(path: Path) -> tuple[AETSoftwareRecipe, dict[str, Any]]:
    """Return a validated recipe and its qualification without output or writes."""
    recipe = load_aet_software_recipe(path)
    return recipe, _qualification_report(recipe)


def dry_run(path: Path) -> dict[str, Any]:
    """Validate and print the exploratory bounds without executing or writing."""
    recipe, qualification = qualify_for_execution(path)
    run_count = sum(run.repetitions for run in recipe.runs)
    maximum_walltime_s = sum(run.repetitions * run.max_walltime_s for run in recipe.runs)
    ready = qualification["ready_to_execute"]
    report = {
        "status": "valid_ready_software_exploratory" if ready else "valid_not_qualified",
        "schema_version": SCHEMA_VERSION,
        "name": recipe.name,
        "purpose": "software_exploratory",
        "scientific_use": False,
        "confirmatory_eligible": False,
        "confirmatory_quality_status": "not_evaluable",
        "cross_solver_energy_comparable": False,
        "host_id": recipe.host_id,
        "execution_layer": "windows-native",
        "gpu_index": recipe.gpu_index,
        "output_root": recipe.output_root,
        "run_definitions": len(recipe.runs),
        "run_count": run_count,
        "maximum_single_run_walltime_s": max(run.max_walltime_s for run in recipe.runs),
        "maximum_walltime_s": maximum_walltime_s,
        "maximum_walltime_hours": maximum_walltime_s / 3600.0,
        "walltime_limit_enforcement": "cooperative_after_complete_work_units",
        "qualification": qualification,
        "ready_to_execute": ready,
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
