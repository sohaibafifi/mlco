"""Closed 100-epoch AM/GNN training and prospective-selection recipe.

Version two is intentionally separate from the 40-epoch pilot.  It trains ten
models, selects one checkpoint per model on a frozen development corpus, and
never reads the later deployment holdout.  CPU-package and GPU counters remain
component measurements, not whole-system energy or carbon accounting.
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

from neuro_co.aet.experiments.training_debt_recipe import (
    Architecture,
    Artifact,
    Training,
    _architecture,
    _artifact,
    _mapping,
    _number,
    _sha,
    _string,
)

SCHEMA_VERSION = "aet-journal-training-debt-v2/v1"
KIND = "aet-journal-training-debt-v2"
OUTPUT_ROOT = "experiments/aet-journal/raw/training-debt-v2/cvrp50-epoch100-seeds2-6"
GPU_DEVICE_ID_SHA256 = "686de95a427cb6c0734302fcaf694ddb9c37ba897579d9a66b75c838efeb04cf"
SUPPORTED_ARCHITECTURES = ("am", "gnn")
CANDIDATE_EPOCHS = tuple(range(90, 101))
CHECKPOINT_EPOCHS = (*range(0, 90, 10), *CANDIDATE_EPOCHS)

CLASSIFICATION: dict[str, Any] = {
    "purpose": "software_exploratory_training_and_prospective_selection",
    "scientific_use": False,
    "aet_eligible": False,
    "whole_system_energy": False,
    "cross_solver_energy_comparable": False,
    "carbon_accounting": "none",
    "final_holdout_used": False,
}

_SAFE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class TrainingDebtV2RecipeValidationError(ValueError):
    """Raised when a v2 recipe differs from the frozen protocol."""


@dataclass(frozen=True, slots=True)
class Selection:
    mode_id: str
    candidate_epochs: tuple[int, ...]
    criterion: str
    tie_break: str
    n_starts: int
    augmentations: int
    forced_first_actions: bool
    batch_size: int
    inference_precision: str


@dataclass(frozen=True, slots=True)
class TrainingDebtV2Recipe:
    name: str
    host_id: str
    gpu_index: int
    gpu_device_id_sha256: str
    output_root: str
    preflight_report: str
    base_recipe: Artifact
    selection_source_root: str
    selection_source_status: str
    selection_source_artifacts: tuple[Artifact, ...]
    corpus: Artifact
    corpus_content_sha256: str
    reference: Artifact
    reference_lock: Artifact
    architectures: tuple[Architecture, ...]
    training: Training
    selection: Selection
    minimum_training_duration_s: float
    minimum_selection_duration_s: float
    maximum_seed_walltime_s: float
    maximum_campaign_walltime_s: float
    attestation_max_age_s: float
    eta_calibration_steps: int

    # These aliases allow reuse of the tested v1 hardware/source qualification
    # helpers while keeping the public v2 terminology explicit.
    @property
    def quality_source_root(self) -> str:
        return self.selection_source_root

    @property
    def quality_source_status(self) -> str:
        return self.selection_source_status

    @property
    def quality_source_artifacts(self) -> tuple[Artifact, ...]:
        return self.selection_source_artifacts

    @property
    def minimum_measured_duration_s(self) -> float:
        return self.minimum_training_duration_s


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    if set(value) != keys:
        missing = sorted(keys - set(value))
        unknown = sorted(set(value) - keys)
        raise TrainingDebtV2RecipeValidationError(
            f"{where} keys differ; missing={missing}, unknown={unknown}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if type(value) is not type(expected) or value != expected:
        raise TrainingDebtV2RecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TrainingDebtV2RecipeValidationError(f"{where} must be an integer >= {minimum}")
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
        raise TrainingDebtV2RecipeValidationError(f"{where} must be a safe relative POSIX path")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrainingDebtV2RecipeValidationError(f"cannot load recipe: {exc}") from exc
    if not isinstance(raw, dict):
        raise TrainingDebtV2RecipeValidationError("recipe must be a mapping")
    return raw


def load_aet_training_debt_v2_recipe(path: Path) -> TrainingDebtV2Recipe:
    value = _load_yaml(path)
    _closed(
        value,
        {
            "schema_version",
            "kind",
            "name",
            "classification",
            "platform",
            "output_root",
            "qualification",
            "base_recipe",
            "selection_source",
            "architectures",
            "training",
            "selection",
            "measurement",
            "limits",
        },
        "recipe",
    )
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
    expected_qualification = {
        "require_git_clean": False,
        "require_windows_emi": True,
        "require_nvml_total_energy_counter": True,
        "allow_energy_fallback": False,
    }
    for key, expected in expected_qualification.items():
        _exact(qualification[key], expected, f"recipe.qualification.{key}")

    source = _mapping(value["selection_source"], "recipe.selection_source")
    _closed(
        source,
        {
            "root",
            "expected_status",
            "split",
            "artifacts",
            "corpus",
            "corpus_content_sha256",
            "reference",
            "reference_lock",
        },
        "recipe.selection_source",
    )
    _exact(source["split"], "development", "recipe.selection_source.split")
    source_root = _path(source["root"], "recipe.selection_source.root")
    if "holdout" in source_root.casefold():
        raise TrainingDebtV2RecipeValidationError("selection source must not be a holdout path")
    artifacts_raw = source["artifacts"]
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise TrainingDebtV2RecipeValidationError(
            "recipe.selection_source.artifacts must be non-empty"
        )
    artifacts = tuple(
        _artifact(item, f"recipe.selection_source.artifacts[{index}]")
        for index, item in enumerate(artifacts_raw)
    )
    if len({item.path for item in artifacts}) != len(artifacts):
        raise TrainingDebtV2RecipeValidationError("selection source has duplicate artifacts")

    architectures_raw = value["architectures"]
    if not isinstance(architectures_raw, list):
        raise TrainingDebtV2RecipeValidationError("recipe.architectures must be a list")
    architectures = tuple(
        _architecture(item, f"recipe.architectures[{index}]")
        for index, item in enumerate(architectures_raw)
    )
    if tuple(item.architecture_id for item in architectures) != SUPPORTED_ARCHITECTURES:
        raise TrainingDebtV2RecipeValidationError("recipe.architectures must be [am, gnn]")

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
    expected_training = {
        "algorithm": "pomo",
        "epochs": 100,
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
    scheduler = _mapping(training_raw["scheduler"], "recipe.training.scheduler")
    _closed(scheduler, {"name", "milestones", "gamma"}, "recipe.training.scheduler")
    milestones = tuple(scheduler["milestones"]) if isinstance(scheduler["milestones"], list) else ()
    checkpoints = (
        tuple(training_raw["checkpoint_epochs"])
        if isinstance(training_raw["checkpoint_epochs"], list)
        else ()
    )
    _exact(scheduler["name"], "multistep-epoch", "recipe.training.scheduler.name")
    _exact(milestones, (90, 95), "recipe.training.scheduler.milestones")
    _exact(scheduler["gamma"], 0.1, "recipe.training.scheduler.gamma")
    _exact(checkpoints, CHECKPOINT_EPOCHS, "recipe.training.checkpoint_epochs")
    training = Training(
        algorithm="pomo",
        seeds=seeds,
        epochs=100,
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

    selection_raw = _mapping(value["selection"], "recipe.selection")
    _closed(
        selection_raw,
        {
            "mode_id",
            "candidate_epochs",
            "criterion",
            "tie_break",
            "n_starts",
            "augmentations",
            "forced_first_actions",
            "batch_size",
            "inference_precision",
        },
        "recipe.selection",
    )
    selection_values = {
        "mode_id": "greedy-1x1",
        "candidate_epochs": list(CANDIDATE_EPOCHS),
        "criterion": "lowest_mean_gap_pct",
        "tie_break": "earliest_epoch",
        "n_starts": 1,
        "augmentations": 1,
        "forced_first_actions": False,
        "batch_size": 256,
        "inference_precision": "fp32",
    }
    for key, expected in selection_values.items():
        _exact(selection_raw[key], expected, f"recipe.selection.{key}")
    selection = Selection(
        mode_id="greedy-1x1",
        candidate_epochs=CANDIDATE_EPOCHS,
        criterion="lowest_mean_gap_pct",
        tie_break="earliest_epoch",
        n_starts=1,
        augmentations=1,
        forced_first_actions=False,
        batch_size=256,
        inference_precision="fp32",
    )

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
            "minimum_training_duration_s",
            "minimum_selection_duration_s",
            "maximum_seed_walltime_s",
            "maximum_campaign_walltime_s",
            "attestation_max_age_s",
            "eta_calibration_steps",
        },
        "recipe.limits",
    )
    minimum_training = _number(
        limits["minimum_training_duration_s"],
        "recipe.limits.minimum_training_duration_s",
        positive=True,
    )
    minimum_selection = _number(
        limits["minimum_selection_duration_s"],
        "recipe.limits.minimum_selection_duration_s",
        positive=True,
    )
    maximum_seed = _number(
        limits["maximum_seed_walltime_s"],
        "recipe.limits.maximum_seed_walltime_s",
        positive=True,
    )
    maximum_campaign = _number(
        limits["maximum_campaign_walltime_s"],
        "recipe.limits.maximum_campaign_walltime_s",
        positive=True,
    )
    attestation_max = _number(
        limits["attestation_max_age_s"],
        "recipe.limits.attestation_max_age_s",
        positive=True,
    )
    calibration_steps = _integer(
        limits["eta_calibration_steps"], "recipe.limits.eta_calibration_steps", minimum=1
    )
    if (
        minimum_training != 60.0
        or minimum_selection != 1.0
        or maximum_seed != 36_000.0
        or maximum_campaign != 216_000.0
        or attestation_max != 216_000.0
        or calibration_steps != 3
    ):
        raise TrainingDebtV2RecipeValidationError("recipe.limits differs from the frozen campaign")

    name = _string(value["name"], "recipe.name")
    if _SAFE.fullmatch(name) is None:
        raise TrainingDebtV2RecipeValidationError("recipe.name is not a safe identifier")
    return TrainingDebtV2Recipe(
        name=name,
        host_id=platform["host_id"],
        gpu_index=platform["gpu_index"],
        gpu_device_id_sha256=platform["gpu_device_id_sha256"],
        output_root=OUTPUT_ROOT,
        preflight_report=_path(
            qualification["preflight_report"], "recipe.qualification.preflight_report"
        ),
        base_recipe=_artifact(value["base_recipe"], "recipe.base_recipe"),
        selection_source_root=source_root,
        selection_source_status=_string(
            source["expected_status"], "recipe.selection_source.expected_status"
        ),
        selection_source_artifacts=artifacts,
        corpus=_artifact(source["corpus"], "recipe.selection_source.corpus"),
        corpus_content_sha256=_sha(
            source["corpus_content_sha256"],
            "recipe.selection_source.corpus_content_sha256",
        ),
        reference=_artifact(source["reference"], "recipe.selection_source.reference"),
        reference_lock=_artifact(
            source["reference_lock"], "recipe.selection_source.reference_lock"
        ),
        architectures=architectures,
        training=training,
        selection=selection,
        minimum_training_duration_s=minimum_training,
        minimum_selection_duration_s=minimum_selection,
        maximum_seed_walltime_s=maximum_seed,
        maximum_campaign_walltime_s=maximum_campaign,
        attestation_max_age_s=attestation_max,
        eta_calibration_steps=calibration_steps,
    )


def recipe_summary(recipe: TrainingDebtV2Recipe) -> dict[str, Any]:
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
        "checkpoint_selection": {
            "mode_id": recipe.selection.mode_id,
            "candidate_epochs": list(recipe.selection.candidate_epochs),
            "criterion": recipe.selection.criterion,
            "tie_break": recipe.selection.tie_break,
            "source_split": "development",
            "final_holdout_used": False,
        },
        "output_root": recipe.output_root,
        "maximum_seed_walltime_s": recipe.maximum_seed_walltime_s,
        "maximum_campaign_walltime_s": recipe.maximum_campaign_walltime_s,
        "classification": CLASSIFICATION,
    }


def runtime_qualification(
    recipe: TrainingDebtV2Recipe,
    *,
    repository_root: Path | None = None,
    active_architecture_probe: bool = True,
) -> dict[str, Any]:
    """Use the v1 hardware probe, then correct its ETA to 100 epochs."""

    from neuro_co.aet.experiments import training_debt_runner as v1

    result = v1.runtime_qualification(
        recipe,  # type: ignore[arg-type]
        repository_root=repository_root,
        active_architecture_probe=active_architecture_probe,
    )
    for check in result.get("architecture_checks", {}).values():
        estimate = check.get("estimated_seed_walltime_s_from_am_anchor")
        if isinstance(estimate, (int, float)) and math.isfinite(float(estimate)):
            check["estimated_seed_walltime_s_from_am_anchor"] = float(estimate) * 2.5
            check["estimate_scaled_from_v1_epochs"] = {"from": 40, "to": 100}
    result["schema_version"] = "aet-training-debt-v2-runtime-qualification/v1"
    result["final_holdout_used"] = False
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--check-runtime", action="store_true")
    parser.add_argument("--wait-for-preflight", action="store_true")
    parser.add_argument("--wait-timeout-s", type=float, default=0.0)
    parser.add_argument("--poll-interval-s", type=float, default=5.0)
    args = parser.parse_args(argv)
    deadline = time.monotonic() + max(0.0, args.wait_timeout_s)
    while True:
        try:
            recipe = load_aet_training_debt_v2_recipe(args.recipe)
            summary = recipe_summary(recipe)
            if args.check_runtime:
                qualification = runtime_qualification(recipe)
                summary["qualification"] = qualification
                summary["ready_to_execute"] = qualification["ready_to_execute"]
                if not qualification["ready_to_execute"]:
                    raise TrainingDebtV2RecipeValidationError("; ".join(qualification["errors"]))
            print(json.dumps(summary, indent=2, sort_keys=True))
            return 0
        except (TrainingDebtV2RecipeValidationError, OSError) as exc:
            if not args.wait_for_preflight or time.monotonic() >= deadline:
                print(f"training debt v2 recipe invalid: {exc}", file=sys.stderr)
                return 2
            time.sleep(max(0.1, args.poll_interval_s))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_EPOCHS",
    "CHECKPOINT_EPOCHS",
    "CLASSIFICATION",
    "Selection",
    "TrainingDebtV2Recipe",
    "TrainingDebtV2RecipeValidationError",
    "load_aet_training_debt_v2_recipe",
    "recipe_summary",
    "runtime_qualification",
]
