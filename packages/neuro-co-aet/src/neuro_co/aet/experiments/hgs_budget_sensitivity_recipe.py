"""Closed recipe for the same-corpus HGS budget-sensitivity sidecar.

The sidecar remeasures four fixed HGS budgets on the corpus used by the
completed AM/GNN batch frontier.  It remains software-exploratory evidence.
The budget-10 remeasurement is an engineering drift bridge: the new and old
surfaces may be combined only when costs reproduce exactly and mean CPU
package energy remains within the frozen relative band.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import yaml

from neuro_co.aet.experiments import software_recipe as software
from neuro_co.aet.experiments.deployment_energy_recipe import (
    GPU_DEVICE_ID_SHA256,
    QUALITY_SOURCE_ARTIFACTS,
    QUALITY_SOURCE_GIT_SHA,
    QUALITY_SOURCE_ROOT,
    QUALITY_SOURCE_SNAPSHOT_SHA256,
    Artifact,
    SourceBundle,
)

SCHEMA_VERSION = "aet-journal-hgs-budget-sensitivity/v1"
KIND = "aet-journal-hgs-budget-sensitivity"
OUTPUT_ROOT = "experiments/aet-journal/raw/hgs-budget-sensitivity/cvrp50-seed2723"
FRONTIER_SOURCE_ROOT = "experiments/aet-journal/raw/batch-frontier/cvrp50-seed2723"
FRONTIER_MANIFEST_SHA256 = "c12d0f68ecf72366f80277dc4ab88a3656a4aa199384f45b9ef730ee10f7426c"
FRONTIER_CHECKSUMS_SHA256 = "d3ec2b16cb0581db895a73b7ce4741c330c823f144efe0a2887abd0dc5cb0473"
FRONTIER_SUMMARY_SHA256 = "25de08c79144c0d434f40eefac5c2b46ea892295fe036dbffc774c167dc1cd55"
DATASET_CONTENT_SHA256 = "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
BUDGETS = (10, 30, 100, 300)
HGS_SEEDS = (50_101, 51_101, 52_101, 53_101, 54_101)
ROUND_ORDERS = (
    (10, 30, 100, 300),
    (300, 10, 30, 100),
    (30, 100, 300, 10),
    (100, 300, 10, 30),
    (300, 100, 30, 10),
)
CLASSIFICATION: dict[str, Any] = {
    "purpose": "software_exploratory",
    "scientific_use": False,
    "confirmatory_eligible": False,
    "whole_system_energy": False,
    "carbon_accounting": "none",
    "outcome_triggered_follow_up": "disclosed",
    "cross_solver_energy_comparability": "conditional_on_engineering_drift_bridge",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "qualification",
    "frontier_source",
    "quality_source",
    "dataset",
    "hgs_policy",
    "schedule",
    "measurement",
    "quality_gate",
    "engineering_drift_bridge",
    "limits",
}


class HGSBudgetSensitivityRecipeValidationError(ValueError):
    """Raised when the recipe differs from the frozen sidecar design."""


@dataclass(frozen=True, slots=True)
class FrontierSource:
    root: str
    manifest_schema: str
    expected_status: str
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True, slots=True)
class SensitivityDataset:
    dataset_id: str
    problem: str
    size: int
    capacity: float
    max_demand: int
    num_instances: int
    seed: int
    artifact: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class SensitivityHGSPolicy:
    solver: str
    budget_kind: str
    budgets: tuple[int, ...]
    scaling_factor: int
    collect_stats: bool
    cpu_threads: int
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MeasurementPolicy:
    backend: str
    domains: tuple[str, ...]
    minimum_block_duration_s: float
    duration_policy: str
    fallback: bool
    pue: float
    whole_system: bool


@dataclass(frozen=True, slots=True)
class FrozenQualityGate:
    metric: str
    threshold_pct: float
    comparison: str
    maximum_invalid_instances: int
    require_finite: bool
    require_each_seed_below_threshold: bool
    require_t_ucb_below_threshold: bool
    require_bootstrap_mean_ucb_below_threshold: bool
    require_bootstrap_q95_ucb_below_threshold: bool
    bootstrap_method: str
    bootstrap_replicates: int
    bootstrap_generator: str
    bootstrap_seed: int
    bootstrap_quantile: float
    t_critical_value: float
    t_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class EngineeringDriftBridge:
    remeasurement_budget: int
    require_exact_per_instance_cost_reproducibility: bool
    cpu_energy_metric: str
    max_relative_mean_cpu_energy_difference: float
    failure_effect: str


@dataclass(frozen=True, slots=True)
class SensitivityBlock:
    round_index: int
    order_index: int
    budget: int
    hgs_seed: int
    relative_path: str


@dataclass(frozen=True, slots=True)
class HGSBudgetSensitivityRecipe:
    name: str
    execution_layer: str
    host_id: str
    gpu_index: int
    gpu_device_id_sha256: str
    output_root: str
    preflight_report: str
    frontier_source: FrontierSource
    quality_source: SourceBundle
    dataset: SensitivityDataset
    hgs_policy: SensitivityHGSPolicy
    round_orders: tuple[tuple[int, ...], ...]
    measurement: MeasurementPolicy
    quality_gate: FrozenQualityGate
    drift_bridge: EngineeringDriftBridge
    minimum_block_duration_s: float
    maximum_block_walltime_s: float
    campaign_attestation_max_age_s: float
    maximum_campaign_walltime_s: float


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise HGSBudgetSensitivityRecipeValidationError(
            f"{where} keys differ; missing={missing!r}, unknown={unknown!r}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if type(value) is not type(expected) or value != expected:
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be finite and positive")
    return result


def _relative(value: Any, where: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be a relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} or not _SAFE.fullmatch(part) for part in posix.parts):
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} is not a safe relative path")
    if suffix is not None and posix.suffix != suffix:
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must end with {suffix}")
    return posix.as_posix()


def _artifact(value: Any, where: str) -> Artifact:
    raw = _mapping(value, where)
    _closed(raw, {"path", "sha256"}, where)
    path = _relative(raw["path"], f"{where}.path")
    sha256 = raw["sha256"]
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise HGSBudgetSensitivityRecipeValidationError(
            f"{where}.sha256 must be a lowercase SHA-256"
        )
    return Artifact(path, sha256)


def _integer_tuple(value: Any, where: str, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise HGSBudgetSensitivityRecipeValidationError(f"{where} must be a list")
    return tuple(_integer(item, f"{where}[]", minimum) for item in value)


def load_recipe(path: Path) -> HGSBudgetSensitivityRecipe:
    """Load and strictly validate the only authorized sensitivity recipe."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HGSBudgetSensitivityRecipeValidationError(
            f"cannot read recipe {path}: {exc}"
        ) from exc
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
    execution_layer = _exact(
        platform["execution_layer"], "windows-native", "platform.execution_layer"
    )
    host_id = _exact(platform["host_id"], "win-a4500-01", "platform.host_id")
    _exact(platform["accelerator_label"], "NVIDIA RTX A4500", "platform.accelerator_label")
    gpu_index = _exact(
        _integer(platform["gpu_index"], "platform.gpu_index"), 0, "platform.gpu_index"
    )
    gpu_hash = _exact(
        platform["gpu_device_id_sha256"],
        GPU_DEVICE_ID_SHA256,
        "platform.gpu_device_id_sha256",
    )

    qualification = _mapping(raw["qualification"], "qualification")
    _closed(qualification, {"preflight_report"}, "qualification")
    preflight_report = _exact(
        _relative(qualification["preflight_report"], "qualification.preflight_report", ".json"),
        "experiments/aet-journal/qualification/windows-hgs-budget-sensitivity-preflight.json",
        "qualification.preflight_report",
    )

    frontier_raw = _mapping(raw["frontier_source"], "frontier_source")
    _closed(
        frontier_raw,
        {"root", "manifest_schema", "expected_status", "artifacts"},
        "frontier_source",
    )
    frontier_artifacts_raw = frontier_raw["artifacts"]
    if not isinstance(frontier_artifacts_raw, list):
        raise HGSBudgetSensitivityRecipeValidationError("frontier_source.artifacts must be a list")
    frontier_artifacts = tuple(
        _artifact(item, "frontier_source.artifacts[]") for item in frontier_artifacts_raw
    )
    _exact(
        tuple((item.path, item.sha256) for item in frontier_artifacts),
        (
            ("manifest.json", FRONTIER_MANIFEST_SHA256),
            ("SHA256SUMS", FRONTIER_CHECKSUMS_SHA256),
            ("batch-frontier-summary.json", FRONTIER_SUMMARY_SHA256),
        ),
        "frontier_source.artifacts",
    )
    frontier_source = FrontierSource(
        root=_exact(
            _relative(frontier_raw["root"], "frontier_source.root"),
            FRONTIER_SOURCE_ROOT,
            "frontier_source.root",
        ),
        manifest_schema=_exact(
            frontier_raw["manifest_schema"],
            "aet-batch-frontier-manifest/v1",
            "frontier_source.manifest_schema",
        ),
        expected_status=_exact(
            frontier_raw["expected_status"], "complete", "frontier_source.expected_status"
        ),
        artifacts=frontier_artifacts,
    )

    quality_raw = _mapping(raw["quality_source"], "quality_source")
    _closed(
        quality_raw,
        {"root", "expected_status", "source_git_sha", "source_snapshot_sha256", "artifacts"},
        "quality_source",
    )
    quality_artifacts_raw = quality_raw["artifacts"]
    if not isinstance(quality_artifacts_raw, list):
        raise HGSBudgetSensitivityRecipeValidationError("quality_source.artifacts must be a list")
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
            _relative(quality_raw["root"], "quality_source.root"),
            QUALITY_SOURCE_ROOT,
            "quality_source.root",
        ),
        expected_status=_exact(
            quality_raw["expected_status"],
            "complete_confirmatory_passed",
            "quality_source.expected_status",
        ),
        source_git_sha=_exact(
            quality_raw["source_git_sha"], QUALITY_SOURCE_GIT_SHA, "quality_source.source_git_sha"
        ),
        source_snapshot_sha256=_exact(
            quality_raw["source_snapshot_sha256"],
            QUALITY_SOURCE_SNAPSHOT_SHA256,
            "quality_source.source_snapshot_sha256",
        ),
        artifacts=quality_artifacts,
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
    dataset = SensitivityDataset(
        dataset_id=_exact(dataset_raw["id"], "cvrp50-hgs10-confirmatory-seed2723", "dataset.id"),
        problem=_exact(dataset_raw["problem"], "cvrp", "dataset.problem"),
        size=_exact(_integer(dataset_raw["size"], "dataset.size", 1), 50, "dataset.size"),
        capacity=_exact(
            _number(dataset_raw["capacity"], "dataset.capacity", positive=True),
            40.0,
            "dataset.capacity",
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
        content_sha256=_exact(
            dataset_raw["content_sha256"], DATASET_CONTENT_SHA256, "dataset.content_sha256"
        ),
    )

    hgs_raw = _mapping(raw["hgs_policy"], "hgs_policy")
    _closed(
        hgs_raw,
        {
            "solver",
            "budget_kind",
            "budgets",
            "scaling_factor",
            "collect_stats",
            "cpu_threads",
            "seeds",
        },
        "hgs_policy",
    )
    budgets = _exact(
        _integer_tuple(hgs_raw["budgets"], "hgs_policy.budgets", 1), BUDGETS, "hgs_policy.budgets"
    )
    seeds = _exact(
        _integer_tuple(hgs_raw["seeds"], "hgs_policy.seeds"), HGS_SEEDS, "hgs_policy.seeds"
    )
    hgs_policy = SensitivityHGSPolicy(
        solver=_exact(hgs_raw["solver"], "pyvrp-hgs", "hgs_policy.solver"),
        budget_kind=_exact(hgs_raw["budget_kind"], "max_iterations", "hgs_policy.budget_kind"),
        budgets=budgets,
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
        seeds=seeds,
    )

    schedule_raw = _mapping(raw["schedule"], "schedule")
    _closed(schedule_raw, {"design", "round_orders"}, "schedule")
    _exact(
        schedule_raw["design"],
        "balanced_counter_rotating_latin_schedule",
        "schedule.design",
    )
    orders_raw = schedule_raw["round_orders"]
    if not isinstance(orders_raw, list):
        raise HGSBudgetSensitivityRecipeValidationError("schedule.round_orders must be a list")
    round_orders = tuple(
        _integer_tuple(order, f"schedule.round_orders[{index}]", 1)
        for index, order in enumerate(orders_raw)
    )
    _exact(round_orders, ROUND_ORDERS, "schedule.round_orders")
    for order in round_orders:
        if set(order) != set(BUDGETS) or len(order) != len(BUDGETS):
            raise HGSBudgetSensitivityRecipeValidationError(
                "every schedule round must contain each budget exactly once"
            )
    for position in range(len(BUDGETS)):
        counts = {
            budget: sum(order[position] == budget for order in round_orders) for budget in BUDGETS
        }
        if any(count not in {1, 2} for count in counts.values()):
            raise HGSBudgetSensitivityRecipeValidationError(
                "every budget must occur once or twice in each schedule position"
            )

    measurement_raw = _mapping(raw["measurement"], "measurement")
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
    _exact(measurement_raw, expected_measurement, "measurement")
    measurement = MeasurementPolicy(
        backend=str(measurement_raw["backend"]),
        domains=tuple(measurement_raw["domains"]),
        minimum_block_duration_s=120.0,
        duration_policy=str(measurement_raw["duration_policy"]),
        fallback=False,
        pue=1.0,
        whole_system=False,
    )

    gate_raw = _mapping(raw["quality_gate"], "quality_gate")
    expected_gate = {
        "metric": "mean_gap_to_locked_reference_pct",
        "threshold_pct": 5.0,
        "comparison": "strict_less_than",
        "maximum_invalid_instances": 0,
        "require_finite": True,
        "require_each_seed_below_threshold": True,
        "require_t_ucb_below_threshold": True,
        "require_bootstrap_mean_ucb_below_threshold": True,
        "require_bootstrap_q95_ucb_below_threshold": True,
        "bootstrap_method": "crossed_seed_by_instance_percentile",
        "bootstrap_replicates": 10_000,
        "bootstrap_generator": "numpy-pcg64",
        "bootstrap_seed": 3496,
        "bootstrap_quantile": 0.95,
        "t_critical_value": 2.13184678632665,
        "t_degrees_of_freedom": 4,
    }
    _exact(gate_raw, expected_gate, "quality_gate")
    quality_gate = FrozenQualityGate(**gate_raw)

    bridge_raw = _mapping(raw["engineering_drift_bridge"], "engineering_drift_bridge")
    expected_bridge = {
        "remeasurement_budget": 10,
        "require_exact_per_instance_cost_reproducibility": True,
        "cpu_energy_metric": "mean_cpu_package_j_per_instance",
        "max_relative_mean_cpu_energy_difference": 0.05,
        "failure_effect": "combined_surface_ineligible",
    }
    _exact(bridge_raw, expected_bridge, "engineering_drift_bridge")
    drift_bridge = EngineeringDriftBridge(**bridge_raw)

    limits_raw = _mapping(raw["limits"], "limits")
    expected_limits = {
        "maximum_block_walltime_s": 900,
        "campaign_attestation_max_age_s": 7200,
        "maximum_campaign_walltime_s": 7200,
    }
    _exact(limits_raw, expected_limits, "limits")
    return HGSBudgetSensitivityRecipe(
        name=_exact(raw["name"], "aet-hgs-budget-sensitivity-cvrp50-windows", "recipe.name"),
        execution_layer=execution_layer,
        host_id=host_id,
        gpu_index=gpu_index,
        gpu_device_id_sha256=gpu_hash,
        output_root=_exact(
            _relative(raw["output_root"], "output_root"), OUTPUT_ROOT, "output_root"
        ),
        preflight_report=preflight_report,
        frontier_source=frontier_source,
        quality_source=quality_source,
        dataset=dataset,
        hgs_policy=hgs_policy,
        round_orders=round_orders,
        measurement=measurement,
        quality_gate=quality_gate,
        drift_bridge=drift_bridge,
        minimum_block_duration_s=120.0,
        maximum_block_walltime_s=float(limits_raw["maximum_block_walltime_s"]),
        campaign_attestation_max_age_s=float(limits_raw["campaign_attestation_max_age_s"]),
        maximum_campaign_walltime_s=float(limits_raw["maximum_campaign_walltime_s"]),
    )


def expected_blocks(recipe: HGSBudgetSensitivityRecipe) -> tuple[SensitivityBlock, ...]:
    """Return the fixed twenty-block counter-rotating schedule."""

    blocks: list[SensitivityBlock] = []
    for round_index, order in enumerate(recipe.round_orders):
        for order_index, budget in enumerate(order):
            blocks.append(
                SensitivityBlock(
                    round_index=round_index,
                    order_index=order_index,
                    budget=budget,
                    hgs_seed=recipe.hgs_policy.seeds[round_index],
                    relative_path=(
                        f"blocks/round-{round_index:02d}/"
                        f"order-{order_index:02d}-hgs-i{budget:03d}.json"
                    ),
                )
            )
    return tuple(blocks)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_report(
    root: str, artifacts: tuple[Artifact, ...], workspace_root: Path
) -> dict[str, Any]:
    source_root = workspace_root.joinpath(*PurePosixPath(root).parts)
    valid = source_root.is_dir() and not source_root.is_symlink()
    records: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = source_root.joinpath(*PurePosixPath(artifact.path).parts)
        matches = path.is_file() and not path.is_symlink() and _sha256_file(path) == artifact.sha256
        records.append({"path": artifact.path, "sha256_matches": matches})
        valid = valid and matches
    return {"root": root, "valid": valid, "artifacts": records}


def _manifest_status(source_root: Path) -> dict[str, Any]:
    path = source_root / "manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"readable": False, "payload": None}
    return {
        "readable": isinstance(payload, dict),
        "payload": payload if isinstance(payload, dict) else None,
    }


def qualify_for_execution(
    recipe: HGSBudgetSensitivityRecipe,
    workspace_root: Path,
) -> dict[str, Any]:
    """Perform read-only native-host and frozen-source checks."""

    frontier = _source_report(
        recipe.frontier_source.root, recipe.frontier_source.artifacts, workspace_root
    )
    frontier_root = workspace_root.joinpath(*PurePosixPath(recipe.frontier_source.root).parts)
    frontier_manifest = _manifest_status(frontier_root)
    frontier_payload = frontier_manifest["payload"]
    frontier_semantics = bool(
        isinstance(frontier_payload, dict)
        and frontier_payload.get("schema_version") == recipe.frontier_source.manifest_schema
        and frontier_payload.get("status") == recipe.frontier_source.expected_status
        and frontier_payload.get("batch_frontier_summary_sha256") == FRONTIER_SUMMARY_SHA256
        and isinstance(frontier_payload.get("dataset"), dict)
        and frontier_payload["dataset"].get("content_sha256") == recipe.dataset.content_sha256
    )
    frontier["manifest_semantics_match"] = frontier_semantics
    frontier["valid"] = bool(frontier["valid"] and frontier_semantics)

    quality = _source_report(
        recipe.quality_source.root, recipe.quality_source.artifacts, workspace_root
    )
    quality_root = workspace_root.joinpath(*PurePosixPath(recipe.quality_source.root).parts)
    quality_manifest = _manifest_status(quality_root)
    quality_payload = quality_manifest["payload"]
    quality_status_matches = bool(
        isinstance(quality_payload, dict)
        and quality_payload.get("status") == recipe.quality_source.expected_status
    )
    quality["manifest_status_matches"] = quality_status_matches
    quality["valid"] = bool(quality["valid"] and quality_status_matches)

    probe = SimpleNamespace(
        preflight_report=recipe.preflight_report,
        host_id=recipe.host_id,
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    native = software._qualification_report(probe)  # type: ignore[arg-type]
    ready = bool(native["ready_to_execute"] and frontier["valid"] and quality["valid"])
    return {
        "native_windows_counters_ready": native["ready_to_execute"],
        "native": native,
        "frontier_source": frontier,
        "quality_source": quality,
        "ready_to_execute": ready,
    }


def dry_run(path: Path, workspace_root: Path | None = None) -> dict[str, Any]:
    """Validate and qualify the sidecar without writing or measuring."""

    recipe = load_recipe(path)
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    qualification = qualify_for_execution(recipe, root)
    blocks = expected_blocks(recipe)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "valid_ready_software_exploratory"
            if qualification["ready_to_execute"]
            else "valid_not_qualified"
        ),
        "classification": CLASSIFICATION,
        "name": recipe.name,
        "output_root": recipe.output_root,
        "budgets": list(recipe.hgs_policy.budgets),
        "rounds": len(recipe.round_orders),
        "block_count": len(blocks),
        "minimum_block_duration_s": recipe.minimum_block_duration_s,
        "minimum_measured_walltime_s": len(blocks) * recipe.minimum_block_duration_s,
        "maximum_campaign_walltime_s": recipe.maximum_campaign_walltime_s,
        "engineering_drift_bridge": {
            "remeasurement_budget": recipe.drift_bridge.remeasurement_budget,
            "required_for_combined_surface": True,
            "failure_effect": recipe.drift_bridge.failure_effect,
        },
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
    "BUDGETS",
    "CLASSIFICATION",
    "DATASET_CONTENT_SHA256",
    "FRONTIER_CHECKSUMS_SHA256",
    "FRONTIER_MANIFEST_SHA256",
    "FRONTIER_SUMMARY_SHA256",
    "HGS_SEEDS",
    "ROUND_ORDERS",
    "SCHEMA_VERSION",
    "EngineeringDriftBridge",
    "FrozenQualityGate",
    "HGSBudgetSensitivityRecipe",
    "HGSBudgetSensitivityRecipeValidationError",
    "SensitivityBlock",
    "dry_run",
    "expected_blocks",
    "load_recipe",
    "qualify_for_execution",
]
