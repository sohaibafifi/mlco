"""Closed recipe for the native-Windows CVRP50 deployment-energy pilot.

The pilot measures two already fixed deployment policies on one fresh corpus.
It is deliberately exploratory: component counters are not whole-system
energy, the policies are not cross-solver energy comparable, and no AET or
carbon quantity may be computed from this recipe.
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

SCHEMA_VERSION = "aet-journal-deployment-energy-pilot/v1"
KIND = "aet-journal-deployment-energy-pilot"
GPU_DEVICE_ID_SHA256 = "686de95a427cb6c0734302fcaf694ddb9c37ba897579d9a66b75c838efeb04cf"
OUTPUT_ROOT = "experiments/aet-journal/raw/deployment-energy/cvrp50-seed2724"
QUALITY_SOURCE_ROOT = "experiments/aet-journal/raw/hgs10-confirmatory/cvrp50-seed2723"
CHECKPOINT_SOURCE_ROOT = "experiments/aet-journal/raw/quality-replication/cvrp50-epoch40-seeds2-6"
QUALITY_SOURCE_GIT_SHA = "e8a06e2ac5b9279ba4eccd3b3facede0a86794c0"
QUALITY_SOURCE_SNAPSHOT_SHA256 = "9e74a8cc0ee96f5cc2a3ee58d9aa653dde95c5a8d21e586e5113063680d4b899"
QUALITY_SOURCE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("manifest.json", "4920b5ce99523d568f274e6da7c4f8b143c4622c0c400b6a6d0bdc4dc2f38673"),
    ("SHA256SUMS", "c9d620adad2838a0fcd3dae573d0958ec051d57368689b122c2fd8ecf75d73c8"),
    ("run-state.json", "326ba38642a916da9c3ef17aa4f7a47182cf475ece0d63995e9f8c42c0946b16"),
    (
        "holdout-assessment.json",
        "1397cf0ee6cf0a779c0d918bcbbb123087ce451a2b55fc9f64ec19aaa0370416",
    ),
    ("recipe.yaml", "fe71cc6b9f1b3c8db5eb764cefc79c06fcfcbd5ec39a9137a1aae19ce4ad1387"),
    (
        "reference/reference-lock.json",
        "0490e17a0e26a3250c7303f141633ba9a4635655c275069f1e4a6bb96f96a381",
    ),
    (
        "shared/cvrp50-hgs10-confirmatory-seed2723.manifest.json",
        "9e8d436a0e9477533648abdcaef9c6fd9703f7751ecf08d781dc797b310aff1c",
    ),
    (
        "shared/cvrp50-hgs10-confirmatory-seed2723.npz",
        "4c29e404141c82f6b09fe9ec5c0ce5e79e5ad575a43939bb4d79521fd576c05f",
    ),
)
QUALITY_SOURCE_CORPUS_CONTENT_SHA256 = (
    "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
)
CHECKPOINT_SOURCE_GIT_SHA = "bf8d5c5dfb1a566019cd56135efa91f2e460ee6b"
CHECKPOINT_SOURCE_SNAPSHOT_SHA256 = (
    "3dcc187f5e875f6427e92c4c1a707461a101c8d37f1abaca029d2c03a9cd8e19"
)
CHECKPOINT_SOURCE_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("manifest.json", "df47d74fe890f35103e1aa761b925ca7c082f741af8f732c1b689129e971f49f"),
    ("SHA256SUMS", "e59d5fb1449b75f0439a06ce587e7129ba0f6e068f0a920dc7a3021e45f0202f"),
    (
        "replication-assessment.json",
        "7cc4b1b2e257c6bc5380684776f39347105c2fd647d5445d7c33a12e3774f320",
    ),
    ("recipe.yaml", "e0617bd5fd43057763128d62005b9f12ef1e26308191a6bc077dd803d6b4160c"),
    ("run-state.json", "afa50d5b27f7655e49e9fa3f35604a51ef83098969f777c11c4233db0d12a4bb"),
    (
        "reference/reference-lock.json",
        "90a418104cc98b55405ea19413c0ac85f990f3bb7117d710afd77e9ea2a9e57d",
    ),
)
CHECKPOINTS: tuple[tuple[int, str, str], ...] = (
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
HGS_SEEDS = (50_101, 51_101, 52_101, 53_101, 54_101)
ROUND_ORDERS = (
    ("neural", "hgs"),
    ("hgs", "neural"),
    ("neural", "hgs"),
    ("hgs", "neural"),
    ("neural", "hgs"),
)
CLASSIFICATION: dict[str, Any] = {
    "purpose": "software_exploratory",
    "scientific_use": False,
    "confirmatory_eligible": False,
    "cross_solver_energy_comparable": False,
    "whole_system_energy": False,
    "aet_eligible": False,
    "carbon_accounting": "none",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_TOP_KEYS = {
    "schema_version",
    "kind",
    "name",
    "classification",
    "platform",
    "output_root",
    "qualification",
    "quality_source",
    "checkpoint_source",
    "dataset",
    "neural_policy",
    "hgs_policy",
    "schedule",
    "measurement",
    "limits",
}


class DeploymentEnergyRecipeValidationError(ValueError):
    """Raised when the deployment-energy recipe is not the closed pilot."""


@dataclass(frozen=True, slots=True)
class Artifact:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Checkpoint:
    training_seed: int
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceBundle:
    root: str
    expected_status: str
    source_git_sha: str
    source_snapshot_sha256: str
    artifacts: tuple[Artifact, ...]


@dataclass(frozen=True, slots=True)
class Dataset:
    dataset_id: str
    problem: str
    size: int
    capacity: float
    max_demand: int
    num_instances: int
    seed: int
    artifact: str
    forbidden_content_sha256: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class NeuralPolicy:
    mode_id: str
    checkpoint_epoch: int
    n_starts: int
    augmentations: int
    forced_first_actions: bool
    batch_size: int
    inference_precision: str
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HGSPolicy:
    solver: str
    max_iterations: int
    scaling_factor: int
    collect_stats: bool
    cpu_threads: int
    seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EnergyBlock:
    round_index: int
    order_index: int
    policy: str
    training_seed: int
    evaluation_seed: int
    hgs_seed: int
    relative_path: str


@dataclass(frozen=True, slots=True)
class DeploymentEnergyRecipe:
    name: str
    host_id: str
    gpu_index: int
    gpu_device_id_sha256: str
    output_root: str
    preflight_report: str
    quality_source: SourceBundle
    checkpoint_source: SourceBundle
    checkpoints: tuple[Checkpoint, ...]
    dataset: Dataset
    neural_policy: NeuralPolicy
    hgs_policy: HGSPolicy
    round_orders: tuple[tuple[str, str], ...]
    minimum_block_duration_s: float
    maximum_block_walltime_s: float
    campaign_attestation_max_age_s: float
    maximum_campaign_walltime_s: float


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DeploymentEnergyRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    missing = sorted(keys - set(value))
    unknown = sorted(set(value) - keys)
    if missing or unknown:
        raise DeploymentEnergyRecipeValidationError(
            f"{where} keys differ; missing={missing!r}, unknown={unknown!r}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if value != expected or type(value) is not type(expected):
        raise DeploymentEnergyRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DeploymentEnergyRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DeploymentEnergyRecipeValidationError(f"{where} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise DeploymentEnergyRecipeValidationError(f"{where} must be a positive number")
    return result


def _relative(value: Any, where: str, suffix: str | None = None) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DeploymentEnergyRecipeValidationError(f"{where} must be a relative POSIX path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise DeploymentEnergyRecipeValidationError(f"{where} must be relative")
    if any(part in {"", ".", ".."} or not _SAFE.fullmatch(part) for part in posix.parts):
        raise DeploymentEnergyRecipeValidationError(f"{where} is not a safe relative path")
    if suffix is not None and posix.suffix != suffix:
        raise DeploymentEnergyRecipeValidationError(f"{where} must end with {suffix}")
    return posix.as_posix()


def _sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DeploymentEnergyRecipeValidationError(f"{where} must be a lowercase SHA-256")
    return value


def _artifact(value: Any, where: str) -> Artifact:
    raw = _mapping(value, where)
    _closed(raw, {"path", "sha256"}, where)
    return Artifact(_relative(raw["path"], f"{where}.path"), _sha(raw["sha256"], f"{where}.sha256"))


def _source(
    value: Any,
    where: str,
    *,
    expected_root: str,
    expected_status: str,
    expected_git: str,
    expected_snapshot: str,
    expected_artifacts: tuple[tuple[str, str], ...],
) -> SourceBundle:
    raw = _mapping(value, where)
    _closed(
        raw,
        {"root", "expected_status", "source_git_sha", "source_snapshot_sha256", "artifacts"},
        where,
    )
    artifacts_raw = raw["artifacts"]
    if not isinstance(artifacts_raw, list):
        raise DeploymentEnergyRecipeValidationError(f"{where}.artifacts must be a list")
    artifacts = tuple(_artifact(item, f"{where}.artifacts[]") for item in artifacts_raw)
    _exact(
        tuple((item.path, item.sha256) for item in artifacts),
        expected_artifacts,
        f"{where}.artifacts",
    )
    git_sha = raw["source_git_sha"]
    if not isinstance(git_sha, str) or _GIT_SHA.fullmatch(git_sha) is None:
        raise DeploymentEnergyRecipeValidationError(f"{where}.source_git_sha must be a Git SHA")
    return SourceBundle(
        root=_exact(_relative(raw["root"], f"{where}.root"), expected_root, f"{where}.root"),
        expected_status=_exact(raw["expected_status"], expected_status, f"{where}.expected_status"),
        source_git_sha=_exact(git_sha, expected_git, f"{where}.source_git_sha"),
        source_snapshot_sha256=_exact(
            _sha(raw["source_snapshot_sha256"], f"{where}.source_snapshot_sha256"),
            expected_snapshot,
            f"{where}.source_snapshot_sha256",
        ),
        artifacts=artifacts,
    )


def load_recipe(path: Path) -> DeploymentEnergyRecipe:
    """Load and strictly validate the one authorized pilot recipe."""

    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DeploymentEnergyRecipeValidationError(f"cannot read recipe {path}: {exc}") from exc
    raw = _mapping(raw, "recipe")
    _closed(raw, _TOP_KEYS, "recipe")
    _exact(raw["schema_version"], SCHEMA_VERSION, "recipe.schema_version")
    _exact(raw["kind"], KIND, "recipe.kind")

    classification = _mapping(raw["classification"], "classification")
    _closed(classification, set(CLASSIFICATION), "classification")
    _exact(classification, CLASSIFICATION, "classification")

    platform = _mapping(raw["platform"], "platform")
    _closed(
        platform,
        {"execution_layer", "host_id", "accelerator_label", "gpu_index", "gpu_device_id_sha256"},
        "platform",
    )
    _exact(platform["execution_layer"], "windows-native", "platform.execution_layer")
    _exact(platform["accelerator_label"], "NVIDIA RTX A4500", "platform.accelerator_label")
    gpu_index = _exact(
        _integer(platform["gpu_index"], "platform.gpu_index"), 0, "platform.gpu_index"
    )
    gpu_hash = _exact(
        _sha(platform["gpu_device_id_sha256"], "platform.gpu_device_id_sha256"),
        GPU_DEVICE_ID_SHA256,
        "platform.gpu_device_id_sha256",
    )

    qualification = _mapping(raw["qualification"], "qualification")
    _closed(qualification, {"preflight_report"}, "qualification")

    quality_source = _source(
        raw["quality_source"],
        "quality_source",
        expected_root=QUALITY_SOURCE_ROOT,
        expected_status="complete_confirmatory_passed",
        expected_git=QUALITY_SOURCE_GIT_SHA,
        expected_snapshot=QUALITY_SOURCE_SNAPSHOT_SHA256,
        expected_artifacts=QUALITY_SOURCE_ARTIFACTS,
    )

    checkpoint_raw = _mapping(raw["checkpoint_source"], "checkpoint_source")
    _closed(
        checkpoint_raw,
        {
            "root",
            "expected_status",
            "source_git_sha",
            "source_snapshot_sha256",
            "artifacts",
            "checkpoint_epoch",
            "checkpoints",
        },
        "checkpoint_source",
    )
    checkpoint_source = _source(
        {
            key: checkpoint_raw[key]
            for key in (
                "root",
                "expected_status",
                "source_git_sha",
                "source_snapshot_sha256",
                "artifacts",
            )
        },
        "checkpoint_source",
        expected_root=CHECKPOINT_SOURCE_ROOT,
        expected_status="complete_replication_passed",
        expected_git=CHECKPOINT_SOURCE_GIT_SHA,
        expected_snapshot=CHECKPOINT_SOURCE_SNAPSHOT_SHA256,
        expected_artifacts=CHECKPOINT_SOURCE_ARTIFACTS,
    )
    _exact(
        _integer(checkpoint_raw["checkpoint_epoch"], "checkpoint_source.checkpoint_epoch"),
        40,
        "checkpoint_source.checkpoint_epoch",
    )
    checkpoints_raw = checkpoint_raw["checkpoints"]
    if not isinstance(checkpoints_raw, list):
        raise DeploymentEnergyRecipeValidationError("checkpoint_source.checkpoints must be a list")
    checkpoints: list[Checkpoint] = []
    for item in checkpoints_raw:
        record = _mapping(item, "checkpoint_source.checkpoints[]")
        _closed(record, {"training_seed", "path", "sha256"}, "checkpoint_source.checkpoints[]")
        checkpoints.append(
            Checkpoint(
                _integer(record["training_seed"], "checkpoint training_seed"),
                _relative(record["path"], "checkpoint path", ".pt"),
                _sha(record["sha256"], "checkpoint sha256"),
            )
        )
    _exact(
        tuple((item.training_seed, item.path, item.sha256) for item in checkpoints),
        CHECKPOINTS,
        "checkpoint_source.checkpoints",
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
            "artifact",
            "forbidden_content_sha256",
        },
        "dataset",
    )
    forbidden_raw = dataset_raw["forbidden_content_sha256"]
    if not isinstance(forbidden_raw, list):
        raise DeploymentEnergyRecipeValidationError(
            "dataset.forbidden_content_sha256 must be a list"
        )
    forbidden = tuple(_sha(value, "dataset.forbidden_content_sha256[]") for value in forbidden_raw)
    expected_forbidden = (
        "4c0558caf0d7d7cda263959d12237396a201557c7cf1eb2bb22666d17077ae88",
        "e05572b79f212dc8bfd331fe451c8d7f9e43ca111ecc33fe4c4fe8c34dffa7b2",
        QUALITY_SOURCE_CORPUS_CONTENT_SHA256,
    )
    _exact(forbidden, expected_forbidden, "dataset.forbidden_content_sha256")
    dataset = Dataset(
        dataset_id=_exact(dataset_raw["id"], "cvrp50-deployment-energy-seed2724", "dataset.id"),
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
        seed=_exact(_integer(dataset_raw["seed"], "dataset.seed"), 2724, "dataset.seed"),
        artifact=_exact(
            _relative(dataset_raw["artifact"], "dataset.artifact", ".npz"),
            "shared/cvrp50-deployment-energy-seed2724.npz",
            "dataset.artifact",
        ),
        forbidden_content_sha256=forbidden,
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
            "batch_size",
            "inference_precision",
            "evaluation_seeds",
        },
        "neural_policy",
    )
    eval_seeds_raw = neural_raw["evaluation_seeds"]
    if not isinstance(eval_seeds_raw, list):
        raise DeploymentEnergyRecipeValidationError("neural_policy.evaluation_seeds must be a list")
    eval_seeds = tuple(
        _integer(value, "neural_policy.evaluation_seeds[]") for value in eval_seeds_raw
    )
    _exact(
        eval_seeds, (102_000, 103_000, 104_000, 105_000, 106_000), "neural_policy.evaluation_seeds"
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
        batch_size=_exact(
            _integer(neural_raw["batch_size"], "neural_policy.batch_size", 1),
            4,
            "neural_policy.batch_size",
        ),
        inference_precision=_exact(
            neural_raw["inference_precision"], "fp32", "neural_policy.inference_precision"
        ),
        evaluation_seeds=eval_seeds,
    )

    hgs_raw = _mapping(raw["hgs_policy"], "hgs_policy")
    _closed(
        hgs_raw,
        {"solver", "max_iterations", "scaling_factor", "collect_stats", "cpu_threads", "seeds"},
        "hgs_policy",
    )
    hgs_seeds_raw = hgs_raw["seeds"]
    if not isinstance(hgs_seeds_raw, list):
        raise DeploymentEnergyRecipeValidationError("hgs_policy.seeds must be a list")
    hgs_seeds = tuple(_integer(value, "hgs_policy.seeds[]") for value in hgs_seeds_raw)
    _exact(hgs_seeds, HGS_SEEDS, "hgs_policy.seeds")
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
        seeds=hgs_seeds,
    )

    schedule = _mapping(raw["schedule"], "schedule")
    _closed(schedule, {"paired_rounds", "fixed_orders"}, "schedule")
    _exact(
        _integer(schedule["paired_rounds"], "schedule.paired_rounds", 1),
        5,
        "schedule.paired_rounds",
    )
    orders_raw = schedule["fixed_orders"]
    if not isinstance(orders_raw, list) or any(not isinstance(item, list) for item in orders_raw):
        raise DeploymentEnergyRecipeValidationError("schedule.fixed_orders must be a list of lists")
    orders = tuple(tuple(str(value) for value in item) for item in orders_raw)
    _exact(orders, ROUND_ORDERS, "schedule.fixed_orders")
    orders = ROUND_ORDERS

    measurement = _mapping(raw["measurement"], "measurement")
    _closed(
        measurement,
        {
            "backend",
            "fallback",
            "domains",
            "minimum_block_duration_s",
            "duration_policy",
            "pue",
            "report_embodied",
            "whole_system",
            "exclusive_attestation",
            "gpu_process_lists",
        },
        "measurement",
    )
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
    _closed(
        limits,
        {
            "maximum_block_walltime_s",
            "campaign_attestation_max_age_s",
            "maximum_campaign_walltime_s",
        },
        "limits",
    )
    return DeploymentEnergyRecipe(
        name=_exact(raw["name"], "aet-deployment-energy-pilot-cvrp50-windows", "recipe.name"),
        host_id=_exact(platform["host_id"], "win-a4500-01", "platform.host_id"),
        gpu_index=gpu_index,
        gpu_device_id_sha256=gpu_hash,
        output_root=_exact(
            _relative(raw["output_root"], "output_root"), OUTPUT_ROOT, "output_root"
        ),
        preflight_report=_exact(
            _relative(qualification["preflight_report"], "qualification.preflight_report", ".json"),
            "experiments/aet-journal/qualification/windows-deployment-energy-preflight.json",
            "qualification.preflight_report",
        ),
        quality_source=quality_source,
        checkpoint_source=checkpoint_source,
        checkpoints=tuple(checkpoints),
        dataset=dataset,
        neural_policy=neural,
        hgs_policy=hgs,
        round_orders=orders,
        minimum_block_duration_s=float(measurement["minimum_block_duration_s"]),
        maximum_block_walltime_s=_exact(
            _number(limits["maximum_block_walltime_s"], "limits.maximum_block_walltime_s"),
            900.0,
            "limits.maximum_block_walltime_s",
        ),
        campaign_attestation_max_age_s=_exact(
            _number(
                limits["campaign_attestation_max_age_s"], "limits.campaign_attestation_max_age_s"
            ),
            7200.0,
            "limits.campaign_attestation_max_age_s",
        ),
        maximum_campaign_walltime_s=_exact(
            _number(limits["maximum_campaign_walltime_s"], "limits.maximum_campaign_walltime_s"),
            7200.0,
            "limits.maximum_campaign_walltime_s",
        ),
    )


def validate_recipe(raw: dict[str, Any]) -> DeploymentEnergyRecipe:
    """Validate an in-memory recipe through the same closed YAML contract."""

    serialized = yaml.safe_dump(raw, sort_keys=False)
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "recipe.yaml"
        path.write_text(serialized, encoding="utf-8")
        return load_recipe(path)


def expected_blocks(recipe: DeploymentEnergyRecipe) -> tuple[EnergyBlock, ...]:
    """Return the fixed ten-block order for atomic execution and resume."""

    blocks: list[EnergyBlock] = []
    for round_index, order in enumerate(recipe.round_orders):
        training_seed = recipe.checkpoints[round_index].training_seed
        evaluation_seed = recipe.neural_policy.evaluation_seeds[round_index]
        hgs_seed = recipe.hgs_policy.seeds[round_index]
        for order_index, policy in enumerate(order):
            relative = f"blocks/round-{round_index:02d}-{order_index:02d}-{policy}.json"
            blocks.append(
                EnergyBlock(
                    round_index,
                    order_index,
                    policy,
                    training_seed,
                    evaluation_seed,
                    hgs_seed,
                    relative,
                )
            )
    return tuple(blocks)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_report(source: SourceBundle, root: Path) -> dict[str, Any]:
    source_root = root.joinpath(*PurePosixPath(source.root).parts)
    artifacts: list[dict[str, Any]] = []
    valid = source_root.is_dir() and not source_root.is_symlink()
    for artifact in source.artifacts:
        path = source_root.joinpath(*PurePosixPath(artifact.path).parts)
        matches = path.is_file() and not path.is_symlink() and _sha256_file(path) == artifact.sha256
        artifacts.append({"path": artifact.path, "sha256_matches": matches})
        valid = valid and matches
    return {"root": source.root, "valid": valid, "artifacts": artifacts}


def qualify_for_execution(recipe: DeploymentEnergyRecipe, root: Path) -> dict[str, Any]:
    """Perform read-only source and native-Windows qualification."""

    quality_source = _source_report(recipe.quality_source, root)
    checkpoint_source = _source_report(recipe.checkpoint_source, root)
    checkpoint_root = root.joinpath(*PurePosixPath(recipe.checkpoint_source.root).parts)
    checkpoints_match = all(
        (
            checkpoint_root.joinpath(*PurePosixPath(item.path).parts).is_file()
            and _sha256_file(checkpoint_root.joinpath(*PurePosixPath(item.path).parts))
            == item.sha256
        )
        for item in recipe.checkpoints
    )
    probe = SimpleNamespace(
        preflight_report=recipe.preflight_report,
        host_id=recipe.host_id,
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    native = software._qualification_report(probe)  # type: ignore[arg-type]
    ready = bool(
        native["ready_to_execute"]
        and quality_source["valid"]
        and checkpoint_source["valid"]
        and checkpoints_match
    )
    return {
        "native_windows_counters_ready": native["ready_to_execute"],
        "native": native,
        "quality_source": quality_source,
        "checkpoint_source": checkpoint_source,
        "all_five_checkpoints_match": checkpoints_match,
        "ready_to_execute": ready,
    }


def dry_run(path: Path, workspace_root: Path | None = None) -> dict[str, Any]:
    """Validate without creating output or generating/opening seed 2724."""

    recipe = load_recipe(path)
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    qualification = qualify_for_execution(recipe, root)
    blocks = expected_blocks(recipe)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "valid_ready_software_exploratory"
        if qualification["ready_to_execute"]
        else "valid_not_qualified",
        "classification": CLASSIFICATION,
        "name": recipe.name,
        "output_root": recipe.output_root,
        "fresh_dataset_seed": recipe.dataset.seed,
        "fresh_dataset_opened": False,
        "block_count": len(blocks),
        "paired_rounds": 5,
        "minimum_block_duration_s": recipe.minimum_block_duration_s,
        "maximum_campaign_walltime_s": recipe.maximum_campaign_walltime_s,
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
