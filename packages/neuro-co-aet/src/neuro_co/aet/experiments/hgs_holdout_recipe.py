"""Closed recipe contracts for the CVRP50 HGS quality holdouts.

The production recipe opens exactly one previously reserved corpus: 512 CVRP50
instances generated from seed 2722.  It evaluates the already frozen neural
policy together with HGS budget 3 as the primary comparator and budget 10 as a
non-rescuing sensitivity.  This module is intentionally quality-only.  Energy,
carbon, preflight reports, and exclusive-host attestations are outside its
schema.  A second closed profile confirms the already fixed HGS-10 and neural
policies on seed 2723 without re-running a frontier or HGS-3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCHEMA_VERSION = "aet-journal-hgs-holdout/v1"
KIND = "aet-journal-hgs-holdout"
CONFIRMATORY_SCHEMA_VERSION = "aet-journal-hgs10-confirmatory/v1"
CONFIRMATORY_KIND = "aet-journal-hgs10-confirmatory"
RUN_STATE_SCHEMA = "aet-hgs-holdout-run-state/v1"

PRODUCTION_OUTPUT_ROOT = "experiments/aet-journal/raw/hgs-holdout/cvrp50-holdout-seed2722"
PRODUCTION_SELECTION_ROOT = "experiments/aet-journal/raw/hgs-frontier/cvrp50-selection-seed2721"
PRODUCTION_REPLICATION_ROOT = (
    "experiments/aet-journal/raw/quality-replication/cvrp50-epoch40-seeds2-6"
)
CONFIRMATORY_OUTPUT_ROOT = "experiments/aet-journal/raw/hgs10-confirmatory/cvrp50-seed2723"
CONFIRMATORY_POLICY_ROOT = "experiments/aet-journal/raw/hgs-holdout/cvrp50-holdout-seed2722"

SELECTION_GIT_SHA = "d0e06e73d29cc57e726ea0295d63c8dfa6e0792c"
SELECTION_SOURCE_SNAPSHOT_SHA256 = (
    "acea1bdb6dfa8133a9723a7789b49b69e10ffde2a273bb783cea2284875fd961"
)
SELECTION_ARTIFACTS: dict[str, tuple[str, str]] = {
    "manifest": (
        "manifest.json",
        "cf026bcf4bc8c88277c72a3574c2b3e5deb2eba3cc800c7b5ad06aa7da2e1a18",
    ),
    "checksums": (
        "SHA256SUMS",
        "65149ec433b77c31b38aac1d5383c9b96c787ca97248ef77f1214302e15964a3",
    ),
    "run_state": (
        "run-state.json",
        "00a6e4f5f56f816b75022ac639b0cb2cbd5bf3ecb42857c30d935868e4e05a02",
    ),
    "assessment": (
        "selection-assessment.json",
        "e62e37aa82131b8afb50bd9bdc99af0fd3dd775730aec4ccf48994b45515cd27",
    ),
    "recipe": (
        "recipe.yaml",
        "fa7729d41687f4fc54fe5f4d24f9162f08a50f2291b5461d6071f59655200da1",
    ),
    "reference_lock": (
        "reference/reference-lock.json",
        "0d6d8a5e6d08a619b4f4ea9d2db1adee09f6cf15a6c245aa2b2efdce22765f06",
    ),
    "neural_quality_gate": (
        "neural-quality-gate.json",
        "7c5f38985aba94dbc7aec76907837739118f2960b33657fd26f1278e8575fd4d",
    ),
    "replication_receipt": (
        "provenance/quality-replication-receipt.json",
        "bbed56e758b63a90b79814ad646d245a0f677cfff8116b905afeafe75001a20b",
    ),
    "selection_corpus_manifest": (
        "shared/cvrp50-hgs-frontier-selection-seed2721.manifest.json",
        "f5b93f4eaebee8827dae7e50242500c14112e8797a861e017183588af1b2999f",
    ),
}
SELECTION_CORPUS_CONTENT_SHA256 = "4c0558caf0d7d7cda263959d12237396a201557c7cf1eb2bb22666d17077ae88"

CONFIRMATORY_POLICY_GIT_SHA = "d50cfad03c7a443e2680279ecaab49577560917f"
CONFIRMATORY_POLICY_SOURCE_SNAPSHOT_SHA256 = (
    "25d7757c0a2500fc5666ce5d2ca1ca8d3398f776efba5be38678b641de7f96e7"
)
CONFIRMATORY_POLICY_ARTIFACTS: dict[str, tuple[str, str]] = {
    "manifest": (
        "manifest.json",
        "acefcc51f537ee3ecd169c2344b1b8a53051d78ada6aa0396b77d4e426969a6b",
    ),
    "checksums": (
        "SHA256SUMS",
        "d908951eab8dbaf5102da05272eda4c229d122ff08cb8d3206856d738a3e388a",
    ),
    "run_state": (
        "run-state.json",
        "e278994f966c212e911b2f0db09d53bac04f37b5afd665dd5ce0f3b08b39b382",
    ),
    "assessment": (
        "holdout-assessment.json",
        "05a36796a9a8a445f0d1c5814d279b4911d9f509a8c5a3e6eb4d3292e37eddfb",
    ),
    "recipe": (
        "recipe.yaml",
        "29c87eccb7d09327c446f3a66b5c04c5a13acd7d49a9c5543326d2b486743b6d",
    ),
    "reference_lock": (
        "reference/reference-lock.json",
        "f842e67a0a2aa4876795790f5a7f9d26c6b4b5fd1062c2d051795cd74f6e3819",
    ),
    "replication_receipt": (
        "provenance/quality-replication-receipt.json",
        "bbed56e758b63a90b79814ad646d245a0f677cfff8116b905afeafe75001a20b",
    ),
    "prior_corpus_manifest": (
        "shared/cvrp50-hgs-holdout-seed2722.manifest.json",
        "0d1097c1acc3f29e00f94618fc81b85531327cc9ddf2153af7e6e76e0f7f6862",
    ),
}
CONFIRMATORY_PRIOR_CORPUS_CONTENT_SHA256 = (
    "e05572b79f212dc8bfd331fe451c8d7f9e43ca111ecc33fe4c4fe8c34dffa7b2"
)

REPLICATION_GIT_SHA = "bf8d5c5dfb1a566019cd56135efa91f2e460ee6b"
REPLICATION_SOURCE_SNAPSHOT_SHA256 = (
    "3dcc187f5e875f6427e92c4c1a707461a101c8d37f1abaca029d2c03a9cd8e19"
)
REPLICATION_ARTIFACTS: dict[str, tuple[str, str]] = {
    "manifest": (
        "manifest.json",
        "df47d74fe890f35103e1aa761b925ca7c082f741af8f732c1b689129e971f49f",
    ),
    "checksums": (
        "SHA256SUMS",
        "e59d5fb1449b75f0439a06ce587e7129ba0f6e068f0a920dc7a3021e45f0202f",
    ),
    "assessment": (
        "replication-assessment.json",
        "7cc4b1b2e257c6bc5380684776f39347105c2fd647d5445d7c33a12e3774f320",
    ),
    "recipe": (
        "recipe.yaml",
        "e0617bd5fd43057763128d62005b9f12ef1e26308191a6bc077dd803d6b4160c",
    ),
    "run_state": (
        "run-state.json",
        "afa50d5b27f7655e49e9fa3f35604a51ef83098969f777c11c4233db0d12a4bb",
    ),
    "reference_lock": (
        "reference/reference-lock.json",
        "90a418104cc98b55405ea19413c0ac85f990f3bb7117d710afd77e9ea2a9e57d",
    ),
}
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

HOLDOUT_SEED = 2722
CONFIRMATORY_HOLDOUT_SEED = 2723
HOLDOUT_INSTANCES = 512
PRIMARY_BUDGET = 3
SENSITIVITY_BUDGET = 10
TRAINING_SEEDS = (2, 3, 4, 5, 6)
NEURAL_EVALUATION_SEEDS = (92_000, 93_000, 94_000, 95_000, 96_000)
HGS_SEEDS = tuple(range(40_101, 50_000, 1_000))
REFERENCE_HGS_SEEDS = (31_101, 32_101, 33_101)
REFERENCE_ORTOOLS_SEED = 34_101
BUDGET_ORDERS = tuple(
    (PRIMARY_BUDGET, SENSITIVITY_BUDGET) if index % 2 == 0 else (SENSITIVITY_BUDGET, PRIMARY_BUDGET)
    for index in range(10)
)
CONFIRMATORY_BUDGET = 10
CONFIRMATORY_BUDGET_ORDERS = tuple((CONFIRMATORY_BUDGET,) for _ in HGS_SEEDS)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_HEX_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class HGSHoldoutRecipeValidationError(ValueError):
    """Raised when the holdout recipe departs from its frozen contract."""


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
class SelectionSource:
    root: str
    expected_status: str
    source_git_sha: str
    source_snapshot_sha256: str
    selected_budget: int
    sensitivity_budget: int
    selection_corpus_content_sha256: str
    manifest: ArtifactEvidence
    checksums: ArtifactEvidence
    run_state: ArtifactEvidence
    assessment: ArtifactEvidence
    recipe: ArtifactEvidence
    reference_lock: ArtifactEvidence
    neural_quality_gate: ArtifactEvidence
    replication_receipt: ArtifactEvidence
    selection_corpus_manifest: ArtifactEvidence


@dataclass(frozen=True, slots=True)
class ConfirmatoryPolicySource:
    root: str
    expected_status: str
    source_git_sha: str
    source_snapshot_sha256: str
    confirmed_budget: int
    prior_corpus_content_sha256: str
    manifest: ArtifactEvidence
    checksums: ArtifactEvidence
    run_state: ArtifactEvidence
    assessment: ArtifactEvidence
    recipe: ArtifactEvidence
    reference_lock: ArtifactEvidence
    replication_receipt: ArtifactEvidence
    prior_corpus_manifest: ArtifactEvidence


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
class SourceSnapshotConfig:
    schema_version: str
    canonicalize_text_line_endings: bool
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
class HoldoutDataset:
    problem: str
    size: int
    capacity: float
    max_demand: int
    holdout: DatasetSplit
    forbidden_selection_content_sha256: str
    additional_forbidden_content_sha256: tuple[str, ...] = ()


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
class HoldoutReference:
    policy: str
    locked_before_policy_evaluation: bool
    hgs: HGSReference
    ortools: ORToolsReference


@dataclass(frozen=True, slots=True)
class NeuralPolicy:
    mode_id: str
    checkpoint_epoch: int
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class HGSPolicies:
    solver: str
    budget_kind: str
    primary_budget: int
    sensitivity_budget: int | None
    seeds: tuple[int, ...]
    budget_orders: tuple[tuple[int, ...], ...]
    scaling_factor: int
    collect_stats: bool


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    method: str
    replicates: int
    generator: str
    seed: int
    confidence_level: float
    report_mean: bool
    report_q95: bool
    neural_t_critical_value: float
    neural_t_degrees_of_freedom: int
    hgs_t_critical_value: float
    hgs_t_degrees_of_freedom: int


@dataclass(frozen=True, slots=True)
class QualityGate:
    metric: str
    maximum_mean_gap_pct: float
    comparison: str
    maximum_invalid_instances: int
    require_finite: bool
    require_each_seed_below_threshold: bool
    require_each_seed_empirical_q95_below_threshold: bool
    require_pooled_empirical_q95_below_threshold: bool
    require_t_ucb_below_threshold: bool
    require_bootstrap_mean_ucb_below_threshold: bool
    require_bootstrap_q95_ucb_below_threshold: bool
    primary_joint_rule: str
    sensitivity_can_rescue_primary: bool


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    max_newly_completed_rounds_per_invocation: int
    energy_measurement: str
    carbon_accounting: str
    preflight_required: bool
    exclusive_attestation_required: bool
    timing_scientific_use: bool


@dataclass(frozen=True, slots=True)
class AETHGSHoldoutRecipe:
    name: str
    output_root: str
    host_id: str
    execution_layer: str
    gpu_index: int
    expected_accelerator_label: str
    selection_source: SelectionSource | ConfirmatoryPolicySource
    replication_source: ReplicationSource
    source_snapshot: SourceSnapshotConfig
    dataset: HoldoutDataset
    reference: HoldoutReference
    neural_policy: NeuralPolicy
    hgs_policies: HGSPolicies
    bootstrap: BootstrapConfig
    gate: QualityGate
    execution: ExecutionConfig

    @property
    def neural_prerequisite(self) -> NeuralPolicy:
        """Compatibility view for the sealed replication-source validator."""

        return self.neural_policy

    @property
    def policy_source(self) -> ConfirmatoryPolicySource | None:
        """Return the sealed seed-2722 policy source for the confirmatory profile."""

        if isinstance(self.selection_source, ConfirmatoryPolicySource):
            return self.selection_source
        return None

    @property
    def confirmatory_profile(self) -> bool:
        return isinstance(self.selection_source, ConfirmatoryPolicySource)


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise HGSHoldoutRecipeValidationError(f"{where} must be a mapping")
    return value


def _closed(value: dict[str, Any], keys: set[str], where: str) -> None:
    unknown = sorted(repr(key) for key in set(value) - keys)
    missing = sorted(keys - set(value))
    if unknown:
        raise HGSHoldoutRecipeValidationError(f"{where} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise HGSHoldoutRecipeValidationError(
            f"{where} is missing required keys: {', '.join(missing)}"
        )


def _exact(value: Any, expected: Any, where: str) -> Any:
    if type(value) is not type(expected) or value != expected:
        raise HGSHoldoutRecipeValidationError(f"{where} must be exactly {expected!r}")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise HGSHoldoutRecipeValidationError(f"{where} must be an integer >= {minimum}")
    return value


def _number(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HGSHoldoutRecipeValidationError(f"{where} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        qualifier = "positive" if positive else "non-negative"
        raise HGSHoldoutRecipeValidationError(f"{where} must be {qualifier} and finite")
    return result


def _sha256(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise HGSHoldoutRecipeValidationError(f"{where} must be a lowercase SHA-256")
    return value


def _git_sha(value: Any, where: str) -> str:
    if not isinstance(value, str) or _HEX_GIT_SHA.fullmatch(value) is None:
        raise HGSHoldoutRecipeValidationError(f"{where} must be a lowercase Git SHA-1")
    return value


def _relative_path(value: Any, where: str, *, suffix: str | None = None) -> str:
    if not isinstance(value, str):
        raise HGSHoldoutRecipeValidationError(f"{where} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise HGSHoldoutRecipeValidationError(f"{where} must be a safe relative POSIX path")
    for part in path.parts:
        if not _SAFE_COMPONENT.fullmatch(part):
            raise HGSHoldoutRecipeValidationError(f"{where} contains an unsafe component")
        if part.split(".", 1)[0].lower() in _WINDOWS_RESERVED_NAMES:
            raise HGSHoldoutRecipeValidationError(f"{where} uses a Windows-reserved name")
    if suffix is not None and path.suffix.lower() != suffix:
        raise HGSHoldoutRecipeValidationError(f"{where} must end with {suffix}")
    return path.as_posix()


def _slug(value: Any, where: str) -> str:
    if not isinstance(value, str) or _SAFE_COMPONENT.fullmatch(value) is None:
        raise HGSHoldoutRecipeValidationError(f"{where} must be a safe slug")
    return value


def _tuple_of_ints(value: Any, where: str, *, minimum: int = 0) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise HGSHoldoutRecipeValidationError(f"{where} must be a list")
    return tuple(
        _integer(item, f"{where}[{index}]", minimum=minimum) for index, item in enumerate(value)
    )


def _artifact(value: Any, where: str, expected: tuple[str, str]) -> ArtifactEvidence:
    raw = _mapping(value, where)
    _closed(raw, {"path", "sha256"}, where)
    path = _relative_path(raw["path"], f"{where}.path")
    digest = _sha256(raw["sha256"], f"{where}.sha256")
    _exact(path, expected[0], f"{where}.path")
    _exact(digest, expected[1], f"{where}.sha256")
    return ArtifactEvidence(path=path, sha256=digest)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HGSHoldoutRecipeValidationError(f"cannot load holdout recipe: {exc}") from exc
    return _mapping(loaded, "recipe")


def _parse_selection_source(value: Any) -> SelectionSource:
    raw = _mapping(value, "selection_source")
    keys = {
        "root",
        "expected_status",
        "source_git_sha",
        "source_snapshot_sha256",
        "selected_budget",
        "sensitivity_budget",
        "selection_corpus_content_sha256",
        *SELECTION_ARTIFACTS,
    }
    _closed(raw, keys, "selection_source")
    root = _relative_path(raw["root"], "selection_source.root")
    _exact(root, PRODUCTION_SELECTION_ROOT, "selection_source.root")
    _exact(raw["expected_status"], "complete_selection_passed", "selection_source.expected_status")
    git_sha = _git_sha(raw["source_git_sha"], "selection_source.source_git_sha")
    snapshot = _sha256(raw["source_snapshot_sha256"], "selection_source.source_snapshot_sha256")
    _exact(git_sha, SELECTION_GIT_SHA, "selection_source.source_git_sha")
    _exact(
        snapshot,
        SELECTION_SOURCE_SNAPSHOT_SHA256,
        "selection_source.source_snapshot_sha256",
    )
    selected = _integer(raw["selected_budget"], "selection_source.selected_budget", minimum=1)
    sensitivity = _integer(
        raw["sensitivity_budget"], "selection_source.sensitivity_budget", minimum=1
    )
    _exact(selected, PRIMARY_BUDGET, "selection_source.selected_budget")
    _exact(sensitivity, SENSITIVITY_BUDGET, "selection_source.sensitivity_budget")
    content = _sha256(
        raw["selection_corpus_content_sha256"],
        "selection_source.selection_corpus_content_sha256",
    )
    _exact(
        content,
        SELECTION_CORPUS_CONTENT_SHA256,
        "selection_source.selection_corpus_content_sha256",
    )
    artifacts = {
        name: _artifact(raw[name], f"selection_source.{name}", expected)
        for name, expected in SELECTION_ARTIFACTS.items()
    }
    return SelectionSource(
        root=root,
        expected_status="complete_selection_passed",
        source_git_sha=git_sha,
        source_snapshot_sha256=snapshot,
        selected_budget=selected,
        sensitivity_budget=sensitivity,
        selection_corpus_content_sha256=content,
        **artifacts,
    )


def _parse_confirmatory_policy_source(value: Any) -> ConfirmatoryPolicySource:
    raw = _mapping(value, "policy_source")
    keys = {
        "root",
        "expected_status",
        "source_git_sha",
        "source_snapshot_sha256",
        "confirmed_budget",
        "prior_corpus_content_sha256",
        *CONFIRMATORY_POLICY_ARTIFACTS,
    }
    _closed(raw, keys, "policy_source")
    root = _relative_path(raw["root"], "policy_source.root")
    _exact(root, CONFIRMATORY_POLICY_ROOT, "policy_source.root")
    _exact(
        raw["expected_status"],
        "complete_primary_nonpass_sensitivity_passed",
        "policy_source.expected_status",
    )
    git_sha = _git_sha(raw["source_git_sha"], "policy_source.source_git_sha")
    snapshot = _sha256(raw["source_snapshot_sha256"], "policy_source.source_snapshot_sha256")
    _exact(git_sha, CONFIRMATORY_POLICY_GIT_SHA, "policy_source.source_git_sha")
    _exact(
        snapshot,
        CONFIRMATORY_POLICY_SOURCE_SNAPSHOT_SHA256,
        "policy_source.source_snapshot_sha256",
    )
    budget = _integer(raw["confirmed_budget"], "policy_source.confirmed_budget", minimum=1)
    _exact(budget, CONFIRMATORY_BUDGET, "policy_source.confirmed_budget")
    content = _sha256(
        raw["prior_corpus_content_sha256"],
        "policy_source.prior_corpus_content_sha256",
    )
    _exact(
        content,
        CONFIRMATORY_PRIOR_CORPUS_CONTENT_SHA256,
        "policy_source.prior_corpus_content_sha256",
    )
    artifacts = {
        name: _artifact(raw[name], f"policy_source.{name}", expected)
        for name, expected in CONFIRMATORY_POLICY_ARTIFACTS.items()
    }
    return ConfirmatoryPolicySource(
        root=root,
        expected_status="complete_primary_nonpass_sensitivity_passed",
        source_git_sha=git_sha,
        source_snapshot_sha256=snapshot,
        confirmed_budget=budget,
        prior_corpus_content_sha256=content,
        **artifacts,
    )


def _parse_replication_source(value: Any) -> ReplicationSource:
    raw = _mapping(value, "replication_source")
    _closed(
        raw,
        {
            "root",
            "expected_status",
            "source_git_sha",
            "source_snapshot_sha256",
            *REPLICATION_ARTIFACTS,
            "checkpoint_epoch",
            "mode_id",
            "checkpoints",
        },
        "replication_source",
    )
    root = _relative_path(raw["root"], "replication_source.root")
    _exact(root, PRODUCTION_REPLICATION_ROOT, "replication_source.root")
    _exact(
        raw["expected_status"],
        "complete_replication_passed",
        "replication_source.expected_status",
    )
    git_sha = _git_sha(raw["source_git_sha"], "replication_source.source_git_sha")
    snapshot = _sha256(raw["source_snapshot_sha256"], "replication_source.source_snapshot_sha256")
    _exact(git_sha, REPLICATION_GIT_SHA, "replication_source.source_git_sha")
    _exact(
        snapshot,
        REPLICATION_SOURCE_SNAPSHOT_SHA256,
        "replication_source.source_snapshot_sha256",
    )
    artifacts = {
        name: _artifact(raw[name], f"replication_source.{name}", expected)
        for name, expected in REPLICATION_ARTIFACTS.items()
    }
    epoch = _integer(raw["checkpoint_epoch"], "replication_source.checkpoint_epoch", minimum=1)
    _exact(epoch, 40, "replication_source.checkpoint_epoch")
    mode_id = _slug(raw["mode_id"], "replication_source.mode_id")
    _exact(mode_id, "pomo-50x8", "replication_source.mode_id")
    checkpoint_values = raw["checkpoints"]
    if not isinstance(checkpoint_values, list) or len(checkpoint_values) != len(CHECKPOINTS):
        raise HGSHoldoutRecipeValidationError(
            "replication_source.checkpoints must contain the five frozen checkpoints"
        )
    checkpoints: list[CheckpointEvidence] = []
    for index, (value_item, expected) in enumerate(
        zip(checkpoint_values, CHECKPOINTS, strict=True)
    ):
        item = _mapping(value_item, f"replication_source.checkpoints[{index}]")
        _closed(
            item, {"training_seed", "path", "sha256"}, f"replication_source.checkpoints[{index}]"
        )
        seed = _integer(
            item["training_seed"], f"replication_source.checkpoints[{index}].training_seed"
        )
        path = _relative_path(
            item["path"], f"replication_source.checkpoints[{index}].path", suffix=".pt"
        )
        digest = _sha256(item["sha256"], f"replication_source.checkpoints[{index}].sha256")
        _exact((seed, path, digest), expected, f"replication_source.checkpoints[{index}]")
        checkpoints.append(CheckpointEvidence(seed, path, digest))
    return ReplicationSource(
        root=root,
        expected_status="complete_replication_passed",
        source_git_sha=git_sha,
        source_snapshot_sha256=snapshot,
        checkpoint_epoch=epoch,
        mode_id=mode_id,
        checkpoints=tuple(checkpoints),
        **artifacts,
    )


def _parse_source_snapshot(value: Any) -> SourceSnapshotConfig:
    raw = _mapping(value, "source_snapshot")
    _closed(
        raw,
        {
            "schema_version",
            "canonicalize_text_line_endings",
            "freeze_at_initialization",
            "require_match_on_resume",
            "papers_and_experiment_outputs_excluded",
            "include_patterns",
        },
        "source_snapshot",
    )
    _exact(
        raw["schema_version"],
        "aet-hgs-holdout-source-snapshot/v1",
        "source_snapshot.schema_version",
    )
    for key in (
        "canonicalize_text_line_endings",
        "freeze_at_initialization",
        "require_match_on_resume",
        "papers_and_experiment_outputs_excluded",
    ):
        _exact(raw[key], True, f"source_snapshot.{key}")
    patterns = raw["include_patterns"]
    if (
        not isinstance(patterns, list)
        or not patterns
        or not all(isinstance(item, str) for item in patterns)
    ):
        raise HGSHoldoutRecipeValidationError(
            "source_snapshot.include_patterns must be a non-empty string list"
        )
    required = {
        "pyproject.toml",
        "uv.lock",
        "packages/neuro-co-aet/pyproject.toml",
        "packages/neuro-co-aet/src/**/*.py",
        "packages/neuro-co-core/src/**/*.py",
        "packages/neuro-co-problems/src/**/*.py",
        "recipes/aet_journal/hgs_holdout_cvrp50_windows.yaml",
        "scripts/aet_journal/run_hgs_holdout_windows.ps1",
    }
    if len(set(patterns)) != len(patterns) or not required.issubset(patterns):
        raise HGSHoldoutRecipeValidationError(
            "source_snapshot.include_patterns must uniquely cover the holdout implementation"
        )
    for index, pattern in enumerate(patterns):
        path = PurePosixPath(pattern)
        if (
            path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise HGSHoldoutRecipeValidationError(
                f"source_snapshot.include_patterns[{index}] is unsafe"
            )
        if pattern.startswith("papers/") or pattern.startswith("experiments/"):
            raise HGSHoldoutRecipeValidationError(
                "paper and experiment outputs cannot enter the source snapshot"
            )
    return SourceSnapshotConfig(
        schema_version="aet-hgs-holdout-source-snapshot/v1",
        canonicalize_text_line_endings=True,
        freeze_at_initialization=True,
        require_match_on_resume=True,
        papers_and_experiment_outputs_excluded=True,
        include_patterns=tuple(patterns),
    )


def _parse_dataset(value: Any) -> HoldoutDataset:
    raw = _mapping(value, "dataset")
    _closed(
        raw,
        {
            "problem",
            "size",
            "capacity",
            "max_demand",
            "holdout",
            "forbidden_selection_content_sha256",
        },
        "dataset",
    )
    _exact(raw["problem"], "cvrp", "dataset.problem")
    size = _integer(raw["size"], "dataset.size", minimum=1)
    capacity = _number(raw["capacity"], "dataset.capacity", positive=True)
    max_demand = _integer(raw["max_demand"], "dataset.max_demand", minimum=1)
    _exact(size, 50, "dataset.size")
    _exact(capacity, 40.0, "dataset.capacity")
    _exact(max_demand, 9, "dataset.max_demand")
    split = _mapping(raw["holdout"], "dataset.holdout")
    _closed(split, {"id", "num_instances", "seed", "artifact"}, "dataset.holdout")
    split_id = _slug(split["id"], "dataset.holdout.id")
    count = _integer(split["num_instances"], "dataset.holdout.num_instances", minimum=1)
    seed = _integer(split["seed"], "dataset.holdout.seed")
    artifact = _relative_path(split["artifact"], "dataset.holdout.artifact", suffix=".npz")
    _exact(split_id, "cvrp50-hgs-holdout-seed2722", "dataset.holdout.id")
    _exact(count, HOLDOUT_INSTANCES, "dataset.holdout.num_instances")
    _exact(seed, HOLDOUT_SEED, "dataset.holdout.seed")
    _exact(artifact, "shared/cvrp50-hgs-holdout-seed2722.npz", "dataset.holdout.artifact")
    forbidden = _sha256(
        raw["forbidden_selection_content_sha256"], "dataset.forbidden_selection_content_sha256"
    )
    _exact(forbidden, SELECTION_CORPUS_CONTENT_SHA256, "dataset.forbidden_selection_content_sha256")
    return HoldoutDataset(
        problem="cvrp",
        size=size,
        capacity=capacity,
        max_demand=max_demand,
        holdout=DatasetSplit(split_id, count, seed, artifact),
        forbidden_selection_content_sha256=forbidden,
    )


def _parse_reference(value: Any) -> HoldoutReference:
    raw = _mapping(value, "reference")
    _closed(raw, {"policy", "locked_before_policy_evaluation", "hgs", "ortools"}, "reference")
    _exact(raw["policy"], "best_validated_per_instance", "reference.policy")
    _exact(
        raw["locked_before_policy_evaluation"], True, "reference.locked_before_policy_evaluation"
    )
    hgs = _mapping(raw["hgs"], "reference.hgs")
    _closed(hgs, {"solver", "seeds", "max_iterations"}, "reference.hgs")
    _exact(hgs["solver"], "pyvrp-hgs", "reference.hgs.solver")
    hgs_seeds = _tuple_of_ints(hgs["seeds"], "reference.hgs.seeds")
    _exact(hgs_seeds, REFERENCE_HGS_SEEDS, "reference.hgs.seeds")
    iterations = _integer(hgs["max_iterations"], "reference.hgs.max_iterations", minimum=1)
    _exact(iterations, 1000, "reference.hgs.max_iterations")
    ortools = _mapping(raw["ortools"], "reference.ortools")
    _closed(ortools, {"solver", "seed", "solution_limit", "scaling_factor"}, "reference.ortools")
    _exact(ortools["solver"], "ortools-routing-gls", "reference.ortools.solver")
    ortools_seed = _integer(ortools["seed"], "reference.ortools.seed")
    solution_limit = _integer(
        ortools["solution_limit"], "reference.ortools.solution_limit", minimum=1
    )
    scaling = _integer(ortools["scaling_factor"], "reference.ortools.scaling_factor", minimum=1)
    _exact(ortools_seed, REFERENCE_ORTOOLS_SEED, "reference.ortools.seed")
    _exact(solution_limit, 200, "reference.ortools.solution_limit")
    _exact(scaling, 1_000_000, "reference.ortools.scaling_factor")
    return HoldoutReference(
        policy="best_validated_per_instance",
        locked_before_policy_evaluation=True,
        hgs=HGSReference("pyvrp-hgs", hgs_seeds, iterations),
        ortools=ORToolsReference("ortools-routing-gls", ortools_seed, solution_limit, scaling),
    )


def _parse_neural_policy(value: Any) -> NeuralPolicy:
    raw = _mapping(value, "neural_policy")
    _closed(
        raw, {"mode_id", "checkpoint_epoch", "training_seeds", "evaluation_seeds"}, "neural_policy"
    )
    mode = _slug(raw["mode_id"], "neural_policy.mode_id")
    epoch = _integer(raw["checkpoint_epoch"], "neural_policy.checkpoint_epoch", minimum=1)
    training = _tuple_of_ints(raw["training_seeds"], "neural_policy.training_seeds")
    evaluation = _tuple_of_ints(raw["evaluation_seeds"], "neural_policy.evaluation_seeds")
    _exact(mode, "pomo-50x8", "neural_policy.mode_id")
    _exact(epoch, 40, "neural_policy.checkpoint_epoch")
    _exact(training, TRAINING_SEEDS, "neural_policy.training_seeds")
    _exact(evaluation, NEURAL_EVALUATION_SEEDS, "neural_policy.evaluation_seeds")
    return NeuralPolicy(mode, epoch, training, evaluation)


def _parse_hgs_policies(value: Any) -> HGSPolicies:
    raw = _mapping(value, "hgs_policies")
    _closed(
        raw,
        {
            "solver",
            "budget_kind",
            "primary_budget",
            "sensitivity_budget",
            "seeds",
            "budget_orders",
            "scaling_factor",
            "collect_stats",
        },
        "hgs_policies",
    )
    _exact(raw["solver"], "pyvrp-hgs", "hgs_policies.solver")
    _exact(raw["budget_kind"], "max_iterations", "hgs_policies.budget_kind")
    primary = _integer(raw["primary_budget"], "hgs_policies.primary_budget", minimum=1)
    sensitivity = _integer(raw["sensitivity_budget"], "hgs_policies.sensitivity_budget", minimum=1)
    _exact(primary, PRIMARY_BUDGET, "hgs_policies.primary_budget")
    _exact(sensitivity, SENSITIVITY_BUDGET, "hgs_policies.sensitivity_budget")
    seeds = _tuple_of_ints(raw["seeds"], "hgs_policies.seeds")
    _exact(seeds, HGS_SEEDS, "hgs_policies.seeds")
    orders_raw = raw["budget_orders"]
    if not isinstance(orders_raw, list):
        raise HGSHoldoutRecipeValidationError("hgs_policies.budget_orders must be a list")
    orders = tuple(
        _tuple_of_ints(row, f"hgs_policies.budget_orders[{index}]", minimum=1)
        for index, row in enumerate(orders_raw)
    )
    _exact(orders, BUDGET_ORDERS, "hgs_policies.budget_orders")
    scaling = _integer(raw["scaling_factor"], "hgs_policies.scaling_factor", minimum=1)
    _exact(scaling, 1_000_000, "hgs_policies.scaling_factor")
    _exact(raw["collect_stats"], False, "hgs_policies.collect_stats")
    return HGSPolicies(
        solver="pyvrp-hgs",
        budget_kind="max_iterations",
        primary_budget=primary,
        sensitivity_budget=sensitivity,
        seeds=seeds,
        budget_orders=orders,
        scaling_factor=scaling,
        collect_stats=False,
    )


def _parse_bootstrap(value: Any) -> BootstrapConfig:
    raw = _mapping(value, "bootstrap")
    _closed(
        raw,
        {
            "method",
            "replicates",
            "generator",
            "seed",
            "confidence_level",
            "report_mean",
            "report_q95",
            "neural_t_critical_value",
            "neural_t_degrees_of_freedom",
            "hgs_t_critical_value",
            "hgs_t_degrees_of_freedom",
        },
        "bootstrap",
    )
    _exact(raw["method"], "crossed_seed_by_instance_percentile", "bootstrap.method")
    _exact(raw["generator"], "numpy-pcg64", "bootstrap.generator")
    replicates = _integer(raw["replicates"], "bootstrap.replicates", minimum=1)
    seed = _integer(raw["seed"], "bootstrap.seed")
    confidence = _number(raw["confidence_level"], "bootstrap.confidence_level", positive=True)
    neural_critical = _number(
        raw["neural_t_critical_value"], "bootstrap.neural_t_critical_value", positive=True
    )
    neural_df = _integer(
        raw["neural_t_degrees_of_freedom"], "bootstrap.neural_t_degrees_of_freedom", minimum=1
    )
    hgs_critical = _number(
        raw["hgs_t_critical_value"], "bootstrap.hgs_t_critical_value", positive=True
    )
    hgs_df = _integer(
        raw["hgs_t_degrees_of_freedom"], "bootstrap.hgs_t_degrees_of_freedom", minimum=1
    )
    _exact(replicates, 10_000, "bootstrap.replicates")
    _exact(seed, 3495, "bootstrap.seed")
    _exact(confidence, 0.95, "bootstrap.confidence_level")
    _exact(raw["report_mean"], True, "bootstrap.report_mean")
    _exact(raw["report_q95"], True, "bootstrap.report_q95")
    _exact(neural_critical, 2.13184678632665, "bootstrap.neural_t_critical_value")
    _exact(neural_df, 4, "bootstrap.neural_t_degrees_of_freedom")
    _exact(hgs_critical, 1.8331129326536335, "bootstrap.hgs_t_critical_value")
    _exact(hgs_df, 9, "bootstrap.hgs_t_degrees_of_freedom")
    return BootstrapConfig(
        method="crossed_seed_by_instance_percentile",
        replicates=replicates,
        generator="numpy-pcg64",
        seed=seed,
        confidence_level=confidence,
        report_mean=True,
        report_q95=True,
        neural_t_critical_value=neural_critical,
        neural_t_degrees_of_freedom=neural_df,
        hgs_t_critical_value=hgs_critical,
        hgs_t_degrees_of_freedom=hgs_df,
    )


def _parse_gate(value: Any) -> QualityGate:
    raw = _mapping(value, "quality_gate")
    _closed(
        raw,
        {
            "metric",
            "maximum_mean_gap_pct",
            "comparison",
            "maximum_invalid_instances",
            "require_finite",
            "require_each_seed_below_threshold",
            "require_each_seed_empirical_q95_below_threshold",
            "require_pooled_empirical_q95_below_threshold",
            "require_t_ucb_below_threshold",
            "require_bootstrap_mean_ucb_below_threshold",
            "require_bootstrap_q95_ucb_below_threshold",
            "primary_joint_rule",
            "sensitivity_can_rescue_primary",
        },
        "quality_gate",
    )
    _exact(raw["metric"], "mean_gap_to_locked_reference_pct", "quality_gate.metric")
    threshold = _number(
        raw["maximum_mean_gap_pct"], "quality_gate.maximum_mean_gap_pct", positive=True
    )
    _exact(threshold, 5.0, "quality_gate.maximum_mean_gap_pct")
    _exact(raw["comparison"], "strict_less_than", "quality_gate.comparison")
    invalid = _integer(raw["maximum_invalid_instances"], "quality_gate.maximum_invalid_instances")
    _exact(invalid, 0, "quality_gate.maximum_invalid_instances")
    for key in (
        "require_finite",
        "require_each_seed_below_threshold",
        "require_each_seed_empirical_q95_below_threshold",
        "require_pooled_empirical_q95_below_threshold",
        "require_t_ucb_below_threshold",
        "require_bootstrap_mean_ucb_below_threshold",
        "require_bootstrap_q95_ucb_below_threshold",
    ):
        _exact(raw[key], True, f"quality_gate.{key}")
    _exact(raw["primary_joint_rule"], "neural_and_hgs_primary", "quality_gate.primary_joint_rule")
    _exact(
        raw["sensitivity_can_rescue_primary"], False, "quality_gate.sensitivity_can_rescue_primary"
    )
    return QualityGate(
        metric="mean_gap_to_locked_reference_pct",
        maximum_mean_gap_pct=threshold,
        comparison="strict_less_than",
        maximum_invalid_instances=invalid,
        require_finite=True,
        require_each_seed_below_threshold=True,
        require_each_seed_empirical_q95_below_threshold=True,
        require_pooled_empirical_q95_below_threshold=True,
        require_t_ucb_below_threshold=True,
        require_bootstrap_mean_ucb_below_threshold=True,
        require_bootstrap_q95_ucb_below_threshold=True,
        primary_joint_rule="neural_and_hgs_primary",
        sensitivity_can_rescue_primary=False,
    )


def _parse_execution(value: Any) -> ExecutionConfig:
    raw = _mapping(value, "execution")
    _closed(
        raw,
        {
            "max_newly_completed_rounds_per_invocation",
            "energy_measurement",
            "carbon_accounting",
            "preflight_required",
            "exclusive_attestation_required",
            "timing_scientific_use",
        },
        "execution",
    )
    maximum = _integer(
        raw["max_newly_completed_rounds_per_invocation"],
        "execution.max_newly_completed_rounds_per_invocation",
        minimum=1,
    )
    _exact(maximum, 1, "execution.max_newly_completed_rounds_per_invocation")
    _exact(raw["energy_measurement"], "none", "execution.energy_measurement")
    _exact(raw["carbon_accounting"], "none", "execution.carbon_accounting")
    _exact(raw["preflight_required"], False, "execution.preflight_required")
    _exact(raw["exclusive_attestation_required"], False, "execution.exclusive_attestation_required")
    _exact(raw["timing_scientific_use"], False, "execution.timing_scientific_use")
    return ExecutionConfig(maximum, "none", "none", False, False, False)


def _load_confirmatory_recipe(raw: dict[str, Any], path: Path) -> AETHGSHoldoutRecipe:
    """Apply the small confirmatory delta to the already closed holdout policy."""

    _closed(
        raw,
        {
            "schema_version",
            "kind",
            "name",
            "classification",
            "platform",
            "output_root",
            "inherits",
            "policy_source",
            "dataset",
            "hgs_policy",
            "quality_gate",
        },
        "recipe",
    )
    _exact(raw["schema_version"], CONFIRMATORY_SCHEMA_VERSION, "schema_version")
    _exact(raw["kind"], CONFIRMATORY_KIND, "kind")
    name = _slug(raw["name"], "name")
    _exact(name, "aet-hgs10-confirmatory-cvrp50-windows", "name")
    expected_classification = {
        "purpose": "confirmatory_quality_check",
        "stage": "confirmatory",
        "scientific_use": True,
        "quality_evidence_role": "prospective_confirmatory_validation",
        "confirmatory_eligible": True,
        "energy_measurement": "none",
        "aet_eligible": False,
        "cross_solver_energy_comparable": False,
    }
    _exact(
        _mapping(raw["classification"], "classification"), expected_classification, "classification"
    )
    expected_platform = {
        "execution_layer": "windows-native",
        "host_id": "win-a4500-01",
        "accelerator_label": "NVIDIA RTX A4500",
        "gpu_index": 0,
    }
    _exact(_mapping(raw["platform"], "platform"), expected_platform, "platform")
    output_root = _relative_path(raw["output_root"], "output_root")
    _exact(output_root, CONFIRMATORY_OUTPUT_ROOT, "output_root")
    inherited = _relative_path(raw["inherits"], "inherits", suffix=".yaml")
    _exact(inherited, "recipes/aet_journal/hgs_holdout_cvrp50_windows.yaml", "inherits")
    source = _parse_confirmatory_policy_source(raw["policy_source"])

    dataset = _mapping(raw["dataset"], "dataset")
    _closed(dataset, {"holdout", "forbidden_content_sha256"}, "dataset")
    split = _mapping(dataset["holdout"], "dataset.holdout")
    expected_split = {
        "id": "cvrp50-hgs10-confirmatory-seed2723",
        "num_instances": HOLDOUT_INSTANCES,
        "seed": CONFIRMATORY_HOLDOUT_SEED,
        "artifact": "shared/cvrp50-hgs10-confirmatory-seed2723.npz",
    }
    _exact(split, expected_split, "dataset.holdout")
    forbidden_raw = dataset["forbidden_content_sha256"]
    if not isinstance(forbidden_raw, list):
        raise HGSHoldoutRecipeValidationError("dataset.forbidden_content_sha256 must be a list")
    forbidden = tuple(
        _sha256(item, f"dataset.forbidden_content_sha256[{index}]")
        for index, item in enumerate(forbidden_raw)
    )
    _exact(
        forbidden,
        (SELECTION_CORPUS_CONTENT_SHA256, CONFIRMATORY_PRIOR_CORPUS_CONTENT_SHA256),
        "dataset.forbidden_content_sha256",
    )
    hgs_policy = _mapping(raw["hgs_policy"], "hgs_policy")
    _closed(hgs_policy, {"budget", "seeds"}, "hgs_policy")
    _exact(hgs_policy["budget"], CONFIRMATORY_BUDGET, "hgs_policy.budget")
    seeds = _tuple_of_ints(hgs_policy["seeds"], "hgs_policy.seeds")
    _exact(seeds, HGS_SEEDS, "hgs_policy.seeds")
    quality_gate = _mapping(raw["quality_gate"], "quality_gate")
    _closed(quality_gate, {"joint_rule"}, "quality_gate")
    joint_rule = quality_gate["joint_rule"]
    _exact(joint_rule, "neural_and_hgs10", "quality_gate.joint_rule")

    base_path = path.parents[2] / inherited
    base = load_aet_hgs_holdout_recipe(base_path)

    source_patterns = (
        *(
            pattern
            for pattern in base.source_snapshot.include_patterns
            if pattern
            not in {
                "recipes/aet_journal/hgs_holdout_cvrp50_windows.yaml",
                "scripts/aet_journal/run_hgs_holdout_windows.ps1",
            }
        ),
        "recipes/aet_journal/hgs_holdout_cvrp50_windows.yaml",
        "recipes/aet_journal/hgs10_confirmatory_cvrp50_windows.yaml",
        "scripts/aet_journal/run_hgs10_confirmatory_windows.ps1",
    )
    return replace(
        base,
        name=name,
        output_root=output_root,
        selection_source=source,
        source_snapshot=replace(base.source_snapshot, include_patterns=source_patterns),
        dataset=replace(
            base.dataset,
            holdout=DatasetSplit(
                expected_split["id"],
                expected_split["num_instances"],
                expected_split["seed"],
                expected_split["artifact"],
            ),
            forbidden_selection_content_sha256=forbidden[0],
            additional_forbidden_content_sha256=forbidden[1:],
        ),
        hgs_policies=replace(
            base.hgs_policies,
            primary_budget=CONFIRMATORY_BUDGET,
            sensitivity_budget=None,
            seeds=seeds,
            budget_orders=CONFIRMATORY_BUDGET_ORDERS,
        ),
        gate=replace(base.gate, primary_joint_rule=joint_rule),
    )


def load_aet_hgs_holdout_recipe(path: str | Path) -> AETHGSHoldoutRecipe:
    """Load one of the two closed production quality-holdout profiles."""

    recipe_path = Path(path)
    raw = _load_yaml(recipe_path)
    if (
        raw.get("schema_version") == CONFIRMATORY_SCHEMA_VERSION
        or raw.get("kind") == CONFIRMATORY_KIND
    ):
        return _load_confirmatory_recipe(raw, recipe_path)
    _closed(
        raw,
        {
            "schema_version",
            "kind",
            "name",
            "classification",
            "platform",
            "output_root",
            "selection_source",
            "replication_source",
            "source_snapshot",
            "dataset",
            "reference",
            "neural_policy",
            "hgs_policies",
            "bootstrap",
            "quality_gate",
            "execution",
        },
        "recipe",
    )
    _exact(raw["schema_version"], SCHEMA_VERSION, "schema_version")
    _exact(raw["kind"], KIND, "kind")
    name = _slug(raw["name"], "name")
    _exact(name, "aet-hgs-holdout-cvrp50-windows", "name")
    classification = _mapping(raw["classification"], "classification")
    _closed(
        classification,
        {
            "purpose",
            "stage",
            "scientific_use",
            "quality_evidence_role",
            "confirmatory_eligible",
            "energy_measurement",
            "aet_eligible",
            "cross_solver_energy_comparable",
        },
        "classification",
    )
    expected_classification = {
        "purpose": "frozen_quality_holdout",
        "stage": "holdout",
        "scientific_use": True,
        "quality_evidence_role": "internally_preregistered_holdout_validation",
        "confirmatory_eligible": False,
        "energy_measurement": "none",
        "aet_eligible": False,
        "cross_solver_energy_comparable": False,
    }
    for key, expected in expected_classification.items():
        _exact(classification[key], expected, f"classification.{key}")
    platform = _mapping(raw["platform"], "platform")
    _closed(platform, {"execution_layer", "host_id", "accelerator_label", "gpu_index"}, "platform")
    _exact(platform["execution_layer"], "windows-native", "platform.execution_layer")
    _exact(platform["host_id"], "win-a4500-01", "platform.host_id")
    _exact(platform["accelerator_label"], "NVIDIA RTX A4500", "platform.accelerator_label")
    gpu_index = _integer(platform["gpu_index"], "platform.gpu_index")
    _exact(gpu_index, 0, "platform.gpu_index")
    output_root = _relative_path(raw["output_root"], "output_root")
    _exact(output_root, PRODUCTION_OUTPUT_ROOT, "output_root")
    recipe = AETHGSHoldoutRecipe(
        name=name,
        output_root=output_root,
        host_id="win-a4500-01",
        execution_layer="windows-native",
        gpu_index=gpu_index,
        expected_accelerator_label="NVIDIA RTX A4500",
        selection_source=_parse_selection_source(raw["selection_source"]),
        replication_source=_parse_replication_source(raw["replication_source"]),
        source_snapshot=_parse_source_snapshot(raw["source_snapshot"]),
        dataset=_parse_dataset(raw["dataset"]),
        reference=_parse_reference(raw["reference"]),
        neural_policy=_parse_neural_policy(raw["neural_policy"]),
        hgs_policies=_parse_hgs_policies(raw["hgs_policies"]),
        bootstrap=_parse_bootstrap(raw["bootstrap"]),
        gate=_parse_gate(raw["quality_gate"]),
        execution=_parse_execution(raw["execution"]),
    )
    if set(recipe.neural_policy.evaluation_seeds) & set(recipe.hgs_policies.seeds):
        raise HGSHoldoutRecipeValidationError("neural and HGS evaluation seeds must be disjoint")
    all_candidate_seeds = set(recipe.hgs_policies.seeds)
    if all_candidate_seeds & set(recipe.reference.hgs.seeds):
        raise HGSHoldoutRecipeValidationError("reference and candidate HGS seeds must be disjoint")
    return recipe


def _repository_root(path: Path) -> Path:
    for candidate in (path.parent, *path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "uv.lock").is_file():
            return candidate.resolve()
    raise HGSHoldoutRecipeValidationError("cannot locate repository root")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_qualification(root: Path, artifacts: dict[str, ArtifactEvidence]) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for name, artifact in artifacts.items():
        path = root.joinpath(*PurePosixPath(artifact.path).parts)
        exists = path.is_file()
        observed = _hash_file(path) if exists else None
        checks[name] = {
            "path": path.as_posix(),
            "exists": exists,
            "expected_sha256": artifact.sha256,
            "observed_sha256": observed,
            "sha256_matches": observed == artifact.sha256,
        }
    return checks


def _selection_source_qualification(recipe: AETHGSHoldoutRecipe, root: Path) -> dict[str, Any]:
    source = recipe.selection_source
    if isinstance(source, ConfirmatoryPolicySource):
        return _confirmatory_policy_source_qualification(source, root)
    source_root = root.joinpath(*PurePosixPath(source.root).parts)
    artifacts = {name: getattr(source, name) for name in SELECTION_ARTIFACTS}
    checks = _artifact_qualification(source_root, artifacts)
    semantics: dict[str, bool] = {}
    error: str | None = None
    if all(item["sha256_matches"] for item in checks.values()):
        try:
            state = json.loads((source_root / source.run_state.path).read_text(encoding="utf-8"))
            assessment = json.loads(
                (source_root / source.assessment.path).read_text(encoding="utf-8")
            )
            manifest = json.loads((source_root / source.manifest.path).read_text(encoding="utf-8"))
            corpus_manifest = json.loads(
                (source_root / source.selection_corpus_manifest.path).read_text(encoding="utf-8")
            )
            semantics = {
                "run_state_status_matches": state.get("status") == source.expected_status,
                "assessment_status_matches": assessment.get("status") == source.expected_status,
                "manifest_status_matches": manifest.get("status") == source.expected_status,
                "selected_budget_matches": state.get("selected_budget")
                == source.selected_budget
                == assessment.get("selected_budget"),
                "sensitivity_was_passing": source.sensitivity_budget
                in assessment.get("passing_budgets", []),
                "energy_did_not_select": assessment.get("energy_used_for_selection") is False,
                "holdout_was_not_executed": assessment.get("holdout_executed") is False,
                "aet_was_not_computed": assessment.get("aet_was_computed") is False,
                "manifest_hash_matches": state.get("manifest_sha256") == source.manifest.sha256,
                "checksums_hash_matches": state.get("checksums_sha256") == source.checksums.sha256,
                "recipe_hash_matches": state.get("recipe_sha256") == source.recipe.sha256,
                "reference_hash_matches": state.get("reference_lock_sha256")
                == source.reference_lock.sha256,
                "neural_gate_hash_matches": state.get("neural_quality_gate_sha256")
                == source.neural_quality_gate.sha256,
                "git_sha_matches": state.get("git_sha") == source.source_git_sha,
                "snapshot_hash_matches": state.get("source_snapshot", {}).get("sha256")
                == source.source_snapshot_sha256,
                "selection_corpus_hash_matches": corpus_manifest.get("content_sha256")
                == source.selection_corpus_content_sha256,
                "selection_seed_matches": corpus_manifest.get("seed") == 2721,
                "selection_size_matches": corpus_manifest.get("num_instances") == HOLDOUT_INSTANCES,
            }
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    ready = bool(
        checks
        and all(item["sha256_matches"] for item in checks.values())
        and semantics
        and all(semantics.values())
        and error is None
    )
    return {
        "root": source_root.as_posix(),
        "artifact_checks": checks,
        "semantic_checks": semantics,
        "error": error,
        "ready": ready,
    }


def _confirmatory_policy_source_qualification(
    source: ConfirmatoryPolicySource,
    root: Path,
) -> dict[str, Any]:
    source_root = root.joinpath(*PurePosixPath(source.root).parts)
    artifacts = {name: getattr(source, name) for name in CONFIRMATORY_POLICY_ARTIFACTS}
    checks = _artifact_qualification(source_root, artifacts)
    semantics: dict[str, bool] = {}
    error: str | None = None
    if all(item["sha256_matches"] for item in checks.values()):
        try:
            state = json.loads((source_root / source.run_state.path).read_text(encoding="utf-8"))
            assessment = json.loads(
                (source_root / source.assessment.path).read_text(encoding="utf-8")
            )
            manifest = json.loads((source_root / source.manifest.path).read_text(encoding="utf-8"))
            corpus_manifest = json.loads(
                (source_root / source.prior_corpus_manifest.path).read_text(encoding="utf-8")
            )
            semantics = {
                "run_state_status_matches": state.get("status") == source.expected_status,
                "assessment_status_matches": assessment.get("status") == source.expected_status,
                "manifest_status_matches": manifest.get("status") == source.expected_status,
                "neural_passed": assessment.get("policies", {}).get("neural", {}).get("passed")
                is True,
                "hgs10_passed": assessment.get("policies", {}).get("hgs_b10", {}).get("passed")
                is True,
                "joint_hgs10_decision_passed": assessment.get("sensitivity_decision", {}).get(
                    "joint_passed"
                )
                is True,
                "confirmed_budget_matches": assessment.get("sensitivity_decision", {}).get("budget")
                == source.confirmed_budget,
                "energy_was_not_measured": assessment.get("energy_measurement") == "none",
                "aet_was_not_computed": assessment.get("aet_was_computed") is False,
                "manifest_hash_matches": state.get("manifest_sha256") == source.manifest.sha256,
                "checksums_hash_matches": state.get("checksums_sha256") == source.checksums.sha256,
                "assessment_hash_matches": state.get("assessment_sha256")
                == source.assessment.sha256,
                "reference_hash_matches": state.get("reference_lock_sha256")
                == source.reference_lock.sha256,
                "git_sha_matches": state.get("git_sha") == source.source_git_sha,
                "snapshot_hash_matches": state.get("source_snapshot", {}).get("sha256")
                == source.source_snapshot_sha256,
                "prior_corpus_hash_matches": corpus_manifest.get("content_sha256")
                == source.prior_corpus_content_sha256,
                "prior_seed_matches": corpus_manifest.get("seed") == HOLDOUT_SEED,
                "prior_size_matches": corpus_manifest.get("num_instances") == HOLDOUT_INSTANCES,
            }
        except (OSError, UnicodeError, ValueError, TypeError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
    ready = bool(
        checks
        and all(item["sha256_matches"] for item in checks.values())
        and semantics
        and all(semantics.values())
        and error is None
    )
    return {
        "root": source_root.as_posix(),
        "artifact_checks": checks,
        "semantic_checks": semantics,
        "error": error,
        "ready": ready,
    }


def _replication_source_qualification(recipe: AETHGSHoldoutRecipe, root: Path) -> dict[str, Any]:
    source = recipe.replication_source
    source_root = root.joinpath(*PurePosixPath(source.root).parts)
    artifacts = {name: getattr(source, name) for name in REPLICATION_ARTIFACTS}
    checks = _artifact_qualification(source_root, artifacts)
    checkpoints: list[dict[str, Any]] = []
    for checkpoint in source.checkpoints:
        path = source_root.joinpath(*PurePosixPath(checkpoint.path).parts)
        exists = path.is_file()
        observed = _hash_file(path) if exists else None
        checkpoints.append(
            {
                "training_seed": checkpoint.training_seed,
                "path": path.as_posix(),
                "exists": exists,
                "expected_sha256": checkpoint.sha256,
                "observed_sha256": observed,
                "sha256_matches": observed == checkpoint.sha256,
            }
        )
    ready = bool(
        checks
        and all(item["sha256_matches"] for item in checks.values())
        and len(checkpoints) == 5
        and all(item["sha256_matches"] for item in checkpoints)
    )
    return {
        "root": source_root.as_posix(),
        "artifact_checks": checks,
        "checkpoint_checks": checkpoints,
        "ready": ready,
    }


def _runtime_report(recipe: AETHGSHoldoutRecipe) -> dict[str, Any]:
    report: dict[str, Any] = {
        "platform": sys.platform,
        "windows_native": sys.platform == "win32",
        "torch_importable": False,
        "cuda_available": False,
        "gpu_index_valid": False,
        "gpu_name": None,
        "gpu_identity_matches": False,
        "error": None,
    }
    try:
        import torch

        report["torch_importable"] = True
        report["torch_version"] = torch.__version__
        report["cuda_available"] = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if report["cuda_available"] else 0
        report["cuda_device_count"] = count
        report["gpu_index_valid"] = 0 <= recipe.gpu_index < count
        if report["gpu_index_valid"]:
            name = str(torch.cuda.get_device_name(recipe.gpu_index))
            report["gpu_name"] = name
            report["gpu_identity_matches"] = name == recipe.expected_accelerator_label
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    report["ready"] = bool(
        report["windows_native"]
        and report["torch_importable"]
        and report["cuda_available"]
        and report["gpu_index_valid"]
        and report["gpu_identity_matches"]
        and report["error"] is None
    )
    return report


def runtime_qualification(
    recipe: AETHGSHoldoutRecipe,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Check sources and the native-Windows CUDA runtime without writing anything."""

    root = Path.cwd().resolve() if repository_root is None else Path(repository_root).resolve()
    selection = _selection_source_qualification(recipe, root)
    replication_source = _replication_source_qualification(recipe, root)
    runtime = _runtime_report(recipe)
    output = root.joinpath(*PurePosixPath(recipe.output_root).parts)
    output_absent = not output.exists()
    resumable = (output / "run-state.json").is_file()
    output_safe = output_absent or resumable
    ready = bool(
        selection["ready"] and replication_source["ready"] and runtime["ready"] and output_safe
    )
    source_key = "policy_source" if recipe.confirmatory_profile else "selection_source"
    return {
        source_key: selection,
        "replication_source": replication_source,
        "runtime": runtime,
        "output": {
            "path": output.as_posix(),
            "absent_before_first_open": output_absent,
            "resumable_run_state_present": resumable,
            "safe": output_safe,
        },
        "energy_measurement": "none",
        "preflight_required": False,
        "exclusive_attestation_required": False,
        "ready_to_execute": ready,
    }


def dry_run(path: str | Path) -> dict[str, Any]:
    """Validate a closed protocol without generating or loading its corpus."""

    recipe_path = Path(path).resolve(strict=True)
    recipe = load_aet_hgs_holdout_recipe(recipe_path)
    root = _repository_root(recipe_path)
    before = set(root.rglob("*"))
    qualification = runtime_qualification(recipe, repository_root=root)
    after = set(root.rglob("*"))
    if before != after:
        raise HGSHoldoutRecipeValidationError("holdout dry-run performed a filesystem write")
    ready = qualification["ready_to_execute"] is True
    confirmatory = recipe.confirmatory_profile
    policies = (
        {
            "neural": recipe.neural_policy.mode_id,
            "hgs_budget": recipe.hgs_policies.primary_budget,
            "joint_rule": recipe.gate.primary_joint_rule,
        }
        if confirmatory
        else {
            "neural": recipe.neural_policy.mode_id,
            "hgs_primary_budget": recipe.hgs_policies.primary_budget,
            "hgs_sensitivity_budget": recipe.hgs_policies.sensitivity_budget,
            "sensitivity_can_rescue_primary": False,
        }
    )
    return {
        "schema_version": CONFIRMATORY_SCHEMA_VERSION if confirmatory else SCHEMA_VERSION,
        "name": recipe.name,
        "purpose": "confirmatory_quality_check" if confirmatory else "frozen_quality_holdout",
        "status": (
            "valid_ready_confirmatory_quality"
            if ready and confirmatory
            else "valid_ready_quality_holdout"
            if ready
            else "valid_not_qualified"
        ),
        "ready_to_execute": ready,
        "output_root": recipe.output_root,
        "holdout": {
            "seed": recipe.dataset.holdout.seed,
            "num_instances": recipe.dataset.holdout.num_instances,
            "generated_or_loaded": False,
            "protection": "dry_run_never_opens_holdout",
        },
        "policies": policies,
        "energy_measurement": "none",
        "carbon_accounting": "none",
        "preflight_required": False,
        "exclusive_attestation_required": False,
        "qualification": qualification,
        "writes_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = dry_run(args.recipe)
    except (HGSHoldoutRecipeValidationError, OSError, ValueError) as exc:
        print(f"HGS holdout recipe invalid: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_ready and payload["ready_to_execute"] is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
