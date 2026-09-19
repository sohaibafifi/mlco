"""Resumable multi-seed quality replication for the AET journal study.

This runner records no energy and never computes AET.  It imports the sealed
seed-1 discovery bundle only as provenance, builds one new common development
corpus and reference, then trains and evaluates fresh seeds 2 through 6.  At
most one seed is newly completed by an invocation so that long Windows runs
have an explicit, automation-friendly resume boundary.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import unicodedata
import uuid
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import neuro_co.aet.experiments.quality_runner as quality
from neuro_co.aet.experiments.quality_recipe import AETQualityRecipe, DatasetSplit, EvaluationMode
from neuro_co.aet.experiments.quality_replication_recipe import (
    AETQualityReplicationRecipe,
    load_aet_quality_replication_recipe,
    runtime_qualification,
)

RUN_STATE_SCHEMA = "aet-quality-replication-run-state/v1"
REFERENCE_LOCK_SCHEMA = "aet-quality-replication-reference-lock/v1"
EVALUATION_SCHEMA = "aet-quality-replication-evaluation/v1"
SEED_RESULT_SCHEMA = "aet-quality-replication-seed-result/v1"
ASSESSMENT_SCHEMA = "aet-quality-replication-assessment/v1"
MANIFEST_SCHEMA = "aet-quality-replication-manifest/v1"
IMPORT_RECEIPT_SCHEMA = "aet-quality-replication-discovery-import/v1"

COMPLETE_STATUSES = {
    "complete_replication_passed",
    "complete_replication_nonpass",
}
INCOMPLETE_STATUS = "incomplete_seeds_pending"
T_CRITICAL_ONE_SIDED_95_DF4 = 2.13184678632665
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_MEMBER_BYTES = 1_073_741_824
MAX_ZIP_TOTAL_BYTES = 2_147_483_648
MAX_ZIP_COMPRESSION_RATIO = 1_000.0
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


def _classification() -> dict[str, Any]:
    return {
        "purpose": "quality_exploratory",
        "scientific_use": False,
        "aet_eligible": False,
        "energy_measurement": "none",
        "discovery_seed_inference_excluded": True,
    }


def _strict_json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-like values without accepting bool/int aliases."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _strict_json_equal(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_json_equal(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


class QualityReplicationError(RuntimeError):
    """Raised for a technical or integrity failure that can be retried."""


class QualityReplicationQualificationError(QualityReplicationError):
    """Raised when runtime qualification or discovery provenance is invalid."""


@dataclass(frozen=True, slots=True)
class ReplicationResult:
    path: Path
    status: str
    complete: bool
    manifest_path: Path | None = None
    manifest_sha256: str | None = None
    primary_classification: str | None = None
    completed_seeds: tuple[int, ...] = ()
    remaining_seeds: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscoveryEvidence:
    outer_sha256: str
    manifest_sha256: str
    checksums_sha256: str
    status: str
    git_sha: str
    recipe_semantic_sha256: str
    checkpoint_epoch: int
    checkpoint_sha256: str


def _yaml_semantic_sha256(payload: bytes) -> str:
    try:
        import yaml

        value = yaml.safe_load(payload)
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except Exception as exc:
        raise QualityReplicationQualificationError(
            f"cannot canonicalize YAML semantics: {exc}"
        ) from exc
    return quality._sha256_bytes(canonical)


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise QualityReplicationQualificationError(f"{label} is not a lowercase SHA-256")
    return value


def _canonical_zip_member(raw_name: str) -> tuple[str, bool]:
    """Canonicalize slash styles and reject paths unsafe on POSIX or Windows."""

    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise QualityReplicationQualificationError("discovery ZIP has an empty or NUL path")
    normalized = raw_name.replace("\\", "/")
    is_directory = normalized.endswith("/")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise QualityReplicationQualificationError(
            f"discovery ZIP has an absolute member path: {raw_name!r}"
        )
    raw_parts = normalized[:-1].split("/") if is_directory else normalized.split("/")
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise QualityReplicationQualificationError(
            f"discovery ZIP has an unsafe member path: {raw_name!r}"
        )
    parts: list[str] = []
    for raw_part in raw_parts:
        part = unicodedata.normalize("NFC", raw_part)
        if (
            any(ord(character) < 32 for character in part)
            or any(character in '<>:"|?*' for character in part)
            or part.endswith((" ", "."))
            or part.split(".", maxsplit=1)[0].casefold() in _WINDOWS_RESERVED
        ):
            raise QualityReplicationQualificationError(
                f"discovery ZIP has a Windows-unsafe member path: {raw_name!r}"
            )
        parts.append(part)
    return "/".join(parts), is_directory


def _zip_member_is_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = info.external_attr >> 16
    return bool(info.create_system == 3 and unix_mode and stat.S_ISLNK(unix_mode))


def _zip_member_is_special(info: zipfile.ZipInfo, *, is_directory: bool) -> bool:
    unix_mode = info.external_attr >> 16
    if info.create_system != 3 or unix_mode == 0:
        return False
    file_type = stat.S_IFMT(unix_mode)
    if file_type == 0:
        return False
    return file_type != (stat.S_IFDIR if is_directory else stat.S_IFREG)


def _extract_zip_safely(archive_path: Path, destination: Path) -> tuple[str, ...]:
    """Extract a bounded ZIP after validating its complete canonical member tree."""

    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise QualityReplicationQualificationError(
            f"cannot open discovery ZIP {archive_path}: {exc}"
        ) from exc
    with archive:
        infos = archive.infolist()
        if not infos or len(infos) > MAX_ZIP_ENTRIES:
            raise QualityReplicationQualificationError(
                "discovery ZIP has an empty or oversized member inventory"
            )
        canonical: dict[str, tuple[zipfile.ZipInfo, bool]] = {}
        casefolded: dict[str, str] = {}
        files: set[str] = set()
        directories: set[str] = set()
        total_size = 0
        for info in infos:
            name, directory = _canonical_zip_member(info.filename)
            folded = name.casefold()
            if name in canonical:
                raise QualityReplicationQualificationError(
                    f"discovery ZIP has a duplicate canonical path: {name}"
                )
            if folded in casefolded and casefolded[folded] != name:
                raise QualityReplicationQualificationError(
                    "discovery ZIP has a case-colliding canonical path: "
                    f"{casefolded[folded]} and {name}"
                )
            if info.flag_bits & 0x1:
                raise QualityReplicationQualificationError(
                    f"discovery ZIP has an encrypted member: {name}"
                )
            if _zip_member_is_symlink(info) or _zip_member_is_special(info, is_directory=directory):
                raise QualityReplicationQualificationError(
                    f"discovery ZIP has a link or special member: {name}"
                )
            if info.file_size < 0 or info.compress_size < 0:
                raise QualityReplicationQualificationError(
                    f"discovery ZIP has invalid sizes for member: {name}"
                )
            if not directory:
                if info.file_size > MAX_ZIP_MEMBER_BYTES:
                    raise QualityReplicationQualificationError(
                        f"discovery ZIP member is too large: {name}"
                    )
                total_size += info.file_size
                if total_size > MAX_ZIP_TOTAL_BYTES:
                    raise QualityReplicationQualificationError(
                        "discovery ZIP expands beyond the total size limit"
                    )
                if (info.file_size > 0 and info.compress_size == 0) or (
                    info.compress_size > 0
                    and info.file_size / info.compress_size > MAX_ZIP_COMPRESSION_RATIO
                ):
                    raise QualityReplicationQualificationError(
                        f"discovery ZIP member has an unsafe compression ratio: {name}"
                    )
            canonical[name] = (info, directory)
            casefolded[folded] = name
            (directories if directory else files).add(name)

        for file_name in files:
            parts = PurePosixPath(file_name).parts
            if any("/".join(parts[:index]) in files for index in range(1, len(parts))):
                raise QualityReplicationQualificationError(
                    f"discovery ZIP file shadows a parent path: {file_name}"
                )
            prefix = file_name + "/"
            if any(other.startswith(prefix) for other in canonical if other != file_name):
                raise QualityReplicationQualificationError(
                    f"discovery ZIP file shadows a child path: {file_name}"
                )

        destination.mkdir(parents=True, exist_ok=False)
        root = destination.resolve(strict=True)
        for name in sorted(canonical, key=lambda value: (value.count("/"), value)):
            info, directory = canonical[name]
            target = destination.joinpath(*PurePosixPath(name).parts)
            resolved = target.resolve(strict=False)
            try:
                resolved.relative_to(root)
            except ValueError as exc:  # pragma: no cover - canonical path check is primary
                raise QualityReplicationQualificationError(
                    f"discovery ZIP member escaped extraction root: {name}"
                ) from exc
            if directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
            written = 0
            try:
                with archive.open(info, "r") as source, temporary.open("xb") as sink:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > info.file_size or written > MAX_ZIP_MEMBER_BYTES:
                            raise QualityReplicationQualificationError(
                                f"discovery ZIP member exceeded its declared size: {name}"
                            )
                        sink.write(chunk)
                if written != info.file_size:
                    raise QualityReplicationQualificationError(
                        f"discovery ZIP member size changed while reading: {name}"
                    )
                os.replace(temporary, target)
            except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise QualityReplicationQualificationError(
                    f"cannot extract discovery ZIP member {name}: {exc}"
                ) from exc
            finally:
                if temporary.exists():
                    temporary.unlink()
    return tuple(sorted(files))


def _crossed_bootstrap_means(
    matrices: Mapping[str, Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap seed and instance axes with shared draws for every mode."""

    import numpy as np

    if replicates < 1:
        raise QualityReplicationError("bootstrap replicates must be positive")
    arrays = {mode: np.asarray(matrix, dtype=np.float64) for mode, matrix in matrices.items()}
    if not arrays:
        raise QualityReplicationError("bootstrap needs at least one mode matrix")
    shapes = {array.shape for array in arrays.values()}
    if len(shapes) != 1:
        raise QualityReplicationError("bootstrap mode matrices must have one common shape")
    shape = next(iter(shapes))
    if len(shape) != 2:
        raise QualityReplicationError("bootstrap mode matrices must be two-dimensional")
    n_seeds, n_instances = shape
    if n_seeds < 2 or n_instances < 1:
        raise QualityReplicationError("bootstrap needs at least two seeds and one instance")
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = {mode: np.empty(replicates, dtype=np.float64) for mode in arrays}
    for replicate in range(replicates):
        seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        instance_indices = rng.integers(0, n_instances, size=n_instances)
        for mode, array in arrays.items():
            selected = array[seed_indices[:, None], instance_indices[None, :]]
            samples[mode][replicate] = selected.mean()
    return samples


def _mode_classification(
    *,
    finite: bool,
    invalid_instances: int,
    mean_gap_pct: float | None,
    maximum_seed_mean_gap_pct: float | None,
    t_ucb_pct: float | None,
    bootstrap_ucb_pct: float | None,
    threshold_pct: float,
) -> str:
    if not finite or invalid_instances > 0:
        return "invalid"
    if (
        t_ucb_pct is None
        or bootstrap_ucb_pct is None
        or mean_gap_pct is None
        or maximum_seed_mean_gap_pct is None
    ):
        return "invalid"
    if (
        maximum_seed_mean_gap_pct < threshold_pct
        and t_ucb_pct < threshold_pct
        and bootstrap_ucb_pct < threshold_pct
    ):
        return "pass"
    if mean_gap_pct >= threshold_pct or maximum_seed_mean_gap_pct >= threshold_pct:
        return "fail_quality"
    return "inconclusive"


def _summarize_gap_matrices(
    matrices: Mapping[str, Any],
    *,
    invalid_instances: Mapping[str, int],
    threshold_pct: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    bootstrap_quantile: float = 0.95,
    t_critical: float = T_CRITICAL_ONE_SIDED_95_DF4,
) -> dict[str, dict[str, Any]]:
    import numpy as np

    arrays = {mode: np.asarray(matrix, dtype=np.float64) for mode, matrix in matrices.items()}
    if set(invalid_instances) != set(arrays):
        raise QualityReplicationError(
            "invalid-instance inventory must exactly match bootstrap modes"
        )
    bootstrap = _crossed_bootstrap_means(
        arrays, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    summaries: dict[str, dict[str, Any]] = {}
    for mode, matrix in arrays.items():
        if matrix.ndim != 2 or matrix.shape[0] != 5:
            raise QualityReplicationError(
                f"replication matrix for {mode} must have shape [5, instances]"
            )
        finite = bool(np.isfinite(matrix).all())
        seed_means = matrix.mean(axis=1)
        mean = float(seed_means.mean()) if finite else None
        sample_std = float(seed_means.std(ddof=1)) if finite else None
        t_ucb = (
            float(mean + t_critical * sample_std / math.sqrt(matrix.shape[0]))
            if mean is not None and sample_std is not None
            else None
        )
        bootstrap_ucb = (
            float(np.quantile(bootstrap[mode], bootstrap_quantile, method="linear"))
            if finite
            else None
        )
        invalid = int(invalid_instances[mode])
        classification = _mode_classification(
            finite=finite,
            invalid_instances=invalid,
            mean_gap_pct=mean,
            maximum_seed_mean_gap_pct=(float(seed_means.max()) if finite else None),
            t_ucb_pct=t_ucb,
            bootstrap_ucb_pct=bootstrap_ucb,
            threshold_pct=threshold_pct,
        )
        summaries[mode] = {
            "classification": classification,
            "finite": finite,
            "invalid_instances": invalid,
            "gap_matrix_shape": list(matrix.shape),
            "gap_matrix_pct": matrix.tolist() if finite else None,
            "per_seed_mean_gap_pct": seed_means.tolist() if finite else None,
            "mean_of_seed_means_gap_pct": mean,
            "sample_std_of_seed_means_gap_pct": sample_std,
            "median_seed_mean_gap_pct": float(np.median(seed_means)) if finite else None,
            "minimum_seed_mean_gap_pct": float(seed_means.min()) if finite else None,
            "maximum_seed_mean_gap_pct": float(seed_means.max()) if finite else None,
            "t_ucb_95_pct": t_ucb,
            "bootstrap_ucb_95_pct": bootstrap_ucb,
            "maximum_mean_gap_pct": threshold_pct,
        }
    return summaries


def _attribute(value: Any, *names: str) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
        if isinstance(value, Mapping) and name in value:
            return value[name]
    joined = ", ".join(names)
    raise QualityReplicationQualificationError(f"recipe is missing required field: {joined}")


def _discovery_archive_path(recipe: AETQualityReplicationRecipe, root: Path) -> Path:
    discovery = _attribute(recipe, "discovery_zip", "discovery", "discovery_bundle")
    raw = str(_attribute(discovery, "archive", "path", "archive_path"))
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else root / candidate


def _base_recipe_path(recipe: AETQualityReplicationRecipe, root: Path) -> Path:
    base = _attribute(recipe, "base_recipe", "source_recipe")
    raw = str(_attribute(base, "path", "recipe_path"))
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else root / candidate
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise QualityReplicationQualificationError(
            "base quality recipe must be stored inside the repository"
        ) from exc
    return path.resolve(strict=True)


def _expected_discovery(recipe: AETQualityReplicationRecipe) -> dict[str, Any]:
    discovery = _attribute(recipe, "discovery_zip", "discovery", "discovery_bundle")
    return {
        "outer_sha256": _require_sha256(
            _attribute(discovery, "sha256", "archive_sha256"), "discovery archive hash"
        ),
        "manifest_sha256": _require_sha256(
            _attribute(discovery, "manifest_sha256"), "discovery manifest hash"
        ),
        "checksums_sha256": _require_sha256(
            _attribute(discovery, "checksums_sha256", "checksum_inventory_sha256"),
            "discovery checksum inventory hash",
        ),
        "status": str(_attribute(discovery, "status", "expected_status")),
        "git_sha": str(_attribute(discovery, "git_sha", "expected_git_sha")),
        "checkpoint_epoch": int(
            _attribute(discovery, "checkpoint_epoch", "expected_checkpoint_epoch")
        ),
        "checkpoint_sha256": _require_sha256(
            _attribute(discovery, "checkpoint_sha256", "expected_checkpoint_sha256"),
            "discovery checkpoint hash",
        ),
    }


def _expected_base_semantic_sha256(recipe: AETQualityReplicationRecipe) -> str:
    base = _attribute(recipe, "base_recipe", "source_recipe")
    return _require_sha256(
        _attribute(base, "semantic_sha256", "sha256"), "base recipe semantic hash"
    )


def _inspect_discovery_archive(
    archive_path: Path,
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
) -> DiscoveryEvidence:
    """Validate the discovery ZIP and its complete old-run semantics."""

    expected = _expected_discovery(recipe)
    if not archive_path.is_file():
        raise QualityReplicationQualificationError(
            f"sealed discovery archive is unavailable: {archive_path}"
        )
    outer_sha256 = quality._sha256_file(archive_path)
    if outer_sha256 != expected["outer_sha256"]:
        raise QualityReplicationQualificationError("sealed discovery archive hash changed")
    with tempfile.TemporaryDirectory(prefix="aet-discovery-validate-") as raw_temporary:
        temporary = Path(raw_temporary)
        extracted = temporary / "bundle"
        inventory = set(_extract_zip_safely(archive_path, extracted))
        required = {
            "SHA256SUMS",
            "manifest.json",
            "run-state.json",
            "recipe.yaml",
            "training/result.json",
            f"training/checkpoints/epoch-{expected['checkpoint_epoch']:03d}.pt",
        }
        missing = sorted(required - inventory)
        if missing:
            raise QualityReplicationQualificationError(
                f"sealed discovery archive lacks required members: {missing}"
            )
        state = quality._load_json(extracted / "run-state.json")
        if (
            state.get("status") != expected["status"]
            or state.get("git_sha") != expected["git_sha"]
            or state.get("manifest_sha256") != expected["manifest_sha256"]
            or state.get("checksums_sha256") != expected["checksums_sha256"]
        ):
            raise QualityReplicationQualificationError(
                "sealed discovery run-state disagrees with frozen provenance"
            )
        if quality._sha256_file(extracted / "manifest.json") != expected["manifest_sha256"]:
            raise QualityReplicationQualificationError("sealed discovery manifest hash changed")
        if quality._sha256_file(extracted / "SHA256SUMS") != expected["checksums_sha256"]:
            raise QualityReplicationQualificationError(
                "sealed discovery checksum inventory hash changed"
            )
        manifest = quality._load_json(extracted / "manifest.json")
        expected_primary = recipe.discovery_zip.expected_primary_mode
        primary_selection = (
            manifest.get("checkpoint_selection", {}).get("selections", {}).get(expected_primary)
        )
        primary_gate = (
            manifest.get("quality_gate", {}).get("mode_results", {}).get(expected_primary)
        )
        if (
            not isinstance(primary_selection, dict)
            or primary_selection.get("epoch") != expected["checkpoint_epoch"]
            or primary_selection.get("checkpoint_sha256") != expected["checkpoint_sha256"]
            or not isinstance(primary_gate, dict)
            or primary_gate.get("passed") is not True
            or primary_gate.get("selected_epoch") != expected["checkpoint_epoch"]
            or primary_gate.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        ):
            raise QualityReplicationQualificationError(
                "sealed discovery primary-mode checkpoint evidence changed"
            )
        try:
            quality._verify_checksums(extracted, expected_sha256=expected["checksums_sha256"])
            (
                validated_manifest,
                validated_manifest_sha256,
                _passing_modes,
                validated_status,
            ) = quality._validate_completed_bundle(base_recipe, extracted, state)
        except quality.QualityPilotError as exc:
            raise QualityReplicationQualificationError(
                f"sealed discovery bundle failed its original validator: {exc}"
            ) from exc
        if (
            validated_manifest != extracted / "manifest.json"
            or validated_manifest_sha256 != expected["manifest_sha256"]
            or validated_status != expected["status"]
        ):
            raise QualityReplicationQualificationError(
                "sealed discovery semantic validator returned unexpected evidence"
            )

        try:
            quality.load_aet_quality_recipe(extracted / "recipe.yaml")
        except Exception as exc:
            raise QualityReplicationQualificationError(
                f"sealed discovery recipe is invalid: {exc}"
            ) from exc
        archived_semantic = _yaml_semantic_sha256((extracted / "recipe.yaml").read_bytes())
        expected_semantic = _expected_base_semantic_sha256(recipe)
        if archived_semantic != expected_semantic:
            raise QualityReplicationQualificationError(
                "discovery recipe does not match the frozen semantic recipe hash"
            )

        training = quality._load_json(extracted / "training" / "result.json")
        checkpoint_records = training.get("checkpoint_records")
        if not isinstance(checkpoint_records, list) or training.get("seed") != 1:
            raise QualityReplicationQualificationError(
                "sealed discovery training metadata does not identify seed 1"
            )
        matching = [
            record
            for record in checkpoint_records
            if isinstance(record, dict) and record.get("epoch") == expected["checkpoint_epoch"]
        ]
        checkpoint_path = (
            extracted / "training" / "checkpoints" / f"epoch-{expected['checkpoint_epoch']:03d}.pt"
        )
        if (
            len(matching) != 1
            or matching[0].get("path")
            != f"training/checkpoints/epoch-{expected['checkpoint_epoch']:03d}.pt"
            or matching[0].get("sha256") != expected["checkpoint_sha256"]
            or quality._sha256_file(checkpoint_path) != expected["checkpoint_sha256"]
        ):
            raise QualityReplicationQualificationError(
                "sealed discovery epoch-40 checkpoint metadata changed"
            )
        checkpoint = quality._load_torch_mapping(checkpoint_path)
        if (
            checkpoint.get("completed_epochs") != expected["checkpoint_epoch"]
            or checkpoint.get("git_sha") != expected["git_sha"]
            or checkpoint.get("training", {}).get("seed") != 1
            or checkpoint.get("recipe_sha256") != state.get("recipe_sha256")
        ):
            raise QualityReplicationQualificationError(
                "sealed discovery checkpoint payload metadata changed"
            )
    return DiscoveryEvidence(
        outer_sha256=outer_sha256,
        manifest_sha256=expected["manifest_sha256"],
        checksums_sha256=expected["checksums_sha256"],
        status=expected["status"],
        git_sha=expected["git_sha"],
        recipe_semantic_sha256=_expected_base_semantic_sha256(recipe),
        checkpoint_epoch=expected["checkpoint_epoch"],
        checkpoint_sha256=expected["checkpoint_sha256"],
    )


def _atomic_copy_verified(source: Path, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
    try:
        with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        if quality._sha256_file(temporary) != expected_sha256:
            raise QualityReplicationQualificationError(
                "discovery archive changed during atomic import"
            )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _discovery_receipt(
    evidence: DiscoveryEvidence, *, archive_relative_path: str
) -> dict[str, Any]:
    return {
        "schema_version": IMPORT_RECEIPT_SCHEMA,
        "status": "validated_and_imported",
        "archive_path": archive_relative_path,
        "archive_sha256": evidence.outer_sha256,
        "source_manifest_sha256": evidence.manifest_sha256,
        "source_checksums_sha256": evidence.checksums_sha256,
        "source_status": evidence.status,
        "source_git_sha": evidence.git_sha,
        "source_recipe_semantic_sha256": evidence.recipe_semantic_sha256,
        "source_training_seed": 1,
        "source_checkpoint_epoch": evidence.checkpoint_epoch,
        "source_checkpoint_sha256": evidence.checkpoint_sha256,
        "evaluated": False,
        "included_in_aggregation": False,
        "reason": "pilot outcome observed before replication freeze",
    }


def _record_invocation(
    output: Path,
    state: dict[str, Any],
    qualification: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> None:
    invocations = state.setdefault("invocations", [])
    if not isinstance(invocations, list):
        raise QualityReplicationError("replication invocation history is invalid")
    invocations.append(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "qualification": qualification,
            "runtime_identity": runtime_identity,
        }
    )
    quality._atomic_write_json(output / "run-state.json", state)


def _prepare_output(
    *,
    recipe_path: Path,
    recipe_bytes: bytes,
    recipe: AETQualityReplicationRecipe,
    base_recipe_path: Path,
    base_recipe_bytes: bytes,
    base_recipe: AETQualityRecipe,
    root: Path,
    runtime_identity: dict[str, Any],
    resume: bool,
    external_archive: Path | None,
    external_evidence: DiscoveryEvidence | None,
) -> tuple[Path, dict[str, Any], DiscoveryEvidence]:
    recipe_sha256 = quality._sha256_bytes(recipe_bytes)
    base_recipe_file_sha256 = quality._sha256_bytes(base_recipe_bytes)
    base_recipe_semantic_sha256 = _yaml_semantic_sha256(base_recipe_bytes)
    if base_recipe_semantic_sha256 != _expected_base_semantic_sha256(recipe):
        raise QualityReplicationQualificationError(
            "live base recipe does not match the frozen semantic recipe hash"
        )
    uv_lock_path = root / "uv.lock"
    try:
        uv_lock_bytes = uv_lock_path.read_bytes()
    except OSError as exc:
        raise QualityReplicationQualificationError("uv.lock is unavailable") from exc
    uv_lock_sha256 = quality._sha256_bytes(uv_lock_bytes)
    source_snapshot = quality._source_snapshot(root)
    git = quality._git_snapshot(root)
    output = quality._safe_output_target(root, str(_attribute(recipe, "output_root")))
    frozen_recipe = output / "recipe.yaml"
    frozen_base_recipe = output / "base-recipe.yaml"
    frozen_lock = output / "environment" / "uv.lock"
    archive_relative = "provenance/discovery/windows-a4500-quality-pilot.zip"
    internal_archive = output / archive_relative
    receipt_path = output / "provenance" / "discovery" / "import-receipt.json"

    if output.exists():
        if not resume:
            raise QualityReplicationError(f"output already exists; use --resume: {output}")
        state = quality._load_json(output / "run-state.json")
        expected_state = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha256,
            "base_recipe_file_sha256": base_recipe_file_sha256,
            "base_recipe_semantic_sha256": base_recipe_semantic_sha256,
            "uv_lock_sha256": uv_lock_sha256,
            "classification": _classification(),
            "discovery_archive_sha256": _expected_discovery(recipe)["outer_sha256"],
            "fresh_training_seeds": list(_fresh_seeds(recipe)),
            "discovery_seed": 1,
            "discovery_seed_evaluated": False,
            "discovery_seed_included_in_aggregation": False,
        }
        if any(
            not _strict_json_equal(state.get(key), value) for key, value in expected_state.items()
        ):
            raise QualityReplicationQualificationError(
                "existing replication was created from another recipe, base recipe, or lockfile"
            )
        if not _strict_json_equal(state.get("runtime_identity"), runtime_identity):
            raise QualityReplicationQualificationError(
                "runtime or GPU identity changed since replication initialization"
            )
        if state.get("source_snapshot", {}).get("sha256") != source_snapshot["sha256"]:
            raise QualityReplicationQualificationError(
                "replication runtime source fingerprint changed since initialization"
            )
        if state.get("git_sha") != git["sha"]:
            raise QualityReplicationQualificationError(
                "replication Git commit changed since initialization"
            )
        frozen_expected = (
            (frozen_recipe, recipe_sha256),
            (frozen_base_recipe, base_recipe_file_sha256),
            (frozen_lock, uv_lock_sha256),
            (internal_archive, _expected_discovery(recipe)["outer_sha256"]),
        )
        for path, expected_sha256 in frozen_expected:
            if not quality._file_matches_sha256(path, expected_sha256):
                raise QualityReplicationQualificationError(
                    f"frozen replication input changed or disappeared: {path}"
                )
        evidence = _inspect_discovery_archive(internal_archive, recipe, base_recipe)
        if quality._load_json(receipt_path) != _discovery_receipt(
            evidence, archive_relative_path=archive_relative
        ):
            raise QualityReplicationQualificationError("discovery import receipt changed")
        if state.get("status") not in COMPLETE_STATUSES:
            quality._record_partial_cleanup(output, state)
        return output, state, evidence

    if external_archive is None or external_evidence is None:
        raise QualityReplicationQualificationError(
            "initial replication requires the sealed discovery archive"
        )
    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "initialized",
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe_sha256,
        "base_recipe_file_sha256": base_recipe_file_sha256,
        "base_recipe_semantic_sha256": base_recipe_semantic_sha256,
        "uv_lock_sha256": uv_lock_sha256,
        "git_sha": git["sha"],
        "git": git,
        "source_snapshot": source_snapshot,
        "runtime_identity": runtime_identity,
        "classification": _classification(),
        "discovery_archive_sha256": external_evidence.outer_sha256,
        "fresh_training_seeds": list(_fresh_seeds(recipe)),
        "discovery_seed": 1,
        "discovery_seed_evaluated": False,
        "discovery_seed_included_in_aggregation": False,
        "invocations": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "environment").mkdir()
        (staging / "provenance" / "discovery").mkdir(parents=True)
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "base-recipe.yaml").write_bytes(base_recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(uv_lock_bytes)
        staging_archive = staging / archive_relative
        _atomic_copy_verified(external_archive, staging_archive, external_evidence.outer_sha256)
        copied_evidence = _inspect_discovery_archive(staging_archive, recipe, base_recipe)
        if copied_evidence != external_evidence:
            raise QualityReplicationQualificationError(
                "discovery evidence changed after atomic import"
            )
        quality._atomic_write_json(
            staging / "provenance" / "discovery" / "import-receipt.json",
            _discovery_receipt(copied_evidence, archive_relative_path=archive_relative),
        )
        quality._atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state, external_evidence


def _fresh_seeds(recipe: AETQualityReplicationRecipe) -> tuple[int, ...]:
    raw = _attribute(recipe, "fresh_seeds", "training_seeds", "seeds")
    return tuple(int(seed) for seed in raw)


def _fixed_epoch(recipe: AETQualityReplicationRecipe) -> int:
    return int(_attribute(recipe, "fixed_epoch", "checkpoint_epoch", "evaluation_epoch"))


def _development_spec(recipe: AETQualityReplicationRecipe) -> DatasetSplit:
    dataset = _attribute(recipe, "dataset")
    return _attribute(dataset, "development", "dev", "replication")


def _mode_ids(recipe: AETQualityReplicationRecipe) -> tuple[str, ...]:
    evaluation = _attribute(recipe, "evaluation")
    modes = _attribute(evaluation, "modes", "mode_ids")
    return tuple(mode.mode_id if hasattr(mode, "mode_id") else str(mode) for mode in modes)


def _role_mode_id(recipe: AETQualityReplicationRecipe, role: str) -> str:
    evaluation = _attribute(recipe, "evaluation")
    direct_names = {
        "primary": ("primary_mode_id", "primary"),
        "secondary": ("secondary_mode_id", "secondary"),
        "descriptive": (
            "descriptive_mode_id",
            "diagnostic_mode_id",
            "descriptive",
            "greedy_mode_id",
        ),
    }
    try:
        return str(_attribute(evaluation, *direct_names[role]))
    except QualityReplicationQualificationError:
        roles = _attribute(evaluation, "roles")
        return str(_attribute(roles, role))


def _bootstrap_settings(recipe: AETQualityReplicationRecipe) -> tuple[int, int, float, float]:
    bootstrap = _attribute(recipe, "bootstrap", "statistics")
    replicates = int(_attribute(bootstrap, "replicates", "bootstrap_replicates"))
    seed = int(_attribute(bootstrap, "seed", "bootstrap_seed"))
    quantile = float(_attribute(bootstrap, "quantile", "one_sided_quantile", "confidence_level"))
    t_critical = float(
        _attribute(
            bootstrap,
            "t_critical",
            "t_critical_df4",
            "seed_t_critical_value",
        )
    )
    return replicates, seed, quantile, t_critical


def _gate_threshold(recipe: AETQualityReplicationRecipe) -> float:
    gate = _attribute(recipe, "quality_gate", "gate")
    return float(_attribute(gate, "maximum_mean_gap_pct", "threshold_pct"))


def _maximum_invalid(recipe: AETQualityReplicationRecipe) -> int:
    gate = _attribute(recipe, "quality_gate", "gate")
    return int(_attribute(gate, "maximum_invalid_instances", "max_invalid_instances"))


def _validate_base_compatibility(
    recipe: AETQualityReplicationRecipe, base_recipe: AETQualityRecipe
) -> None:
    dataset_fields = ("problem", "size", "capacity", "max_demand")
    if any(
        getattr(recipe.dataset, field) != getattr(base_recipe.dataset, field)
        for field in dataset_fields
    ):
        raise QualityReplicationQualificationError(
            "replication dataset generator differs from the base recipe"
        )
    if recipe.model != base_recipe.model:
        raise QualityReplicationQualificationError("replication model differs from the base recipe")
    training = recipe.training
    base_training = base_recipe.training
    for field in (
        "algorithm",
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
    ):
        if getattr(training, field) != getattr(base_training, field):
            raise QualityReplicationQualificationError(
                f"replication training.{field} differs from the base recipe"
            )
    if (
        training.fixed_epoch != training.epochs
        or training.max_newly_completed_seeds_per_invocation != 1
    ):
        raise QualityReplicationQualificationError(
            "replication must evaluate the fixed final epoch and complete one seed per invocation"
        )
    base_modes = {mode.mode_id: mode for mode in base_recipe.evaluation.modes}
    for mode in recipe.evaluation.modes:
        if base_modes.get(mode.mode_id) != mode:
            raise QualityReplicationQualificationError(
                f"replication evaluation mode differs from base recipe: {mode.mode_id}"
            )


def _prepare_development_corpus(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    output: Path,
) -> quality.Corpus:
    spec = _development_spec(recipe)
    path = quality._relative(output, spec.artifact)
    if path.exists():
        corpus = quality._load_corpus(base_recipe, "development", spec, path)
    else:
        corpus = quality._generate_corpus(base_recipe, "development", spec, path)
    if int(corpus.coords.shape[0]) != spec.num_instances or corpus.split != "development":
        raise QualityReplicationError("development corpus metadata is inconsistent")
    return corpus


def _reference_lock_payload(
    recipe: AETQualityReplicationRecipe,
    corpus: quality.Corpus,
    reference_path: Path,
) -> dict[str, Any]:
    reference = quality._load_json(reference_path)
    candidates = reference.get("candidate_artifacts")
    if not isinstance(candidates, list):
        raise QualityReplicationError("replication reference candidate inventory is invalid")
    return {
        "schema_version": REFERENCE_LOCK_SCHEMA,
        "status": "locked",
        "locked_before_fresh_seed_training_and_evaluation": True,
        "policy": recipe.reference.policy,
        "base_recipe_semantic_sha256": _expected_base_semantic_sha256(recipe),
        "entry": {
            "path": "reference/development/reference.json",
            "sha256": quality._sha256_file(reference_path),
            "dataset_content_sha256": corpus.content_sha256,
        },
        "candidate_entries": candidates,
    }


def _load_anchored_reference(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_path = output / "reference" / "reference-lock.json"
    anchored_sha256 = state.get("reference_lock_sha256")
    if not isinstance(anchored_sha256, str) or not quality._file_matches_sha256(
        lock_path, anchored_sha256
    ):
        raise QualityReplicationError("anchored replication reference lock changed")
    lock = quality._load_json(lock_path)
    entry = lock.get("entry")
    reference_path = output / "reference" / "development" / "reference.json"
    if (
        lock.get("schema_version") != REFERENCE_LOCK_SCHEMA
        or lock.get("status") != "locked"
        or lock.get("locked_before_fresh_seed_training_and_evaluation") is not True
        or lock.get("policy") != recipe.reference.policy
        or lock.get("base_recipe_semantic_sha256") != _expected_base_semantic_sha256(recipe)
        or not isinstance(entry, dict)
        or entry.get("path") != "reference/development/reference.json"
        or entry.get("dataset_content_sha256") != corpus.content_sha256
        or not quality._file_matches_sha256(reference_path, entry.get("sha256", ""))
    ):
        raise QualityReplicationError("anchored replication reference metadata changed")
    reference = quality._load_json(reference_path)
    candidate_artifacts = reference.get("candidate_artifacts")
    if (
        not isinstance(candidate_artifacts, list)
        or lock.get("candidate_entries") != candidate_artifacts
    ):
        raise QualityReplicationError("replication reference candidate inventory is invalid")
    for candidate in candidate_artifacts:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
            raise QualityReplicationError("replication reference candidate entry is invalid")
        path = quality._relative(output, candidate["path"])
        if not quality._file_matches_sha256(path, candidate.get("sha256", "")):
            raise QualityReplicationError("anchored replication reference candidate changed")
    reference_recipe = replace(base_recipe, reference=recipe.reference)
    rebuilt = quality._build_reference(reference_recipe, corpus, output)
    if rebuilt != reference or quality._sha256_file(reference_path) != entry["sha256"]:
        raise QualityReplicationError("anchored replication reference semantics changed")
    return reference, lock


def _prepare_development_reference(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if state.get("reference_lock_sha256") is not None:
        return _load_anchored_reference(recipe, base_recipe, output, state, corpus)
    if (output / "seeds").exists() or (output / "evaluation").exists():
        raise QualityReplicationError(
            "fresh neural artifacts exist without a pre-training reference anchor"
        )
    reference_recipe = replace(base_recipe, reference=recipe.reference)
    reference = quality._build_reference(reference_recipe, corpus, output)
    reference_path = output / "reference" / "development" / "reference.json"
    lock_path = output / "reference" / "reference-lock.json"
    expected = _reference_lock_payload(recipe, corpus, reference_path)
    if lock_path.exists():
        lock = quality._load_json(lock_path)
        comparable = dict(lock)
        comparable.pop("locked_at", None)
        if comparable != expected:
            raise QualityReplicationError("unanchored replication reference lock changed")
    else:
        lock = {**expected, "locked_at": datetime.now(UTC).isoformat()}
        quality._atomic_write_json(lock_path, lock)
    state["reference_lock_sha256"] = quality._sha256_file(lock_path)
    state["reference_locked_before_fresh_seed_training_and_evaluation"] = True
    quality._atomic_write_json(output / "run-state.json", state)
    return reference, lock


def _seed_recipe(
    base_recipe: AETQualityRecipe,
    recipe: AETQualityReplicationRecipe,
    seed: int,
) -> AETQualityRecipe:
    training = replace(
        base_recipe.training,
        seed=seed,
    )
    return replace(base_recipe, training=training)


def _mode_objects(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
) -> tuple[EvaluationMode, ...]:
    by_id = {mode.mode_id: mode for mode in base_recipe.evaluation.modes}
    mode_ids = _mode_ids(recipe)
    try:
        return tuple(by_id[mode_id] for mode_id in mode_ids)
    except KeyError as exc:
        raise QualityReplicationQualificationError(
            f"replication mode is absent from the base recipe: {exc.args[0]}"
        ) from exc


def _expected_seed_progress(seed_recipe: AETQualityRecipe) -> tuple[int, int]:
    epochs = seed_recipe.training.epochs
    items = epochs * seed_recipe.training.instances_per_epoch
    steps = epochs * math.ceil(
        seed_recipe.training.instances_per_epoch / seed_recipe.training.batch_size
    )
    return items, steps


def _current_model_identity(seed_recipe: AETQualityRecipe) -> dict[str, Any]:
    from neuro_co.core.factory import make_env

    env = make_env(
        seed_recipe.dataset.problem,
        size=seed_recipe.dataset.size,
        capacity=seed_recipe.dataset.capacity,
        max_demand=seed_recipe.dataset.max_demand,
    )
    model = quality._make_mlco_am(seed_recipe, env)
    return quality._model_identity(seed_recipe, model)


def _validate_training_result(
    recipe: AETQualityReplicationRecipe,
    seed_recipe: AETQualityRecipe,
    output: Path,
    state: dict[str, Any],
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed_dir = output / "seeds" / f"seed-{seed:03d}"
    result_path = seed_dir / "training" / "result.json"
    latest_path = seed_dir / "training" / "latest.pt"
    checkpoint_path = seed_dir / "training" / "checkpoints" / f"epoch-{_fixed_epoch(recipe):03d}.pt"
    result = quality._load_json(result_path)
    expected_items, expected_steps = _expected_seed_progress(seed_recipe)
    records = result.get("checkpoint_records")
    if (
        result.get("status") != "complete"
        or result.get("seed") != seed
        or result.get("epochs") != _fixed_epoch(recipe)
        or result.get("items_processed") != expected_items
        or result.get("optimizer_steps") != expected_steps
        or result.get("energy_measurement") != "none"
        or result.get("aet_eligible") is not False
        or not isinstance(records, list)
        or len(records) != len(seed_recipe.training.checkpoint_epochs)
    ):
        raise QualityReplicationError(f"completed training metadata changed for seed {seed}")
    records_by_epoch: dict[int, dict[str, Any]] = {}
    steps_per_epoch = math.ceil(
        seed_recipe.training.instances_per_epoch / seed_recipe.training.batch_size
    )
    for record in records:
        if not isinstance(record, dict) or isinstance(record.get("epoch"), bool):
            raise QualityReplicationError(f"checkpoint record changed for seed {seed}")
        epoch = record.get("epoch")
        if not isinstance(epoch, int) or epoch in records_by_epoch:
            raise QualityReplicationError(f"checkpoint epoch inventory changed for seed {seed}")
        path = seed_dir / "training" / "checkpoints" / f"epoch-{epoch:03d}.pt"
        if (
            record.get("path") != f"training/checkpoints/epoch-{epoch:03d}.pt"
            or record.get("items_processed") != epoch * seed_recipe.training.instances_per_epoch
            or record.get("optimizer_steps") != epoch * steps_per_epoch
            or not quality._file_matches_sha256(path, record.get("sha256", ""))
        ):
            raise QualityReplicationError(f"checkpoint record changed for seed {seed}")
        records_by_epoch[epoch] = record
    if set(records_by_epoch) != set(seed_recipe.training.checkpoint_epochs):
        raise QualityReplicationError(f"checkpoint epochs changed for seed {seed}")
    record = records_by_epoch[_fixed_epoch(recipe)]
    checkpoint_files = {
        path.name for path in (seed_dir / "training" / "checkpoints").glob("*.pt") if path.is_file()
    }
    expected_checkpoint_files = {
        f"epoch-{epoch:03d}.pt" for epoch in seed_recipe.training.checkpoint_epochs
    }
    if checkpoint_files != expected_checkpoint_files:
        raise QualityReplicationError(f"unexpected checkpoint inventory for seed {seed}")
    if not latest_path.is_file():
        raise QualityReplicationError(f"latest checkpoint is missing for seed {seed}")
    model_identity = _current_model_identity(seed_recipe)
    if result.get("model_identity") != model_identity:
        raise QualityReplicationError(f"model identity changed for seed {seed}")
    paths_to_validate = [
        (
            f"epoch-{epoch}",
            seed_dir / "training" / "checkpoints" / f"epoch-{epoch:03d}.pt",
            epoch,
        )
        for epoch in seed_recipe.training.checkpoint_epochs
    ]
    paths_to_validate.append(("latest", latest_path, _fixed_epoch(recipe)))
    for label, path, expected_epoch in paths_to_validate:
        payload = quality._load_torch_mapping(path)
        quality._validate_checkpoint_metadata(payload, seed_recipe, state, model_identity)
        if (
            payload.get("completed_epochs") != expected_epoch
            or payload.get("items_processed")
            != expected_epoch * seed_recipe.training.instances_per_epoch
            or payload.get("optimizer_steps") != expected_epoch * steps_per_epoch
        ):
            raise QualityReplicationError(
                f"{label} checkpoint progress metadata changed for seed {seed}"
            )
    return result, {
        "epoch": _fixed_epoch(recipe),
        "path": checkpoint_path.relative_to(output).as_posix(),
        "sha256": record["sha256"],
        "items_processed": expected_items,
        "optimizer_steps": expected_steps,
    }


def _evaluation_seed(seed: int, mode_index: int) -> int:
    return 80_000 + seed * 1_000 + mode_index * 100


def _validate_cached_evaluation(
    result: dict[str, Any],
    *,
    recipe_sha256: str,
    seed: int,
    checkpoint_record: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    mode: EvaluationMode,
    eval_seed: int,
    label: str,
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "classification",
        "training_seed",
        "discovery_seed",
        "included_in_aggregation",
        "split",
        "epoch",
        "checkpoint_path",
        "checkpoint_sha256",
        "dataset_content_sha256",
        "reference_sha256",
        "mode",
        "evaluation_seed",
        "elapsed_s",
        "routes",
        "validation",
        "quality",
        "recipe_sha256",
    }
    expected_quality_keys = {
        "mean_cost",
        "reference_mean_cost",
        "mean_gap_pct",
        "median_gap_pct",
        "p95_gap_pct",
        "maximum_gap_pct",
        "minimum_gap_pct",
        "gaps_pct",
    }
    expected_validation_keys = {
        "complete",
        "validator",
        "instance_count",
        "validated_instance_count",
        "invalid_instance_count",
        "failure_count",
        "costs",
        "mean_cost",
        "violations",
    }
    elapsed = result.get("elapsed_s")
    if (
        set(result) != expected_keys
        or not isinstance(result.get("quality"), dict)
        or set(result["quality"]) != expected_quality_keys
        or not isinstance(result.get("validation"), dict)
        or set(result["validation"]) != expected_validation_keys
        or isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise QualityReplicationError(f"cached replication evaluation schema changed: {label}")
    expected = {
        "schema_version": EVALUATION_SCHEMA,
        "classification": _classification(),
        "training_seed": seed,
        "discovery_seed": False,
        "included_in_aggregation": True,
        "split": "development",
        "epoch": checkpoint_record["epoch"],
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "mode": asdict(mode),
        "evaluation_seed": eval_seed,
        "recipe_sha256": recipe_sha256,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise QualityReplicationError(f"cached replication evaluation changed: {label}")
    validation = quality._revalidate_stored_routes(result, corpus, label, require_complete=False)
    quality._revalidate_quality_metrics(result, reference, label)
    if validation.get("instance_count") != int(corpus.coords.shape[0]):
        raise QualityReplicationError(f"cached evaluation has wrong row count: {label}")
    return result


def _evaluate_seed_mode(
    recipe: AETQualityReplicationRecipe,
    seed_recipe: AETQualityRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    seed: int,
    checkpoint_record: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    mode: EvaluationMode,
    mode_index: int,
) -> dict[str, Any]:
    result_path = output / "evaluation" / mode.mode_id / f"seed-{seed:03d}.json"
    eval_seed = _evaluation_seed(seed, mode_index)
    if result_path.exists():
        return _validate_cached_evaluation(
            quality._load_json(result_path),
            recipe_sha256=state["recipe_sha256"],
            seed=seed,
            checkpoint_record=checkpoint_record,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            mode=mode,
            eval_seed=eval_seed,
            label=str(result_path),
        )
    checkpoint_path = quality._relative(output, checkpoint_record["path"])
    if not quality._file_matches_sha256(checkpoint_path, checkpoint_record["sha256"]):
        raise QualityReplicationError(f"checkpoint changed before seed {seed} evaluation")
    started = time.perf_counter()
    routes = quality._evaluate_routes(
        seed_recipe,
        checkpoint_path,
        corpus,
        mode,
        eval_seed=eval_seed,
    )
    elapsed_s = time.perf_counter() - started
    validation = quality.validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        routes,
    )
    gap = quality._gap_summary(validation["costs"], reference["costs"])
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "classification": _classification(),
        "training_seed": seed,
        "discovery_seed": False,
        "included_in_aggregation": True,
        "split": "development",
        "epoch": checkpoint_record["epoch"],
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "mode": asdict(mode),
        "evaluation_seed": eval_seed,
        "elapsed_s": elapsed_s,
        "routes": routes,
        "validation": validation,
        "quality": gap,
        "recipe_sha256": state["recipe_sha256"],
    }
    quality._atomic_write_json(result_path, result)
    return result


def _seed_result_payload(
    *,
    output: Path,
    seed: int,
    training_result: dict[str, Any],
    checkpoint_record: dict[str, Any],
    evaluations: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    seed_dir = output / "seeds" / f"seed-{seed:03d}"
    training_path = seed_dir / "training" / "result.json"
    latest_path = seed_dir / "training" / "latest.pt"
    return {
        "schema_version": SEED_RESULT_SCHEMA,
        "status": "complete",
        "training_seed": seed,
        "discovery_seed": False,
        "included_in_aggregation": True,
        "training_result": training_result,
        "training_result_path": training_path.relative_to(output).as_posix(),
        "training_result_sha256": quality._sha256_file(training_path),
        "latest_checkpoint_path": latest_path.relative_to(output).as_posix(),
        "latest_checkpoint_sha256": quality._sha256_file(latest_path),
        "evaluated_checkpoint": checkpoint_record,
        "evaluations": {
            mode_id: {
                "path": (Path("evaluation") / mode_id / f"seed-{seed:03d}.json").as_posix(),
                "sha256": quality._sha256_file(
                    output / "evaluation" / mode_id / f"seed-{seed:03d}.json"
                ),
                "invalid_instances": result["validation"]["invalid_instance_count"],
                "mean_gap_pct": result["quality"]["mean_gap_pct"],
            }
            for mode_id, result in evaluations.items()
        },
        "only_epoch_40_evaluated": checkpoint_record["epoch"] == 40,
    }


def _load_completed_seed(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    seed: int,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if seed == 1 or seed not in _fresh_seeds(recipe):
        raise QualityReplicationError(f"non-fresh seed entered replication results: {seed}")
    seed_recipe = _seed_recipe(base_recipe, recipe, seed)
    training_result, checkpoint_record = _validate_training_result(
        recipe, seed_recipe, output, state, seed
    )
    modes = _mode_objects(recipe, base_recipe)
    evaluations: dict[str, dict[str, Any]] = {}
    for mode_index, mode in enumerate(modes):
        path = output / "evaluation" / mode.mode_id / f"seed-{seed:03d}.json"
        if not path.is_file():
            raise QualityReplicationError(
                f"completed seed {seed} is missing evaluation {mode.mode_id}"
            )
        evaluations[mode.mode_id] = _validate_cached_evaluation(
            quality._load_json(path),
            recipe_sha256=state["recipe_sha256"],
            seed=seed,
            checkpoint_record=checkpoint_record,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            mode=mode,
            eval_seed=_evaluation_seed(seed, mode_index),
            label=str(path),
        )
    seed_result_path = output / "seeds" / f"seed-{seed:03d}" / "seed-result.json"
    stored = quality._load_json(seed_result_path)
    expected = _seed_result_payload(
        output=output,
        seed=seed,
        training_result=training_result,
        checkpoint_record=checkpoint_record,
        evaluations=evaluations,
    )
    if stored != expected:
        raise QualityReplicationError(f"completed seed result changed for seed {seed}")
    return stored, evaluations


def _completed_seed_prefix(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, dict[str, Any]]]]:
    completed: dict[int, dict[str, Any]] = {}
    evaluations: dict[int, dict[str, dict[str, Any]]] = {}
    missing_seen = False
    for seed in _fresh_seeds(recipe):
        result_path = output / "seeds" / f"seed-{seed:03d}" / "seed-result.json"
        if not result_path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise QualityReplicationError(
                "completed fresh seeds are not a contiguous ordered prefix"
            )
        completed[seed], evaluations[seed] = _load_completed_seed(
            recipe,
            base_recipe,
            output=output,
            state=state,
            seed=seed,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
        )
    seed_one_artifacts = [
        path.relative_to(output).as_posix()
        for root in (output / "seeds", output / "evaluation")
        if root.exists()
        for path in root.rglob("*seed-001*")
    ]
    if seed_one_artifacts:
        raise QualityReplicationError(
            f"discovery seed 1 entered fresh inference artifacts: {seed_one_artifacts}"
        )
    return completed, evaluations


def _complete_one_seed(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    seed: int,
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    seed_recipe = _seed_recipe(base_recipe, recipe, seed)
    seed_dir = output / "seeds" / f"seed-{seed:03d}"
    try:
        quality._train(seed_recipe, seed_dir, state)
    except quality.QualityPilotError as exc:
        if str(exc).startswith("training walltime limit reached after a complete epoch"):
            state["status"] = INCOMPLETE_STATUS
            state["active_training_seed"] = seed
            state["resume_reason"] = "per-invocation training walltime reached"
            quality._atomic_write_json(output / "run-state.json", state)
            return None
        raise QualityReplicationError(f"training seed {seed} failed: {exc}") from exc
    training_result, checkpoint_record = _validate_training_result(
        recipe, seed_recipe, output, state, seed
    )
    evaluations: dict[str, dict[str, Any]] = {}
    for mode_index, mode in enumerate(_mode_objects(recipe, base_recipe)):
        result = _evaluate_seed_mode(
            recipe,
            seed_recipe,
            output=output,
            state=state,
            seed=seed,
            checkpoint_record=checkpoint_record,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            mode=mode,
            mode_index=mode_index,
        )
        evaluations[mode.mode_id] = result
        print(
            json.dumps(
                {
                    "stage": "replication-evaluation",
                    "training_seed": seed,
                    "mode": mode.mode_id,
                    "mean_gap_pct": result["quality"]["mean_gap_pct"],
                    "invalid_instances": result["validation"]["invalid_instance_count"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    payload = _seed_result_payload(
        output=output,
        seed=seed,
        training_result=training_result,
        checkpoint_record=checkpoint_record,
        evaluations=evaluations,
    )
    quality._atomic_write_json(seed_dir / "seed-result.json", payload)
    validated, validated_evaluations = _load_completed_seed(
        recipe,
        base_recipe,
        output=output,
        state=state,
        seed=seed,
        corpus=corpus,
        reference=reference,
        reference_sha256=reference_sha256,
    )
    return validated, validated_evaluations


def _build_assessment(
    recipe: AETQualityReplicationRecipe,
    evaluations: Mapping[int, Mapping[str, dict[str, Any]]],
) -> dict[str, Any]:
    import numpy as np

    seeds = _fresh_seeds(recipe)
    mode_ids = _mode_ids(recipe)
    if tuple(sorted(evaluations)) != seeds or 1 in evaluations:
        raise QualityReplicationError(
            "assessment seed inventory must be exactly fresh seeds 2 to 6"
        )
    expected_instances = _development_spec(recipe).num_instances
    matrices: dict[str, Any] = {}
    invalid_counts: dict[str, int] = {}
    evaluation_inventory: dict[str, list[dict[str, Any]]] = {}
    for mode_id in mode_ids:
        rows: list[list[float]] = []
        invalid = 0
        entries: list[dict[str, Any]] = []
        for seed in seeds:
            try:
                result = evaluations[seed][mode_id]
            except KeyError as exc:
                raise QualityReplicationError(
                    f"assessment lacks {mode_id} evaluation for seed {seed}"
                ) from exc
            gaps = result.get("quality", {}).get("gaps_pct")
            if not isinstance(gaps, list) or len(gaps) != expected_instances:
                raise QualityReplicationError(
                    f"assessment gap vector has wrong size for {mode_id}, seed {seed}"
                )
            rows.append([float(value) for value in gaps])
            invalid += int(result["validation"]["invalid_instance_count"])
            entries.append(
                {
                    "training_seed": seed,
                    "mean_gap_pct": result["quality"]["mean_gap_pct"],
                    "invalid_instances": result["validation"]["invalid_instance_count"],
                }
            )
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.shape != (len(seeds), expected_instances):
            raise QualityReplicationError(f"assessment matrix has wrong shape for {mode_id}")
        matrices[mode_id] = matrix
        invalid_counts[mode_id] = invalid
        evaluation_inventory[mode_id] = entries

    replicates, bootstrap_seed, quantile, t_critical = _bootstrap_settings(recipe)
    summaries = _summarize_gap_matrices(
        matrices,
        invalid_instances=invalid_counts,
        threshold_pct=_gate_threshold(recipe),
        bootstrap_replicates=replicates,
        bootstrap_seed=bootstrap_seed,
        bootstrap_quantile=quantile,
        t_critical=t_critical,
    )
    primary = _role_mode_id(recipe, "primary")
    secondary = _role_mode_id(recipe, "secondary")
    descriptive = _role_mode_id(recipe, "descriptive")
    if set((primary, secondary, descriptive)) != set(mode_ids):
        raise QualityReplicationError("replication role inventory disagrees with mode inventory")
    primary_classification = summaries[primary]["classification"]
    primary_passed = primary_classification == "pass"
    status = "complete_replication_passed" if primary_passed else "complete_replication_nonpass"
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": status,
        "classification": _classification(),
        "fresh_training_seeds": list(seeds),
        "discovery_training_seed": 1,
        "discovery_seed_evaluated": False,
        "discovery_seed_included_in_aggregation": False,
        "common_development_instances": expected_instances,
        "mode_order": list(mode_ids),
        "mode_statistics": summaries,
        "evaluation_inventory": evaluation_inventory,
        "method": {
            "unit_of_replication": "independent_training_seed",
            "per_seed_statistic": "mean of per-instance percentage gaps",
            "t_ucb": "one-sided Student t upper confidence bound across five seed means",
            "t_critical_one_sided_95_df4": t_critical,
            "crossed_bootstrap": {
                "generator": "numpy.random.Generator(PCG64)",
                "seed": bootstrap_seed,
                "replicates": replicates,
                "one_sided_quantile": quantile,
                "resampled_axes": ["training_seed", "instance"],
                "shared_resample_indices_across_modes": True,
            },
            "strict_threshold": True,
            "threshold_operator": "<",
            "maximum_mean_gap_pct": _gate_threshold(recipe),
        },
        "roles": {
            "primary": {
                "mode_id": primary,
                "classification": primary_classification,
                "passed": primary_passed,
                "alone_decides_final_status": True,
            },
            "secondary": {
                "mode_id": secondary,
                "classification": summaries[secondary]["classification"],
                "inferentially_interpreted": primary_passed,
                "interpretation": (
                    "inferential_secondary" if primary_passed else "not_interpreted_primary_nonpass"
                ),
                "can_change_final_status": False,
            },
            "descriptive": {
                "mode_id": descriptive,
                "classification": summaries[descriptive]["classification"],
                "interpretation": "descriptive_only",
                "can_change_final_status": False,
            },
        },
        "primary_gate": {
            "mode_id": primary,
            "all_routes_valid": summaries[primary]["invalid_instances"] <= _maximum_invalid(recipe),
            "all_values_finite": summaries[primary]["finite"],
            "t_ucb_strictly_below_threshold": (
                summaries[primary]["t_ucb_95_pct"] is not None
                and summaries[primary]["t_ucb_95_pct"] < _gate_threshold(recipe)
            ),
            "bootstrap_ucb_strictly_below_threshold": (
                summaries[primary]["bootstrap_ucb_95_pct"] is not None
                and summaries[primary]["bootstrap_ucb_95_pct"] < _gate_threshold(recipe)
            ),
            "each_seed_mean_strictly_below_threshold": (
                summaries[primary]["maximum_seed_mean_gap_pct"] is not None
                and summaries[primary]["maximum_seed_mean_gap_pct"] < _gate_threshold(recipe)
            ),
            "classification": primary_classification,
            "passed": primary_passed,
        },
        "aet_was_computed": False,
        "energy_was_measured": False,
    }


def _expected_artifact_inventory(
    recipe: AETQualityReplicationRecipe,
) -> set[str]:
    expected = {
        "recipe.yaml",
        "base-recipe.yaml",
        "environment/uv.lock",
        "provenance/discovery/windows-a4500-quality-pilot.zip",
        "provenance/discovery/import-receipt.json",
        _development_spec(recipe).artifact,
        str(Path(_development_spec(recipe).artifact).with_suffix(".manifest.json")).replace(
            "\\", "/"
        ),
        "reference/reference-lock.json",
        "reference/development/reference.json",
        "replication-assessment.json",
        "manifest.json",
        "SHA256SUMS",
        "run-state.json",
    }
    for seed in recipe.reference.hgs.seeds:
        expected.add(f"reference/development/candidates/pyvrp-hgs-seed{seed}.json")
    expected.add(
        "reference/development/candidates/"
        f"ortools-routing-gls-seed{recipe.reference.ortools.seed}.json"
    )
    for seed in _fresh_seeds(recipe):
        prefix = f"seeds/seed-{seed:03d}/training"
        expected.update(
            {
                f"{prefix}/latest.pt",
                f"{prefix}/result.json",
                f"seeds/seed-{seed:03d}/seed-result.json",
            }
        )
        expected.update(
            f"{prefix}/checkpoints/epoch-{epoch:03d}.pt"
            for epoch in recipe.training.checkpoint_epochs
        )
        expected.update(
            f"evaluation/{mode_id}/seed-{seed:03d}.json" for mode_id in _mode_ids(recipe)
        )
    return expected


def _assert_exact_artifact_inventory(recipe: AETQualityReplicationRecipe, output: Path) -> None:
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    expected = _expected_artifact_inventory(recipe)
    if actual != expected:
        raise QualityReplicationError(
            "replication artifact inventory changed; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _manifest_payload(
    recipe: AETQualityReplicationRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    qualification: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_lock: dict[str, Any],
    seed_results: Mapping[int, dict[str, Any]],
    evaluations: Mapping[int, Mapping[str, dict[str, Any]]],
    assessment: dict[str, Any],
    started_at: str,
    ended_at: str,
) -> dict[str, Any]:
    receipt_path = output / "provenance" / "discovery" / "import-receipt.json"
    assessment_path = output / "replication-assessment.json"
    return {
        "schema_version": MANIFEST_SCHEMA,
        "status": assessment["status"],
        "classification": {**_classification(), "exclusive_access_required": False},
        "started_at_this_invocation": started_at,
        "ended_at": ended_at,
        "recipe": {
            "path": "recipe.yaml",
            "sha256": state["recipe_sha256"],
            "name": recipe.name,
        },
        "base_recipe": {
            "path": "base-recipe.yaml",
            "file_sha256": state["base_recipe_file_sha256"],
            "semantic_sha256": state["base_recipe_semantic_sha256"],
        },
        "source": {
            "git_sha": state["git_sha"],
            "git_at_initialization": state["git"],
            "uv_lock_sha256": state["uv_lock_sha256"],
            "scoped_source_snapshot": state["source_snapshot"],
        },
        "runtime_controls": state["runtime_controls"],
        "execution_qualification": qualification,
        "execution_invocations": state["invocations"],
        "run_state_history": {
            "created_at": state["created_at"],
            "resume_cleanups": state.get("resume_cleanups", []),
        },
        "discovery_provenance": {
            "archive_path": "provenance/discovery/windows-a4500-quality-pilot.zip",
            "archive_sha256": state["discovery_archive_sha256"],
            "receipt_path": "provenance/discovery/import-receipt.json",
            "receipt_sha256": quality._sha256_file(receipt_path),
            "receipt": quality._load_json(receipt_path),
        },
        "dataset": {
            "split": "development",
            "path": corpus.path.relative_to(output).as_posix(),
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "instances": int(corpus.coords.shape[0]),
            "shared_across_all_fresh_seeds": True,
        },
        "reference_lock": {
            "path": "reference/reference-lock.json",
            "sha256": state["reference_lock_sha256"],
            "payload": reference_lock,
            "reference_sha256": quality._sha256_file(
                output / "reference" / "development" / "reference.json"
            ),
            "reference_mean_cost": reference["mean_cost"],
        },
        "fresh_seed_results": {
            str(seed): {
                "path": f"seeds/seed-{seed:03d}/seed-result.json",
                "sha256": quality._sha256_file(
                    output / "seeds" / f"seed-{seed:03d}" / "seed-result.json"
                ),
                "payload": seed_results[seed],
            }
            for seed in _fresh_seeds(recipe)
        },
        "evaluation_order": list(_mode_ids(recipe)),
        "evaluation_count": sum(len(values) for values in evaluations.values()),
        "checkpoint_selection_performed": False,
        "evaluated_checkpoint_epoch": _fixed_epoch(recipe),
        "assessment": {
            "path": "replication-assessment.json",
            "sha256": quality._sha256_file(assessment_path),
            "payload": assessment,
        },
        "seed_1": {
            "role": "provenance_only",
            "evaluated": False,
            "included_in_aggregation": False,
        },
        "aet_was_computed": False,
        "energy_was_measured": False,
    }


def _validate_completed_run_state(
    recipe: AETQualityReplicationRecipe,
    state: dict[str, Any],
) -> None:
    seeds = list(_fresh_seeds(recipe))
    expected_keys = {
        "schema_version",
        "status",
        "created_at",
        "recipe_sha256",
        "base_recipe_file_sha256",
        "base_recipe_semantic_sha256",
        "uv_lock_sha256",
        "git_sha",
        "git",
        "source_snapshot",
        "runtime_identity",
        "classification",
        "discovery_archive_sha256",
        "fresh_training_seeds",
        "discovery_seed",
        "discovery_seed_evaluated",
        "discovery_seed_included_in_aggregation",
        "invocations",
        "runtime_controls",
        "reference_lock_sha256",
        "reference_locked_before_fresh_seed_training_and_evaluation",
        "completed_at",
        "completed_seeds",
        "remaining_seeds",
        "active_training_seed",
        "manifest_sha256",
        "checksums_sha256",
        "resume_cleanups",
    }
    if set(state) != expected_keys:
        raise QualityReplicationError("completed replication run-state schema changed")
    expected = {
        "schema_version": RUN_STATE_SCHEMA,
        "classification": _classification(),
        "discovery_archive_sha256": _expected_discovery(recipe)["outer_sha256"],
        "fresh_training_seeds": seeds,
        "discovery_seed": 1,
        "discovery_seed_evaluated": False,
        "discovery_seed_included_in_aggregation": False,
        "reference_locked_before_fresh_seed_training_and_evaluation": True,
        "completed_seeds": seeds,
        "remaining_seeds": [],
        "active_training_seed": None,
    }
    if any(not _strict_json_equal(state.get(key), value) for key, value in expected.items()):
        raise QualityReplicationError("completed replication run-state invariants changed")
    if state.get("status") not in COMPLETE_STATUSES:
        raise QualityReplicationError("replication run-state is not complete")
    for key in ("reference_lock_sha256", "manifest_sha256", "checksums_sha256"):
        value = state.get(key)
        if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
            raise QualityReplicationError(f"completed replication run-state has an invalid {key}")
    for key in ("created_at", "completed_at"):
        value = state.get(key)
        if not isinstance(value, str) or not value:
            raise QualityReplicationError(f"completed replication has an invalid {key}")
    if not isinstance(state.get("resume_cleanups"), list):
        raise QualityReplicationError("completed replication cleanup history changed")
    controls = state.get("runtime_controls")
    if (
        not isinstance(controls, dict)
        or controls.get("energy_measurement") != "none"
        or controls.get("parallel_workloads_allowed") is not True
        or controls.get("exclusive_attestation_required") is not False
    ):
        raise QualityReplicationError("completed replication runtime controls changed")


def _validate_completed_bundle(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    *,
    output: Path,
    state: dict[str, Any],
    qualification: dict[str, Any] | None = None,
) -> ReplicationResult:
    _validate_completed_run_state(recipe, state)
    checksum_sha256 = state.get("checksums_sha256")
    if not isinstance(checksum_sha256, str):
        raise QualityReplicationError("completed replication lacks a checksum anchor")
    quality._verify_checksums(output, expected_sha256=checksum_sha256)
    _assert_exact_artifact_inventory(recipe, output)
    if quality._stale_partial_artifacts(output):
        raise QualityReplicationError("completed replication contains partial artifacts")

    internal_archive = output / "provenance" / "discovery" / "windows-a4500-quality-pilot.zip"
    evidence = _inspect_discovery_archive(internal_archive, recipe, base_recipe)
    receipt = quality._load_json(output / "provenance" / "discovery" / "import-receipt.json")
    if not _strict_json_equal(
        receipt,
        _discovery_receipt(
            evidence,
            archive_relative_path="provenance/discovery/windows-a4500-quality-pilot.zip",
        ),
    ):
        raise QualityReplicationError("completed replication discovery receipt changed")

    corpus = quality._load_corpus(
        base_recipe,
        "development",
        _development_spec(recipe),
        quality._relative(output, _development_spec(recipe).artifact),
    )
    reference, reference_lock = _load_anchored_reference(recipe, base_recipe, output, state, corpus)
    reference_sha256 = reference_lock["entry"]["sha256"]
    seed_results, evaluations = _completed_seed_prefix(
        recipe,
        base_recipe,
        output=output,
        state=state,
        corpus=corpus,
        reference=reference,
        reference_sha256=reference_sha256,
    )
    if tuple(seed_results) != _fresh_seeds(recipe):
        raise QualityReplicationError("completed replication lacks fresh seeds")
    assessment = quality._load_json(output / "replication-assessment.json")
    recomputed_assessment = _build_assessment(recipe, evaluations)
    if not _strict_json_equal(assessment, recomputed_assessment):
        raise QualityReplicationError("completed replication assessment changed")
    manifest_path = output / "manifest.json"
    manifest = quality._load_json(manifest_path)
    if quality._sha256_file(manifest_path) != state.get("manifest_sha256"):
        raise QualityReplicationError("completed replication manifest hash changed")
    invocations = state.get("invocations")
    if (
        not isinstance(invocations, list)
        or not invocations
        or not isinstance(invocations[-1], dict)
    ):
        raise QualityReplicationError("completed replication invocation history changed")
    final_qualification = invocations[-1].get("qualification")
    if not isinstance(final_qualification, dict):
        raise QualityReplicationError("completed replication qualification history changed")
    expected_manifest = _manifest_payload(
        recipe,
        output=output,
        state=state,
        qualification=final_qualification,
        corpus=corpus,
        reference=reference,
        reference_lock=reference_lock,
        seed_results=seed_results,
        evaluations=evaluations,
        assessment=assessment,
        started_at=str(manifest.get("started_at_this_invocation")),
        ended_at=str(manifest.get("ended_at")),
    )
    if (
        not _strict_json_equal(manifest, expected_manifest)
        or manifest.get("status") != state.get("status")
        or manifest.get("ended_at") != state.get("completed_at")
    ):
        raise QualityReplicationError("completed replication manifest relationships changed")
    if (
        not _strict_json_equal(manifest.get("execution_invocations"), invocations)
        or not _strict_json_equal(manifest.get("execution_qualification"), final_qualification)
        or not _strict_json_equal(
            invocations[-1].get("runtime_identity"), state.get("runtime_identity")
        )
    ):
        raise QualityReplicationError("completed replication invocation history changed")
    primary_classification = assessment["roles"]["primary"]["classification"]
    expected_status = (
        "complete_replication_passed"
        if primary_classification == "pass"
        else "complete_replication_nonpass"
    )
    if state["status"] != expected_status or assessment["status"] != expected_status:
        raise QualityReplicationError("completed replication primary gate status changed")
    return ReplicationResult(
        path=output,
        status=expected_status,
        complete=True,
        manifest_path=manifest_path,
        manifest_sha256=state["manifest_sha256"],
        primary_classification=primary_classification,
        completed_seeds=_fresh_seeds(recipe),
        remaining_seeds=(),
    )


def _revalidate_live_inputs(
    *,
    recipe_path: Path,
    recipe_bytes: bytes,
    base_recipe_path: Path,
    base_recipe_bytes: bytes,
    root: Path,
    output: Path,
    state: dict[str, Any],
) -> None:
    if (
        recipe_path.read_bytes() != recipe_bytes
        or not quality._file_matches_sha256(output / "recipe.yaml", state["recipe_sha256"])
        or base_recipe_path.read_bytes() != base_recipe_bytes
        or not quality._file_matches_sha256(
            output / "base-recipe.yaml", state["base_recipe_file_sha256"]
        )
        or not quality._file_matches_sha256(
            output / "environment" / "uv.lock", state["uv_lock_sha256"]
        )
        or not quality._file_matches_sha256(root / "uv.lock", state["uv_lock_sha256"])
    ):
        raise QualityReplicationError("replication frozen input changed during execution")
    if quality._source_snapshot(root)["sha256"] != state["source_snapshot"]["sha256"]:
        raise QualityReplicationError("replication runtime source changed during execution")
    if quality._git_snapshot(root)["sha"] != state["git_sha"]:
        raise QualityReplicationError("replication Git commit changed during execution")
    if _yaml_semantic_sha256(base_recipe_bytes) != state["base_recipe_semantic_sha256"]:
        raise QualityReplicationError("base recipe semantics changed during execution")


def _finalize(
    recipe: AETQualityReplicationRecipe,
    base_recipe: AETQualityRecipe,
    *,
    recipe_path: Path,
    recipe_bytes: bytes,
    base_recipe_path: Path,
    base_recipe_bytes: bytes,
    root: Path,
    output: Path,
    state: dict[str, Any],
    qualification: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_lock: dict[str, Any],
    seed_results: Mapping[int, dict[str, Any]],
    evaluations: Mapping[int, Mapping[str, dict[str, Any]]],
    started_at: datetime,
) -> ReplicationResult:
    assessment = _build_assessment(recipe, evaluations)
    assessment_path = output / "replication-assessment.json"
    quality._atomic_write_json(assessment_path, assessment)
    if quality._load_json(assessment_path) != _build_assessment(recipe, evaluations):
        raise QualityReplicationError("replication assessment changed after atomic write")
    _revalidate_live_inputs(
        recipe_path=recipe_path,
        recipe_bytes=recipe_bytes,
        base_recipe_path=base_recipe_path,
        base_recipe_bytes=base_recipe_bytes,
        root=root,
        output=output,
        state=state,
    )
    ended_at = datetime.now(UTC)
    manifest = _manifest_payload(
        recipe,
        output=output,
        state=state,
        qualification=qualification,
        corpus=corpus,
        reference=reference,
        reference_lock=reference_lock,
        seed_results=seed_results,
        evaluations=evaluations,
        assessment=assessment,
        started_at=started_at.isoformat(),
        ended_at=ended_at.isoformat(),
    )
    manifest_path = output / "manifest.json"
    quality._atomic_write_json(manifest_path, manifest)
    checksum_path = quality._write_checksums(output)
    candidate_state = {
        **state,
        "status": assessment["status"],
        "completed_at": ended_at.isoformat(),
        "completed_seeds": list(_fresh_seeds(recipe)),
        "remaining_seeds": [],
        "active_training_seed": None,
        "manifest_sha256": quality._sha256_file(manifest_path),
        "checksums_sha256": quality._sha256_file(checksum_path),
        "resume_cleanups": list(state.get("resume_cleanups", [])),
    }
    candidate_state.pop("resume_reason", None)
    result = _validate_completed_bundle(
        recipe,
        base_recipe,
        output=output,
        state=candidate_state,
    )
    quality._atomic_write_json(output / "run-state.json", candidate_state)
    state.clear()
    state.update(candidate_state)
    return result


def _hardware_qualification_for_resume(
    base_recipe: AETQualityRecipe,
) -> dict[str, Any]:
    qualification = quality.runtime_qualification(base_recipe)
    return {
        **qualification,
        "replication_input_source": "frozen_internal_discovery_archive",
    }


def execute_quality_replication(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    resume: bool = True,
) -> ReplicationResult:
    """Execute or resume the frozen multi-seed quality replication."""

    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
    except OSError as exc:
        raise QualityReplicationQualificationError(
            f"replication workspace or recipe is unavailable: {exc}"
        ) from exc
    if root != Path.cwd().resolve(strict=True):
        raise QualityReplicationQualificationError("workspace_root must be the current repository")
    try:
        source_recipe.relative_to(root)
    except ValueError as exc:
        raise QualityReplicationQualificationError(
            "replication recipe must be stored inside the repository"
        ) from exc
    recipe_bytes = source_recipe.read_bytes()
    try:
        recipe = load_aet_quality_replication_recipe(source_recipe)
    except Exception as exc:
        raise QualityReplicationQualificationError(f"replication recipe is invalid: {exc}") from exc
    if source_recipe.read_bytes() != recipe_bytes:
        raise QualityReplicationQualificationError(
            "replication recipe changed while it was being loaded"
        )
    base_recipe_path = _base_recipe_path(recipe, root)
    base_recipe_bytes = base_recipe_path.read_bytes()
    try:
        base_recipe = quality.load_aet_quality_recipe(base_recipe_path)
    except Exception as exc:
        raise QualityReplicationQualificationError(
            f"base quality recipe is invalid: {exc}"
        ) from exc
    if base_recipe_path.read_bytes() != base_recipe_bytes:
        raise QualityReplicationQualificationError("base recipe changed while it was being loaded")
    if _yaml_semantic_sha256(base_recipe_bytes) != _expected_base_semantic_sha256(recipe):
        raise QualityReplicationQualificationError("base recipe semantic hash changed")
    _validate_base_compatibility(recipe, base_recipe)

    output = quality._safe_output_target(root, recipe.output_root)
    output_exists = output.exists()
    external_archive: Path | None = None
    external_evidence: DiscoveryEvidence | None = None
    if output_exists:
        qualification = _hardware_qualification_for_resume(base_recipe)
    else:
        qualification = runtime_qualification(recipe, repository_root=root)
        external_archive = _discovery_archive_path(recipe, root)
        external_evidence = _inspect_discovery_archive(external_archive, recipe, base_recipe)
    if qualification.get("ready_to_execute") is not True:
        raise QualityReplicationQualificationError(
            "quality replication requires native Windows, CUDA, Git, PyVRP, and OR-Tools"
        )
    runtime_identity = quality._runtime_identity(base_recipe, qualification)
    with quality._output_lock(output):
        if source_recipe.read_bytes() != recipe_bytes:
            raise QualityReplicationError("replication recipe changed before initialization")
        if base_recipe_path.read_bytes() != base_recipe_bytes:
            raise QualityReplicationError("base recipe changed before initialization")
        output, state, _evidence = _prepare_output(
            recipe_path=source_recipe,
            recipe_bytes=recipe_bytes,
            recipe=recipe,
            base_recipe_path=base_recipe_path,
            base_recipe_bytes=base_recipe_bytes,
            base_recipe=base_recipe,
            root=root,
            runtime_identity=runtime_identity,
            resume=resume if output_exists else False,
            external_archive=external_archive,
            external_evidence=external_evidence,
        )
        if state.get("status") in COMPLETE_STATUSES:
            return _validate_completed_bundle(
                recipe,
                base_recipe,
                output=output,
                state=state,
            )

        runtime_controls = quality._configure_runtime(recipe.gpu_index)
        if "runtime_controls" in state and not _strict_json_equal(
            state["runtime_controls"], runtime_controls
        ):
            raise QualityReplicationQualificationError(
                "replication runtime controls changed between invocations"
            )
        state["runtime_controls"] = runtime_controls
        _record_invocation(output, state, qualification, runtime_identity)
        started_at = datetime.now(UTC)
        corpus = _prepare_development_corpus(recipe, base_recipe, output)
        reference, reference_lock = _prepare_development_reference(
            recipe, base_recipe, output, state, corpus
        )
        reference_sha256 = reference_lock["entry"]["sha256"]
        seed_results, evaluations = _completed_seed_prefix(
            recipe,
            base_recipe,
            output=output,
            state=state,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
        )
        remaining = tuple(seed for seed in _fresh_seeds(recipe) if seed not in seed_results)
        if remaining:
            seed = remaining[0]
            completed = _complete_one_seed(
                recipe,
                base_recipe,
                output=output,
                state=state,
                seed=seed,
                corpus=corpus,
                reference=reference,
                reference_sha256=reference_sha256,
            )
            if completed is None:
                return ReplicationResult(
                    path=output,
                    status=INCOMPLETE_STATUS,
                    complete=False,
                    completed_seeds=tuple(seed_results),
                    remaining_seeds=remaining,
                )
            seed_results[seed], evaluations[seed] = completed
            remaining = tuple(
                candidate for candidate in _fresh_seeds(recipe) if candidate not in seed_results
            )
            if remaining:
                state["status"] = INCOMPLETE_STATUS
                state["completed_seeds"] = list(seed_results)
                state["remaining_seeds"] = list(remaining)
                state["active_training_seed"] = None
                state["resume_reason"] = "one-new-seed-per-invocation boundary"
                quality._atomic_write_json(output / "run-state.json", state)
                return ReplicationResult(
                    path=output,
                    status=INCOMPLETE_STATUS,
                    complete=False,
                    completed_seeds=tuple(seed_results),
                    remaining_seeds=remaining,
                )
        return _finalize(
            recipe,
            base_recipe,
            recipe_path=source_recipe,
            recipe_bytes=recipe_bytes,
            base_recipe_path=base_recipe_path,
            base_recipe_bytes=base_recipe_bytes,
            root=root,
            output=output,
            state=state,
            qualification=qualification,
            corpus=corpus,
            reference=reference,
            reference_lock=reference_lock,
            seed_results=seed_results,
            evaluations=evaluations,
            started_at=started_at,
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", action="store_true")
    resume_group.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute_quality_replication(args.recipe, resume=args.resume or not args.no_resume)
    except (QualityReplicationQualificationError, quality.QualityPilotQualificationError) as exc:
        print(f"quality replication not qualified: {exc}", file=sys.stderr, flush=True)
        return 2
    except (QualityReplicationError, quality.QualityPilotError) as exc:
        print(f"quality replication failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "purpose": "quality_exploratory",
                "scientific_use": False,
                "aet_eligible": False,
                "energy_measurement": "none",
                "path": result.path.as_posix(),
                "complete": result.complete,
                "manifest": (
                    result.manifest_path.as_posix() if result.manifest_path is not None else None
                ),
                "manifest_sha256": result.manifest_sha256,
                "primary_classification": result.primary_classification,
                "completed_seeds": result.completed_seeds,
                "remaining_seeds": result.remaining_seeds,
                "discovery_seed_evaluated": False,
                "aet_was_computed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not result.complete:
        return 4
    return 0 if result.primary_classification == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
