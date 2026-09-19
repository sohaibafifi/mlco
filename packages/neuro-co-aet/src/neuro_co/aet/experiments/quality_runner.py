"""Resumable quality-only runner for the first AET journal pilot.

This runner intentionally records no energy.  It first locks a quality
reference built from two classical solver families, then trains one POMO + AM
seed, selects a checkpoint independently for each declared inference policy,
and applies the quality gate once on a held-out corpus.  It never computes an
AET value.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from neuro_co.aet.experiments.quality_recipe import (
    AETQualityRecipe,
    DatasetSplit,
    EvaluationMode,
    load_aet_quality_recipe,
    runtime_qualification,
)

RUN_STATE_SCHEMA = "aet-quality-pilot-run-state/v1"
CORPUS_SCHEMA = "aet-quality-pilot-corpus/v1"
REFERENCE_CANDIDATE_SCHEMA = "aet-quality-reference-candidate/v1"
REFERENCE_SCHEMA = "aet-quality-reference/v1"
REFERENCE_LOCK_SCHEMA = "aet-quality-reference-lock/v1"
CHECKPOINT_SCHEMA = "aet-quality-checkpoint/v1"
EVALUATION_SCHEMA = "aet-quality-evaluation/v1"
SELECTION_SCHEMA = "aet-quality-checkpoint-selection/v1"
GATE_SCHEMA = "aet-quality-gate/v1"
MANIFEST_SCHEMA = "aet-quality-pilot-manifest/v1"
SOURCE_SNAPSHOT_SCHEMA = "aet-quality-source-snapshot/v1"

_SOURCE_PACKAGE_ROOTS = (
    "packages/neuro-co-aet/src",
    "packages/neuro-co-core/src",
    "packages/neuro-co-problems/src",
)
_SOURCE_CONFIG_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "packages/neuro-co-aet/pyproject.toml",
    "packages/neuro-co-core/pyproject.toml",
    "packages/neuro-co-problems/pyproject.toml",
)
_COMPLETE_GATE_STATUSES = {
    "complete_quality_gate_passed",
    "complete_quality_gate_failed",
}


def _quality_classification() -> dict[str, Any]:
    return {
        "purpose": "quality_exploratory",
        "scientific_use": False,
        "aet_eligible": False,
        "energy_measurement": "none",
    }


class QualityPilotError(RuntimeError):
    """Raised when the quality pilot cannot proceed without corrupting its contract."""


class QualityPilotQualificationError(QualityPilotError):
    """Raised before execution when native Windows CUDA qualification is incomplete."""


@dataclass(frozen=True, slots=True)
class PilotResult:
    path: Path
    manifest_path: Path
    manifest_sha256: str
    passing_modes: tuple[str, ...]
    status: str


@dataclass(frozen=True, slots=True)
class Corpus:
    split: str
    split_id: str
    coords: Any
    demands: Any
    capacity: float
    content_sha256: str
    file_sha256: str
    path: Path


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_matches_sha256(path: Path, expected: str) -> bool:
    try:
        return path.is_file() and _sha256_file(path) == expected
    except OSError:
        return False


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(value))


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _stale_partial_artifacts(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name.startswith(".") and ".partial-" in path.name
    )


def _remove_stale_partial_artifacts(root: Path) -> list[str]:
    removed: list[str] = []
    for path in _stale_partial_artifacts(root):
        relative = path.relative_to(root).as_posix()
        path.unlink()
        removed.append(relative)
    return removed


def _record_partial_cleanup(root: Path, state: dict[str, Any]) -> None:
    removed = _remove_stale_partial_artifacts(root)
    if not removed:
        return
    cleanups = state.setdefault("resume_cleanups", [])
    if not isinstance(cleanups, list):
        raise QualityPilotError("run-state resume cleanup history is invalid")
    cleanups.append(
        {
            "at": datetime.now(UTC).isoformat(),
            "kind": "stale_atomic_partial_artifacts",
            "removed": removed,
        }
    )
    _atomic_write_json(root / "run-state.json", state)


def _invalidate_active_run(output: Path, state: dict[str, Any], reason: str) -> None:
    state["status"] = "invalidated_input_changed"
    state["invalidated_at"] = datetime.now(UTC).isoformat()
    state["invalidation_reason"] = reason
    _atomic_write_json(output / "run-state.json", state)
    raise QualityPilotError(f"{reason}; start a new output rather than resuming this one")


@contextmanager
def _output_lock(output: Path):
    """Prevent two resumable runners from mutating one output concurrently."""

    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output.parent / f".{output.name}.runner.lock"
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise QualityPilotError(
                f"another quality-pilot process is active for {output}"
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _write_training_checkpoint_pair(
    checkpoint_dir: Path,
    latest_path: Path,
    payload: dict[str, Any],
    *,
    epoch: int,
    checkpoint_epochs: tuple[int, ...],
) -> None:
    """Persist a planned snapshot before advancing the resumable latest pointer."""

    if epoch in checkpoint_epochs:
        _atomic_torch_save(checkpoint_dir / f"epoch-{epoch:03d}.pt", payload)
    _atomic_torch_save(latest_path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QualityPilotError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityPilotError(f"JSON artifact is not an object: {path}")
    return value


def _relative(root: Path, value: str) -> Path:
    return root.joinpath(*PurePosixPath(value).parts)


def _safe_output_target(workspace_root: Path, output_root: str) -> Path:
    root = workspace_root.resolve(strict=True)
    if root != Path.cwd().resolve(strict=True):
        raise QualityPilotError("workspace_root must be the current repository")
    target = _relative(root, output_root)
    cursor = root
    for component in target.relative_to(root).parts:
        cursor = cursor / component
        if (cursor.exists() or cursor.is_symlink()) and cursor.is_symlink():
            raise QualityPilotError(f"refusing symlinked output path: {cursor}")
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise QualityPilotError("output_root resolves outside the repository") from exc
    return resolved


def _git_snapshot(root: Path) -> dict[str, Any]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "diff", "--binary", "HEAD"], cwd=root, stderr=subprocess.DEVNULL
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QualityPilotQualificationError(
            "quality pilot must run inside a Git checkout"
        ) from exc
    fingerprint = hashlib.sha256()
    fingerprint.update(diff)
    for raw_name in sorted(part for part in untracked.split(b"\0") if part):
        name = raw_name.decode("utf-8", errors="surrogateescape")
        candidate = root / name
        if candidate.is_file():
            fingerprint.update(raw_name)
            fingerprint.update(bytes.fromhex(_sha256_file(candidate)))
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(diff) or bool(untracked),
        "worktree_fingerprint_sha256": fingerprint.hexdigest(),
        "clean_required": False,
    }


def _source_snapshot(root: Path) -> dict[str, Any]:
    """Fingerprint only the Python/runtime configuration used by this pilot."""

    relative_paths: set[str] = set(_SOURCE_CONFIG_PATHS)
    for relative_root in _SOURCE_PACKAGE_ROOTS:
        package_root = root / relative_root
        if not package_root.is_dir():
            raise QualityPilotQualificationError(
                f"quality-pilot source root is unavailable: {relative_root}"
            )
        relative_paths.update(
            path.relative_to(root).as_posix()
            for path in package_root.rglob("*.py")
            if path.is_file()
        )

    files: list[dict[str, str]] = []
    digest = hashlib.sha256()
    digest.update((SOURCE_SNAPSHOT_SCHEMA + "\0").encode("ascii"))
    for relative_path in sorted(relative_paths):
        path = root / relative_path
        if not path.is_file():
            raise QualityPilotQualificationError(
                f"quality-pilot source file is unavailable: {relative_path}"
            )
        file_sha256 = _sha256_file(path)
        encoded_path = relative_path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(bytes.fromhex(file_sha256))
        files.append({"path": relative_path, "sha256": file_sha256})
    return {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA,
        "scope": {
            "python": [f"{package_root}/**/*.py" for package_root in _SOURCE_PACKAGE_ROOTS],
            "configuration": list(_SOURCE_CONFIG_PATHS),
            "papers_and_experiment_outputs_excluded": True,
        },
        "file_count": len(files),
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _library_versions() -> dict[str, str | None]:
    distributions = {
        "torch": "torch",
        "numpy": "numpy",
        "pyvrp": "pyvrp",
        "ortools": "ortools",
    }
    versions: dict[str, str | None] = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _runtime_identity(recipe: AETQualityRecipe, qualification: dict[str, Any]) -> dict[str, Any]:
    import platform

    import torch

    properties = torch.cuda.get_device_properties(recipe.gpu_index)
    raw_uuid = getattr(properties, "uuid", None)
    driver_version: str | None = None
    smi_uuid: str | None = None
    try:
        smi_output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={recipe.gpu_index}",
                "--query-gpu=driver_version,uuid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).strip()
        driver_version, smi_uuid = (part.strip() for part in smi_output.split(",", maxsplit=1))
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        pass
    uuid_text = str(raw_uuid) if raw_uuid is not None else smi_uuid
    return {
        "schema_version": "aet-quality-runtime-identity/v1",
        "declared_host_id": recipe.host_id,
        "platform": sys.platform,
        "python": platform.python_version(),
        "libraries": _library_versions(),
        "torch_compiled_cuda": torch.version.cuda,
        "nvidia_driver_version": driver_version,
        "cuda_device_count": qualification["cuda_device_count"],
        "selected_gpu_index": recipe.gpu_index,
        "selected_gpu_name": qualification["selected_gpu_name"],
        "selected_gpu_total_memory_b": int(properties.total_memory),
        "selected_gpu_compute_capability": list(torch.cuda.get_device_capability(recipe.gpu_index)),
        "selected_gpu_uuid_sha256": (
            _sha256_bytes(uuid_text.encode("utf-8")) if uuid_text is not None else None
        ),
    }


def _record_invocation(
    output: Path,
    state: dict[str, Any],
    qualification: dict[str, Any],
    runtime_identity: dict[str, Any],
) -> None:
    invocations = state.setdefault("invocations", [])
    if not isinstance(invocations, list):
        raise QualityPilotError("run-state invocation history is invalid")
    invocations.append(
        {
            "started_at": datetime.now(UTC).isoformat(),
            "runtime_identity": runtime_identity,
            "qualification": qualification,
        }
    )
    _atomic_write_json(output / "run-state.json", state)


def _configure_runtime(gpu_index: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
        raise QualityPilotQualificationError(f"CUDA GPU index {gpu_index} is unavailable")
    torch.cuda.set_device(gpu_index)
    torch.use_deterministic_algorithms(False)
    return {
        "torch_deterministic_algorithms": False,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "selected_gpu_index": gpu_index,
        "selected_gpu_name": torch.cuda.get_device_name(gpu_index),
        "parallel_workloads_allowed": True,
        "exclusive_attestation_required": False,
        "energy_measurement": "none",
    }


def _prepare_output(
    recipe_path: Path,
    recipe: AETQualityRecipe,
    workspace_root: Path,
    *,
    resume: bool,
    recipe_bytes: bytes | None = None,
    runtime_identity: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    if recipe_bytes is None:
        recipe_bytes = recipe_path.read_bytes()
    recipe_sha256 = _sha256_bytes(recipe_bytes)
    lock_path = workspace_root / "uv.lock"
    try:
        lock_bytes = lock_path.read_bytes()
    except OSError as exc:
        raise QualityPilotQualificationError("uv.lock is unavailable") from exc
    lock_sha256 = _sha256_bytes(lock_bytes)
    git = _git_snapshot(workspace_root)
    source_snapshot = _source_snapshot(workspace_root)
    output = _safe_output_target(workspace_root, recipe.output_root)
    state_path = output / "run-state.json"
    if output.exists():
        if not resume:
            raise QualityPilotError(f"output already exists; use --resume: {output}")
        state = _load_json(state_path)
        expected = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha256,
            "uv_lock_sha256": lock_sha256,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise QualityPilotError("existing pilot was created from another recipe or lockfile")
        if state.get("runtime_identity") != runtime_identity:
            raise QualityPilotError("runtime or GPU identity changed since initialization")
        if str(state.get("status", "")).startswith("invalidated_"):
            raise QualityPilotError(
                "quality pilot was invalidated by an input change; start a new output"
            )
        frozen_recipe = output / "recipe.yaml"
        frozen_lock = output / "environment" / "uv.lock"
        if (
            not frozen_recipe.is_file()
            or _sha256_file(frozen_recipe) != recipe_sha256
            or not frozen_lock.is_file()
            or _sha256_file(frozen_lock) != lock_sha256
        ):
            raise QualityPilotError("frozen recipe or lockfile changed since initialization")
        stored_source_snapshot = state.get("source_snapshot")
        if not isinstance(stored_source_snapshot, dict) or (
            stored_source_snapshot.get("sha256") != source_snapshot["sha256"]
        ):
            raise QualityPilotError("quality-pilot source fingerprint changed since initialization")
        if state.get("status") not in _COMPLETE_GATE_STATUSES:
            _record_partial_cleanup(output, state)
        return output, state

    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": "initialized",
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe_sha256,
        "uv_lock_sha256": lock_sha256,
        "git_sha": git["sha"],
        "git": git,
        "source_snapshot": source_snapshot,
        "runtime_identity": runtime_identity,
        "classification": _quality_classification(),
        "uncommitted_worktree_reconstruction_limitation": (
            "the scoped runtime source was fingerprinted but not copied; unrelated dirty files "
            "remain outside the resume gate"
            if git["dirty"]
            else None
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _content_sha256(coords: Any, demands: Any, capacity: float) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(b"aet-quality-cvrp-corpus/v1\0")
    for name, value in (("coords", coords), ("demands", demands)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    digest.update(float(capacity).hex().encode("ascii"))
    return digest.hexdigest()


def _expected_corpus_arrays(recipe: AETQualityRecipe, spec: DatasetSplit) -> tuple[Any, Any]:
    import numpy as np
    import torch

    from neuro_co.core.factory import make_env

    env = make_env(
        recipe.dataset.problem,
        size=recipe.dataset.size,
        capacity=recipe.dataset.capacity,
        max_demand=recipe.dataset.max_demand,
    )
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    state = env.reset(spec.num_instances, generator=generator, device="cpu")
    coords = np.ascontiguousarray(state.coords.detach().cpu().numpy(), dtype=np.float32)
    demands = np.ascontiguousarray(state.demand.detach().cpu().numpy(), dtype=np.float32)
    return coords, demands


def _corpus_manifest(corpus: Corpus, spec: DatasetSplit) -> dict[str, Any]:
    return {
        "schema_version": CORPUS_SCHEMA,
        "split": corpus.split,
        "split_id": corpus.split_id,
        "num_instances": spec.num_instances,
        "seed": spec.seed,
        "content_sha256": corpus.content_sha256,
        "file_sha256": corpus.file_sha256,
    }


def _ensure_corpus_manifest(corpus: Corpus, spec: DatasetSplit) -> None:
    path = corpus.path.with_suffix(".manifest.json")
    expected = _corpus_manifest(corpus, spec)
    if path.exists():
        if _load_json(path) != expected:
            raise QualityPilotError(f"corpus side manifest changed: {path}")
    else:
        _atomic_write_json(path, expected)


def _generate_corpus(
    recipe: AETQualityRecipe,
    split_name: str,
    spec: DatasetSplit,
    path: Path,
) -> Corpus:
    import numpy as np

    coords, demands = _expected_corpus_arrays(recipe, spec)
    content_sha256 = _content_sha256(coords, demands, recipe.dataset.capacity)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    try:
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(CORPUS_SCHEMA),
            split=np.asarray(split_name),
            split_id=np.asarray(spec.split_id),
            coords=coords,
            demands=demands,
            capacity=np.asarray(recipe.dataset.capacity, dtype=np.float64),
            seed=np.asarray(spec.seed, dtype=np.int64),
            content_sha256=np.asarray(content_sha256),
        )
        generated = temporary.with_suffix(temporary.suffix + ".npz")
        if generated.exists() and generated != temporary:
            temporary = generated
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    corpus = Corpus(
        split=split_name,
        split_id=spec.split_id,
        coords=coords,
        demands=demands,
        capacity=recipe.dataset.capacity,
        content_sha256=content_sha256,
        file_sha256=_sha256_file(path),
        path=path,
    )
    _ensure_corpus_manifest(corpus, spec)
    return corpus


def _load_corpus(
    recipe: AETQualityRecipe,
    split_name: str,
    spec: DatasetSplit,
    path: Path,
) -> Corpus:
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = str(archive["schema_version"].item())
            embedded_split = str(archive["split"].item())
            split_id = str(archive["split_id"].item())
            coords = np.ascontiguousarray(archive["coords"], dtype=np.float32)
            demands = np.ascontiguousarray(archive["demands"], dtype=np.float32)
            capacity = float(archive["capacity"].item())
            seed = int(archive["seed"].item())
            embedded_hash = str(archive["content_sha256"].item())
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise QualityPilotError(f"cannot load corpus {path}") from exc
    expected_coords = (spec.num_instances, recipe.dataset.size + 1, 2)
    expected_demands = expected_coords[:2]
    actual_hash = _content_sha256(coords, demands, capacity)
    deterministic_coords, deterministic_demands = _expected_corpus_arrays(recipe, spec)
    if (
        schema != CORPUS_SCHEMA
        or embedded_split != split_name
        or split_id != spec.split_id
        or coords.shape != expected_coords
        or demands.shape != expected_demands
        or capacity != recipe.dataset.capacity
        or seed != spec.seed
        or embedded_hash != actual_hash
        or not np.array_equal(coords, deterministic_coords)
        or not np.array_equal(demands, deterministic_demands)
    ):
        raise QualityPilotError(f"corpus metadata or content changed: {path}")
    corpus = Corpus(
        split=split_name,
        split_id=split_id,
        coords=coords,
        demands=demands,
        capacity=capacity,
        content_sha256=actual_hash,
        file_sha256=_sha256_file(path),
        path=path,
    )
    _ensure_corpus_manifest(corpus, spec)
    return corpus


def _prepare_corpora(recipe: AETQualityRecipe, output: Path) -> dict[str, Corpus]:
    corpora: dict[str, Corpus] = {}
    for split_name, spec in (
        ("selection", recipe.dataset.selection),
        ("holdout", recipe.dataset.holdout),
    ):
        path = _relative(output, spec.artifact)
        if path.exists():
            corpora[split_name] = _load_corpus(recipe, split_name, spec, path)
        else:
            corpora[split_name] = _generate_corpus(recipe, split_name, spec, path)
    if corpora["selection"].content_sha256 == corpora["holdout"].content_sha256:
        raise QualityPilotError("selection and holdout corpora unexpectedly match")
    return corpora


def validate_routes(
    coords: Any,
    demands: Any,
    capacity: float,
    routes_by_instance: Any,
) -> dict[str, Any]:
    """Validate coverage, capacity, and cost independently of both solvers."""

    import operator

    import numpy as np

    points = np.asarray(coords, dtype=np.float64)
    loads = np.asarray(demands, dtype=np.float64)
    routes = tuple(routes_by_instance)
    if points.ndim != 3 or points.shape[2] != 2:
        raise QualityPilotError("validation coords must have shape [batch, nodes, 2]")
    if loads.shape != points.shape[:2] or len(routes) != points.shape[0]:
        raise QualityPilotError("solution count or demand shape disagrees with corpus")
    if not np.isfinite(points).all() or not np.isfinite(loads).all():
        raise QualityPilotError("validation inputs must be finite")
    if not math.isfinite(capacity) or capacity <= 0:
        raise QualityPilotError("validation capacity must be finite and positive")

    expected = list(range(1, points.shape[1]))
    costs: list[float] = []
    violations: list[dict[str, Any]] = []
    for instance_index, raw_routes in enumerate(routes):
        flattened: list[int] = []
        total_cost = 0.0
        for route_index, raw_route in enumerate(raw_routes):
            route: list[int] = []
            for raw_customer in raw_route:
                try:
                    route.append(operator.index(raw_customer))
                except TypeError:
                    violations.append(
                        {
                            "instance": instance_index,
                            "route": route_index,
                            "kind": "non_integer_customer",
                        }
                    )
            if not route:
                violations.append(
                    {"instance": instance_index, "route": route_index, "kind": "empty_route"}
                )
                continue
            invalid = [customer for customer in route if customer not in expected]
            if invalid:
                violations.append(
                    {
                        "instance": instance_index,
                        "route": route_index,
                        "kind": "customer_out_of_range",
                        "customers": invalid,
                    }
                )
                continue
            route_load = float(loads[instance_index, route].sum())
            if route_load > capacity + 1e-9:
                violations.append(
                    {
                        "instance": instance_index,
                        "route": route_index,
                        "kind": "capacity_exceeded",
                        "load": route_load,
                    }
                )
            flattened.extend(route)
            for origin, destination in pairwise((0, *route, 0)):
                total_cost += float(
                    np.linalg.norm(
                        points[instance_index, origin] - points[instance_index, destination]
                    )
                )
        if sorted(flattened) != expected:
            violations.append(
                {
                    "instance": instance_index,
                    "kind": "customer_coverage",
                    "missing": sorted(set(expected) - set(flattened)),
                    "duplicates": sorted(
                        customer for customer in set(flattened) if flattened.count(customer) > 1
                    ),
                }
            )
        costs.append(total_cost)
    invalid_instance_count = len({int(violation["instance"]) for violation in violations})
    return {
        "complete": not violations,
        "validator": "independent-euclidean-cvrp-route-checker",
        "instance_count": points.shape[0],
        "validated_instance_count": points.shape[0] - invalid_instance_count,
        "invalid_instance_count": invalid_instance_count,
        "failure_count": len(violations),
        "costs": costs,
        "mean_cost": float(np.mean(costs)),
        "violations": violations,
    }


def _assert_valid(validation: dict[str, Any], label: str) -> None:
    if validation.get("complete") is not True:
        raise QualityPilotError(f"{label} returned invalid CVRP routes")


def _candidate_payload(
    *,
    corpus: Corpus,
    family: str,
    solver: str,
    seed: int,
    settings: dict[str, Any],
    results: list[Any],
    elapsed_s: float,
) -> dict[str, Any]:
    routes = tuple(result.routes for result in results)
    validation = validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        routes,
    )
    _assert_valid(validation, f"{corpus.split} {solver} seed {seed}")
    solver_results = []
    for result in results:
        record = {
            "instance_index": int(result.instance_index),
            "effective_seed": int(result.seed),
            "integer_cost": int(result.integer_cost),
            "reported_cost": float(result.cost),
            "scaling_factor": int(result.scaling_factor),
            "search_status": str(getattr(result, "search_status", "feasible_complete")),
        }
        for field in (
            "max_iterations",
            "limit_kind",
            "max_runtime_s",
            "solution_limit",
            "search_status",
        ):
            if hasattr(result, field):
                record[field] = getattr(result, field)
        solver_results.append(record)
    return {
        "schema_version": REFERENCE_CANDIDATE_SCHEMA,
        "classification": {
            "purpose": "quality_reference",
            "energy_measurement": "none",
        },
        "split": corpus.split,
        "dataset_content_sha256": corpus.content_sha256,
        "family": family,
        "solver": solver,
        "seed": seed,
        "settings": settings,
        "elapsed_s": elapsed_s,
        "routes": routes,
        "solver_reported_costs": [float(result.cost) for result in results],
        "solver_results": solver_results,
        "validation": validation,
    }


def _revalidate_stored_routes(
    payload: dict[str, Any],
    corpus: Corpus,
    label: str,
    *,
    require_complete: bool = True,
) -> dict[str, Any]:
    validation = validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        payload.get("routes", ()),
    )
    if require_complete:
        _assert_valid(validation, label)
    stored = payload.get("validation")
    if not isinstance(stored, dict) or _json_bytes(validation) != _json_bytes(stored):
        raise QualityPilotError(f"stored route validation changed: {label}")
    return validation


def _validate_candidate_result_metadata(
    payload: dict[str, Any], corpus: Corpus, *, base_seed: int, label: str
) -> None:
    records = payload.get("solver_results")
    reported_costs = payload.get("solver_reported_costs")
    elapsed_s = payload.get("elapsed_s")
    if (
        payload.get("classification")
        != {"purpose": "quality_reference", "energy_measurement": "none"}
        or payload.get("split") != corpus.split
        or not isinstance(elapsed_s, (int, float))
        or isinstance(elapsed_s, bool)
        or not math.isfinite(float(elapsed_s))
        or float(elapsed_s) < 0.0
        or not isinstance(records, list)
        or len(records) != corpus.coords.shape[0]
        or not isinstance(reported_costs, list)
        or len(reported_costs) != corpus.coords.shape[0]
    ):
        raise QualityPilotError(f"reference candidate result metadata changed: {label}")
    for instance_index, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or record.get("instance_index") != instance_index
            or record.get("effective_seed") != base_seed + instance_index
            or isinstance(record.get("integer_cost"), bool)
            or not isinstance(record.get("integer_cost"), int)
            or not isinstance(record.get("search_status"), str)
            or isinstance(record.get("reported_cost"), bool)
            or not isinstance(record.get("reported_cost"), (int, float))
            or isinstance(reported_costs[instance_index], bool)
            or not isinstance(reported_costs[instance_index], (int, float))
            or not math.isclose(
                float(record.get("reported_cost", math.nan)),
                float(reported_costs[instance_index]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise QualityPilotError(f"reference candidate result metadata changed: {label}")


def _run_reference_candidates(
    recipe: AETQualityRecipe,
    corpus: Corpus,
    output: Path,
) -> list[dict[str, Any]]:
    from neuro_co.problems.cvrp.ortools import solve_corpus_sequential as solve_ortools
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential as solve_hgs

    candidate_dir = output / "reference" / corpus.split / "candidates"
    candidates: list[dict[str, Any]] = []
    for seed in recipe.reference.hgs.seeds:
        path = candidate_dir / f"pyvrp-hgs-seed{seed}.json"
        family = "hybrid-genetic-search"
        settings = {"max_iterations": recipe.reference.hgs.max_iterations}
        if path.exists():
            payload = _load_json(path)
        else:
            started = time.perf_counter()
            results = solve_hgs(
                corpus.coords,
                corpus.demands,
                corpus.capacity,
                seed=seed,
                max_iterations=recipe.reference.hgs.max_iterations,
                collect_stats=False,
            )
            payload = _candidate_payload(
                corpus=corpus,
                family=family,
                solver=recipe.reference.hgs.solver,
                seed=seed,
                settings=settings,
                results=results,
                elapsed_s=time.perf_counter() - started,
            )
            _atomic_write_json(path, payload)
        if (
            payload.get("schema_version") != REFERENCE_CANDIDATE_SCHEMA
            or payload.get("dataset_content_sha256") != corpus.content_sha256
            or payload.get("family") != family
            or payload.get("solver") != recipe.reference.hgs.solver
            or payload.get("seed") != seed
            or payload.get("settings") != settings
        ):
            raise QualityPilotError(f"reference candidate disagrees with recipe: {path}")
        _revalidate_stored_routes(payload, corpus, str(path))
        _validate_candidate_result_metadata(payload, corpus, base_seed=seed, label=str(path))
        for record in payload["solver_results"]:
            if (
                record.get("max_iterations") != recipe.reference.hgs.max_iterations
                or record.get("search_status") != "feasible_complete"
                or record.get("scaling_factor") != 1_000_000
            ):
                raise QualityPilotError(f"reference candidate limit metadata changed: {path}")
        candidates.append(payload)

    ortools_seed = recipe.reference.ortools.seed
    ortools_path = candidate_dir / f"ortools-routing-gls-seed{ortools_seed}.json"
    ortools_family = "ortools-routing"
    ortools_settings = {
        "solution_limit": recipe.reference.ortools.solution_limit,
        "scaling_factor": recipe.reference.ortools.scaling_factor,
        "search": "parallel-cheapest-insertion-plus-guided-local-search",
        "num_search_workers": 1,
        "timing_sensitivity": False,
    }
    if ortools_path.exists():
        ortools_payload = _load_json(ortools_path)
    else:
        started = time.perf_counter()
        ortools_results = solve_ortools(
            corpus.coords,
            corpus.demands,
            corpus.capacity,
            seed=ortools_seed,
            solution_limit=recipe.reference.ortools.solution_limit,
            scaling_factor=recipe.reference.ortools.scaling_factor,
        )
        ortools_payload = _candidate_payload(
            corpus=corpus,
            family=ortools_family,
            solver=recipe.reference.ortools.solver,
            seed=ortools_seed,
            settings=ortools_settings,
            results=ortools_results,
            elapsed_s=time.perf_counter() - started,
        )
        _atomic_write_json(ortools_path, ortools_payload)
    if (
        ortools_payload.get("schema_version") != REFERENCE_CANDIDATE_SCHEMA
        or ortools_payload.get("dataset_content_sha256") != corpus.content_sha256
        or ortools_payload.get("family") != ortools_family
        or ortools_payload.get("solver") != recipe.reference.ortools.solver
        or ortools_payload.get("seed") != ortools_seed
        or ortools_payload.get("settings") != ortools_settings
    ):
        raise QualityPilotError(f"reference candidate disagrees with recipe: {ortools_path}")
    _revalidate_stored_routes(ortools_payload, corpus, str(ortools_path))
    _validate_candidate_result_metadata(
        ortools_payload, corpus, base_seed=ortools_seed, label=str(ortools_path)
    )
    for record in ortools_payload["solver_results"]:
        if (
            record.get("limit_kind") != "solutions"
            or record.get("solution_limit") != recipe.reference.ortools.solution_limit
            or record.get("max_runtime_s") is not None
            or not isinstance(record.get("search_status"), str)
        ):
            raise QualityPilotError(
                f"reference candidate limit or search metadata changed: {ortools_path}"
            )
    candidates.append(ortools_payload)
    return candidates


def _build_reference(
    recipe: AETQualityRecipe,
    corpus: Corpus,
    output: Path,
) -> dict[str, Any]:
    import numpy as np

    path = output / "reference" / corpus.split / "reference.json"
    candidates = _run_reference_candidates(recipe, corpus, output)
    candidate_artifacts: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate["solver"] == recipe.reference.hgs.solver:
            filename = f"pyvrp-hgs-seed{candidate['seed']}.json"
        elif candidate["solver"] == recipe.reference.ortools.solver:
            filename = f"ortools-routing-gls-seed{candidate['seed']}.json"
        else:  # pragma: no cover - guarded by strict candidate validation
            raise QualityPilotError("unexpected solver in reference candidates")
        candidate_path = output / "reference" / corpus.split / "candidates" / filename
        candidate_artifacts.append(
            {
                "path": candidate_path.relative_to(output).as_posix(),
                "sha256": _sha256_file(candidate_path),
                "family": candidate["family"],
                "solver": candidate["solver"],
                "base_seed": candidate["seed"],
                "settings": candidate["settings"],
            }
        )
    candidate_costs = [candidate["validation"]["costs"] for candidate in candidates]
    routes: list[Any] = []
    costs: list[float] = []
    sources: list[dict[str, Any]] = []
    for instance_index in range(corpus.coords.shape[0]):
        best_index = min(
            range(len(candidates)),
            key=lambda index: (float(candidate_costs[index][instance_index]), index),
        )
        candidate = candidates[best_index]
        solver_result = candidate["solver_results"][instance_index]
        routes.append(candidate["routes"][instance_index])
        costs.append(float(candidate_costs[best_index][instance_index]))
        sources.append(
            {
                "family": candidate["family"],
                "solver": candidate["solver"],
                "base_seed": candidate["seed"],
                "effective_seed": solver_result["effective_seed"],
                "integer_cost": solver_result["integer_cost"],
                "reported_cost": solver_result["reported_cost"],
                "candidate_sha256": candidate_artifacts[best_index]["sha256"],
            }
        )
    validation = validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        routes,
    )
    _assert_valid(validation, f"combined {corpus.split} reference")
    if not np.allclose(validation["costs"], costs, rtol=0.0, atol=1e-10):
        raise QualityPilotError("combined reference costs changed during route validation")
    source_counts: dict[str, int] = {}
    for source in sources:
        key = f"{source['solver']}:seed{source['base_seed']}"
        source_counts[key] = source_counts.get(key, 0) + 1
    if path.exists():
        reference = _load_json(path)
        expected_exact = {
            "schema_version": REFERENCE_SCHEMA,
            "status": "complete",
            "split": corpus.split,
            "dataset_content_sha256": corpus.content_sha256,
            "policy": recipe.reference.policy,
            "independent_solver_families": sorted(
                {candidate["family"] for candidate in candidates}
            ),
            "candidate_count": len(candidates),
            "candidate_artifacts": candidate_artifacts,
            "source_counts": source_counts,
            "future_timed_baseline_is_separate": True,
            "energy_measurement": "none",
        }
        if any(reference.get(key) != value for key, value in expected_exact.items()):
            raise QualityPilotError(f"locked reference disagrees with candidates: {path}")
        _revalidate_stored_routes(reference, corpus, str(path))
        if _json_bytes(reference.get("routes")) != _json_bytes(routes) or _json_bytes(
            reference.get("sources")
        ) != _json_bytes(sources):
            raise QualityPilotError(f"locked reference selection changed: {path}")
        try:
            costs_match = np.allclose(reference.get("costs", ()), costs, rtol=0.0, atol=1e-10)
        except (TypeError, ValueError):
            costs_match = False
        if not costs_match:
            raise QualityPilotError(f"locked reference costs changed: {path}")
        if not math.isclose(
            float(reference.get("mean_cost", math.nan)),
            float(np.mean(costs)),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise QualityPilotError(f"locked reference mean cost changed: {path}")
        return reference

    reference = {
        "schema_version": REFERENCE_SCHEMA,
        "status": "complete",
        "split": corpus.split,
        "dataset_content_sha256": corpus.content_sha256,
        "policy": recipe.reference.policy,
        "independent_solver_families": sorted({candidate["family"] for candidate in candidates}),
        "candidate_count": len(candidates),
        "candidate_artifacts": candidate_artifacts,
        "source_counts": source_counts,
        "sources": sources,
        "routes": routes,
        "costs": costs,
        "mean_cost": float(np.mean(costs)),
        "validation": validation,
        "future_timed_baseline_is_separate": True,
        "energy_measurement": "none",
    }
    _atomic_write_json(path, reference)
    return reference


def _prepare_references(
    recipe: AETQualityRecipe,
    corpora: dict[str, Corpus],
    output: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    references = {
        split: _build_reference(recipe, corpus, output) for split, corpus in corpora.items()
    }
    lock_path = output / "reference" / "reference-lock.json"
    expected_entries = {
        split: {
            "path": f"reference/{split}/reference.json",
            "sha256": _sha256_file(output / "reference" / split / "reference.json"),
            "dataset_content_sha256": corpora[split].content_sha256,
        }
        for split in ("selection", "holdout")
    }
    if lock_path.exists():
        lock = _load_json(lock_path)
        if (
            lock.get("schema_version") != REFERENCE_LOCK_SCHEMA
            or lock.get("status") != "locked"
            or lock.get("locked_before_neural_evaluation") is not True
            or lock.get("policy") != recipe.reference.policy
            or lock.get("solver_families") != ["hybrid-genetic-search", "ortools-routing"]
            or lock.get("entries") != expected_entries
        ):
            raise QualityPilotError("reference lock does not match current reference artifacts")
    else:
        lock = {
            "schema_version": REFERENCE_LOCK_SCHEMA,
            "status": "locked",
            "locked_at": datetime.now(UTC).isoformat(),
            "locked_before_neural_evaluation": True,
            "policy": recipe.reference.policy,
            "solver_families": ["hybrid-genetic-search", "ortools-routing"],
            "entries": expected_entries,
        }
        _atomic_write_json(lock_path, lock)
    return references, lock


def _verify_anchored_reference_tree(
    output: Path,
    run_state: dict[str, Any],
    corpora: dict[str, Corpus],
) -> None:
    expected_lock_sha256 = run_state.get("reference_lock_sha256")
    lock_path = output / "reference" / "reference-lock.json"
    if (
        not isinstance(expected_lock_sha256, str)
        or len(expected_lock_sha256) != 64
        or not _file_matches_sha256(lock_path, expected_lock_sha256)
    ):
        raise QualityPilotError("anchored reference lock is missing or changed")
    lock = _load_json(lock_path)
    entries = lock.get("entries")
    if not isinstance(entries, dict):
        raise QualityPilotError("anchored reference lock entries are invalid")
    for split in ("selection", "holdout"):
        entry = entries.get(split)
        reference_path = output / "reference" / split / "reference.json"
        if (
            not isinstance(entry, dict)
            or entry.get("dataset_content_sha256") != corpora[split].content_sha256
            or not _file_matches_sha256(reference_path, entry.get("sha256", ""))
        ):
            raise QualityPilotError(f"anchored {split} reference is missing or changed")
        reference = _load_json(reference_path)
        for candidate in reference.get("candidate_artifacts", ()):
            if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
                raise QualityPilotError(f"anchored {split} candidate inventory is invalid")
            if not _file_matches_sha256(
                _relative(output, candidate["path"]), candidate.get("sha256", "")
            ):
                raise QualityPilotError(f"anchored {split} candidate is missing or changed")


def _make_mlco_am(recipe: AETQualityRecipe, env: Any) -> Any:
    from neuro_co.core.factory import make_model

    if recipe.model.architecture != "mlco-am":
        raise QualityPilotError(
            f"unsupported quality-pilot architecture: {recipe.model.architecture}"
        )
    return make_model(
        env,
        backbone="am",
        hidden_dim=recipe.model.hidden_dim,
        num_layers=recipe.model.num_layers,
        num_heads=recipe.model.num_heads,
    )


def _model_identity(recipe: AETQualityRecipe, model: Any) -> dict[str, Any]:
    structure = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }
        for name, tensor in model.state_dict().items()
    ]
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != recipe.model.expected_parameter_count:
        raise QualityPilotError(
            "MLCO AM parameter count mismatch: "
            f"got {parameter_count}, expected {recipe.model.expected_parameter_count}"
        )
    return {
        "implementation": "MLCO AM",
        "recipe_architecture": recipe.model.architecture,
        "factory_backbone": "am",
        "configuration": asdict(recipe.model),
        "parameter_count": parameter_count,
        "state_dict_structure_sha256": _sha256_bytes(_json_bytes(structure)),
        "state_dict_signature_fields": ["name", "shape", "dtype"],
        "historical_external_implementation": False,
    }


def _make_training_stack(
    recipe: AETQualityRecipe, device: Any
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    import torch

    from neuro_co.core.factory import make_algo, make_env

    env = make_env(
        recipe.dataset.problem,
        size=recipe.dataset.size,
        capacity=recipe.dataset.capacity,
        max_demand=recipe.dataset.max_demand,
    )
    model = _make_mlco_am(recipe, env)
    model_identity = _model_identity(recipe, model)
    training = recipe.training
    algo = make_algo(
        training.algorithm,
        model,
        env,
        device=str(device),
        batch_size=training.batch_size,
        n_starts=training.n_starts,
        lr=training.learning_rate,
        optimizer=training.optimizer,
        weight_decay=training.weight_decay,
        grad_clip=training.gradient_clip,
        precision=training.precision,
        eval_batch_size=recipe.dataset.selection.num_instances,
        eval_augment=1,
        lr_warmup_steps=0,
        lr_total_steps=0,
    )
    if algo.sched is not None:
        raise QualityPilotError("POMO unexpectedly created a step-based scheduler")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        algo.opt,
        milestones=list(training.scheduler.milestones),
        gamma=training.scheduler.gamma,
    )
    return env, model, algo, scheduler, model_identity


def _checkpoint_payload(
    recipe: AETQualityRecipe,
    *,
    model: Any,
    algo: Any,
    scheduler: Any,
    generator: Any,
    epoch: int,
    items_processed: int,
    cumulative_training_walltime_s: float,
    recipe_sha256: str,
    git_sha: str,
    last_epoch_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    import torch

    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "recipe_sha256": recipe_sha256,
        "git_sha": git_sha,
        "problem": recipe.dataset.problem,
        "size": recipe.dataset.size,
        "capacity": recipe.dataset.capacity,
        "max_demand": recipe.dataset.max_demand,
        "architecture": asdict(recipe.model),
        "model_identity": _model_identity(recipe, model),
        "training": asdict(recipe.training),
        "completed_epochs": epoch,
        "items_processed": items_processed,
        "cumulative_training_walltime_s": cumulative_training_walltime_s,
        "optimizer_steps": int(algo._step),
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer": algo.opt.state_dict(),
        "epoch_scheduler": scheduler.state_dict(),
        "train_generator_state": generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "last_epoch_metrics": last_epoch_metrics,
        "saved_at": datetime.now(UTC).isoformat(),
    }


def _validate_checkpoint_metadata(
    payload: dict[str, Any],
    recipe: AETQualityRecipe,
    run_state: dict[str, Any],
    model_identity: dict[str, Any],
) -> None:
    expected = {
        "schema_version": CHECKPOINT_SCHEMA,
        "recipe_sha256": run_state["recipe_sha256"],
        "git_sha": run_state["git_sha"],
        "problem": recipe.dataset.problem,
        "size": recipe.dataset.size,
        "capacity": recipe.dataset.capacity,
        "max_demand": recipe.dataset.max_demand,
        "architecture": asdict(recipe.model),
        "model_identity": model_identity,
        "training": asdict(recipe.training),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise QualityPilotError("checkpoint metadata disagrees with this pilot")


def _load_torch_mapping(path: Path) -> dict[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise QualityPilotError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QualityPilotError(f"checkpoint is not a mapping: {path}")
    return value


def _restore_training_state(
    payload: dict[str, Any],
    *,
    model: Any,
    algo: Any,
    scheduler: Any,
    generator: Any,
) -> tuple[int, int, float]:
    import torch

    model.load_state_dict(payload["model"], strict=True)
    algo.opt.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["epoch_scheduler"])
    generator.set_state(payload["train_generator_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    cuda_states = payload.get("torch_cuda_rng_states")
    if not isinstance(cuda_states, list) or len(cuda_states) != torch.cuda.device_count():
        raise QualityPilotError("checkpoint CUDA RNG state count disagrees with visible GPUs")
    torch.cuda.set_rng_state_all(cuda_states)
    algo._step = int(payload["optimizer_steps"])
    return (
        int(payload["completed_epochs"]),
        int(payload["items_processed"]),
        float(payload["cumulative_training_walltime_s"]),
    )


def _epoch_metrics(rows: list[tuple[int, dict[str, float]]]) -> dict[str, float]:
    if not rows:
        raise QualityPilotError("training epoch completed no optimizer step")
    total = sum(weight for weight, _ in rows)
    keys = rows[0][1].keys()
    return {key: sum(weight * metrics[key] for weight, metrics in rows) / total for key in keys}


def _train(
    recipe: AETQualityRecipe,
    output: Path,
    run_state: dict[str, Any],
) -> dict[str, Any]:
    import torch

    training = recipe.training
    device = torch.device(f"cuda:{recipe.gpu_index}")
    random.seed(training.seed)
    torch.manual_seed(training.seed)
    torch.cuda.manual_seed_all(training.seed)
    env, model, algo, scheduler, model_identity = _make_training_stack(recipe, device)
    _ = env
    generator = torch.Generator(device=device).manual_seed(training.seed)
    checkpoint_dir = output / "training" / "checkpoints"
    latest_path = output / "training" / "latest.pt"
    epoch = 0
    items_processed = 0
    cumulative_training_walltime_s = 0.0
    if latest_path.exists():
        payload = _load_torch_mapping(latest_path)
        _validate_checkpoint_metadata(payload, recipe, run_state, model_identity)
        epoch, items_processed, cumulative_training_walltime_s = _restore_training_state(
            payload,
            model=model,
            algo=algo,
            scheduler=scheduler,
            generator=generator,
        )
    else:
        payload = _checkpoint_payload(
            recipe,
            model=model,
            algo=algo,
            scheduler=scheduler,
            generator=generator,
            epoch=0,
            items_processed=0,
            cumulative_training_walltime_s=0.0,
            recipe_sha256=run_state["recipe_sha256"],
            git_sha=run_state["git_sha"],
            last_epoch_metrics=None,
        )
        _write_training_checkpoint_pair(
            checkpoint_dir,
            latest_path,
            payload,
            epoch=0,
            checkpoint_epochs=training.checkpoint_epochs,
        )

    if epoch < 0 or epoch > training.epochs:
        raise QualityPilotError(f"invalid completed epoch in latest checkpoint: {epoch}")
    expected_items = epoch * training.instances_per_epoch
    expected_optimizer_steps = epoch * math.ceil(training.instances_per_epoch / training.batch_size)
    if items_processed != expected_items:
        raise QualityPilotError(
            f"checkpoint item count mismatch: got {items_processed}, expected {expected_items}"
        )
    if int(algo._step) != expected_optimizer_steps:
        raise QualityPilotError(
            "checkpoint optimizer-step count mismatch: "
            f"got {int(algo._step)}, expected {expected_optimizer_steps}"
        )

    started = time.perf_counter()
    while epoch < training.epochs:
        epoch_started = time.perf_counter()
        epoch_rows: list[tuple[int, dict[str, float]]] = []
        remaining = training.instances_per_epoch
        while remaining > 0:
            current_batch = min(training.batch_size, remaining)
            algo.cfg.batch_size = current_batch
            metrics = {key: float(value) for key, value in algo.train_step(generator).items()}
            if any(not math.isfinite(value) for value in metrics.values()):
                raise QualityPilotError(f"non-finite training metric at epoch {epoch + 1}")
            epoch_rows.append((current_batch, metrics))
            remaining -= current_batch
            items_processed += current_batch
        algo.cfg.batch_size = training.batch_size
        scheduler.step()
        epoch += 1
        cumulative_training_walltime_s += time.perf_counter() - epoch_started
        metrics = _epoch_metrics(epoch_rows)
        payload = _checkpoint_payload(
            recipe,
            model=model,
            algo=algo,
            scheduler=scheduler,
            generator=generator,
            epoch=epoch,
            items_processed=items_processed,
            cumulative_training_walltime_s=cumulative_training_walltime_s,
            recipe_sha256=run_state["recipe_sha256"],
            git_sha=run_state["git_sha"],
            last_epoch_metrics=metrics,
        )
        _write_training_checkpoint_pair(
            checkpoint_dir,
            latest_path,
            payload,
            epoch=epoch,
            checkpoint_epochs=training.checkpoint_epochs,
        )
        elapsed_s = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "training",
                    "completed_epoch": epoch,
                    "items_processed": items_processed,
                    "optimizer_steps": int(algo._step),
                    "learning_rate": float(algo.opt.param_groups[0]["lr"]),
                    "cumulative_training_walltime_s": cumulative_training_walltime_s,
                    "elapsed_s_this_invocation": elapsed_s,
                    "metrics": metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if elapsed_s > training.max_walltime_per_invocation_s and epoch < training.epochs:
            raise QualityPilotError(
                "training walltime limit reached after a complete epoch; rerun the same command"
            )

    checkpoint_records: list[dict[str, Any]] = []
    for planned_epoch in training.checkpoint_epochs:
        path = checkpoint_dir / f"epoch-{planned_epoch:03d}.pt"
        if not path.is_file():
            raise QualityPilotError(f"planned checkpoint is missing: {path}")
        checkpoint = _load_torch_mapping(path)
        _validate_checkpoint_metadata(checkpoint, recipe, run_state, model_identity)
        if checkpoint.get("completed_epochs") != planned_epoch:
            raise QualityPilotError(f"checkpoint epoch metadata mismatch: {path}")
        expected_checkpoint_items = planned_epoch * training.instances_per_epoch
        expected_checkpoint_steps = planned_epoch * math.ceil(
            training.instances_per_epoch / training.batch_size
        )
        if (
            checkpoint.get("items_processed") != expected_checkpoint_items
            or checkpoint.get("optimizer_steps") != expected_checkpoint_steps
        ):
            raise QualityPilotError(f"checkpoint progress metadata mismatch: {path}")
        checkpoint_records.append(
            {
                "epoch": planned_epoch,
                "path": path.relative_to(output).as_posix(),
                "sha256": _sha256_file(path),
                "items_processed": checkpoint["items_processed"],
                "optimizer_steps": checkpoint["optimizer_steps"],
            }
        )
    result = {
        "status": "complete",
        "algorithm": training.algorithm,
        "seed": training.seed,
        "epochs": training.epochs,
        "items_processed": items_processed,
        "optimizer_steps": int(algo._step),
        "cumulative_training_walltime_s": cumulative_training_walltime_s,
        "model_identity": model_identity,
        "parameter_count": model_identity["parameter_count"],
        "training_objective": {
            "implementation": "MLCO POMO group-mean advantage",
            "historical_reward_zscore_applied": False,
            "historical_reproduction": False,
        },
        "checkpoint_records": checkpoint_records,
        "energy_measurement": "none",
        "aet_eligible": False,
    }
    _atomic_write_json(output / "training" / "result.json", result)
    return result


def _initial_state(env: Any, coords: Any, demands: Any, device: Any) -> Any:
    import torch

    coords_tensor = torch.as_tensor(coords, dtype=torch.float32, device=device)
    demands_tensor = torch.as_tensor(demands, dtype=torch.float32, device=device)
    state = env.reset(demands_tensor.shape[0], device=device)
    return state.replace(coords=coords_tensor, demand=demands_tensor)


def _routes_from_actions(actions: Any, active: Any) -> tuple[tuple[tuple[int, ...], ...], ...]:
    routes_by_instance: list[tuple[tuple[int, ...], ...]] = []
    for row_actions, row_active in zip(actions, active, strict=True):
        routes: list[tuple[int, ...]] = []
        current: list[int] = []
        for raw_action, is_active in zip(row_actions, row_active, strict=True):
            if not bool(is_active):
                continue
            action = int(raw_action)
            if action == 0:
                if current:
                    routes.append(tuple(current))
                    current = []
            else:
                current.append(action)
        if current:
            raise QualityPilotError("neural rollout did not close its final route at the depot")
        routes_by_instance.append(tuple(routes))
    return tuple(routes_by_instance)


def _greedy_routes(model: Any, env: Any, state: Any) -> tuple[tuple[tuple[int, ...], ...], ...]:
    import torch

    from neuro_co.core.trace import rollout_trace

    trace = rollout_trace(model, env, state, decode="greedy")
    if trace.reward is None or not trace.steps:
        raise QualityPilotError("greedy evaluation produced an empty trace")
    actions = torch.stack([step.action for step in trace.steps], dim=1).detach().cpu().numpy()
    active = torch.stack([step.active for step in trace.steps], dim=1).detach().cpu().numpy()
    return _routes_from_actions(actions, active)


def _multistart_routes(
    model: Any,
    env: Any,
    state: Any,
    *,
    n_starts: int,
    augmentations: int,
    seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    import numpy as np
    import torch

    from neuro_co.core.algos.multistart import distinct_first_actions, replicate
    from neuro_co.core.augment import augment_state
    from neuro_co.core.env import get_dynamic_decoder_context

    base_batch = int(state.coords.shape[0])
    augmented = augment_state(state, augmentations)
    features = env.build_features(augmented)
    node_embs, graph_emb = model.encode(features)
    first_mask = env.pomo_first_mask(augmented)
    if int(first_mask.sum(dim=1).min().item()) < n_starts:
        raise QualityPilotError("evaluation requested more distinct starts than valid customers")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    first_actions = distinct_first_actions(first_mask, n_starts, generator=generator)
    expanded = replicate(augmented, n_starts)
    node_embs = node_embs.repeat_interleave(n_starts, dim=0)
    graph_emb = graph_emb.repeat_interleave(n_starts, dim=0)
    decoder_cache = model.precompute_decoder_cache(node_embs)
    candidate_count = int(first_actions.shape[0])
    done_acc = torch.zeros(candidate_count, dtype=torch.bool, device=features.device)
    final_reward = torch.zeros(candidate_count, dtype=features.dtype, device=features.device)
    actions: list[Any] = []
    active_rows: list[Any] = []

    active_rows.append((~done_acc).clone())
    actions.append(first_actions)
    expanded, reward, done = env.step(expanded, first_actions)
    final_reward = final_reward + reward
    done_acc = done_acc | done

    for _ in range(env.max_steps(expanded) - 1):
        active = ~done_acc
        mask = env.action_mask(expanded)
        first_idx, current_idx = env.decoder_context(expanded)
        logits = model.decode_step(
            node_embs,
            graph_emb,
            first_idx,
            current_idx,
            mask,
            dynamic_context=get_dynamic_decoder_context(env, expanded),
            decoder_cache=decoder_cache,
        )
        action = logits.argmax(dim=-1)
        actions.append(action)
        active_rows.append(active.clone())
        expanded, reward, done = env.step(expanded, action)
        final_reward = final_reward + reward * active.to(reward.dtype)
        done_acc = done_acc | done
        if bool(done_acc.all()):
            break
    if not bool(done_acc.all()):
        raise QualityPilotError("multistart evaluation exceeded the environment step bound")

    action_matrix = torch.stack(actions, dim=1).detach().cpu().numpy()
    active_matrix = torch.stack(active_rows, dim=1).detach().cpu().numpy()
    candidate_routes = _routes_from_actions(action_matrix, active_matrix)
    reward_grid = (
        final_reward.view(augmentations, base_batch, n_starts)
        .permute(1, 0, 2)
        .reshape(base_batch, augmentations * n_starts)
    )
    choices = reward_grid.argmax(dim=1).detach().cpu().numpy()
    selected: list[tuple[tuple[int, ...], ...]] = []
    for batch_index, choice in enumerate(choices):
        aug_index, start_index = np.unravel_index(int(choice), (augmentations, n_starts))
        flat_index = (aug_index * base_batch + batch_index) * n_starts + start_index
        selected.append(candidate_routes[flat_index])
    return tuple(selected)


def _evaluate_routes(
    recipe: AETQualityRecipe,
    checkpoint_path: Path,
    corpus: Corpus,
    mode: EvaluationMode,
    *,
    eval_seed: int,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    import torch

    from neuro_co.core.factory import make_env

    checkpoint = _load_torch_mapping(checkpoint_path)
    device = torch.device(f"cuda:{recipe.gpu_index}")
    env = make_env(
        recipe.dataset.problem,
        size=recipe.dataset.size,
        capacity=recipe.dataset.capacity,
        max_demand=recipe.dataset.max_demand,
    )
    model = _make_mlco_am(recipe, env)
    if checkpoint.get("model_identity") != _model_identity(recipe, model):
        raise QualityPilotError("checkpoint MLCO AM identity disagrees with this pilot")
    model.load_state_dict(checkpoint["model"], strict=True)
    if mode.inference_precision != "fp32":
        raise QualityPilotError(f"unsupported inference precision: {mode.inference_precision}")
    model.to(device=device, dtype=torch.float32).eval()
    all_routes: list[tuple[tuple[int, ...], ...]] = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, corpus.coords.shape[0], mode.batch_size):
            stop = min(start + mode.batch_size, corpus.coords.shape[0])
            state = _initial_state(
                env,
                corpus.coords[start:stop],
                corpus.demands[start:stop],
                device,
            )
            if mode.n_starts == 1:
                if mode.augmentations != 1 or mode.forced_first_actions:
                    raise QualityPilotError("greedy mode has inconsistent policy fields")
                routes = _greedy_routes(model, env, state)
            else:
                routes = _multistart_routes(
                    model,
                    env,
                    state,
                    n_starts=mode.n_starts,
                    augmentations=mode.augmentations,
                    seed=eval_seed + start,
                )
            all_routes.extend(routes)
    return tuple(all_routes)


def _gap_summary(costs: Any, reference_costs: Any) -> dict[str, Any]:
    import numpy as np

    candidate = np.asarray(costs, dtype=np.float64)
    reference = np.asarray(reference_costs, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1 or candidate.size < 1:
        raise QualityPilotError("candidate and reference costs must be equal non-empty vectors")
    if not np.isfinite(candidate).all() or not np.isfinite(reference).all():
        raise QualityPilotError("quality vectors contain a non-finite value")
    if np.any(reference <= 0):
        raise QualityPilotError("reference costs must be strictly positive")
    gaps = 100.0 * (candidate - reference) / reference
    return {
        "mean_cost": float(candidate.mean()),
        "reference_mean_cost": float(reference.mean()),
        "mean_gap_pct": float(gaps.mean()),
        "median_gap_pct": float(np.median(gaps)),
        "p95_gap_pct": float(np.percentile(gaps, 95)),
        "maximum_gap_pct": float(gaps.max()),
        "minimum_gap_pct": float(gaps.min()),
        "gaps_pct": gaps.tolist(),
    }


def _revalidate_quality_metrics(
    result: dict[str, Any], reference: dict[str, Any], label: str
) -> None:
    recomputed_gap = _gap_summary(result["validation"]["costs"], reference["costs"])
    for key in (
        "mean_cost",
        "reference_mean_cost",
        "mean_gap_pct",
        "median_gap_pct",
        "p95_gap_pct",
        "maximum_gap_pct",
        "minimum_gap_pct",
    ):
        if not math.isclose(
            float(recomputed_gap[key]),
            float(result.get("quality", {}).get(key, math.nan)),
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise QualityPilotError(f"stored quality metric changed: {label}")
    try:
        gaps_match = all(
            math.isclose(float(actual), float(stored), rel_tol=0.0, abs_tol=1e-10)
            for actual, stored in zip(
                recomputed_gap["gaps_pct"],
                result.get("quality", {}).get("gaps_pct", ()),
                strict=True,
            )
        )
    except (TypeError, ValueError):
        gaps_match = False
    if not gaps_match:
        raise QualityPilotError(f"stored per-instance quality gaps changed: {label}")


def _evaluate_one(
    recipe: AETQualityRecipe,
    *,
    output: Path,
    run_state: dict[str, Any],
    checkpoint_record: dict[str, Any],
    corpus: Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
    mode: EvaluationMode,
    result_path: Path,
) -> dict[str, Any]:
    if result_path.exists():
        result = _load_json(result_path)
        expected = {
            "schema_version": EVALUATION_SCHEMA,
            "classification": _quality_classification(),
            "split": corpus.split,
            "epoch": checkpoint_record["epoch"],
            "checkpoint_path": checkpoint_record["path"],
            "checkpoint_sha256": checkpoint_record["sha256"],
            "dataset_content_sha256": corpus.content_sha256,
            "reference_sha256": reference_sha256,
            "mode": asdict(mode),
            "recipe_sha256": run_state["recipe_sha256"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise QualityPilotError(f"existing evaluation disagrees with inputs: {result_path}")
        _revalidate_stored_routes(
            result,
            corpus,
            str(result_path),
            require_complete=False,
        )
        _revalidate_quality_metrics(result, reference, str(result_path))
        return result

    checkpoint_path = output / checkpoint_record["path"]
    if _sha256_file(checkpoint_path) != checkpoint_record["sha256"]:
        raise QualityPilotError(f"checkpoint changed before evaluation: {checkpoint_path}")
    started = time.perf_counter()
    routes = _evaluate_routes(
        recipe,
        checkpoint_path,
        corpus,
        mode,
        eval_seed=50_000 + int(checkpoint_record["epoch"]) * 100,
    )
    elapsed_s = time.perf_counter() - started
    validation = validate_routes(
        corpus.coords,
        corpus.demands,
        corpus.capacity,
        routes,
    )
    gap = _gap_summary(validation["costs"], reference["costs"])
    result = {
        "schema_version": EVALUATION_SCHEMA,
        "classification": _quality_classification(),
        "split": corpus.split,
        "epoch": checkpoint_record["epoch"],
        "checkpoint_path": checkpoint_record["path"],
        "checkpoint_sha256": checkpoint_record["sha256"],
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "mode": asdict(mode),
        "elapsed_s": elapsed_s,
        "routes": routes,
        "validation": validation,
        "quality": gap,
        "recipe_sha256": run_state["recipe_sha256"],
    }
    _atomic_write_json(result_path, result)
    return result


def _select_checkpoints(
    recipe: AETQualityRecipe,
    *,
    output: Path,
    run_state: dict[str, Any],
    training_result: dict[str, Any],
    corpus: Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    all_results: dict[str, list[dict[str, Any]]] = {}
    for mode in recipe.evaluation.modes:
        mode_results: list[dict[str, Any]] = []
        for checkpoint in training_result["checkpoint_records"]:
            epoch = int(checkpoint["epoch"])
            result_path = (
                output / "evaluation" / "selection" / mode.mode_id / f"epoch-{epoch:03d}.json"
            )
            result = _evaluate_one(
                recipe,
                output=output,
                run_state=run_state,
                checkpoint_record=checkpoint,
                corpus=corpus,
                reference=reference,
                reference_sha256=reference_sha256,
                mode=mode,
                result_path=result_path,
            )
            mode_results.append(result)
            print(
                json.dumps(
                    {
                        "stage": "selection-evaluation",
                        "mode": mode.mode_id,
                        "epoch": epoch,
                        "mean_gap_pct": result["quality"]["mean_gap_pct"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        def selection_key(value: dict[str, Any]) -> tuple[bool, float, int]:
            mean_cost = float(value["quality"]["mean_cost"])
            feasible = bool(
                value["validation"]["invalid_instance_count"]
                <= recipe.quality_gate.maximum_invalid_instances
                and math.isfinite(mean_cost)
            )
            return (
                not feasible,
                mean_cost if math.isfinite(mean_cost) else math.inf,
                value["epoch"],
            )

        selected = min(mode_results, key=selection_key)
        selection_invalid_instances = int(selected["validation"]["invalid_instance_count"])
        selection_finite = math.isfinite(float(selected["quality"]["mean_cost"]))
        selection_feasible = bool(
            selection_finite
            and selection_invalid_instances <= recipe.quality_gate.maximum_invalid_instances
        )
        selections[mode.mode_id] = {
            "epoch": selected["epoch"],
            "checkpoint_path": selected["checkpoint_path"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_mean_cost": selected["quality"]["mean_cost"],
            "selection_mean_gap_pct": selected["quality"]["mean_gap_pct"],
            "selection_finite": selection_finite,
            "selection_invalid_instances": selection_invalid_instances,
            "selection_feasible": selection_feasible,
        }
        all_results[mode.mode_id] = [
            {
                "epoch": result["epoch"],
                "mean_cost": result["quality"]["mean_cost"],
                "mean_gap_pct": result["quality"]["mean_gap_pct"],
                "path": (
                    Path("evaluation")
                    / "selection"
                    / mode.mode_id
                    / f"epoch-{int(result['epoch']):03d}.json"
                ).as_posix(),
            }
            for result in mode_results
        ]
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "status": "complete",
        "split": "selection",
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "selection_rule": recipe.evaluation.selection_rule,
        "policy_specific_checkpoint_selection": True,
        "selections": selections,
        "all_results": all_results,
        "holdout_was_not_used": True,
    }
    _atomic_write_json(output / "evaluation" / "checkpoint-selection.json", selection)
    return selection


def _apply_holdout_gate(
    recipe: AETQualityRecipe,
    *,
    output: Path,
    run_state: dict[str, Any],
    training_result: dict[str, Any],
    selection: dict[str, Any],
    corpus: Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    records_by_epoch = {
        int(record["epoch"]): record for record in training_result["checkpoint_records"]
    }
    mode_results: dict[str, Any] = {}
    for mode in recipe.evaluation.modes:
        selected = selection["selections"][mode.mode_id]
        checkpoint = records_by_epoch[int(selected["epoch"])]
        if (
            selected.get("checkpoint_path") != checkpoint["path"]
            or selected.get("checkpoint_sha256") != checkpoint["sha256"]
        ):
            raise QualityPilotError(
                f"selection for {mode.mode_id} disagrees with the declared checkpoint"
            )
        result_path = output / "evaluation" / "holdout" / f"{mode.mode_id}.json"
        result = _evaluate_one(
            recipe,
            output=output,
            run_state=run_state,
            checkpoint_record=checkpoint,
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
            mode=mode,
            result_path=result_path,
        )
        values = [
            result["quality"][key]
            for key in (
                "mean_cost",
                "reference_mean_cost",
                "mean_gap_pct",
                "median_gap_pct",
                "p95_gap_pct",
                "maximum_gap_pct",
                "minimum_gap_pct",
            )
        ]
        finite = all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
        invalid_instances = int(result["validation"]["invalid_instance_count"])
        selection_feasible = selected.get("selection_feasible") is True
        passed = bool(
            selection_feasible
            and (finite or not recipe.quality_gate.require_finite)
            and invalid_instances <= recipe.quality_gate.maximum_invalid_instances
            and result["quality"]["mean_gap_pct"] <= recipe.quality_gate.maximum_mean_gap_pct
        )
        mode_results[mode.mode_id] = {
            "passed": passed,
            "selected_epoch": selected["epoch"],
            "checkpoint_sha256": selected["checkpoint_sha256"],
            "selection_feasible": selection_feasible,
            "selection_invalid_instances": selected.get("selection_invalid_instances"),
            "finite": finite,
            "invalid_instances": invalid_instances,
            "maximum_invalid_instances": recipe.quality_gate.maximum_invalid_instances,
            "mean_gap_pct": result["quality"]["mean_gap_pct"],
            "maximum_mean_gap_pct": recipe.quality_gate.maximum_mean_gap_pct,
            "result_path": result_path.relative_to(output).as_posix(),
            "result_sha256": _sha256_file(result_path),
        }
        print(
            json.dumps(
                {
                    "stage": "holdout-gate",
                    "mode": mode.mode_id,
                    "selected_epoch": selected["epoch"],
                    "mean_gap_pct": result["quality"]["mean_gap_pct"],
                    "passed": passed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    passing_modes = sorted(mode for mode, result in mode_results.items() if result["passed"])
    status = "complete_quality_gate_passed" if passing_modes else "complete_quality_gate_failed"
    gate = {
        "schema_version": GATE_SCHEMA,
        "status": status,
        "metric": recipe.quality_gate.metric,
        "split": "holdout",
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "selection_and_holdout_are_disjoint": True,
        "mode_results": mode_results,
        "passing_modes": passing_modes,
        "at_least_one_mode_passed": bool(passing_modes),
        "assessment_scope": {
            "kind": "exploratory_multi_mode_holdout_screening",
            "confirmatory": False,
            "multiple_holdout_modes_screened": True,
            "multiplicity_adjusted": False,
            "historical_reproduction": False,
            "note": (
                "The multi-mode holdout gate is exploratory screening, not confirmatory "
                "evidence, and this MLCO pilot is not a reproduction of the historical pre-port run."
            ),
        },
        "aet_calculation_permitted_by_this_artifact": False,
        "next_step": (
            "replicate passing policies across additional training seeds before confirmatory evaluation"
            if passing_modes
            else (
                "stop before any energy campaign, revise the MLCO training and evaluation "
                "protocol, then rerun this screening pilot"
            )
        ),
    }
    _atomic_write_json(output / "quality-gate.json", gate)
    return gate


def _revalidate_final_inputs(
    recipe: AETQualityRecipe,
    *,
    workspace_root: Path,
    source_recipe: Path,
    output: Path,
    run_state: dict[str, Any],
    corpora: dict[str, Corpus],
    references: dict[str, dict[str, Any]],
    reference_lock: dict[str, Any],
    training_result: dict[str, Any],
) -> None:
    """Recheck long-lived inputs immediately before sealing the bundle."""

    if _source_snapshot(workspace_root)["sha256"] != run_state["source_snapshot"]["sha256"]:
        _invalidate_active_run(output, run_state, "quality-pilot source changed during execution")
    if (
        not _file_matches_sha256(source_recipe, run_state["recipe_sha256"])
        or not _file_matches_sha256(output / "recipe.yaml", run_state["recipe_sha256"])
        or not _file_matches_sha256(output / "environment" / "uv.lock", run_state["uv_lock_sha256"])
    ):
        _invalidate_active_run(output, run_state, "recipe or lockfile changed during execution")

    specs = {
        "selection": recipe.dataset.selection,
        "holdout": recipe.dataset.holdout,
    }
    for split, original in corpora.items():
        current = _load_corpus(recipe, split, specs[split], original.path)
        if (
            current.content_sha256 != original.content_sha256
            or current.file_sha256 != original.file_sha256
        ):
            raise QualityPilotError(f"{split} corpus changed during pilot execution")

    lock_path = output / "reference" / "reference-lock.json"
    if (
        _load_json(lock_path) != reference_lock
        or run_state.get("reference_locked_before_neural_artifacts") is not True
        or run_state.get("reference_lock_sha256") != _sha256_file(lock_path)
    ):
        raise QualityPilotError("reference lock changed during pilot execution")
    for split, original in references.items():
        entry = reference_lock["entries"][split]
        expected_path = f"reference/{split}/reference.json"
        if entry.get("path") != expected_path:
            raise QualityPilotError(f"reference lock path changed for {split}")
        path = output / expected_path
        if _sha256_file(path) != entry.get("sha256"):
            raise QualityPilotError(f"locked {split} reference changed during pilot execution")
        current = _load_json(path)
        if current != original:
            raise QualityPilotError(f"{split} reference payload changed during pilot execution")
        _revalidate_stored_routes(current, corpora[split], str(path))
        for candidate in current.get("candidate_artifacts", ()):
            candidate_path = _relative(output, candidate["path"])
            if not candidate_path.is_file() or _sha256_file(candidate_path) != candidate.get(
                "sha256"
            ):
                raise QualityPilotError(
                    f"{split} reference candidate changed during pilot execution"
                )

    for checkpoint in training_result["checkpoint_records"]:
        path = _relative(output, checkpoint["path"])
        if _sha256_file(path) != checkpoint["sha256"]:
            raise QualityPilotError(f"checkpoint changed during evaluation: {path}")


def _write_checksums(root: Path) -> Path:
    partials = [path.relative_to(root).as_posix() for path in _stale_partial_artifacts(root)]
    if partials:
        raise QualityPilotError(f"atomic partial artifacts remain before finalization: {partials}")
    checksum_path = root / "SHA256SUMS"
    entries: list[str] = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == checksum_path or path == root / "run-state.json":
            continue
        entries.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    _atomic_write_bytes(checksum_path, ("\n".join(entries) + "\n").encode("utf-8"))
    for entry in entries:
        expected, relative = entry.split("  ", maxsplit=1)
        if _sha256_file(root / relative) != expected:
            raise QualityPilotError(f"checksum verification failed: {relative}")
    return checksum_path


def _verify_checksums(root: Path, *, expected_sha256: str | None = None) -> None:
    checksum_path = root / "SHA256SUMS"
    try:
        checksum_bytes = checksum_path.read_bytes()
        lines = checksum_bytes.decode("utf-8").splitlines()
    except OSError as exc:
        raise QualityPilotError("completed pilot has no readable SHA256SUMS") from exc
    except UnicodeDecodeError as exc:
        raise QualityPilotError("completed pilot has non-UTF-8 checksums") from exc
    if expected_sha256 is not None and _sha256_bytes(checksum_bytes) != expected_sha256:
        raise QualityPilotError("completed pilot checksum inventory hash changed")
    if not lines:
        raise QualityPilotError("completed pilot has an empty SHA256SUMS")
    expected_inventory: set[str] = set()
    for line in lines:
        try:
            expected, relative = line.split("  ", maxsplit=1)
        except ValueError as exc:
            raise QualityPilotError("completed pilot has malformed checksums") from exc
        relative_path = PurePosixPath(relative)
        if (
            relative_path.is_absolute()
            or not relative_path.parts
            or any(part in {"", ".", ".."} for part in relative_path.parts)
            or relative in expected_inventory
        ):
            raise QualityPilotError("completed pilot has unsafe or duplicate checksums")
        expected_inventory.add(relative)
        path = _relative(root, relative)
        if len(expected) != 64 or not path.is_file() or _sha256_file(path) != expected:
            raise QualityPilotError(f"completed pilot checksum mismatch: {relative}")
    actual_inventory = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path not in {checksum_path, root / "run-state.json"}
    }
    if actual_inventory != expected_inventory:
        missing = sorted(expected_inventory - actual_inventory)
        extra = sorted(actual_inventory - expected_inventory)
        raise QualityPilotError(
            f"completed pilot artifact inventory changed; missing={missing}, extra={extra}"
        )


def _validate_completed_bundle(
    recipe: AETQualityRecipe,
    output: Path,
    run_state: dict[str, Any],
) -> tuple[Path, str, tuple[str, ...], str]:
    """Validate semantic links inside an already checksummed result bundle."""

    manifest_path = output / "manifest.json"
    manifest = _load_json(manifest_path)
    gate = _load_json(output / "quality-gate.json")
    selection = _load_json(output / "evaluation" / "checkpoint-selection.json")
    training = _load_json(output / "training" / "result.json")
    expected_mode_ids = {mode.mode_id for mode in recipe.evaluation.modes}
    mode_by_id = {mode.mode_id: mode for mode in recipe.evaluation.modes}
    expected_manifest_classification = {
        **_quality_classification(),
        "exclusive_access_required": False,
    }

    if (
        run_state.get("classification") != _quality_classification()
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("classification") != expected_manifest_classification
        or gate.get("schema_version") != GATE_SCHEMA
        or selection.get("schema_version") != SELECTION_SCHEMA
        or manifest.get("recipe", {}).get("sha256") != run_state["recipe_sha256"]
        or manifest.get("source", {}).get("uv_lock_sha256") != run_state["uv_lock_sha256"]
        or manifest.get("source", {}).get("scoped_source_snapshot") != run_state["source_snapshot"]
        or manifest.get("execution_invocations") != run_state.get("invocations")
        or manifest.get("training") != training
        or manifest.get("checkpoint_selection") != selection
        or manifest.get("quality_gate") != gate
    ):
        raise QualityPilotError("completed pilot manifest relationships are inconsistent")
    invocations = run_state.get("invocations")
    if (
        not isinstance(invocations, list)
        or not invocations
        or not isinstance(invocations[-1], dict)
        or invocations[-1].get("runtime_identity") != run_state.get("runtime_identity")
        or manifest.get("execution_qualification") != invocations[-1].get("qualification")
    ):
        raise QualityPilotError("completed pilot runtime history is inconsistent")

    mode_results = gate.get("mode_results")
    selections = selection.get("selections")
    if (
        not isinstance(mode_results, dict)
        or set(mode_results) != expected_mode_ids
        or not isinstance(selections, dict)
        or set(selections) != expected_mode_ids
    ):
        raise QualityPilotError("completed pilot mode inventory is inconsistent")
    if selection.get("status") != "complete" or selection.get("split") != "selection":
        raise QualityPilotError("completed pilot checkpoint selection is inconsistent")
    computed_passing = sorted(
        mode_id
        for mode_id, result in mode_results.items()
        if isinstance(result, dict) and result.get("passed") is True
    )
    passing_modes = gate.get("passing_modes")
    if passing_modes != computed_passing:
        raise QualityPilotError("completed pilot passing-mode inventory is inconsistent")
    status = "complete_quality_gate_passed" if passing_modes else "complete_quality_gate_failed"
    if manifest.get("status") != status or gate.get("status") != status:
        raise QualityPilotError("completed pilot quality-gate status is inconsistent")

    checkpoint_records = training.get("checkpoint_records")
    if not isinstance(checkpoint_records, list):
        raise QualityPilotError("completed pilot checkpoint inventory is invalid")
    records_by_epoch: dict[int, dict[str, Any]] = {}
    for record in checkpoint_records:
        if (
            not isinstance(record, dict)
            or isinstance(record.get("epoch"), bool)
            or not isinstance(record.get("epoch"), int)
        ):
            raise QualityPilotError("completed pilot checkpoint record is invalid")
        epoch = record["epoch"]
        expected_path = f"training/checkpoints/epoch-{epoch:03d}.pt"
        expected_items = epoch * recipe.training.instances_per_epoch
        expected_steps = epoch * math.ceil(
            recipe.training.instances_per_epoch / recipe.training.batch_size
        )
        path = _relative(output, expected_path)
        if (
            epoch in records_by_epoch
            or record.get("path") != expected_path
            or record.get("items_processed") != expected_items
            or record.get("optimizer_steps") != expected_steps
            or not path.is_file()
            or _sha256_file(path) != record.get("sha256")
        ):
            raise QualityPilotError("completed pilot checkpoint record is inconsistent")
        records_by_epoch[epoch] = record
    if set(records_by_epoch) != set(recipe.training.checkpoint_epochs):
        raise QualityPilotError("completed pilot checkpoint epochs disagree with the recipe")

    corpora: dict[str, Corpus] = {}
    dataset_manifest = manifest.get("dataset")
    if not isinstance(dataset_manifest, dict):
        raise QualityPilotError("completed pilot dataset manifest is invalid")
    for split, spec in (
        ("selection", recipe.dataset.selection),
        ("holdout", recipe.dataset.holdout),
    ):
        corpus_path = _relative(output, spec.artifact)
        side_manifest = corpus_path.with_suffix(".manifest.json")
        if not corpus_path.is_file() or not side_manifest.is_file():
            raise QualityPilotError(f"completed pilot {split} corpus artifacts are missing")
        corpus = _load_corpus(recipe, split, spec, corpus_path)
        corpora[split] = corpus
        if dataset_manifest.get(split) != {
            "path": spec.artifact,
            "file_sha256": corpus.file_sha256,
            "content_sha256": corpus.content_sha256,
            "instances": spec.num_instances,
        }:
            raise QualityPilotError(f"completed pilot {split} corpus manifest is inconsistent")

    lock_path = output / "reference" / "reference-lock.json"
    lock = _load_json(lock_path)
    manifest_lock = manifest.get("reference_lock")
    if (
        lock.get("schema_version") != REFERENCE_LOCK_SCHEMA
        or lock.get("status") != "locked"
        or lock.get("locked_before_neural_evaluation") is not True
        or lock.get("policy") != recipe.reference.policy
        or lock.get("solver_families") != ["hybrid-genetic-search", "ortools-routing"]
        or not isinstance(manifest_lock, dict)
        or manifest_lock.get("path") != "reference/reference-lock.json"
        or manifest_lock.get("sha256") != _sha256_file(lock_path)
        or manifest_lock.get("anchored_in_run_state_before_neural_artifacts") is not True
        or run_state.get("reference_locked_before_neural_artifacts") is not True
        or run_state.get("reference_lock_sha256") != _sha256_file(lock_path)
        or manifest_lock.get("entries") != lock.get("entries")
    ):
        raise QualityPilotError("completed pilot reference lock is inconsistent")
    reference_hashes: dict[str, str] = {}
    reference_payloads: dict[str, dict[str, Any]] = {}
    lock_entries = lock.get("entries")
    if not isinstance(lock_entries, dict):
        raise QualityPilotError("completed pilot reference lock entries are invalid")
    for split in ("selection", "holdout"):
        reference_path = output / "reference" / split / "reference.json"
        entry = lock_entries.get(split)
        if (
            not isinstance(entry, dict)
            or entry.get("path") != f"reference/{split}/reference.json"
            or entry.get("dataset_content_sha256") != corpora[split].content_sha256
            or not reference_path.is_file()
            or entry.get("sha256") != _sha256_file(reference_path)
        ):
            raise QualityPilotError(f"completed pilot {split} reference lock is inconsistent")
        reference = _load_json(reference_path)
        if (
            reference.get("schema_version") != REFERENCE_SCHEMA
            or reference.get("status") != "complete"
            or reference.get("split") != split
            or reference.get("dataset_content_sha256") != corpora[split].content_sha256
            or reference.get("policy") != recipe.reference.policy
            or reference.get("independent_solver_families")
            != ["hybrid-genetic-search", "ortools-routing"]
            or reference.get("candidate_count") != len(recipe.reference.hgs.seeds) + 1
            or reference.get("future_timed_baseline_is_separate") is not True
            or reference.get("energy_measurement") != "none"
        ):
            raise QualityPilotError(f"completed pilot {split} reference is inconsistent")
        _revalidate_stored_routes(reference, corpora[split], str(reference_path))
        for candidate in reference.get("candidate_artifacts", ()):
            if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
                raise QualityPilotError(
                    f"completed pilot {split} reference candidate inventory is invalid"
                )
            candidate_path = _relative(output, candidate["path"])
            if not candidate_path.is_file() or candidate.get("sha256") != _sha256_file(
                candidate_path
            ):
                raise QualityPilotError(
                    f"completed pilot {split} reference candidate is inconsistent"
                )
        reference_hashes[split] = entry["sha256"]
        reference_payloads[split] = reference

    if (
        selection.get("dataset_content_sha256") != corpora["selection"].content_sha256
        or selection.get("reference_sha256") != reference_hashes["selection"]
        or gate.get("dataset_content_sha256") != corpora["holdout"].content_sha256
        or gate.get("reference_sha256") != reference_hashes["holdout"]
    ):
        raise QualityPilotError("completed pilot evaluation inputs are inconsistent")

    for mode_id in sorted(expected_mode_ids):
        selected = selections[mode_id]
        if (
            not isinstance(selected, dict)
            or isinstance(selected.get("epoch"), bool)
            or not isinstance(selected.get("epoch"), int)
        ):
            raise QualityPilotError(f"completed pilot selection is invalid for {mode_id}")
        checkpoint = records_by_epoch.get(selected["epoch"])
        if (
            checkpoint is None
            or selected.get("checkpoint_path") != checkpoint["path"]
            or selected.get("checkpoint_sha256") != checkpoint["sha256"]
        ):
            raise QualityPilotError(f"completed pilot selection is inconsistent for {mode_id}")
        selection_mean_cost = selected.get("selection_mean_cost")
        selection_invalid_instances = selected.get("selection_invalid_instances")
        selection_finite = bool(
            isinstance(selection_mean_cost, (int, float))
            and not isinstance(selection_mean_cost, bool)
            and math.isfinite(selection_mean_cost)
        )
        selection_feasible = bool(
            selection_finite
            and isinstance(selection_invalid_instances, int)
            and not isinstance(selection_invalid_instances, bool)
            and selection_invalid_instances <= recipe.quality_gate.maximum_invalid_instances
        )
        if (
            selected.get("selection_finite") is not selection_finite
            or selected.get("selection_feasible") is not selection_feasible
        ):
            raise QualityPilotError(
                f"completed pilot selection feasibility is inconsistent for {mode_id}"
            )
        gate_result = mode_results[mode_id]
        expected_result_path = f"evaluation/holdout/{mode_id}.json"
        result_path = _relative(output, expected_result_path)
        if (
            not isinstance(gate_result, dict)
            or gate_result.get("result_path") != expected_result_path
            or gate_result.get("checkpoint_sha256") != checkpoint["sha256"]
            or not result_path.is_file()
            or gate_result.get("result_sha256") != _sha256_file(result_path)
        ):
            raise QualityPilotError(f"completed pilot holdout gate is inconsistent for {mode_id}")
        result = _load_json(result_path)
        expected_result = {
            "schema_version": EVALUATION_SCHEMA,
            "classification": _quality_classification(),
            "split": "holdout",
            "epoch": checkpoint["epoch"],
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "dataset_content_sha256": corpora["holdout"].content_sha256,
            "reference_sha256": reference_hashes["holdout"],
            "mode": asdict(mode_by_id[mode_id]),
            "recipe_sha256": run_state["recipe_sha256"],
        }
        if any(result.get(key) != value for key, value in expected_result.items()):
            raise QualityPilotError(
                f"completed pilot holdout result metadata is inconsistent for {mode_id}"
            )
        _revalidate_stored_routes(
            result,
            corpora["holdout"],
            str(result_path),
            require_complete=False,
        )
        _revalidate_quality_metrics(result, reference_payloads["holdout"], str(result_path))
        quality_values = [
            result["quality"][key]
            for key in (
                "mean_cost",
                "reference_mean_cost",
                "mean_gap_pct",
                "median_gap_pct",
                "p95_gap_pct",
                "maximum_gap_pct",
                "minimum_gap_pct",
            )
        ]
        finite = all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            for value in quality_values
        )
        invalid_instances = int(result["validation"]["invalid_instance_count"])
        mean_gap_pct = result["quality"]["mean_gap_pct"]
        passed = bool(
            selection_feasible
            and (finite or not recipe.quality_gate.require_finite)
            and invalid_instances <= recipe.quality_gate.maximum_invalid_instances
            and mean_gap_pct <= recipe.quality_gate.maximum_mean_gap_pct
        )
        expected_gate_fields = {
            "passed": passed,
            "selected_epoch": checkpoint["epoch"],
            "checkpoint_sha256": checkpoint["sha256"],
            "selection_feasible": selection_feasible,
            "selection_invalid_instances": selection_invalid_instances,
            "finite": finite,
            "invalid_instances": invalid_instances,
            "maximum_invalid_instances": recipe.quality_gate.maximum_invalid_instances,
            "mean_gap_pct": mean_gap_pct,
            "maximum_mean_gap_pct": recipe.quality_gate.maximum_mean_gap_pct,
            "result_path": expected_result_path,
            "result_sha256": _sha256_file(result_path),
        }
        if gate_result != expected_gate_fields:
            raise QualityPilotError(
                f"completed pilot holdout decision is inconsistent for {mode_id}"
            )

    manifest_sha256 = _sha256_file(manifest_path)
    if run_state.get("manifest_sha256") != manifest_sha256:
        raise QualityPilotError("completed pilot manifest hash disagrees with run state")
    return manifest_path, manifest_sha256, tuple(computed_passing), status


def _execute_quality_pilot_unlocked(
    source_recipe: Path,
    recipe: AETQualityRecipe,
    qualification: dict[str, Any],
    runtime_identity: dict[str, Any],
    root: Path,
    recipe_bytes: bytes,
    *,
    resume: bool = False,
) -> PilotResult:
    output, run_state = _prepare_output(
        source_recipe,
        recipe,
        root,
        resume=resume,
        recipe_bytes=recipe_bytes,
        runtime_identity=runtime_identity,
    )
    if run_state.get("status") in _COMPLETE_GATE_STATUSES:
        checksum_sha256 = run_state.get("checksums_sha256")
        if not isinstance(checksum_sha256, str) or len(checksum_sha256) != 64:
            raise QualityPilotError("completed pilot has no anchored checksum inventory")
        _verify_checksums(output, expected_sha256=checksum_sha256)
        manifest_path, manifest_sha256, passing_modes, expected_status = _validate_completed_bundle(
            recipe, output, run_state
        )
        if run_state.get("status") != expected_status:
            raise QualityPilotError("completed pilot quality-gate status is inconsistent")
        return PilotResult(
            path=output,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            passing_modes=passing_modes,
            status=expected_status,
        )
    _record_invocation(output, run_state, qualification, runtime_identity)
    runtime_controls = _configure_runtime(recipe.gpu_index)
    started_at = datetime.now(UTC)
    corpora = _prepare_corpora(recipe, output)
    neural_artifacts_exist = (output / "training" / "latest.pt").exists() or (
        output / "evaluation"
    ).exists()
    if run_state.get("reference_lock_sha256") is not None:
        _verify_anchored_reference_tree(output, run_state, corpora)
    elif neural_artifacts_exist:
        raise QualityPilotError(
            "neural artifacts exist without a pre-training reference-lock anchor"
        )
    references, reference_lock = _prepare_references(recipe, corpora, output)
    reference_lock_path = output / "reference" / "reference-lock.json"
    reference_lock_sha256 = _sha256_file(reference_lock_path)
    if run_state.get("reference_lock_sha256") is None:
        run_state["reference_lock_sha256"] = reference_lock_sha256
        run_state["reference_locked_before_neural_artifacts"] = True
        _atomic_write_json(output / "run-state.json", run_state)
    elif run_state["reference_lock_sha256"] != reference_lock_sha256:
        raise QualityPilotError("reference lock changed after its pre-training anchor")
    reference_hashes = {
        split: reference_lock["entries"][split]["sha256"] for split in ("selection", "holdout")
    }
    for split in reference_hashes:
        if _sha256_file(output / "reference" / split / "reference.json") != reference_hashes[split]:
            raise QualityPilotError("reference changed after locking")

    training_result = _train(recipe, output, run_state)
    selection = _select_checkpoints(
        recipe,
        output=output,
        run_state=run_state,
        training_result=training_result,
        corpus=corpora["selection"],
        reference=references["selection"],
        reference_sha256=reference_hashes["selection"],
    )
    gate = _apply_holdout_gate(
        recipe,
        output=output,
        run_state=run_state,
        training_result=training_result,
        selection=selection,
        corpus=corpora["holdout"],
        reference=references["holdout"],
        reference_sha256=reference_hashes["holdout"],
    )
    _revalidate_final_inputs(
        recipe,
        workspace_root=root,
        source_recipe=source_recipe,
        output=output,
        run_state=run_state,
        corpora=corpora,
        references=references,
        reference_lock=reference_lock,
        training_result=training_result,
    )
    ended_at = datetime.now(UTC)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": gate["status"],
        "classification": {
            **_quality_classification(),
            "exclusive_access_required": False,
        },
        "started_at_this_invocation": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "recipe": {
            "path": "recipe.yaml",
            "sha256": run_state["recipe_sha256"],
            "name": recipe.name,
        },
        "source": {
            "git_sha": run_state["git_sha"],
            "git_at_initialization": run_state["git"],
            "uv_lock_sha256": run_state["uv_lock_sha256"],
            "scoped_source_snapshot": run_state["source_snapshot"],
        },
        "runtime_controls": runtime_controls,
        "execution_qualification": qualification,
        "execution_invocations": run_state["invocations"],
        "library_versions": _library_versions(),
        "dataset": {
            split: {
                "path": corpus.path.relative_to(output).as_posix(),
                "file_sha256": corpus.file_sha256,
                "content_sha256": corpus.content_sha256,
                "instances": int(corpus.coords.shape[0]),
            }
            for split, corpus in corpora.items()
        },
        "reference_lock": {
            "path": "reference/reference-lock.json",
            "sha256": _sha256_file(output / "reference" / "reference-lock.json"),
            "entries": reference_lock["entries"],
            "anchored_in_run_state_before_neural_artifacts": True,
        },
        "training": training_result,
        "checkpoint_selection": selection,
        "quality_gate": gate,
        "historical_mismatch_prevented": {
            "checkpoint_selection_policy_is_explicit": True,
            "each_inference_policy_selects_its_own_checkpoint": True,
            "greedy_and_multistart_augmented_policies_are_not_conflated": True,
        },
        "model_provenance": {
            "implementation": "MLCO AM",
            "historical_external_implementation": False,
            "historical_reproduction": False,
            "configuration": training_result["model_identity"]["configuration"],
            "parameter_count": training_result["model_identity"]["parameter_count"],
            "state_dict_structure_sha256": training_result["model_identity"][
                "state_dict_structure_sha256"
            ],
            "historical_reward_zscore_applied": False,
        },
        "evidence_scope": gate["assessment_scope"],
        "aet_was_computed": False,
    }
    manifest_path = output / "manifest.json"
    _atomic_write_json(manifest_path, manifest)
    checksum_path = _write_checksums(output)
    run_state["status"] = gate["status"]
    run_state["completed_at"] = ended_at.isoformat()
    run_state["manifest_sha256"] = _sha256_file(manifest_path)
    run_state["checksums_sha256"] = _sha256_file(checksum_path)
    _atomic_write_json(output / "run-state.json", run_state)
    _verify_checksums(output, expected_sha256=run_state["checksums_sha256"])
    validated_path, validated_sha256, passing_modes, validated_status = _validate_completed_bundle(
        recipe, output, run_state
    )
    if validated_status != gate["status"]:
        raise QualityPilotError("finalized quality-gate status changed during validation")
    return PilotResult(
        path=output,
        manifest_path=validated_path,
        manifest_sha256=validated_sha256,
        passing_modes=passing_modes,
        status=validated_status,
    )


def execute_quality_pilot(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    resume: bool = False,
) -> PilotResult:
    """Run or safely resume the complete quality-only pilot."""

    root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
    source_recipe = Path(recipe_path).resolve(strict=True)
    try:
        source_recipe.relative_to(root)
    except ValueError as exc:
        raise QualityPilotQualificationError("recipe must be stored inside the repository") from exc
    recipe_bytes = source_recipe.read_bytes()
    recipe = load_aet_quality_recipe(source_recipe)
    if source_recipe.read_bytes() != recipe_bytes:
        raise QualityPilotQualificationError("recipe changed while it was being loaded")
    qualification = runtime_qualification(recipe)
    if qualification.get("ready_to_execute") is not True:
        raise QualityPilotQualificationError(
            "quality pilot requires native Windows, CUDA, Git, PyVRP, and OR-Tools"
        )
    output = _safe_output_target(root, recipe.output_root)
    runtime_identity = _runtime_identity(recipe, qualification)
    with _output_lock(output):
        if source_recipe.read_bytes() != recipe_bytes:
            raise QualityPilotError("recipe changed before pilot initialization")
        return _execute_quality_pilot_unlocked(
            source_recipe,
            recipe,
            qualification,
            runtime_identity,
            root,
            recipe_bytes,
            resume=resume,
        )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute_quality_pilot(args.recipe, resume=args.resume)
    except QualityPilotQualificationError as exc:
        print(f"quality pilot not qualified: {exc}", file=sys.stderr, flush=True)
        return 2
    except QualityPilotError as exc:
        print(f"quality pilot failed: {exc}", file=sys.stderr, flush=True)
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
                "manifest": result.manifest_path.as_posix(),
                "manifest_sha256": result.manifest_sha256,
                "passing_modes": result.passing_modes,
                "aet_was_computed": False,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if result.passing_modes else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PilotResult",
    "QualityPilotError",
    "QualityPilotQualificationError",
    "execute_quality_pilot",
    "validate_routes",
]
