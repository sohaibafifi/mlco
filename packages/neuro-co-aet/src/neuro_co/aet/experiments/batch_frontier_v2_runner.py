"""Measure greedy AM/GNN batches against 16-worker HGS at 10 s per instance.

Every promoted JSON block is a durable schedule prefix.  Re-running with
``--resume`` continues at the first uncommitted block. Neural blocks repeat
the 512-instance corpus until 120 seconds have elapsed. An HGS block maps the
512 instances to 16 processes, with an independent ten-second stop criterion
and deterministic seed for every instance. The single CPU-package counter
brackets the whole pool and its aggregate energy is divided by 512 original
instances. Per-process energy is never summed.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from statistics import mean, median, stdev
from typing import Any, cast

from neuro_co.aet.experiments import batch_frontier_runner as legacy
from neuro_co.aet.experiments import deployment_energy_runner as energy_runner
from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_recipe
from neuro_co.aet.experiments.batch_frontier_v2_recipe import (
    ARCHITECTURES,
    CLASSIFICATION,
    TRAINING_SEEDS,
    BatchFrontierV2Recipe,
    FrontierBlock,
    expected_blocks,
    load_recipe,
    minimum_measured_walltime_s,
    qualify_for_execution,
)

RUN_STATE_SCHEMA = "aet-batch-frontier-v2-run-state/v1"
CAPACITY_SCHEMA = "aet-batch-frontier-v2-capacity-probe/v1"
BLOCK_SCHEMA = "aet-batch-frontier-v2-block/v1"
SUMMARY_SCHEMA = "aet-batch-frontier-v2-summary/v1"
MANIFEST_SCHEMA = "aet-batch-frontier-v2-manifest/v1"
ESTIMATE_SCHEMA = "aet-batch-frontier-v2-estimate/v1"
COMPLETE_STATUS = "complete"
INCOMPLETE_STATUS = "incomplete"
SUMMARY_FILENAME = "batch-frontier-v2-summary.json"
HGS_MEASUREMENT_BOUNDARY = "warm_process_pool_ready_before_tracker_through_all_512_results_returned"
HGS_ENERGY_NORMALIZATION = (
    "aggregate_cpu_package_energy_for_whole_pool_divided_by_512_original_instances"
)
_HGS_WARMUP_BARRIER: Any | None = None


class BatchFrontierV2Error(RuntimeError):
    """Raised when the corrected campaign cannot safely continue."""


class BatchFrontierV2QualificationError(BatchFrontierV2Error):
    """Raised before measurement when an immutable input is not qualified."""


@dataclass(frozen=True, slots=True)
class BatchFrontierV2Result:
    path: Path
    status: str
    completed_blocks: int
    manifest_path: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class IndexedHGSResult:
    """AET-local global index wrapped around one generic core solve result."""

    instance_index: int
    routes: tuple[tuple[int, ...], ...]
    integer_cost: int
    cost: float
    seed: int
    limit_kind: str
    max_iterations: int | None
    max_runtime_s: float | None
    scaling_factor: int
    worker_pid: int


def _hgs_worker_initializer(warmup_barrier: Any | None = None) -> None:
    """Prevent nested BLAS/OpenMP parallelism in every spawned HGS worker."""

    global _HGS_WARMUP_BARRIER
    _HGS_WARMUP_BARRIER = warmup_barrier
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"


def _solve_hgs_instance_task(task: tuple[Any, ...]) -> Any:
    """Solve one globally indexed instance with its own fresh MaxRuntime."""

    (
        instance_index,
        coords,
        demands,
        capacity,
        instance_seed,
        max_runtime_s,
        scaling_factor,
        collect_stats,
        synchronize_warmup_workers,
    ) = task
    if synchronize_warmup_workers:
        barrier = _HGS_WARMUP_BARRIER
        if barrier is None:
            raise RuntimeError("HGS warmup barrier is unavailable")
        barrier.wait(timeout=120)
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    result = solve_corpus_sequential(
        coords[None, ...],
        demands[None, ...],
        capacity,
        seed=instance_seed,
        max_runtime_s=max_runtime_s,
        scaling_factor=scaling_factor,
        collect_stats=collect_stats,
    )[0]
    return IndexedHGSResult(
        instance_index=int(instance_index),
        routes=tuple(tuple(int(customer) for customer in route) for route in result.routes),
        integer_cost=int(result.integer_cost),
        cost=float(result.cost),
        seed=int(result.seed),
        limit_kind=str(result.limit_kind),
        max_iterations=result.max_iterations,
        max_runtime_s=result.max_runtime_s,
        scaling_factor=int(result.scaling_factor),
        worker_pid=os.getpid(),
    )


def classification() -> dict[str, Any]:
    return dict(CLASSIFICATION)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, _json_bytes(payload))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BatchFrontierV2Error(f"cannot read JSON artifact {path}") from exc
    if not isinstance(payload, dict):
        raise BatchFrontierV2Error(f"JSON artifact is not an object: {path}")
    return payload


def _relative(root: Path, value: str) -> Path:
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise BatchFrontierV2Error(f"unsafe relative path: {value!r}")
    return root.joinpath(*parsed.parts)


def _sha_field(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BatchFrontierV2QualificationError(f"{where} is not a lowercase SHA-256")
    return value


def _safe_output(root: Path, value: str) -> Path:
    try:
        return legacy._safe_output(root, value)
    except Exception as exc:
        raise BatchFrontierV2Error(str(exc)) from exc


def _training_root(recipe: BatchFrontierV2Recipe, root: Path, override: Path | None) -> Path:
    candidate = override or _relative(root, recipe.training_source.root)
    candidate = candidate if candidate.is_absolute() else root / candidate
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise BatchFrontierV2QualificationError(
            "training source must remain inside the repository"
        ) from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise BatchFrontierV2QualificationError("training source is not a regular directory")
    return resolved


def _resolve_training_source(
    recipe: BatchFrontierV2Recipe,
    root: Path,
    override: Path | None,
) -> tuple[dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    """Bind the ten prospectively selected epoch-90-to-100 checkpoints."""

    source = _training_root(recipe, root, override)
    manifest_path = source / recipe.training_source.manifest_path
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema_version") != recipe.training_source.manifest_schema
        or manifest.get("status") != recipe.training_source.expected_status
    ):
        raise BatchFrontierV2QualificationError("training-v2 bundle is not complete")
    raw_entries = manifest.get("checkpoint_entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 10:
        raise BatchFrontierV2QualificationError(
            "training-v2 manifest must expose exactly ten selected checkpoints"
        )
    expected = {(architecture, seed) for architecture in ARCHITECTURES for seed in TRAINING_SEEDS}
    resolved: dict[tuple[str, int], dict[str, Any]] = {}
    models: list[dict[str, Any]] = []
    from neuro_co.aet.experiments.training_debt_runner import model_state_sha256

    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise BatchFrontierV2QualificationError("checkpoint entry is malformed")
        raw_architecture = raw.get("architecture")
        raw_seed = raw.get("training_seed")
        selected_epoch = raw.get("selected_epoch")
        if (
            not isinstance(raw_architecture, str)
            or isinstance(raw_seed, bool)
            or not isinstance(raw_seed, int)
            or isinstance(selected_epoch, bool)
            or not isinstance(selected_epoch, int)
            or not recipe.training_source.selected_epoch_min
            <= selected_epoch
            <= recipe.training_source.selected_epoch_max
        ):
            raise BatchFrontierV2QualificationError("selected checkpoint key or epoch is invalid")
        architecture = raw_architecture
        seed = raw_seed
        key = (architecture, seed)
        if key not in expected or key in resolved:
            raise BatchFrontierV2QualificationError("selected checkpoint key or epoch is invalid")
        relative = raw.get("path")
        if not isinstance(relative, str):
            raise BatchFrontierV2QualificationError("selected checkpoint path is invalid")
        checkpoint = _relative(source, relative)
        checkpoint_sha = _sha_field(raw.get("checkpoint_sha256"), "checkpoint_sha256")
        if (
            checkpoint.is_symlink()
            or not checkpoint.is_file()
            or _sha256_file(checkpoint) != checkpoint_sha
        ):
            raise BatchFrontierV2QualificationError(f"selected checkpoint changed: {relative}")
        identity = raw.get("model_identity")
        configuration = raw.get("model_configuration")
        backend = raw.get("backend_identity")
        if (
            not isinstance(identity, dict)
            or not isinstance(configuration, dict)
            or not isinstance(backend, dict)
        ):
            raise BatchFrontierV2QualificationError("selected model identity is incomplete")
        identity_sha = _sha_field(raw.get("model_identity_sha256"), "model_identity_sha256")
        identity_payload = dict(identity)
        if (
            identity_payload.pop("model_identity_sha256", None) != identity_sha
            or _canonical_sha256(identity_payload) != identity_sha
        ):
            raise BatchFrontierV2QualificationError("selected model identity hash changed")
        try:
            legacy._validate_training_model_metadata(
                str(architecture), identity, configuration, backend
            )
        except Exception as exc:
            raise BatchFrontierV2QualificationError(str(exc)) from exc
        payload = quality._load_torch_mapping(checkpoint)
        if (
            payload.get("architecture") != architecture
            or payload.get("training_seed") != seed
            or payload.get("completed_epochs") != selected_epoch
            or payload.get("model_configuration") != configuration
            or payload.get("model_identity") != identity
        ):
            raise BatchFrontierV2QualificationError("selected checkpoint metadata changed")
        state_sha = _sha_field(raw.get("model_state_sha256"), "model_state_sha256")
        if model_state_sha256(payload.get("model", {})) != state_sha:
            raise BatchFrontierV2QualificationError("selected checkpoint model state changed")
        record = {
            "architecture": architecture,
            "seed": seed,
            "training_seed": seed,
            "selected_epoch": selected_epoch,
            "path": relative,
            "checkpoint_path": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "model_state_sha256": state_sha,
            "model_identity": identity,
            "model_identity_sha256": identity_sha,
            "model_configuration": configuration,
            "backend_identity": backend,
            "architecture_recipe_sha256": raw.get("architecture_recipe_sha256"),
        }
        resolved[(architecture, seed)] = record
        models.append({key: value for key, value in record.items() if key != "checkpoint_path"})
    if set(resolved) != expected:
        raise BatchFrontierV2QualificationError("training-v2 checkpoint matrix is incomplete")

    architectures = manifest.get("architectures")
    if not isinstance(architectures, dict) or set(architectures) != set(ARCHITECTURES):
        raise BatchFrontierV2QualificationError("training-v2 architecture summaries are missing")
    debt: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        record = architectures[architecture]
        value = record.get("recipe_training_energy_j_mean") if isinstance(record, dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise BatchFrontierV2QualificationError(
                f"training-v2 energy debt is missing for {architecture}"
            )
        selection_included = (
            record.get("recipe_training_energy_j_mean_includes_checkpoint_selection") is True
        )
        if not selection_included:
            raise BatchFrontierV2QualificationError(
                f"training-v2 selection energy is not included for {architecture}"
            )
        debt[architecture] = {
            "recipe_training_energy_j_mean": float(value),
            "selection_energy_included": selection_included,
        }
    receipt = {
        "root": source.relative_to(root).as_posix(),
        "training_manifest_path": recipe.training_source.manifest_path,
        "training_manifest_sha256": _sha256_file(manifest_path),
        "training_manifest_schema": manifest["schema_version"],
        "training_status": manifest["status"],
        "architecture_training_debt": debt,
        "models": sorted(models, key=lambda item: (item["architecture"], item["training_seed"])),
    }
    return receipt, resolved


def _quality_source_context(
    recipe: BatchFrontierV2Recipe, root: Path
) -> tuple[dict[str, Any], list[float]]:
    """Validate and load only the sealed corpus/reference, not the old policy gate."""

    source = _relative(root, recipe.quality_source.root)
    artifacts: list[dict[str, str]] = []
    for artifact in recipe.quality_source.artifacts:
        path = _relative(source, artifact.path)
        if path.is_symlink() or not path.is_file() or _sha256_file(path) != artifact.sha256:
            raise BatchFrontierV2QualificationError(
                f"sealed quality artifact changed: {artifact.path}"
            )
        artifacts.append({"path": artifact.path, "sha256": artifact.sha256})
    lock = _load_json(source / "reference" / "reference-lock.json")
    reference_entry = lock.get("reference")
    if not isinstance(reference_entry, dict) or not isinstance(reference_entry.get("path"), str):
        raise BatchFrontierV2QualificationError("sealed reference lock is malformed")
    reference_path = _relative(source, reference_entry["path"])
    reference_sha = _sha_field(reference_entry.get("sha256"), "reference sha256")
    if _sha256_file(reference_path) != reference_sha:
        raise BatchFrontierV2QualificationError("sealed reference changed")
    reference = _load_json(reference_path)
    costs = reference.get("costs")
    if (
        reference.get("status") != "complete"
        or not isinstance(costs, list)
        or len(costs) != recipe.dataset.num_instances
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in costs
        )
    ):
        raise BatchFrontierV2QualificationError("sealed reference cost vector is invalid")
    receipt = {
        "root": recipe.quality_source.root,
        "artifacts": artifacts,
        "reference_path": reference_entry["path"],
        "reference_sha256": reference_sha,
        "corpus_content_sha256": recipe.dataset.artifact
        and "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf",
    }
    return receipt, [float(value) for value in costs]


def _configure_runtime(recipe: BatchFrontierV2Recipe) -> dict[str, Any]:
    try:
        identity = legacy._configure_runtime(recipe)  # type: ignore[arg-type]
    except Exception as exc:
        raise BatchFrontierV2QualificationError(str(exc)) from exc
    identity["hgs_baseline"] = {
        "baseline_id": "HGS-10s, 16 workers",
        "executor": recipe.hgs_policy.executor,
        "process_start_method": "spawn",
        "parallel_workers": recipe.hgs_policy.parallel_workers,
        "cpu_threads_per_worker": recipe.hgs_policy.cpu_threads_per_worker,
        "per_instance_independent_limit": True,
        "deterministic_instance_seed_mapping": "instance_i_uses_base_seed_plus_i",
        "measurement_boundary": HGS_MEASUREMENT_BOUNDARY,
        "pool_creation_included_in_energy": False,
        "worker_imports_included_in_energy": False,
        "warmup_included_in_energy": False,
        "pool_shutdown_included_in_energy": False,
    }
    return identity


def _exclusive_attestation(recipe: BatchFrontierV2Recipe) -> dict[str, Any]:
    try:
        return legacy._exclusive_attestation(recipe)  # type: ignore[arg-type]
    except Exception as exc:
        raise BatchFrontierV2QualificationError(str(exc)) from exc


def _assert_campaign_active(
    recipe: BatchFrontierV2Recipe, attestation: dict[str, Any], process_started: float
) -> None:
    try:
        legacy._assert_campaign_active(recipe, attestation, process_started)  # type: ignore[arg-type]
    except Exception as exc:
        raise BatchFrontierV2Error(str(exc)) from exc


def _prepare_output(
    recipe_path: Path,
    recipe: BatchFrontierV2Recipe,
    root: Path,
    *,
    resume: bool,
    runtime_identity: dict[str, Any],
    source_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = _safe_output(root, recipe.output_root)
    recipe_bytes = recipe_path.read_bytes()
    lock_bytes = (root / "uv.lock").read_bytes()
    git = software_recipe._current_git_snapshot((_relative(root, recipe.preflight_report),))
    if git.get("available") is not True:
        raise BatchFrontierV2QualificationError("Git source identity is unavailable")
    source_bytes = _json_bytes(source_receipt)
    identity = {
        "schema_version": RUN_STATE_SCHEMA,
        "recipe_sha256": _sha256_bytes(recipe_bytes),
        "uv_lock_sha256": _sha256_bytes(lock_bytes),
        "git_sha": git.get("sha"),
        "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
        "runtime_identity": runtime_identity,
        "source_receipt": source_receipt,
        "source_receipt_sha256": _sha256_bytes(source_bytes),
        "classification": classification(),
    }
    state_path = output / "run-state.json"
    if output.exists():
        if not resume:
            raise BatchFrontierV2Error(f"output already exists; use --resume: {output}")
        state = _load_json(state_path)
        if any(state.get(key) != value for key, value in identity.items()):
            raise BatchFrontierV2Error("resume identity differs from initialized frontier-v2")
        if (output / "recipe.yaml").read_bytes() != recipe_bytes:
            raise BatchFrontierV2Error("frozen recipe changed")
        if (output / "environment" / "uv.lock").read_bytes() != lock_bytes:
            raise BatchFrontierV2Error("frozen lockfile changed")
        if (output / "source-receipt.json").read_bytes() != source_bytes:
            raise BatchFrontierV2Error("frozen source receipt changed")
        return output, state
    state = {
        **identity,
        "status": INCOMPLETE_STATUS,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": None,
        "capacity_probe_sha256": None,
        "feasible_batches": None,
        "schedule": None,
        "completed_blocks": [],
        "block_sha256": {},
        "block_attempts": [],
        "current_block": None,
        "sessions": [],
    }
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        _atomic_write_bytes(staging / "source-receipt.json", source_bytes)
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _prepare_corpus(
    recipe: BatchFrontierV2Recipe,
    root: Path,
    output: Path,
    state: dict[str, Any],
) -> energy_runner.Corpus:
    try:
        return legacy._prepare_corpus(recipe, root, output, state)  # type: ignore[arg-type]
    except Exception as exc:
        raise BatchFrontierV2Error(str(exc)) from exc


def _build_model(
    recipe: BatchFrontierV2Recipe, entry: dict[str, Any], *, load_checkpoint: bool
) -> tuple[Any, Any]:
    try:
        return legacy._build_model(cast(Any, recipe), entry, load_checkpoint=load_checkpoint)
    except Exception as exc:
        raise BatchFrontierV2QualificationError(str(exc)) from exc


def _random_entry(architecture: str) -> dict[str, Any]:
    return legacy._random_entry(architecture)


def _greedy_cycle(
    recipe: BatchFrontierV2Recipe,
    corpus: energy_runner.Corpus,
    env: Any,
    model: Any,
    *,
    batch_size: int,
    limit: int | None = None,
) -> tuple[Any, ...]:
    """Run exactly one unaugmented argmax rollout per original instance."""

    import torch

    stop_at = corpus.coords.shape[0] if limit is None else limit
    routes: list[Any] = []
    device = torch.device(f"cuda:{recipe.gpu_index}")
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, stop_at, batch_size):
            stop = min(start + batch_size, stop_at)
            state = quality._initial_state(
                env, corpus.coords[start:stop], corpus.demands[start:stop], device
            )
            routes.extend(quality._greedy_routes(model, env, state))
    torch.cuda.synchronize(device)
    return tuple(routes)


def _capacity_probe(
    recipe: BatchFrontierV2Recipe,
    corpus: energy_runner.Corpus,
    entries: dict[tuple[str, int], dict[str, Any]] | None,
    *,
    load_checkpoints: bool,
) -> dict[str, Any]:
    import torch

    architectures: dict[str, Any] = {}
    device = torch.device(f"cuda:{recipe.gpu_index}")
    properties = torch.cuda.get_device_properties(device)
    for architecture in ARCHITECTURES:
        torch.manual_seed(2700 + ARCHITECTURES.index(architecture))
        entry = (
            entries[(architecture, recipe.capacity_probe.checkpoint_seed)]
            if entries is not None
            else _random_entry(architecture)
        )
        env, model = _build_model(recipe, entry, load_checkpoint=load_checkpoints)
        records: list[dict[str, Any]] = []
        feasible: list[int] = []
        failed = False
        for batch_size in recipe.neural_policy.batch_candidates:
            if failed:
                records.append(
                    {"batch_size": batch_size, "status": "not_probed_after_first_infeasible"}
                )
                continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            try:
                routes = _greedy_cycle(
                    recipe, corpus, env, model, batch_size=batch_size, limit=batch_size
                )
                validation = quality.validate_routes(
                    corpus.coords[:batch_size],
                    corpus.demands[:batch_size],
                    corpus.capacity,
                    routes,
                )
                if validation.get("complete") is not True:
                    raise BatchFrontierV2Error("capacity probe produced invalid greedy routes")
                feasible.append(batch_size)
                records.append(
                    {
                        "batch_size": batch_size,
                        "status": "feasible",
                        "elapsed_s": time.perf_counter() - started,
                        "peak_memory_allocated_b": int(torch.cuda.max_memory_allocated(device)),
                        "peak_memory_reserved_b": int(torch.cuda.max_memory_reserved(device)),
                        "device_total_memory_b": int(properties.total_memory),
                        "route_output_sha256": energy_runner._route_hash(routes, validation),
                    }
                )
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if (
                    not isinstance(exc, torch.OutOfMemoryError)
                    and "out of memory" not in str(exc).lower()
                ):
                    raise
                failed = True
                records.append(
                    {
                        "batch_size": batch_size,
                        "status": "infeasible_cuda_oom",
                        "elapsed_s": time.perf_counter() - started,
                        "device_total_memory_b": int(properties.total_memory),
                    }
                )
                torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
        if not feasible:
            raise BatchFrontierV2QualificationError(f"{architecture} has no feasible batch")
        architectures[architecture] = {
            "feasible_prefix": feasible,
            "maximum_feasible_batch_size": feasible[-1],
            "records": records,
        }
    return {
        "schema_version": CAPACITY_SCHEMA,
        "status": COMPLETE_STATUS,
        "measured_energy": False,
        "checkpoint_backed": load_checkpoints,
        "decode_policy": "greedy-1x1",
        "gpu_index": recipe.gpu_index,
        "batch_candidates": list(recipe.neural_policy.batch_candidates),
        "architectures": architectures,
    }


def _require_full_batch_grid(
    recipe: BatchFrontierV2Recipe, capacity: dict[str, Any]
) -> dict[str, tuple[int, ...]]:
    architectures = capacity.get("architectures")
    if not isinstance(architectures, dict):
        raise BatchFrontierV2QualificationError("capacity probe lacks architecture records")
    feasible: dict[str, tuple[int, ...]] = {}
    for architecture in ARCHITECTURES:
        record = architectures.get(architecture)
        values = record.get("feasible_prefix") if isinstance(record, dict) else None
        if not isinstance(values, list):
            raise BatchFrontierV2QualificationError("capacity probe is malformed")
        feasible[architecture] = tuple(int(value) for value in values)
        if feasible[architecture] != recipe.neural_policy.batch_candidates:
            raise BatchFrontierV2QualificationError(
                f"{architecture} cannot execute the preregistered full batch grid"
            )
    return feasible


def estimate_only(recipe_path: Path, workspace_root: Path | None = None) -> dict[str, Any]:
    """Probe capacity without writes and report a true fixed-budget time floor."""

    import numpy as np

    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    _configure_runtime(recipe)
    source = _relative(_relative(root, recipe.quality_source.root), recipe.dataset.artifact)
    with np.load(source, allow_pickle=False) as archive:
        corpus = energy_runner.Corpus(
            archive["coords"],
            archive["demands"],
            float(archive["capacity"].item()),
            str(archive["content_sha256"].item()),
            _sha256_file(source),
            source,
        )
    capacity = _capacity_probe(recipe, corpus, None, load_checkpoints=False)
    feasible = _require_full_batch_grid(recipe, capacity)
    schedule = expected_blocks(recipe, feasible)
    completed_schedule = _validated_existing_progress(recipe_path, recipe, root, schedule)
    remaining_schedule = schedule[len(completed_schedule) :]
    total_neural_blocks = sum(block.policy == "neural" for block in schedule)
    total_hgs_blocks = sum(block.policy == "hgs" for block in schedule)
    completed_neural_blocks = sum(block.policy == "neural" for block in completed_schedule)
    completed_hgs_blocks = sum(block.policy == "hgs" for block in completed_schedule)
    remaining_neural_blocks = sum(block.policy == "neural" for block in remaining_schedule)
    remaining_hgs_blocks = sum(block.policy == "hgs" for block in remaining_schedule)
    hgs_block_floor_s = (
        math.ceil(recipe.dataset.num_instances / recipe.hgs_policy.parallel_workers)
        * recipe.hgs_policy.max_runtime_s
    )
    total_floor = minimum_measured_walltime_s(recipe, feasible)
    remaining_neural_floor = remaining_neural_blocks * recipe.minimum_block_duration_s
    remaining_hgs_floor = remaining_hgs_blocks * hgs_block_floor_s
    remaining_floor = remaining_neural_floor + remaining_hgs_floor
    return {
        "schema_version": ESTIMATE_SCHEMA,
        "status": COMPLETE_STATUS,
        "writes_performed": False,
        "energy_measured": False,
        "capacity_probe": capacity,
        "resolved_feasible_batches": {
            architecture: list(values) for architecture, values in feasible.items()
        },
        "resolved_block_count": len(schedule),
        "resolved_neural_block_count": total_neural_blocks,
        "resolved_hgs_block_count": total_hgs_blocks,
        "total_block_count": len(schedule),
        "completed_block_count": len(completed_schedule),
        "remaining_block_count": len(remaining_schedule),
        "total_neural_block_count": total_neural_blocks,
        "completed_neural_block_count": completed_neural_blocks,
        "remaining_neural_block_count": remaining_neural_blocks,
        "total_hgs_block_count": total_hgs_blocks,
        "completed_hgs_block_count": completed_hgs_blocks,
        "remaining_hgs_block_count": remaining_hgs_blocks,
        "neural_measured_floor_s": total_neural_blocks * recipe.minimum_block_duration_s,
        "hgs_measured_floor_s": total_hgs_blocks * hgs_block_floor_s,
        "minimum_measured_walltime_s": total_floor,
        "minimum_measured_walltime_hours": total_floor / 3600,
        "remaining_neural_measured_floor_s": remaining_neural_floor,
        "remaining_hgs_measured_floor_s": remaining_hgs_floor,
        "remaining_minimum_measured_walltime_s": remaining_floor,
        "remaining_minimum_measured_walltime_hours": remaining_floor / 3600,
        "progress_source": ("existing_run_state" if completed_schedule else "no_completed_blocks"),
        "estimate_excludes_pool_warmup_checkpoint_loading_and_finalization": True,
    }


def _validated_existing_progress(
    recipe_path: Path,
    recipe: BatchFrontierV2Recipe,
    root: Path,
    schedule: Sequence[FrontierBlock],
) -> tuple[FrontierBlock, ...]:
    """Read and authenticate a durable schedule prefix without changing it."""

    output = _safe_output(root, recipe.output_root)
    if not output.exists():
        return ()
    state_path = output / "run-state.json"
    if not state_path.is_file():
        raise BatchFrontierV2Error("existing frontier-v2 output lacks run-state.json")
    state = _load_json(state_path)
    recipe_bytes = recipe_path.read_bytes()
    if state.get("schema_version") != RUN_STATE_SCHEMA or state.get(
        "recipe_sha256"
    ) != _sha256_bytes(recipe_bytes):
        raise BatchFrontierV2Error("existing frontier-v2 recipe identity changed")
    frozen_recipe = output / "recipe.yaml"
    if not frozen_recipe.is_file() or frozen_recipe.read_bytes() != recipe_bytes:
        raise BatchFrontierV2Error("existing frontier-v2 frozen recipe changed")
    completed = state.get("completed_blocks")
    hashes = state.get("block_sha256")
    if not isinstance(completed, list) or not isinstance(hashes, dict):
        raise BatchFrontierV2Error("existing frontier-v2 block prefix is malformed")
    expected_paths = [block.relative_path for block in schedule]
    if completed != expected_paths[: len(completed)]:
        raise BatchFrontierV2Error("existing frontier-v2 blocks are not a schedule prefix")
    stored_schedule = state.get("schedule")
    if stored_schedule is None:
        if completed:
            raise BatchFrontierV2Error("existing frontier-v2 schedule is missing")
    elif stored_schedule != [asdict(block) for block in schedule]:
        raise BatchFrontierV2Error("existing frontier-v2 schedule changed")
    if set(hashes) != set(completed):
        raise BatchFrontierV2Error("existing frontier-v2 block hash index changed")
    for relative_path in completed:
        path = _relative(output, relative_path)
        if not path.is_file() or _sha256_file(path) != hashes[relative_path]:
            raise BatchFrontierV2Error(f"existing frontier-v2 block changed: {relative_path}")
    if state.get("status") == COMPLETE_STATUS and len(completed) != len(schedule):
        raise BatchFrontierV2Error("completed frontier-v2 output lacks the full schedule")
    return tuple(schedule[: len(completed)])


def _execute_neural_block(
    recipe: BatchFrontierV2Recipe,
    block: FrontierBlock,
    *,
    corpus: energy_runner.Corpus,
    entry: dict[str, Any],
    attestation: dict[str, Any],
) -> dict[str, Any]:
    import torch

    if block.architecture is None or block.batch_size is None:
        raise BatchFrontierV2Error("neural block lacks architecture or batch size")
    env, model = _build_model(recipe, entry, load_checkpoint=True)
    warmup_limit = min(block.batch_size, recipe.dataset.num_instances)
    warmup = _greedy_cycle(
        recipe, corpus, env, model, batch_size=block.batch_size, limit=warmup_limit
    )
    if (
        quality.validate_routes(
            corpus.coords[:warmup_limit],
            corpus.demands[:warmup_limit],
            corpus.capacity,
            warmup,
        ).get("complete")
        is not True
    ):
        raise BatchFrontierV2Error("neural warmup produced invalid greedy routes")
    before = energy_runner._diagnostic_gpu_snapshot(recipe.gpu_index)
    tracker = energy_runner._make_strict_tracker(
        f"batch-frontier-v2-{block.architecture}-b{block.batch_size}-r{block.round_index}",
        recipe,  # type: ignore[arg-type]
    )
    repetitions = 0
    first_routes: tuple[Any, ...] | None = None
    last_routes: tuple[Any, ...] | None = None
    wall_started = time.perf_counter()
    with tracker as active:
        measured_started = time.perf_counter()
        while True:
            current = _greedy_cycle(recipe, corpus, env, model, batch_size=block.batch_size)
            first_routes = current if first_routes is None else first_routes
            last_routes = current
            repetitions += 1
            if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
                raise BatchFrontierV2Error("neural block exceeded maximum wall time")
            if time.perf_counter() - measured_started >= recipe.minimum_block_duration_s:
                break
        active.n_items = repetitions * recipe.dataset.num_instances
    after = energy_runner._diagnostic_gpu_snapshot(recipe.gpu_index)
    if first_routes is None or last_routes is None:
        raise BatchFrontierV2Error("neural block completed no corpus repetition")
    first_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, first_routes
    )
    last_validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, last_routes
    )
    first_hash = energy_runner._route_hash(first_routes, first_validation)
    last_hash = energy_runner._route_hash(last_routes, last_validation)
    if (
        first_validation.get("complete") is not True
        or last_validation.get("complete") is not True
        or first_hash != last_hash
    ):
        raise BatchFrontierV2Error("greedy output is invalid or nondeterministic")
    instances = repetitions * recipe.dataset.num_instances
    energy = energy_runner._checked_energy(
        tracker,
        recipe,  # type: ignore[arg-type]
        items_processed=instances,
    )
    duration = float(energy["duration_s"])
    del model
    torch.cuda.empty_cache()
    return {
        "schema_version": BLOCK_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": "neural",
        "decode_policy": "greedy-1x1",
        "architecture": block.architecture,
        "batch_size": block.batch_size,
        "training_seed": block.training_seed,
        "selected_epoch": entry["selected_epoch"],
        "evaluation_seed": block.evaluation_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "attestation": attestation,
        "minimum_duration_s": recipe.minimum_block_duration_s,
        "duration_policy": "repeat_complete_corpus_until_minimum_duration",
        "normalization_unit": "original_problem_instance",
        "repetitions": repetitions,
        "instances_per_repetition": recipe.dataset.num_instances,
        "instances_processed": instances,
        "duration_s": duration,
        "throughput_instances_per_s": instances / duration,
        "energy": energy,
        "validation": first_validation,
        "first_measured_output_sha256": first_hash,
        "last_measured_output_sha256": last_hash,
        "first_and_last_outputs_identical": True,
        "checkpoint": {
            "path": entry["path"],
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "model_state_sha256": entry["model_state_sha256"],
            "model_identity_sha256": entry["model_identity_sha256"],
        },
        "gpu_process_snapshot_before": before,
        "gpu_process_snapshot_after": after,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _validated_runtime_hgs_results(
    results: Sequence[Any],
    *,
    expected_instances: int,
    base_seed: int,
    max_runtime_s: float,
    scaling_factor: int,
) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    if len(results) != expected_instances:
        raise BatchFrontierV2Error("HGS did not return one result per original instance")
    routes: list[Any] = []
    metadata: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if (
            result.instance_index != index
            or result.seed != base_seed + index
            or result.limit_kind != "time"
            or result.max_iterations is not None
            or not math.isclose(result.max_runtime_s, max_runtime_s, rel_tol=0.0, abs_tol=0.0)
            or result.scaling_factor != scaling_factor
            or isinstance(result.worker_pid, bool)
            or not isinstance(result.worker_pid, int)
            or result.worker_pid <= 0
        ):
            raise BatchFrontierV2Error("HGS result disagrees with the 10-second policy")
        routes.append(tuple(tuple(int(customer) for customer in route) for route in result.routes))
        metadata.append(
            {
                "instance_index": index,
                "effective_seed": int(result.seed),
                "limit_kind": "time",
                "max_iterations": None,
                "max_runtime_s": float(result.max_runtime_s),
                "integer_cost": int(result.integer_cost),
                "reported_cost": float(result.cost),
                "scaling_factor": int(result.scaling_factor),
            }
        )
    return tuple(routes), metadata


def _hgs_tasks(
    recipe: BatchFrontierV2Recipe,
    corpus: energy_runner.Corpus,
    *,
    base_seed: int,
    count: int,
    synchronize_warmup_workers: bool,
) -> list[tuple[Any, ...]]:
    if count < 1 or count > recipe.dataset.num_instances:
        raise BatchFrontierV2Error("HGS task count is outside the sealed corpus")
    return [
        (
            index,
            corpus.coords[index],
            corpus.demands[index],
            corpus.capacity,
            base_seed + index,
            recipe.hgs_policy.max_runtime_s,
            recipe.hgs_policy.scaling_factor,
            recipe.hgs_policy.collect_stats,
            synchronize_warmup_workers,
        )
        for index in range(count)
    ]


def _create_hgs_executor(
    recipe: BatchFrontierV2Recipe,
) -> tuple[concurrent.futures.ProcessPoolExecutor, Any]:
    """Create all workers outside the energy-measurement boundary."""

    context = multiprocessing.get_context("spawn")
    warmup_barrier = context.Barrier(recipe.hgs_policy.parallel_workers)
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=recipe.hgs_policy.parallel_workers,
        mp_context=context,
        initializer=_hgs_worker_initializer,
        initargs=(warmup_barrier,),
    )
    return executor, warmup_barrier


def _warm_hgs_executor(
    recipe: BatchFrontierV2Recipe,
    corpus: energy_runner.Corpus,
    executor: concurrent.futures.ProcessPoolExecutor,
    *,
    base_seed: int,
) -> tuple[list[Any], tuple[int, ...]]:
    """Force every worker through one real 10-second solve before measurement."""

    tasks = _hgs_tasks(
        recipe,
        corpus,
        base_seed=base_seed,
        count=recipe.hgs_policy.parallel_workers,
        synchronize_warmup_workers=True,
    )
    results = list(executor.map(_solve_hgs_instance_task, tasks, chunksize=1))
    routes, _ = _validated_runtime_hgs_results(
        results,
        expected_instances=recipe.hgs_policy.parallel_workers,
        base_seed=base_seed,
        max_runtime_s=recipe.hgs_policy.max_runtime_s,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    validation = quality.validate_routes(
        corpus.coords[: recipe.hgs_policy.parallel_workers],
        corpus.demands[: recipe.hgs_policy.parallel_workers],
        corpus.capacity,
        routes,
    )
    worker_pids = tuple(sorted({int(result.worker_pid) for result in results}))
    if (
        validation.get("complete") is not True
        or len(worker_pids) != recipe.hgs_policy.parallel_workers
    ):
        raise BatchFrontierV2Error(
            "HGS pool warmup did not validate one real solve on every worker"
        )
    return results, worker_pids


def _parallel_hgs_cycle(
    recipe: BatchFrontierV2Recipe,
    corpus: energy_runner.Corpus,
    executor: concurrent.futures.ProcessPoolExecutor,
    *,
    base_seed: int,
) -> list[Any]:
    """Measure the sealed corpus on an already warm Windows process pool."""

    if corpus.coords.shape[0] != 512 or corpus.demands.shape[0] != 512:
        raise BatchFrontierV2Error("parallel HGS requires exactly 512 sealed instances")
    tasks = _hgs_tasks(
        recipe,
        corpus,
        base_seed=base_seed,
        count=recipe.dataset.num_instances,
        synchronize_warmup_workers=False,
    )
    results = list(executor.map(_solve_hgs_instance_task, tasks, chunksize=1))
    if [result.instance_index for result in results] != list(range(512)):
        raise BatchFrontierV2Error("parallel HGS result order changed")
    return results


def _execute_hgs_block(
    recipe: BatchFrontierV2Recipe,
    block: FrontierBlock,
    *,
    corpus: energy_runner.Corpus,
    attestation: dict[str, Any],
) -> dict[str, Any]:
    executor, warmup_barrier = _create_hgs_executor(recipe)
    try:
        _warmup_results, warmup_worker_pids = _warm_hgs_executor(
            recipe, corpus, executor, base_seed=block.hgs_seed
        )
        before = energy_runner._diagnostic_gpu_snapshot(recipe.gpu_index)
        tracker = energy_runner._make_strict_tracker(
            f"batch-frontier-v2-hgs10s-r{block.round_index}",
            recipe,  # type: ignore[arg-type]
        )
        wall_started = time.perf_counter()
        with tracker as active:
            results = _parallel_hgs_cycle(recipe, corpus, executor, base_seed=block.hgs_seed)
            active.n_items = recipe.dataset.num_instances
        if time.perf_counter() - wall_started > recipe.maximum_block_walltime_s:
            raise BatchFrontierV2Error("HGS block exceeded maximum wall time")
        after = energy_runner._diagnostic_gpu_snapshot(recipe.gpu_index)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        del warmup_barrier
    routes, metadata = _validated_runtime_hgs_results(
        results,
        expected_instances=recipe.dataset.num_instances,
        base_seed=block.hgs_seed,
        max_runtime_s=recipe.hgs_policy.max_runtime_s,
        scaling_factor=recipe.hgs_policy.scaling_factor,
    )
    validation = quality.validate_routes(corpus.coords, corpus.demands, corpus.capacity, routes)
    route_hash = energy_runner._route_hash(routes, validation, metadata)
    if validation.get("complete") is not True:
        raise BatchFrontierV2Error("HGS produced invalid routes")
    energy = energy_runner._checked_energy(
        tracker,
        recipe,  # type: ignore[arg-type]
        items_processed=recipe.dataset.num_instances,
    )
    energy.update(
        {
            "primary_hgs_energy_domain": "aggregate_cpu_package",
            "primary_hgs_energy_j_per_instance": energy["cpu_package_energy_j_per_instance"],
            "gpu_energy_in_primary_hgs_baseline": False,
            "gpu_energy_role": "idle_host_gpu_diagnostic",
            "per_process_energy_summed": False,
        }
    )
    duration = float(energy["duration_s"])
    return {
        "schema_version": BLOCK_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": "hgs",
        "baseline_id": "HGS-10s, 16 workers",
        "architecture": None,
        "batch_size": None,
        "training_seed": block.training_seed,
        "hgs_seed": block.hgs_seed,
        "hgs_max_runtime_s_per_instance": recipe.hgs_policy.max_runtime_s,
        "hgs_execution": "windows_process_pool",
        "parallel_workers": recipe.hgs_policy.parallel_workers,
        "cpu_threads_per_worker": recipe.hgs_policy.cpu_threads_per_worker,
        "per_instance_independent_limit": True,
        "deterministic_instance_seed_mapping": "instance_i_uses_base_seed_plus_i",
        "measurement_boundary": HGS_MEASUREMENT_BOUNDARY,
        "pool_creation_included_in_energy": False,
        "worker_imports_included_in_energy": False,
        "warmup_included_in_energy": False,
        "pool_shutdown_included_in_energy": False,
        "warmup_real_solve_count": recipe.hgs_policy.parallel_workers,
        "warmup_distinct_worker_count": len(warmup_worker_pids),
        "warmup_worker_pid_set_sha256": _canonical_sha256(warmup_worker_pids),
        "dataset_content_sha256": corpus.content_sha256,
        "attestation": attestation,
        "minimum_duration_s": math.ceil(
            recipe.dataset.num_instances / recipe.hgs_policy.parallel_workers
        )
        * recipe.hgs_policy.max_runtime_s,
        "duration_policy": "one_complete_corpus_parallel_pool_fixed_runtime_per_instance",
        "normalization_unit": "original_problem_instance",
        "energy_normalization": HGS_ENERGY_NORMALIZATION,
        "per_process_energy_summed": False,
        "repetitions": 1,
        "instances_per_repetition": recipe.dataset.num_instances,
        "instances_processed": recipe.dataset.num_instances,
        "duration_s": duration,
        "throughput_instances_per_s": recipe.dataset.num_instances / duration,
        "energy": energy,
        "validation": validation,
        "first_measured_output_sha256": route_hash,
        "last_measured_output_sha256": route_hash,
        "first_and_last_outputs_identical": True,
        "solver_result_metadata": metadata,
        "gpu_process_snapshot_before": before,
        "gpu_process_snapshot_after": after,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _validate_block(
    payload: dict[str, Any],
    recipe: BatchFrontierV2Recipe,
    block: FrontierBlock,
    corpus: energy_runner.Corpus,
) -> None:
    expected = {
        "schema_version": BLOCK_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "round": block.round_index,
        "order_within_round": block.order_index,
        "policy": block.policy,
        "architecture": block.architecture,
        "batch_size": block.batch_size,
        "training_seed": block.training_seed,
        "dataset_content_sha256": corpus.content_sha256,
        "normalization_unit": "original_problem_instance",
        "instances_per_repetition": recipe.dataset.num_instances,
        "first_and_last_outputs_identical": True,
    }
    if block.policy == "neural":
        expected.update(
            {
                "decode_policy": "greedy-1x1",
                "evaluation_seed": block.evaluation_seed,
                "minimum_duration_s": recipe.minimum_block_duration_s,
                "duration_policy": "repeat_complete_corpus_until_minimum_duration",
            }
        )
    else:
        expected.update(
            {
                "hgs_seed": block.hgs_seed,
                "baseline_id": "HGS-10s, 16 workers",
                "hgs_max_runtime_s_per_instance": recipe.hgs_policy.max_runtime_s,
                "hgs_execution": "windows_process_pool",
                "parallel_workers": recipe.hgs_policy.parallel_workers,
                "cpu_threads_per_worker": recipe.hgs_policy.cpu_threads_per_worker,
                "per_instance_independent_limit": True,
                "deterministic_instance_seed_mapping": "instance_i_uses_base_seed_plus_i",
                "measurement_boundary": HGS_MEASUREMENT_BOUNDARY,
                "pool_creation_included_in_energy": False,
                "worker_imports_included_in_energy": False,
                "warmup_included_in_energy": False,
                "pool_shutdown_included_in_energy": False,
                "warmup_real_solve_count": recipe.hgs_policy.parallel_workers,
                "warmup_distinct_worker_count": recipe.hgs_policy.parallel_workers,
                "minimum_duration_s": math.ceil(
                    recipe.dataset.num_instances / recipe.hgs_policy.parallel_workers
                )
                * recipe.hgs_policy.max_runtime_s,
                "duration_policy": ("one_complete_corpus_parallel_pool_fixed_runtime_per_instance"),
                "energy_normalization": HGS_ENERGY_NORMALIZATION,
                "per_process_energy_summed": False,
                "repetitions": 1,
            }
        )
    if any(payload.get(key) != value for key, value in expected.items()):
        raise BatchFrontierV2Error(f"cached block identity changed: {block.relative_path}")
    repetitions = payload.get("repetitions")
    validation = payload.get("validation")
    costs = validation.get("costs") if isinstance(validation, dict) else None
    if (
        isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or payload.get("instances_processed") != repetitions * recipe.dataset.num_instances
        or not isinstance(validation, dict)
        or validation.get("complete") is not True
        or not isinstance(costs, list)
        or len(costs) != recipe.dataset.num_instances
        or payload.get("first_measured_output_sha256") != payload.get("last_measured_output_sha256")
    ):
        raise BatchFrontierV2Error(f"cached block metrics changed: {block.relative_path}")
    energy = payload.get("energy")
    if not isinstance(energy, dict):
        raise BatchFrontierV2Error("cached block lacks energy")
    for key in (
        "cpu_package_energy_j_per_instance",
        "gpu_energy_j_per_instance",
        "observed_component_energy_j_per_instance",
    ):
        value = energy.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise BatchFrontierV2Error(f"cached block energy changed: {key}")


def _series_summary(values: list[float]) -> dict[str, Any]:
    average = mean(values)
    sample_sd = stdev(values) if len(values) > 1 else 0.0
    half_width = 2.7764451051977987 * sample_sd / math.sqrt(5) if len(values) == 5 else None
    return {
        "n": len(values),
        "mean": average,
        "median": median(values),
        "sample_standard_deviation": sample_sd,
        "minimum": min(values),
        "maximum": max(values),
        "descriptive_t_interval_95": (
            [average - half_width, average + half_width] if half_width is not None else None
        ),
        "positive_count": sum(value > 0 for value in values),
        "negative_count": sum(value < 0 for value in values),
        "zero_count": sum(value == 0 for value in values),
        "inferential_use": False,
    }


def _direct_quality_gate(
    payloads_by_round: dict[int, dict[str, Any]],
    reference_costs: list[float],
    *,
    threshold_pct: float,
) -> dict[str, Any]:
    """Apply the full frozen 5% gate directly to routes from this campaign."""

    import numpy as np

    from neuro_co.aet.experiments import hgs_holdout_runner as holdout
    from neuro_co.aet.experiments import training_debt_runner as training

    gap_rows: list[list[float]] = []
    invalid = 0
    for round_index in range(5):
        validation = payloads_by_round[round_index].get("validation")
        if not isinstance(validation, dict):
            raise BatchFrontierV2Error("quality block lacks route validation")
        gap = quality._gap_summary(validation.get("costs"), reference_costs)
        values = gap.get("gaps_pct")
        if not isinstance(values, list) or len(values) != len(reference_costs):
            raise BatchFrontierV2Error("quality gap vector is malformed")
        gap_rows.append([float(value) for value in values])
        count = validation.get("invalid_instance_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise BatchFrontierV2Error("invalid-instance count is malformed")
        invalid += count
    matrix = np.asarray(gap_rows, dtype=np.float64)
    bootstrap_seed = 3495
    bootstrap_replicates = 10_000
    bootstrap_quantile = 0.95
    mean_samples, q95_samples = training._bootstrap_samples(
        matrix,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    summary = holdout._summarize_policy(
        matrix,
        policy_id="batch_frontier_v2_measured_routes",
        policy_role="direct_deployment_quality_qualification",
        invalid_instances=invalid,
        threshold_pct=threshold_pct,
        t_critical_value=2.13184678632665,
        t_degrees_of_freedom=4,
        bootstrap_mean_samples=mean_samples,
        bootstrap_q95_samples=q95_samples,
        bootstrap_quantile=bootstrap_quantile,
    )
    passed = summary.get("passed") is True
    return {
        "quality_status": "feasible" if passed else "infeasible",
        "frontier_eligible": passed,
        "quality_gate": {
            "passed": passed,
            "status": "quality_passed" if passed else "quality_nonpass",
            "rule": (
                "zero invalid; each seed mean and q95, pooled q95, t-UCB, "
                "bootstrap mean UCB, and bootstrap q95 UCB strictly below 5%"
            ),
            "threshold_pct": threshold_pct,
            "summary": summary,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_quantile": bootstrap_quantile,
            "reference_instance_count": len(reference_costs),
            "route_sources": "routes_measured_in_batch_frontier_v2",
            "inherits_pomo50x8_gate": False,
        },
    }


def batch_frontier_v2_summary(
    recipe: BatchFrontierV2Recipe,
    source_receipt: dict[str, Any],
    capacity: dict[str, Any],
    payloads: Sequence[dict[str, Any]],
    *,
    reference_costs: list[float],
) -> dict[str, Any]:
    hgs_payloads = [payload for payload in payloads if payload["policy"] == "hgs"]
    hgs_by_round = {int(payload["round"]): payload for payload in hgs_payloads}
    if len(hgs_payloads) != 5 or set(hgs_by_round) != set(range(5)):
        raise BatchFrontierV2Error("summary requires one HGS-10s block per round")
    hgs_quality = _direct_quality_gate(
        hgs_by_round, reference_costs, threshold_pct=recipe.quality_threshold_pct
    )
    architectures: dict[str, Any] = {}
    for architecture in ARCHITECTURES:
        batches: list[dict[str, Any]] = []
        for batch_size in capacity["architectures"][architecture]["feasible_prefix"]:
            neural_by_round = {
                int(payload["round"]): payload
                for payload in payloads
                if payload["policy"] == "neural"
                and payload["architecture"] == architecture
                and payload["batch_size"] == batch_size
            }
            if set(neural_by_round) != set(range(5)):
                raise BatchFrontierV2Error("summary requires five neural blocks per batch cell")
            quality_gate = _direct_quality_gate(
                neural_by_round,
                reference_costs,
                threshold_pct=recipe.quality_threshold_pct,
            )
            pairs: list[dict[str, Any]] = []
            primary: list[float] = []
            same_host: list[float] = []
            for round_index in range(5):
                neural = neural_by_round[round_index]
                hgs = hgs_by_round[round_index]
                neural_total = float(neural["energy"]["observed_component_energy_j_per_instance"])
                hgs_cpu = float(hgs["energy"]["cpu_package_energy_j_per_instance"])
                hgs_total = float(hgs["energy"]["observed_component_energy_j_per_instance"])
                primary.append(hgs_cpu - neural_total)
                same_host.append(hgs_total - neural_total)
                pairs.append(
                    {
                        "round": round_index,
                        "training_seed": neural["training_seed"],
                        "selected_epoch": neural["selected_epoch"],
                        "neural_observed_components_j_per_instance": neural_total,
                        "hgs_cpu_package_j_per_instance": hgs_cpu,
                        "hgs_observed_components_j_per_instance": hgs_total,
                        "delta_hgs_minus_neural_j_per_instance": hgs_cpu - neural_total,
                        "same_host_observed_delta_hgs_minus_neural_j_per_instance": hgs_total
                        - neural_total,
                        "neural_throughput_instances_per_s": neural["throughput_instances_per_s"],
                        "hgs_throughput_instances_per_s": hgs["throughput_instances_per_s"],
                    }
                )
            batches.append(
                {
                    "batch_size": batch_size,
                    **quality_gate,
                    "pairs": pairs,
                    "delta_hgs_minus_neural_j_per_instance": _series_summary(primary),
                    "same_host_observed_delta_hgs_minus_neural_j_per_instance": _series_summary(
                        same_host
                    ),
                }
            )
        architectures[architecture] = {
            "capacity": capacity["architectures"][architecture],
            "training_debt": source_receipt["training"]["architecture_training_debt"][architecture],
            "batches": batches,
        }
    hgs_cpu = [
        float(payload["energy"]["cpu_package_energy_j_per_instance"]) for payload in hgs_payloads
    ]
    hgs_total = [
        float(payload["energy"]["observed_component_energy_j_per_instance"])
        for payload in hgs_payloads
    ]
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "dataset_content_sha256": (
            "172b67d71a944ff1c39abf1d7925e7dd807388e4b309d9d8cbedf54b9863c2bf"
        ),
        "normalization_unit": "original_problem_instance",
        "neural_policy": {
            "mode_id": recipe.neural_policy.mode_id,
            "n_starts": recipe.neural_policy.n_starts,
            "augmentations": recipe.neural_policy.augmentations,
            "forced_first_actions": recipe.neural_policy.forced_first_actions,
            "inference_precision": recipe.neural_policy.inference_precision,
        },
        "training_source": {
            "training_manifest_sha256": source_receipt["training"]["training_manifest_sha256"],
            "architecture_training_debt": source_receipt["training"]["architecture_training_debt"],
            "models": source_receipt["training"]["models"],
        },
        "primary_delta_definition": (
            "HGS CPU-package energy minus neural CPU-package-plus-GPU energy per original instance"
        ),
        "architectures": architectures,
        "hgs_reference": {
            "policy": "pyvrp-hgs",
            "baseline_id": "HGS-10s, 16 workers",
            "max_runtime_s": recipe.hgs_policy.max_runtime_s,
            "max_runtime_s_per_instance": recipe.hgs_policy.max_runtime_s,
            "parallel": True,
            "parallel_workers": recipe.hgs_policy.parallel_workers,
            "cpu_threads_per_worker": recipe.hgs_policy.cpu_threads_per_worker,
            "per_instance_independent_limit": True,
            "executor": recipe.hgs_policy.executor,
            "deterministic_instance_seed_mapping": "instance_i_uses_base_seed_plus_i",
            "measurement_boundary": HGS_MEASUREMENT_BOUNDARY,
            "pool_creation_included_in_energy": False,
            "worker_imports_included_in_energy": False,
            "warmup_included_in_energy": False,
            "pool_shutdown_included_in_energy": False,
            "warmup_real_solve_count_per_block": recipe.hgs_policy.parallel_workers,
            "energy_normalization": HGS_ENERGY_NORMALIZATION,
            "per_process_energy_summed": False,
            "gpu_energy_role": "idle_host_gpu_diagnostic_excluded_from_primary_hgs_energy",
            "rounds": 5,
            "quality_status": hgs_quality["quality_status"],
            "frontier_eligible": hgs_quality["frontier_eligible"],
            "quality_gate": hgs_quality["quality_gate"],
            "cpu_package_energy_j_per_instance": _series_summary(hgs_cpu),
            "observed_component_energy_j_per_instance": _series_summary(hgs_total),
        },
        "quality_was_computed_from_v2_routes": True,
        "pomo50x8_quality_was_inherited": False,
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "aet_computed": False,
    }


def _write_checksums(output: Path) -> str:
    rows: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in {"SHA256SUMS", "run-state.json"} or ".partial-" in path.name:
            continue
        rows.append(f"{_sha256_file(path)}  {relative}")
    _atomic_write_bytes(output / "SHA256SUMS", ("\n".join(rows) + "\n").encode())
    return _sha256_file(output / "SHA256SUMS")


def _complete(
    recipe: BatchFrontierV2Recipe,
    output: Path,
    state: dict[str, Any],
    capacity: dict[str, Any],
    payloads: list[dict[str, Any]],
    reference_costs: list[float],
) -> BatchFrontierV2Result:
    summary = batch_frontier_v2_summary(
        recipe,
        state["source_receipt"],
        capacity,
        payloads,
        reference_costs=reference_costs,
    )
    summary_path = output / SUMMARY_FILENAME
    _atomic_write_json(summary_path, summary)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": classification(),
        "dataset": state["dataset"],
        "capacity_probe_sha256": state["capacity_probe_sha256"],
        "resolved_block_count": len(payloads),
        "completed_block_count": len(payloads),
        "paired_rounds": recipe.paired_rounds,
        "hgs_blocks": recipe.paired_rounds,
        "neural_decode_policy": "greedy-1x1",
        "hgs_baseline_id": "HGS-10s, 16 workers",
        "hgs_max_runtime_s_per_instance": recipe.hgs_policy.max_runtime_s,
        "hgs_parallel_workers": recipe.hgs_policy.parallel_workers,
        "hgs_per_instance_independent_limit": True,
        "hgs_measurement_boundary": HGS_MEASUREMENT_BOUNDARY,
        "hgs_pool_creation_included_in_energy": False,
        "hgs_worker_imports_included_in_energy": False,
        "hgs_warmup_included_in_energy": False,
        "hgs_pool_shutdown_included_in_energy": False,
        "summary_path": SUMMARY_FILENAME,
        "batch_frontier_v2_summary_sha256": _sha256_file(summary_path),
        "source_receipt_sha256": state["source_receipt_sha256"],
        "runtime_identity": state["runtime_identity"],
        "training_was_run_by_frontier": False,
        "aet_was_computed": False,
        "carbon_was_computed": False,
    }
    _atomic_write_json(output / "manifest.json", manifest)
    state.update(
        {
            "status": COMPLETE_STATUS,
            "completed_at": datetime.now(UTC).isoformat(),
            "batch_frontier_v2_summary_sha256": manifest["batch_frontier_v2_summary_sha256"],
            "manifest_sha256": _sha256_file(output / "manifest.json"),
            "checksums_sha256": _write_checksums(output),
        }
    )
    _atomic_write_json(output / "run-state.json", state)
    return BatchFrontierV2Result(
        output,
        COMPLETE_STATUS,
        len(payloads),
        output / "manifest.json",
        state["manifest_sha256"],
    )


def run_batch_frontier_v2(
    recipe_path: Path,
    workspace_root: Path | None = None,
    *,
    resume: bool = False,
    training_source: Path | None = None,
) -> BatchFrontierV2Result:
    root = (workspace_root or Path.cwd()).resolve(strict=True)
    recipe = load_recipe(recipe_path)
    qualification = qualify_for_execution(recipe, root, training_source)
    if qualification["ready_to_execute"] is not True:
        raise BatchFrontierV2QualificationError("native Windows or source qualification failed")
    training_receipt, entries = _resolve_training_source(recipe, root, training_source)
    quality_receipt, reference_costs = _quality_source_context(recipe, root)
    source_receipt = {"quality": quality_receipt, "training": training_receipt}
    attestation = _exclusive_attestation(recipe)
    runtime_identity = _configure_runtime(recipe)
    output, state = _prepare_output(
        recipe_path,
        recipe,
        root,
        resume=resume,
        runtime_identity=runtime_identity,
        source_receipt=source_receipt,
    )
    corpus = _prepare_corpus(recipe, root, output, state)
    sessions = state.get("sessions")
    if not isinstance(sessions, list):
        raise BatchFrontierV2Error("session history is malformed")
    if state.get("status") != COMPLETE_STATUS:
        try:
            preflight = legacy._preserve_preflight(recipe, root, output)  # type: ignore[arg-type]
        except Exception as exc:
            raise BatchFrontierV2QualificationError(str(exc)) from exc
        sessions.append(
            {
                **attestation,
                "preflight_evidence": preflight,
                "process_started_at": datetime.now(UTC).isoformat(),
                "resume": resume,
                "completed_block_count_at_start": len(state.get("completed_blocks", [])),
            }
        )
        _atomic_write_json(output / "run-state.json", state)
    process_started = time.perf_counter()

    capacity_path = output / "capacity-probe.json"
    if state.get("capacity_probe_sha256") is None:
        capacity = _capacity_probe(recipe, corpus, entries, load_checkpoints=True)
        _require_full_batch_grid(recipe, capacity)
        _atomic_write_json(capacity_path, capacity)
        state["capacity_probe_sha256"] = _sha256_file(capacity_path)
        state["feasible_batches"] = {
            architecture: record["feasible_prefix"]
            for architecture, record in capacity["architectures"].items()
        }
        schedule = expected_blocks(
            recipe,
            {key: tuple(value) for key, value in state["feasible_batches"].items()},
        )
        state["schedule"] = [asdict(block) for block in schedule]
        _atomic_write_json(output / "run-state.json", state)
    else:
        if (
            not capacity_path.is_file()
            or _sha256_file(capacity_path) != state["capacity_probe_sha256"]
        ):
            raise BatchFrontierV2Error("durable capacity probe changed")
        capacity = _load_json(capacity_path)
        _require_full_batch_grid(recipe, capacity)
        schedule = expected_blocks(
            recipe,
            {key: tuple(value) for key, value in (state.get("feasible_batches") or {}).items()},
        )
        if state.get("schedule") != [asdict(block) for block in schedule]:
            raise BatchFrontierV2Error("durable schedule changed")

    completed = state.get("completed_blocks")
    hashes = state.get("block_sha256")
    if not isinstance(completed, list) or not isinstance(hashes, dict):
        raise BatchFrontierV2Error("durable block prefix is malformed")
    if completed != [block.relative_path for block in schedule[: len(completed)]]:
        raise BatchFrontierV2Error("completed blocks are not a strict schedule prefix")
    payloads: list[dict[str, Any]] = []
    for block in schedule[: len(completed)]:
        path = _relative(output, block.relative_path)
        if _sha256_file(path) != hashes.get(block.relative_path):
            raise BatchFrontierV2Error(f"durable block changed: {block.relative_path}")
        payload = _load_json(path)
        _validate_block(payload, recipe, block, corpus)
        payloads.append(payload)
    if state.get("status") == COMPLETE_STATUS:
        manifest = output / "manifest.json"
        if _sha256_file(manifest) != state.get("manifest_sha256"):
            raise BatchFrontierV2Error("completed manifest changed")
        return BatchFrontierV2Result(
            output, COMPLETE_STATUS, len(payloads), manifest, _sha256_file(manifest)
        )

    import torch

    for block in schedule[len(payloads) :]:
        _assert_campaign_active(recipe, attestation, process_started)
        torch.cuda.empty_cache()
        path = _relative(output, block.relative_path)
        if path.exists():
            current = state.get("current_block")
            if not isinstance(current, dict) or current.get("relative_path") != block.relative_path:
                raise BatchFrontierV2Error("unanchored block artifact exists")
            payload = _load_json(path)
            _validate_block(payload, recipe, block, corpus)
            digest = _sha256_file(path)
            state["completed_blocks"].append(block.relative_path)
            state["block_sha256"][block.relative_path] = digest
            state["current_block"] = None
            _atomic_write_json(output / "run-state.json", state)
            payloads.append(payload)
            continue
        previous = state.get("current_block")
        if previous is not None:
            if (
                not isinstance(previous, dict)
                or previous.get("relative_path") != block.relative_path
            ):
                raise BatchFrontierV2Error("interrupted attempt is not the next schedule block")
            state["block_attempts"][previous["attempt_index"]]["outcome"] = "interrupted"
        attempt = {
            "attempt_index": len(state["block_attempts"]),
            "relative_path": block.relative_path,
            "round": block.round_index,
            "policy": block.policy,
            "architecture": block.architecture,
            "batch_size": block.batch_size,
            "session_id": attestation["session_id"],
            "started_at": datetime.now(UTC).isoformat(),
            "outcome": "running",
        }
        state["block_attempts"].append(attempt)
        state["current_block"] = {
            "attempt_index": attempt["attempt_index"],
            "relative_path": block.relative_path,
        }
        _atomic_write_json(output / "run-state.json", state)
        if block.policy == "hgs":
            payload = _execute_hgs_block(recipe, block, corpus=corpus, attestation=attestation)
        else:
            payload = _execute_neural_block(
                recipe,
                block,
                corpus=corpus,
                entry=entries[(str(block.architecture), block.training_seed)],
                attestation=attestation,
            )
        _validate_block(payload, recipe, block, corpus)
        _atomic_write_json(path, payload)
        digest = _sha256_file(path)
        attempt.update(
            {
                "outcome": "complete",
                "completed_at": datetime.now(UTC).isoformat(),
                "artifact_sha256": digest,
            }
        )
        state["completed_blocks"].append(block.relative_path)
        state["block_sha256"][block.relative_path] = digest
        state["current_block"] = None
        _atomic_write_json(output / "run-state.json", state)
        payloads.append(payload)
        print(
            json.dumps(
                {
                    "status": "block_complete",
                    "completed": len(payloads),
                    "total": len(schedule),
                    "round": block.round_index,
                    "policy": block.policy,
                    "architecture": block.architecture,
                    "batch_size": block.batch_size,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return _complete(recipe, output, state, capacity, payloads, reference_costs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--training-source", type=Path)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.estimate_output is not None and not args.estimate_only:
            raise BatchFrontierV2QualificationError("--estimate-output requires --estimate-only")
        if args.estimate_only:
            estimate = estimate_only(args.recipe)
            if args.estimate_output is not None:
                destination = args.estimate_output.resolve(strict=False)
                try:
                    destination.relative_to(Path.cwd().resolve(strict=True))
                except ValueError as exc:
                    raise BatchFrontierV2QualificationError(
                        "--estimate-output must remain inside the repository"
                    ) from exc
                _atomic_write_json(destination, estimate)
            print(json.dumps(estimate, indent=2, sort_keys=True), flush=True)
            return 0
        result = run_batch_frontier_v2(
            args.recipe,
            resume=args.resume,
            training_source=args.training_source,
        )
    except BatchFrontierV2Error as exc:
        print(f"batch frontier v2 failed: {exc}", file=sys.stderr, flush=True)
        return 2
    except Exception as exc:
        print(
            f"batch frontier v2 failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "classification": classification(),
                "path": str(result.path),
                "completed_blocks": result.completed_blocks,
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
