"""Native-Windows exploratory AET smoke runner.

The runner deliberately supports one closed workload.  It qualifies the exact
source snapshot and direct CPU/GPU counters before creating output, then runs a
short CVRP training, inference, and PyVRP baseline chain on one shared corpus.
Every promoted bundle is permanently marked exploratory and non-confirmatory.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time
import uuid
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

from neuro_co.aet.experiments.software_recipe import (
    AETSoftwareRecipe,
    SoftwareRun,
    qualify_for_execution,
)

SOFTWARE_SMOKE_MANIFEST_SCHEMA = "aet-software-smoke-manifest/v1"
ROUTE_VALIDATION_SCHEMA = "aet-software-route-validation/v1"
DATASET_MANIFEST_SCHEMA = "aet-software-dataset-manifest/v1"
STAGE_RESULT_SCHEMA = "aet-software-stage-result/v1"
EXCLUSIVE_ATTESTED_ENV = "AET_SOFTWARE_EXCLUSIVE_ATTESTED"
EXCLUSIVE_ATTESTED_AT_ENV = "AET_SOFTWARE_EXCLUSIVE_ATTESTED_AT"
ATTESTATION_MAX_AGE_S = 15 * 60


class SoftwareSmokeError(RuntimeError):
    """Base error for a software smoke that cannot be completed safely."""


class SoftwareSmokeQualificationError(SoftwareSmokeError):
    """Raised before output writes when live qualification is incomplete."""


@dataclass(frozen=True)
class SoftwareSmokeResult:
    """Identity and diagnostic result of one atomically promoted smoke bundle."""

    path: Path
    manifest_path: Path
    manifest_sha256: str
    diagnostic_gap_within_tolerance: bool


@dataclass(frozen=True)
class _Corpus:
    coords: Any
    demands: Any
    capacity: float
    content_sha256: str
    file_sha256: str


@dataclass(frozen=True)
class _TrainingOutcome:
    checkpoint: dict[str, Any]
    energy: dict[str, Any]
    steps: int
    items_processed: int
    last_metrics: dict[str, float]


@dataclass(frozen=True)
class _InferenceOutcome:
    routes: tuple[tuple[tuple[int, ...], ...], ...]
    model_costs: tuple[float, ...]
    energy: dict[str, Any]
    cycles: int
    items_processed: int
    warmup_items: int


@dataclass(frozen=True)
class _BaselineOutcome:
    routes: tuple[tuple[tuple[int, ...], ...], ...]
    integer_costs: tuple[int, ...]
    scaled_costs: tuple[float, ...]
    energy: dict[str, Any]
    cycles: int
    items_processed: int
    warmup_items: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _exclusive_use_attestation() -> dict[str, Any]:
    """Require a recent operator attestation before creating any smoke output."""

    if os.environ.get(EXCLUSIVE_ATTESTED_ENV) != "1":
        raise SoftwareSmokeQualificationError(
            "exclusive-host attestation is absent; use run_software_smoke_windows.ps1"
        )
    raw_timestamp = os.environ.get(EXCLUSIVE_ATTESTED_AT_ENV)
    if not raw_timestamp:
        raise SoftwareSmokeQualificationError("exclusive-host attestation has no UTC timestamp")
    try:
        attested_at = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SoftwareSmokeQualificationError(
            "exclusive-host attestation timestamp is invalid"
        ) from exc
    if attested_at.tzinfo is None:
        raise SoftwareSmokeQualificationError(
            "exclusive-host attestation timestamp must include a UTC offset"
        )
    attested_at = attested_at.astimezone(UTC)
    age_s = (datetime.now(UTC) - attested_at).total_seconds()
    if age_s < -60.0 or age_s > ATTESTATION_MAX_AGE_S:
        raise SoftwareSmokeQualificationError(
            "exclusive-host attestation is stale; restart run_software_smoke_windows.ps1"
        )
    return {
        "attested": True,
        "attested_at": attested_at.isoformat(),
        "statement": "all other CPU and GPU workloads were stopped before launch",
        "operator_supplied": True,
        "exclusivity_basis": "operator_attestation",
        "cpu_exclusivity_verified_by_software": False,
        "gpu_exclusivity_verified_by_software": False,
        "process_gate_enforced": False,
    }


def _nvml_running_processes(pynvml: Any, handle: Any, kind: str) -> tuple[str, list[int]]:
    """Query one NVML process class without treating an error as an empty list."""

    function_names = (
        f"nvmlDeviceGet{kind}RunningProcesses_v3",
        f"nvmlDeviceGet{kind}RunningProcesses_v2",
        f"nvmlDeviceGet{kind}RunningProcesses",
    )
    last_error: BaseException | None = None
    for function_name in function_names:
        function = getattr(pynvml, function_name, None)
        if function is None:
            continue
        for _ in range(3):
            try:
                processes = function(handle)
                return function_name, sorted({int(process.pid) for process in processes})
            except Exception as exc:
                last_error = exc
                error_name = type(exc).__name__
                if error_name == "NVMLError_InsufficientSize":
                    continue
                if error_name in {
                    "NVMLError_FunctionNotFound",
                    "NVMLLibraryMismatchError",
                }:
                    break
                raise SoftwareSmokeError(
                    f"NVML {kind.lower()} process query failed: {error_name}: {exc}"
                ) from exc
    if last_error is None:
        raise SoftwareSmokeError(f"NVML {kind.lower()} process query API is unavailable")
    raise SoftwareSmokeError(
        f"NVML {kind.lower()} process query is unavailable: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _gpu_process_snapshot(gpu_index: int) -> dict[str, Any]:
    """Capture lower-bound process evidence for one physical GPU."""

    try:
        import pynvml  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover, qualified runtime dependency
        raise SoftwareSmokeError(f"pynvml cannot be imported: {exc}") from exc

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        compute_api, compute_pids = _nvml_running_processes(pynvml, handle, "Compute")
        graphics_api, graphics_pids = _nvml_running_processes(pynvml, handle, "Graphics")
        driver_model: list[int] | None = None
        try:
            raw_driver_model = pynvml.nvmlDeviceGetDriverModel(handle)
            if isinstance(raw_driver_model, (list, tuple)):
                driver_model = [int(value) for value in raw_driver_model]
            else:
                driver_model = [int(raw_driver_model)]
        except Exception:
            driver_model = None
        utilization: dict[str, int] | None = None
        try:
            raw_utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            utilization = {
                "gpu_pct": int(raw_utilization.gpu),
                "memory_pct": int(raw_utilization.memory),
            }
        except Exception:
            utilization = None
    except SoftwareSmokeError:
        raise
    except Exception as exc:
        raise SoftwareSmokeError(
            f"NVML GPU process snapshot failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        with suppress(Exception):
            pynvml.nvmlShutdown()

    own_pid = os.getpid()
    snapshot = {
        "captured_at": datetime.now(UTC).isoformat(),
        "gpu_index": gpu_index,
        "own_pid": own_pid,
        "compute_query_api": compute_api,
        "graphics_query_api": graphics_api,
        "compute_pids": compute_pids,
        "graphics_pids": graphics_pids,
        "foreign_compute_pids": [pid for pid in compute_pids if pid != own_pid],
        "foreign_graphics_pids": [pid for pid in graphics_pids if pid != own_pid],
        "driver_model": driver_model,
        "utilization": utilization,
    }
    return snapshot


def _gpu_process_audit(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, Any]:
    """Record endpoint PID snapshots without overriding operator attestation."""

    foreign_compute_pids = sorted(
        set(before["foreign_compute_pids"]) | set(after["foreign_compute_pids"])
    )
    foreign_graphics_pids = sorted(
        set(before["foreign_graphics_pids"]) | set(after["foreign_graphics_pids"])
    )
    return {
        "before": before,
        "after": after,
        "foreign_compute_processes_detected": bool(foreign_compute_pids),
        "foreign_compute_pids": foreign_compute_pids,
        "foreign_graphics_processes_detected": bool(foreign_graphics_pids),
        "foreign_graphics_pids": foreign_graphics_pids,
        "operator_attestation_is_authoritative": True,
        "process_lists_are_diagnostic_only": True,
        "process_gate_enforced": False,
        "continuous_monitoring": False,
        "exclusive_device_verified_by_software": False,
        "scope": "endpoint_process_snapshots_only",
    }


def _relative_path(root: Path, value: str) -> Path:
    return root.joinpath(*PurePosixPath(value).parts)


def _assert_no_symlink_ancestors(root: Path, target: Path) -> None:
    relative = target.relative_to(root)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if (cursor.exists() or cursor.is_symlink()) and cursor.is_symlink():
            raise SoftwareSmokeError(f"refusing symlinked output path component: {cursor}")


def _safe_output_target(workspace_root: Path, output_root: str) -> Path:
    root = workspace_root.resolve(strict=True)
    if root != Path.cwd().resolve(strict=True):
        raise SoftwareSmokeError("workspace_root must be the current repository working directory")
    target = _relative_path(root, output_root)
    _assert_no_symlink_ancestors(root, target)
    resolved = target.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SoftwareSmokeError("output_root resolves outside the repository") from exc
    return resolved


def _content_sha256(coords: Any, demands: Any, capacity: float) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(b"aet-cvrp-corpus/v1\0")
    for name, value in (("coords", coords), ("demands", demands)):
        array = np.ascontiguousarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(array.tobytes(order="C"))
    digest.update(float(capacity).hex().encode("ascii"))
    return digest.hexdigest()


def _generate_corpus(recipe: AETSoftwareRecipe, path: Path) -> _Corpus:
    import numpy as np
    import torch

    from neuro_co.core.factory import make_env

    spec = recipe.workload.dataset
    env = make_env(
        spec.problem,
        size=spec.size,
        capacity=spec.capacity,
        max_demand=spec.max_demand,
    )
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    state = env.reset(spec.num_instances, generator=generator, device="cpu")
    coords = np.ascontiguousarray(state.coords.detach().cpu().numpy(), dtype=np.float32)
    demands = np.ascontiguousarray(state.demand.detach().cpu().numpy(), dtype=np.float32)
    content_sha256 = _content_sha256(coords, demands, spec.capacity)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        coords=coords,
        demands=demands,
        capacity=np.asarray(spec.capacity, dtype=np.float64),
        dataset_seed=np.asarray(spec.seed, dtype=np.int64),
        content_sha256=np.asarray(content_sha256),
    )
    return _Corpus(
        coords=coords,
        demands=demands,
        capacity=spec.capacity,
        content_sha256=content_sha256,
        file_sha256=_sha256_file(path),
    )


def _load_corpus(recipe: AETSoftwareRecipe, path: Path, expected_hash: str) -> _Corpus:
    import numpy as np

    spec = recipe.workload.dataset
    try:
        with np.load(path, allow_pickle=False) as archive:
            coords = np.ascontiguousarray(archive["coords"], dtype=np.float32)
            demands = np.ascontiguousarray(archive["demands"], dtype=np.float32)
            capacity = float(archive["capacity"].item())
            dataset_seed = int(archive["dataset_seed"].item())
            embedded_hash = str(archive["content_sha256"].item())
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise SoftwareSmokeError("shared corpus artifact cannot be loaded") from exc
    expected_coords = (spec.num_instances, spec.size + 1, 2)
    expected_demands = (spec.num_instances, spec.size + 1)
    if coords.shape != expected_coords or demands.shape != expected_demands:
        raise SoftwareSmokeError("shared corpus shape disagrees with the closed recipe")
    if capacity != spec.capacity or dataset_seed != spec.seed:
        raise SoftwareSmokeError("shared corpus metadata disagrees with the closed recipe")
    actual_hash = _content_sha256(coords, demands, capacity)
    if actual_hash != expected_hash or embedded_hash != expected_hash:
        raise SoftwareSmokeError("shared corpus content hash changed")
    return _Corpus(
        coords=coords,
        demands=demands,
        capacity=capacity,
        content_sha256=actual_hash,
        file_sha256=_sha256_file(path),
    )


def _configure_runtime() -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise SoftwareSmokeError(
            "PyTorch inter-op threads were initialized before the controlled smoke"
        ) from exc
    if not torch.cuda.is_available():
        raise SoftwareSmokeError("CUDA became unavailable after successful qualification")
    return {
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _make_tracker(label: str, gpu_index: int) -> Any:
    from neuro_co.aet import EnergyTracker

    return EnergyTracker(
        label,
        backend="hwcounters",
        pue=1.0,
        report_embodied=False,
        allow_fallback=False,
        required_domains={"cpu", "gpu"},
        gpu_indices=[gpu_index],
        items=0,
    )


def _safe_gpu_metadata_value(field: str, value: Any) -> str:
    """Render GPU metadata without exposing a device identifier."""

    if field != "device_id_sha256":
        return repr(value)
    if value is None:
        return "None"
    if isinstance(value, str):
        return f"<redacted SHA-256 string length={len(value)}>"
    return f"<redacted {type(value).__name__}>"


def _checked_energy(
    tracker: Any,
    *,
    minimum_duration_s: float,
    gpu_index: int,
    gpu_device_id_sha256: str,
) -> dict[str, Any]:
    from neuro_co.aet.energy.nvml import NVML_TOTAL_ENERGY_COUNTER

    reading = tracker.reading
    if reading is None:
        raise SoftwareSmokeError("energy tracker returned no reading")
    payload = reading.to_dict()
    if payload.get("backend") != "hwcounters":
        raise SoftwareSmokeError("software smoke did not use direct hardware counters")
    if set(payload.get("energy_domains", ())) != {"cpu", "gpu"}:
        raise SoftwareSmokeError("software smoke did not measure both CPU and GPU components")
    if payload.get("measurement_scope") != "it_components":
        raise SoftwareSmokeError("software smoke measurement scope is not IT components")
    for key in ("energy_j", "energy_cpu_j", "energy_gpu_j"):
        value = payload.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise SoftwareSmokeError(f"{key} must be a positive direct-counter value")
    if payload.get("energy_dram_j") != 0.0:
        raise SoftwareSmokeError("modeled or measured DRAM energy is forbidden in this smoke")
    duration = payload.get("duration_s")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(duration)
        or duration < minimum_duration_s
    ):
        raise SoftwareSmokeError("measured block is shorter than minimum_block_duration_s")
    if payload.get("items_processed", 0) <= 0:
        raise SoftwareSmokeError("measured block processed no workload items")
    hardware = payload.get("hardware")
    if not isinstance(hardware, dict):
        raise SoftwareSmokeError("energy reading has no hardware metadata")
    if hardware.get("attribution_scope") != "machine_wide_component_counters":
        raise SoftwareSmokeError("energy attribution scope is not machine-wide components")
    if hardware.get("exclusive_host_required") is not True:
        raise SoftwareSmokeError("energy reading does not require exclusive host access")
    cpu = hardware.get("cpu_measurement")
    if (
        not isinstance(cpu, dict)
        or cpu.get("measurement_mode") != "windows_emi_cpu_package_counter"
        or cpu.get("codecarbon_version") != "3.3.1"
        or cpu.get("includes_dram") is not False
        or not isinstance(cpu.get("initial_selected_channel_count"), int)
        or cpu["initial_selected_channel_count"] < 1
        or cpu.get("last_measured_channel_count") != cpu["initial_selected_channel_count"]
        or cpu.get("attribution_scope") != "machine_wide_cpu_package"
        or cpu.get("exclusive_host_required") is not True
    ):
        raise SoftwareSmokeError("energy reading lacks the Windows EMI CPU package counter")
    devices = hardware.get("gpu_devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise SoftwareSmokeError("energy reading must bind exactly one GPU")
    device = devices[0]
    if not isinstance(device, dict):
        raise SoftwareSmokeError("energy reading GPU metadata is not an object")
    expected_gpu_metadata = {
        "measurement_mode": NVML_TOTAL_ENERGY_COUNTER,
        "index": gpu_index,
        "device_id_sha256": gpu_device_id_sha256,
        "attribution_scope": "physical_device_all_processes",
        "exclusive_device_required": True,
        "sampling_interval_s": None,
        "sample_count": 2,
        "dropped_sample_count": 0,
    }
    for field, expected in expected_gpu_metadata.items():
        observed = device.get(field)
        if field == "exclusive_device_required":
            matches = observed is True
        elif field == "sampling_interval_s":
            matches = observed is None
        else:
            matches = observed == expected
        if not matches:
            observed_safe = _safe_gpu_metadata_value(field, observed)
            expected_safe = (
                "<qualified GPU SHA-256>"
                if field == "device_id_sha256"
                else _safe_gpu_metadata_value(field, expected)
            )
            raise SoftwareSmokeError(
                f"NVML GPU metadata field {field!r} mismatch: "
                f"observed={observed_safe}; expected={expected_safe}"
            )
    extra = payload.get("extra")
    if not isinstance(extra, dict):
        raise SoftwareSmokeError("energy reading has no tracker metadata")
    if extra.get("tdp_fallback") is not False or extra.get("allow_fallback") is not False:
        raise SoftwareSmokeError("an energy fallback was used")
    return payload


def _run_by_stage(recipe: AETSoftwareRecipe, stage: str) -> SoftwareRun:
    matches = [run for run in recipe.runs if run.stage == stage]
    if len(matches) != 1:  # guarded by the closed recipe; defense in depth
        raise SoftwareSmokeError(f"closed software recipe has no unique {stage} run")
    return matches[0]


def _check_elapsed(start: float, run: SoftwareRun) -> float:
    elapsed = time.perf_counter() - start
    if elapsed > run.max_walltime_s:
        raise SoftwareSmokeError(f"{run.run_id} exceeded its maximum wall time")
    return elapsed


def _execute_training(
    recipe: AETSoftwareRecipe,
    run: SoftwareRun,
    *,
    git_sha: str,
) -> _TrainingOutcome:
    import torch

    from neuro_co.core.factory import make_algo, make_env, make_model

    dataset = recipe.workload.dataset
    model_spec = recipe.workload.model
    training = recipe.workload.training
    device = torch.device(f"cuda:{recipe.gpu_index}")
    torch.manual_seed(run.seed)
    torch.cuda.manual_seed_all(run.seed)
    env = make_env(
        dataset.problem,
        size=dataset.size,
        capacity=dataset.capacity,
        max_demand=dataset.max_demand,
    )
    model = make_model(
        env,
        backbone=model_spec.architecture,
        hidden_dim=model_spec.hidden_dim,
        num_layers=model_spec.num_layers,
        num_heads=model_spec.num_heads,
    )
    algo = make_algo(
        training.algorithm,
        model,
        env,
        device=str(device),
        batch_size=training.batch_size,
        eval_batch_size=training.batch_size,
        lr=training.learning_rate,
    )
    generator = torch.Generator(device=device).manual_seed(run.seed)
    torch.cuda.synchronize(device)
    gpu_processes_before = _gpu_process_snapshot(recipe.gpu_index)
    tracker = _make_tracker(run.run_id, recipe.gpu_index)
    steps = 0
    last_metrics: dict[str, float] = {}
    with tracker as active_tracker:
        started = time.perf_counter()
        while True:
            last_metrics = {key: float(value) for key, value in algo.train_step(generator).items()}
            steps += 1
            torch.cuda.synchronize(device)
            elapsed = _check_elapsed(started, run)
            if (
                steps >= training.minimum_steps
                and elapsed >= recipe.measurement["minimum_block_duration_s"]
            ):
                break
        active_tracker.n_items = steps * training.batch_size
    gpu_processes_after = _gpu_process_snapshot(recipe.gpu_index)
    energy = _checked_energy(
        tracker,
        minimum_duration_s=recipe.measurement["minimum_block_duration_s"],
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    energy["gpu_process_audit"] = _gpu_process_audit(
        gpu_processes_before,
        gpu_processes_after,
    )
    arch = {
        "backbone": model_spec.architecture,
        "hidden_dim": model_spec.hidden_dim,
        "num_layers": model_spec.num_layers,
        "num_heads": model_spec.num_heads,
    }
    checkpoint = {
        "schema_version": "aet-software-checkpoint/v1",
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "arch": arch,
        "problem": dataset.problem,
        "size": dataset.size,
        "capacity": dataset.capacity,
        "max_demand": dataset.max_demand,
        "algorithm": training.algorithm,
        "training_seed": run.seed,
        "training_steps": steps,
        "training_items": steps * training.batch_size,
        "git_sha": git_sha,
    }
    return _TrainingOutcome(
        checkpoint=checkpoint,
        energy=energy,
        steps=steps,
        items_processed=steps * training.batch_size,
        last_metrics=last_metrics,
    )


def _initial_cvrp_state(env: Any, corpus: _Corpus, device: Any) -> Any:
    import torch

    coords = torch.as_tensor(corpus.coords, dtype=torch.float32, device=device)
    demands = torch.as_tensor(corpus.demands, dtype=torch.float32, device=device)
    state = env.reset(demands.shape[0], device=device)
    return state.replace(coords=coords, demand=demands)


def _trace_routes_and_costs(
    trace: Any,
) -> tuple[
    tuple[tuple[tuple[int, ...], ...], ...],
    tuple[float, ...],
]:
    import numpy as np
    import torch

    if trace.reward is None or not trace.steps:
        raise SoftwareSmokeError("neural inference returned an empty trace")
    actions = torch.stack([step.action for step in trace.steps], dim=1).detach().cpu().numpy()
    active = torch.stack([step.active for step in trace.steps], dim=1).detach().cpu().numpy()
    costs = (-trace.reward.detach().cpu()).numpy().astype(np.float64, copy=False)
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
            raise SoftwareSmokeError("neural route trace did not return to the depot")
        routes_by_instance.append(tuple(routes))
    return tuple(routes_by_instance), tuple(float(value) for value in costs)


def _execute_inference(
    recipe: AETSoftwareRecipe,
    run: SoftwareRun,
    *,
    corpus: _Corpus,
    checkpoint_path: Path,
    checkpoint_sha256: str,
) -> _InferenceOutcome:
    import torch

    from neuro_co.core.factory import make_env, make_model
    from neuro_co.core.trace import rollout_trace

    dataset = recipe.workload.dataset
    device = torch.device(f"cuda:{recipe.gpu_index}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise SoftwareSmokeError("training checkpoint is not a mapping")
    expected_metadata = {
        "schema_version": "aet-software-checkpoint/v1",
        "problem": dataset.problem,
        "size": dataset.size,
        "capacity": dataset.capacity,
        "max_demand": dataset.max_demand,
        "algorithm": recipe.workload.training.algorithm,
    }
    if any(checkpoint.get(key) != value for key, value in expected_metadata.items()):
        raise SoftwareSmokeError("training checkpoint metadata disagrees with the recipe")
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise SoftwareSmokeError("training checkpoint changed before inference")
    env = make_env(
        dataset.problem,
        size=dataset.size,
        capacity=dataset.capacity,
        max_demand=dataset.max_demand,
    )
    arch = checkpoint.get("arch")
    if not isinstance(arch, dict):
        raise SoftwareSmokeError("training checkpoint has no architecture mapping")
    model = make_model(env, **arch)
    model.load_state_dict(checkpoint.get("model"), strict=True)
    model.to(device).eval()
    state = _initial_cvrp_state(env, corpus, device)
    with torch.no_grad():
        warmup = rollout_trace(model, env, state, decode="greedy")
        _trace_routes_and_costs(warmup)
    torch.cuda.synchronize(device)

    gpu_processes_before = _gpu_process_snapshot(recipe.gpu_index)
    tracker = _make_tracker(run.run_id, recipe.gpu_index)
    cycles = 0
    routes: tuple[tuple[tuple[int, ...], ...], ...] = ()
    costs: tuple[float, ...] = ()
    with tracker as active_tracker:
        started = time.perf_counter()
        while True:
            with torch.no_grad():
                trace = rollout_trace(model, env, state, decode="greedy")
                routes, costs = _trace_routes_and_costs(trace)
            cycles += 1
            torch.cuda.synchronize(device)
            elapsed = _check_elapsed(started, run)
            if elapsed >= recipe.measurement["minimum_block_duration_s"]:
                break
        active_tracker.n_items = cycles * dataset.num_instances
    gpu_processes_after = _gpu_process_snapshot(recipe.gpu_index)
    energy = _checked_energy(
        tracker,
        minimum_duration_s=recipe.measurement["minimum_block_duration_s"],
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    energy["gpu_process_audit"] = _gpu_process_audit(
        gpu_processes_before,
        gpu_processes_after,
    )
    return _InferenceOutcome(
        routes=routes,
        model_costs=costs,
        energy=energy,
        cycles=cycles,
        items_processed=cycles * dataset.num_instances,
        warmup_items=dataset.num_instances,
    )


def _execute_baseline(
    recipe: AETSoftwareRecipe,
    run: SoftwareRun,
    *,
    corpus: _Corpus,
) -> _BaselineOutcome:
    from neuro_co.problems.cvrp.pyvrp import solve_corpus_sequential

    baseline = recipe.workload.baseline
    warmup = solve_corpus_sequential(
        corpus.coords[:1],
        corpus.demands[:1],
        corpus.capacity,
        seed=run.seed,
        max_iterations=baseline.max_iterations,
        collect_stats=False,
    )
    if len(warmup) != 1:
        raise SoftwareSmokeError("PyVRP warm-up did not solve exactly one instance")

    gpu_processes_before = _gpu_process_snapshot(recipe.gpu_index)
    tracker = _make_tracker(run.run_id, recipe.gpu_index)
    cycles = 0
    results: list[Any] = []
    with tracker as active_tracker:
        started = time.perf_counter()
        while True:
            results = solve_corpus_sequential(
                corpus.coords,
                corpus.demands,
                corpus.capacity,
                seed=run.seed,
                max_iterations=baseline.max_iterations,
                collect_stats=False,
            )
            cycles += 1
            elapsed = _check_elapsed(started, run)
            if elapsed >= recipe.measurement["minimum_block_duration_s"]:
                break
        active_tracker.n_items = cycles * recipe.workload.dataset.num_instances
    gpu_processes_after = _gpu_process_snapshot(recipe.gpu_index)
    energy = _checked_energy(
        tracker,
        minimum_duration_s=recipe.measurement["minimum_block_duration_s"],
        gpu_index=recipe.gpu_index,
        gpu_device_id_sha256=recipe.gpu_device_id_sha256,
    )
    energy["gpu_process_audit"] = _gpu_process_audit(
        gpu_processes_before,
        gpu_processes_after,
    )
    return _BaselineOutcome(
        routes=tuple(result.routes for result in results),
        integer_costs=tuple(int(result.integer_cost) for result in results),
        scaled_costs=tuple(float(result.cost) for result in results),
        energy=energy,
        cycles=cycles,
        items_processed=cycles * recipe.workload.dataset.num_instances,
        warmup_items=1,
    )


def validate_routes(
    coords: Any,
    demands: Any,
    capacity: float,
    routes_by_instance: Any,
) -> dict[str, Any]:
    """Independently validate CVRP coverage, capacity, and Euclidean cost."""

    import operator

    import numpy as np

    points = np.asarray(coords, dtype=np.float64)
    loads = np.asarray(demands, dtype=np.float64)
    routes = tuple(routes_by_instance)
    if points.ndim != 3 or points.shape[2] != 2:
        raise SoftwareSmokeError("validation coords must have shape [batch, nodes, 2]")
    if loads.shape != points.shape[:2]:
        raise SoftwareSmokeError("validation demands do not match coords")
    if len(routes) != points.shape[0]:
        raise SoftwareSmokeError("solution count does not match corpus size")
    if not np.isfinite(points).all() or not np.isfinite(loads).all():
        raise SoftwareSmokeError("validation inputs must be finite")
    if not math.isfinite(capacity) or capacity <= 0:
        raise SoftwareSmokeError("validation capacity must be finite and positive")

    costs: list[float] = []
    violations: list[dict[str, Any]] = []
    num_clients = points.shape[1] - 1
    expected_clients = list(range(1, num_clients + 1))
    for instance_index, raw_routes in enumerate(routes):
        flattened: list[int] = []
        total_cost = 0.0
        for route_index, raw_route in enumerate(raw_routes):
            route: list[int] = []
            for raw_customer in raw_route:
                try:
                    customer = operator.index(raw_customer)
                except TypeError:
                    violations.append(
                        {
                            "instance": instance_index,
                            "route": route_index,
                            "kind": "non_integer_customer",
                        }
                    )
                    continue
                route.append(customer)
            if not route:
                violations.append(
                    {
                        "instance": instance_index,
                        "route": route_index,
                        "kind": "empty_route",
                    }
                )
                continue
            invalid = [customer for customer in route if customer < 1 or customer > num_clients]
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
            sequence = [0, *route, 0]
            for origin, destination in pairwise(sequence):
                total_cost += float(
                    np.linalg.norm(
                        points[instance_index, origin] - points[instance_index, destination]
                    )
                )
        if sorted(flattened) != expected_clients:
            missing = sorted(set(expected_clients) - set(flattened))
            duplicates = sorted(
                customer for customer in set(flattened) if flattened.count(customer) > 1
            )
            violations.append(
                {
                    "instance": instance_index,
                    "kind": "customer_coverage",
                    "missing": missing,
                    "duplicates": duplicates,
                }
            )
        costs.append(total_cost)
    complete = not violations
    return {
        "schema_version": ROUTE_VALIDATION_SCHEMA,
        "complete": complete,
        "independent": True,
        "validator": "euclidean-cvrp-route-checker",
        "instance_count": points.shape[0],
        "validated_instance_count": points.shape[0] if complete else 0,
        "failure_count": len(violations),
        "costs": costs,
        "mean_cost": float(np.mean(costs)),
        "violations": violations,
    }


def _assert_valid(validation: dict[str, Any], label: str) -> None:
    if validation.get("complete") is not True:
        raise SoftwareSmokeError(f"{label} returned infeasible or incomplete CVRP routes")


def _stage_record(
    *,
    run: SoftwareRun,
    corpus: _Corpus,
    energy: dict[str, Any],
    cycles: int,
    items_processed: int,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": STAGE_RESULT_SCHEMA,
        "classification": {
            "purpose": "software_exploratory",
            "scientific_use": False,
            "confirmatory_eligible": False,
        },
        "run": asdict(run),
        "dataset_content_sha256": corpus.content_sha256,
        "dataset_file_sha256": corpus.file_sha256,
        "workload_cycle_count": cycles,
        "items_processed": items_processed,
        "energy": energy,
        "details": details,
    }


def _library_versions() -> dict[str, str | None]:
    distributions = {
        "torch": "torch",
        "numpy": "numpy",
        "pyvrp": "pyvrp",
        "codecarbon": "codecarbon",
        "pynvml": "nvidia-ml-py",
    }
    versions: dict[str, str | None] = {}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = None
    return versions


def _write_checksums(root: Path) -> Path:
    checksum_path = root / "SHA256SUMS"
    entries = []
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        if path == checksum_path:
            continue
        entries.append(f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}")
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    for entry in entries:
        expected, relative = entry.split("  ", maxsplit=1)
        if _sha256_file(root / relative) != expected:
            raise SoftwareSmokeError(f"artifact checksum verification failed: {relative}")
    return checksum_path


def execute_aet_software_smoke(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> SoftwareSmokeResult:
    """Execute and atomically promote the qualified exploratory software smoke."""

    root = Path.cwd() if workspace_root is None else Path(workspace_root)
    root = root.resolve(strict=True)
    source_recipe_path = Path(recipe_path).resolve(strict=True)
    try:
        source_recipe_path.relative_to(root)
    except ValueError as exc:
        raise SoftwareSmokeQualificationError(
            "software recipe must be stored inside the current repository"
        ) from exc
    recipe_before = source_recipe_path.read_bytes()
    recipe, qualification = qualify_for_execution(source_recipe_path)
    if qualification.get("ready_to_execute") is not True:
        raise SoftwareSmokeQualificationError(
            "native-Windows software smoke qualification is incomplete; no output was written"
        )
    if source_recipe_path.read_bytes() != recipe_before:
        raise SoftwareSmokeQualificationError("software recipe changed during qualification")
    preflight_path = _relative_path(root, recipe.preflight_report).resolve(strict=True)
    try:
        preflight_path.relative_to(root)
    except ValueError as exc:
        raise SoftwareSmokeQualificationError(
            "preflight report resolves outside the repository"
        ) from exc
    preflight_before = preflight_path.read_bytes()
    recipe_after, qualification_after = qualify_for_execution(source_recipe_path)
    if recipe_after != recipe or qualification_after != qualification:
        raise SoftwareSmokeQualificationError("execution qualification changed before launch")
    if preflight_path.read_bytes() != preflight_before:
        raise SoftwareSmokeQualificationError("preflight report changed before launch")
    try:
        lockfile_path = (root / "uv.lock").resolve(strict=True)
        lockfile_before = lockfile_path.read_bytes()
    except OSError as exc:
        raise SoftwareSmokeQualificationError("uv.lock is unavailable") from exc
    exclusive_use_attestation = _exclusive_use_attestation()
    try:
        launch_gpu_process_snapshot = _gpu_process_snapshot(recipe.gpu_index)
    except SoftwareSmokeError as exc:
        raise SoftwareSmokeQualificationError(str(exc)) from exc

    target = _safe_output_target(root, recipe.output_root)
    if target.exists() or target.is_symlink():
        raise SoftwareSmokeError(f"refusing to overwrite existing software smoke: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".{target.name}.partial-{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():  # pragma: no cover, UUID collision
        raise SoftwareSmokeError("software smoke staging path already exists")

    started_at = datetime.now(UTC)
    staging.mkdir()
    try:
        (staging / "qualification").mkdir()
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_before)
        (staging / "qualification" / "preflight.json").write_bytes(preflight_before)
        (staging / "environment" / "uv.lock").write_bytes(lockfile_before)
        runtime_controls = _configure_runtime()

        dataset_path = _relative_path(staging, recipe.workload.dataset.artifact)
        corpus = _generate_corpus(recipe, dataset_path)
        _write_json(
            dataset_path.with_suffix(".manifest.json"),
            {
                "schema_version": DATASET_MANIFEST_SCHEMA,
                "complete": True,
                "dataset": asdict(recipe.workload.dataset),
                "content_sha256": corpus.content_sha256,
                "file_sha256": corpus.file_sha256,
                "coords_shape": list(corpus.coords.shape),
                "demands_shape": list(corpus.demands.shape),
            },
        )

        git_sha = qualification.get("current_git_commit")
        if not isinstance(git_sha, str) or not git_sha:
            raise SoftwareSmokeQualificationError("qualified Git SHA is unavailable")
        training_run = _run_by_stage(recipe, "training")
        training = _execute_training(recipe, training_run, git_sha=git_sha)
        checkpoint_path = _relative_path(staging, recipe.workload.training.checkpoint_artifact)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(training.checkpoint, checkpoint_path)
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        training_stage = _stage_record(
            run=training_run,
            corpus=corpus,
            energy=training.energy,
            cycles=training.steps,
            items_processed=training.items_processed,
            details={
                "algorithm": recipe.workload.training.algorithm,
                "batch_size": recipe.workload.training.batch_size,
                "training_steps": training.steps,
                "last_metrics": training.last_metrics,
                "checkpoint": checkpoint_path.relative_to(staging).as_posix(),
                "checkpoint_sha256": checkpoint_sha256,
            },
        )
        training_stage_path = _relative_path(staging, training_run.output_subdir) / "result.json"
        _write_json(training_stage_path, training_stage)

        inference_corpus = _load_corpus(recipe, dataset_path, corpus.content_sha256)
        inference_run = _run_by_stage(recipe, "inference")
        inference = _execute_inference(
            recipe,
            inference_run,
            corpus=inference_corpus,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha256,
        )
        inference_validation = validate_routes(
            inference_corpus.coords,
            inference_corpus.demands,
            inference_corpus.capacity,
            inference.routes,
        )
        _assert_valid(inference_validation, "neural inference")
        validated_inference_costs = tuple(float(value) for value in inference_validation["costs"])
        max_model_cost_error = max(
            abs(left - right)
            for left, right in zip(
                inference.model_costs,
                validated_inference_costs,
                strict=True,
            )
        )
        if max_model_cost_error > 1e-4:
            raise SoftwareSmokeError(
                "neural model costs disagree with independent route validation"
            )
        inference_dir = _relative_path(staging, inference_run.output_subdir)
        _write_json(inference_dir / "validation.json", inference_validation)
        _write_json(
            inference_dir / "solutions.json",
            {
                "dataset_content_sha256": corpus.content_sha256,
                "checkpoint_sha256": checkpoint_sha256,
                "routes": inference.routes,
                "model_costs": inference.model_costs,
            },
        )
        inference_stage = _stage_record(
            run=inference_run,
            corpus=corpus,
            energy=inference.energy,
            cycles=inference.cycles,
            items_processed=inference.items_processed,
            details={
                "policy": "am-greedy",
                "checkpoint_from": recipe.workload.inference.checkpoint_from,
                "checkpoint_sha256": checkpoint_sha256,
                "warmup_items": inference.warmup_items,
                "max_model_cost_validation_error": max_model_cost_error,
            },
        )
        inference_stage_path = inference_dir / "result.json"
        _write_json(inference_stage_path, inference_stage)

        baseline_corpus = _load_corpus(recipe, dataset_path, corpus.content_sha256)
        baseline_run = _run_by_stage(recipe, "baseline")
        baseline = _execute_baseline(recipe, baseline_run, corpus=baseline_corpus)
        baseline_validation = validate_routes(
            baseline_corpus.coords,
            baseline_corpus.demands,
            baseline_corpus.capacity,
            baseline.routes,
        )
        _assert_valid(baseline_validation, "PyVRP baseline")
        validated_baseline_costs = tuple(float(value) for value in baseline_validation["costs"])
        max_scaled_cost_error = max(
            abs(left - right)
            for left, right in zip(
                baseline.scaled_costs,
                validated_baseline_costs,
                strict=True,
            )
        )
        if max_scaled_cost_error > 1e-4:
            raise SoftwareSmokeError(
                "PyVRP scaled costs disagree with independent route validation"
            )
        baseline_dir = _relative_path(staging, baseline_run.output_subdir)
        _write_json(baseline_dir / "validation.json", baseline_validation)
        _write_json(
            baseline_dir / "solutions.json",
            {
                "dataset_content_sha256": corpus.content_sha256,
                "routes": baseline.routes,
                "integer_costs": baseline.integer_costs,
                "scaled_costs": baseline.scaled_costs,
            },
        )
        baseline_stage = _stage_record(
            run=baseline_run,
            corpus=corpus,
            energy=baseline.energy,
            cycles=baseline.cycles,
            items_processed=baseline.items_processed,
            details={
                "solver": recipe.workload.baseline.solver,
                "max_iterations": recipe.workload.baseline.max_iterations,
                "warmup_items": baseline.warmup_items,
                "max_scaled_cost_validation_error": max_scaled_cost_error,
            },
        )
        baseline_stage_path = baseline_dir / "result.json"
        _write_json(baseline_stage_path, baseline_stage)

        if any(reference <= 0.0 for reference in validated_baseline_costs):
            raise SoftwareSmokeError("quality reference costs must be strictly positive")
        import numpy as np

        gaps = tuple(
            100.0 * (neural - reference) / reference
            for neural, reference in zip(
                validated_inference_costs,
                validated_baseline_costs,
                strict=True,
            )
        )
        if any(not math.isfinite(value) for value in gaps):
            raise SoftwareSmokeError("quality gap contains a non-finite value")
        mean_gap_pct = sum(gaps) / len(gaps)
        diagnostic_gap_within_tolerance = mean_gap_pct <= recipe.workload.quality.maximum_gap_pct
        quality = {
            "classification": {
                "purpose": "software_exploratory",
                "scientific_use": False,
                "confirmatory_eligible": False,
            },
            "metric": recipe.workload.quality.metric,
            "reference": "pyvrp-hgs-smoke-budget",
            "independent_reference": False,
            "confirmatory_quality_status": "not_evaluable",
            "confirmatory_quality_reason": "no_independent_reference",
            "maximum_gap_pct": recipe.workload.quality.maximum_gap_pct,
            "mean_gap_pct": mean_gap_pct,
            "median_gap_pct": float(np.median(gaps)),
            "maximum_observed_gap_pct": max(gaps),
            "diagnostic_gap_within_tolerance": diagnostic_gap_within_tolerance,
            "gaps_pct": gaps,
        }
        _write_json(staging / "quality.json", quality)

        final_recipe, final_qualification = qualify_for_execution(source_recipe_path)
        if final_recipe != recipe or final_qualification != qualification:
            raise SoftwareSmokeQualificationError(
                "source or execution qualification changed during the smoke"
            )
        if source_recipe_path.read_bytes() != recipe_before:
            raise SoftwareSmokeQualificationError("software recipe changed during the smoke")
        if preflight_path.read_bytes() != preflight_before:
            raise SoftwareSmokeQualificationError("preflight report changed during the smoke")
        if lockfile_path.read_bytes() != lockfile_before:
            raise SoftwareSmokeQualificationError("uv.lock changed during the smoke")

        ended_at = datetime.now(UTC)
        manifest = {
            "schema_version": SOFTWARE_SMOKE_MANIFEST_SCHEMA,
            "status": "complete",
            "classification": {
                "purpose": "software_exploratory",
                "scientific_use": False,
                "confirmatory_eligible": False,
                "whole_system_energy": False,
                "cross_solver_energy_comparable": False,
            },
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "recipe": {
                "path": "recipe.yaml",
                "sha256": _sha256_bytes(recipe_before),
                "name": recipe.name,
            },
            "qualification": {
                "preflight_path": "qualification/preflight.json",
                "preflight_sha256": _sha256_bytes(preflight_before),
                "host_id": recipe.host_id,
                "execution_layer": "windows-native",
                "git_sha": git_sha,
                "git_dirty": qualification.get("current_worktree_dirty"),
                "worktree_fingerprint_sha256": qualification.get(
                    "current_worktree_fingerprint_sha256"
                ),
                "source_reconstructible_from_git_commit": not bool(
                    qualification.get("current_worktree_dirty")
                ),
                "dirty_source_reconstruction_limitation": (
                    "dirty tracked and untracked content is fingerprinted but not copied"
                    if qualification.get("current_worktree_dirty")
                    else None
                ),
                "gpu_index": recipe.gpu_index,
                "gpu_device_id_sha256": recipe.gpu_device_id_sha256,
            },
            "measurement": {
                **recipe.measurement,
                "whole_system_energy": False,
                "cross_solver_energy_comparable": False,
                "comparison_exclusion_reasons": [
                    "machine_wide_component_counters",
                    "gpu_idle_energy_in_cpu_baseline",
                    "unequal_setup_boundaries",
                    "no_wall_meter",
                ],
                "walltime_limit_enforcement": (
                    "cooperative_after_complete_training_steps_rollouts_or_corpora"
                ),
                "excluded_components": [
                    "dram",
                    "storage",
                    "motherboard",
                    "psu_losses",
                    "cooling",
                ],
            },
            "runtime_controls": runtime_controls,
            "environment": {
                "lockfile": "environment/uv.lock",
                "lockfile_sha256": _sha256_bytes(lockfile_before),
            },
            "exclusive_use_attestation": exclusive_use_attestation,
            "launch_gpu_process_snapshot": launch_gpu_process_snapshot,
            "library_versions": _library_versions(),
            "workload": {
                "dataset": asdict(recipe.workload.dataset),
                "model": asdict(recipe.workload.model),
                "training": asdict(recipe.workload.training),
                "inference": asdict(recipe.workload.inference),
                "baseline": asdict(recipe.workload.baseline),
                "quality": asdict(recipe.workload.quality),
                "duration_policy": recipe.workload.duration_policy,
            },
            "dataset": {
                "path": dataset_path.relative_to(staging).as_posix(),
                "content_sha256": corpus.content_sha256,
                "file_sha256": corpus.file_sha256,
            },
            "stages": {
                "training": training_stage_path.relative_to(staging).as_posix(),
                "inference": inference_stage_path.relative_to(staging).as_posix(),
                "baseline": baseline_stage_path.relative_to(staging).as_posix(),
            },
            "quality": quality,
        }
        manifest_path = staging / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_checksums(staging)
        staging.rename(target)
        final_manifest = target / "manifest.json"
        return SoftwareSmokeResult(
            path=target,
            manifest_path=final_manifest,
            manifest_sha256=_sha256_file(final_manifest),
            diagnostic_gap_within_tolerance=diagnostic_gap_within_tolerance,
        )
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def main(argv: list[str] | None = None) -> int:
    """Run one software smoke recipe."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m neuro_co.aet.experiments.software_smoke_runner"
    )
    parser.add_argument("recipe", type=Path)
    args = parser.parse_args(argv)
    try:
        result = execute_aet_software_smoke(args.recipe)
    except SoftwareSmokeQualificationError as exc:
        print(f"software smoke not qualified: {exc}", file=sys.stderr, flush=True)
        return 2
    except SoftwareSmokeError as exc:
        print(f"software smoke failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": "complete",
                "purpose": "software_exploratory",
                "scientific_use": False,
                "confirmatory_eligible": False,
                "confirmatory_quality_status": "not_evaluable",
                "cross_solver_energy_comparable": False,
                "path": result.path.as_posix(),
                "manifest": result.manifest_path.as_posix(),
                "manifest_sha256": result.manifest_sha256,
                "diagnostic_gap_within_tolerance": (result.diagnostic_gap_within_tolerance),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


__all__ = [
    "SoftwareSmokeError",
    "SoftwareSmokeQualificationError",
    "SoftwareSmokeResult",
    "execute_aet_software_smoke",
    "main",
    "validate_routes",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
