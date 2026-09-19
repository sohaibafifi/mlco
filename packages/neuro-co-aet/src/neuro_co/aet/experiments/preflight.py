"""Report whether a host is safe for a measured AET smoke or primary run.

This command performs no solver training or evaluation. With `--active-probe`,
it opens and closes direct hardware counters for a short qualification sample.
It never records hostnames, serial numbers, hardware UUIDs, or environment
variables.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neuro_co.aet.experiments.journal_recipe import (
    CalibrationValidationError,
    validate_calibration_manifest,
)

SCHEMA_VERSION = "aet-preflight/v1"
SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX_EXCLUSIVE = (3, 13)
REQUIRED_CODECARBON_WINDOWS_EMI = "3.3.1"


def command(*args: str, cwd: Path | None = None, binary: bool = False) -> bytes | str:
    return subprocess.check_output(
        list(args),
        cwd=cwd,
        stderr=subprocess.DEVNULL,
        text=not binary,
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def public_path(path: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def git_snapshot(root: Path, excluded: set[Path]) -> dict[str, Any]:
    try:
        sha = str(command("git", "rev-parse", "HEAD", cwd=root)).strip()
        branch = str(command("git", "branch", "--show-current", cwd=root)).strip()
        diff = command("git", "diff", "--binary", "HEAD", cwd=root, binary=True)
        untracked_raw = command(
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
            cwd=root,
            binary=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    digest = hashlib.sha256()
    digest.update(diff if isinstance(diff, bytes) else diff.encode())
    untracked_paths: list[str] = []
    raw_bytes = untracked_raw if isinstance(untracked_raw, bytes) else untracked_raw.encode()
    for raw in sorted(part for part in raw_bytes.split(b"\0") if part):
        rel = raw.decode("utf-8", errors="surrogateescape")
        candidate = (root / rel).resolve()
        if candidate in excluded or not candidate.is_file():
            continue
        untracked_paths.append(rel)
        digest.update(rel.encode("utf-8", errors="surrogateescape"))
        digest.update(bytes.fromhex(sha256_file(candidate)))
    diff_bytes = diff if isinstance(diff, bytes) else diff.encode()
    return {
        "available": True,
        "sha": sha,
        "branch": branch,
        "dirty": bool(diff_bytes) or bool(untracked_paths),
        "worktree_fingerprint_sha256": digest.hexdigest(),
        "untracked_file_count": len(untracked_paths),
    }


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def version_is_exact(value: str | None, required: str) -> bool:
    """Accept one stable dependency release when private interfaces are pinned."""
    if value is None or value != required:
        return False
    return re.fullmatch(r"\d+\.\d+\.\d+", value) is not None


def import_probe(module: str) -> dict[str, Any]:
    try:
        importlib.import_module(module)
        return {"importable": True}
    except Exception as exc:
        return {"importable": False, "error": f"{type(exc).__name__}: {exc}"}


def strict_tracker_probe() -> dict[str, Any]:
    try:
        from neuro_co.aet import EnergyTracker

        parameters = inspect.signature(EnergyTracker.__init__).parameters
        required_parameters = {
            "allow_fallback",
            "required_domains",
            "gpu_indices",
            "wall_meter_sampler",
        }
        missing_parameters = sorted(required_parameters - set(parameters))

        class _ContractProbe:
            @staticmethod
            def start() -> None:
                return None

            @staticmethod
            def stop() -> Any:
                return None

        wall_meter_contract_supported = False
        component_contract_supported = False
        if not missing_parameters:
            tracker = EnergyTracker(
                backend="wall_meter",
                wall_meter_sampler=_ContractProbe(),
                pue=1.0,
                report_embodied=False,
                allow_fallback=False,
            )
            wall_meter_contract_supported = tracker._resolve_chain(
                "wall_meter", allow_fallback=False
            ) == ["wall_meter"]
            component_tracker = EnergyTracker(
                backend="hwcounters",
                pue=1.0,
                report_embodied=False,
                allow_fallback=False,
                required_domains={"cpu", "gpu"},
                gpu_indices=[0],
            )
            component_contract_supported = component_tracker._resolve_chain(
                "hwcounters", allow_fallback=False
            ) == ["hwcounters"]
        supported = (
            not missing_parameters
            and wall_meter_contract_supported
            and component_contract_supported
        )
        return {
            "importable": True,
            "supported": supported,
            "required_parameters": sorted(required_parameters),
            "missing_parameters": missing_parameters,
            "wall_meter_contract_supported": wall_meter_contract_supported,
            "component_contract_supported": component_contract_supported,
        }
    except Exception as exc:
        return {
            "importable": False,
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _optional_nvml_call(callable_value: Any, *args: Any) -> Any:
    try:
        return callable_value(*args)
    except Exception:
        return None


def _decode_nvml_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _gpu_identity_sha256(value: Any) -> str | None:
    """Hash a normalized GPU UUID without retaining the source identifier."""
    text = _decode_nvml_text(value)
    if text is None:
        return None
    normalized = text.strip().lower()
    if normalized.startswith("gpu-"):
        normalized = normalized[4:]
    normalized = normalized.strip("{}")
    if not normalized:
        return None
    return sha256_bytes(normalized.encode("utf-8"))


def _torch_uuid(torch: Any, index: int) -> Any:
    try:
        properties = torch.cuda.get_device_properties(index)
    except Exception:
        return None
    return getattr(properties, "uuid", None)


def gpu_qualification_probe(
    active: bool,
    selected_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Qualify one physical GPU with CUDA work bracketed by its NVML counter."""
    nvml_result: dict[str, Any] = {
        "available": False,
        "active_probe": active,
        "device_count": 0,
        "selected_index": selected_index,
        "selection_valid": False,
        "selected_device": None,
        "devices": [],
    }
    torch_result: dict[str, Any] = {
        "available": False,
        "active_probe": active,
        "selected_index": selected_index,
        "requested_nvml_index": selected_index,
        "selected_cuda_index": None,
        "selection_valid": False,
        "identity_verified": False,
        "identity_verification_mode": "unavailable",
        "identity_mismatch": False,
        "visibility_remapped": False,
        "compute_probe_passed": False,
        "nvml_counter_bracketed": False,
        "nvml_counter_positive": False,
        "nvml_sample_energy_delta_mj": None,
    }

    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
    except Exception as exc:
        nvml_result["error"] = f"{type(exc).__name__}: {exc}"
        return nvml_result, torch_result

    handles: dict[int, Any] = {}
    identity_hashes: dict[int, str | None] = {}
    devices: list[dict[str, Any]] = []
    try:
        driver_version = _decode_nvml_text(_optional_nvml_call(pynvml.nvmlSystemGetDriverVersion))
        cuda_driver_version_raw = _optional_nvml_call(pynvml.nvmlSystemGetCudaDriverVersion)
        if isinstance(cuda_driver_version_raw, int):
            cuda_driver_version = (
                f"{cuda_driver_version_raw // 1000}.{(cuda_driver_version_raw % 1000) // 10}"
            )
        else:
            cuda_driver_version = None

        count = int(pynvml.nvmlDeviceGetCount())
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            handles[index] = handle
            name = _decode_nvml_text(pynvml.nvmlDeviceGetName(handle)) or "unknown"
            memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
            raw_uuid = _optional_nvml_call(pynvml.nvmlDeviceGetUUID, handle)
            raw_uuid_text = _decode_nvml_text(raw_uuid)
            device_id_sha256 = (
                sha256_bytes(raw_uuid_text.encode("utf-8")) if raw_uuid_text is not None else None
            )
            identity_sha256 = _gpu_identity_sha256(raw_uuid)
            identity_hashes[index] = identity_sha256
            power_limit_mw = _optional_nvml_call(pynvml.nvmlDeviceGetPowerManagementLimit, handle)
            power_limit_w = (
                float(power_limit_mw) / 1000.0 if isinstance(power_limit_mw, (int, float)) else None
            )
            persistence_mode = _optional_nvml_call(pynvml.nvmlDeviceGetPersistenceMode, handle)
            current_graphics_clock_mhz = _optional_nvml_call(
                pynvml.nvmlDeviceGetClockInfo,
                handle,
                getattr(pynvml, "NVML_CLOCK_GRAPHICS", 0),
            )
            current_memory_clock_mhz = _optional_nvml_call(
                pynvml.nvmlDeviceGetClockInfo,
                handle,
                getattr(pynvml, "NVML_CLOCK_MEM", 2),
            )
            try:
                int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                counter_supported = True
            except Exception:
                counter_supported = False
            sample_power_w: float | None = None
            if counter_supported:
                power_sampling_supported = True
            else:
                try:
                    sample_power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                    power_sampling_supported = (
                        math.isfinite(sample_power_w) and sample_power_w >= 0.0
                    )
                except Exception:
                    power_sampling_supported = False
            measurement_mode = "total_energy_counter" if counter_supported else "power_integration"
            devices.append(
                {
                    "index": index,
                    "name": name,
                    "memory_total_b": memory,
                    "device_id_sha256": device_id_sha256,
                    "identity_sha256": identity_sha256,
                    "power_limit_w": power_limit_w,
                    "persistence_mode": persistence_mode,
                    "current_graphics_clock_mhz": current_graphics_clock_mhz,
                    "current_memory_clock_mhz": current_memory_clock_mhz,
                    "total_energy_counter_supported": counter_supported,
                    "counter_monotonic": None,
                    "counter_positive": None,
                    "counter_bracketed_cuda_work": False,
                    "sample_energy_delta_mj": None,
                    "power_sampling_supported": power_sampling_supported,
                    "sample_power_w": sample_power_w,
                    "measurement_mode": measurement_mode,
                    "measurement_mode_supported": (counter_supported or power_sampling_supported),
                    "selected": index == selected_index,
                }
            )

        selected = next(
            (device for device in devices if device["index"] == selected_index),
            None,
        )
        nvml_result.update(
            {
                "available": bool(devices),
                "device_count": len(devices),
                "selection_valid": selected is not None,
                "driver_version": driver_version,
                "cuda_driver_version": cuda_driver_version,
                "selected_device": selected,
                "devices": devices,
            }
        )

        try:
            import torch
        except Exception as exc:
            torch_result["error"] = f"{type(exc).__name__}: {exc}"
            return nvml_result, torch_result

        torch_result["torch_cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
            cuda_device_count = int(torch.cuda.device_count()) if cuda_available else 0
        except Exception as exc:
            torch_result["error"] = f"{type(exc).__name__}: {exc}"
            return nvml_result, torch_result
        torch_result.update(
            {
                "available": cuda_available,
                "device_count": cuda_device_count,
            }
        )
        if not cuda_available or selected is None:
            return nvml_result, torch_result

        torch_identities = {
            index: _gpu_identity_sha256(_torch_uuid(torch, index))
            for index in range(cuda_device_count)
        }
        selected_identity = identity_hashes.get(selected_index)
        identified_cuda_devices = {
            index: identity for index, identity in torch_identities.items() if identity is not None
        }
        cuda_index: int | None = None
        if selected_identity is not None and identified_cuda_devices:
            matches = [
                index
                for index, identity in identified_cuda_devices.items()
                if identity == selected_identity
            ]
            if len(matches) == 1:
                cuda_index = matches[0]
                torch_result.update(
                    {
                        "identity_verified": True,
                        "identity_verification_mode": "uuid_hash",
                    }
                )
            else:
                torch_result.update(
                    {
                        "identity_mismatch": True,
                        "identity_verification_mode": "uuid_hash_mismatch",
                    }
                )
        elif count == 1 and cuda_device_count == 1 and selected_index == 0:
            cuda_index = 0
            torch_result.update(
                {
                    "identity_verified": True,
                    "identity_verification_mode": "single_device_count",
                }
            )

        if cuda_index is None:
            return nvml_result, torch_result
        torch_result.update(
            {
                "selected_cuda_index": cuda_index,
                "selection_valid": True,
                "visibility_remapped": cuda_index != selected_index,
                "selected_device_identity_sha256": torch_identities.get(cuda_index),
            }
        )
        try:
            torch_result["selected_device_name"] = str(torch.cuda.get_device_name(cuda_index))
        except Exception as exc:
            torch_result["selection_valid"] = False
            torch_result["error"] = f"{type(exc).__name__}: {exc}"
            return nvml_result, torch_result
        if not active:
            return nvml_result, torch_result
        if selected.get("total_energy_counter_supported") is not True:
            return nvml_result, torch_result

        handle = handles[selected_index]
        start_mj: int | None = None
        stop_mj: int | None = None
        try:
            start_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            device = torch.device(f"cuda:{cuda_index}")
            left = torch.ones((2048, 2048), device=device)
            product = left
            for _ in range(8):
                product = left @ left
            checksum = float(product.sum().item())
            torch.cuda.synchronize(device)
            torch_result.update(
                {
                    "compute_probe_passed": (math.isfinite(checksum) and checksum > 0.0),
                    "compute_probe_checksum": checksum,
                    "compute_probe_matrix_size": 2048,
                    "compute_probe_repetitions": 8,
                }
            )
        except Exception as exc:
            torch_result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if start_mj is not None:
                try:
                    stop_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                except Exception as exc:
                    torch_result["counter_error"] = f"{type(exc).__name__}: {exc}"

        if start_mj is not None and stop_mj is not None:
            delta_mj = stop_mj - start_mj
            counter_monotonic = delta_mj >= 0
            counter_positive = delta_mj > 0
            selected.update(
                {
                    "counter_monotonic": counter_monotonic,
                    "counter_positive": counter_positive,
                    "counter_bracketed_cuda_work": True,
                    "sample_energy_delta_mj": delta_mj,
                }
            )
            torch_result.update(
                {
                    "nvml_counter_bracketed": True,
                    "nvml_counter_positive": counter_positive,
                    "nvml_sample_energy_delta_mj": delta_mj,
                }
            )
        return nvml_result, torch_result
    except Exception as exc:
        nvml_result["error"] = f"{type(exc).__name__}: {exc}"
        return nvml_result, torch_result
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()


def nvml_probe(active: bool, selected_index: int) -> dict[str, Any]:
    """Compatibility wrapper around the joint GPU qualification probe."""
    return gpu_qualification_probe(active, selected_index)[0]


def torch_cuda_probe(active: bool, selected_index: int) -> dict[str, Any]:
    """Compatibility wrapper around the joint GPU qualification probe."""
    return gpu_qualification_probe(active, selected_index)[1]


def rapl_probe(active: bool) -> dict[str, Any]:
    sysfs_files = sorted(Path("/sys/class/powercap").glob("**/energy_uj"))
    readable = [str(path) for path in sysfs_files if os.access(path, os.R_OK)]
    pyrapl_version = distribution_version("pyRAPL")
    result: dict[str, Any] = {
        "available": False,
        "active_probe": active,
        "platform_linux": sys.platform.startswith("linux"),
        "pyrapl_version": pyrapl_version,
        "readable_energy_files": readable,
    }
    if not sys.platform.startswith("linux") or pyrapl_version is None:
        return result
    if not active:
        result["available"] = bool(readable)
        return result
    try:
        from neuro_co.aet.energy.rapl import RaplSampler

        sampler = RaplSampler()
        sampler.start()
        deadline = time.perf_counter() + 0.05
        value = 0
        while time.perf_counter() < deadline:
            value += 1
        energy_wh = sampler.stop()
        result.update(
            {
                "available": energy_wh >= 0.0,
                "sample_energy_wh": energy_wh,
                "sample_work": value,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def windows_emi_probe(active: bool) -> dict[str, Any]:
    """Probe Windows 11 CPU package energy through CodeCarbon's EMI backend."""
    codecarbon_version = distribution_version("codecarbon")
    version_supported = version_is_exact(
        codecarbon_version,
        REQUIRED_CODECARBON_WINDOWS_EMI,
    )
    result: dict[str, Any] = {
        "available": False,
        "active_probe": active,
        "platform_windows": sys.platform == "win32",
        "codecarbon_version": codecarbon_version,
        "required_codecarbon_version": REQUIRED_CODECARBON_WINDOWS_EMI,
        "version_supported": version_supported,
        "backend_mode": "windows_emi",
        "interface_class": "WindowsEMI",
        "fallback_used": False,
        "measurement_scope": "cpu_package",
        "ram_included": False,
    }
    if sys.platform != "win32" or not version_supported:
        return result
    try:
        from neuro_co.aet.energy.windows_emi import WindowsEmiSampler

        sampler = WindowsEmiSampler()
        if not active:
            result["available"] = sampler.available
            return result
        sampler.start()
        deadline = time.perf_counter() + 0.25
        sample_work = 0
        while time.perf_counter() < deadline:
            sample_work += 1
        energy_wh = sampler.stop()
        result.update(
            {
                "available": energy_wh > 0.0,
                "counter_positive": energy_wh > 0.0,
                "sample_energy_wh": energy_wh,
                "sample_energy_j": energy_wh * 3600.0,
                "sample_work": sample_work,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def memory_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    if sys.platform == "darwin":
        try:
            return int(str(command("sysctl", "-n", "hw.memsize")).strip())
        except Exception:
            return None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    return None


def calibration_probe(
    path: Path | None,
    *,
    verified_evidence_files: set[Path] | None = None,
) -> dict[str, Any]:
    if path is None:
        return {"provided": False, "passed": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "provided": True,
            "passed": False,
            "path": public_path(path),
            "error": f"{type(exc).__name__}: {exc}",
        }
    try:
        validated = validate_calibration_manifest(payload, manifest_path=path)
    except CalibrationValidationError as exc:
        raw = payload if isinstance(payload, dict) else {}
        return {
            "provided": True,
            "passed": False,
            "path": public_path(path),
            "sha256": sha256_file(path),
            "schema_version": raw.get("schema_version"),
            "calibration_id": raw.get("calibration_id"),
            "host_id": raw.get("host_id"),
            "execution_layer": raw.get("execution_layer"),
            "validation_error": str(exc),
        }
    if verified_evidence_files is not None:
        verified_evidence_files.update(validated.evidence_files)
    return {
        "provided": True,
        "passed": True,
        "path": public_path(path),
        "sha256": sha256_file(path),
        "schema_version": "aet-calibration/v1",
        "calibration_id": validated.calibration_id,
        "host_id": validated.host_id,
        "execution_layer": validated.execution_layer,
        "wall_system_primary": True,
        "hardware_complete": True,
        "gpu_index": validated.gpu_index,
        "gpu_device_id_sha256": validated.gpu_device_id_sha256,
        "wall_metadata_complete": True,
        "evidence_complete": True,
        "summary_complete": True,
        "numeric_criteria_pass": True,
        "workload_records_complete": True,
    }


def execution_layer() -> str:
    if sys.platform == "win32":
        return "windows-native"
    if sys.platform.startswith("linux"):
        try:
            version = Path("/proc/version").read_text().lower()
        except OSError:
            version = ""
        return "wsl2" if "microsoft" in version else "native-linux"
    if sys.platform == "darwin":
        return "macos"
    return sys.platform


def windows_build() -> int | None:
    if sys.platform != "win32":
        return None
    try:
        return int(sys.getwindowsversion().build)
    except (AttributeError, TypeError, ValueError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-id", required=True, help="Non-identifying study host label")
    parser.add_argument("--cpu-label", default=None)
    parser.add_argument("--accelerator-label", default=None)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--calibration-manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--active-probe", action="store_true")
    requirement = parser.add_mutually_exclusive_group()
    requirement.add_argument("--require-smoke", action="store_true")
    requirement.add_argument("--require-software-smoke", action="store_true")
    requirement.add_argument("--require-primary", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(str(command("git", "rev-parse", "--show-toplevel")).strip()).resolve()
    output = args.output.resolve() if args.output is not None else None
    calibration_evidence_files: set[Path] = set()
    calibration = calibration_probe(
        args.calibration_manifest,
        verified_evidence_files=calibration_evidence_files,
    )
    excluded_git_artifacts = {
        path.resolve() for path in (output, args.calibration_manifest) if path is not None
    }
    excluded_git_artifacts.update(calibration_evidence_files)
    git = git_snapshot(root, excluded_git_artifacts)
    nvml, torch_cuda = gpu_qualification_probe(args.active_probe, args.gpu_index)
    rapl = rapl_probe(args.active_probe)
    windows_emi = windows_emi_probe(args.active_probe)
    strict_tracker = strict_tracker_probe()
    python_supported = SUPPORTED_PYTHON_MIN <= sys.version_info[:2] < SUPPORTED_PYTHON_MAX_EXCLUSIVE
    imports = {
        "neuro_co.aet": import_probe("neuro_co.aet"),
        "neuro_co.aet.experiments.software_smoke_runner": import_probe(
            "neuro_co.aet.experiments.software_smoke_runner"
        ),
        "neuro_co.problems.cvrp.pyvrp": import_probe("neuro_co.problems.cvrp.pyvrp"),
    }
    current_execution_layer = execution_layer()
    current_windows_build = windows_build()
    windows_11_native = (
        current_execution_layer == "windows-native"
        and current_windows_build is not None
        and current_windows_build >= 22000
    )
    git_clean = bool(git.get("available")) and not bool(git.get("dirty"))
    runtime_ready = (
        python_supported
        and all(value["importable"] for value in imports.values())
        and bool(git.get("available"))
    )
    environment_ready = runtime_ready and git_clean
    source_snapshot_ready = bool(
        git.get("available") and git.get("sha") and git.get("worktree_fingerprint_sha256")
    )
    selected_device = nvml.get("selected_device") or {}
    gpu_component_ready = (
        args.active_probe
        and bool(nvml.get("available"))
        and bool(nvml.get("selection_valid"))
        and bool(selected_device.get("measurement_mode_supported"))
    )
    direct_gpu_counter_ready = (
        gpu_component_ready
        and selected_device.get("total_energy_counter_supported") is True
        and selected_device.get("counter_monotonic") is True
        and selected_device.get("counter_positive") is True
        and selected_device.get("counter_bracketed_cuda_work") is True
    )
    torch_cuda_ready = (
        args.active_probe
        and torch_cuda.get("available") is True
        and torch_cuda.get("selection_valid") is True
        and torch_cuda.get("requested_nvml_index") == args.gpu_index
        and torch_cuda.get("selected_cuda_index") == args.gpu_index
        and torch_cuda.get("visibility_remapped") is False
        and torch_cuda.get("identity_verified") is True
        and torch_cuda.get("compute_probe_passed") is True
        and torch_cuda.get("nvml_counter_bracketed") is True
        and torch_cuda.get("nvml_counter_positive") is True
    )
    cpu_component_ready = args.active_probe and bool(rapl.get("available"))
    windows_emi_ready = (
        args.active_probe
        and windows_11_native
        and windows_emi.get("available") is True
        and windows_emi.get("backend_mode") == "windows_emi"
        and windows_emi.get("counter_positive") is True
        and windows_emi.get("ram_included") is False
    )
    calibration_compatible = (
        bool(calibration.get("passed"))
        and calibration.get("host_id") == args.host_id
        and calibration.get("execution_layer") == current_execution_layer
        and calibration.get("gpu_index") == args.gpu_index
        and calibration.get("gpu_device_id_sha256") == selected_device.get("device_id_sha256")
    )
    system_energy_ready = cpu_component_ready or calibration_compatible
    smoke_ready = (
        environment_ready
        and gpu_component_ready
        and system_energy_ready
        and bool(strict_tracker.get("supported"))
    )
    primary_ready = smoke_ready and calibration_compatible
    exploratory_software_ready = (
        runtime_ready
        and source_snapshot_ready
        and windows_11_native
        and direct_gpu_counter_ready
        and torch_cuda_ready
        and windows_emi_ready
        and bool(strict_tracker.get("component_contract_supported"))
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "host_id": args.host_id,
        "privacy": {
            "hostname_recorded": False,
            "serial_number_recorded": False,
            "hardware_uuid_recorded": False,
            "environment_variables_recorded": False,
        },
        "system": {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "windows_build": current_windows_build,
            "machine": platform.machine(),
            "execution_layer": current_execution_layer,
            "cpu_label": args.cpu_label or platform.processor() or platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "memory_total_b": memory_bytes(),
            "accelerator_label": args.accelerator_label,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_supported": python_supported,
            "uv": distribution_version("uv"),
            "torch": distribution_version("torch"),
            "codecarbon": distribution_version("codecarbon"),
            "pynvml": distribution_version("nvidia-ml-py"),
            "pyrapl": distribution_version("pyRAPL"),
            "imports": imports,
        },
        "git": git,
        "strict_tracker": strict_tracker,
        "backends": {
            "nvml": nvml,
            "torch_cuda": torch_cuda,
            "rapl": rapl,
            "windows_emi": windows_emi,
            "codecarbon_allowed_primary": False,
            "software_primary": "hwcounters",
            "tdp_allowed_primary": False,
        },
        "calibration": calibration,
        "readiness": {
            "environment_ready": environment_ready,
            "runtime_ready": runtime_ready,
            "git_clean": git_clean,
            "source_snapshot_ready": source_snapshot_ready,
            "gpu_component_ready": gpu_component_ready,
            "direct_gpu_counter_ready": direct_gpu_counter_ready,
            "torch_cuda_ready": torch_cuda_ready,
            "cpu_component_ready": cpu_component_ready,
            "windows_11_native": windows_11_native,
            "windows_emi_ready": windows_emi_ready,
            "calibration_compatible": calibration_compatible,
            "system_energy_ready": system_energy_ready,
            "measured_smoke_ready": smoke_ready,
            "exploratory_software_ready": exploratory_software_ready,
            "confirmatory_primary_ready": primary_ready,
        },
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    if args.require_primary and not primary_ready:
        return 2
    if args.require_software_smoke and not exploratory_software_ready:
        return 2
    if args.require_smoke and not smoke_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
