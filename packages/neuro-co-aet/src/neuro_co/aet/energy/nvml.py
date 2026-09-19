"""NVML GPU energy sampler with explicit physical-device selection.

Each selected GPU uses the cumulative NVML energy counter when available.
Devices without that counter fall back individually to sampled power
integration. Public metadata contains only SHA-256 device identifiers, never
raw NVML UUIDs.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections.abc import Collection
from contextlib import suppress
from typing import Any, Final

NVML_TOTAL_ENERGY_COUNTER: Final = "nvml_total_energy_counter"
NVML_POWER_INTEGRATION: Final = "nvml_power_integration"


class NvmlDeviceSelectionError(RuntimeError):
    """Raised when an explicitly requested physical GPU cannot be selected."""


class NvmlSampler:
    """Measure energy for selected GPUs via pynvml.

    ``gpu_indices`` and ``gpu_uuids`` are mutually exclusive. When neither is
    supplied, all visible NVML devices are measured for backwards
    compatibility. UUIDs are retained only in private in-memory state during
    setup; :attr:`device_metadata` exposes their SHA-256 digests.
    """

    def __init__(
        self,
        gpu_indices: Collection[int] | None = None,
        gpu_uuids: Collection[str] | None = None,
        poll_hz: float = 10.0,
    ) -> None:
        if gpu_indices is not None and gpu_uuids is not None:
            raise ValueError("gpu_indices and gpu_uuids are mutually exclusive")
        selected_indices = _validate_indices(gpu_indices)
        selected_uuids = _validate_uuids(gpu_uuids)
        if isinstance(poll_hz, bool):
            raise TypeError("poll_hz must be a real number")
        try:
            poll_hz_value = float(poll_hz)
        except (TypeError, ValueError) as exc:
            raise TypeError("poll_hz must be a real number") from exc
        if not math.isfinite(poll_hz_value) or poll_hz_value <= 0.0:
            raise ValueError("poll_hz must be finite and greater than zero")

        try:
            import pynvml  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(f"pynvml not installed: {exc}") from exc

        self.pynvml = pynvml
        self.poll_dt = 1.0 / poll_hz_value
        self._initialized = False
        self._started = False
        self._stopped = False
        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()
        self._poll_lock = threading.Lock()
        self._poll_energy_wh: dict[int, float] = {}
        self._poll_sample_count: dict[int, int] = {}
        self._poll_dropped_sample_count: dict[int, int] = {}
        self._poll_error: BaseException | None = None
        self._counter_start_mj: dict[int, int] = {}
        self._counter_sample_count: dict[int, int] = {}

        try:
            pynvml.nvmlInit()
            self._initialized = True
            inventory = self._inventory()
            if not inventory:
                raise RuntimeError("no NVML devices detected")
            selected = self._select_devices(
                inventory,
                gpu_indices=selected_indices,
                gpu_uuids=selected_uuids,
            )
            self.gpu_indices = [entry[0] for entry in selected]
            self.handles = [entry[1] for entry in selected]
            self._device_id_hashes = [_hash_device_id(entry[2]) for entry in selected]
            self._measurement_modes = [self._measurement_mode(handle) for handle in self.handles]
        except Exception:
            self._shutdown()
            raise

    @property
    def measurement_modes(self) -> tuple[str, ...]:
        """Return the selected devices' measurement modes in selection order."""

        return tuple(self._measurement_modes)

    @property
    def device_metadata(self) -> list[dict[str, Any]]:
        """Return artifact-safe metadata for every selected physical GPU."""

        metadata: list[dict[str, Any]] = []
        for position, (index, device_id_hash, mode) in enumerate(
            zip(
                self.gpu_indices,
                self._device_id_hashes,
                self._measurement_modes,
                strict=True,
            )
        ):
            integrated = mode == NVML_POWER_INTEGRATION
            metadata.append(
                {
                    "index": index,
                    "device_id_sha256": device_id_hash,
                    "measurement_mode": mode,
                    "attribution_scope": "physical_device_all_processes",
                    "exclusive_device_required": True,
                    "sampling_interval_s": self.poll_dt if integrated else None,
                    "sample_count": (
                        self._poll_sample_count.get(position, 0)
                        if integrated
                        else self._counter_sample_count.get(position, 0)
                    ),
                    "dropped_sample_count": (
                        self._poll_dropped_sample_count.get(position, 0) if integrated else 0
                    ),
                }
            )
        return metadata

    def start(self) -> None:
        """Start all selected counters and power samplers."""

        if self._started:
            raise RuntimeError("NVML sampler has already been started")
        if self._stopped:
            raise RuntimeError("NVML sampler has already been stopped")
        self._started = True
        try:
            self._counter_start_mj = {
                position: int(self.pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
                for position, (handle, mode) in enumerate(
                    zip(self.handles, self._measurement_modes, strict=True)
                )
                if mode == NVML_TOTAL_ENERGY_COUNTER
            }
            self._counter_sample_count = {position: 1 for position in self._counter_start_mj}
            integrated_positions = [
                position
                for position, mode in enumerate(self._measurement_modes)
                if mode == NVML_POWER_INTEGRATION
            ]
            if integrated_positions:
                self._poll_stop.clear()
                self._poll_error = None
                self._poll_energy_wh = {position: 0.0 for position in integrated_positions}
                self._poll_sample_count = {position: 0 for position in integrated_positions}
                self._poll_dropped_sample_count = {position: 0 for position in integrated_positions}
                self._poll_thread = threading.Thread(
                    target=self._poll_loop,
                    args=(integrated_positions,),
                    daemon=True,
                )
                self._poll_thread.start()
        except Exception:
            self._shutdown()
            raise

    def stop(self) -> float:
        """Stop sampling and return aggregate GPU energy in Wh."""

        if not self._started:
            raise RuntimeError("NVML sampler has not been started")
        if self._stopped:
            raise RuntimeError("NVML sampler has already been stopped")
        self._stopped = True
        try:
            self._poll_stop.set()
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=max(2.0, self.poll_dt * 4.0))
                if self._poll_thread.is_alive():
                    raise RuntimeError("NVML power-integration thread did not stop")
            if self._poll_error is not None:
                raise RuntimeError("NVML power sampling failed") from self._poll_error

            counter_energy_wh = 0.0
            for position, start_mj in self._counter_start_mj.items():
                stop_mj = int(
                    self.pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handles[position])
                )
                self._counter_sample_count[position] += 1
                if stop_mj < start_mj:
                    raise RuntimeError("NVML total-energy counter decreased during measurement")
                counter_energy_wh += (stop_mj - start_mj) / 3_600_000.0
            with self._poll_lock:
                integrated_energy_wh = sum(self._poll_energy_wh.values())
            return counter_energy_wh + integrated_energy_wh
        finally:
            self._shutdown()

    def close(self) -> None:
        """Release NVML when a prepared sampler is abandoned before a run."""

        if self._started and not self._stopped:
            self.stop()
            return
        self._shutdown()

    def _inventory(self) -> list[tuple[int, Any, str]]:
        device_count = int(self.pynvml.nvmlDeviceGetCount())
        inventory: list[tuple[int, Any, str]] = []
        for index in range(device_count):
            handle = self.pynvml.nvmlDeviceGetHandleByIndex(index)
            raw_uuid = _normalize_uuid(self.pynvml.nvmlDeviceGetUUID(handle))
            inventory.append((index, handle, raw_uuid))
        return inventory

    def _select_devices(
        self,
        inventory: list[tuple[int, Any, str]],
        *,
        gpu_indices: tuple[int, ...] | None,
        gpu_uuids: tuple[str, ...] | None,
    ) -> list[tuple[int, Any, str]]:
        if gpu_indices is not None:
            by_index = {entry[0]: entry for entry in inventory}
            missing = [index for index in gpu_indices if index not in by_index]
            if missing:
                rendered = ", ".join(str(index) for index in missing)
                raise NvmlDeviceSelectionError(f"NVML GPU index not found: {rendered}")
            return [by_index[index] for index in gpu_indices]
        if gpu_uuids is not None:
            by_uuid = {entry[2].casefold(): entry for entry in inventory}
            missing = [uuid for uuid in gpu_uuids if uuid.casefold() not in by_uuid]
            if missing:
                hashes = ", ".join(_hash_device_id(uuid)[:12] for uuid in missing)
                raise NvmlDeviceSelectionError(f"NVML GPU UUID hash prefix not found: {hashes}")
            return [by_uuid[uuid.casefold()] for uuid in gpu_uuids]
        return inventory

    def _measurement_mode(self, handle: Any) -> str:
        try:
            self.pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        except Exception:
            return NVML_POWER_INTEGRATION
        return NVML_TOTAL_ENERGY_COUNTER

    def _poll_loop(self, positions: list[int]) -> None:
        try:
            powers_w = {position: self._read_power_w(position) for position in positions}
            with self._poll_lock:
                for position in positions:
                    self._poll_sample_count[position] += 1
            last = time.perf_counter()
            while True:
                self._poll_stop.wait(self.poll_dt)
                now = time.perf_counter()
                dt = max(0.0, now - last)
                with self._poll_lock:
                    for position, power_w in powers_w.items():
                        self._poll_energy_wh[position] += power_w * dt / 3600.0
                        missed = max(0, int(dt / self.poll_dt) - 1)
                        self._poll_dropped_sample_count[position] += missed
                if self._poll_stop.is_set():
                    return
                powers_w = {position: self._read_power_w(position) for position in positions}
                with self._poll_lock:
                    for position in positions:
                        self._poll_sample_count[position] += 1
                last = now
        except BaseException as exc:
            self._poll_error = exc
            self._poll_stop.set()

    def _read_power_w(self, position: int) -> float:
        power_mw = float(self.pynvml.nvmlDeviceGetPowerUsage(self.handles[position]))
        if not math.isfinite(power_mw) or power_mw < 0.0:
            raise RuntimeError("NVML returned invalid power usage")
        return power_mw / 1000.0

    def _shutdown(self) -> None:
        if not self._initialized:
            return
        with suppress(Exception):
            self.pynvml.nvmlShutdown()
        self._initialized = False


def _validate_indices(gpu_indices: Collection[int] | None) -> tuple[int, ...] | None:
    if gpu_indices is None:
        return None
    if isinstance(gpu_indices, (str, bytes)):
        raise TypeError("gpu_indices must be a collection of integers")
    try:
        indices = tuple(gpu_indices)
    except TypeError as exc:
        raise TypeError("gpu_indices must be a collection of integers") from exc
    if not indices:
        raise ValueError("gpu_indices must not be empty")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError("gpu_indices must contain only integers")
    if any(index < 0 for index in indices):
        raise ValueError("gpu_indices must contain only non-negative integers")
    if len(set(indices)) != len(indices):
        raise ValueError("gpu_indices must not contain duplicates")
    return indices


def _validate_uuids(gpu_uuids: Collection[str] | None) -> tuple[str, ...] | None:
    if gpu_uuids is None:
        return None
    if isinstance(gpu_uuids, (str, bytes)):
        raise TypeError("gpu_uuids must be a collection of strings")
    try:
        raw_uuids = tuple(gpu_uuids)
    except TypeError as exc:
        raise TypeError("gpu_uuids must be a collection of strings") from exc
    if not raw_uuids:
        raise ValueError("gpu_uuids must not be empty")
    if any(not isinstance(uuid, str) for uuid in raw_uuids):
        raise TypeError("gpu_uuids must contain only strings")
    uuids = tuple(uuid.strip() for uuid in raw_uuids)
    if any(not uuid for uuid in uuids):
        raise ValueError("gpu_uuids must contain only non-empty strings")
    if len({uuid.casefold() for uuid in uuids}) != len(uuids):
        raise ValueError("gpu_uuids must not contain duplicates")
    return uuids


def _normalize_uuid(raw_uuid: object) -> str:
    if isinstance(raw_uuid, bytes):
        value = raw_uuid.decode("utf-8", errors="strict")
    else:
        value = str(raw_uuid)
    value = value.strip()
    if not value:
        raise RuntimeError("NVML returned an empty GPU UUID")
    return value


def _hash_device_id(raw_uuid: str) -> str:
    return hashlib.sha256(raw_uuid.encode("utf-8")).hexdigest()
