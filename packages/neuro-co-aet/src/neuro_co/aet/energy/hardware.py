"""Best-effort hardware-SKU detection for AET embodied-carbon lookup.

Returns a string compatible with the embodied-carbon table in
`neuro_co.aet.energy.embodied`. Used by `EnergyTracker(hardware_id=…)`
callers (training drivers, baseline runners) when no explicit
identifier is set.

`torch` is imported lazily so this stays callable on a CPU-only
install. `autodetect_hardware_id()` returns `"generic-cpu"` if torch
isn't available.
"""

from __future__ import annotations

# Substrings matched against `torch.cuda.get_device_name(0).lower()`.
_GPU_KEYS: tuple[str, ...] = (
    "a100",
    "h100",
    "h200",
    "b100",
    "b200",
    "v100",
    "rtx-4090",
    "rtx-3090",
)

_APPLE_KEYS: tuple[str, ...] = ("m5", "m4", "m3", "m2", "m1")


def autodetect_hardware_id() -> str:
    """Identify the active accelerator (CUDA / Apple) or fall back to CPU.

    Order:
    1. CUDA device name → `"nvidia-<sku>"` (or `"generic-gpu"`).
    2. `platform.processor()` → `"apple-mN"` / `"intel-xeon-…"` /
       `"amd-epyc-…"`.
    3. `"generic-cpu"`.
    """
    try:
        import torch
    except ImportError:
        return "generic-cpu"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0).lower()
        for key in _GPU_KEYS:
            if key in name:
                return f"nvidia-{key}"
        return "generic-gpu"
    try:
        import platform

        proc = (platform.processor() or "").lower()
        if "apple" in proc:
            for k in _APPLE_KEYS:
                if k in proc:
                    return f"apple-{k}"
        if "xeon" in proc:
            return "intel-xeon-8358"
        if "epyc" in proc:
            return "amd-epyc-7763"
    except Exception:
        pass
    return "generic-cpu"
