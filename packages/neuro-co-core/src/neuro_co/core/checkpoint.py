"""Atomic Torch checkpoints and explicit training/RNG state, without a trainer dependency."""

from __future__ import annotations

import os
import random
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch


def atomic_save(payload: Any, path: Path | str) -> None:
    """Replace a checkpoint only after serialization and flushing succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as out:
            temporary = Path(out.name)
            torch.save(payload, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _device(device: torch.device | str) -> str:
    device = torch.device(device)
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        return f"cuda:{index}"
    if device.type not in {"cpu", "mps"}:
        raise ValueError(f"unsupported RNG device: {device}")
    return device.type


def capture_rng_state(device: torch.device | str) -> dict[str, Any]:
    """Capture host RNGs and only the selected accelerator's RNG on CPU."""
    selected = _device(device)
    numpy_state = cast(tuple, np.random.get_state())
    accelerator = None
    if selected.startswith("cuda:"):
        accelerator = torch.cuda.get_rng_state(torch.device(selected)).cpu()
    elif selected == "mps":
        accelerator = torch.mps.get_rng_state().cpu()
    return {
        "device": selected,
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
        "torch_cpu": torch.get_rng_state().cpu(),
        "accelerator": accelerator,
    }


def _numpy_state(state: dict) -> tuple:
    name, keys, position, gaussian, cached = state["numpy"]
    return name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached


def validate_rng_state(state: dict, device: torch.device | str) -> None:
    """Validate without changing any global random stream."""
    selected = _device(device)
    if state.get("device") != selected:
        raise ValueError("checkpoint RNG device differs from the selected training device")
    random.Random().setstate(state["python"])
    np.random.RandomState().set_state(_numpy_state(state))
    torch.Generator().set_state(state["torch_cpu"].cpu())
    accelerator = state.get("accelerator")
    if selected == "cpu":
        if accelerator is not None:
            raise ValueError("CPU checkpoint contains accelerator RNG state")
    elif (
        not isinstance(accelerator, torch.Tensor)
        or accelerator.dtype != torch.uint8
        or accelerator.ndim != 1
        or accelerator.numel() == 0
    ):
        raise ValueError("checkpoint accelerator RNG state is invalid")


def restore_rng_state(state: dict, device: torch.device | str) -> None:
    """Restore only the validated host and selected-device random streams."""
    validate_rng_state(state, device)
    random.setstate(state["python"])
    np.random.set_state(_numpy_state(state))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["device"].startswith("cuda:"):
        torch.cuda.set_rng_state(state["accelerator"].cpu(), torch.device(state["device"]))
    elif state["device"] == "mps":
        torch.mps.set_rng_state(state["accelerator"].cpu())


def _type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _config(algo: object) -> dict | None:
    config = getattr(algo, "cfg", None)
    return asdict(config) if is_dataclass(config) and not isinstance(config, type) else None


def _components(algo: object) -> dict:
    return {
        name: component
        for name in ("opt", "sched", "precision", "ema")
        if (component := getattr(algo, name, None)) is not None
    }


def training_state(algo: object) -> dict[str, Any]:
    """Save conventional algorithm components separately from model weights."""
    return {
        "algorithm": _type_name(algo),
        "config": _config(algo),
        "step": getattr(algo, "_step", None),
        "components": {
            name: {"type": _type_name(component), "state": component.state_dict()}
            for name, component in _components(algo).items()
        },
    }


def validate_training_state(algo: object, state: dict) -> None:
    """Check component/config compatibility before callers mutate model weights."""
    if state.get("algorithm") != _type_name(algo) or state.get("config") != _config(algo):
        raise ValueError("checkpoint algorithm or training configuration differs")
    components = _components(algo)
    saved = state.get("components", {})
    if set(saved) != set(components):
        raise ValueError("checkpoint training components differ")
    if state.get("step") is not None and (type(state["step"]) is not int or state["step"] < 0):
        raise ValueError("checkpoint algorithm step is invalid")
    for name, component in components.items():
        record = saved[name]
        if record.get("type") != _type_name(component) or not isinstance(record.get("state"), dict):
            raise ValueError(f"checkpoint {name} component differs")
        values = record["state"]
        validator = getattr(component, "validate_state_dict", None)
        if validator is not None:
            validator(values)
        if name == "opt":
            groups = values.get("param_groups", [])
            if len(groups) != len(component.param_groups) or any(
                len(a["params"]) != len(b["params"])
                for a, b in zip(groups, component.param_groups, strict=True)
            ):
                raise ValueError("checkpoint optimizer parameter groups differ")


def restore_training_state(algo: object, state: dict) -> None:
    """Restore optimizer, scheduler, precision, EMA and algorithm step."""
    validate_training_state(algo, state)
    for name, component in _components(algo).items():
        component.load_state_dict(state["components"][name]["state"])
    if state["step"] is not None:
        cast(Any, algo)._step = state["step"]
