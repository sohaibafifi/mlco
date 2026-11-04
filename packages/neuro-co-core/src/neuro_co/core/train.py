"""Minimal trainer. Replaces Lightning.

Owns the outer loop: periodic eval, checkpointing (latest + best),
resume, optional logging hooks. Algos own optimizer + rollout + AMP +
DDP grad sync + EMA + LR schedule.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .algo import Algo
from .checkpoint import (
    atomic_save,
    capture_rng_state,
    restore_rng_state,
    restore_training_state,
    training_state,
    validate_rng_state,
    validate_training_state,
)


@dataclass(slots=True)
class TrainState:
    step: int = 0
    epoch: int = 0
    best_metric: float = float("-inf")


@dataclass(slots=True)
class TrainHooks:
    """Optional callbacks. Each takes `(trainer, metrics)`."""

    on_step_end: Callable[["Trainer", dict[str, float]], None] | None = None
    on_eval_end: Callable[["Trainer", dict[str, float]], None] | None = None


class Trainer:
    """Minimal trainer loop. Algo-agnostic.

    Args:
        algo: object satisfying `Algo` protocol.
        steps: total optimizer steps.
        eval_every: run `algo.eval_step` every N steps. 0 disables.
        ckpt_every: write a `step_*.pt` checkpoint every N steps. 0 disables.
        ckpt_dir: directory for checkpoints (latest, best, periodic).
        device: target device.
        seed: torch generator seed for sampling reproducibility.
        hooks: optional callbacks for logging integration.
        best_metric_key: eval metric to track for best.pt (higher = better).
        resume_from: optional checkpoint path to restore before training.
    """

    def __init__(
        self,
        algo: Algo,
        *,
        steps: int,
        eval_every: int = 0,
        ckpt_every: int = 0,
        ckpt_dir: Path | str | None = None,
        device: torch.device | str = "cpu",
        seed: int = 0,
        hooks: TrainHooks | None = None,
        best_metric_key: str = "eval_reward",
        resume_from: Path | str | None = None,
    ) -> None:
        self.algo = algo
        self.steps = steps
        self.eval_every = eval_every
        self.ckpt_every = ckpt_every
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.device = torch.device(device)
        self.hooks = hooks or TrainHooks()
        self.best_metric_key = best_metric_key
        self.state = TrainState()
        generator_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.rng = torch.Generator(device=generator_device).manual_seed(seed)
        if resume_from is not None:
            self.load(resume_from)

    def fit(self) -> TrainState:
        remaining = max(0, self.steps - self.state.step)
        for _ in range(remaining):
            metrics = self.algo.train_step(self.rng)
            self.state.step += 1
            if self.hooks.on_step_end is not None:
                self.hooks.on_step_end(self, metrics)
            if self.eval_every and self.state.step % self.eval_every == 0:
                self._run_eval()
            if self.ckpt_every and self.state.step % self.ckpt_every == 0:
                self._save_ckpt(
                    self.ckpt_dir / f"step_{self.state.step:08d}.pt" if self.ckpt_dir else None
                )
        # Always persist the final state as latest.pt.
        if self.ckpt_dir is not None and remaining > 0:
            self._save_ckpt(self.ckpt_dir / "latest.pt")
        return self.state

    def _run_eval(self) -> dict[str, float]:
        metrics = self.algo.eval_step(self.rng)
        if self.hooks.on_eval_end is not None:
            self.hooks.on_eval_end(self, metrics)
        # Track best (higher = better). Save best.pt on improvement, under
        # eval (EMA) weights so the exported model matches what scored best.
        val = metrics.get(self.best_metric_key)
        if val is not None and val > self.state.best_metric:
            self.state.best_metric = val
            if self.ckpt_dir is not None:
                self._save_ckpt(self.ckpt_dir / "best.pt", eval_weights=True, mirror_latest=False)
        return metrics

    def _save_ckpt(
        self, path: Path | None, *, eval_weights: bool = False, mirror_latest: bool = True
    ) -> None:
        """Write a checkpoint.

        eval_weights: save under the algo's EMA weights (for best.pt /
            deployment). latest/periodic use raw weights so resume restores
            the true optimizer-aligned parameters.
        mirror_latest: also refresh latest.pt (skip for best.pt).
        """
        if path is None or self.ckpt_dir is None:
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ew = getattr(self.algo, "eval_weights", None)
        ctx: AbstractContextManager = nullcontext()
        if eval_weights and callable(ew):
            ctx = cast(AbstractContextManager, ew())
        with ctx:
            payload = {
                "checkpoint_version": 1,
                "resume_training": not (
                    eval_weights and callable(ew) and getattr(self.algo, "ema", None) is not None
                ),
                "training_state": training_state(self.algo),
                "global_rng_state": capture_rng_state(self.device),
                "step": self.state.step,
                "epoch": self.state.epoch,
                "best_metric": self.state.best_metric,
                "algo_state": _maybe_state_dict(self.algo),
                "rng_state": self.rng.get_state().cpu(),
            }
            atomic_save(payload, path)
            if mirror_latest and path != self.ckpt_dir / "latest.pt":
                atomic_save(payload, self.ckpt_dir / "latest.pt")

    def load(self, path: Path | str) -> None:
        """Restore a training checkpoint; legacy checkpoints keep their saved subset."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        version = payload.get("checkpoint_version")
        if version not in (None, 1):
            raise ValueError("unsupported trainer checkpoint version")
        if payload.get("resume_training") is False:
            raise ValueError("EMA evaluation checkpoint cannot resume training; use latest.pt")
        algo_state = payload.get("algo_state")
        if version == 1:
            validate_training_state(self.algo, payload["training_state"])
            validate_rng_state(payload["global_rng_state"], self.device)
            _validate_algo_state(self.algo, algo_state)
        rng_state = payload.get("rng_state")
        if rng_state is not None:
            torch.Generator(device=self.rng.device).set_state(rng_state.cpu())
        state = TrainState(
            step=int(payload.get("step", 0)),
            epoch=int(payload.get("epoch", 0)),
            best_metric=float(payload.get("best_metric", float("-inf"))),
        )
        if state.step < 0 or state.epoch < 0:
            raise ValueError("checkpoint progress must be nonnegative")
        load_fn = getattr(self.algo, "load_state_dict", None)
        if algo_state is not None and callable(load_fn):
            load_fn(algo_state)
        if version == 1:
            restore_training_state(self.algo, payload["training_state"])
            restore_rng_state(payload["global_rng_state"], self.device)
        elif hasattr(self.algo, "_step"):
            cast(Any, self.algo)._step = state.step
        if rng_state is not None:
            self.rng.set_state(rng_state.cpu())
        self.state = state


def _validate_algo_state(algo: object, saved: dict | None) -> None:
    current = _maybe_state_dict(algo)
    if current is None and saved is None:
        return
    if not isinstance(saved, dict) or current is None or set(current) != set(saved):
        raise ValueError("checkpoint model state keys differ")
    for key, value in current.items():
        if isinstance(value, torch.Tensor) and (
            not isinstance(saved[key], torch.Tensor)
            or value.shape != saved[key].shape
            or value.dtype != saved[key].dtype
        ):
            raise ValueError(f"checkpoint model shape or dtype differs: {key}")


def _maybe_state_dict(obj: object) -> dict | None:
    fn = getattr(obj, "state_dict", None)
    if not callable(fn):
        return None
    out = fn()
    return out if isinstance(out, dict) else None
