"""Evaluation support with a fixed seeded dataset and optional EMA weights.

Algorithms inherit this mixin to reuse held-out instances across evaluations.
The eval_weights() context substitutes EMA shadow weights during evaluation.

The algorithm supplies env, device, cfg.eval_batch_size, cfg.eval_seed, and
optionally ema."""

import contextlib

import torch

from ..state import State


class EvalSupport:
    """Mixin providing `_get_eval_state()` and `eval_weights()`."""

    _eval_state: State | None = None

    def _get_eval_state(self) -> State:
        state = self._eval_state
        if state is None:
            # CPU generator + env-side device move keeps this portable across
            # cpu / cuda / mps (mps lacks a device-bound Generator).
            g = torch.Generator().manual_seed(self.cfg.eval_seed)  # type: ignore[attr-defined]
            state = self.env.reset(  # type: ignore[attr-defined]
                self.cfg.eval_batch_size,  # type: ignore[attr-defined]
                generator=g,
                device=self.device,  # type: ignore[attr-defined]
            )
            self._eval_state = state
        return state

    @contextlib.contextmanager
    def eval_weights(self):
        """Swap EMA shadow weights in for the duration; restore after.

        No-op when the algo has no EMA. Trainer uses this when saving best.pt
        so the exported weights match what eval scored."""
        ema = getattr(self, "ema", None)
        if ema is not None:
            ema.apply()
        try:
            yield
        finally:
            if ema is not None:
                ema.restore()


__all__ = ["EvalSupport"]
