"""Pure-functional Env Protocol.

Envs are stateless modules holding constants only. Per-batch state
travels via `State` subclasses (pytree-registered, frozen dataclass).

Algos consume the Protocol; concrete envs (TSP, CVRP, CVRPTW, ...)
implement it. Algos stay env-agnostic: no per-env variants.
"""

from typing import Generic, Protocol, TypeVar, runtime_checkable

import torch
from jaxtyping import Bool, Float, Int

from .state import State

S = TypeVar("S", bound=State)


@runtime_checkable
class Env(Protocol, Generic[S]):
    """Pure-functional CO environment.

    Constants live on the env (e.g. problem size); per-batch state lives
    in `S`. All methods are pure.
    """

    encoder_in_dim: int
    """Dimensionality of the per-node feature tensor returned by
    `build_features`. Caller uses this when constructing an `Encoder`."""

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> S:
        """Sample a fresh batch of problems."""
        ...

    def step(
        self,
        state: S,
        action: Int[torch.Tensor, "b"],
    ) -> tuple[S, Float[torch.Tensor, "b"], Bool[torch.Tensor, "b"]]:
        """Apply action. Return (next_state, per-sample reward, done flag)."""
        ...

    def action_mask(self, state: S) -> Bool[torch.Tensor, "b a"]:
        """True = action permitted. Masked entries get -inf bias in logits."""
        ...

    # Features and decoder context shared by training algorithms.

    def build_features(self, state: S) -> Float[torch.Tensor, "b n d_in"]:
        """Per-node feature tensor consumed by the encoder.

        Examples:
            TSP   -> coords (b, n, 2)
            CVRP  -> cat(coords, demand_norm) (b, n+1, 3)
            CVRPTW-> cat(coords, demand_norm, tw_early, tw_late, t_now) (b, n+1, 6)
        """
        ...

    def decoder_context(self, state: S) -> tuple[Int[torch.Tensor, "b"], Int[torch.Tensor, "b"]]:
        """`(first_idx, current_idx)` indices the pointer decoder uses to
        build context. For envs without a "first" semantics (CVRP, CVRPTW)
        return depot index (0)."""
        ...

    def max_steps(self, state: S) -> int:
        """Upper bound on rollout length. Algos cap loops at this value;
        envs that finish earlier signal via the `done` flag."""
        ...

    def pomo_first_mask(self, state: S) -> Bool[torch.Tensor, "b a"]:
        """Mask of actions valid as the *first* POMO action. By default
        excludes the current node (e.g. depot for CVRP, start city for TSP).
        Used by POMO to pick `n_starts` distinct first actions per problem."""
        ...


def get_dynamic_decoder_context(env: Env[S], state: S) -> Float[torch.Tensor, "b c"] | None:
    """Return an optional, state-dependent decoder context.

    The original environment protocol exposes only node indices. Some
    constructive problems also need numeric state at every decode step. CVRP,
    for example, needs the remaining vehicle capacity. Keeping this extension
    optional preserves structural compatibility for environments that only
    implement ``decoder_context``.
    """

    context_fn = getattr(env, "dynamic_decoder_context", None)
    if context_fn is None:
        return None
    context = context_fn(state)
    if not isinstance(context, torch.Tensor):
        raise TypeError("dynamic_decoder_context must return a torch.Tensor")
    return context


__all__ = ["Env", "get_dynamic_decoder_context"]
