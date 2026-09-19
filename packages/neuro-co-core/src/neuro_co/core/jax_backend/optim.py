"""Small pure-JAX Adam and AdamW optimizers.

The update order matches the ordinary PyTorch optimizers used by the core
POMO implementation: global gradient clipping happens before the optimizer,
Adam applies coupled L2 decay, and AdamW applies decoupled parameter decay.
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class AdamState(NamedTuple):
    count: jax.Array
    first_moment: Any
    second_moment: Any


@dataclass(frozen=True, slots=True)
class AdamConfig:
    learning_rate: float = 1.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    decoupled_weight_decay: bool = False
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta values must be in [0, 1)")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip < 0.0:
            raise ValueError("grad_clip must be non-negative")


def init_adam(params: Any) -> AdamState:
    """Create zero moments with the same tree structure and dtypes as params."""

    return AdamState(
        count=jnp.asarray(0, dtype=jnp.int32),
        first_moment=jax.tree.map(jnp.zeros_like, params),
        second_moment=jax.tree.map(jnp.zeros_like, params),
    )


def global_norm(tree: Any) -> jax.Array:
    """Euclidean norm across every array leaf in a pytree."""

    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float32)
    squares = [jnp.sum(jnp.square(leaf.astype(jnp.float32))) for leaf in leaves]
    return jnp.sqrt(jnp.sum(jnp.stack(squares)))


def clip_by_global_norm(grads: Any, max_norm: float) -> tuple[Any, jax.Array]:
    """Clip a gradient pytree like ``torch.nn.utils.clip_grad_norm_``."""

    norm = global_norm(grads)
    if max_norm <= 0.0:
        return grads, norm
    scale = jnp.minimum(
        1.0,
        jnp.asarray(max_norm, jnp.float32) / (norm + jnp.asarray(1.0e-6, jnp.float32)),
    )
    return jax.tree.map(lambda grad: grad * scale.astype(grad.dtype), grads), norm


def adam_update(
    params: Any,
    grads: Any,
    state: AdamState,
    config: AdamConfig,
    *,
    learning_rate: float | jax.Array | None = None,
) -> tuple[Any, AdamState, jax.Array]:
    """Apply one Adam or AdamW update and return the pre-clipping grad norm."""

    grads, grad_norm = clip_by_global_norm(grads, config.grad_clip)
    if config.weight_decay and not config.decoupled_weight_decay:
        grads = jax.tree.map(
            lambda grad, param: grad + config.weight_decay * param,
            grads,
            params,
        )

    count = state.count + jnp.asarray(1, dtype=state.count.dtype)
    first_moment = jax.tree.map(
        lambda moment, grad: config.beta1 * moment + (1.0 - config.beta1) * grad,
        state.first_moment,
        grads,
    )
    second_moment = jax.tree.map(
        lambda moment, grad: config.beta2 * moment + (1.0 - config.beta2) * jnp.square(grad),
        state.second_moment,
        grads,
    )
    count_float = count.astype(jnp.float32)
    bias1 = 1.0 - jnp.power(jnp.asarray(config.beta1, jnp.float32), count_float)
    bias2 = 1.0 - jnp.power(jnp.asarray(config.beta2, jnp.float32), count_float)
    rate = jnp.asarray(
        config.learning_rate if learning_rate is None else learning_rate,
        dtype=jnp.float32,
    )

    def update(param: jax.Array, first: jax.Array, second: jax.Array) -> jax.Array:
        first_hat = first / bias1.astype(first.dtype)
        second_hat = second / bias2.astype(second.dtype)
        base = param
        if config.weight_decay and config.decoupled_weight_decay:
            base = base * (1.0 - rate.astype(param.dtype) * config.weight_decay)
        delta = (
            rate.astype(first.dtype)
            * first_hat
            / (jnp.sqrt(second_hat) + jnp.asarray(config.eps, dtype=second_hat.dtype))
        )
        return base - delta.astype(param.dtype)

    new_params = jax.tree.map(update, params, first_moment, second_moment)
    new_state = AdamState(
        count=count,
        first_moment=first_moment,
        second_moment=second_moment,
    )
    return new_params, new_state, grad_norm


__all__ = [
    "AdamConfig",
    "AdamState",
    "adam_update",
    "clip_by_global_norm",
    "global_norm",
    "init_adam",
]
