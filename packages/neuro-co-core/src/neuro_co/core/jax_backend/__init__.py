"""Pure-JAX model, optimization, and training primitives.

Import this backend explicitly so the ordinary core API remains independent
of JAX runtime initialization.
"""

from .am import (
    AMBlockParams,
    AMEncoderParams,
    AttentionModelParams,
    JaxAttentionModel,
    LayerNormParams,
    LinearParams,
    PointerDecoderCache,
    PointerDecoderParams,
)
from .optim import (
    AdamConfig,
    AdamState,
    adam_update,
    clip_by_global_norm,
    global_norm,
    init_adam,
)
from .pomo import (
    JaxPOMO,
    POMOLossMetrics,
    POMOTrainMetrics,
    POMOTrainState,
    distinct_first_actions,
    pomo_advantage,
    pomo_loss,
)

__all__ = [
    "AMBlockParams",
    "AMEncoderParams",
    "AdamConfig",
    "AdamState",
    "AttentionModelParams",
    "JaxAttentionModel",
    "JaxPOMO",
    "LayerNormParams",
    "LinearParams",
    "POMOLossMetrics",
    "POMOTrainMetrics",
    "POMOTrainState",
    "PointerDecoderCache",
    "PointerDecoderParams",
    "adam_update",
    "clip_by_global_norm",
    "distinct_first_actions",
    "global_norm",
    "init_adam",
    "pomo_advantage",
    "pomo_loss",
]
