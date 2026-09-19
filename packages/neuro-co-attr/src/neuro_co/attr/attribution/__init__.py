"""Attribution methods for autoregressive CO policies.

- `AttributionTrace`: common dataclass returned by every method.
- `gradient_attribution`: gradient x feature (cheap baseline).
- `contrastive_attribution`: gradient of `log pi(a) - log pi(b)`.
- `integrated_gradients`: Riemann-sum path integral.
- `deeplift_attribution`: DeepLIFT-Rescale with completeness diagnostics.

Method-specific modules live alongside this file; `_common` holds the
shared rollout driver + trace packing, built on `neuro-co-core`.
"""

from __future__ import annotations

from neuro_co.attr.attribution._common import AttributionTrace
from neuro_co.attr.attribution.contrastive import contrastive_attribution
from neuro_co.attr.attribution.deeplift import deeplift_attribution
from neuro_co.attr.attribution.gradient import gradient_attribution
from neuro_co.attr.attribution.ig import integrated_gradients

__all__ = [
    "AttributionTrace",
    "contrastive_attribution",
    "deeplift_attribution",
    "gradient_attribution",
    "integrated_gradients",
]
