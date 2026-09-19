"""`Decoder` Protocol.

Pointer-style decoder consumes node embeddings + graph embedding + per-step
context indices, returns logits over actions. Concrete class:
  - `decoders/pointer.py:PointerDecoder`

If non-pointer decoders are added later (e.g., MLP head, RNN-based), this
Protocol is the shared contract.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Bool, Float, Int
from torch import Tensor


@runtime_checkable
class Decoder(Protocol):
    """One-step decoder. Returns masked logits over the action set."""

    def __call__(
        self,
        node_embs: Float[Tensor, "b n d"],
        graph_emb: Float[Tensor, "b d"],
        first_idx: Int[Tensor, "b"],
        current_idx: Int[Tensor, "b"],
        mask: Bool[Tensor, "b n"],
        dynamic_context: Float[Tensor, "b c"] | None = None,
        decoder_cache: object | None = None,
    ) -> Float[Tensor, "b n"]: ...


__all__ = ["Decoder"]
