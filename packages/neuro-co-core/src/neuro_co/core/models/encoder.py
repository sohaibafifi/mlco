"""Encoder protocol for node embeddings and a pooled graph embedding.

AMEncoder, GNNEncoder, MatNetEncoder, and SSMEncoder implement this interface.
"""

from typing import Protocol, runtime_checkable

from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class Encoder(Protocol):
    """Maps a per-node feature tensor to (node_embs, graph_emb)."""

    def __call__(
        self,
        x: Float[Tensor, "b n d_in"],
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b d"]]: ...


def pool_mean(node_embs: Float[Tensor, "b n d"]) -> Float[Tensor, "b d"]:
    """Default graph-embedding pool. Encoders can use or ignore."""
    return node_embs.mean(dim=1)


def pool_max(node_embs: Float[Tensor, "b n d"]) -> Float[Tensor, "b d"]:
    """Max-pool variant. Available to encoders that prefer it."""
    return node_embs.max(dim=1).values


__all__ = ["Encoder", "pool_max", "pool_mean"]
