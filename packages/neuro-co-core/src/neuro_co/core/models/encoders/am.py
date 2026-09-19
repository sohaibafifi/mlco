"""Attention-Model encoder (Kool 2019). Satisfies `Encoder` Protocol.

Pre-norm multi-head self-attention stack with FFN. Mean-pool for graph
embedding. Uses `F.scaled_dot_product_attention` (flash backend when
available). No TensorDict, no dynamic shapes.
"""

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ..encoder import pool_mean


class _MHABlock(nn.Module):
    """Pre-norm multi-head self-attention + FFN."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.GELU(),
            nn.Linear(ff_mult * d, d),
        )
        self.heads = heads
        self.d = d
        self.dropout = dropout

    def forward(self, x: Float[Tensor, "b n d"]) -> Float[Tensor, "b n d"]:
        h = self.norm1(x)
        b, n, d = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        att = att.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(att)
        x = x + self.ff(self.norm2(x))
        return x


class AMEncoder(nn.Module):
    """Embed -> stack of MHA blocks -> (node_embs, graph_emb).

    Satisfies `Encoder` Protocol.
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [_MHABlock(hidden_dim, num_heads, ff_mult, dropout) for _ in range(num_layers)]
        )

    def forward(
        self, x: Float[Tensor, "b n d_in"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b d"]]:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return h, pool_mean(h)
