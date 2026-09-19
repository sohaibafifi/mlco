"""MatNet-style edge-aware encoder (Kwon et al. 2021). Satisfies `Encoder`.

Consumes an `(b, n, n)` relation/distance matrix instead of node coords.
Each attention layer uses a *mixed score*: content attention `q·k` plus a
learned per-head bias derived from the edge weight `D_ij`. This injects the
problem matrix directly into attention, the key MatNet idea: without the
full dual row/col machinery (for square problems like ATSP, rows == cols).

Node init: project each row of `D` (its out-distances) to the hidden dim.
Edge bias: `MLP(D_ij) -> (heads,)`, added to attention logits via SDPA's
additive `attn_mask`. Stays compile-friendly (no Python branching).
"""

import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ..encoder import pool_mean


class _MixedScoreBlock(nn.Module):
    """Pre-norm MHA with additive per-head edge bias + FFN."""

    def __init__(self, d: int, heads: int, ff_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ff = nn.Sequential(nn.Linear(d, ff_mult * d), nn.GELU(), nn.Linear(ff_mult * d, d))
        self.heads = heads
        self.dropout = dropout

    def forward(
        self, x: Float[Tensor, "b n d"], edge_bias: Float[Tensor, "b h n n"]
    ) -> Float[Tensor, "b n d"]:
        h = self.norm1(x)
        b, n, d = h.shape
        qkv = self.qkv(h).reshape(b, n, 3, self.heads, d // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        att = F.scaled_dot_product_attention(
            q, k, v, attn_mask=edge_bias, dropout_p=self.dropout if self.training else 0.0
        )
        att = att.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(att)
        x = x + self.ff(self.norm2(x))
        return x


class MatNetEncoder(nn.Module):
    """Edge-aware encoder over an `(b, n, n)` matrix. Satisfies `Encoder`."""

    def __init__(
        self,
        problem_size: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.row_embed = nn.Linear(problem_size, hidden_dim)
        # Edge weight (scalar) -> per-head additive bias.
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, num_heads), nn.GELU(), nn.Linear(num_heads, num_heads)
        )
        self.blocks = nn.ModuleList(
            [_MixedScoreBlock(hidden_dim, num_heads, ff_mult, dropout) for _ in range(num_layers)]
        )
        self.heads = num_heads

    def forward(
        self, x: Float[Tensor, "b n n"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b d"]]:
        edge_bias = self.edge_mlp(x.unsqueeze(-1))  # (b, n, n, heads)
        edge_bias = edge_bias.permute(0, 3, 1, 2).contiguous()  # (b, heads, n, n)
        h = self.row_embed(x)  # (b, n, d)
        for blk in self.blocks:
            h = blk(h, edge_bias)
        return h, pool_mean(h)
