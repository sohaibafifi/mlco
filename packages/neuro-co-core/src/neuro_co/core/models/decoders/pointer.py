"""Pointer-attention decoder (Kool 2019 style). Satisfies `Decoder` Protocol.

Two stages:
  1. Multi-head attention from a context query over node embeddings -> refined query.
  2. Single-head pointer attention with `tanh` clip -> logits over nodes.

Mask via large negative bias (compile-friendly). Context is normally built
from `(graph_emb, first_node_emb, current_node_emb)`. Environments may instead
supply one dynamic scalar, in which case the query uses
`(graph_emb, current_node_emb, dynamic_scalar_embedding)`. This gives CVRP its
normalized remaining-capacity context without changing checkpoint shapes.
"""

from typing import NamedTuple

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from ...model import NEG_INF


class PointerDecoderCache(NamedTuple):
    """Fixed node projections shared by every step of one rollout."""

    glimpse_key: Tensor
    glimpse_value: Tensor
    logit_key: Tensor


class PointerDecoder(nn.Module):
    """Pointer-attention decoder."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 8, tanh_clip: float = 10.0) -> None:
        super().__init__()
        self.context_proj = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.point_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.point_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.heads = num_heads
        self.d = hidden_dim
        self.tanh_clip = tanh_clip

    def precompute(self, node_embs: Float[Tensor, "b n d"]) -> PointerDecoderCache:
        """Project fixed node embeddings once while retaining autograd."""

        b, n, d = node_embs.shape
        return PointerDecoderCache(
            glimpse_key=self.k_proj(node_embs)
            .view(b, n, self.heads, d // self.heads)
            .transpose(1, 2),
            glimpse_value=self.v_proj(node_embs)
            .view(b, n, self.heads, d // self.heads)
            .transpose(1, 2),
            logit_key=self.point_k(node_embs),
        )

    def forward(
        self,
        node_embs: Float[Tensor, "b n d"],
        graph_emb: Float[Tensor, "b d"],
        first_idx: Int[Tensor, "b"],
        current_idx: Int[Tensor, "b"],
        mask: Bool[Tensor, "b n"],
        dynamic_context: Float[Tensor, "b c"] | None = None,
        decoder_cache: object | None = None,
    ) -> Float[Tensor, "b n"]:
        b, n, d = node_embs.shape
        first_emb = _gather(node_embs, first_idx)
        cur_emb = _gather(node_embs, current_idx)
        if dynamic_context is None:
            ctx = torch.cat([graph_emb, first_emb, cur_emb], dim=-1)
        else:
            if dynamic_context.shape != (b, 1):
                raise ValueError(
                    "dynamic decoder context must have shape "
                    f"({b}, 1), got {tuple(dynamic_context.shape)}"
                )
            dynamic_emb = dynamic_context.to(
                device=node_embs.device,
                dtype=node_embs.dtype,
            ).expand(b, d)
            ctx = torch.cat([graph_emb, cur_emb, dynamic_emb], dim=-1)
        q = self.context_proj(ctx)

        qh = self.q_proj(q).view(b, 1, self.heads, d // self.heads).transpose(1, 2)
        if decoder_cache is None:
            cache = self.precompute(node_embs)
        elif isinstance(decoder_cache, PointerDecoderCache):
            cache = decoder_cache
        else:
            raise TypeError("decoder_cache must be a PointerDecoderCache")
        kh = cache.glimpse_key
        vh = cache.glimpse_value
        pk = cache.logit_key
        expected_attention_shape = (b, self.heads, n, d // self.heads)
        if kh.shape != expected_attention_shape or vh.shape != expected_attention_shape:
            raise ValueError("decoder cache attention projections have incompatible shapes")
        if pk.shape != (b, n, d):
            raise ValueError("decoder cache logit projection has an incompatible shape")
        attn_mask = mask.view(b, 1, 1, n).expand(b, self.heads, 1, n)
        bias = torch.where(
            attn_mask,
            torch.zeros((), device=q.device),
            torch.full((), NEG_INF, device=q.device),
        )
        att = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=bias)
        att = att.transpose(1, 2).reshape(b, 1, d).squeeze(1)
        q2 = self.out_proj(att)

        pq = self.point_q(q2)
        logits = torch.einsum("bd,bnd->bn", pq, pk) / (d**0.5)
        logits = torch.tanh(logits) * self.tanh_clip
        logits = torch.where(mask, logits, logits.new_full((), NEG_INF))
        return logits


def _gather(x: Float[Tensor, "b n d"], idx: Int[Tensor, "b"]) -> Float[Tensor, "b d"]:
    b, _, d = x.shape
    return x.gather(1, idx.view(b, 1, 1).expand(b, 1, d)).squeeze(1)
