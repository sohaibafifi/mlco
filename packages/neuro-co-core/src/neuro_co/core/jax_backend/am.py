"""Pure-JAX Attention Model.

The parameter tree intentionally mirrors the PyTorch Attention Model so the
same weights can be evaluated by both backends. Linear weights use PyTorch's
``(out_features, in_features)`` layout; this keeps conversion explicit and
lossless while remaining efficient after JIT compilation.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

NEG_INF = -1.0e9


class LinearParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array | None


class LayerNormParams(NamedTuple):
    weight: jax.Array
    bias: jax.Array


class AMBlockParams(NamedTuple):
    norm1: LayerNormParams
    norm2: LayerNormParams
    qkv: LinearParams
    proj: LinearParams
    ff1: LinearParams
    ff2: LinearParams


class AMEncoderParams(NamedTuple):
    embed: LinearParams
    blocks: tuple[AMBlockParams, ...]


class PointerDecoderParams(NamedTuple):
    context_proj: LinearParams
    q_proj: LinearParams
    k_proj: LinearParams
    v_proj: LinearParams
    out_proj: LinearParams
    point_q: LinearParams
    point_k: LinearParams


class AttentionModelParams(NamedTuple):
    encoder: AMEncoderParams
    decoder: PointerDecoderParams


class PointerDecoderCache(NamedTuple):
    glimpse_key: jax.Array
    glimpse_value: jax.Array
    logit_key: jax.Array


@dataclass(frozen=True, slots=True)
class JaxAttentionModel:
    """Functional AM encoder and pointer decoder with an explicit parameter tree."""

    in_dim: int = 2
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.0
    tanh_clip: float = 10.0
    precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads:
            raise ValueError("num_heads must divide hidden_dim")
        if self.ff_mult <= 0:
            raise ValueError("ff_mult must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.tanh_clip <= 0.0:
            raise ValueError("tanh_clip must be positive")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")

    @property
    def compute_dtype(self) -> jnp.dtype:
        return {
            "fp32": jnp.dtype(jnp.float32),
            "bf16": jnp.dtype(jnp.bfloat16),
            "fp16": jnp.dtype(jnp.float16),
        }[self.precision]

    def init(self, key: jax.Array) -> AttentionModelParams:
        """Initialize parameters with the same distributions as ``nn.Linear``."""

        keys = iter(jax.random.split(key, 1 + 4 * self.num_layers + 7))
        embed = _init_linear(next(keys), self.in_dim, self.hidden_dim, bias=True)
        blocks: list[AMBlockParams] = []
        for _ in range(self.num_layers):
            blocks.append(
                AMBlockParams(
                    norm1=_init_layer_norm(self.hidden_dim),
                    norm2=_init_layer_norm(self.hidden_dim),
                    qkv=_init_linear(next(keys), self.hidden_dim, 3 * self.hidden_dim, bias=False),
                    proj=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
                    ff1=_init_linear(
                        next(keys),
                        self.hidden_dim,
                        self.ff_mult * self.hidden_dim,
                        bias=True,
                    ),
                    ff2=_init_linear(
                        next(keys),
                        self.ff_mult * self.hidden_dim,
                        self.hidden_dim,
                        bias=True,
                    ),
                )
            )
        decoder = PointerDecoderParams(
            context_proj=_init_linear(next(keys), 3 * self.hidden_dim, self.hidden_dim, bias=False),
            q_proj=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
            k_proj=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
            v_proj=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
            out_proj=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
            point_q=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
            point_k=_init_linear(next(keys), self.hidden_dim, self.hidden_dim, bias=False),
        )
        return AttentionModelParams(
            encoder=AMEncoderParams(embed=embed, blocks=tuple(blocks)),
            decoder=decoder,
        )

    def encode(
        self,
        params: AttentionModelParams,
        features: jax.Array,
        *,
        training: bool = False,
        key: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Encode node features once and return node and graph embeddings."""

        if features.ndim != 3 or features.shape[-1] != self.in_dim:
            raise ValueError(
                f"features must have shape (batch, nodes, {self.in_dim}), got {features.shape}"
            )
        if self.precision == "fp16" and jax.default_backend() == "cpu":
            raise ValueError("fp16 computation requires a non-CPU JAX backend")
        if training and self.dropout > 0.0 and key is None:
            raise ValueError("dropout training requires a PRNG key")
        layer_keys = (
            jax.random.split(key, self.num_layers) if key is not None else (None,) * self.num_layers
        )
        dtype = self.compute_dtype
        h = _linear(params.encoder.embed, features, dtype)
        for block, layer_key in zip(params.encoder.blocks, layer_keys, strict=True):
            h = self._encoder_block(block, h, training=training, key=layer_key)
        return h, jnp.mean(h, axis=1)

    def _encoder_block(
        self,
        params: AMBlockParams,
        x: jax.Array,
        *,
        training: bool,
        key: jax.Array | None,
    ) -> jax.Array:
        dtype = self.compute_dtype
        h = _layer_norm(params.norm1, x, dtype)
        batch, nodes, hidden = h.shape
        head_dim = hidden // self.num_heads
        qkv = _linear(params.qkv, h, dtype).reshape(batch, nodes, 3, self.num_heads, head_dim)
        q, k, v = (qkv[:, :, index] for index in range(3))
        if training and self.dropout > 0.0:
            if key is None:
                raise ValueError("dropout training requires a PRNG key")
            attention = _dropout_attention(q, k, v, self.dropout, key)
        else:
            attention = jax.nn.dot_product_attention(q, k, v)
        attention = attention.reshape(batch, nodes, hidden)
        x = x + _linear(params.proj, attention, dtype)
        ff = _linear(params.ff1, _layer_norm(params.norm2, x, dtype), dtype)
        ff = jax.nn.gelu(ff, approximate=False)
        return x + _linear(params.ff2, ff, dtype)

    def precompute_decoder_cache(
        self, params: AttentionModelParams, node_embs: jax.Array
    ) -> PointerDecoderCache:
        """Project the fixed node embeddings once for a complete rollout."""

        batch, nodes, hidden = node_embs.shape
        head_dim = hidden // self.num_heads
        decoder = params.decoder
        return PointerDecoderCache(
            glimpse_key=_linear(decoder.k_proj, node_embs, self.compute_dtype).reshape(
                batch, nodes, self.num_heads, head_dim
            ),
            glimpse_value=_linear(decoder.v_proj, node_embs, self.compute_dtype).reshape(
                batch, nodes, self.num_heads, head_dim
            ),
            logit_key=_linear(decoder.point_k, node_embs, self.compute_dtype),
        )

    def decode_step(
        self,
        params: AttentionModelParams,
        node_embs: jax.Array,
        graph_emb: jax.Array,
        first_idx: jax.Array,
        current_idx: jax.Array,
        mask: jax.Array,
        *,
        dynamic_context: jax.Array | None = None,
        decoder_cache: PointerDecoderCache | None = None,
    ) -> jax.Array:
        """Return masked pointer logits for one constructive decoding step."""

        batch, nodes, hidden = node_embs.shape
        if mask.shape != (batch, nodes):
            raise ValueError(f"mask must have shape {(batch, nodes)}, got {mask.shape}")
        first_emb = _gather(node_embs, first_idx)
        current_emb = _gather(node_embs, current_idx)
        if dynamic_context is None:
            context = jnp.concatenate((graph_emb, first_emb, current_emb), axis=-1)
        else:
            if dynamic_context.shape != (batch, 1):
                raise ValueError(
                    f"dynamic_context must have shape {(batch, 1)}, got {dynamic_context.shape}"
                )
            dynamic_emb = jnp.broadcast_to(
                dynamic_context.astype(self.compute_dtype), (batch, hidden)
            )
            context = jnp.concatenate((graph_emb, current_emb, dynamic_emb), axis=-1)

        decoder = params.decoder
        query = _linear(decoder.context_proj, context, self.compute_dtype)
        head_dim = hidden // self.num_heads
        query = _linear(decoder.q_proj, query, self.compute_dtype).reshape(
            batch, 1, self.num_heads, head_dim
        )
        cache = (
            decoder_cache
            if decoder_cache is not None
            else self.precompute_decoder_cache(params, node_embs)
        )
        attention = jax.nn.dot_product_attention(
            query,
            cache.glimpse_key,
            cache.glimpse_value,
            mask=mask[:, None, None, :],
        ).reshape(batch, hidden)
        refined = _linear(decoder.out_proj, attention, self.compute_dtype)
        pointer_query = _linear(decoder.point_q, refined, self.compute_dtype)
        logits = jnp.einsum("bd,bnd->bn", pointer_query, cache.logit_key)
        logits = logits / jnp.sqrt(jnp.asarray(hidden, dtype=logits.dtype))
        logits = jnp.tanh(logits) * self.tanh_clip
        return jnp.where(mask, logits, jnp.asarray(NEG_INF, dtype=logits.dtype))

    def from_torch_state_dict(self, state_dict: Mapping[str, Any]) -> AttentionModelParams:
        """Convert a matching PyTorch state dict without importing PyTorch."""

        expected = _state_dict_names(self.num_layers)
        missing = expected.difference(state_dict)
        unexpected = set(state_dict).difference(expected)
        if missing or unexpected:
            raise ValueError(
                f"state_dict keys differ: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        def array(name: str) -> jax.Array:
            value = state_dict[name]
            detach = getattr(value, "detach", None)
            if callable(detach):
                value = detach()
            cpu = getattr(value, "cpu", None)
            if callable(cpu):
                value = cpu()
            numpy = getattr(value, "numpy", None)
            if callable(numpy):
                value = numpy()
            return jnp.asarray(np.asarray(value, dtype=np.float32))

        def linear(prefix: str, *, bias: bool) -> LinearParams:
            return LinearParams(
                weight=array(f"{prefix}.weight"),
                bias=array(f"{prefix}.bias") if bias else None,
            )

        blocks = tuple(
            AMBlockParams(
                norm1=LayerNormParams(
                    array(f"encoder.blocks.{index}.norm1.weight"),
                    array(f"encoder.blocks.{index}.norm1.bias"),
                ),
                norm2=LayerNormParams(
                    array(f"encoder.blocks.{index}.norm2.weight"),
                    array(f"encoder.blocks.{index}.norm2.bias"),
                ),
                qkv=linear(f"encoder.blocks.{index}.qkv", bias=False),
                proj=linear(f"encoder.blocks.{index}.proj", bias=False),
                ff1=linear(f"encoder.blocks.{index}.ff.0", bias=True),
                ff2=linear(f"encoder.blocks.{index}.ff.2", bias=True),
            )
            for index in range(self.num_layers)
        )
        decoder = PointerDecoderParams(
            context_proj=linear("decoder.context_proj", bias=False),
            q_proj=linear("decoder.q_proj", bias=False),
            k_proj=linear("decoder.k_proj", bias=False),
            v_proj=linear("decoder.v_proj", bias=False),
            out_proj=linear("decoder.out_proj", bias=False),
            point_q=linear("decoder.point_q", bias=False),
            point_k=linear("decoder.point_k", bias=False),
        )
        params = AttentionModelParams(
            encoder=AMEncoderParams(
                embed=linear("encoder.embed", bias=True),
                blocks=blocks,
            ),
            decoder=decoder,
        )
        self._validate_params(params)
        return params

    def _validate_params(self, params: AttentionModelParams) -> None:
        """Fail early when imported parameters do not match this model."""

        probe = jnp.zeros((1, 2, self.in_dim), dtype=jnp.float32)
        node_embs, graph_emb = self.encode(params, probe)
        if node_embs.shape != (1, 2, self.hidden_dim) or graph_emb.shape != (
            1,
            self.hidden_dim,
        ):
            raise ValueError("parameter shapes do not match the model configuration")


def _init_linear(
    key: jax.Array, in_features: int, out_features: int, *, bias: bool
) -> LinearParams:
    bound = 1.0 / np.sqrt(in_features)
    weight = jax.random.uniform(
        key,
        (out_features, in_features),
        minval=-bound,
        maxval=bound,
        dtype=jnp.float32,
    )
    bias_value = None
    if bias:
        bias_value = jax.random.uniform(
            jax.random.fold_in(key, 1),
            (out_features,),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        )
    return LinearParams(weight=weight, bias=bias_value)


def _init_layer_norm(hidden_dim: int) -> LayerNormParams:
    return LayerNormParams(
        weight=jnp.ones((hidden_dim,), dtype=jnp.float32),
        bias=jnp.zeros((hidden_dim,), dtype=jnp.float32),
    )


def _linear(params: LinearParams, x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    output = jnp.einsum("...i,oi->...o", x.astype(dtype), params.weight.astype(dtype))
    if params.bias is not None:
        output = output + params.bias.astype(dtype)
    return output


def _layer_norm(params: LayerNormParams, x: jax.Array, dtype: jnp.dtype) -> jax.Array:
    stable = x.astype(jnp.float32)
    mean = jnp.mean(stable, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(stable - mean), axis=-1, keepdims=True)
    normalized = (stable - mean) * jax.lax.rsqrt(variance + 1.0e-5)
    normalized = normalized * params.weight + params.bias
    return normalized.astype(dtype)


def _dropout_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    dropout: float,
    rng: jax.Array,
) -> jax.Array:
    head_dim = query.shape[-1]
    scores = jnp.einsum("bthd,bshd->bhts", query, key)
    scores = scores / jnp.sqrt(jnp.asarray(head_dim, dtype=scores.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    keep = jax.random.bernoulli(rng, 1.0 - dropout, weights.shape)
    weights = jnp.where(keep, weights / (1.0 - dropout), 0.0)
    return jnp.einsum("bhts,bshd->bthd", weights, value)


def _gather(x: jax.Array, indices: jax.Array) -> jax.Array:
    return x[jnp.arange(x.shape[0]), indices]


def _state_dict_names(num_layers: int) -> set[str]:
    names = {"encoder.embed.weight", "encoder.embed.bias"}
    for index in range(num_layers):
        prefix = f"encoder.blocks.{index}"
        names.update(
            {
                f"{prefix}.norm1.weight",
                f"{prefix}.norm1.bias",
                f"{prefix}.norm2.weight",
                f"{prefix}.norm2.bias",
                f"{prefix}.qkv.weight",
                f"{prefix}.proj.weight",
                f"{prefix}.ff.0.weight",
                f"{prefix}.ff.0.bias",
                f"{prefix}.ff.2.weight",
                f"{prefix}.ff.2.bias",
            }
        )
    for name in (
        "context_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "point_q",
        "point_k",
    ):
        names.add(f"decoder.{name}.weight")
    return names


__all__ = [
    "AMBlockParams",
    "AMEncoderParams",
    "AttentionModelParams",
    "JaxAttentionModel",
    "LayerNormParams",
    "LinearParams",
    "PointerDecoderCache",
    "PointerDecoderParams",
]
