"""Constructive policy composed from an encoder and a decoder."""

from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from .decoder import Decoder
from .encoder import Encoder


class ConstructivePolicy(nn.Module):
    """Encoder-decoder constructive policy. Encoder + decoder are DI'd.

    Methods:
        encode(x): run encoder once per problem.
        precompute_decoder_cache(node_embs): optional fixed decoder projections.
        decode_step(...): one decoder call.
        forward(x): convenience returning encode output.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        # Stored as plain attrs so type checkers see the Protocol type.
        # `nn.Module.__setattr__` registers them as sub-modules iff they are nn.Module.
        self.encoder = encoder  # type: ignore[assignment]
        self.decoder = decoder  # type: ignore[assignment]

    def encode(
        self, x: Float[Tensor, "b n d_in"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b d"]]:
        return self.encoder(x)

    def precompute_decoder_cache(self, node_embs: Float[Tensor, "b n d"]) -> object | None:
        """Build an ephemeral cache when the injected decoder supports one."""

        precompute = getattr(self.decoder, "precompute", None)
        if precompute is None:
            return None
        return precompute(node_embs)

    def decode_step(
        self,
        node_embs: Float[Tensor, "b n d"],
        graph_emb: Float[Tensor, "b d"],
        first_idx: Int[Tensor, "b"],
        current_idx: Int[Tensor, "b"],
        mask: Bool[Tensor, "b n"],
        dynamic_context: Float[Tensor, "b c"] | None = None,
        decoder_cache: object | None = None,
    ) -> Float[Tensor, "b n"]:
        if decoder_cache is None:
            return self.decoder(
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=dynamic_context,
            )
        return self.decoder(
            node_embs,
            graph_emb,
            first_idx,
            current_idx,
            mask,
            dynamic_context=dynamic_context,
            decoder_cache=decoder_cache,
        )

    def forward(
        self, x: Float[Tensor, "b n d_in"]
    ) -> tuple[Float[Tensor, "b n d"], Float[Tensor, "b d"]]:
        return self.encode(x)
