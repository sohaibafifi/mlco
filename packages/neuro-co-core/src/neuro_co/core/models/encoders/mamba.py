"""Mamba encoder for node features, with optional forward and reverse scans."""

from torch import Tensor, nn

from .ssm_block import Backend, SSMBlock


class SSMEncoder(nn.Module):
    """Mamba encoder implementing the core Encoder protocol.

    Args:
        in_dim: per-node feature width (env.encoder_in_dim).
        hidden_dim: embedding dim.
        num_layers: stacked SSM blocks (per direction).
        d_state, d_conv, expand: Mamba hyperparams.
        bidirectional: sum forward and reversed scans. This is not permutation invariant.
        backend: "cuda" for native kernels, "mambapy" for PyTorch, or None to choose automatically.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        bidirectional: bool = True,
        backend: Backend | None = None,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden_dim)
        self.fwd = SSMBlock(hidden_dim, num_layers, d_state, d_conv, expand, backend)
        self.bwd = (
            SSMBlock(hidden_dim, num_layers, d_state, d_conv, expand, backend)
            if bidirectional
            else None
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.embed(x)
        out = self.fwd(h)
        if self.bwd is not None:
            out = out + self.bwd(h.flip(dims=[1])).flip(dims=[1])
        out = self.norm(out)
        return out, out.mean(dim=1)
