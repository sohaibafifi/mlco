"""Policy factories pairing each encoder with the core pointer decoder."""

from .decoders.pointer import PointerDecoder
from .encoders.am import AMEncoder
from .encoders.mamba import SSMEncoder
from .encoders.matnet import MatNetEncoder
from .encoders.ssm_block import Backend
from .policy import ConstructivePolicy


def AttentionModel(  # noqa: N802 (preserve paper-faithful name)
    in_dim: int = 2,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    ff_mult: int = 4,
    dropout: float = 0.0,
    tanh_clip: float = 10.0,
) -> ConstructivePolicy:
    """Kool 2019 Attention Model preset.

    Encoder = transformer (AMEncoder), decoder = pointer attention.
    Equivalent to `ConstructivePolicy(AMEncoder(...), PointerDecoder(...))`.
    """
    return ConstructivePolicy(
        encoder=AMEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        ),
        decoder=PointerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, tanh_clip=tanh_clip),
    )


def MatNetModel(  # noqa: N802 (paper-faithful name)
    problem_size: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    ff_mult: int = 4,
    dropout: float = 0.0,
    tanh_clip: float = 10.0,
) -> ConstructivePolicy:
    """MatNet preset for matrix problems (ATSP, FFSP).

    Encoder = edge-aware mixed-score (MatNetEncoder) over the `(b, n, n)`
    matrix; decoder = pointer attention. `problem_size` = matrix width.
    """
    return ConstructivePolicy(
        encoder=MatNetEncoder(
            problem_size=problem_size,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        ),
        decoder=PointerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, tanh_clip=tanh_clip),
    )


def GNNModel(  # noqa: N802 (factory, mirrors AttentionModel naming)
    in_dim: int = 3,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    *,
    k_sparse: int = 10,
    dropout: float = 0.1,
    rbf_k: int = 16,
    fourier_feats: int = 1,
    residual: bool = True,
    tanh_clip: float = 10.0,
) -> ConstructivePolicy:
    """Sparse GNN encoder plus the core pointer decoder."""

    from .encoders.gnn import GNNEncoder

    return ConstructivePolicy(
        encoder=GNNEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            k_sparse=k_sparse,
            dropout=dropout,
            rbf_k=rbf_k,
            fourier_feats=fourier_feats,
            residual=residual,
        ),
        decoder=PointerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, tanh_clip=tanh_clip),
    )


def MambaModel(  # noqa: N802 (factory, mirrors core.AttentionModel naming)
    in_dim: int,
    hidden_dim: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    d_state: int = 16,
    d_conv: int = 4,
    expand: int = 2,
    bidirectional: bool = True,
    tanh_clip: float = 10.0,
    backend: Backend | None = None,
) -> ConstructivePolicy:
    """Mamba encoder and pointer decoder.

    Use the same environment and algorithm interfaces as AttentionModel with
    any core env:

        env = TSPEnv(size=50)
        model = MambaModel(in_dim=env.encoder_in_dim)
        algo = POMO(model, env, POMOConfig(...))
    """
    return ConstructivePolicy(
        encoder=SSMEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            bidirectional=bidirectional,
            backend=backend,
        ),
        decoder=PointerDecoder(hidden_dim=hidden_dim, num_heads=num_heads, tanh_clip=tanh_clip),
    )


__all__ = ["AttentionModel", "GNNModel", "MambaModel", "MatNetModel"]
