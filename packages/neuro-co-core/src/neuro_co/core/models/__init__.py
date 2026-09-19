from .decoder import Decoder
from .decoders.pointer import PointerDecoder
from .encoder import Encoder, pool_max, pool_mean
from .encoders.am import AMEncoder
from .encoders.mamba import SSMEncoder
from .encoders.matnet import MatNetEncoder
from .policy import ConstructivePolicy
from .presets import AttentionModel, GNNModel, MambaModel, MatNetModel

__all__ = [
    "AMEncoder",
    "AttentionModel",
    "ConstructivePolicy",
    "Decoder",
    "Encoder",
    "GNNEncoder",
    "GNNModel",
    "MambaModel",
    "MatNetEncoder",
    "MatNetModel",
    "PointerDecoder",
    "SSMEncoder",
    "pool_max",
    "pool_mean",
]


def __getattr__(name: str):
    if name == "GNNEncoder":
        from .encoders.gnn import GNNEncoder

        return GNNEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
