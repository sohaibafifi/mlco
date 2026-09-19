from .am import AMEncoder
from .mamba import SSMEncoder
from .matnet import MatNetEncoder

__all__ = ["AMEncoder", "GNNEncoder", "MatNetEncoder", "SSMEncoder"]


def __getattr__(name: str):
    if name == "GNNEncoder":
        from .gnn import GNNEncoder

        return GNNEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
