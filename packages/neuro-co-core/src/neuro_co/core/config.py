"""`tyro` config dataclasses.

CLI parsing via `tyro.cli`. Sweeps via W&B Sweeps or Optuna directly.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class OptimConfig:
    lr: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip: float = 1.0


@dataclass(slots=True)
class TrainConfig:
    steps: int = 100_000
    eval_every: int = 1_000
    ckpt_every: int = 5_000
    batch_size: int = 512
    device: str = "cpu"
    seed: int = 0
    compile: bool = False
    compile_mode: str = "default"  # "default" | "reduce-overhead" | "max-autotune"
    precision: str = "fp32"  # "fp32" | "bf16" | "fp16"


@dataclass(slots=True)
class ModelConfig:
    """Generic model hyperparams. Algo-specific configs extend this."""

    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 8
    dropout: float = 0.0


@dataclass(slots=True)
class EnvConfig:
    """Problem-instance hyperparams. Env-specific configs extend this."""

    name: str = "tsp"
    size: int = 20


@dataclass(slots=True)
class LogConfig:
    project: str = "neuro-co-core"
    run_name: str | None = None
    backend: str = "stdout"


@dataclass(slots=True)
class Config:
    """Top-level config. Compose into experiment-specific configs."""

    train: TrainConfig = field(default_factory=TrainConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    log: LogConfig = field(default_factory=LogConfig)


def parse(argv: list[str] | None = None) -> Config:
    """Parse CLI args into a `Config`. Tiny wrapper around `tyro.cli`.

    Importing tyro lazily so the package stays importable without tyro at
    library-only call sites.
    """
    import tyro

    return tyro.cli(Config, args=argv)
