"""End-to-end TSP REINFORCE training script.

Install neuro-co-core and neuro-co-problems before running this example.

Single process:
    uv run --no-sync python packages/neuro-co-core/examples/train_tsp.py \\
        --train.steps 200 --env.size 20 --train.batch_size 256

Multi GPU (N processes, one per GPU):
    torchrun --nproc_per_node=8 packages/neuro-co-core/examples/train_tsp.py \\
        --train.steps 200 --env.size 20 --train.batch_size 256 \\
        --train.device cuda --train.precision bf16
"""

import sys

import torch

from neuro_co.core import (
    Config,
    DistEnv,
    Trainer,
    TrainHooks,
    parse,
)
from neuro_co.core.algos.reinforce import REINFORCE, REINFORCEConfig
from neuro_co.core.distributed import init as dist_init
from neuro_co.core.distributed import shutdown as dist_shutdown
from neuro_co.core.models import AttentionModel
from neuro_co.problems.tsp.env import TSPEnv


def main(argv: list[str] | None = None) -> int:
    cfg: Config = parse(argv)
    dist = DistEnv.from_env()
    dist_init(dist)

    # Pick per-rank device: cuda:LOCAL_RANK if cuda chosen.
    device = _resolve_device(cfg.train.device, dist.local_rank)
    # Per-rank seed so each rank samples a different batch.
    seed = cfg.train.seed + dist.rank

    env = TSPEnv(size=cfg.env.size)
    model = AttentionModel(
        in_dim=env.encoder_in_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
    )
    if cfg.train.compile:
        model.encode = torch.compile(model.encode, mode=cfg.train.compile_mode, fullgraph=True)  # type: ignore[assignment]
        model.decode_step = torch.compile(
            model.decode_step, mode=cfg.train.compile_mode, fullgraph=True
        )  # type: ignore[assignment]
        if dist.is_main:
            print(f"[compile] mode={cfg.train.compile_mode}")

    algo = REINFORCE(
        model=model,
        env=env,
        cfg=REINFORCEConfig(
            batch_size=cfg.train.batch_size,
            lr=cfg.optim.lr,
            grad_clip=cfg.optim.grad_clip,
            precision=cfg.train.precision,  # type: ignore[arg-type]
        ),
        device=device,
    )

    def _print_train(_, m: dict[str, float]) -> None:
        if dist.is_main and algo._step % 10 == 0:
            print(
                f"step={algo._step:6d}  loss={m['loss']:+.4f}  "
                f"R_student={m['reward_student']:+.4f}  R_baseline={m['reward_baseline']:+.4f}"
            )

    def _print_eval(_, m: dict[str, float]) -> None:
        if dist.is_main:
            print(f"  [eval] tour_length={m['eval_tour_length']:.4f}")

    trainer = Trainer(
        algo=algo,
        steps=cfg.train.steps,
        eval_every=cfg.train.eval_every,
        ckpt_every=cfg.train.ckpt_every if dist.is_main else 0,
        device=device,
        seed=seed,
        hooks=TrainHooks(on_step_end=_print_train, on_eval_end=_print_eval),
    )
    trainer.fit()
    dist_shutdown(dist)
    return 0


def _resolve_device(name: str, local_rank: int) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device(name)


if __name__ == "__main__":
    sys.exit(main())
