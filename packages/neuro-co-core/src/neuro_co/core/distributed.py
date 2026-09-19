"""Distributed training helpers using manual gradient reduction.

Each rank constructs its algorithm with a separate RNG seed. Reduce gradients
after loss.backward() and before the optimizer step so ranks keep the same
weights. Policies call encode() and decode_step() separately rather than a DDP
forward() wrapper.

Launch with torchrun, for example:
    torchrun --nproc_per_node=8 packages/neuro-co-problems/examples/train_tsp.py ...

torchrun supplies WORLD_SIZE, RANK, LOCAL_RANK, MASTER_ADDR, and MASTER_PORT."""

import os
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True, frozen=True)
class DistEnv:
    """Read torchrun env vars. Defaults = single-process."""

    world_size: int
    rank: int
    local_rank: int
    backend: str

    @classmethod
    def from_env(cls, backend: str = "nccl") -> "DistEnv":
        return cls(
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            backend=backend,
        )

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init(env: DistEnv) -> None:
    """Initialize process group if needed. No-op for single-process."""
    if not env.enabled:
        return
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(env.backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(env.local_rank)


def shutdown(env: DistEnv) -> None:
    if env.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def all_reduce_grads(model: nn.Module, world_size: int) -> None:
    """Average gradients across ranks. Call after `loss.backward()`."""
    if world_size <= 1:
        return
    for p in model.parameters():
        if p.grad is not None:
            torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
            p.grad.div_(world_size)


def broadcast_params(model: nn.Module, src: int = 0) -> None:
    """Broadcast model params from `src` to all ranks. Use after rank-0-only
    decisions like baseline refresh in REINFORCE."""
    if not torch.distributed.is_initialized():
        return
    for p in model.parameters():
        torch.distributed.broadcast(p.data, src=src)


def all_reduce_mean(t: torch.Tensor, world_size: int) -> torch.Tensor:
    """Mean of a scalar/tensor across ranks. For metrics."""
    if world_size <= 1:
        return t
    out = t.detach().clone()
    torch.distributed.all_reduce(out, op=torch.distributed.ReduceOp.SUM)
    return out / world_size


__all__ = [
    "DistEnv",
    "all_reduce_grads",
    "all_reduce_mean",
    "broadcast_params",
    "init",
    "shutdown",
]
