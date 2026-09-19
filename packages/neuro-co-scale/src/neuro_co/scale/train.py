"""REINFORCE training loop for the hierarchical VRP policy.

The trainer is a thin, partition-based analogue of core POMO. Core POMO drives a
core ``Env``; the hierarchical policy instead decodes over a precomputed
partition, so it needs its own rollout. The objective is identical in spirit:
sample ``n_starts`` rollouts per instance, use their mean cost as a baseline,
and apply the REINFORCE gradient with that group-mean advantage.

Reward is negative route cost, so minimizing cost is maximizing reward. With
``advantage = cost - mean_cost`` (detached), the loss is
``(advantage * log_prob).mean()`` minimized by gradient descent: rollouts that
beat the group mean get their log-probability pushed up.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from copy import copy, deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .decode import HierarchicalPolicy
from .types import Partition, VRPInstance

PartitionFn = Callable[[VRPInstance], Partition]
InstanceSampler = Callable[[int], list[VRPInstance]]
TrainingState = dict[str, Any]
_TRAINING_STATE_VERSION = 1


@dataclass(slots=True)
class TrainConfig:
    steps: int = 200
    n_starts: int = 16  # rollouts (POMO-style starts) per instance
    batch_instances: int = 8
    lr: float = 1e-4
    grad_clip: float = 1.0
    lr_warmup_steps: int = 0  # >0 enables linear warmup then cosine decay to 0
    seed: int = 0
    device: str = "cpu"


def _lr_factor(step: int, warmup: int, total: int) -> float:
    """Linear warmup to 1.0 over ``warmup`` steps, then cosine decay to 0."""

    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True, slots=True)
class TrainStep:
    step: int
    loss: float
    mean_cost: float
    best_cost: float
    baseline_cost: float


def reinforce_loss(costs: Tensor, log_prob: Tensor) -> tuple[Tensor, Tensor]:
    """Group-mean-baseline REINFORCE loss for one instance's rollouts."""

    if costs.shape != log_prob.shape:
        raise ValueError("costs and log_prob must have the same shape")
    if costs.numel() < 2:
        raise ValueError("need at least two rollouts to form a group baseline")
    baseline = costs.mean()
    advantage = (costs - baseline).detach().to(log_prob.device)
    return (advantage * log_prob).mean(), baseline


def _cpu_snapshot(value: Any) -> Any:
    """Copy a state tree without retaining live tensors or accelerator memory."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        result = copy(value)  # Preserve OrderedDict metadata on model state dicts.
        for key, item in value.items():
            result[key] = _cpu_snapshot(item)
        return result
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    return deepcopy(value)


def _policy_config(policy: HierarchicalPolicy) -> dict[str, Any]:
    builder = policy.action_set_builder
    return {
        "reanchor": policy.reanchor,
        "use_checkpoint": policy.use_checkpoint,
        "checkpoint_chunk": policy.checkpoint_chunk,
        "action_set_builder": f"{type(builder).__module__}.{type(builder).__qualname__}",
        "neighbor_span": getattr(builder, "neighbor_span", None),
    }


def _training_state(
    policy: HierarchicalPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    generator: torch.Generator,
    cfg: TrainConfig,
    history: list[TrainStep],
) -> TrainingState:
    return _cpu_snapshot(
        {
            "format_version": _TRAINING_STATE_VERSION,
            "completed_step": len(history),
            "state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
            "sampling_generator_state": generator.get_state(),
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "train_config": asdict(cfg),
            "policy_config": _policy_config(policy),
            "history": [asdict(record) for record in history],
        }
    )


def _restore_training_state(
    state: Mapping[str, Any],
    policy: HierarchicalPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    generator: torch.Generator,
    cfg: TrainConfig,
) -> list[TrainStep]:
    if state.get("format_version") != _TRAINING_STATE_VERSION:
        raise ValueError("unsupported training-state format_version")
    saved_cfg = dict(state["train_config"])
    current_cfg = asdict(cfg)
    # Device migration is permitted for recovery, but exact numerical replay is
    # only promised with the same hardware, runtime, and deterministic settings.
    saved_cfg.pop("device", None)
    current_cfg.pop("device", None)
    if saved_cfg != current_cfg:
        raise ValueError(
            "resume train_config differs; preserve the original total steps and schedule"
        )
    if state["policy_config"] != _policy_config(policy):
        raise ValueError("resume policy_config differs from the saved decoding configuration")
    completed = state["completed_step"]
    if not isinstance(completed, int) or not 0 <= completed <= cfg.steps:
        raise ValueError("invalid completed_step in training state")
    history = [TrainStep(**record) for record in state["history"]]
    if len(history) != completed or [record.step for record in history] != list(range(completed)):
        raise ValueError("training-state history does not match completed_step")
    saved_scheduler = state["scheduler_state_dict"]
    if (saved_scheduler is None) != (scheduler is None):
        raise ValueError("training-state scheduler does not match the training configuration")

    policy.load_state_dict(state["state_dict"])
    # Loading the optimizer maps its tensor state to its parameters' devices.
    # Copy first so subsequent training cannot mutate the supplied checkpoint.
    optimizer.load_state_dict(_cpu_snapshot(state["optimizer_state_dict"]))
    if scheduler is not None:
        scheduler.load_state_dict(deepcopy(saved_scheduler))
    generator.set_state(state["sampling_generator_state"].cpu())
    torch.random.set_rng_state(state["torch_rng_state"].cpu())
    cuda_states = state["cuda_rng_states"]
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA device count differs from the saved RNG state")
        torch.cuda.set_rng_state_all([rng.cpu() for rng in cuda_states])
    return history


def train_hierarchical(
    policy: HierarchicalPolicy,
    *,
    partition_fn: PartitionFn,
    instances: InstanceSampler,
    cfg: TrainConfig,
    on_step: Callable[[TrainStep], None] | None = None,
    resume_state: Mapping[str, Any] | None = None,
    on_checkpoint: Callable[[TrainingState], None] | None = None,
    checkpoint_every: int = 1,
    max_steps: int | None = None,
) -> list[TrainStep]:
    """Train ``policy`` with partition-based REINFORCE.

    ``partition_fn`` maps an instance to its partition; ``instances`` returns the
    batch of instances for a given step. Both are injected so the trainer stays
    decoupled from any specific partitioner or dataset. For reproducible resume,
    they must be deterministic functions of their inputs, or their independent
    state must be restored by the caller.

    ``cfg.steps`` is the original total optimization horizon. ``max_steps`` caps
    updates in this call without changing that horizon or the learning-rate
    schedule. A resumed call returns the full history, including prior updates.

    ``on_checkpoint`` receives an independent CPU snapshot after ``on_step`` at
    every ``checkpoint_every`` updates and at the call's final completed update.
    Its ``completed_step`` counts completed optimizer updates, so the next
    zero-based sampler step is that value. Snapshots contain model, optimizer,
    scheduler, sampling-generator and global PyTorch CPU/CUDA RNG states, plus
    training/decoding configuration and history. The callback should serialize
    without consuming training RNG streams or mutating the policy. Exact replay
    assumes identical runtime, hardware, callbacks, and external sampler state.
    """

    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.n_starts < 2:
        raise ValueError("n_starts must be at least 2 for a group baseline")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive")
    device = torch.device(cfg.device)
    policy.to(device).train()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    scheduler = None
    if cfg.lr_warmup_steps > 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: _lr_factor(s, cfg.lr_warmup_steps, cfg.steps)
        )
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    history: list[TrainStep] = []
    if resume_state is not None:
        history = _restore_training_state(
            resume_state, policy, optimizer, scheduler, generator, cfg
        )
    start = len(history)
    end = cfg.steps if max_steps is None else min(cfg.steps, start + max_steps)
    for step in range(start, end):
        batch = instances(step)
        if not batch:
            raise ValueError("instances() returned an empty batch")

        optimizer.zero_grad(set_to_none=True)
        partitions = [partition_fn(instance) for instance in batch]
        # One batched decode over all instances x samples; per-instance group-mean
        # baseline (POMO) computed across the sample dimension.
        rollout = policy.rollout_batched(
            batch, partitions, num_samples=cfg.n_starts, mode="sample", generator=generator
        )
        costs = rollout.costs  # (B, S)
        log_prob = rollout.log_prob  # (B, S)
        baseline = costs.mean(dim=1, keepdim=True)
        advantage = (costs - baseline).detach().to(log_prob.device)
        total_loss = (advantage * log_prob).mean()
        total_loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        record = TrainStep(
            step=step,
            loss=float(total_loss.detach()),
            mean_cost=float(costs.mean()),
            best_cost=float(costs.min()),
            baseline_cost=float(baseline.mean()),
        )
        history.append(record)
        if on_step is not None:
            on_step(record)
        if on_checkpoint is not None and ((step + 1) % checkpoint_every == 0 or step + 1 == end):
            on_checkpoint(_training_state(policy, optimizer, scheduler, generator, cfg, history))
    return history
