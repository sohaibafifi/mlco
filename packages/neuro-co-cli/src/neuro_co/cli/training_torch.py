"""Torch POMO adapter for the shared training protocol."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.env import Env, get_dynamic_decoder_context
from neuro_co.core.models import AttentionModel
from neuro_co.core.state import State
from neuro_co.problems.cvrp.env import CVRPEnv, CVRPState
from neuro_co.problems.tsp.env import TSPEnv, TSPState

if TYPE_CHECKING:
    from .training import TrainingConfig


class TorchTrainer:
    checkpoint_suffix = ".pt"

    def __init__(self, config: TrainingConfig, weights: dict[str, np.ndarray], device: str) -> None:
        self.config = config
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        if config.problem == "tsp":
            self.env: Env = TSPEnv(size=config.size)
        elif config.problem == "cvrp":
            self.env = CVRPEnv(
                size=config.size, capacity=config.capacity, max_demand=config.max_demand
            )
        else:
            raise ValueError(f"Matched Torch training does not support {config.problem!r}")
        self.model = AttentionModel(
            in_dim=self.env.encoder_in_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=0.0,
        ).float()
        self.model.load_state_dict(
            {name: torch.from_numpy(np.asarray(value)) for name, value in weights.items()}
        )
        assert config.n_starts is not None
        self.algo = POMO(
            self.model,
            self.env,
            POMOConfig(
                batch_size=config.batch_size,
                n_starts=config.n_starts,
                lr=config.lr,
                optimizer=cast(Literal["adam", "adamw"], config.optimizer),
                weight_decay=config.weight_decay,
                grad_clip=config.grad_clip,
                eval_batch_size=config.eval_batch_size,
                precision="fp32",
            ),
            device=self.device,
        )
        self.rng = torch.Generator(device=self.device).manual_seed(config.seed)
        self.info = {
            "backend": "torch",
            "device": str(self.device),
            "platform": self.device.type,
            "framework_version": str(torch.__version__),
        }

    def _state(self, batch: dict[str, np.ndarray]) -> State:
        coords = torch.as_tensor(batch["coords"], dtype=torch.float32, device=self.device)
        expected_nodes = self.config.size + (self.config.problem == "cvrp")
        if coords.ndim != 3 or coords.shape[1:] != (expected_nodes, 2):
            raise ValueError(f"Expected coordinates shaped (batch, {expected_nodes}, 2)")
        b = coords.shape[0]
        visited = torch.zeros((b, expected_nodes), dtype=torch.bool, device=self.device)
        indices = torch.zeros(b, dtype=torch.long, device=self.device)
        length = torch.zeros(b, dtype=torch.float32, device=self.device)
        if self.config.problem == "tsp":
            visited[:, 0] = True
            return cast(Any, TSPState)(
                coords=coords,
                visited=visited,
                current=indices,
                first=indices.clone(),
                step_count=indices.clone(),
                tour_length=length,
            )
        demand = torch.as_tensor(batch["demand"], dtype=torch.float32, device=self.device)
        if demand.shape != coords.shape[:2]:
            raise ValueError("Demand must have one value per node, including the depot")
        return cast(Any, CVRPState)(
            coords=coords,
            demand=demand,
            visited=visited,
            current=indices,
            remaining_capacity=torch.full(
                (b,), self.config.capacity, dtype=torch.float32, device=self.device
            ),
            tour_length=length,
            step_count=indices.clone(),
        )

    def train_step(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        return self.algo.train_step(self.rng, state=self._state(batch))

    @torch.no_grad()
    def evaluate(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        self.model.eval()
        state = self._state(batch)
        features = self.env.build_features(state)
        node_embs, graph_emb = self.model.encode(features)
        cache = self.model.precompute_decoder_cache(node_embs)
        done_acc = torch.zeros(features.shape[0], dtype=torch.bool, device=self.device)
        reward_sum = torch.zeros(features.shape[0], dtype=torch.float32, device=self.device)
        for _ in range(self.env.max_steps(state)):
            active = ~done_acc
            mask = self.env.action_mask(state) | done_acc.unsqueeze(1)
            first, current = self.env.decoder_context(state)
            logits = self.model.decode_step(
                node_embs,
                graph_emb,
                first,
                current,
                mask,
                dynamic_context=get_dynamic_decoder_context(self.env, state),
                decoder_cache=cache,
            )
            action = logits.masked_fill(~mask, -torch.inf).argmax(dim=-1)
            state, reward, done = self.env.step(state, action.masked_fill(done_acc, 0))
            reward_sum += torch.where(active, reward, 0.0)
            done_acc |= done
            if bool(done_acc.all()):
                break
        if not bool(done_acc.all()):
            raise RuntimeError("Torch greedy evaluation exceeded the environment step bound")
        costs = -reward_sum.cpu().numpy()
        if not np.isfinite(costs).all():
            raise RuntimeError("Torch greedy evaluation produced non-finite costs")
        return costs

    def save(self, path: Path, step: int) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "arch": {
                    "backbone": "am",
                    "hidden_dim": self.config.hidden_dim,
                    "num_layers": self.config.num_layers,
                    "num_heads": self.config.num_heads,
                },
                "epoch": max(0, (step - 1) // self.config.steps_per_epoch),
                "step": step,
                "optimizer": self.algo.opt.state_dict(),
                "sampler_state": self.rng.get_state(),
            },
            path,
        )

    def load(self, path: Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.algo.opt.load_state_dict(checkpoint["optimizer"])
        if "sampler_state" in checkpoint:
            self.rng.set_state(checkpoint["sampler_state"].cpu())
        self.algo._step = checkpoint.get("step", 0)
        return int(self.algo._step)

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
