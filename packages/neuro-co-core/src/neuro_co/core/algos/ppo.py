"""PPO for sequential CO problems. Env-agnostic.

Single class working on any `Env` Protocol implementation. Clipped
surrogate + value head + entropy bonus. Variable-length rollouts handled
via `action_valid` mask (envs that terminate early stop incurring loss).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

from ..decode import log_prob
from ..distributed import DistEnv, all_reduce_grads
from ..env import Env, get_dynamic_decoder_context
from ..models.policy import ConstructivePolicy
from ..precision import Precision, PrecisionType
from ..state import State
from ..train_utils import WeightEMA, build_scheduler
from ._eval import EvalSupport


@dataclass(slots=True)
class PPOConfig:
    batch_size: int = 256
    epochs_per_rollout: int = 4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 1.0
    lr: float = 1e-4
    eval_batch_size: int = 512
    normalize_advantage: bool = True
    hidden_dim: int = 128
    precision: PrecisionType = "fp32"
    ema_decay: float = 0.0
    lr_warmup_steps: int = 0
    lr_total_steps: int = 0
    eval_augment: int = 1
    eval_seed: int = 12_345


class ValueHead(nn.Module):
    """Maps graph embedding (b, d) -> scalar value (b,)."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, graph_emb: Float[Tensor, "b d"]) -> Float[Tensor, "b"]:
        return self.net(graph_emb).squeeze(-1)


class PPO(EvalSupport, nn.Module):
    """PPO trainer. Generic over `Env`."""

    def __init__(
        self,
        model: ConstructivePolicy,
        env: Env,
        cfg: PPOConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.value = ValueHead(cfg.hidden_dim).to(device)
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        self.opt = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value.parameters()), lr=cfg.lr
        )
        dev_type = self.device.type if self.device.type != "mps" else "cpu"
        self.precision = Precision(cfg.precision, device=dev_type)
        self.dist = DistEnv.from_env()
        self.ema = WeightEMA(self.model, cfg.ema_decay) if cfg.ema_decay > 0 else None
        self.sched = build_scheduler(
            self.opt, warmup_steps=cfg.lr_warmup_steps, total_steps=cfg.lr_total_steps
        )
        self._step = 0

    def train_step(self, rng: torch.Generator) -> dict[str, float]:
        self.model.train()
        self.value.train()
        state = self.env.reset(self.cfg.batch_size, generator=rng, device=self.device)

        with torch.no_grad():
            traj = self._rollout(state, rng=rng)

        metrics: dict[str, float] = {}
        for _ in range(self.cfg.epochs_per_rollout):
            metrics = self._ppo_update(state, traj)

        if self.sched is not None:
            self.sched.step()
        if self.ema is not None:
            self.ema.update()

        self._step += 1
        metrics["reward"] = traj["reward"].mean().item()
        return metrics

    def eval_step(self, rng: torch.Generator) -> dict[str, float]:
        self.model.eval()
        with self.eval_weights(), torch.no_grad():
            traj = self._rollout(self._get_eval_state(), rng=None)
        return {
            "eval_reward": traj["reward"].mean().item(),
            "eval_tour_length": (-traj["reward"]).mean().item(),
        }

    @torch.no_grad()
    def _rollout(self, state: State, rng: torch.Generator | None) -> dict[str, Tensor]:
        feats = self.env.build_features(state)
        node_embs, graph_emb = self.model.encode(feats)
        baseline_value = self.value(graph_emb).detach()
        b = feats.shape[0]
        max_steps = self.env.max_steps(state)

        actions = torch.zeros(b, max_steps, dtype=torch.long, device=feats.device)
        old_logp = torch.zeros(b, max_steps, device=feats.device)
        action_valid = torch.zeros(b, max_steps, dtype=torch.bool, device=feats.device)

        cur = state
        done_acc = torch.zeros(b, dtype=torch.bool, device=feats.device)
        reward_acc = torch.zeros(b, device=feats.device)
        for t in range(max_steps):
            mask = self.env.action_mask(cur)
            first_idx, current_idx = self.env.decoder_context(cur)
            logits = self.model.decode_step(
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=get_dynamic_decoder_context(self.env, cur),
            )
            if rng is not None:
                probs = logits.softmax(dim=-1)
                action = torch.multinomial(probs, 1, generator=rng).squeeze(-1)
            else:
                action = logits.argmax(dim=-1)
            active = (~done_acc).to(reward_acc.dtype)
            old_logp[:, t] = log_prob(logits, action)
            actions[:, t] = action
            action_valid[:, t] = ~done_acc
            cur, r, done = self.env.step(cur, action)
            # Mask reward from finished episodes: variable-length envs re-emit
            # reward on the absorbing state, which would double-count returns.
            reward_acc = reward_acc + r * active
            done_acc = done_acc | done
            if bool(done_acc.all()):
                break

        return {
            "actions": actions,
            "old_logp": old_logp,
            "action_valid": action_valid,
            "reward": reward_acc,
            "baseline_value": baseline_value,
        }

    def _ppo_update(self, state: State, traj: dict[str, Tensor]) -> dict[str, float]:
        with self.precision.autocast():
            return self._ppo_update_inner(state, traj)

    def _ppo_update_inner(self, state: State, traj: dict[str, Tensor]) -> dict[str, float]:
        feats = self.env.build_features(state)
        node_embs, graph_emb = self.model.encode(feats)
        value = self.value(graph_emb)
        b = feats.shape[0]

        new_logp_sum = torch.zeros(b, device=feats.device)
        old_logp_sum = (traj["old_logp"] * traj["action_valid"].float()).sum(dim=1)
        entropy_total = torch.zeros_like(new_logp_sum)
        valid_counts = traj["action_valid"].sum(dim=1).clamp_min(1)

        cur = state
        done_acc = torch.zeros(b, dtype=torch.bool, device=feats.device)
        max_steps = self.env.max_steps(state)
        for t in range(max_steps):
            mask = self.env.action_mask(cur)
            first_idx, current_idx = self.env.decoder_context(cur)
            logits = self.model.decode_step(
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=get_dynamic_decoder_context(self.env, cur),
            )
            action = traj["actions"][:, t]
            valid = traj["action_valid"][:, t].float()
            new_logp_sum = new_logp_sum + log_prob(logits, action) * valid
            logp_full = F.log_softmax(logits, dim=-1)
            p_full = logp_full.exp()
            entropy_total = entropy_total - (p_full * logp_full).sum(dim=-1) * valid
            cur, _, done = self.env.step(cur, action)
            done_acc = done_acc | done
            if done_acc.all():
                break

        reward = traj["reward"].detach()
        advantage = reward - traj["baseline_value"]
        if self.cfg.normalize_advantage and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)

        ratio = (new_logp_sum - old_logp_sum).exp()
        unclipped = ratio * advantage
        clipped = ratio.clamp(1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio) * advantage
        policy_loss = -torch.minimum(unclipped, clipped).mean()
        value_loss = F.mse_loss(value, reward)
        entropy = (entropy_total / valid_counts).mean()
        loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

        self.opt.zero_grad(set_to_none=True)
        self.precision.backward(loss)
        if self.cfg.grad_clip > 0:
            self.precision.unscale_(self.opt)
            nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value.parameters()), self.cfg.grad_clip
            )
        all_reduce_grads(self.model, self.dist.world_size)
        all_reduce_grads(self.value, self.dist.world_size)
        self.precision.step(self.opt)
        self.precision.update()

        return {
            "loss": loss.detach().item(),
            "policy_loss": policy_loss.detach().item(),
            "value_loss": value_loss.detach().item(),
            "entropy": entropy.detach().item(),
            "ratio_mean": ratio.detach().mean().item(),
        }
