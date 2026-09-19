"""REINFORCE with a greedy-rollout baseline.

The algorithm accepts any core Env implementation, which supplies the encoder
features and decoder context."""

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from ..augment import augment_state, best_over_aug
from ..decode import log_prob
from ..distributed import DistEnv, all_reduce_grads, all_reduce_mean, broadcast_params
from ..env import Env, get_dynamic_decoder_context
from ..models.policy import ConstructivePolicy
from ..precision import Precision, PrecisionType
from ..state import State
from ..train_utils import WeightEMA, build_scheduler
from ._eval import EvalSupport


@dataclass(slots=True)
class REINFORCEConfig:
    batch_size: int = 256
    baseline_refresh_every: int = 1_000
    grad_clip: float = 1.0
    lr: float = 1e-4
    optimizer: Literal["adamw", "adam"] = "adamw"
    weight_decay: float = 0.01
    eval_batch_size: int = 512
    precision: PrecisionType = "fp32"
    eval_augment: int = 1  # 1 = none, 8 = dihedral x8 instance augmentation
    ema_decay: float = 0.0
    lr_warmup_steps: int = 0
    lr_total_steps: int = 0
    eval_seed: int = 12_345  # fixed held-out eval instances


class REINFORCE(EvalSupport, nn.Module):
    """REINFORCE-with-baseline trainer. Generic over `Env`."""

    def __init__(
        self,
        model: ConstructivePolicy,
        env: Env,
        cfg: REINFORCEConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.baseline = copy.deepcopy(model).to(device)
        for p in self.baseline.parameters():
            p.requires_grad_(False)
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(
                model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"unsupported optimizer: {cfg.optimizer!r}")
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
        self.baseline.eval()
        state = self.env.reset(self.cfg.batch_size, generator=rng, device=self.device)

        with self.precision.autocast():
            student_reward, sum_logp = _rollout(
                self.model, self.env, state, sample_actions=True, rng=rng
            )
            with torch.no_grad():
                baseline_reward, _ = _rollout(
                    self.baseline, self.env, state, sample_actions=False, rng=None
                )
            advantage = (student_reward - baseline_reward).detach()
            loss = -(advantage * sum_logp).mean()

        self.opt.zero_grad(set_to_none=True)
        self.precision.backward(loss)
        if self.cfg.grad_clip > 0:
            self.precision.unscale_(self.opt)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        all_reduce_grads(self.model, self.dist.world_size)
        self.precision.step(self.opt)
        self.precision.update()
        if self.sched is not None:
            self.sched.step()
        if self.ema is not None:
            self.ema.update()

        self._step += 1
        if self._step % self.cfg.baseline_refresh_every == 0:
            self._maybe_refresh_baseline(rng)

        return {
            "loss": loss.detach().item(),
            "reward_student": student_reward.mean().item(),
            "reward_baseline": baseline_reward.mean().item(),
            "advantage": advantage.mean().item(),
        }

    def eval_step(self, rng: torch.Generator) -> dict[str, float]:
        self.model.eval()
        n_aug = self.cfg.eval_augment
        b = self.cfg.eval_batch_size
        with self.eval_weights(), torch.no_grad():
            state = augment_state(self._get_eval_state(), n_aug)
            reward, _ = _rollout(self.model, self.env, state, sample_actions=False, rng=None)
            reward = best_over_aug(reward, n_aug, b)
        return {"eval_reward": reward.mean().item(), "eval_tour_length": (-reward).mean().item()}

    @torch.no_grad()
    def _maybe_refresh_baseline(self, rng: torch.Generator) -> None:
        self.model.eval()
        self.baseline.eval()
        state = self.env.reset(self.cfg.eval_batch_size, generator=rng, device=self.device)
        r_student, _ = _rollout(self.model, self.env, state, sample_actions=False, rng=None)
        r_baseline, _ = _rollout(self.baseline, self.env, state, sample_actions=False, rng=None)
        # Collective decision across ranks: average reward, then all-reduce
        # broadcasts no longer needed because all ranks reach same conclusion.
        s_mean = all_reduce_mean(r_student.mean(), self.dist.world_size)
        b_mean = all_reduce_mean(r_baseline.mean(), self.dist.world_size)
        if s_mean > b_mean:
            self.baseline.load_state_dict(self.model.state_dict())
            broadcast_params(self.baseline, src=0)


def _rollout(
    model: ConstructivePolicy,
    env: Env,
    state: State,
    *,
    sample_actions: bool,
    rng: torch.Generator | None,
) -> tuple[Float[Tensor, "b"], Float[Tensor, "b"]]:
    """Full rollout. Uses Env Protocol: works for TSP, CVRP, CVRPTW."""
    feats = env.build_features(state)
    node_embs, graph_emb = model.encode(feats)
    b = feats.shape[0]
    sum_logp = torch.zeros(b, device=feats.device)
    final_reward = torch.zeros_like(sum_logp)
    done_acc = torch.zeros(b, dtype=torch.bool, device=feats.device)
    max_steps = env.max_steps(state)
    for _ in range(max_steps):
        # Mask contributions from already-finished episodes. Variable-length
        # envs (CVRP/CVRPTW/OP/PDP/mTSP) finish at different steps and re-emit
        # reward on the absorbing depot state; without this mask the surplus
        # log-probs corrupt the gradient and reward is double-counted.
        active = (~done_acc).to(sum_logp.dtype)
        mask = env.action_mask(state)
        first_idx, current_idx = env.decoder_context(state)
        logits = model.decode_step(
            node_embs,
            graph_emb,
            first_idx,
            current_idx,
            mask,
            dynamic_context=get_dynamic_decoder_context(env, state),
        )
        action = _pick(logits, sample=sample_actions, rng=rng)
        sum_logp = sum_logp + log_prob(logits, action) * active
        state, reward, done = env.step(state, action)
        final_reward = final_reward + reward * active
        done_acc = done_acc | done
        if bool(done_acc.all()):
            break
    return final_reward, sum_logp


def _pick(
    logits: Float[Tensor, "b n"],
    *,
    sample: bool,
    rng: torch.Generator | None,
) -> Int[Tensor, "b"]:
    if not sample:
        return logits.argmax(dim=-1)
    probs = logits.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng).squeeze(-1)
