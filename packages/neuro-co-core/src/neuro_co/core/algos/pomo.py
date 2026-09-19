"""POMO REINFORCE (Kwon 2020). Env-agnostic.

Single algo class working on any `Env` Protocol implementation. Vary
the first decoded action per replica: works for TSP (start fixed at
city 0, varying second-visited city) and CVRP (depot fixed, varying
first customer). Group-mean baseline; no separate baseline network.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

from ..augment import augment_state, best_over_aug
from ..decode import log_prob
from ..distributed import DistEnv, all_reduce_grads
from ..env import Env, get_dynamic_decoder_context
from ..models.policy import ConstructivePolicy
from ..precision import Precision, PrecisionType
from ..state import State
from ..train_utils import WeightEMA, build_scheduler
from ._eval import EvalSupport
from .multistart import distinct_first_actions, pomo_advantage, replicate


@dataclass(slots=True)
class POMOConfig:
    batch_size: int = 64
    n_starts: int = 20
    lr: float = 1e-4
    optimizer: Literal["adamw", "adam"] = "adamw"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_batch_size: int = 256
    precision: PrecisionType = "fp32"
    eval_augment: int = 1  # 1 = none, 8 = dihedral x8 instance augmentation
    ema_decay: float = 0.0  # 0 = off; eval uses EMA weights when > 0
    lr_warmup_steps: int = 0
    lr_total_steps: int = 0  # > 0 enables cosine decay
    eval_seed: int = 12_345  # fixed held-out eval instances


class POMO(EvalSupport, nn.Module):
    """POMO-REINFORCE. Generic over `Env`."""

    def __init__(
        self,
        model: ConstructivePolicy,
        env: Env,
        cfg: POMOConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model.to(device)
        self.env = env
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.optimizer == "adamw":
            self.opt = torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(
                self.model.parameters(),
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
        with self.precision.autocast():
            reward, sum_logp = self._pomo_rollout(rng=rng, sample_actions=True)
            adv = pomo_advantage(reward, self.cfg.n_starts).detach()
            loss = -(adv * sum_logp).mean()

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
        return {
            "loss": loss.detach().item(),
            "reward_mean": reward.mean().item(),
            "reward_max_per_group": reward.view(-1, self.cfg.n_starts)
            .max(dim=1)
            .values.mean()
            .item(),
            "advantage_std": adv.std(unbiased=False).item(),
        }

    def eval_step(self, rng: torch.Generator) -> dict[str, float]:
        """Greedy eval on a fixed held-out set. `cfg.eval_augment=8` runs
        dihedral x8 augmentation and keeps the best tour per instance."""
        self.model.eval()
        n_aug = self.cfg.eval_augment
        b = self.cfg.eval_batch_size
        with self.eval_weights(), torch.no_grad():
            state = augment_state(self._get_eval_state(), n_aug)
            reward = _greedy_rollout(self.model, self.env, state)
            reward = best_over_aug(reward, n_aug, b)
        return {
            "eval_reward": reward.mean().item(),
            "eval_tour_length": (-reward).mean().item(),
        }

    def _pomo_rollout(
        self,
        rng: torch.Generator,
        sample_actions: bool,
    ) -> tuple[Float[Tensor, "bn"], Float[Tensor, "bn"]]:
        b = self.cfg.batch_size
        n_starts = self.cfg.n_starts
        state = self.env.reset(b, generator=rng, device=self.device)

        feats = self.env.build_features(state)
        node_embs, graph_emb = self.model.encode(feats)
        node_embs = node_embs.repeat_interleave(n_starts, dim=0)
        graph_emb = graph_emb.repeat_interleave(n_starts, dim=0)
        decoder_cache = self.model.precompute_decoder_cache(node_embs)

        first_mask = self.env.pomo_first_mask(state)
        # `distinct_first_actions` cycles through the permitted set when fewer
        # than n_starts are available (e.g. CVRPTW windows), so no hard cap on
        # n_starts vs runtime-valid. Only guard the degenerate empty mask.
        if int(first_mask.sum(dim=1).min().item()) == 0:
            raise ValueError("a problem has zero permitted first actions")
        first_actions = distinct_first_actions(first_mask, n_starts, generator=rng)

        state_exp = replicate(state, n_starts)
        bn = b * n_starts
        sum_logp = torch.zeros(bn, device=self.device)
        final_reward = torch.zeros_like(sum_logp)
        done_acc = torch.zeros(bn, dtype=torch.bool, device=self.device)

        # Step 0 is assigned by POMO, not sampled from the policy. Its
        # score-function contribution is therefore log(1) = 0.
        state_exp, reward, done = self.env.step(state_exp, first_actions)
        final_reward = final_reward + reward
        done_acc = done_acc | done

        # Remaining steps: mask finished episodes (variable-length envs
        # re-emit reward on the absorbing state; see reinforce._rollout).
        max_steps = self.env.max_steps(state_exp) - 1
        for _ in range(max_steps):
            active = (~done_acc).to(sum_logp.dtype)
            mask = self.env.action_mask(state_exp)
            first_idx, current_idx = self.env.decoder_context(state_exp)
            logits = self.model.decode_step(
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=get_dynamic_decoder_context(self.env, state_exp),
                decoder_cache=decoder_cache,
            )
            action = _pick(logits, sample=sample_actions, rng=rng)
            sum_logp = sum_logp + log_prob(logits, action) * active
            state_exp, reward, done = self.env.step(state_exp, action)
            final_reward = final_reward + reward * active
            done_acc = done_acc | done
            if bool(done_acc.all()):
                break
        return final_reward, sum_logp


def _greedy_rollout(
    model: ConstructivePolicy,
    env: Env,
    state: State,
) -> Float[Tensor, "b"]:
    feats = env.build_features(state)
    node_embs, graph_emb = model.encode(feats)
    b = feats.shape[0]
    final = torch.zeros(b, device=feats.device)
    done_acc = torch.zeros(b, dtype=torch.bool, device=feats.device)
    for _ in range(env.max_steps(state)):
        active = (~done_acc).to(final.dtype)
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
        action = logits.argmax(dim=-1)
        state, reward, done = env.step(state, action)
        final = final + reward * active
        done_acc = done_acc | done
        if bool(done_acc.all()):
            break
    return final


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
