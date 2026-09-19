"""Collect policy decisions and activations for downstream analysis.

rollout_trace retains the state, action, reward, and log probability at each
step. Attribution and probing packages share this decoding loop. Set grad=True
to make encoder features a leaf tensor and keep autograd enabled for gradient
attribution. Finished episodes contribute nothing after done. This module
records tensors; downstream packages interpret them."""

from dataclasses import dataclass, field

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .decode import log_prob
from .env import Env, get_dynamic_decoder_context
from .models.policy import ConstructivePolicy
from .state import State


@dataclass(slots=True)
class Step:
    """One decode step, batched."""

    mask: Bool[Tensor, "b a"]
    logits: Float[Tensor, "b a"]  # masked logits the decoder produced
    action: Int[Tensor, "b"]
    logp: Float[Tensor, "b"]  # log-prob of the taken action
    current: Int[Tensor, "b"]
    first: Int[Tensor, "b"]
    active: Bool[Tensor, "b"]  # False once the episode has finished


@dataclass(slots=True)
class Trace:
    """Full record of one policy rollout over a batch of instances."""

    features: Float[Tensor, "b n d_in"]  # encoder input (leaf when grad=True)
    node_embs: Float[Tensor, "b n d"]
    graph_emb: Float[Tensor, "b d"]
    steps: list[Step] = field(default_factory=list)
    reward: Float[Tensor, "b"] | None = None

    @property
    def actions(self) -> Int[Tensor, "b t"]:
        """Stack per-step actions into `(b, T)`."""
        return torch.stack([s.action for s in self.steps], dim=1)

    @property
    def logits(self) -> Float[Tensor, "b t a"]:
        return torch.stack([s.logits for s in self.steps], dim=1)

    @property
    def tour_length(self) -> Float[Tensor, "b"] | None:
        return None if self.reward is None else -self.reward


def rollout_trace(
    model: ConstructivePolicy,
    env: Env,
    state: State,
    *,
    decode: str = "greedy",
    rng: torch.Generator | None = None,
    grad: bool = False,
) -> Trace:
    """Run `model` on `env` from `state`, recording every decode step.

    Args:
        decode: "greedy" (argmax) or "sample" (categorical).
        rng: generator for "sample".
        grad: if True, the encoder input is a leaf with `requires_grad` and
            no `no_grad` guard is used: for gradient-based attribution.

    Returns a `Trace`. Reward is the env's terminal reward, post-done masked.
    """
    if decode not in ("greedy", "sample"):
        raise ValueError(f"decode must be 'greedy' or 'sample', got {decode!r}")

    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        features = env.build_features(state)
        if grad:
            features = features.detach().requires_grad_(True)
        node_embs, graph_emb = model.encode(features)
        decoder_cache = model.precompute_decoder_cache(node_embs)

        b = features.shape[0]
        reward = torch.zeros(b, device=features.device)
        done_acc = torch.zeros(b, dtype=torch.bool, device=features.device)
        trace = Trace(features=features, node_embs=node_embs, graph_emb=graph_emb)

        for _ in range(env.max_steps(state)):
            active = ~done_acc
            mask = env.action_mask(state)
            first_idx, current_idx = env.decoder_context(state)
            logits = model.decode_step(
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=get_dynamic_decoder_context(env, state),
                decoder_cache=decoder_cache,
            )
            if decode == "greedy":
                action = logits.argmax(dim=-1)
            else:
                action = torch.multinomial(logits.softmax(-1), 1, generator=rng).squeeze(-1)
            trace.steps.append(
                Step(
                    mask=mask,
                    logits=logits,
                    action=action,
                    logp=log_prob(logits, action),
                    current=current_idx,
                    first=first_idx,
                    active=active.clone(),
                )
            )
            state, r, done = env.step(state, action)
            reward = reward + r * active.to(reward.dtype)
            done_acc = done_acc | done
            if bool(done_acc.all()):
                break
        trace.reward = reward
    return trace


def layer_activations(encoder, features: Float[Tensor, "b n d_in"]) -> list[Float[Tensor, "b n d"]]:
    """Capture per-layer node embeddings from an encoder.

    For probing internal representations: returns the output of each
    encoder layer (plus the final pooled-free node embeddings). Works on
    any encoder exposing a `.blocks` / `.layers` ModuleList (AM, MatNet);
    falls back to `[final node_embs]` for opaque encoders (e.g. SSM).

    Uses forward hooks: no encoder modification needed. Runs under
    `no_grad`; for gradient-based probing capture activations yourself.
    """
    mods = getattr(encoder, "blocks", None) or getattr(encoder, "layers", None)
    acts: list[Tensor] = []
    if mods is None:
        with torch.no_grad():
            node_embs, _ = encoder(features)
        return [node_embs]

    handles = []
    for m in mods:
        handles.append(m.register_forward_hook(lambda _m, _i, out: acts.append(_first_tensor(out))))
    try:
        with torch.no_grad():
            encoder(features)
    finally:
        for h in handles:
            h.remove()
    return acts


def _first_tensor(out):
    """Unwrap a layer output (Tensor or tuple) to its leading Tensor."""
    while isinstance(out, tuple):
        out = out[0]
    return out


__all__ = ["Step", "Trace", "layer_activations", "rollout_trace"]
