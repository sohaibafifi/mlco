"""Mask-aware constraint-family intervention attribution.

Feature gradients cannot observe a constraint whose effect enters the policy
through a discrete action mask. This module estimates family sensitivity by
intervening on one constraint family at a time, retaining only feasible
states, and measuring the full action-distribution response, including mask
changes.

For each family and decoding cell, the score is:

1. ``1 + 1 / d`` when a sampled feasible intervention flips the argmax,
   where ``d`` is the smallest normalized intervention distance found;
2. mean total-variation distance from the original policy distribution when
   no sampled flip is found.

The first branch always outranks the second because total variation is at
most one. Distances are normalized using the same natural field scales as the
high-budget counterfactual oracle, but samples and random seeds are separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.attr.attribution._common import _decode_logp
from neuro_co.cax.cp_counterfactual import (
    FAMILY_PERTURB_FIELDS,
    _field_scale,
    _sample_candidate,
)
from neuro_co.cax.feasibility import is_feasible


@dataclass
class ConstraintInterventionAttribution:
    """Per-cell constraint-family intervention diagnostics."""

    family_names: list[str]
    scores: torch.Tensor
    min_flip_distance: torch.Tensor
    mean_total_variation: torch.Tensor
    feasible_rate: torch.Tensor
    mask_change_rate: torch.Tensor
    actions: torch.Tensor
    epsilon: float
    num_samples: int
    seed: int
    mask_mode: str

    def top_family_per_step(self) -> torch.Tensor:
        """Return the highest-scoring family index for every cell."""
        return self.scores.argmax(dim=-1)


def constraint_intervention_attribution(
    policy: Any,
    env: Any,
    state: Any,
    *,
    problem: str,
    epsilon: float = 0.2,
    num_samples: int = 32,
    max_steps: int | None = 8,
    seed: int = 9100,
    feasibility_mode: str = "arithmetic",
    mask_mode: str = "recomputed",
) -> ConstraintInterventionAttribution:
    """Estimate mask-aware sensitivity to each constraint family.

    This is an explainer, not the evaluation oracle. It uses its own random
    stream, a caller-controlled sample budget, and no post-hoc CP certificate
    by default. The evaluation oracle can therefore use more samples and
    CP-SAT without sharing candidates with this method.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if feasibility_mode not in {"arithmetic", "cp_sat"}:
        raise ValueError("feasibility_mode must be 'arithmetic' or 'cp_sat'")
    if mask_mode not in {"recomputed", "fixed"}:
        raise ValueError("mask_mode must be 'recomputed' or 'fixed'")

    key = problem.lower()
    if key not in FAMILY_PERTURB_FIELDS:
        raise KeyError(f"no perturbable fields for problem={problem!r}")
    family_fields = FAMILY_PERTURB_FIELDS[key]
    family_names = list(family_fields)

    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    batch_size = int(env.build_features(state).shape[0])
    env_steps = int(env.max_steps(state))
    step_cap = env_steps if max_steps is None else min(int(max_steps), env_steps)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    score_steps: list[torch.Tensor] = []
    distance_steps: list[torch.Tensor] = []
    variation_steps: list[torch.Tensor] = []
    feasible_steps: list[torch.Tensor] = []
    mask_change_steps: list[torch.Tensor] = []
    action_steps: list[torch.Tensor] = []
    done_acc: torch.Tensor | None = None

    with torch.no_grad():
        for _ in range(step_cap):
            original_logp = _decode_logp(policy, env, state, env.build_features(state))
            original_prob = original_logp.exp()
            original_action = original_logp.argmax(dim=-1)
            original_mask = env.action_mask(state)

            family_scores: list[torch.Tensor] = []
            family_distances: list[torch.Tensor] = []
            family_variations: list[torch.Tensor] = []
            family_feasible_rates: list[torch.Tensor] = []
            family_mask_change_rates: list[torch.Tensor] = []

            for family_name, fields in family_fields.items():
                min_distance = torch.full((batch_size,), float("inf"))
                variation_sum = torch.zeros(batch_size)
                feasible_count = torch.zeros(batch_size)
                mask_change_count = torch.zeros(batch_size)

                for _sample in range(num_samples):
                    candidate, deltas = _sample_candidate(
                        state,
                        fields,
                        key=key,
                        epsilon=epsilon,
                        sigma=epsilon / 3.0,
                        generator=generator,
                        env=env,
                        family=family_name,
                    )
                    feasible = is_feasible(
                        candidate,
                        env,
                        problem,
                        mode=feasibility_mode,
                    ).cpu()
                    candidate_features = env.build_features(candidate)
                    candidate_mask = env.action_mask(candidate)
                    if mask_mode == "recomputed":
                        candidate_logp = _decode_logp(
                            policy,
                            env,
                            candidate,
                            candidate_features,
                        )
                    else:
                        node_embs, graph_emb = policy.encode(candidate_features)
                        first_idx, current_idx = env.decoder_context(candidate)
                        candidate_logits = policy.decode_step(
                            node_embs,
                            graph_emb,
                            first_idx,
                            current_idx,
                            original_mask,
                        )
                        candidate_logp = torch.log_softmax(candidate_logits, dim=-1)
                    variation = (
                        0.5 * (candidate_logp.exp() - original_prob).abs().sum(dim=-1)
                    ).cpu()
                    flipped = (candidate_logp.argmax(dim=-1) != original_action).cpu()

                    field_distances = [
                        delta.abs().flatten(1).mean(dim=-1).cpu()
                        / max(
                            epsilon * _field_scale(state, field, key=key, env=env),
                            1e-12,
                        )
                        for field, delta in deltas.items()
                    ]
                    distance = torch.stack(field_distances, dim=-1).mean(dim=-1)
                    accepted_flip = feasible & flipped
                    min_distance = torch.where(
                        accepted_flip & (distance < min_distance),
                        distance,
                        min_distance,
                    )
                    variation_sum += torch.where(feasible, variation, 0.0)
                    feasible_count += feasible.float()
                    mask_changed = (candidate_mask != original_mask).any(dim=-1).cpu()
                    mask_change_count += (feasible & mask_changed).float()

                mean_variation = variation_sum / feasible_count.clamp_min(1.0)
                score = torch.where(
                    torch.isfinite(min_distance),
                    1.0 + 1.0 / (min_distance + 1e-4),
                    mean_variation,
                )
                family_scores.append(score)
                family_distances.append(min_distance)
                family_variations.append(mean_variation)
                family_feasible_rates.append(feasible_count / float(num_samples))
                family_mask_change_rates.append(mask_change_count / feasible_count.clamp_min(1.0))

            score_steps.append(torch.stack(family_scores, dim=-1))
            distance_steps.append(torch.stack(family_distances, dim=-1))
            variation_steps.append(torch.stack(family_variations, dim=-1))
            feasible_steps.append(torch.stack(family_feasible_rates, dim=-1))
            mask_change_steps.append(torch.stack(family_mask_change_rates, dim=-1))
            action_steps.append(original_action.cpu())

            state, _, done = env.step(state, original_action.detach())
            done_acc = done if done_acc is None else (done_acc | done)
            if bool(done_acc.all()):
                break

    return ConstraintInterventionAttribution(
        family_names=family_names,
        scores=torch.stack(score_steps, dim=1),
        min_flip_distance=torch.stack(distance_steps, dim=1),
        mean_total_variation=torch.stack(variation_steps, dim=1),
        feasible_rate=torch.stack(feasible_steps, dim=1),
        mask_change_rate=torch.stack(mask_change_steps, dim=1),
        actions=torch.stack(action_steps, dim=1),
        epsilon=float(epsilon),
        num_samples=int(num_samples),
        seed=int(seed),
        mask_mode=mask_mode,
    )
