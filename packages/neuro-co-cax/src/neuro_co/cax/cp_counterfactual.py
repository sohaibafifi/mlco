"""CP-certified counterfactuals for neural CO policies.

Standard Wachter (2017) counterfactuals minimise `||delta||` such that
`argmax pi(x + delta) != argmax pi(x)`, with no guarantee the perturbed
instance stays *feasible*. This module instead returns only feasible
flipping counterfactuals:

    minimise   ||delta||_1
    subject to delta in [-epsilon, epsilon]
               instance(x + delta) is feasible        (arithmetic / CP)
               argmax pi(x + delta) != argmax pi(x)   (verified)

Family-directed sample + verify draws candidates for one declared constraint
family at a time. This avoids labelling a mixed perturbation by whichever
field happens to have the most coordinates. Distances are the mean absolute
perturbation divided by ``epsilon`` times the natural field scale, averaged
over fields. Coordinates, demands, time windows, prizes, and processing
times are therefore comparable despite different raw units and dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from neuro_co.attr.attribution._common import _decode_logp
from neuro_co.cax.feasibility import is_feasible

# Per-problem family -> instance state-fields a counterfactual may perturb.
FAMILY_PERTURB_FIELDS: dict[str, dict[str, tuple[str, ...]]] = {
    "cvrptw": {
        "capacity": ("demand",),
        "time_window": ("tw_early", "tw_late"),
        "spatial": ("coords",),
    },
    "vrptw": {
        "capacity": ("demand",),
        "time_window": ("tw_early", "tw_late"),
        "spatial": ("coords",),
    },
    "op": {
        "prize": ("prize",),
        "travel_budget": ("coords",),
    },
    "fjsp": {
        "processing": ("proc_times",),
        "eligibility": ("ops_ma_adj", "proc_times"),
        "precedence": ("op_in_job",),
    },
}


@dataclass
class CounterfactualReport:
    """Per-step counterfactual diagnostics.

    delta: `dict[state_field, Tensor]` of perturbations `[T, *field_shape]`.
    flipped: `[B, T]` bool indicating that a feasible flip was found.
    new_action: `[B, T]` long counterfactual argmax (orig where not flipped).
    delta_l1: `[B, T]` normalized distance of the chosen delta.
    family_flipped: `[B, T, K]` feasible flip found per targeted family.
    family_delta_l1: `[B, T, K]` normalized distance, zero when no flip.
    top_family: `[B, T]` minimum-distance family index, -1 when no flip.
    """

    delta: dict[str, torch.Tensor]
    flipped: torch.Tensor
    new_action: torch.Tensor
    delta_l1: torch.Tensor
    family_names: list[str]
    family_flipped: torch.Tensor
    family_delta_l1: torch.Tensor
    top_family: torch.Tensor
    epsilon: float
    method: str = "sample_verify"


def _field_zero_delta(reference: torch.Tensor) -> torch.Tensor:
    """Return a numeric zero delta even for boolean/integer state fields."""
    if reference.is_floating_point():
        return torch.zeros_like(reference)
    return torch.zeros_like(reference, dtype=torch.float32)


def _field_delta(edited: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Numeric edited-minus-reference delta for any state-field dtype."""
    if edited.is_floating_point() and reference.is_floating_point():
        return edited - reference
    return edited.float() - reference.float()


def cp_counterfactual(
    policy: Any,
    env: Any,
    state: Any,
    *,
    problem: str,
    epsilon: float = 0.1,
    max_shots: int = 32,
    sigma: float | None = None,
    max_steps: int | None = 8,
    seed: int = 0,
    feasibility_mode: str = "arithmetic",
    certificate_time_limit_s: float = 1.0,
    certificate_workers: int = 1,
) -> CounterfactualReport:
    """Sample-and-verify feasible counterfactuals per decoding step."""
    key = problem.lower()
    if key not in FAMILY_PERTURB_FIELDS:
        raise KeyError(f"no perturbable fields for problem={problem!r}")
    family_fields = FAMILY_PERTURB_FIELDS[key]
    family_names = list(family_fields)
    all_fields = tuple(dict.fromkeys(f for fields in family_fields.values() for f in fields))
    sig = epsilon / 3.0 if sigma is None else sigma

    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    b = state.coords.shape[0] if hasattr(state, "coords") else state.proc_times.shape[0]
    env_steps = env.max_steps(state)
    t_max = env_steps if max_steps is None else min(max_steps, env_steps)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    delta_store: dict[str, list[torch.Tensor]] = {f: [] for f in all_fields}
    flipped = torch.zeros(b, t_max, dtype=torch.bool)
    new_action = torch.zeros(b, t_max, dtype=torch.long)
    delta_l1 = torch.zeros(b, t_max, dtype=torch.float32)
    k_families = len(family_names)
    family_flipped = torch.zeros(b, t_max, k_families, dtype=torch.bool)
    family_delta_l1 = torch.zeros(b, t_max, k_families, dtype=torch.float32)
    top_family = torch.full((b, t_max), -1, dtype=torch.long)
    done_acc: torch.Tensor | None = None
    actual_t = 0

    with torch.no_grad():
        for t in range(t_max):
            actual_t = t + 1
            orig = _decode_logp(policy, env, state, env.build_features(state)).argmax(dim=-1)
            overall_l1 = torch.full((b,), float("inf"))
            overall_delta = {f: _field_zero_delta(getattr(state, f)) for f in all_fields}
            overall_new = orig.clone()
            overall_family = torch.full((b,), -1, dtype=torch.long)

            for family_idx, family_name in enumerate(family_names):
                fields = family_fields[family_name]
                best_l1 = torch.full((b,), float("inf"))
                best_values = {f: getattr(state, f).clone() for f in fields}
                best_new = orig.clone()

                for _ in range(max_shots):
                    cand, deltas = _sample_candidate(
                        state,
                        fields,
                        key=key,
                        epsilon=epsilon,
                        sigma=sig,
                        generator=gen,
                        env=env,
                        family=family_name,
                    )
                    # CP-SAT is a stage-2 certificate on the winner. The bulk
                    # search uses the cheap necessary arithmetic filter.
                    feas = is_feasible(cand, env, problem, mode="arithmetic")
                    act = _decode_logp(policy, env, cand, env.build_features(cand)).argmax(dim=-1)
                    flip = (act != orig).cpu()
                    distance = torch.stack(
                        [
                            d.abs().flatten(1).mean(dim=-1).cpu()
                            / max(
                                epsilon * _field_scale(state, field, key=key, env=env),
                                1e-12,
                            )
                            for field, d in deltas.items()
                        ],
                        dim=-1,
                    ).mean(dim=-1)
                    take = feas.cpu() & flip & (distance < best_l1)
                    if take.any():
                        for bi in take.nonzero(as_tuple=False).flatten().tolist():
                            best_l1[bi] = distance[bi]
                            best_new[bi] = act[bi]
                            for field in fields:
                                best_values[field][bi] = getattr(cand, field)[bi]

                found = best_l1.isfinite()
                if feasibility_mode == "cp_sat" and found.any():
                    certified = is_feasible(
                        state.replace(**best_values),
                        env,
                        problem,
                        mode="cp_sat",
                        time_limit_s=certificate_time_limit_s,
                        max_workers=certificate_workers,
                    )
                    found = found & certified.cpu()

                family_flipped[:, t, family_idx] = found
                family_delta_l1[:, t, family_idx] = torch.where(
                    found,
                    best_l1,
                    torch.zeros_like(best_l1),
                )
                take_family = found & (best_l1 < overall_l1)
                if take_family.any():
                    for bi in take_family.nonzero(as_tuple=False).flatten().tolist():
                        overall_l1[bi] = best_l1[bi]
                        overall_new[bi] = best_new[bi]
                        overall_family[bi] = family_idx
                        for field in all_fields:
                            overall_delta[field][bi].zero_()
                        for field in fields:
                            overall_delta[field][bi] = _field_delta(
                                best_values[field][bi],
                                getattr(state, field)[bi],
                            )

            for field in all_fields:
                delta_store[field].append(overall_delta[field].cpu())
            found = overall_l1.isfinite()
            flipped[:, t] = found
            delta_l1[:, t] = torch.where(found, overall_l1, torch.zeros_like(overall_l1))
            new_action[:, t] = torch.where(found, overall_new.cpu(), orig.cpu())
            top_family[:, t] = overall_family

            state, _, done = env.step(state, orig)
            done_acc = done if done_acc is None else (done_acc | done)
            if done_acc is not None and bool(done_acc.all()):
                break

    delta_final = {f: torch.stack(v, dim=0) for f, v in delta_store.items()}
    return CounterfactualReport(
        delta=delta_final,
        flipped=flipped[:, :actual_t],
        new_action=new_action[:, :actual_t],
        delta_l1=delta_l1[:, :actual_t],
        family_names=family_names,
        family_flipped=family_flipped[:, :actual_t],
        family_delta_l1=family_delta_l1[:, :actual_t],
        top_family=top_family[:, :actual_t],
        epsilon=float(epsilon),
        method=f"family_relative_sample_verify_{feasibility_mode}",
    )


def _sample_candidate(
    state: Any,
    fields: tuple[str, ...],
    *,
    key: str,
    epsilon: float,
    sigma: float,
    generator: torch.Generator,
    env: Any,
    family: str | None = None,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Sample, project to the instance domain, and return actual deltas."""
    if key == "fjsp" and family == "eligibility":
        return _sample_fjsp_eligibility(
            state,
            epsilon=epsilon,
            generator=generator,
            env=env,
        )
    if key == "fjsp" and family == "precedence":
        return _sample_fjsp_precedence(
            state,
            epsilon=epsilon,
            generator=generator,
            env=env,
        )
    edits: dict[str, torch.Tensor] = {}
    for field in fields:
        ref = getattr(state, field)
        scale = _field_scale(state, field, key=key, env=env)
        noise = (
            torch.empty_like(ref.float().cpu())
            .normal_(0.0, sigma * scale, generator=generator)
            .clamp_(-epsilon * scale, epsilon * scale)
            .to(ref.device)
        )
        # A decoding-state intervention must not rewrite the prefix that has
        # already been executed. For routing states, preserve the depot, the
        # current node, and every visited node. Otherwise coordinates or
        # attributes from the past are changed while current_time,
        # remaining_capacity, or length_used still describe the old prefix.
        if hasattr(state, "visited") and ref.ndim >= 2 and ref.shape[1] == state.visited.shape[1]:
            mutable = (~state.visited).clone()
            mutable[:, 0] = False
            if hasattr(state, "current"):
                mutable.scatter_(1, state.current.unsqueeze(1), False)
            while mutable.ndim < noise.ndim:
                mutable = mutable.unsqueeze(-1)
            noise = noise * mutable.to(noise.dtype)
        elif (
            key == "fjsp"
            and hasattr(state, "op_done")
            and ref.ndim >= 2
            and ref.shape[1] == state.op_done.shape[1]
        ):
            mutable = ~state.op_done
            while mutable.ndim < noise.ndim:
                mutable = mutable.unsqueeze(-1)
            noise = noise * mutable.to(noise.dtype)
        edits[field] = (ref.float() + noise).to(ref.dtype)

    if "coords" in edits:
        edits["coords"] = edits["coords"].clamp(0.0, 1.0)
        edits["coords"][:, 0] = state.coords[:, 0]
    if key in ("cvrptw", "vrptw"):
        if "demand" in edits:
            edits["demand"] = edits["demand"].clamp(
                0.0,
                float(getattr(env, "max_demand", edits["demand"].max().item())),
            )
            edits["demand"][:, 0] = state.demand[:, 0]
        if "tw_early" in edits and "tw_late" in edits:
            early = torch.minimum(edits["tw_early"], edits["tw_late"])
            late = torch.maximum(edits["tw_early"], edits["tw_late"])
            horizon = float(getattr(env, "horizon", late.max().item()))
            edits["tw_early"] = early.clamp(0.0, horizon)
            edits["tw_late"] = late.clamp(0.0, horizon)
            edits["tw_early"][:, 0] = state.tw_early[:, 0]
            edits["tw_late"][:, 0] = state.tw_late[:, 0]
    elif key == "op" and "prize" in edits:
        edits["prize"] = edits["prize"].clamp_min(0.0)
        edits["prize"][:, 0] = state.prize[:, 0]
    elif key == "fjsp" and "proc_times" in edits:
        max_proc = float(getattr(env, "max_proc", edits["proc_times"].max().item()))
        positive = edits["proc_times"].clamp(1.0, max_proc)
        edits["proc_times"] = torch.where(
            state.ops_ma_adj,
            positive,
            torch.zeros_like(edits["proc_times"]),
        )

    deltas = {field: edits[field] - getattr(state, field) for field in fields}
    return state.replace(**edits), deltas


def _sample_fjsp_eligibility(
    state: Any,
    *,
    epsilon: float,
    generator: torch.Generator,
    env: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Toggle one future operation-machine edge while keeping every op valid."""
    del epsilon
    old_adj = state.ops_ma_adj
    old_adj_cpu = old_adj.detach().cpu()
    active_cpu = (~state.op_done).detach().cpu()
    toggle_cpu = torch.zeros_like(old_adj_cpu)
    for batch_idx in range(old_adj_cpu.shape[0]):
        eligible_count = old_adj_cpu[batch_idx].sum(dim=-1)
        removable = old_adj_cpu[batch_idx] & (eligible_count > 1).unsqueeze(-1)
        addable = ~old_adj_cpu[batch_idx]
        editable = active_cpu[batch_idx].unsqueeze(-1) & (removable | addable)
        choices = editable.nonzero(as_tuple=False)
        if choices.numel() == 0:
            continue
        choice_idx = int(torch.randint(choices.shape[0], (), generator=generator).item())
        operation, machine = choices[choice_idx].tolist()
        toggle_cpu[batch_idx, operation, machine] = True
    toggle = toggle_cpu.to(old_adj.device)
    new_adj = old_adj ^ toggle

    eligible_count = old_adj.sum(dim=-1).clamp_min(1)
    mean_duration = (
        (state.proc_times.sum(dim=-1) / eligible_count)
        .round()
        .clamp(1.0, float(getattr(env, "max_proc", 9.0)))
    )
    expanded_mean = mean_duration.unsqueeze(-1).expand_as(state.proc_times)
    new_proc = torch.where(
        new_adj,
        torch.where(old_adj, state.proc_times, expanded_mean),
        torch.zeros_like(state.proc_times),
    )
    candidate = state.replace(ops_ma_adj=new_adj, proc_times=new_proc)
    return candidate, {"ops_ma_adj": _field_delta(new_adj, old_adj)}


def _sample_fjsp_precedence(
    state: Any,
    *,
    epsilon: float,
    generator: torch.Generator,
    env: Any,
) -> tuple[Any, dict[str, torch.Tensor]]:
    """Swap one future operation pair without rewriting the executed prefix."""
    del env, epsilon
    order_cpu = state.op_in_job.detach().cpu().clone()
    job_cpu = state.job_id.detach().cpu()
    active_cpu = (~state.op_done).detach().cpu()

    for batch_idx in range(order_cpu.shape[0]):
        candidates = []
        for job in job_cpu[batch_idx].unique(sorted=True).tolist():
            operations = (
                ((job_cpu[batch_idx] == int(job)) & active_cpu[batch_idx])
                .nonzero(as_tuple=False)
                .flatten()
            )
            if operations.numel() >= 2:
                candidates.append(operations)
        if not candidates:
            continue
        chosen = int(torch.randint(len(candidates), (), generator=generator).item())
        operations = candidates[chosen]
        pair = operations[torch.randperm(operations.numel(), generator=generator)[:2]]
        left, right = int(pair[0]), int(pair[1])
        saved = order_cpu[batch_idx, left].clone()
        order_cpu[batch_idx, left] = order_cpu[batch_idx, right]
        order_cpu[batch_idx, right] = saved

    new_order = order_cpu.to(state.op_in_job.device)
    candidate = state.replace(op_in_job=new_order)
    return candidate, {"op_in_job": _field_delta(new_order, state.op_in_job)}


def _field_scale(state: Any, field: str, *, key: str, env: Any) -> float:
    """Natural raw-unit scale used for relative perturbation budgets."""
    if field in ("coords", "prize"):
        return 1.0
    if field == "demand":
        return float(getattr(env, "max_demand", 1.0))
    if field in ("tw_early", "tw_late"):
        return float(getattr(env, "horizon", 1.0))
    if key == "fjsp" and field == "proc_times":
        return float(getattr(env, "max_proc", 1.0))
    ref = getattr(state, field).float()
    return max(float((ref.max() - ref.min()).item()), 1.0)
