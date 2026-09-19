"""Constraint attribution and sufficient subsets on core states."""

import torch

from neuro_co.attr.attribution import gradient_attribution
from neuro_co.cax.constraint_map import aggregate_trace_by_family, get_constraints
from neuro_co.cax.cp_minimal_subset import cp_minimal_subset, pac_sample_count
from neuro_co.cax.lambda_attribution import LambdaAttribution, lambda_attribution
from neuro_co.core.models import AttentionModel
from neuro_co.problems.op.env import OPEnv
from neuro_co.problems.vrptw.env import CVRPTWEnv


def _setup(env, batch: int = 8):
    torch.manual_seed(0)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    state = env.reset(batch, generator=torch.Generator().manual_seed(0))
    return model, state


def test_constraint_map_columns() -> None:
    fams = get_constraints("cvrptw")
    names = [n for n, _ in fams]
    assert names == ["capacity", "time_window", "spatial"]
    assert dict(fams)["time_window"] == (3, 4, 5)


def test_lambda_attribution_cvrptw() -> None:
    env = CVRPTWEnv(size=6, capacity=20.0, horizon=10.0, window_width=2.0)
    model, state = _setup(env)
    la = lambda_attribution(model, env, state, problem="cvrptw")
    assert isinstance(la, LambdaAttribution)
    assert la.constraint_names == ["capacity", "time_window", "spatial"]
    assert la.scores.shape[0] == 8 and la.scores.shape[2] == 3
    assert la.per_family_node_scores.shape[0] == 3
    assert torch.isfinite(la.scores).all()
    assert (la.scores >= 0).all()  # |grad x feat| is non-negative
    assert la.actions.shape == la.scores.shape[:2]
    flat = la.to_attribution_trace(top_k=3)
    assert flat.node_scores.shape[:2] == la.scores.shape[:2]


def test_generic_trace_family_aggregation() -> None:
    env = CVRPTWEnv(size=6, capacity=20.0, horizon=10.0, window_width=2.0)
    model, state = _setup(env)
    trace = gradient_attribution(model, env, state, top_k=3, max_steps=2)
    family_scores = aggregate_trace_by_family(trace, "cvrptw")
    assert family_scores.shape == (8, 2, 3)
    assert torch.isfinite(family_scores).all()


def test_lambda_attribution_weighted() -> None:
    env = OPEnv(size=6, budget=4.0)
    model, state = _setup(env)
    la = lambda_attribution(
        model,
        env,
        state,
        problem="op",
        multipliers={"prize": 2.0, "travel_budget": 1.0},
    )
    assert la.multipliers is not None
    assert la.constraint_names == ["prize", "travel_budget"]
    assert la.multipliers.shape == (8, 2)


def test_multiplier_normalization_is_mean_one() -> None:
    from neuro_co.cax.duals import normalize_multipliers

    weights = normalize_multipliers(
        {"capacity": 2.0, "time_window": 0.0, "spatial": 1.0},
        ["capacity", "time_window", "spatial"],
        floor=0.1,
    )
    assert abs(sum(weights.values()) / 3.0 - 1.0) < 1e-6
    assert min(weights.values()) >= 0.1


def test_pac_sample_count() -> None:
    m = pac_sample_count(0.1, 0.1, n_tests=5)
    assert m > 0 and isinstance(m, int)


def test_cp_minimal_subset_runs() -> None:
    env = CVRPTWEnv(size=5, capacity=20.0, horizon=10.0, window_width=2.0)
    model, state = _setup(env, batch=4)
    rep = cp_minimal_subset(model, env, state, pac_epsilon=0.2, pac_delta=0.2, max_k=3, max_steps=3)
    assert rep.subset_size.shape[0] == 4
    assert rep.samples_drawn > 0
    assert rep.subset.dtype == torch.bool
    assert (rep.lower_confidence_bound <= rep.preserved_rate).all()
    assert rep.method == "greedy_hoeffding_lower_bound"


def test_feasibility_arithmetic() -> None:
    from neuro_co.cax.feasibility import is_feasible

    env = CVRPTWEnv(size=5, capacity=20.0, horizon=10.0, window_width=2.0)
    state = env.reset(8, generator=torch.Generator().manual_seed(0))
    ok = is_feasible(state, env, "cvrptw", mode="arithmetic")
    assert ok.shape == (8,)
    assert ok.dtype == torch.bool
    assert ok.all()  # freshly generated instances are feasible


def test_feasibility_op_and_fjsp() -> None:
    from neuro_co.cax.feasibility import is_feasible
    from neuro_co.problems.fjsp.env import FJSPEnv

    op_env = OPEnv(size=6, budget=4.0)
    op_state = op_env.reset(4, generator=torch.Generator().manual_seed(0))
    assert is_feasible(op_state, op_env, "op").all()
    assert is_feasible(
        op_state,
        op_env,
        "op",
        mode="cp_sat",
        time_limit_s=1.0,
        max_workers=2,
    ).all()

    fj_env = FJSPEnv(size=3, ops_per_job=2, num_machines=4)
    fj_state = fj_env.reset(4, generator=torch.Generator().manual_seed(0))
    assert is_feasible(fj_state, fj_env, "fjsp").all()


def test_cp_counterfactual_runs() -> None:
    from neuro_co.cax.cp_counterfactual import CounterfactualReport, cp_counterfactual

    env = CVRPTWEnv(size=5, capacity=20.0, horizon=10.0, window_width=2.0)
    model, state = _setup(env, batch=4)
    rep = cp_counterfactual(
        model, env, state, problem="cvrptw", epsilon=0.2, max_shots=8, max_steps=3
    )
    assert isinstance(rep, CounterfactualReport)
    assert rep.flipped.shape[0] == 4
    assert set(rep.delta) == {"coords", "demand", "tw_early", "tw_late"}
    assert rep.family_names == ["capacity", "time_window", "spatial"]
    assert rep.family_flipped.shape == (4, 3, 3)
    assert rep.family_delta_l1.shape == (4, 3, 3)
    assert rep.method == "family_relative_sample_verify_arithmetic"


def test_constraint_intervention_runs() -> None:
    from neuro_co.cax.constraint_intervention import (
        ConstraintInterventionAttribution,
        constraint_intervention_attribution,
    )

    env = CVRPTWEnv(size=5, capacity=20.0, horizon=10.0, window_width=2.0)
    model, state = _setup(env, batch=4)
    rep = constraint_intervention_attribution(
        model,
        env,
        state,
        problem="cvrptw",
        epsilon=0.2,
        num_samples=4,
        max_steps=3,
        seed=7,
    )
    assert isinstance(rep, ConstraintInterventionAttribution)
    assert rep.family_names == ["capacity", "time_window", "spatial"]
    assert rep.scores.shape == (4, 3, 3)
    assert rep.min_flip_distance.shape == (4, 3, 3)
    assert rep.mean_total_variation.shape == (4, 3, 3)
    assert rep.feasible_rate.shape == (4, 3, 3)
    assert rep.mask_change_rate.shape == (4, 3, 3)
    assert torch.isfinite(rep.scores).all()
    assert ((rep.feasible_rate >= 0) & (rep.feasible_rate <= 1)).all()
    assert ((rep.mask_change_rate >= 0) & (rep.mask_change_rate <= 1)).all()
    assert rep.mask_mode == "recomputed"


def test_constraint_intervention_fixed_mask_ablation_runs() -> None:
    from neuro_co.cax.constraint_intervention import constraint_intervention_attribution

    env = CVRPTWEnv(size=5, capacity=20.0, horizon=10.0, window_width=2.0)
    model, state = _setup(env, batch=3)
    rep = constraint_intervention_attribution(
        model,
        env,
        state,
        problem="cvrptw",
        epsilon=0.2,
        num_samples=4,
        max_steps=2,
        seed=8,
        mask_mode="fixed",
    )
    assert rep.mask_mode == "fixed"
    assert rep.scores.shape == (3, 2, 3)
    assert rep.mask_change_rate.shape == (3, 2, 3)
    assert torch.isfinite(rep.scores).all()


def test_routing_intervention_preserves_executed_prefix() -> None:
    from neuro_co.cax.cp_counterfactual import _sample_candidate

    env = CVRPTWEnv(size=5, capacity=20.0, horizon=10.0, window_width=2.0)
    _model, state = _setup(env, batch=4)
    action = torch.tensor([1, 2, 3, 4])
    state, _, _ = env.step(state, action)
    candidate, deltas = _sample_candidate(
        state,
        ("coords",),
        key="cvrptw",
        epsilon=0.2,
        sigma=0.2 / 3.0,
        generator=torch.Generator().manual_seed(11),
        env=env,
    )
    rows = torch.arange(action.shape[0])
    assert torch.equal(candidate.coords[:, 0], state.coords[:, 0])
    assert torch.equal(candidate.coords[rows, action], state.coords[rows, action])
    assert torch.equal(
        deltas["coords"][state.visited], torch.zeros_like(deltas["coords"][state.visited])
    )


def test_fjsp_discrete_interventions_preserve_prefix_and_feasibility() -> None:
    from neuro_co.cax.cp_counterfactual import _sample_candidate
    from neuro_co.cax.feasibility import is_feasible
    from neuro_co.problems.fjsp.env import FJSPEnv

    env = FJSPEnv(size=4, ops_per_job=3, num_machines=5)
    _model, state = _setup(env, batch=4)
    first_action = env.action_mask(state).float().argmax(dim=-1)
    state, _, _ = env.step(state, first_action)

    eligibility, eligibility_delta = _sample_candidate(
        state,
        ("ops_ma_adj", "proc_times"),
        key="fjsp",
        family="eligibility",
        epsilon=1.0,
        sigma=1.0 / 3.0,
        generator=torch.Generator().manual_seed(21),
        env=env,
    )
    completed = state.op_done.unsqueeze(-1).expand_as(state.ops_ma_adj)
    assert torch.equal(eligibility.ops_ma_adj[completed], state.ops_ma_adj[completed])
    assert torch.equal(eligibility.proc_times[completed], state.proc_times[completed])
    assert eligibility.ops_ma_adj.any(dim=-1).all()
    assert torch.equal(eligibility.proc_times > 0, eligibility.ops_ma_adj)
    assert eligibility_delta["ops_ma_adj"].abs().sum() > 0
    assert is_feasible(eligibility, env, "fjsp").all()

    precedence, precedence_delta = _sample_candidate(
        state,
        ("op_in_job",),
        key="fjsp",
        family="precedence",
        epsilon=1.0,
        sigma=1.0 / 3.0,
        generator=torch.Generator().manual_seed(22),
        env=env,
    )
    assert torch.equal(precedence.op_in_job[state.op_done], state.op_in_job[state.op_done])
    assert precedence_delta["op_in_job"].abs().sum() > 0
    for batch_idx in range(state.op_done.shape[0]):
        for job in range(env.size):
            mask = state.job_id[batch_idx] == job
            assert torch.equal(
                precedence.op_in_job[batch_idx, mask].sort().values,
                state.op_in_job[batch_idx, mask].sort().values,
            )
    assert is_feasible(precedence, env, "fjsp").all()


def test_fjsp_counterfactual_and_intervention_use_three_families() -> None:
    from neuro_co.cax.constraint_intervention import constraint_intervention_attribution
    from neuro_co.cax.cp_counterfactual import cp_counterfactual
    from neuro_co.problems.fjsp.env import FJSPEnv

    env = FJSPEnv(size=3, ops_per_job=3, num_machines=5)
    model, state = _setup(env, batch=3)
    expected = ["processing", "eligibility", "precedence"]
    counterfactual = cp_counterfactual(
        model,
        env,
        state,
        problem="fjsp",
        epsilon=0.2,
        max_shots=4,
        max_steps=2,
        seed=31,
    )
    intervention = constraint_intervention_attribution(
        model,
        env,
        state,
        problem="fjsp",
        epsilon=0.2,
        num_samples=4,
        max_steps=2,
        seed=32,
    )
    assert counterfactual.family_names == expected
    assert intervention.family_names == expected
    assert counterfactual.family_flipped.shape == (3, 2, 3)
    assert intervention.scores.shape == (3, 2, 3)
    assert torch.isfinite(intervention.scores).all()


def test_greedy_prefix_tracks_active_instances_and_preserves_ids() -> None:
    from neuro_co.cax.prefix import decision_activity, greedy_prefix, select_state

    env = OPEnv(size=8, budget=2.0)
    model, state = _setup(env, batch=6)
    rollout = greedy_prefix(model, env, state, steps=2)

    assert rollout.requested_steps == 2
    assert 1 <= rollout.executed_steps <= 2
    assert rollout.completed.shape == (6,)
    assert rollout.completion_step.shape == (6,)

    active_ids = rollout.active.nonzero(as_tuple=False).flatten()
    active_state = select_state(rollout.state, active_ids)
    assert active_state.coords.shape[0] == int(rollout.active.sum())
    if active_ids.numel():
        assert torch.equal(active_state.coords, rollout.state.coords[active_ids])
        activity = decision_activity(model, env, active_state, max_steps=3)
        assert activity.active.shape == activity.actions.shape
        assert activity.active.shape[0] == active_ids.numel()
        assert activity.active[:, 0].all()
