"""Constraint magnitudes on routes with known loads, arrival times, and lengths."""

import pytest
import torch

from neuro_co.core.envs.cvrptw import CVRPTWEnv
from neuro_co.core.envs.op import OPEnv
from neuro_co.dual.slack import family_slack, family_violation


def test_cvrptw_capacity_and_customer_lateness_use_signed_margins() -> None:
    env = CVRPTWEnv(size=2, capacity=4.0, horizon=10.0)
    state = env.reset(2).replace(
        coords=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]).expand(2, -1, -1),
        demand=torch.tensor([[0.0, 3.0, 3.0], [0.0, 1.0, 2.0]]),
        tw_early=torch.zeros(2, 3),
        tw_late=torch.tensor([[10.0, 10.0, 1.0], [10.0, 10.0, 10.0]]),
    )
    actions = torch.tensor([[1, 2, 0], [1, 2, 0]])

    violation = family_violation(state, env, actions, "cvrptw")
    slack = family_slack(state, env, actions, "cvrptw")

    # Route loads are 6 and 3. Both reach the second customer at time 2.
    torch.testing.assert_close(violation["capacity"], torch.tensor([0.5, 0.0]))
    torch.testing.assert_close(violation["time_window"], torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(violation["spatial"], torch.zeros(2))
    torch.testing.assert_close(slack["capacity"], torch.tensor([0.0, 0.25]))
    torch.testing.assert_close(slack["time_window"], torch.tensor([0.0, 0.8]))


def test_cvrptw_depot_visits_reset_load_and_clock() -> None:
    env = CVRPTWEnv(size=2, capacity=4.0, horizon=10.0)
    state = env.reset(1).replace(
        coords=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        demand=torch.tensor([[0.0, 3.0, 3.0]]),
        tw_early=torch.zeros(1, 3),
        tw_late=torch.tensor([[10.0, 1.5, 2.5]]),
    )
    actions = torch.tensor([[1, 0, 2, 0]])

    violation = family_violation(state, env, actions, "vrptw")
    slack = family_slack(state, env, actions, "vrptw")

    for value in violation.values():
        torch.testing.assert_close(value, torch.zeros(1))
    torch.testing.assert_close(slack["capacity"], torch.tensor([0.25]))
    torch.testing.assert_close(slack["time_window"], torch.tensor([0.05]))


def test_cvrptw_waiting_propagates_to_later_customers() -> None:
    env = CVRPTWEnv(size=2, capacity=4.0, horizon=10.0)
    state = env.reset(1).replace(
        coords=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]),
        demand=torch.tensor([[0.0, 1.0, 1.0]]),
        tw_early=torch.tensor([[0.0, 5.0, 0.0]]),
        tw_late=torch.tensor([[10.0, 5.0, 5.5]]),
    )

    violation = family_violation(state, env, torch.tensor([[1, 2, 0]]), "cvrptw")

    # Waiting until time 5 at customer 1 puts the next arrival at time 6.
    torch.testing.assert_close(violation["time_window"], torch.tensor([0.05]))
    torch.testing.assert_close(violation["capacity"], torch.zeros(1))


@pytest.mark.parametrize("actions", ([[1], [1]], [[1, 0], [1, 0]]))
def test_op_budget_violation_includes_return_to_depot(actions: list[list[int]]) -> None:
    env = OPEnv(size=2, budget=3.0)
    state = env.reset(2).replace(
        coords=torch.tensor(
            [[[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]]
        ),
        prize=torch.tensor([[0.0, 0.7, 0.3], [0.0, 0.6, 0.4]]),
    )
    trajectory = torch.tensor(actions)

    violation = family_violation(state, env, trajectory, "op")
    slack = family_slack(state, env, trajectory, "op")

    # Closed tour lengths are 4 and 1, against a budget of 3.
    torch.testing.assert_close(violation["budget"], torch.tensor([1.0 / 3, 0.0]))
    torch.testing.assert_close(violation["prize"], torch.zeros(2))
    torch.testing.assert_close(violation["spatial"], torch.zeros(2))
    torch.testing.assert_close(slack["budget"], torch.tensor([0.0, 2.0 / 3]))
    torch.testing.assert_close(slack["prize"], torch.tensor([0.7, 0.6]))
