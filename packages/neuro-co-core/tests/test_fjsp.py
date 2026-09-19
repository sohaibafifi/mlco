"""FJSP env tests."""

import torch

from neuro_co.core.envs.fjsp import FJSPEnv


def test_reset_shapes() -> None:
    env = FJSPEnv(size=3, ops_per_job=2, num_machines=4)  # 6 ops
    s = env.reset(4)
    assert s.proc_times.shape == (4, 6, 4)
    assert s.ops_ma_adj.shape == (4, 6, 4)
    assert s.num_eligible.shape == (4, 6)
    assert (s.num_eligible >= 1).all()  # min_eligible guarantee
    assert env.encoder_in_dim == 4


def test_proc_zero_when_ineligible() -> None:
    env = FJSPEnv(size=2, ops_per_job=2, num_machines=5)
    s = env.reset(8, generator=torch.Generator().manual_seed(0))
    # proc_time is 0 exactly where op is ineligible.
    assert torch.equal((s.proc_times > 0), s.ops_ma_adj | (s.proc_times > 0) & s.ops_ma_adj)
    assert torch.equal(s.proc_times.eq(0), ~s.ops_ma_adj | s.proc_times.eq(0))


def test_initial_mask_only_first_ops() -> None:
    env = FJSPEnv(size=3, ops_per_job=3, num_machines=4)
    s = env.reset(2)
    mask = env.action_mask(s)
    # Only op_in_job==0 ready initially.
    assert torch.equal(mask, s.op_in_job == 0)


def test_precedence_enforced() -> None:
    env = FJSPEnv(size=1, ops_per_job=3, num_machines=3)  # one job, ops 0,1,2
    s = env.reset(1)
    assert env.action_mask(s)[0].tolist() == [True, False, False]
    s, _, _ = env.step(s, torch.tensor([0]))
    assert env.action_mask(s)[0].tolist() == [False, True, False]
    s, _, _ = env.step(s, torch.tensor([1]))
    assert env.action_mask(s)[0].tolist() == [False, False, True]


def test_action_mask_respects_explicit_reordered_precedence() -> None:
    env = FJSPEnv(size=1, ops_per_job=3, num_machines=3)
    state = env.reset(1)
    reordered = state.replace(op_in_job=torch.tensor([[2, 0, 1]]))
    assert env.action_mask(reordered)[0].tolist() == [False, True, False]
    reordered, _, _ = env.step(reordered, torch.tensor([1]))
    assert env.action_mask(reordered)[0].tolist() == [False, False, True]
    reordered, _, _ = env.step(reordered, torch.tensor([2]))
    assert env.action_mask(reordered)[0].tolist() == [True, False, False]


def test_full_rollout_makespan() -> None:
    env = FJSPEnv(size=2, ops_per_job=2, num_machines=3)  # 4 ops
    s = env.reset(1, generator=torch.Generator().manual_seed(0))
    done = torch.tensor([False])
    steps = 0
    while not bool(done.all()) and steps < env.max_steps(s):
        a = env.action_mask(s).float().argmax(dim=-1)
        s, reward, done = env.step(s, a)
        steps += 1
    assert done.all()
    assert s.makespan.item() > 0
    assert reward.item() == -s.makespan.item()


def test_features_dim() -> None:
    env = FJSPEnv(size=3, ops_per_job=2, num_machines=4)
    s = env.reset(2)
    assert env.build_features(s).shape == (2, 6, 4)
