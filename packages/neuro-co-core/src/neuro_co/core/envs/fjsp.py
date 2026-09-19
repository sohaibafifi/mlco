"""FJSP env: Flexible Job-Shop Scheduling. Pure functional.

`size` jobs, each a chain of `ops_per_job` operations, run on `num_machines`
machines. An operation is eligible on a random subset of machines, with a
machine-dependent processing time. An op becomes ready once its predecessor
in the same job is done. Objective: minimise makespan (max completion time).

Action space = operation selection (one of `n_ops`). The machine is chosen
greedily as the eligible machine giving the earliest completion (a standard
dispatching reduction), so the problem fits the pointer decoder's
n-action interface. Reward (higher = better) = -makespan at episode end.

State fields mirror the instance contract used by downstream constraint /
feasibility tooling: `proc_times`, `ops_ma_adj`, plus derived `num_eligible`
and `pad_mask` exposed as properties.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..state import State, register_state

_BIG = 1e9


@register_state
@dataclass(frozen=True, slots=True)
class FJSPState(State):
    proc_times: Float[Tensor, "b o m"]  # processing time, op on machine (0 if ineligible)
    ops_ma_adj: Bool[Tensor, "b o m"]  # eligibility: op may run on machine
    job_id: Int[Tensor, "b o"]  # job each op belongs to
    op_in_job: Int[Tensor, "b o"]  # position of op within its job (0-based)
    op_done: Bool[Tensor, "b o"]
    machine_time: Float[Tensor, "b m"]  # next-free time per machine
    job_time: Float[Tensor, "b j"]  # completion time of last scheduled op per job
    op_end: Float[Tensor, "b o"]  # completion time of each op (0 until scheduled)
    makespan: Float[Tensor, "b"]
    step_count: Int[Tensor, "b"]

    @property
    def num_eligible(self) -> Int[Tensor, "b o"]:
        return self.ops_ma_adj.sum(dim=-1)


class FJSPEnv:
    """Pure-functional Flexible Job-Shop env (operation-dispatch form)."""

    encoder_in_dim: int = 4  # [mean_proc, num_eligible_frac, ready, job_progress]

    def __init__(
        self,
        size: int,
        ops_per_job: int = 3,
        num_machines: int = 5,
        max_proc: int = 9,
        min_eligible: int = 1,
    ) -> None:
        if size < 1 or ops_per_job < 1 or num_machines < 1:
            raise ValueError("size, ops_per_job, num_machines must all be >= 1")
        self.size = size
        self.ops_per_job = ops_per_job
        self.num_machines = num_machines
        self.n_ops = size * ops_per_job
        self.max_proc = max_proc
        self.min_eligible = min(min_eligible, num_machines)

    def reset(
        self,
        batch_size: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str = "cpu",
    ) -> FJSPState:
        device = torch.device(device)
        b, o, m = batch_size, self.n_ops, self.num_machines
        gen_dev = generator.device if generator is not None else device

        def rand(*shape):
            t = torch.rand(*shape, generator=generator, device=gen_dev)
            return t if gen_dev == device else t.to(device)

        # Eligibility: threshold random matrix, then guarantee >= min_eligible
        # per op by forcing the top-k machines on.
        scores = rand(b, o, m)
        adj = scores > 0.5
        k = self.min_eligible
        topk = scores.topk(k, dim=-1).indices
        adj = adj.scatter(2, topk, True)

        proc = (rand(b, o, m) * (self.max_proc - 1) + 1).round()
        proc = torch.where(adj, proc, torch.zeros_like(proc))

        ar_o = torch.arange(o, device=device)
        job_id = (ar_o // self.ops_per_job).unsqueeze(0).expand(b, o).contiguous()
        op_in_job = (ar_o % self.ops_per_job).unsqueeze(0).expand(b, o).contiguous()

        return FJSPState(
            proc_times=proc,
            ops_ma_adj=adj,
            job_id=job_id,
            op_in_job=op_in_job,
            op_done=torch.zeros(b, o, dtype=torch.bool, device=device),
            machine_time=torch.zeros(b, m, device=device),
            job_time=torch.zeros(b, self.size, device=device),
            op_end=torch.zeros(b, o, device=device),
            makespan=torch.zeros(b, device=device),
            step_count=torch.zeros(b, dtype=torch.long, device=device),
        )

    def step(
        self,
        state: FJSPState,
        action: Int[Tensor, "b"],
    ) -> tuple[FJSPState, Float[Tensor, "b"], Bool[Tensor, "b"]]:
        b = action.shape[0]
        ar = torch.arange(b, device=action.device)
        job = state.job_id[ar, action]  # (b,)
        prev_end = state.job_time[ar, job]  # job's last-op completion

        adj = state.ops_ma_adj[ar, action]  # (b, m) eligibility of chosen op
        proc = state.proc_times[ar, action]  # (b, m)
        # Completion if scheduled on each machine = max(machine_free, prev_end)+proc.
        start = torch.maximum(state.machine_time, prev_end.unsqueeze(1))  # (b, m)
        completion = start + proc
        completion = torch.where(adj, completion, torch.full_like(completion, _BIG))
        machine = completion.argmin(dim=1)  # (b,) earliest-completion eligible machine
        chosen_end = completion[ar, machine]

        new_machine_time = state.machine_time.clone()
        new_machine_time[ar, machine] = chosen_end
        new_job_time = state.job_time.clone()
        new_job_time[ar, job] = chosen_end
        new_op_end = state.op_end.clone()
        new_op_end[ar, action] = chosen_end
        new_op_done = state.op_done.scatter(1, action.unsqueeze(1), True)
        new_makespan = torch.maximum(state.makespan, chosen_end)
        new_step = state.step_count + 1

        done = new_op_done.all(dim=1)
        reward = torch.where(done, -new_makespan, torch.zeros_like(new_makespan))

        return (
            state.replace(
                op_done=new_op_done,
                machine_time=new_machine_time,
                job_time=new_job_time,
                op_end=new_op_end,
                makespan=new_makespan,
                step_count=new_step,
            ),
            reward,
            done,
        )

    def action_mask(self, state: FJSPState) -> Bool[Tensor, "b o"]:
        """An op is ready iff undone and all earlier ops in its job are done."""
        # Resolve the predecessor from the explicit job/order fields instead
        # of assuming that operation indices remain in canonical order. This
        # is identical on generated instances and also supports valid future
        # precedence interventions used by the explanation audit.
        is_first = state.op_in_job == 0
        same_job = state.job_id.unsqueeze(2) == state.job_id.unsqueeze(1)
        predecessor_order = state.op_in_job.unsqueeze(1) - 1
        matches_order = state.op_in_job.unsqueeze(2) == predecessor_order
        predecessor = same_job & matches_order
        pred_done = (predecessor & state.op_done.unsqueeze(2)).any(dim=1)
        ready = (~state.op_done) & (is_first | pred_done)
        return ready

    # --- Env Protocol extensions.

    def build_features(self, state: FJSPState) -> Float[Tensor, "b o 4"]:
        m = self.num_machines
        elig = state.ops_ma_adj
        mean_proc = (state.proc_times.sum(-1) / elig.sum(-1).clamp_min(1)) / self.max_proc
        num_elig = elig.sum(-1).float() / m
        ready = self.action_mask(state).float()
        job_progress = state.op_in_job.float() / max(self.ops_per_job - 1, 1)
        return torch.stack([mean_proc, num_elig, ready, job_progress], dim=-1)

    def decoder_context(self, state: FJSPState) -> tuple[Int[Tensor, "b"], Int[Tensor, "b"]]:
        # No "first/current node" notion; use the most-recently-finished op as
        # current and op 0 as a fixed anchor. Decoder uses these only for context.
        current = state.op_end.argmax(dim=1)
        first = torch.zeros_like(current)
        return first, current

    def max_steps(self, state: FJSPState) -> int:
        return self.n_ops

    def pomo_first_mask(self, state: FJSPState) -> Bool[Tensor, "b o"]:
        return self.action_mask(state)
