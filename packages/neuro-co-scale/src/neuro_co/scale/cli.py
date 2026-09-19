"""CLI for scale experiment smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from collections.abc import Callable
from pathlib import Path

from .benchmark import (
    BenchmarkGridConfig,
    PartitionBaselineConfig,
    iter_benchmark_records,
    run_partition_baseline,
    write_benchmark_jsonl,
)
from .datasets import (
    load_cvrplib_instance,
    load_cvrplib_solution,
    sample_clustered_instances,
    sample_core_instances,
)
from .partition import (
    RefinementScoreMode,
    capacity_aware_sweep_partition,
    feature_aware_partition,
    grid_partition,
    morton_partition,
    morton_refined_partition,
    sweep_partition,
)
from .types import Partition, VRPInstance

METHOD_CHOICES = (
    "sweep",
    "capacity_sweep",
    "grid",
    "morton",
    "morton_refined",
    "feature_aware",
)
LOCAL_CONSTRUCTOR_CHOICES = ("nearest_neighbor", "time_window")
REFINEMENT_SCORE_CHOICES = ("compactness", "route", "hybrid")
ENCODER_CHOICES = ("cluster_local", "am")
DISTRIBUTION_CHOICES = ("uniform", "clustered")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neuroco-scale")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("partition-smoke", help="run a deterministic partition smoke test")
    p.add_argument("--problem", default="cvrp", choices=["cvrp", "cvrptw"])
    p.add_argument("--size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-customers", type=int, default=10)
    p.add_argument("--capacity", type=float, default=50.0)
    p.add_argument("--capacity-fraction", type=float, default=0.95)
    p.add_argument(
        "--method",
        default="capacity_sweep",
        choices=METHOD_CHOICES,
    )
    p.add_argument(
        "--local-constructor",
        default="nearest_neighbor",
        choices=LOCAL_CONSTRUCTOR_CHOICES,
    )
    p.add_argument(
        "--refinement-score-mode",
        default="hybrid",
        choices=REFINEMENT_SCORE_CHOICES,
    )
    p.add_argument("--refinement-hybrid-shortlist", type=int, default=2)
    p.add_argument("--out", type=Path, default=None)

    b = sub.add_parser("benchmark", help="run a repeatable partition benchmark grid")
    b.add_argument("--problem", default="cvrp", choices=["cvrp", "cvrptw"])
    b.add_argument("--sizes", type=_parse_int_tuple, default=(50, 100, 200))
    b.add_argument("--seeds", type=_parse_seed_tuple, default=(0,))
    b.add_argument("--num-instances", type=int, default=1)
    b.add_argument("--methods", type=_parse_methods, default=METHOD_CHOICES)
    b.add_argument(
        "--local-constructor",
        default="nearest_neighbor",
        choices=LOCAL_CONSTRUCTOR_CHOICES,
    )
    b.add_argument(
        "--refinement-score-mode",
        default="hybrid",
        choices=REFINEMENT_SCORE_CHOICES,
    )
    b.add_argument("--refinement-hybrid-shortlist", type=int, default=2)
    b.add_argument("--max-customers", type=int, default=10)
    b.add_argument("--capacity", type=float, default=50.0)
    b.add_argument("--capacity-fraction", type=float, default=0.95)
    b.add_argument("--horizon", type=float, default=10.0)
    b.add_argument("--window-width", type=float, default=2.0)
    b.add_argument("--out", type=Path, default=None)

    t = sub.add_parser("train", help="train the hierarchical policy with REINFORCE")
    t.add_argument("--problem", default="cvrp", choices=["cvrp", "cvrptw"])
    t.add_argument("--distribution", default="uniform", choices=DISTRIBUTION_CHOICES)
    t.add_argument("--size", type=int, default=100)
    t.add_argument("--max-customers", type=int, default=20)
    t.add_argument(
        "--max-customers-range",
        default="",
        help="e.g. '10,100': draw the cluster size per step so the policy is not locked to "
        "one value (real instances need routes of 4 to 190 customers)",
    )
    t.add_argument("--method", default="morton", choices=METHOD_CHOICES)
    t.add_argument("--encoder", default="cluster_local", choices=ENCODER_CHOICES)
    t.add_argument(
        "--neighbor-span",
        type=int,
        default=0,
        help="0=route confined to cluster (S1); >=1 crosses into Morton neighbors (S2)",
    )
    t.add_argument(
        "--reanchor",
        action="store_true",
        help="S3: re-anchor an open route to a new cluster when its neighborhood is exhausted",
    )
    t.add_argument(
        "--checkpoint",
        action="store_true",
        help="gradient-checkpoint the decode loop (lower memory for large-n training)",
    )
    t.add_argument("--checkpoint-chunk", type=int, default=64, help="steps per checkpoint chunk")
    t.add_argument("--capacity", type=float, default=50.0)
    t.add_argument("--capacity-fraction", type=float, default=0.95)
    t.add_argument("--refinement-score-mode", default="hybrid", choices=REFINEMENT_SCORE_CHOICES)
    t.add_argument("--refinement-hybrid-shortlist", type=int, default=2)
    t.add_argument("--hidden-dim", type=int, default=128)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--steps", type=int, default=200)
    t.add_argument("--batch", type=int, default=8, help="instances per step")
    t.add_argument("--starts", type=int, default=16, help="rollouts per instance")
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--lr-warmup", type=int, default=0, help=">0 enables warmup+cosine decay")
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--log-every", type=int, default=10)
    t.add_argument("--eval-every", type=int, default=0, help="held-out eval period (0=off)")
    t.add_argument("--eval-instances", type=int, default=32)
    t.add_argument("--eval-seed", type=int, default=999_999)
    t.add_argument("--out", type=Path, default=None, help="checkpoint path")
    t.add_argument(
        "--init-ckpt",
        type=Path,
        default=None,
        help="initialize model weights from a checkpoint; starts a new optimizer",
    )

    t.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume full model, optimizer, scheduler and RNG state",
    )
    t.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="atomically save a resumable checkpoint every N updates",
    )
    t.add_argument(
        "--max-steps-this-run",
        type=int,
        default=None,
        help="cap updates in this invocation without changing the total schedule",
    )

    e = sub.add_parser("eval", help="evaluate a checkpoint against heuristic baselines")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument(
        "--instances-dir",
        type=Path,
        default=None,
        help="evaluate on CVRPLIB .vrp files in this dir (their .sol files are the BKS baseline)",
    )
    e.add_argument(
        "--limit", type=int, default=0, help="max instances from --instances-dir (0=all)"
    )
    e.add_argument(
        "--pyvrp-time",
        type=float,
        default=0.0,
        help="if >0, add a PyVRP (HGS) metaheuristic baseline with this per-instance "
        "runtime budget in seconds (requires --instances-dir)",
    )
    e.add_argument(
        "--auto-max-customers",
        action="store_true",
        help="size clusters per instance from its natural route length (capacity / mean demand) "
        "instead of a fixed --max-customers; real instances range from ~4 to ~190",
    )
    e.add_argument(
        "--distribution",
        default="uniform",
        choices=DISTRIBUTION_CHOICES,
        help="eval distribution (can differ from training: cross-distribution eval)",
    )
    e.add_argument("--size", type=int, default=0, help="eval instance size (0 = checkpoint size)")
    e.add_argument(
        "--max-customers", type=int, default=0, help="cluster cap (0 = checkpoint value)"
    )
    e.add_argument("--eval-instances", type=int, default=64)
    e.add_argument("--eval-seed", type=int, default=999_999)
    e.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    e.add_argument("--num-samples", type=int, default=1)
    e.add_argument("--capacity-fraction", type=float, default=0.95)
    e.add_argument(
        "--capacity",
        type=float,
        default=None,
        help="synthetic capacity; defaults to saved config or legacy generator default",
    )
    e.add_argument("--device", default="cpu")

    ns = parser.parse_args(argv)
    if ns.cmd == "partition-smoke":
        env_kwargs = {"capacity": ns.capacity}
        if ns.problem == "cvrptw":
            env_kwargs.update({"horizon": 10.0, "window_width": 2.0})
        instance = sample_core_instances(
            ns.problem,
            size=ns.size,
            num_instances=1,
            seed=ns.seed,
            **env_kwargs,
        )[0]
        _, payload = run_partition_baseline(
            instance,
            PartitionBaselineConfig(
                method=ns.method,
                max_customers=ns.max_customers,
                capacity_fraction=ns.capacity_fraction,
                local_constructor=ns.local_constructor,
                refinement_score_mode=ns.refinement_score_mode,
                refinement_hybrid_shortlist=ns.refinement_hybrid_shortlist,
            ),
        )
        text = json.dumps(payload, indent=2)
        if ns.out is None:
            print(text)
        else:
            ns.out.parent.mkdir(parents=True, exist_ok=True)
            ns.out.write_text(text)
        return 0
    if ns.cmd == "benchmark":
        cfg = BenchmarkGridConfig(
            problem=ns.problem,
            sizes=ns.sizes,
            methods=ns.methods,
            seeds=ns.seeds,
            num_instances=ns.num_instances,
            max_customers=ns.max_customers,
            capacity=ns.capacity,
            capacity_fraction=ns.capacity_fraction,
            local_constructor=ns.local_constructor,
            refinement_score_mode=ns.refinement_score_mode,
            refinement_hybrid_shortlist=ns.refinement_hybrid_shortlist,
            horizon=ns.horizon,
            window_width=ns.window_width,
        )
        if ns.out is not None:
            count = write_benchmark_jsonl(ns.out, cfg)
            print(json.dumps({"out": str(ns.out), "records": count}, sort_keys=True))
        else:
            for record in iter_benchmark_records(cfg):
                print(json.dumps(record, sort_keys=True))
        return 0
    if ns.cmd == "train":
        return _run_train(ns)
    if ns.cmd == "eval":
        return _run_eval(ns)
    raise AssertionError(f"unhandled command {ns.cmd}")


def _partition_fn(
    method: str,
    *,
    max_customers: int,
    capacity_fraction: float = 0.95,
    score_mode: RefinementScoreMode = "hybrid",
    shortlist: int = 2,
) -> Callable[[VRPInstance], Partition]:
    mc = max_customers
    cf = capacity_fraction
    if method == "sweep":
        return lambda instance: sweep_partition(instance, max_customers=mc)
    if method == "capacity_sweep":
        return lambda instance: capacity_aware_sweep_partition(
            instance, max_customers=mc, capacity_fraction=cf
        )
    if method == "grid":
        return lambda instance: grid_partition(instance, max_customers=mc)
    if method == "morton":
        return lambda instance: morton_partition(instance, max_customers=mc, capacity_fraction=cf)
    if method == "morton_refined":
        return lambda instance: morton_refined_partition(
            instance,
            max_customers=mc,
            capacity_fraction=cf,
            score_mode=score_mode,
            hybrid_shortlist=shortlist,
        )
    if method == "feature_aware":
        return lambda instance: feature_aware_partition(
            instance, max_customers=mc, capacity_fraction=cf
        )
    raise ValueError(f"unknown partition method {method!r}")


def natural_route_size(instance: VRPInstance) -> int:
    """Customers a full vehicle serves on average: capacity / mean customer demand."""

    mean_demand = float(instance.demand[1:].mean())
    if mean_demand <= 0:
        raise ValueError("instance has no positive customer demand")
    return max(2, round(instance.capacity / mean_demand))


def _auto_partition_fn(
    method: str,
    *,
    capacity_fraction: float,
    score_mode: RefinementScoreMode = "hybrid",
    shortlist: int = 2,
) -> Callable[[VRPInstance], Partition]:
    """Partition with a per-instance cluster size, not one fixed value for all."""

    def build(instance: VRPInstance) -> Partition:
        return _partition_fn(
            method,
            max_customers=natural_route_size(instance),
            capacity_fraction=capacity_fraction,
            score_mode=score_mode,
            shortlist=shortlist,
        )(instance)

    return build


def _build_partition_fn(ns: argparse.Namespace) -> Callable[[VRPInstance], Partition]:
    return _partition_fn(
        ns.method,
        max_customers=ns.max_customers,
        capacity_fraction=ns.capacity_fraction,
        score_mode=ns.refinement_score_mode,
        shortlist=ns.refinement_hybrid_shortlist,
    )


def _build_policy(
    encoder: str,
    *,
    hidden_dim: int,
    layers: int,
    heads: int,
    neighbor_span: int = 0,
    reanchor: bool = False,
    use_checkpoint: bool = False,
    checkpoint_chunk: int = 64,
):
    from neuro_co.core.models import AMEncoder, PointerDecoder

    from .cluster_encoder import ClusterLocalEncoder
    from .decode import HierarchicalPolicy, NeighborhoodActionSet
    from .hierarchical import LinearClusterSelector

    if encoder == "cluster_local":
        enc: object = ClusterLocalEncoder(
            in_dim=3, hidden_dim=hidden_dim, num_layers=layers, num_heads=heads
        )
    else:
        enc = AMEncoder(in_dim=3, hidden_dim=hidden_dim, num_layers=layers, num_heads=heads)
    # neighbor_span 0 -> S1 (route confined to cluster); >=1 -> S2 (cross-cluster).
    # reanchor -> S3 (mid-route re-anchoring when the neighborhood is exhausted).
    builder = NeighborhoodActionSet(neighbor_span) if neighbor_span > 0 else None
    return HierarchicalPolicy(
        enc,
        LinearClusterSelector(hidden_dim=hidden_dim),
        PointerDecoder(hidden_dim=hidden_dim, num_heads=heads),
        builder,
        reanchor=reanchor,
        use_checkpoint=use_checkpoint,
        checkpoint_chunk=checkpoint_chunk,
    )


def _make_instances(
    problem: str,
    size: int,
    num_instances: int,
    seed: int,
    distribution: str = "uniform",
    capacity: float | None = 50.0,
) -> list[VRPInstance]:
    if distribution == "clustered":
        return sample_clustered_instances(
            size=size,
            num_instances=num_instances,
            seed=seed,
            capacity=50.0 if capacity is None else capacity,
        )
    env_kwargs: dict[str, float] = {} if capacity is None else {"capacity": capacity}
    if problem == "cvrptw":
        env_kwargs.update({"horizon": 10.0, "window_width": 2.0})
    return sample_core_instances(
        problem, size=size, num_instances=num_instances, seed=seed, **env_kwargs
    )


def _run_train(ns: argparse.Namespace) -> int:
    import torch

    from .eval import evaluate_constructor, evaluate_policy, paired_gap
    from .train import TrainConfig, TrainStep, train_hierarchical

    # Seed weight init (uses the global RNG) so a run is reproducible end to end,
    # not just the sampling generator.
    torch.manual_seed(ns.seed)
    policy = _build_policy(
        ns.encoder,
        hidden_dim=ns.hidden_dim,
        layers=ns.layers,
        heads=ns.heads,
        neighbor_span=ns.neighbor_span,
        reanchor=ns.reanchor,
        use_checkpoint=ns.checkpoint,
        checkpoint_chunk=ns.checkpoint_chunk,
    )
    if ns.init_ckpt is not None and ns.resume is not None:
        raise ValueError("--init-ckpt and --resume are mutually exclusive")
    if ns.init_ckpt is not None:
        initial = torch.load(ns.init_ckpt, map_location="cpu", weights_only=False)
        policy.load_state_dict(initial["state_dict"])
    fixed_pf = _build_partition_fn(ns)
    span = (
        tuple(int(x) for x in ns.max_customers_range.split(",")) if ns.max_customers_range else None
    )

    def instances(step: int) -> list[VRPInstance]:
        batch = _make_instances(
            ns.problem, ns.size, ns.batch, ns.seed * 100_000 + step, ns.distribution, ns.capacity
        )
        if span is not None:
            # One cluster size per step: shapes stay uniform inside a step (no padding
            # waste) while the policy still sees the whole range across steps.
            drawn = random.Random(ns.seed * 1_000_003 + step).randint(span[0], span[1])
            for instance in batch:
                instance.metadata["max_customers"] = drawn
        return batch

    def partition_fn(instance: VRPInstance) -> Partition:
        drawn = instance.metadata.get("max_customers")
        if drawn is None:
            return fixed_pf(instance)
        return _partition_fn(
            ns.method,
            max_customers=drawn,
            capacity_fraction=ns.capacity_fraction,
            score_mode=ns.refinement_score_mode,
            shortlist=ns.refinement_hybrid_shortlist,
        )(instance)

    cfg = TrainConfig(
        steps=ns.steps,
        n_starts=ns.starts,
        batch_instances=ns.batch,
        lr=ns.lr,
        grad_clip=ns.grad_clip,
        lr_warmup_steps=ns.lr_warmup,
        seed=ns.seed,
        device=ns.device,
    )

    # Fixed held-out set + heuristic reference for periodic eval.
    held_out: list[VRPInstance] = []
    reference = None
    if ns.eval_every > 0:
        held_out = _make_instances(
            ns.problem, ns.size, ns.eval_instances, ns.eval_seed, ns.distribution, ns.capacity
        )
        reference = evaluate_constructor(held_out, fixed_pf, name=f"{ns.method}+nn")

    def log(record: TrainStep) -> None:
        if ns.log_every > 0 and (record.step % ns.log_every == 0 or record.step == ns.steps - 1):
            print(
                json.dumps(
                    {
                        "step": record.step,
                        "loss": round(record.loss, 5),
                        "mean_cost": round(record.mean_cost, 4),
                        "best_cost": round(record.best_cost, 4),
                        "baseline_cost": round(record.baseline_cost, 4),
                    },
                    sort_keys=True,
                )
            )
        if reference is not None and (
            record.step % ns.eval_every == 0 or record.step == ns.steps - 1
        ):
            policy_eval = evaluate_policy(policy, held_out, fixed_pf, name="policy")
            gap = paired_gap(policy_eval, reference)
            policy.train()
            print(
                json.dumps(
                    {
                        "eval_step": record.step,
                        "policy_cost": round(policy_eval.mean_cost, 4),
                        "ref_cost": round(reference.mean_cost, 4),
                        "ref": reference.name,
                        "gap_pct": round(gap.mean_gap_pct, 2),
                        "win_rate": round(gap.win_rate, 3),
                        "feasible": round(policy_eval.feasible_rate, 3),
                    },
                    sort_keys=True,
                )
            )

    model_config = {
        "encoder": ns.encoder,
        "hidden_dim": ns.hidden_dim,
        "layers": ns.layers,
        "heads": ns.heads,
        "problem": ns.problem,
        "size": ns.size,
        "method": ns.method,
        "max_customers": ns.max_customers,
        "neighbor_span": ns.neighbor_span,
        "reanchor": ns.reanchor,
        "capacity": ns.capacity,
        "capacity_fraction": ns.capacity_fraction,
        "distribution": ns.distribution,
    }
    sampler_config = {
        "max_customers_range": ns.max_customers_range,
        "refinement_score_mode": ns.refinement_score_mode,
        "refinement_hybrid_shortlist": ns.refinement_hybrid_shortlist,
    }
    source_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "cli.py",
            "train.py",
            "decode.py",
            "cluster_encoder.py",
            "datasets.py",
            "partition.py",
        )
    }
    resume_state = None
    previous_wall = 0.0
    if ns.resume is not None:
        saved = torch.load(ns.resume, map_location="cpu", weights_only=False)
        if saved.get("config") != model_config or saved.get("sampler_config") != sampler_config:
            raise ValueError("resume instance/partition/model configuration differs")
        if saved.get("source_sha256") != source_hashes:
            raise ValueError("resume source code differs; retain the original training package")
        if "training_state" not in saved:
            raise ValueError("checkpoint has no optimizer/RNG state; use --init-ckpt for a new run")
        resume_state = saved["training_state"]
        previous_wall = saved.get("training", {}).get("wall_time_s", 0.0)
    started = time.perf_counter()

    def save_state(state) -> None:
        if ns.out is None:
            return
        ns.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": state["state_dict"],
            "config": model_config,
            "sampler_config": sampler_config,
            "source_sha256": source_hashes,
            "training_state": state,
            "training": {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(ns).items()},
                "wall_time_s": previous_wall + time.perf_counter() - started,
                "torch_version": str(torch.__version__),
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "device_name": torch.cuda.get_device_name()
                if ns.device.startswith("cuda")
                else ns.device,
                "initial_checkpoint_sha256": (
                    hashlib.sha256(ns.init_ckpt.read_bytes()).hexdigest()
                    if ns.init_ckpt is not None
                    else None
                ),
                "history": state["history"],
            },
        }
        temporary = ns.out.with_suffix(ns.out.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(ns.out)
        print(
            json.dumps({"checkpoint": str(ns.out), "completed_step": state["completed_step"]}),
            flush=True,
        )

    history = train_hierarchical(
        policy,
        partition_fn=partition_fn,
        instances=instances,
        cfg=cfg,
        on_step=log,
        resume_state=resume_state,
        on_checkpoint=save_state if ns.out is not None else None,
        checkpoint_every=ns.save_every,
        max_steps=ns.max_steps_this_run,
    )
    print(
        json.dumps(
            {
                "out": str(ns.out) if ns.out is not None else None,
                "steps": len(history),
                "total_steps": ns.steps,
                "complete": len(history) == ns.steps,
                "final_mean_cost": round(history[-1].mean_cost, 4),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _run_eval(ns: argparse.Namespace) -> int:
    import torch

    from .eval import (
        MethodEval,
        evaluate_constructor,
        evaluate_policy,
        evaluate_pyvrp,
        paired_gap,
    )

    ckpt = torch.load(ns.ckpt, map_location=ns.device, weights_only=False)
    cfg = ckpt["config"]
    policy = _build_policy(
        cfg["encoder"],
        hidden_dim=cfg["hidden_dim"],
        layers=cfg["layers"],
        heads=cfg["heads"],
        neighbor_span=cfg.get("neighbor_span", 0),
        reanchor=cfg.get("reanchor", False),
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.to(ns.device)

    # Policy is size-agnostic and the partition handles any n, so a checkpoint
    # trained at one size can be evaluated at another (cross-scale).
    size = ns.size if ns.size > 0 else cfg["size"]
    max_customers = ns.max_customers if ns.max_customers > 0 else cfg["max_customers"]
    bks_eval = None
    if ns.instances_dir:
        # Public CVRPLIB instances; .sol files provide a versioned BKS baseline.
        files = sorted(Path(ns.instances_dir).glob("*.vrp"))
        if ns.limit > 0:
            files = files[: ns.limit]
        if not files:
            raise ValueError(f"no .vrp files under {ns.instances_dir}")
        held_out = [load_cvrplib_instance(f) for f in files]
        bks_costs, bks_routes = [], []
        for f in files:
            sol, cost = load_cvrplib_solution(f.with_suffix(".sol"))
            if cost is None:
                raise ValueError(f"{f.with_suffix('.sol')} has no Cost line")
            bks_costs.append(cost)
            bks_routes.append(sol.num_routes)
        bks_eval = MethodEval(
            name="bks",
            mean_cost=sum(bks_costs) / len(bks_costs),
            feasible_rate=1.0,
            mean_routes=sum(bks_routes) / len(bks_routes),
            costs=tuple(bks_costs),
        )
    else:
        held_out = _make_instances(
            cfg["problem"],
            size,
            ns.eval_instances,
            ns.eval_seed,
            ns.distribution,
            ns.capacity if ns.capacity is not None else cfg.get("capacity"),
        )

    def make_pf(method: str) -> Callable[[VRPInstance], Partition]:
        if ns.auto_max_customers:
            return _auto_partition_fn(method, capacity_fraction=ns.capacity_fraction)
        return _partition_fn(
            method, max_customers=max_customers, capacity_fraction=ns.capacity_fraction
        )

    policy_pf = make_pf(cfg["method"])
    generator = None
    if ns.mode == "sample":
        generator = torch.Generator(device=ns.device).manual_seed(ns.eval_seed)
    policy_eval = evaluate_policy(
        policy,
        held_out,
        policy_pf,
        name="policy",
        mode=ns.mode,
        num_samples=ns.num_samples,
        generator=generator,
    )

    baselines = [] if bks_eval is None else [bks_eval]
    for method in ("morton", "sweep"):
        baselines.append(evaluate_constructor(held_out, make_pf(method), name=f"{method}+nn"))
    pyvrp_eval = None
    if ns.pyvrp_time > 0:
        if not ns.instances_dir:
            raise ValueError("--pyvrp-time requires --instances-dir (needs CVRPLIB .vrp files)")
        pyvrp_eval = evaluate_pyvrp(files, time_limit=ns.pyvrp_time, seed=ns.eval_seed)
        baselines.append(pyvrp_eval)

    methods = [policy_eval, *baselines]
    payload = {
        "size": size,
        "train_size": cfg["size"],
        "distribution": ns.distribution,
        "instances": len(held_out),
        "capacities": sorted({inst.capacity for inst in held_out}),
        "checkpoint_sha256": hashlib.sha256(ns.ckpt.read_bytes()).hexdigest(),
        "mode": ns.mode,
        "num_samples": ns.num_samples,
        "methods": {
            m.name: {
                "mean_cost": round(m.mean_cost, 4),
                "feasible": round(m.feasible_rate, 3),
                "mean_routes": round(m.mean_routes, 2),
            }
            for m in methods
        },
        "gaps_vs_baselines": [
            {
                "reference": gap.reference,
                "policy_gap_pct": round(gap.mean_gap_pct, 2),
                "policy_win_rate": round(gap.win_rate, 3),
            }
            for gap in (paired_gap(policy_eval, base) for base in baselines)
        ],
    }
    if ns.instances_dir:
        payload["instances_dir"] = str(ns.instances_dir)
        payload["per_instance"] = [
            {
                "name": inst.name,
                "n": inst.num_customers,
                "policy": round(policy_eval.costs[i], 1),
                "bks": bks_eval.costs[i] if bks_eval else None,
                "gap_bks_pct": round(
                    100.0 * (policy_eval.costs[i] - bks_eval.costs[i]) / bks_eval.costs[i], 2
                )
                if bks_eval
                else None,
                "pyvrp": round(pyvrp_eval.costs[i], 1) if pyvrp_eval else None,
                "pyvrp_gap_bks_pct": round(
                    100.0 * (pyvrp_eval.costs[i] - bks_eval.costs[i]) / bks_eval.costs[i], 2
                )
                if (pyvrp_eval and bks_eval)
                else None,
            }
            for i, inst in enumerate(held_out)
        ]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    if any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return parts


def _parse_seed_tuple(value: str) -> tuple[int, ...]:
    parts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parts:
        raise argparse.ArgumentTypeError("value must contain at least one seed")
    if any(part < 0 for part in parts):
        raise argparse.ArgumentTypeError("all seeds must be non-negative integers")
    return parts


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(methods).difference(METHOD_CHOICES))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown methods: {', '.join(unknown)}")
    if not methods:
        raise argparse.ArgumentTypeError("value must contain at least one method")
    return methods


if __name__ == "__main__":
    raise SystemExit(main())
