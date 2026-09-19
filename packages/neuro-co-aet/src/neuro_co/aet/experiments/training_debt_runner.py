"""Measure the fixed-recipe AM/GNN training debt on native Windows.

One invocation executes every missing architecture/seed cell.  A seed is
measured from initialization through its final checkpoint in one uninterrupted
energy interval.  Before a partial seed staging directory is removed on resume,
the interrupted attempt and any available energy record are preserved in an
append-only event ledger.  Component-counter readings from separate attempts
are never added into a fictional whole-run energy value.  Already completed
seeds are preserved and fully revalidated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from neuro_co.aet.experiments import quality_runner as quality
from neuro_co.aet.experiments import software_recipe as software_recipe
from neuro_co.aet.experiments import software_smoke_runner as software_runner
from neuro_co.aet.experiments.quality_recipe import EvaluationMode
from neuro_co.aet.experiments.training_debt_recipe import (
    CLASSIFICATION,
    Architecture,
    TrainingDebtRecipe,
    TrainingDebtRecipeValidationError,
    load_aet_training_debt_recipe,
    runtime_qualification,
)

RUN_STATE_SCHEMA = "aet-training-debt-run-state/v1"
CHECKPOINT_SCHEMA = "aet-training-debt-checkpoint/v1"
TRAINING_RESULT_SCHEMA = "aet-training-debt-training-result/v1"
ENERGY_SCHEMA = "aet-training-debt-energy/v1"
EVALUATION_SCHEMA = "aet-training-debt-quality-evaluation/v1"
SEED_RESULT_SCHEMA = "aet-training-debt-seed-result/v1"
QUALITY_SCHEMA = "aet-training-debt-quality-qualification/v1"
SUMMARY_SCHEMA = "aet-training-debt-summary/v1"
MANIFEST_SCHEMA = "aet-training-debt-manifest/v1"
SOURCE_RECEIPT_SCHEMA = "aet-training-debt-source-receipt/v1"
ESTIMATE_SCHEMA = "aet-training-debt-estimate/v1"
ATTEMPT_LEDGER_SCHEMA = "aet-training-debt-attempt-ledger/v1"
COMPLETE_STATUS = "complete"
INCOMPLETE_STATUS = "incomplete_training_cells_pending"


class TrainingDebtError(RuntimeError):
    """Raised when measured training cannot continue safely."""


class TrainingDebtQualificationError(TrainingDebtError):
    """Raised before a measured training cell starts."""


@dataclass(frozen=True, slots=True)
class TrainingDebtResult:
    path: Path
    status: str
    complete: bool
    completed_cells: tuple[str, ...]
    remaining_cells: tuple[str, ...]
    manifest_path: Path | None = None
    manifest_sha256: str | None = None


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return quality._sha256_file(path)


def _load_json(path: Path) -> dict[str, Any]:
    return quality._load_json(path)


def _atomic_write_json(path: Path, value: Any) -> None:
    quality._atomic_write_json(path, value)


def _relative(root: Path, value: str) -> Path:
    return root.joinpath(*PurePosixPath(value).parts)


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    )


def _architecture_recipe_payload(
    recipe: TrainingDebtRecipe, architecture: Architecture
) -> dict[str, Any]:
    return {
        "architecture": architecture.architecture_id,
        "backbone": architecture.backbone,
        "model_configuration": architecture.model_configuration,
        "problem": "cvrp",
        "size": 50,
        "capacity": 40.0,
        "max_demand": 9,
        "training": asdict(recipe.training),
    }


def architecture_recipe_sha256(recipe: TrainingDebtRecipe, architecture: Architecture) -> str:
    return _canonical_sha256(_architecture_recipe_payload(recipe, architecture))


def model_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor names, exact metadata, and raw contiguous CPU bytes."""

    import torch

    digest = hashlib.sha256()
    digest.update(b"aet-model-state/v1\0")
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TrainingDebtError("model state must map string names to tensors")
        cpu = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(cpu.dtype), "shape": list(cpu.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _backend_identity(architecture: Architecture, model: Any) -> dict[str, Any]:
    encoder = getattr(model, "encoder", None)
    base = {
        "architecture": architecture.architecture_id,
        "backbone": architecture.backbone,
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "encoder_class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "decoder_class": (
            f"{type(getattr(model, 'decoder', None)).__module__}."
            f"{type(getattr(model, 'decoder', None)).__qualname__}"
        ),
    }
    if architecture.architecture_id == "am":
        return {**base, "layer_backend": "torch"}

    from torch_geometric.nn import TransformerConv

    from neuro_co.core.models import GNNEncoder

    if not isinstance(encoder, GNNEncoder):
        raise TrainingDebtQualificationError("GNN encoder class changed")
    if not encoder.blocks or not all(
        isinstance(block.gnn, TransformerConv) for block in encoder.blocks
    ):
        raise TrainingDebtQualificationError("GNN message-passing backend changed")
    try:
        pyg_version = importlib.metadata.version("torch-geometric")
    except importlib.metadata.PackageNotFoundError as exc:
        raise TrainingDebtQualificationError("torch-geometric is absent") from exc
    return {
        **base,
        "layer_backend": "torch_geometric.TransformerConv",
        "torch_geometric_version": pyg_version,
        "edge_index_backend": "torch_cdist_topk",
        "encoder_attributes": {
            "in_dim": encoder.in_dim,
            "hidden_dim": encoder.hidden_dim,
            "num_layers": encoder.num_layers,
            "num_heads": encoder.num_heads,
            "k_sparse": encoder.k_sparse,
            "dropout": encoder.dropout,
            "rbf_k": encoder.rbf_k,
            "fourier_feats": encoder.fourier_feats,
            "residual": encoder.residual,
        },
    }


def _model_identity(
    recipe: TrainingDebtRecipe, architecture: Architecture, model: Any
) -> dict[str, Any]:
    structure = [
        {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in model.state_dict().items()
    ]
    identity = {
        "architecture": architecture.architecture_id,
        "factory_backbone": architecture.backbone,
        "model_configuration": architecture.model_configuration,
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "state_dict_structure_sha256": _canonical_sha256(structure),
        "backend_identity": _backend_identity(architecture, model),
    }
    return {**identity, "model_identity_sha256": _canonical_sha256(identity)}


def _make_stack(
    recipe: TrainingDebtRecipe, architecture: Architecture, seed: int
) -> tuple[Any, Any, Any, Any, Any, dict[str, Any]]:
    import torch

    from neuro_co.core.factory import make_algo, make_env, make_model

    device = torch.device(f"cuda:{recipe.gpu_index}")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    env = make_env("cvrp", size=50, capacity=40.0, max_demand=9)
    model = make_model(
        env,
        backbone=architecture.backbone,
        hidden_dim=architecture.hidden_dim,
        num_layers=architecture.num_layers,
        num_heads=architecture.num_heads,
    )
    identity = _model_identity(recipe, architecture, model)
    training = recipe.training
    algo = make_algo(
        training.algorithm,
        model,
        env,
        device=str(device),
        batch_size=training.batch_size,
        n_starts=training.n_starts,
        lr=training.learning_rate,
        optimizer=training.optimizer,
        weight_decay=training.weight_decay,
        grad_clip=training.gradient_clip,
        precision=training.precision,
        eval_batch_size=4,
        eval_augment=1,
        lr_warmup_steps=0,
        lr_total_steps=0,
    )
    if algo.sched is not None:
        raise TrainingDebtError("POMO unexpectedly constructed a step scheduler")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        algo.opt,
        milestones=list(training.scheduler_milestones),
        gamma=training.scheduler_gamma,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    return env, model, algo, scheduler, generator, identity


def _checkpoint_payload(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    *,
    seed: int,
    model: Any,
    algo: Any,
    scheduler: Any,
    generator: Any,
    model_identity: dict[str, Any],
    epoch: int,
    items_processed: int,
    cumulative_training_walltime_s: float,
    recipe_sha256: str,
    git_sha: str,
    last_epoch_metrics: dict[str, float] | None,
) -> dict[str, Any]:
    import torch

    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "recipe_sha256": recipe_sha256,
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "git_sha": git_sha,
        "architecture": architecture.architecture_id,
        "model_configuration": architecture.model_configuration,
        "model_identity": model_identity,
        "training_seed": seed,
        "training": asdict(recipe.training),
        "completed_epochs": epoch,
        "items_processed": items_processed,
        "cumulative_training_walltime_s": cumulative_training_walltime_s,
        "optimizer_steps": int(algo._step),
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "optimizer": algo.opt.state_dict(),
        "epoch_scheduler": scheduler.state_dict(),
        "train_generator_state": generator.get_state().cpu(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()],
        "last_epoch_metrics": last_epoch_metrics,
        "saved_at": datetime.now(UTC).isoformat(),
    }


def _train_seed_workload(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    *,
    seed: int,
    staging: Path,
    recipe_sha256: str,
    git_sha: str,
) -> dict[str, Any]:
    import torch

    _env, model, algo, scheduler, generator, identity = _make_stack(recipe, architecture, seed)
    checkpoint_dir = staging / "training" / "checkpoints"
    latest_path = staging / "training" / "latest.pt"
    epoch = 0
    items_processed = 0
    cumulative_training_walltime_s = 0.0
    payload = _checkpoint_payload(
        recipe,
        architecture,
        seed=seed,
        model=model,
        algo=algo,
        scheduler=scheduler,
        generator=generator,
        model_identity=identity,
        epoch=0,
        items_processed=0,
        cumulative_training_walltime_s=0.0,
        recipe_sha256=recipe_sha256,
        git_sha=git_sha,
        last_epoch_metrics=None,
    )
    quality._write_training_checkpoint_pair(
        checkpoint_dir,
        latest_path,
        payload,
        epoch=0,
        checkpoint_epochs=recipe.training.checkpoint_epochs,
    )
    started = time.perf_counter()
    last_metrics: dict[str, float] | None = None
    while epoch < recipe.training.epochs:
        epoch_started = time.perf_counter()
        rows: list[tuple[int, dict[str, float]]] = []
        remaining = recipe.training.instances_per_epoch
        while remaining:
            current = min(recipe.training.batch_size, remaining)
            algo.cfg.batch_size = current
            metrics = {key: float(value) for key, value in algo.train_step(generator).items()}
            if any(not math.isfinite(value) for value in metrics.values()):
                raise TrainingDebtError(
                    f"non-finite metric for {architecture.architecture_id} seed {seed}"
                )
            rows.append((current, metrics))
            remaining -= current
            items_processed += current
        algo.cfg.batch_size = recipe.training.batch_size
        scheduler.step()
        epoch += 1
        cumulative_training_walltime_s += time.perf_counter() - epoch_started
        last_metrics = quality._epoch_metrics(rows)
        payload = _checkpoint_payload(
            recipe,
            architecture,
            seed=seed,
            model=model,
            algo=algo,
            scheduler=scheduler,
            generator=generator,
            model_identity=identity,
            epoch=epoch,
            items_processed=items_processed,
            cumulative_training_walltime_s=cumulative_training_walltime_s,
            recipe_sha256=recipe_sha256,
            git_sha=git_sha,
            last_epoch_metrics=last_metrics,
        )
        quality._write_training_checkpoint_pair(
            checkpoint_dir,
            latest_path,
            payload,
            epoch=epoch,
            checkpoint_epochs=recipe.training.checkpoint_epochs,
        )
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "stage": "measured-training",
                    "architecture": architecture.architecture_id,
                    "training_seed": seed,
                    "completed_epoch": epoch,
                    "items_processed": items_processed,
                    "optimizer_steps": int(algo._step),
                    "elapsed_s": elapsed,
                    "metrics": last_metrics,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if elapsed > recipe.maximum_seed_walltime_s and epoch < recipe.training.epochs:
            raise TrainingDebtError(
                f"{architecture.architecture_id} seed {seed} exceeded maximum seed walltime"
            )
    torch.cuda.synchronize(recipe.gpu_index)
    records: list[dict[str, Any]] = []
    steps_per_epoch = math.ceil(recipe.training.instances_per_epoch / recipe.training.batch_size)
    for planned_epoch in recipe.training.checkpoint_epochs:
        path = checkpoint_dir / f"epoch-{planned_epoch:03d}.pt"
        checkpoint = quality._load_torch_mapping(path)
        if (
            checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
            or checkpoint.get("completed_epochs") != planned_epoch
            or checkpoint.get("items_processed")
            != planned_epoch * recipe.training.instances_per_epoch
            or checkpoint.get("optimizer_steps") != planned_epoch * steps_per_epoch
            or checkpoint.get("model_identity") != identity
        ):
            raise TrainingDebtError("planned training checkpoint metadata changed")
        records.append(
            {
                "epoch": planned_epoch,
                "path": f"training/checkpoints/epoch-{planned_epoch:03d}.pt",
                "sha256": _sha256_file(path),
                "items_processed": checkpoint["items_processed"],
                "optimizer_steps": checkpoint["optimizer_steps"],
            }
        )
    result = {
        "schema_version": TRAINING_RESULT_SCHEMA,
        "status": "complete",
        "architecture": architecture.architecture_id,
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "training_seed": seed,
        "epochs": recipe.training.epochs,
        "items_processed": items_processed,
        "optimizer_steps": int(algo._step),
        "cumulative_training_walltime_s": cumulative_training_walltime_s,
        "model_identity": identity,
        "checkpoint_records": records,
        "measured_boundary": "model_initialization_through_final_checkpoint_serialization",
        "energy_record": "../energy.json",
        "quality_evaluation_included_in_energy": False,
    }
    _atomic_write_json(staging / "training" / "result.json", result)
    return result


def _make_tracker(label: str, recipe: TrainingDebtRecipe) -> Any:
    from neuro_co.aet import EnergyTracker

    return EnergyTracker(
        label,
        backend="hwcounters",
        pue=1.0,
        report_embodied=False,
        grid_intensity_g_per_kwh=0.0,
        allow_fallback=False,
        required_domains={"cpu", "gpu"},
        gpu_indices=[recipe.gpu_index],
        items=0,
    )


def _checked_energy(
    tracker: Any, recipe: TrainingDebtRecipe, *, items_processed: int
) -> dict[str, Any]:
    try:
        payload = software_runner._checked_energy(
            tracker,
            minimum_duration_s=recipe.minimum_measured_duration_s,
            gpu_index=recipe.gpu_index,
            gpu_device_id_sha256=recipe.gpu_device_id_sha256,
        )
    except Exception as exc:
        raise TrainingDebtError(f"strict training energy record is invalid: {exc}") from exc
    if payload.get("items_processed") != items_processed:
        raise TrainingDebtError("energy record item count differs from training progress")
    for key in ("co2_operational_kg", "co2_embodied_kg", "co2_total_kg"):
        if payload.get(key) not in (None, 0, 0.0):
            raise TrainingDebtError("training energy record contains carbon output")
    cpu = float(payload["energy_cpu_j"])
    gpu = float(payload["energy_gpu_j"])
    observed = cpu + gpu
    if not math.isclose(float(payload["energy_j"]), observed, rel_tol=1e-12, abs_tol=1e-9):
        raise TrainingDebtError("tracker total differs from CPU package plus GPU")
    return {
        "schema_version": ENERGY_SCHEMA,
        "status": "complete",
        "classification": CLASSIFICATION,
        "cpu_package_energy_j": cpu,
        "gpu_energy_j": gpu,
        "observed_component_energy_j": observed,
        "duration_s": float(payload["duration_s"]),
        "items_processed": items_processed,
        "energy_per_training_instance_j": observed / items_processed,
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "raw_tracker_reading": payload,
    }


def _load_quality_source(
    recipe: TrainingDebtRecipe, root: Path
) -> tuple[quality.Corpus, dict[str, Any], str]:
    source = _relative(root, recipe.quality_source_root)
    corpus_path = _relative(source, recipe.corpus.path)
    reference_path = _relative(source, recipe.reference.path)
    if _sha256_file(corpus_path) != recipe.corpus.sha256:
        raise TrainingDebtQualificationError("quality corpus changed")
    if _sha256_file(reference_path) != recipe.reference.sha256:
        raise TrainingDebtQualificationError("quality reference changed")
    try:
        with np.load(corpus_path, allow_pickle=False) as archive:
            coords = np.ascontiguousarray(archive["coords"], dtype=np.float32)
            demands = np.ascontiguousarray(archive["demands"], dtype=np.float32)
            capacity = float(archive["capacity"].item())
            content_sha256 = str(archive["content_sha256"].item())
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise TrainingDebtQualificationError("quality corpus cannot be loaded") from exc
    if (
        coords.shape != (512, 51, 2)
        or demands.shape != (512, 51)
        or capacity != 40.0
        or content_sha256 != recipe.corpus_content_sha256
    ):
        raise TrainingDebtQualificationError("quality corpus identity changed")
    reference = _load_json(reference_path)
    costs = reference.get("costs")
    if not isinstance(costs, list) or len(costs) != 512:
        raise TrainingDebtQualificationError("quality reference cost vector changed")
    return (
        quality.Corpus(
            split="holdout",
            split_id="cvrp50-hgs10-confirmatory-seed2723",
            coords=coords,
            demands=demands,
            capacity=capacity,
            content_sha256=content_sha256,
            file_sha256=recipe.corpus.sha256,
            path=corpus_path,
        ),
        reference,
        recipe.reference.sha256,
    )


def _evaluate_checkpoint(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    *,
    seed: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    expected_identity: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    import torch

    from neuro_co.core.factory import make_env, make_model

    checkpoint = quality._load_torch_mapping(checkpoint_path)
    env = make_env("cvrp", size=50, capacity=40.0, max_demand=9)
    model = make_model(
        env,
        backbone=architecture.backbone,
        hidden_dim=architecture.hidden_dim,
        num_layers=architecture.num_layers,
        num_heads=architecture.num_heads,
    )
    identity = _model_identity(recipe, architecture, model)
    if identity != expected_identity or checkpoint.get("model_identity") != identity:
        raise TrainingDebtError("checkpoint model identity changed before quality evaluation")
    model.load_state_dict(checkpoint["model"], strict=True)
    device = torch.device(f"cuda:{recipe.gpu_index}")
    model.to(device=device, dtype=torch.float32).eval()
    mode = EvaluationMode(
        mode_id=recipe.evaluation.mode_id,
        n_starts=recipe.evaluation.n_starts,
        augmentations=recipe.evaluation.augmentations,
        forced_first_actions=recipe.evaluation.forced_first_actions,
        batch_size=recipe.evaluation.batch_size,
        inference_precision=recipe.evaluation.inference_precision,
    )
    routes: list[Any] = []
    eval_seed = 90_000 + (0 if architecture.architecture_id == "am" else 10_000) + seed * 100
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for start in range(0, corpus.coords.shape[0], mode.batch_size):
            stop = min(start + mode.batch_size, corpus.coords.shape[0])
            state = quality._initial_state(
                env, corpus.coords[start:stop], corpus.demands[start:stop], device
            )
            routes.extend(
                quality._multistart_routes(
                    model,
                    env,
                    state,
                    n_starts=mode.n_starts,
                    augmentations=mode.augmentations,
                    seed=eval_seed + start,
                )
            )
    torch.cuda.synchronize(recipe.gpu_index)
    elapsed = time.perf_counter() - started
    validation = quality.validate_routes(corpus.coords, corpus.demands, corpus.capacity, routes)
    metrics = quality._gap_summary(validation["costs"], reference["costs"])
    return {
        "schema_version": EVALUATION_SCHEMA,
        "status": "complete",
        "classification": {
            "purpose": "post_training_quality_qualification",
            "energy_measurement": "none",
            "included_in_training_energy": False,
        },
        "architecture": architecture.architecture_id,
        "training_seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "model_state_sha256": model_state_sha256(checkpoint["model"]),
        "model_identity": identity,
        "dataset_content_sha256": corpus.content_sha256,
        "reference_sha256": reference_sha256,
        "mode": asdict(mode),
        "evaluation_seed": eval_seed,
        "elapsed_s": elapsed,
        "routes": routes,
        "validation": validation,
        "quality": metrics,
    }


def _cell_id(architecture: str, seed: int) -> str:
    return f"{architecture}:seed-{seed:03d}"


def _cell_relative(architecture: str, seed: int) -> str:
    return f"architectures/{architecture}/seeds/seed-{seed:03d}"


def _expected_cells(recipe: TrainingDebtRecipe) -> tuple[tuple[Architecture, int], ...]:
    return tuple(
        (architecture, seed)
        for architecture in recipe.architectures
        for seed in recipe.training.seeds
    )


def _attestation(recipe: TrainingDebtRecipe) -> dict[str, Any]:
    if os.environ.get("AET_EXPERIMENT_EXCLUSIVE_ATTESTED") != "1":
        raise TrainingDebtQualificationError("EXCLUSIVE attestation is absent")
    raw_time = os.environ.get("AET_EXPERIMENT_EXCLUSIVE_ATTESTED_AT")
    session_id = os.environ.get("AET_EXPERIMENT_EXCLUSIVE_SESSION_ID")
    try:
        timestamp = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
        parsed_session = str(uuid.UUID(str(session_id)))
    except (TypeError, ValueError) as exc:
        raise TrainingDebtQualificationError("EXCLUSIVE attestation metadata is malformed") from exc
    if timestamp.tzinfo is None:
        raise TrainingDebtQualificationError("EXCLUSIVE timestamp lacks a timezone")
    age = (datetime.now(UTC) - timestamp.astimezone(UTC)).total_seconds()
    if age < -5.0 or age > recipe.attestation_max_age_s:
        raise TrainingDebtQualificationError("EXCLUSIVE attestation is outside its age window")
    return {
        "value": "EXCLUSIVE",
        "attested_at": timestamp.astimezone(UTC).isoformat(),
        "session_id": parsed_session,
        "maximum_age_s": recipe.attestation_max_age_s,
        "gpu_process_lists_are_diagnostic_only": True,
    }


def _runtime_identity(recipe: TrainingDebtRecipe) -> dict[str, Any]:
    import platform

    import torch

    versions: dict[str, str | None] = {}
    for name, distribution in {
        "torch": "torch",
        "numpy": "numpy",
        "codecarbon": "codecarbon",
        "pynvml": "nvidia-ml-py",
        "torch_geometric": "torch-geometric",
        "torch_cluster": "torch-cluster",
    }.items():
        try:
            versions[name] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    properties = torch.cuda.get_device_properties(recipe.gpu_index)
    return {
        "platform": sys.platform,
        "platform_release": platform.release(),
        "python": platform.python_version(),
        "versions": versions,
        "gpu_index": recipe.gpu_index,
        "gpu_name": torch.cuda.get_device_name(recipe.gpu_index),
        "gpu_total_memory_b": int(properties.total_memory),
        "gpu_compute_capability": list(torch.cuda.get_device_capability(recipe.gpu_index)),
        "torch_cuda": torch.version.cuda,
    }


def _configure_runtime(recipe: TrainingDebtRecipe) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise TrainingDebtQualificationError(f"{name} must be exactly 1")
    import torch

    if not torch.cuda.is_available() or recipe.gpu_index >= torch.cuda.device_count():
        raise TrainingDebtQualificationError("selected CUDA device is unavailable")
    torch.cuda.set_device(recipe.gpu_index)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(False)
    return _runtime_identity(recipe)


def _source_receipt(recipe: TrainingDebtRecipe, root: Path) -> dict[str, Any]:
    source = _relative(root, recipe.quality_source_root)
    base = _relative(root, recipe.base_recipe.path)
    if _sha256_file(base) != recipe.base_recipe.sha256:
        raise TrainingDebtQualificationError("base recipe changed")
    artifacts: list[dict[str, str]] = []
    for artifact in (
        *recipe.quality_source_artifacts,
        recipe.corpus,
        recipe.reference,
        recipe.reference_lock,
    ):
        path = _relative(source, artifact.path)
        if _sha256_file(path) != artifact.sha256:
            raise TrainingDebtQualificationError(f"quality source changed: {artifact.path}")
        record = {"path": artifact.path, "sha256": artifact.sha256}
        if record not in artifacts:
            artifacts.append(record)
    return {
        "schema_version": SOURCE_RECEIPT_SCHEMA,
        "base_recipe": {
            "source_path": recipe.base_recipe.path,
            "sha256": recipe.base_recipe.sha256,
        },
        "quality_source": {
            "root": recipe.quality_source_root,
            "expected_status": recipe.quality_source_status,
            "artifacts": artifacts,
            "corpus_content_sha256": recipe.corpus_content_sha256,
        },
        "architecture_recipes": {
            item.architecture_id: {
                "sha256": architecture_recipe_sha256(recipe, item),
                "payload": _architecture_recipe_payload(recipe, item),
            }
            for item in recipe.architectures
        },
    }


def _git_snapshot(root: Path, preflight: Path) -> dict[str, Any]:
    snapshot = software_recipe._current_git_snapshot((preflight,))
    if (
        snapshot.get("available") is not True
        or not isinstance(snapshot.get("sha"), str)
        or not isinstance(snapshot.get("worktree_fingerprint_sha256"), str)
    ):
        raise TrainingDebtQualificationError("measured training requires a Git source snapshot")
    return snapshot


def _prepare_output(
    recipe_path: Path,
    recipe: TrainingDebtRecipe,
    root: Path,
    *,
    resume: bool,
    runtime_identity: dict[str, Any],
    source_receipt: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    output = quality._safe_output_target(root, recipe.output_root)
    recipe_bytes = recipe_path.read_bytes()
    recipe_sha = _sha256_bytes(recipe_bytes)
    lock_bytes = (root / "uv.lock").read_bytes()
    lock_sha = _sha256_bytes(lock_bytes)
    base_bytes = _relative(root, recipe.base_recipe.path).read_bytes()
    preflight = _relative(root, recipe.preflight_report)
    git = _git_snapshot(root, preflight)
    source_bytes = _json_bytes(source_receipt)
    source_sha = _sha256_bytes(source_bytes)
    state_path = output / "run-state.json"
    if output.exists():
        if not resume:
            raise TrainingDebtError(f"output already exists; use --resume: {output}")
        state = _load_json(state_path)
        expected = {
            "schema_version": RUN_STATE_SCHEMA,
            "recipe_sha256": recipe_sha,
            "uv_lock_sha256": lock_sha,
            "base_recipe_sha256": recipe.base_recipe.sha256,
            "git_sha": git.get("sha"),
            "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
            "runtime_identity": runtime_identity,
            "source_receipt_sha256": source_sha,
            "classification": CLASSIFICATION,
        }
        if any(state.get(key) != value for key, value in expected.items()):
            raise TrainingDebtError("resume identity differs from initialized training campaign")
        for relative, digest in (
            ("recipe.yaml", recipe_sha),
            ("environment/uv.lock", lock_sha),
            ("base-recipe.yaml", recipe.base_recipe.sha256),
            ("source-receipt.json", source_sha),
        ):
            if _sha256_file(output / relative) != digest:
                raise TrainingDebtError(f"frozen campaign artifact changed: {relative}")
        return output, state

    state = {
        "schema_version": RUN_STATE_SCHEMA,
        "status": INCOMPLETE_STATUS,
        "created_at": datetime.now(UTC).isoformat(),
        "recipe_sha256": recipe_sha,
        "uv_lock_sha256": lock_sha,
        "base_recipe_sha256": recipe.base_recipe.sha256,
        "git_sha": git.get("sha"),
        "worktree_fingerprint_sha256": git.get("worktree_fingerprint_sha256"),
        "runtime_identity": runtime_identity,
        "source_receipt_sha256": source_sha,
        "classification": CLASSIFICATION,
        "completed_cells": [],
        "invocations": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.initializing-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "environment").mkdir()
        (staging / "recipe.yaml").write_bytes(recipe_bytes)
        (staging / "base-recipe.yaml").write_bytes(base_bytes)
        (staging / "environment" / "uv.lock").write_bytes(lock_bytes)
        (staging / "source-receipt.json").write_bytes(source_bytes)
        _atomic_write_json(
            staging / "attempt-ledger.json",
            {
                "schema_version": ATTEMPT_LEDGER_SCHEMA,
                "append_only_semantics": True,
                "events": [],
            },
        )
        _atomic_write_json(staging / "run-state.json", state)
        os.replace(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output, state


def _recover_staging(output: Path) -> None:
    staging = output / "staging"
    if not staging.exists():
        return
    if staging.is_symlink() or not staging.is_dir():
        raise TrainingDebtError("training staging path is unsafe")
    allowed = {"am", "gnn"}
    for architecture_dir in staging.iterdir():
        if (
            architecture_dir.name not in allowed
            or architecture_dir.is_symlink()
            or not architecture_dir.is_dir()
        ):
            raise TrainingDebtError("training staging contains an unknown artifact")
        for seed_dir in architecture_dir.iterdir():
            prefix, separator, raw_seed = seed_dir.name.partition("-")
            if (
                prefix != "seed"
                or separator != "-"
                or len(raw_seed) != 3
                or not raw_seed.isdecimal()
                or seed_dir.is_symlink()
                or not seed_dir.is_dir()
            ):
                raise TrainingDebtError("training staging contains an unsafe seed artifact")
            seed = int(raw_seed)
            energy_path = seed_dir / "energy.json"
            energy: dict[str, Any] | None = None
            if energy_path.is_file():
                try:
                    energy = _load_json(energy_path)
                except quality.QualityPilotError:
                    energy = None
            ledger = _attempt_ledger(output)
            attempt_id, attempt_is_open = _latest_attempt_for_cell(
                ledger, architecture=architecture_dir.name, seed=seed
            )
            if attempt_id is None:
                attempt_id = str(uuid.uuid4())
            _append_attempt_event(
                output,
                {
                    "event": (
                        "interrupted_attempt_recovered"
                        if attempt_is_open
                        else "attempt_staging_discarded"
                    ),
                    "attempt_id": attempt_id,
                    "architecture": architecture_dir.name,
                    "training_seed": seed,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "energy_status": (
                        "complete_but_excluded_from_primary" if energy is not None else "unknown"
                    ),
                    "energy": energy,
                    "energy_sha256": (
                        _sha256_bytes(_json_bytes(energy)) if energy is not None else None
                    ),
                    "primary_recipe_debt_eligible": False,
                    "attempt_start_event_missing": attempt_id not in _attempt_ids(ledger),
                    "reason": "partial seed measurement cannot be combined across attempts",
                },
            )
            shutil.rmtree(seed_dir)
        architecture_dir.rmdir()
    staging.rmdir()


def _attempt_ledger(output: Path) -> dict[str, Any]:
    path = output / "attempt-ledger.json"
    if not path.exists():
        return {
            "schema_version": ATTEMPT_LEDGER_SCHEMA,
            "append_only_semantics": True,
            "events": [],
        }
    payload = _load_json(path)
    events = payload.get("events")
    if (
        payload.get("schema_version") != ATTEMPT_LEDGER_SCHEMA
        or payload.get("append_only_semantics") is not True
        or not isinstance(events, list)
        or any(not isinstance(event, dict) for event in events)
    ):
        raise TrainingDebtError("attempt ledger is malformed")
    return payload


def _append_attempt_event(output: Path, event: dict[str, Any]) -> None:
    payload = _attempt_ledger(output)
    payload["events"].append(event)
    _atomic_write_json(output / "attempt-ledger.json", payload)


def _attempt_ids(payload: Mapping[str, Any]) -> set[str]:
    return {
        str(event["attempt_id"])
        for event in payload.get("events", ())
        if isinstance(event, dict) and isinstance(event.get("attempt_id"), str)
    }


def _latest_attempt_for_cell(
    payload: Mapping[str, Any], *, architecture: str, seed: int
) -> tuple[str | None, bool]:
    starts: list[str] = []
    terminal: set[str] = set()
    for event in payload.get("events", ()):
        if not isinstance(event, dict):
            continue
        attempt_id = event.get("attempt_id")
        if not isinstance(attempt_id, str):
            continue
        event_architecture = event.get("architecture")
        event_seed = event.get("training_seed")
        if event_architecture != architecture or event_seed != seed:
            continue
        if event.get("event") == "attempt_started":
            starts.append(attempt_id)
        elif event.get("event") in {
            "attempt_completed",
            "attempt_failed",
            "interrupted_attempt_recovered",
            "completed_attempt_recovered",
        }:
            terminal.add(attempt_id)
    if not starts:
        return None, False
    latest = starts[-1]
    return latest, latest not in terminal


def _reconcile_attempt_ledger(output: Path, completed: Mapping[str, dict[str, Any]]) -> None:
    payload = _attempt_ledger(output)
    for cell_id, result in completed.items():
        architecture = result.get("architecture")
        seed = result.get("training_seed")
        if not isinstance(architecture, str) or not isinstance(seed, int):
            raise TrainingDebtError(f"completed cell metadata is malformed: {cell_id}")
        attempt_id, is_open = _latest_attempt_for_cell(
            payload, architecture=architecture, seed=seed
        )
        if attempt_id is not None and is_open:
            _append_attempt_event(
                output,
                {
                    "event": "completed_attempt_recovered",
                    "attempt_id": attempt_id,
                    "architecture": result["architecture"],
                    "training_seed": result["training_seed"],
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "energy_status": "complete",
                    "energy_sha256": _sha256_bytes(_json_bytes(result["energy"])),
                    "primary_recipe_debt_eligible": True,
                },
            )


def _attempt_ledger_summary(output: Path) -> dict[str, Any]:
    payload = _attempt_ledger(output)
    events = payload["events"]
    excluded = {
        event["attempt_id"]
        for event in events
        if isinstance(event.get("attempt_id"), str)
        and event.get("event")
        in {"attempt_failed", "interrupted_attempt_recovered", "attempt_staging_discarded"}
    }
    completed = {
        event["attempt_id"]
        for event in events
        if isinstance(event.get("attempt_id"), str)
        and event.get("event") in {"attempt_completed", "completed_attempt_recovered"}
    }
    path = output / "attempt-ledger.json"
    return {
        "path": "attempt-ledger.json",
        "sha256": _sha256_file(path),
        "event_count": len(events),
        "started_attempt_count": sum(event.get("event") == "attempt_started" for event in events),
        "completed_attempt_count": len(completed),
        "failed_or_interrupted_attempt_count": len(excluded),
        "failed_or_interrupted_energy_is_excluded_from_primary": True,
        "primary_recipe_debt_uses_only_completed_attempts": True,
    }


def _preserve_preflight(recipe: TrainingDebtRecipe, root: Path, output: Path) -> dict[str, str]:
    source = _relative(root, recipe.preflight_report)
    digest = _sha256_file(source)
    relative = f"qualification/preflight-{digest}.json"
    target = output / relative
    if target.exists():
        if _sha256_file(target) != digest:
            raise TrainingDebtError("preserved preflight changed")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        quality._atomic_write_bytes(target, source.read_bytes())
    return {"path": relative, "sha256": digest}


def _checkpoint_entry_from_seed_root(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    seed: int,
    seed_root: Path,
) -> dict[str, Any]:
    checkpoint_relative = (
        f"{_cell_relative(architecture.architecture_id, seed)}/"
        f"training/checkpoints/epoch-{recipe.training.epochs:03d}.pt"
    )
    checkpoint_path = (
        seed_root / "training" / "checkpoints" / (f"epoch-{recipe.training.epochs:03d}.pt")
    )
    checkpoint = quality._load_torch_mapping(checkpoint_path)
    identity = checkpoint.get("model_identity")
    if not isinstance(identity, dict):
        raise TrainingDebtError("final checkpoint model identity is missing")
    state_sha = model_state_sha256(checkpoint.get("model", {}))
    return {
        "architecture": architecture.architecture_id,
        "training_seed": seed,
        "path": checkpoint_relative,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": state_sha,
        "model_identity": identity,
        "model_identity_sha256": identity.get("model_identity_sha256"),
        "model_configuration": architecture.model_configuration,
        "backend_identity": identity.get("backend_identity"),
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "training_result_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/training/result.json"
        ),
        "seed_result_path": f"{_cell_relative(architecture.architecture_id, seed)}/seed-result.json",
        "energy_path": f"{_cell_relative(architecture.architecture_id, seed)}/energy.json",
        "quality_evaluation_path": (
            f"{_cell_relative(architecture.architecture_id, seed)}/quality-evaluation.json"
        ),
        "seed_root_sha256": _canonical_sha256(
            {
                path.relative_to(seed_root).as_posix(): _sha256_file(path)
                for path in sorted(seed_root.rglob("*"))
                if path.is_file() and path.name != "seed-result.json"
            }
        ),
    }


def _checkpoint_entry(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    output: Path,
    seed: int,
) -> dict[str, Any]:
    return _checkpoint_entry_from_seed_root(
        recipe,
        architecture,
        seed,
        output / _cell_relative(architecture.architecture_id, seed),
    )


def _load_completed_cell(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    output: Path,
    seed: int,
    *,
    corpus: quality.Corpus,
    reference: dict[str, Any],
) -> dict[str, Any]:
    root = output / _cell_relative(architecture.architecture_id, seed)
    stored = _load_json(root / "seed-result.json")
    entry = _checkpoint_entry(recipe, architecture, output, seed)
    energy = _load_json(root / "energy.json")
    evaluation = _load_json(root / "quality-evaluation.json")
    if (
        stored.get("schema_version") != SEED_RESULT_SCHEMA
        or stored.get("status") != "complete"
        or stored.get("architecture") != architecture.architecture_id
        or stored.get("training_seed") != seed
        or stored.get("checkpoint") != entry
        or stored.get("energy") != energy
        or stored.get("quality_evaluation_sha256") != _sha256_file(root / "quality-evaluation.json")
    ):
        raise TrainingDebtError(
            f"completed cell changed: {_cell_id(architecture.architecture_id, seed)}"
        )
    if (
        energy.get("schema_version") != ENERGY_SCHEMA
        or energy.get("status") != "complete"
        or energy.get("observed_component_energy_j", 0) <= 0
        or energy.get("duration_s", 0) < recipe.minimum_measured_duration_s
    ):
        raise TrainingDebtError("completed training energy record changed")
    validation = quality.validate_routes(
        corpus.coords, corpus.demands, corpus.capacity, evaluation.get("routes", ())
    )
    metrics = quality._gap_summary(validation["costs"], reference["costs"])
    if evaluation.get("validation") != validation or evaluation.get("quality") != metrics:
        raise TrainingDebtError("completed quality evaluation changed")
    return stored


def _completed_prefix(
    recipe: TrainingDebtRecipe,
    output: Path,
    *,
    corpus: quality.Corpus,
    reference: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    missing_seen = False
    for architecture, seed in _expected_cells(recipe):
        cell_id = _cell_id(architecture.architecture_id, seed)
        path = output / _cell_relative(architecture.architecture_id, seed) / "seed-result.json"
        if not path.exists():
            missing_seen = True
            continue
        if missing_seen:
            raise TrainingDebtError("completed training cells are not an ordered prefix")
        completed[cell_id] = _load_completed_cell(
            recipe,
            architecture,
            output,
            seed,
            corpus=corpus,
            reference=reference,
        )
    return completed


def _execute_cell(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    *,
    root: Path,
    output: Path,
    state: dict[str, Any],
    seed: int,
    attestation: dict[str, Any],
    corpus: quality.Corpus,
    reference: dict[str, Any],
    reference_sha256: str,
) -> dict[str, Any]:
    cell_id = _cell_id(architecture.architecture_id, seed)
    staging = output / "staging" / architecture.architecture_id / f"seed-{seed:03d}"
    final = output / _cell_relative(architecture.architecture_id, seed)
    if staging.exists() or final.exists():
        raise TrainingDebtError(f"fresh training target already exists: {cell_id}")
    attempt_id = str(uuid.uuid4())
    _append_attempt_event(
        output,
        {
            "event": "attempt_started",
            "attempt_id": attempt_id,
            "architecture": architecture.architecture_id,
            "training_seed": seed,
            "recorded_at": datetime.now(UTC).isoformat(),
            "attestation_session_id": attestation["session_id"],
            "energy_status": "unknown",
            "primary_recipe_debt_eligible": False,
        },
    )
    try:
        staging.mkdir(parents=True)
        tracker = _make_tracker(f"training-debt-{architecture.architecture_id}-seed{seed}", recipe)
        started_at = datetime.now(UTC)
        wall_started = time.perf_counter()
        with tracker as active:
            training_result = _train_seed_workload(
                recipe,
                architecture,
                seed=seed,
                staging=staging,
                recipe_sha256=state["recipe_sha256"],
                git_sha=state["git_sha"],
            )
            active.n_items = recipe.training.epochs * recipe.training.instances_per_epoch
        ended_at = datetime.now(UTC)
        elapsed = time.perf_counter() - wall_started
        if elapsed > recipe.maximum_seed_walltime_s:
            raise TrainingDebtError(f"{cell_id} exceeded maximum measured seed walltime")
        energy = _checked_energy(
            tracker,
            recipe,
            items_processed=recipe.training.epochs * recipe.training.instances_per_epoch,
        )
        energy.update(
            {
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
                "started_at": started_at.isoformat(),
                "ended_at": ended_at.isoformat(),
                "attestation": attestation,
            }
        )
        _atomic_write_json(staging / "energy.json", energy)
        _append_attempt_event(
            output,
            {
                "event": "attempt_energy_recorded",
                "attempt_id": attempt_id,
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "recorded_at": datetime.now(UTC).isoformat(),
                "energy_status": "complete_pending_seed_promotion",
                "energy_sha256": _sha256_bytes(_json_bytes(energy)),
                "primary_recipe_debt_eligible": False,
            },
        )
        final_record = next(
            item
            for item in training_result["checkpoint_records"]
            if item["epoch"] == recipe.training.epochs
        )
        checkpoint_path = staging / final_record["path"]
        evaluation = _evaluate_checkpoint(
            recipe,
            architecture,
            seed=seed,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=final_record["sha256"],
            expected_identity=training_result["model_identity"],
            corpus=corpus,
            reference=reference,
            reference_sha256=reference_sha256,
        )
        _atomic_write_json(staging / "quality-evaluation.json", evaluation)
        entry = _checkpoint_entry_from_seed_root(recipe, architecture, seed, staging)
        seed_result = {
            "schema_version": SEED_RESULT_SCHEMA,
            "status": "complete",
            "architecture": architecture.architecture_id,
            "training_seed": seed,
            "checkpoint": entry,
            "energy": energy,
            "training_result_sha256": _sha256_file(staging / "training" / "result.json"),
            "quality_evaluation_sha256": _sha256_file(staging / "quality-evaluation.json"),
            "quality": {
                "invalid_instances": evaluation["validation"]["invalid_instance_count"],
                "mean_gap_pct": evaluation["quality"]["mean_gap_pct"],
                "empirical_q95_gap_pct": evaluation["quality"]["p95_gap_pct"],
                "included_in_training_energy": False,
            },
        }
        _atomic_write_json(staging / "seed-result.json", seed_result)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, final)
        completed = _load_completed_cell(
            recipe,
            architecture,
            output,
            seed,
            corpus=corpus,
            reference=reference,
        )
        _append_attempt_event(
            output,
            {
                "event": "attempt_completed",
                "attempt_id": attempt_id,
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "recorded_at": datetime.now(UTC).isoformat(),
                "energy_status": "complete",
                "energy_sha256": _sha256_bytes(_json_bytes(completed["energy"])),
                "primary_recipe_debt_eligible": True,
            },
        )
        return completed
    except BaseException as exc:
        energy_path = staging / "energy.json"
        recorded_energy: dict[str, Any] | None = None
        if energy_path.is_file():
            try:
                recorded_energy = _load_json(energy_path)
            except quality.QualityPilotError:
                recorded_energy = None
        _append_attempt_event(
            output,
            {
                "event": "attempt_failed",
                "attempt_id": attempt_id,
                "architecture": architecture.architecture_id,
                "training_seed": seed,
                "recorded_at": datetime.now(UTC).isoformat(),
                "energy_status": (
                    "complete_but_excluded_from_primary"
                    if recorded_energy is not None
                    else "unknown"
                ),
                "energy": recorded_energy,
                "energy_sha256": (
                    _sha256_bytes(_json_bytes(recorded_energy))
                    if recorded_energy is not None
                    else None
                ),
                "primary_recipe_debt_eligible": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        # Keep the partial attempt for forensic visibility until the next
        # explicit resume.  Resume discards exactly this seed and starts its
        # indivisible measurement again from epoch zero.
        raise


def _bootstrap_samples(
    matrix: np.ndarray, *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(replicates, dtype=np.float64)
    q95 = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        instances = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        seeds = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        selected = matrix[seeds[:, None], instances[None, :]]
        means[index] = selected.mean()
        q95[index] = np.quantile(selected, 0.95, method="linear")
    return means, q95


def _quality_qualification(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    output: Path,
    seed_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    from neuro_co.aet.experiments import hgs_holdout_runner as holdout

    evaluations = [
        _load_json(
            output
            / _cell_relative(architecture.architecture_id, int(result["training_seed"]))
            / "quality-evaluation.json"
        )
        for result in seed_results
    ]
    matrix = np.asarray([item["quality"]["gaps_pct"] for item in evaluations], dtype=np.float64)
    mean_samples, q95_samples = _bootstrap_samples(
        matrix,
        replicates=recipe.quality_gate.bootstrap_replicates,
        seed=recipe.quality_gate.bootstrap_seed
        + (0 if architecture.architecture_id == "am" else 1),
    )
    invalid = sum(int(item["validation"]["invalid_instance_count"]) for item in evaluations)
    summary = holdout._summarize_policy(
        matrix,
        policy_id=f"{architecture.architecture_id}_pomo_epoch40_seeds2_6",
        policy_role="training_debt_checkpoint_qualification",
        invalid_instances=invalid,
        threshold_pct=recipe.quality_gate.threshold_pct,
        t_critical_value=recipe.quality_gate.t_critical_value,
        t_degrees_of_freedom=recipe.quality_gate.t_degrees_of_freedom,
        bootstrap_mean_samples=mean_samples,
        bootstrap_q95_samples=q95_samples,
        bootstrap_quantile=recipe.quality_gate.bootstrap_quantile,
    )
    return {
        "schema_version": QUALITY_SCHEMA,
        "status": "quality_passed" if summary["passed"] else "quality_nonpass",
        "architecture": architecture.architecture_id,
        "checkpoint_count": len(seed_results),
        "dataset_content_sha256": evaluations[0]["dataset_content_sha256"],
        "reference_sha256": evaluations[0]["reference_sha256"],
        "mode": evaluations[0]["mode"],
        "threshold_pct": recipe.quality_gate.threshold_pct,
        "summary": summary,
        "energy_measurement": "none",
        "included_in_training_energy": False,
        "frontier_eligible": bool(summary["passed"]),
    }


def _energy_statistics(values: Sequence[float]) -> dict[str, Any]:
    average = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    half_width = 2.7764451051977987 * sample_sd / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": average,
        "median": statistics.median(values),
        "sample_standard_deviation": sample_sd,
        "minimum": min(values),
        "maximum": max(values),
        "descriptive_t_interval_95": [average - half_width, average + half_width],
        "unit": "joule",
    }


def _architecture_summary(
    recipe: TrainingDebtRecipe,
    architecture: Architecture,
    output: Path,
    results: Sequence[dict[str, Any]],
    quality_qualification: dict[str, Any],
) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    analyzer_seeds: list[dict[str, Any]] = []
    for result in results:
        energy = result["energy"]
        checkpoint = result["checkpoint"]
        per_seed.append(
            {
                "training_seed": result["training_seed"],
                "observed_component_energy_j": energy["observed_component_energy_j"],
                "cpu_package_energy_j": energy["cpu_package_energy_j"],
                "gpu_energy_j": energy["gpu_energy_j"],
                "duration_s": energy["duration_s"],
                "checkpoint_path": checkpoint["path"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "model_state_sha256": checkpoint["model_state_sha256"],
                "model_identity_sha256": checkpoint["model_identity_sha256"],
            }
        )
        analyzer_seeds.append(
            {
                "seed": result["training_seed"],
                "energy_j": energy["observed_component_energy_j"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "model_state_sha256": checkpoint["model_state_sha256"],
                "model_identity_sha256": checkpoint["model_identity_sha256"],
            }
        )
    energies = [float(item["observed_component_energy_j"]) for item in per_seed]
    cpu = [float(item["cpu_package_energy_j"]) for item in per_seed]
    gpu = [float(item["gpu_energy_j"]) for item in per_seed]
    durations = [float(item["duration_s"]) for item in per_seed]
    model_identity = results[0]["checkpoint"]["model_identity"]
    if any(item["checkpoint"]["model_identity"] != model_identity for item in results[1:]):
        raise TrainingDebtError("model identity differs across training seeds")
    quality_path = f"architectures/{architecture.architecture_id}/quality-qualification.json"
    quality_sha256 = _sha256_file(output / quality_path)
    quality_passed = bool(quality_qualification["frontier_eligible"])
    return {
        "status": "complete",
        "architecture": architecture.architecture_id,
        "architecture_recipe_sha256": architecture_recipe_sha256(recipe, architecture),
        "model_configuration": architecture.model_configuration,
        "model_identity": model_identity,
        "model_identity_sha256": model_identity["model_identity_sha256"],
        "backend_identity": model_identity["backend_identity"],
        "seed_count": len(results),
        "training_seeds": [int(item["training_seed"]) for item in per_seed],
        "recipe_training_energy_j_mean": statistics.mean(energies),
        "energy_debt": {
            "definition": "mean energy of one independently trained model under the frozen recipe",
            "recipe_training_energy_j_mean": statistics.mean(energies),
            "observed_component_energy_j": _energy_statistics(energies),
            "cpu_package_energy_j": _energy_statistics(cpu),
            "gpu_energy_j": _energy_statistics(gpu),
        },
        "duration_s": _energy_statistics(durations),
        "study_cost_measured_five_seeds_j_sum": sum(energies),
        "study_cost_is_not_recipe_training_debt": True,
        "per_seed": per_seed,
        "seeds": analyzer_seeds,
        "selection_debt": {
            "status": "unknown",
            "included_in_primary": False,
        },
        "study_debt": {
            "status": "diagnostic",
            "included_in_primary": False,
            "measured_five_seed_sum_j": sum(energies),
        },
        "quality_status": "feasible" if quality_passed else "infeasible",
        "quality_gate": {
            "passed": quality_passed,
            "status": quality_qualification["status"],
            "path": quality_path,
            "sha256": quality_sha256,
        },
        "quality_qualification_path": quality_path,
        "quality_qualification_sha256": quality_sha256,
        "frontier_eligible": quality_passed,
        "energy_frontier_measurement_required": True,
    }


def _write_checksums(output: Path) -> str:
    records: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in {"SHA256SUMS", "run-state.json"} or relative.startswith("staging/"):
            continue
        records.append(f"{_sha256_file(path)}  {relative}")
    quality._atomic_write_bytes(output / "SHA256SUMS", ("\n".join(records) + "\n").encode())
    return _sha256_file(output / "SHA256SUMS")


def _validate_checksums(output: Path) -> None:
    try:
        lines = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrainingDebtError("completed checksum inventory is unavailable") from exc
    records: dict[str, str] = {}
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise TrainingDebtError("completed checksum inventory is malformed")
        digest, relative = line[:64], line[66:]
        if relative in records or len(digest) != 64:
            raise TrainingDebtError("completed checksum inventory contains a duplicate")
        records[relative] = digest
    expected = {
        item.relative_to(output).as_posix()
        for item in output.rglob("*")
        if item.is_file()
        and item.relative_to(output).as_posix() not in {"SHA256SUMS", "run-state.json"}
        and not item.relative_to(output).as_posix().startswith("staging/")
    }
    if set(records) != expected:
        raise TrainingDebtError("completed checksum inventory differs from output files")
    for relative, digest in records.items():
        if _sha256_file(output / relative) != digest:
            raise TrainingDebtError(f"completed artifact changed: {relative}")


def _validate_completed_output(output: Path, state: dict[str, Any]) -> None:
    summary_path = output / "training-energy-summary.json"
    manifest_path = output / "manifest.json"
    checksums_path = output / "SHA256SUMS"
    if (
        state.get("status") != COMPLETE_STATUS
        or state.get("summary_sha256") != _sha256_file(summary_path)
        or state.get("manifest_sha256") != _sha256_file(manifest_path)
        or state.get("checksums_sha256") != _sha256_file(checksums_path)
    ):
        raise TrainingDebtError("completed training-debt anchors changed")
    summary = _load_json(summary_path)
    manifest = _load_json(manifest_path)
    if (
        summary.get("schema_version") != SUMMARY_SCHEMA
        or summary.get("status") != COMPLETE_STATUS
        or manifest.get("schema_version") != MANIFEST_SCHEMA
        or manifest.get("status") != COMPLETE_STATUS
        or manifest.get("training_energy_summary_sha256") != state["summary_sha256"]
        or manifest.get("checkpoint_entries") != summary.get("checkpoint_entries")
        or manifest.get("attempt_audit") != summary.get("attempt_audit")
        or manifest.get("attempt_audit", {}).get("sha256")
        != _sha256_file(output / "attempt-ledger.json")
        or manifest.get("carbon_was_computed") is not False
        or manifest.get("aet_was_computed") is not False
    ):
        raise TrainingDebtError("completed training-debt manifest changed")
    _validate_checksums(output)


def _finalize(
    recipe: TrainingDebtRecipe,
    output: Path,
    state: dict[str, Any],
    completed: Mapping[str, dict[str, Any]],
) -> TrainingDebtResult:
    _reconcile_attempt_ledger(output, completed)
    attempt_audit = _attempt_ledger_summary(output)
    architecture_summaries: dict[str, dict[str, Any]] = {}
    checkpoint_entries: list[dict[str, Any]] = []
    for architecture in recipe.architectures:
        results = [
            completed[_cell_id(architecture.architecture_id, seed)]
            for seed in recipe.training.seeds
        ]
        qualification = _quality_qualification(recipe, architecture, output, results)
        qualification_path = (
            output / "architectures" / architecture.architecture_id / "quality-qualification.json"
        )
        _atomic_write_json(qualification_path, qualification)
        architecture_summaries[architecture.architecture_id] = _architecture_summary(
            recipe, architecture, output, results, qualification
        )
        checkpoint_entries.extend(result["checkpoint"] for result in results)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": CLASSIFICATION,
        "seed_count_per_architecture": len(recipe.training.seeds),
        "measured_training_run_count": len(completed),
        "training_debt_definition": (
            "architecture-specific mean energy of one independently trained model under the frozen recipe"
        ),
        "architectures": architecture_summaries,
        "checkpoint_entries": checkpoint_entries,
        "attempt_audit": attempt_audit,
        "selection_debt": {"status": "not_measured", "included_in_recipe_training_debt": False},
        "study_debt": {
            "status": "measured_for_this_campaign_only",
            "definition": "sum of all five measured training runs per architecture",
            "included_in_recipe_training_debt": False,
        },
        "whole_system_energy": False,
        "carbon_accounting": "none",
        "aet_computed": False,
    }
    _atomic_write_json(output / "training-energy-summary.json", summary)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "status": COMPLETE_STATUS,
        "classification": CLASSIFICATION,
        "recipe_sha256": state["recipe_sha256"],
        "base_recipe_path": "base-recipe.yaml",
        "base_recipe_sha256": state["base_recipe_sha256"],
        "source_receipt_path": "source-receipt.json",
        "source_receipt_sha256": state["source_receipt_sha256"],
        "git_sha": state["git_sha"],
        "runtime_identity": state["runtime_identity"],
        "architectures": architecture_summaries,
        "checkpoint_entries": checkpoint_entries,
        "attempt_audit": attempt_audit,
        "training_energy_summary_path": "training-energy-summary.json",
        "training_energy_summary_sha256": _sha256_file(output / "training-energy-summary.json"),
        "completed_cells": list(completed),
        "preflight_evidence": state["preflight_evidence"],
        "whole_system_energy": False,
        "carbon_was_computed": False,
        "aet_was_computed": False,
    }
    _atomic_write_json(output / "manifest.json", manifest)
    checksums_sha = _write_checksums(output)
    state.update(
        {
            "status": COMPLETE_STATUS,
            "completed_at": datetime.now(UTC).isoformat(),
            "completed_cells": list(completed),
            "remaining_cells": [],
            "summary_sha256": manifest["training_energy_summary_sha256"],
            "manifest_sha256": _sha256_file(output / "manifest.json"),
            "checksums_sha256": checksums_sha,
        }
    )
    if state["invocations"] and state["invocations"][-1].get("ended_at") is None:
        state["invocations"][-1]["ended_at"] = state["completed_at"]
        state["invocations"][-1]["outcome"] = COMPLETE_STATUS
    _atomic_write_json(output / "run-state.json", state)
    return TrainingDebtResult(
        path=output,
        status=COMPLETE_STATUS,
        complete=True,
        completed_cells=tuple(completed),
        remaining_cells=(),
        manifest_path=output / "manifest.json",
        manifest_sha256=state["manifest_sha256"],
    )


def execute_training_debt(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    resume: bool = True,
) -> TrainingDebtResult:
    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
    except OSError as exc:
        raise TrainingDebtQualificationError(f"workspace or recipe unavailable: {exc}") from exc
    if root != Path.cwd().resolve(strict=True):
        raise TrainingDebtQualificationError("workspace_root must be the current repository")
    try:
        source_recipe.relative_to(root)
    except ValueError as exc:
        raise TrainingDebtQualificationError("recipe must be inside the repository") from exc
    try:
        recipe = load_aet_training_debt_recipe(source_recipe)
    except TrainingDebtRecipeValidationError as exc:
        raise TrainingDebtQualificationError(f"training-debt recipe is invalid: {exc}") from exc
    qualification = runtime_qualification(
        recipe, repository_root=root, active_architecture_probe=True
    )
    if not qualification["ready_to_execute"]:
        raise TrainingDebtQualificationError("; ".join(qualification["errors"]))
    attestation = _attestation(recipe)
    runtime_identity = _configure_runtime(recipe)
    source_receipt = _source_receipt(recipe, root)
    corpus, reference, reference_sha = _load_quality_source(recipe, root)
    campaign_started = time.perf_counter()
    output_target = quality._safe_output_target(root, recipe.output_root)
    with quality._output_lock(output_target):
        output, state = _prepare_output(
            source_recipe,
            recipe,
            root,
            resume=resume if output_target.exists() else False,
            runtime_identity=runtime_identity,
            source_receipt=source_receipt,
        )
        if state.get("status") == COMPLETE_STATUS:
            completed = _completed_prefix(recipe, output, corpus=corpus, reference=reference)
            _validate_completed_output(output, state)
            return TrainingDebtResult(
                path=output,
                status=COMPLETE_STATUS,
                complete=True,
                completed_cells=tuple(completed),
                remaining_cells=(),
                manifest_path=output / "manifest.json",
                manifest_sha256=_sha256_file(output / "manifest.json"),
            )
        if state.get("status") != INCOMPLETE_STATUS:
            raise TrainingDebtError("run state has an unknown status")
        _recover_staging(output)
        preflight = _preserve_preflight(recipe, root, output)
        state.setdefault("preflight_evidence", [])
        if preflight not in state["preflight_evidence"]:
            state["preflight_evidence"].append(preflight)
        state["invocations"].append(
            {
                "started_at": datetime.now(UTC).isoformat(),
                "attestation": attestation,
                "preflight_evidence": preflight,
                "ended_at": None,
                "outcome": None,
            }
        )
        _atomic_write_json(output / "run-state.json", state)
        completed = _completed_prefix(recipe, output, corpus=corpus, reference=reference)
        _reconcile_attempt_ledger(output, completed)
        expected_ids = [
            _cell_id(architecture.architecture_id, seed)
            for architecture, seed in _expected_cells(recipe)
        ]
        for architecture, seed in _expected_cells(recipe):
            cell_id = _cell_id(architecture.architecture_id, seed)
            if cell_id in completed:
                continue
            if time.perf_counter() - campaign_started > recipe.maximum_campaign_walltime_s:
                raise TrainingDebtError("campaign exceeded its maximum walltime")
            completed[cell_id] = _execute_cell(
                recipe,
                architecture,
                root=root,
                output=output,
                state=state,
                seed=seed,
                attestation=attestation,
                corpus=corpus,
                reference=reference,
                reference_sha256=reference_sha,
            )
            state["completed_cells"] = list(completed)
            state["remaining_cells"] = [item for item in expected_ids if item not in completed]
            _atomic_write_json(output / "run-state.json", state)
        return _finalize(recipe, output, state, completed)


def estimate_training_debt(
    recipe_path: str | Path,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run an unmeasured CUDA training-step probe and estimate remaining time."""

    try:
        root = (Path.cwd() if workspace_root is None else Path(workspace_root)).resolve(strict=True)
        source_recipe = Path(recipe_path).resolve(strict=True)
        source_recipe.relative_to(root)
    except (OSError, ValueError) as exc:
        raise TrainingDebtQualificationError(f"workspace or recipe unavailable: {exc}") from exc
    try:
        recipe = load_aet_training_debt_recipe(source_recipe)
    except TrainingDebtRecipeValidationError as exc:
        raise TrainingDebtQualificationError(f"training-debt recipe is invalid: {exc}") from exc
    qualification = runtime_qualification(
        recipe, repository_root=root, active_architecture_probe=True
    )
    if not qualification["ready_to_execute"]:
        raise TrainingDebtQualificationError("; ".join(qualification["errors"]))
    output = quality._safe_output_target(root, recipe.output_root)
    expected = _expected_cells(recipe)
    present: set[str] = set()
    missing_seen = False
    for architecture, seed in expected:
        cell_id = _cell_id(architecture.architecture_id, seed)
        path = output / _cell_relative(architecture.architecture_id, seed) / "seed-result.json"
        if path.is_file():
            if missing_seen:
                raise TrainingDebtError("completed training cells are not an ordered prefix")
            present.add(cell_id)
        else:
            missing_seen = True

    checks = qualification["architecture_checks"]
    architecture_estimates: dict[str, dict[str, Any]] = {}
    remaining_point = 0.0
    remaining_lower = 0.0
    remaining_upper = 0.0
    for architecture in recipe.architectures:
        completed = sum(
            _cell_id(architecture.architecture_id, seed) in present
            for seed in recipe.training.seeds
        )
        remaining = len(recipe.training.seeds) - completed
        point = float(
            checks[architecture.architecture_id]["estimated_seed_walltime_s_from_am_anchor"]
        )
        if architecture.architecture_id == "am":
            lower, upper = 3506.0, 3553.0
            point = 3526.0
            basis = "observed prior AM seed range on this host"
        else:
            lower, upper = point * 0.75, point * 1.5
            basis = "three-step CUDA GNN/AM ratio applied to the observed AM host anchor"
        architecture_estimates[architecture.architecture_id] = {
            "completed_seed_count": completed,
            "remaining_seed_count": remaining,
            "seconds_per_training_seed_point": point,
            "seconds_per_training_seed_lower": lower,
            "seconds_per_training_seed_upper": upper,
            "estimation_basis": basis,
            "median_optimizer_step_s": checks[architecture.architecture_id][
                "median_optimizer_step_s"
            ],
            "parameter_count": checks[architecture.architecture_id]["parameter_count"],
            "backend_metadata": checks[architecture.architecture_id]["backend_metadata"],
        }
        remaining_point += remaining * point
        remaining_lower += remaining * lower
        remaining_upper += remaining * upper
    return {
        "schema_version": ESTIMATE_SCHEMA,
        "status": "ready",
        "generated_at": datetime.now(UTC).isoformat(),
        "classification": {
            "energy_measurement": "none",
            "included_in_training_debt": False,
            "purpose": "pre_attestation_eta_and_architecture_qualification",
        },
        "cuda_probe": {
            "real_forward_backward": True,
            "warmup_steps_per_architecture": 1,
            "timed_steps_per_architecture": recipe.eta_calibration_steps,
            "selected_gpu_index": recipe.gpu_index,
            "architecture_qualification": checks,
        },
        "architectures": architecture_estimates,
        "completed_cells": sorted(present),
        "remaining_training_walltime_s_point": remaining_point,
        "remaining_training_walltime_s_lower": remaining_lower,
        "remaining_training_walltime_s_upper": remaining_upper,
        "maximum_campaign_walltime_s": recipe.maximum_campaign_walltime_s,
        "estimate_is_not_an_energy_measurement": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true")
    group.add_argument("--no-resume", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--estimate-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.estimate_output is not None and not args.estimate_only:
            raise TrainingDebtQualificationError("--estimate-output requires --estimate-only")
        if args.estimate_only:
            estimate = estimate_training_debt(args.recipe)
            if args.estimate_output is not None:
                root = Path.cwd().resolve(strict=True)
                destination = args.estimate_output.resolve(strict=False)
                try:
                    destination.relative_to(root)
                except ValueError as exc:
                    raise TrainingDebtQualificationError(
                        "--estimate-output must remain inside the repository"
                    ) from exc
                _atomic_write_json(destination, estimate)
            print(json.dumps(estimate, indent=2, sort_keys=True))
            return 0
        result = execute_training_debt(args.recipe, resume=args.resume or not args.no_resume)
    except TrainingDebtQualificationError as exc:
        print(f"training debt not qualified: {exc}", file=sys.stderr, flush=True)
        return 2
    except (TrainingDebtError, quality.QualityPilotError) as exc:
        print(f"training debt failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(
        json.dumps(
            {
                "status": result.status,
                "complete": result.complete,
                "path": result.path.as_posix(),
                "completed_cells": list(result.completed_cells),
                "remaining_cells": list(result.remaining_cells),
                "manifest": result.manifest_path.as_posix() if result.manifest_path else None,
                "manifest_sha256": result.manifest_sha256,
                "classification": CLASSIFICATION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TrainingDebtError",
    "TrainingDebtQualificationError",
    "TrainingDebtResult",
    "architecture_recipe_sha256",
    "estimate_training_debt",
    "execute_training_debt",
    "model_state_sha256",
]
