"""End-to-end explainer driver (core).

Takes a built `ConstructivePolicy` + `Env` (dependency injection: no config
framework), samples a batch, runs an attribution method, computes
deletion / sufficiency / sanity faithfulness, optionally fits encoder
concept probes, and writes the result to JSON (+ the instance State).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from neuro_co.attr.attribution import (
    contrastive_attribution,
    gradient_attribution,
    integrated_gradients,
)
from neuro_co.attr.faithfulness import (
    deletion_flip_rate,
    sanity_check,
    sufficiency_keep_rate,
)
from neuro_co.probe.probes import fit_concept_probes

log = logging.getLogger(__name__)

_INSTANCE_FIELDS = (
    "coords",
    "demand",
    "tw_early",
    "tw_late",
    "prize",
    "proc_times",
    "ops_ma_adj",
)


def explain_policy(
    model: Any,
    env: Any,
    *,
    concepts: Mapping[str, Any] | None = None,
    ckpt_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    num_instances: int = 8,
    top_k: int = 5,
    max_steps: int | None = None,
    method: str = "gradient",
    ig_steps: int = 20,
    ig_baseline: str = "zero",
    run_sufficiency: bool = True,
    run_sanity_check: bool = False,
    sanity_trials: int = 1,
    run_encoder_probes: bool = False,
    probe_epochs: int = 200,
    probe_val_frac: float = 0.3,
    probe_lr: float = 1e-2,
    seed: int = 42,
) -> dict[str, Any]:
    """Attribute + evaluate faithfulness for `model` on `env`, write JSON.

    Parameters
    ----------
    model, env
        A built core `ConstructivePolicy` and its `Env`.
    ckpt_path
        Optional state-dict checkpoint to load into `model`.
    method
        `"gradient"`, `"ig"`, or `"contrastive"`.
    """
    output_dir = Path(output_dir or "./outputs/explain")
    output_dir.mkdir(parents=True, exist_ok=True)

    if ckpt_path is not None:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log.warning("missing keys: %s", missing[:5])
        if unexpected:
            log.warning("unexpected keys: %s", unexpected[:5])

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model = model.to(device).eval()
    state = env.reset(num_instances, generator=torch.Generator().manual_seed(seed)).to(device)

    method_l = method.lower()
    if method_l in {"gradient", "grad", "grad_x_input"}:
        trace = gradient_attribution(model, env, state, top_k=top_k, max_steps=max_steps)
    elif method_l in {"ig", "integrated_gradients"}:
        trace = integrated_gradients(
            model,
            env,
            state,
            top_k=top_k,
            max_steps=max_steps,
            ig_steps=ig_steps,
            baseline=ig_baseline,
        )
    elif method_l in {"contrastive", "contrast"}:
        trace = contrastive_attribution(model, env, state, top_k=top_k, max_steps=max_steps)
    else:
        raise ValueError(
            f"unknown attribution method {method!r}; use 'gradient', 'ig', or 'contrastive'"
        )

    deletion = deletion_flip_rate(trace, model, env, state, top_k=top_k)
    suff = sufficiency_keep_rate(trace, model, env, state, top_k=top_k) if run_sufficiency else None
    sanity = (
        sanity_check(trace, model, env, state, top_k=top_k, num_trials=sanity_trials)
        if run_sanity_check
        else None
    )

    payload: dict[str, Any] = {
        "ckpt_path": str(ckpt_path) if ckpt_path else None,
        "num_instances": trace.batch_size,
        "num_steps": trace.num_steps,
        "top_k": top_k,
        "method": method_l,
        "method_params": (
            {"ig_steps": ig_steps, "ig_baseline": ig_baseline}
            if method_l in {"ig", "integrated_gradients"}
            else {}
        ),
        "attribution": {
            "actions": trace.actions.tolist(),
            "log_probs": trace.log_probs.tolist(),
            "top_k_nodes": trace.top_k_nodes.tolist(),
            "top_k_scores": trace.top_k_scores.tolist(),
        },
        "faithfulness": {
            "deletion": {
                "mean_flip_rate": deletion.mean_flip_rate,
                "per_step_flip_rate": deletion.per_step_flip_rate,
                "top_k_used": deletion.top_k_used,
                "num_steps": deletion.num_steps,
                "num_instances": deletion.num_instances,
            },
        },
    }
    if suff is not None:
        payload["faithfulness"]["sufficiency"] = {
            "mean_keep_rate": suff.mean_keep_rate,
            "per_step_keep_rate": suff.per_step_keep_rate,
            "top_k_used": suff.top_k_used,
            "num_steps": suff.num_steps,
        }
    if sanity is not None:
        payload["faithfulness"]["sanity_check"] = {
            "mode": sanity.mode,
            "mean_jaccard": sanity.mean_jaccard,
            "chance_jaccard": sanity.chance_jaccard,
            "num_trials": sanity.num_trials,
            "top_k_used": sanity.top_k_used,
        }

    if run_encoder_probes:
        if not concepts:
            raise ValueError(
                "run_encoder_probes=True requires `concepts`; pass a "
                "dict[str, ConceptFn] (e.g. "
                "`neuro_co.core.concepts.concept_registry.get('<problem>').concepts`)."
            )
        probes = fit_concept_probes(
            model,
            env,
            state,
            concepts=concepts,
            val_frac=probe_val_frac,
            epochs=probe_epochs,
            lr=probe_lr,
            seed=seed,
        )
        payload["encoder_probes"] = [
            {
                "concept": r.concept,
                "train_acc": r.train_acc,
                "val_acc": r.val_acc,
                "val_balanced_acc": r.val_balanced_acc,
                "val_positive_fraction": r.val_positive_fraction,
                "n_train": r.n_train,
                "n_val": r.n_val,
            }
            for r in probes
        ]

    (output_dir / "explanation.json").write_text(json.dumps(payload, indent=2))
    instance_payload = {
        f: getattr(state, f).detach().cpu() for f in _INSTANCE_FIELDS if hasattr(state, f)
    }
    if instance_payload:
        torch.save(instance_payload, output_dir / "instances.pt")
    log.info(
        "wrote explanation: method=%s, num_steps=%d, flip=%.3f%s",
        method_l,
        trace.num_steps,
        deletion.mean_flip_rate,
        f", keep={suff.mean_keep_rate:.3f}" if suff is not None else "",
    )
    return payload
