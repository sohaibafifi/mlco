"""Fit linear concept probes to encoder activations.

Callers supply binary concept functions through `neuro_co.core.concepts`.
The probe fits on node embeddings and reports held-out accuracy, balanced
accuracy, F1, and other diagnostics. Interpret accuracy alongside class balance
and the validation split.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from neuro_co.core.concepts import ConceptFn
from neuro_co.core.trace import layer_activations


@dataclass
class ProbeResult:
    """Outcome of fitting a single binary probe.

    Beyond raw accuracy we report F1, ROC-AUC, precision, recall and the
    confusion matrix on the held-out split, plus the raw `(y_true,
    y_score)` arrays so downstream code can re-plot ROC / PR curves
    without re-running the probe.
    """

    concept: str
    n_train: int
    n_val: int
    train_acc: float
    val_acc: float
    val_balanced_acc: float
    val_positive_fraction: float
    val_f1: float
    val_roc_auc: float
    val_precision: float
    val_recall: float
    confusion_matrix: list[list[int]]
    val_y_true: list[int]
    val_y_score: list[float]
    layer: int = -1  # -1 = final encoder output; >=0 = intermediate layer index


def _encode(policy: Any, features: Tensor) -> Tensor:
    """Encode features `[B, N, d_in]` → per-node embeddings `[B, N, D]`.

    Core `ConstructivePolicy.encode` returns `(node_embs, graph_emb)`;
    we keep the node embeddings.
    """
    node_embs, _ = policy.encode(features)
    return node_embs


def _fit_linear_probe(
    X: Tensor,
    y: Tensor,
    *,
    val_frac: float = 0.3,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 0,
) -> dict[str, Any]:
    """Train a 2-class linear probe with Adam + cross-entropy.

    Returns a dict with train_acc, val_acc, val_balanced_acc,
    val_positive_fraction, n_train, n_val, val_f1, val_roc_auc,
    val_precision, val_recall, confusion_matrix, val_y_true,
    val_y_score.
    """
    device = X.device
    n = X.shape[0]
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx = perm[:n_val].to(device)
    train_idx = perm[n_val:].to(device)

    probe = nn.Linear(X.shape[-1], 2).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(X[train_idx])
        loss = loss_fn(logits, y[train_idx])
        loss.backward()
        opt.step()

    with torch.no_grad():
        val_logits = probe(X[val_idx])
        val_proba = torch.softmax(val_logits, dim=-1)[:, 1]
        train_pred = probe(X[train_idx]).argmax(-1)
        val_pred = val_logits.argmax(-1)
        train_acc = (train_pred == y[train_idx]).float().mean().item()
        val_acc = (val_pred == y[val_idx]).float().mean().item()
        bal = 0.0
        for cls in (0, 1):
            mask = y[val_idx] == cls
            if mask.any():
                bal += (val_pred[mask] == cls).float().mean().item() / 2
        val_pos_frac = float((y[val_idx] == 1).float().mean().item())

    # Confusion matrix [[TN, FP], [FN, TP]] + precision/recall/F1/AUC.
    y_true_np = y[val_idx].cpu().numpy()
    y_pred_np = val_pred.cpu().numpy()
    y_score_np = val_proba.cpu().numpy()
    try:
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).tolist()
        f1 = float(f1_score(y_true_np, y_pred_np, zero_division=0))
        prec = float(precision_score(y_true_np, y_pred_np, zero_division=0))
        rec = float(recall_score(y_true_np, y_pred_np, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true_np, y_score_np))
        except ValueError:
            # Only one class present in y_true → AUC undefined.
            auc = float("nan")
    except ImportError:  # pragma: no cover
        cm = [[0, 0], [0, 0]]
        f1 = prec = rec = auc = float("nan")

    return {
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "val_balanced_acc": float(bal),
        "val_positive_fraction": val_pos_frac,
        "n_train": float(train_idx.numel()),
        "n_val": float(val_idx.numel()),
        "val_f1": f1,
        "val_roc_auc": auc,
        "val_precision": prec,
        "val_recall": rec,
        "confusion_matrix": cm,
        "val_y_true": y_true_np.astype(int).tolist(),
        "val_y_score": y_score_np.astype(float).tolist(),
    }


def encoder_layer_count(policy: Any) -> int:
    """Number of intermediate encoder layers exposed by the policy.

    Returns 0 when the encoder structure is opaque (e.g. SSM).
    """
    enc = getattr(policy, "encoder", None)
    mods = getattr(enc, "blocks", None) or getattr(enc, "layers", None)
    return len(mods) if mods is not None else 0


def fit_concept_probes(
    policy: Any,
    env: Any,
    state: Any,
    *,
    concepts: Mapping[str, ConceptFn],
    val_frac: float = 0.3,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 0,
    layer_indices: list[int] | None = None,
) -> list[ProbeResult]:
    """Encode `state` once, then fit a linear probe per (concept, layer).

    `policy` is a core `ConstructivePolicy`, `env` a core `Env`, `state` a
    core `State`. `concepts` maps `name -> ConceptFn`; get a problem bank
    from `neuro_co.core.concepts.concept_registry`.

    `layer_indices=None` probes the final encoder output only. Passing
    e.g. `[0, 1, 2]` also probes each intermediate encoder layer; results
    carry the index in `ProbeResult.layer`. Concepts returning `None`
    (state lacks their fields) are skipped.
    """
    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    features = env.build_features(state)

    with torch.no_grad():
        final_emb = _encode(policy, features)  # [B, N, D]

    layered: list[tuple[int, Tensor]] = [(-1, final_emb)]
    if layer_indices:
        acts = layer_activations(policy.encoder, features)
        for i in layer_indices:
            if 0 <= i < len(acts):
                layered.append((i, acts[i]))

    results: list[ProbeResult] = []
    for layer_idx, embeddings in layered:
        for name, fn in concepts.items():
            labels = fn(state)
            if labels is None:
                continue
            labels = labels.to(device)
            flat_emb = embeddings.reshape(-1, embeddings.shape[-1])
            flat_lab = labels.reshape(-1)
            keep = flat_lab >= 0
            X = flat_emb[keep]
            y = flat_lab[keep]
            if y.numel() < 4 or len(y.unique()) < 2:
                continue
            stats = _fit_linear_probe(X, y, val_frac=val_frac, epochs=epochs, lr=lr, seed=seed)
            results.append(
                ProbeResult(
                    concept=name,
                    layer=layer_idx,
                    n_train=int(stats["n_train"]),
                    n_val=int(stats["n_val"]),
                    train_acc=stats["train_acc"],
                    val_acc=stats["val_acc"],
                    val_balanced_acc=stats["val_balanced_acc"],
                    val_positive_fraction=stats["val_positive_fraction"],
                    val_f1=stats["val_f1"],
                    val_roc_auc=stats["val_roc_auc"],
                    val_precision=stats["val_precision"],
                    val_recall=stats["val_recall"],
                    confusion_matrix=stats["confusion_matrix"],
                    val_y_true=stats["val_y_true"],
                    val_y_score=stats["val_y_score"],
                )
            )
    return results
