"""Matplotlib visualizations for an `AttributionTrace`.

Two helpers ship with the minimal xai surface:

- `plot_route_attribution(...)`: single instance, depot + customers,
  the chosen route as arrows, and the top-k attributed nodes per step
  highlighted (colored by aggregate attribution mass).
- `plot_flip_rate(...)`: deletion-faithfulness flip rate as a function
  of decoding step.

Both functions accept either an `AttributionTrace` (in-memory) or the
JSON / `instances.pt` artefacts written by `explain_policy`.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "matplotlib is required for neuro_co.xai.viz. "
        "Install with: uv pip install 'neuro-co-xai[plot]'"
    ) from e


def plot_route_attribution(
    locs: np.ndarray,
    actions: Sequence[int],
    top_k_nodes: np.ndarray,
    top_k_scores: np.ndarray,
    *,
    step: int | None = None,
    depot_index: int = 0,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot a single instance's route + top-k attribution overlay.

    Parameters
    ----------
    locs
        `[N, 2]` array of node coordinates (depot at `depot_index`).
    actions
        Length-T sequence of chosen node indices in this rollout.
    top_k_nodes
        `[T, K]` matrix of top-k attributed nodes per step.
    top_k_scores
        `[T, K]` matrix of their attribution scores.
    step
        If given, highlight only the top-k for that step; otherwise
        aggregate scores over all steps.
    depot_index
        Index of the depot node in `locs` (default 0).
    ax
        Optional matplotlib `Axes`; if None, a new figure is created.
    title
        Optional plot title.
    """
    locs = np.asarray(locs)
    top_k_nodes = np.asarray(top_k_nodes)
    top_k_scores = np.asarray(top_k_scores)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    n = locs.shape[0]
    is_depot = np.zeros(n, dtype=bool)
    is_depot[depot_index] = True

    # Aggregate attribution mass per node (over the chosen step window).
    score_per_node = np.zeros(n)
    if step is None:
        for t in range(top_k_nodes.shape[0]):
            for k in range(top_k_nodes.shape[1]):
                idx = int(top_k_nodes[t, k])
                if 0 <= idx < n:
                    score_per_node[idx] += float(top_k_scores[t, k])
        highlight = set(int(i) for i in top_k_nodes.reshape(-1) if 0 <= int(i) < n)
    else:
        s = int(step)
        for k in range(top_k_nodes.shape[1]):
            idx = int(top_k_nodes[s, k])
            if 0 <= idx < n:
                score_per_node[idx] += float(top_k_scores[s, k])
        highlight = {int(i) for i in top_k_nodes[s].tolist() if 0 <= int(i) < n}

    # Route as arrows between consecutive distinct nodes.
    path = [int(a) for a in actions]
    cur = depot_index
    for nxt in path:
        if 0 <= nxt < n and nxt != cur:
            ax.annotate(
                "",
                xy=locs[nxt],
                xytext=locs[cur],
                arrowprops=dict(arrowstyle="->", color="#888", lw=0.6, alpha=0.6),
            )
        cur = nxt

    # Customers: color by aggregate attribution mass.
    customer_mask = ~is_depot
    cust_locs = locs[customer_mask]
    cust_scores = score_per_node[customer_mask]
    sc = ax.scatter(
        cust_locs[:, 0],
        cust_locs[:, 1],
        c=cust_scores,
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.4,
        zorder=2,
    )
    if highlight:
        h = np.array(sorted(highlight))
        h = h[~is_depot[h]]
        if h.size:
            ax.scatter(
                locs[h, 0],
                locs[h, 1],
                s=180,
                facecolors="none",
                edgecolor="red",
                linewidth=1.5,
                zorder=3,
                label=f"top-k @ step {step}" if step is not None else "top-k union",
            )
    # Depot.
    ax.scatter(
        locs[depot_index, 0],
        locs[depot_index, 1],
        marker="s",
        s=120,
        color="black",
        zorder=4,
        label="depot",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    if title is None:
        title = (
            f"Route + top-k attribution (step {step})"
            if step is not None
            else "Route + aggregate top-k attribution"
        )
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="attribution mass")
    return ax


def plot_flip_rate(
    per_step: Sequence[float],
    *,
    ax: Axes | None = None,
    title: str = "Deletion faithfulness: flip rate per step",
) -> Axes:
    """Plot the deletion-faithfulness flip rate as a function of step."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5))
    arr = np.asarray(per_step, dtype=float)
    steps = np.arange(arr.size)
    ax.plot(steps, arr, marker="o", linewidth=1.2, markersize=3, color="#1f77b4")
    if arr.size:
        ax.axhline(
            float(arr.mean()),
            color="red",
            linestyle="--",
            linewidth=0.8,
            label=f"mean={arr.mean():.3f}",
        )
        ax.legend(fontsize=8)
    ax.set_xlabel("decoding step")
    ax.set_ylabel("action flip rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    return ax


def plot_layer_concept_heatmap(
    probe_results: Sequence[Any],
    *,
    metric: str = "val_acc",
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of probe `metric` over (layer, concept).

    Accepts list of `ProbeResult` (or dicts with the same fields).
    Layer `-1` (final encoder output) is rendered as the last row.
    """

    def _get(r: Any, k: str) -> Any:
        return r[k] if isinstance(r, dict) else getattr(r, k)

    if not probe_results:
        raise ValueError("probe_results is empty")
    layers = sorted({int(_get(r, "layer")) for r in probe_results}, key=lambda x: (x == -1, x))
    concepts = sorted({_get(r, "concept") for r in probe_results})
    matrix = np.full((len(layers), len(concepts)), np.nan, dtype=float)
    for r in probe_results:
        i = layers.index(int(_get(r, "layer")))
        j = concepts.index(_get(r, "concept"))
        matrix[i, j] = float(_get(r, metric))
    if ax is None:
        h = max(3.0, 0.45 * len(layers))
        w = max(5.0, 1.3 * len(concepts) + 2.0)
        _, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([f"layer {ll}" if ll != -1 else "final" for ll in layers])
    ax.set_xlabel("concept")
    ax.set_ylabel("encoder layer")
    ax.set_title(f"Probe {metric} per (layer, concept)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black" if matrix[i, j] > 0.75 else "white",
                )
    plt.colorbar(im, ax=ax, label=metric)
    return ax


def plot_probe_metrics_bars(
    probe_results: Sequence[Any],
    *,
    ax: Axes | None = None,
    metrics: Sequence[str] = ("val_acc", "val_balanced_acc", "val_f1", "val_roc_auc"),
) -> Axes:
    """Grouped bar chart: one bar group per probe, one bar per metric."""
    if not probe_results:
        raise ValueError("probe_results is empty")
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.5 * len(probe_results)), 4))
    concepts = [p["concept"] if isinstance(p, dict) else p.concept for p in probe_results]
    x = np.arange(len(concepts))
    width = 0.8 / max(1, len(metrics))
    for i, m in enumerate(metrics):
        values = [(p[m] if isinstance(p, dict) else getattr(p, m)) for p in probe_results]
        offset = (i - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(concepts, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.6, label="chance")
    ax.set_ylabel("score")
    ax.set_title("Probe metrics per concept")
    ax.legend(fontsize=8, ncol=2)
    return ax


def plot_probe_roc(
    probe_results: Sequence[Any],
    *,
    ax: Axes | None = None,
) -> Axes:
    """One ROC curve per probe on the same axes."""
    from sklearn.metrics import roc_curve

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    for p in probe_results:
        y = p["val_y_true"] if isinstance(p, dict) else p.val_y_true
        s = p["val_y_score"] if isinstance(p, dict) else p.val_y_score
        auc = p["val_roc_auc"] if isinstance(p, dict) else p.val_roc_auc
        concept = p["concept"] if isinstance(p, dict) else p.concept
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, linewidth=1.4, label=f"{concept} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=0.6, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Probe ROC curves")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="lower right")
    return ax


def plot_pca_embeddings(
    pca_scores: np.ndarray,
    labels: np.ndarray,
    *,
    concept: str = "concept",
    components: tuple[int, int] = (0, 1),
    ax: Axes | None = None,
) -> Axes:
    """2D scatter of PCA-projected embeddings colored by binary concept label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))
    i, j = components
    if pca_scores.shape[1] <= max(i, j):
        raise ValueError(
            f"requested PCA components {components} but only {pca_scores.shape[1]} available"
        )
    mask_pos = labels == 1
    mask_neg = labels == 0
    ax.scatter(
        pca_scores[mask_neg, i],
        pca_scores[mask_neg, j],
        s=18,
        alpha=0.6,
        color="#4477aa",
        label="0",
    )
    ax.scatter(
        pca_scores[mask_pos, i],
        pca_scores[mask_pos, j],
        s=18,
        alpha=0.7,
        color="#ee6677",
        label="1",
    )
    ax.set_xlabel(f"PC{i + 1}")
    ax.set_ylabel(f"PC{j + 1}")
    ax.set_title(f"PCA - colored by `{concept}`")
    ax.legend(fontsize=8, title="label")
    return ax


def plot_direction_concept_heatmap(
    alignment: Sequence[dict[str, Any]] | Sequence[Any],
    *,
    method_label: str = "PCA",
    ax: Axes | None = None,
) -> Axes:
    """Heatmap of Pearson r per (direction, concept).

    Accepts either dicts (with keys `direction_index`,
    `pearson_per_concept`) or `DirectionAlignment` instances.
    """

    def _get(a: Any, k: str) -> Any:
        return a[k] if isinstance(a, dict) else getattr(a, k)

    if not alignment:
        raise ValueError("alignment is empty")
    concepts = sorted({c for a in alignment for c in _get(a, "pearson_per_concept")})
    matrix = np.array(
        [[float(_get(a, "pearson_per_concept").get(c, 0.0)) for c in concepts] for a in alignment]
    )
    if ax is None:
        h = max(3.0, 0.32 * len(alignment))
        w = max(5.0, 1.1 * len(concepts) + 2.0)
        _, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(alignment)))
    ax.set_yticklabels([f"{method_label}{int(_get(a, 'direction_index'))}" for a in alignment])
    ax.set_xlabel("concept")
    ax.set_ylabel("direction")
    ax.set_title(f"{method_label} direction-concept Pearson r")
    # Annotate cells with numeric r.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black" if abs(matrix[i, j]) < 0.6 else "white",
            )
    plt.colorbar(im, ax=ax, label="r")
    return ax


def plot_scatter_2d_by_label(
    coords: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    ax: Axes | None = None,
) -> Axes:
    """Generic 2D scatter colored by binary `{0, 1}` label."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))
    mask_pos = labels == 1
    mask_neg = labels == 0
    ax.scatter(
        coords[mask_neg, 0],
        coords[mask_neg, 1],
        s=18,
        alpha=0.6,
        color="#4477aa",
        label="0",
    )
    ax.scatter(
        coords[mask_pos, 0],
        coords[mask_pos, 1],
        s=18,
        alpha=0.7,
        color="#ee6677",
        label="1",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, title="label")
    return ax


def plot_tsne_embeddings(
    tsne_coords: np.ndarray,
    labels: np.ndarray,
    *,
    concept: str,
    ax: Axes | None = None,
) -> Axes:
    """2D t-SNE scatter colored by binary concept label.

    `tsne_coords` is the output of `neuro_co.xai.discovered.fit_tsne`.
    """
    return plot_scatter_2d_by_label(
        tsne_coords,
        labels,
        title=f"t-SNE - colored by `{concept}`",
        xlabel="t-SNE 1",
        ylabel="t-SNE 2",
        ax=ax,
    )


def plot_explained_variance(
    explained_variance_ratio: Sequence[float],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Bar chart of per-direction explained variance + cumulative line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))
    arr = np.asarray(explained_variance_ratio, dtype=float)
    x = np.arange(arr.size)
    ax.bar(x, arr, color="#4477aa", label="per-direction")
    ax2 = ax.twinx()
    ax2.plot(x, np.cumsum(arr), color="#ee6677", marker="o", label="cumulative")
    ax.set_xlabel("PCA direction index")
    ax.set_ylabel("explained variance ratio")
    ax2.set_ylabel("cumulative")
    ax2.set_ylim(0.0, 1.05)
    ax.set_title("PCA explained variance")
    return ax


def plot_feature_attribution_bars(
    feature_keys: Sequence[str],
    feature_scores: dict[str, float],
    *,
    ax: Axes | None = None,
) -> Axes:
    """Horizontal bars of mean attribution mass per feature key."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, max(2.0, 0.4 * len(feature_keys))))
    keys = list(feature_keys)
    vals = [float(feature_scores.get(k, 0.0)) for k in keys]
    ax.barh(keys, vals, color="#1f77b4")
    ax.set_xlabel("mean |grad x feature|")
    ax.set_title("Attribution mass per feature")
    return ax


def load_artefacts(run_dir: str | Path) -> dict[str, Any]:
    """Read `explanation.json` and `instances.pt` from a core run dir."""
    import json

    import torch

    run_dir = Path(run_dir)
    explanation = json.loads((run_dir / "explanation.json").read_text())
    instances_path = run_dir / "instances.pt"
    instances = (
        torch.load(instances_path, map_location="cpu", weights_only=False)
        if instances_path.is_file()
        else {}
    )
    return {"explanation": explanation, "instances": instances}
