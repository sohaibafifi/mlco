"""Analyze directions in encoder activations.

PCA and ICA operate on node embeddings. For each direction, the report includes
its Pearson correlation with binary concept labels. t-SNE provides a separate
two-dimensional projection.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from neuro_co.core.concepts import ConceptFn
from neuro_co.probe.probes import _encode


@dataclass
class DirectionAlignment:
    """One PCA direction's best-match concept."""

    direction_index: int
    explained_variance_ratio: float
    best_concept: str | None
    best_abs_pearson: float
    pearson_per_concept: dict[str, float]


@dataclass
class DiscoveredDirections:
    """Output of `discover_directions`."""

    method: str  # "pca" | "ica"
    n_components: int
    embed_dim: int
    embeddings: NDArray[np.float32]
    components: NDArray[np.float32]  # [n_components, embed_dim]
    explained_variance_ratio: list[float]
    concept_labels: dict[str, NDArray[np.int64]]
    alignment: list[DirectionAlignment]
    scores: NDArray[np.float32]  # [n_samples, n_components]


def _flatten_keep_customers(
    embeddings: Tensor,
    concept_labels: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    """Flatten `[B, N, D]` → `[B*N, D]`, drop nodes any concept marks `-1`."""
    flat = embeddings.reshape(-1, embeddings.shape[-1])
    keep = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
    flat_labels: dict[str, Tensor] = {}
    for name, lab in concept_labels.items():
        f = lab.reshape(-1)
        flat_labels[name] = f
        keep = keep & (f >= 0)
    return flat[keep], {name: lab[keep] for name, lab in flat_labels.items()}


def discover_directions(
    policy: Any,
    env: Any,
    state: Any,
    *,
    concepts: Mapping[str, ConceptFn],
    n_components: int = 10,
    method: str = "pca",
    random_state: int = 0,
) -> DiscoveredDirections:
    """Encode `state`, fit a linear decomposition, score each direction.

    `policy` core `ConstructivePolicy`, `env` core `Env`, `state` core
    `State`.

    `method="pca"` runs `sklearn.decomposition.PCA` and reports the
    native explained-variance ratio per direction.

    `method="ica"` runs `sklearn.decomposition.FastICA` and reports a
    variance-share proxy: per-direction score variance divided by the
    total score variance (FastICA itself doesn't expose explained
    variance).
    """
    if method not in {"pca", "ica"}:
        raise ValueError(f"method must be 'pca' or 'ica', got {method!r}")

    device = next(policy.parameters()).device
    policy.eval()
    state = state.to(device)
    features = env.build_features(state)

    with torch.no_grad():
        embeddings = _encode(policy, features)  # [B, N, D]

    labels = {n: fn(state) for n, fn in concepts.items() if fn(state) is not None}
    if not labels:
        raise ValueError("No concept produced labels for this state")
    flat_emb, flat_lab = _flatten_keep_customers(
        embeddings, {n: torch.as_tensor(v).to(device) for n, v in labels.items()}
    )
    if flat_emb.shape[0] < n_components:
        n_components = max(2, flat_emb.shape[0])
    X = flat_emb.detach().cpu().numpy().astype(np.float32)

    if method == "pca":
        from sklearn.decomposition import PCA

        decomp = PCA(n_components=int(n_components))
        scores = decomp.fit_transform(X)
        components = decomp.components_
        n_fit = int(decomp.n_components_)
        variance = [float(v) for v in decomp.explained_variance_ratio_]
    else:
        from sklearn.decomposition import FastICA

        decomp = FastICA(
            n_components=int(n_components),
            random_state=random_state,
            max_iter=500,
            whiten="unit-variance",
        )
        scores = decomp.fit_transform(X)
        components = decomp.components_
        n_fit = int(components.shape[0])
        per_dir = scores.var(axis=0)
        total = float(per_dir.sum()) if per_dir.sum() > 0 else 1.0
        variance = [float(v / total) for v in per_dir]

    concept_arrays: dict[str, NDArray[np.int64]] = {
        n: flat_lab[n].cpu().numpy().astype(np.int64) for n in flat_lab
    }

    alignment: list[DirectionAlignment] = []
    for i in range(n_fit):
        pearson_per: dict[str, float] = {}
        best_name: str | None = None
        best_abs = 0.0
        for cname, y_arr in concept_arrays.items():
            r = _pearson(scores[:, i], y_arr.astype(np.float32))
            pearson_per[cname] = r
            if abs(r) > best_abs:
                best_abs = abs(r)
                best_name = cname
        alignment.append(
            DirectionAlignment(
                direction_index=i,
                explained_variance_ratio=variance[i],
                best_concept=best_name,
                best_abs_pearson=best_abs,
                pearson_per_concept=pearson_per,
            )
        )

    return DiscoveredDirections(
        method=method,
        n_components=n_fit,
        embed_dim=int(flat_emb.shape[-1]),
        embeddings=X,
        components=components.astype(np.float32),
        explained_variance_ratio=variance,
        concept_labels=concept_arrays,
        alignment=alignment,
        scores=scores.astype(np.float32),
    )


def fit_tsne(
    embeddings: NDArray[np.floating],
    *,
    perplexity: float = 30.0,
    random_state: int = 0,
    n_iter: int = 1000,
) -> NDArray[np.float32]:
    """2D t-SNE projection of `embeddings`.

    Adapts `perplexity` if the dataset is too small (perplexity must
    be `< n_samples`).
    """
    from sklearn.manifold import TSNE

    n = int(embeddings.shape[0])
    if perplexity >= n:
        perplexity = max(2.0, n / 4.0)
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        random_state=int(random_state),
        init="pca",
        max_iter=int(n_iter),
    )
    return tsne.fit_transform(embeddings).astype(np.float32)


def _pearson(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)
