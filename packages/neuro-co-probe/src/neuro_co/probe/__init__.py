"""Encoder probes, embedding analysis, and explanation reports.

`fit_concept_probes` fits linear concept classifiers to encoder activations.
`discover_directions` and `fit_tsne` analyze embeddings with PCA, ICA, or t-SNE.
The optional `explain` extra adds attribution and faithfulness checks through
`explain_policy`. Plotting and report helpers load on demand.

Command-line access uses `neuroco probe` and `neuroco-explain-report`.
"""

from __future__ import annotations

from neuro_co.probe.discovered import (
    DirectionAlignment,
    DiscoveredDirections,
    discover_directions,
    fit_tsne,
)
from neuro_co.probe.probes import (
    ProbeResult,
    encoder_layer_count,
    fit_concept_probes,
)

__version__ = "0.2.0"

# Lazy-load matplotlib + report helpers so callers that only want
# probes/discovered don't pay matplotlib's import cost.
_LAZY_VIZ = {
    "plot_route_attribution",
    "plot_flip_rate",
    "plot_feature_attribution_bars",
    "plot_probe_metrics_bars",
    "plot_probe_roc",
    "plot_pca_embeddings",
    "plot_explained_variance",
    "plot_direction_concept_heatmap",
    "plot_layer_concept_heatmap",
    "plot_scatter_2d_by_label",
    "plot_tsne_embeddings",
    "load_artefacts",
}
_LAZY_REPORT = {"write_explanation_report"}
# Explanation needs the optional attribution dependency; probing does not.
_LAZY_EXPLAIN = {"explain_policy"}


def __getattr__(name: str):  # PEP 562 module-level __getattr__
    if name in _LAZY_VIZ:
        from neuro_co.probe import viz

        return getattr(viz, name)
    if name in _LAZY_REPORT:
        from neuro_co.probe import report

        return getattr(report, name)
    if name in _LAZY_EXPLAIN:
        from neuro_co.probe.explainer import explain_policy

        return explain_policy
    raise AttributeError(f"module 'neuro_co.probe' has no attribute {name!r}")


__all__ = [
    "DirectionAlignment",
    "DiscoveredDirections",
    "ProbeResult",
    "__version__",
    "discover_directions",
    "encoder_layer_count",
    "explain_policy",  # lazy via __getattr__ (needs the `explain` extra)
    "fit_concept_probes",
    "fit_tsne",
]


def __dir__() -> list[str]:
    return sorted(__all__)
