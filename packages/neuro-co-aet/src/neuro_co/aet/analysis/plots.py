"""Matplotlib plots for AET tables.

`plot_aet_vs_delta`: AET (instances) vs tolerated quality gap delta.
`plot_asymptotic`: cumulative-energy curves NN vs metaheuristic.
`plot_per_seed_energy`: bar chart of per-seed training energy + IQR.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuro_co.aet.analysis.aet import TrainAggregate
from neuro_co.aet.analysis.asymptotic import energy_curves


def plot_aet_vs_delta(rows: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Group rows by (batch_size, threads_mode) and plot AET_E vs δ."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))
    if not rows:
        ax.set_title("AET vs delta: no data")
    else:
        groups: dict[tuple, list[dict[str, Any]]] = {}
        for r in rows:
            key = (r.get("batch_size"), r.get("threads_mode"))
            groups.setdefault(key, []).append(r)
        for (batch, threads), group in sorted(
            groups.items(), key=lambda kv: (kv[0][0] or 0, str(kv[0][1]))
        ):
            sub = sorted(group, key=lambda r: r.get("delta_pct", 0))
            xs = [
                float(r["delta_pct"])
                for r in sub
                if r.get("aet_E_status") == "finite" and math.isfinite(r["aet_E"])
            ]
            ys = [
                float(r["aet_E"])
                for r in sub
                if r.get("aet_E_status") == "finite" and math.isfinite(r["aet_E"])
            ]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", label=f"batch={batch}, {threads}")
        ax.set_xlabel("δ (tolerated quality gap, %)")
        ax.set_ylabel("AET_E (instances)")
        ax.set_yscale("log")
        ax.set_title("Energy break-even AET vs δ")
        ax.grid(True, which="both", alpha=0.3)
        if ax.has_data():
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_asymptotic(
    train_agg: TrainAggregate,
    rows: list[dict[str, Any]],
    out_path: str | Path,
    *,
    n_values: Sequence[int] | None = None,
) -> Path:
    """Cumulative-energy curves NN vs metaheuristic on log-log axes.

    Picks the median deployment-energy pair over rows with a finite AET.
    """
    out_path = Path(out_path)
    finite_rows = [
        r
        for r in rows
        if r.get("aet_E_status") == "finite" and math.isfinite(float(r.get("aet_E", float("nan"))))
    ]
    fig, ax = plt.subplots(figsize=(7, 5))
    if not finite_rows:
        ax.set_title("Asymptotic regime: no finite AET rows")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    if not math.isfinite(train_agg.energy_wh_mean):
        ax.set_title("Asymptotic regime: no valid training-energy mean")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path

    e_nn_med = _median([r["E_NN_wh_per_inst"] for r in finite_rows])
    e_meta_med = _median([r["E_meta_wh_per_inst"] for r in finite_rows])
    n_vals, nn_curve, meta_curve = energy_curves(
        train_agg.energy_wh_mean, e_nn_med, e_meta_med, list(n_values) if n_values else None
    )
    ax.plot(n_vals, nn_curve, marker="o", color="#1f77b4", label="neural (train + N · E_NN)")
    ax.plot(n_vals, meta_curve, marker="s", color="#d62728", label="metaheuristic (N · E_meta)")

    # IQR fill for training energy uncertainty.
    if math.isfinite(train_agg.energy_wh_p25) and math.isfinite(train_agg.energy_wh_p75):
        lo = [train_agg.energy_wh_p25 + e_nn_med * n for n in n_vals]
        hi = [train_agg.energy_wh_p75 + e_nn_med * n for n in n_vals]
        ax.fill_between(n_vals, lo, hi, alpha=0.18, color="#1f77b4", label="NN (IQR over seeds)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (deployed instances)")
    ax.set_ylabel("Cumulative energy (Wh)")
    ax.set_title("Asymptotic regime: neural solver vs metaheuristic")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_per_seed_energy(records: list[dict[str, Any]], out_path: str | Path) -> Path:
    """Bar chart of per-seed training energy."""
    out_path = Path(out_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    if not records:
        ax.set_title("Per-seed training energy: no data")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return out_path
    seeds = [str(r.get("seed", i)) for i, r in enumerate(records)]
    energies = [_training_energy_wh(record) for record in records]
    ax.bar(seeds, energies, color="#1f77b4")
    ax.set_ylabel("training energy (Wh)")
    ax.set_xlabel("seed")
    ax.set_title("Per-seed training energy")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _median(values: list[float]) -> float:
    cleaned = sorted(v for v in values if isinstance(v, int | float) and math.isfinite(v))
    if not cleaned:
        return float("nan")
    n = len(cleaned)
    return cleaned[n // 2] if n % 2 else (cleaned[n // 2 - 1] + cleaned[n // 2]) / 2.0


def _training_energy_wh(record: dict[str, Any]) -> float:
    energy_j = record.get("energy_j")
    if isinstance(energy_j, int | float) and math.isfinite(float(energy_j)):
        return float(energy_j) / 3600.0
    energy_wh = record.get("energy_wh")
    if isinstance(energy_wh, int | float) and math.isfinite(float(energy_wh)):
        return float(energy_wh)
    return float("nan")
