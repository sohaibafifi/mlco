"""PyTorch-Lightning callback emitting per-epoch energy snapshots.

`EnergyTracker` wraps a whole training run with one context-manager
block, which is suitable for short runs, but for multi-hour jobs you want
per-epoch granularity (an OOM mid-epoch, a hot-swap GPU, or a paper
plot of energy vs. epoch all benefit). `AETCallback` opens a fresh
`EnergyTracker` on `on_train_epoch_start` and closes it on
`on_train_epoch_end`, writing `energy_train_epoch_<idx>.json`
alongside the run-level `energy_train.json` the outer tracker
produces.

The callback has a soft Lightning dependency: importing
`neuro_co.aet.lightning` triggers the Lightning import. The
top-level `neuro_co.aet` namespace exposes `AETCallback` via
`__getattr__` so plain `import neuro_co.aet` keeps the import graph
torch-free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuro_co.aet.energy.tracker import EnergyTracker

try:
    from lightning.pytorch.callbacks import Callback as _LightningCallback
except ImportError:  # pragma: no cover - lightning is optional
    try:
        from pytorch_lightning.callbacks import (
            Callback as _LightningCallback,  # type: ignore[no-redef]
        )
    except ImportError:
        _LightningCallback = None  # type: ignore[assignment, misc]


if _LightningCallback is not None:

    class AETCallback(_LightningCallback):  # type: ignore[misc, valid-type]
        """Per-epoch `EnergyTracker` wrapper.

        Parameters
        ----------
        output_dir
            Directory receiving `energy_train_epoch_<idx>.json`
            snapshots. Created if missing. Typically
            `cfg.paths.output_dir`.
        backend, hardware_id, pue, grid_intensity_g_per_kwh,
        country_iso_code
            Forwarded to `EnergyTracker` per epoch, with the same fields as
            the run-level tracker so per-epoch records are directly
            comparable.
        label_prefix
            Stem of the per-epoch label; the full label becomes
            `f"{label_prefix}_epoch_{idx}"`. Default `"train"`.
        items_attr
            Attribute on the `LightningModule` or `Trainer` holding
            an epoch-level item count for
            `throughput_items_per_s`. Default `None` (no items
            reported per epoch).
        """

        def __init__(
            self,
            output_dir: str | Path,
            *,
            backend: str = "codecarbon",
            hardware_id: str | None = None,
            pue: float = 1.4,
            grid_intensity_g_per_kwh: float = 475.0,
            country_iso_code: str | None = None,
            label_prefix: str = "train",
            items_attr: str | None = None,
        ) -> None:
            super().__init__()
            self.output_dir = Path(output_dir)
            self.backend = backend
            self.hardware_id = hardware_id
            self.pue = pue
            self.grid_intensity = grid_intensity_g_per_kwh
            self.country_iso_code = country_iso_code
            self.label_prefix = label_prefix
            self.items_attr = items_attr
            self._tracker: EnergyTracker | None = None
            self._epoch_records: list[dict[str, Any]] = []

        # ---- Lightning hooks --------------------------------------------

        def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
            idx = int(getattr(trainer, "current_epoch", 0))
            items = self._resolve_items(trainer, pl_module)
            self._tracker = EnergyTracker(
                label=f"{self.label_prefix}_epoch_{idx}",
                backend=self.backend,
                hardware_id=self.hardware_id,
                pue=self.pue,
                grid_intensity_g_per_kwh=self.grid_intensity,
                country_iso_code=self.country_iso_code,
                output_dir=None,
                items=items,
                report_embodied=True,
            )
            self._tracker.__enter__()

        def on_train_epoch_end(self, trainer: Any, _pl_module: Any) -> None:
            if self._tracker is None:
                return
            self._tracker.__exit__(None, None, None)
            reading = self._tracker.reading
            if reading is None:
                self._tracker = None
                return
            idx = int(getattr(trainer, "current_epoch", 0))
            record = reading.to_dict()
            record["extra"] = {
                **record["extra"],
                "epoch": idx,
                "producer": "lightning_aet_callback",
            }
            self._epoch_records.append(record)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / f"energy_train_epoch_{idx}.json").write_text(
                json.dumps(record, indent=2, default=str)
            )
            self._tracker = None

        def on_train_end(self, _trainer: Any, _pl_module: Any) -> None:
            """Write a rollup `energy_train_epochs.json` indexed by epoch."""
            if not self._epoch_records:
                return
            rollup = {
                "schema_version": "aet-training-rollup/v1",
                "label": f"{self.label_prefix}_epochs",
                "per_epoch": self._epoch_records,
                "total_energy_j": sum(r["energy_j"] for r in self._epoch_records),
                "total_co2_kg": sum(r["co2_total_kg"] for r in self._epoch_records),
                "n_epochs": len(self._epoch_records),
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "energy_train_epochs.json").write_text(
                json.dumps(rollup, indent=2, default=str)
            )

        # ---- helpers ----------------------------------------------------

        def _resolve_items(self, trainer: Any, pl_module: Any) -> int:
            if self.items_attr is None:
                return 0
            for obj in (pl_module, trainer):
                v = getattr(obj, self.items_attr, None)
                if v is not None:
                    try:
                        return int(v)
                    except (TypeError, ValueError):
                        return 0
            return 0

else:

    class AETCallback:  # type: ignore[no-redef]
        """Stub raising at instantiation time when `lightning` is missing."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("AETCallback requires `lightning`. Install via `uv add lightning`.")
