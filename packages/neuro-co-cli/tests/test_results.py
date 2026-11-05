"""Aggregate actual training artifacts and versioned runs without layout coupling."""

import json
from typing import Any

import pytest

from neuro_co.cli.artifacts import RUN_SCHEMA, write_run
from neuro_co.cli.results import collect

pytest.importorskip("pandas")


def _json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _manifest(path, **overrides):
    args: dict[str, Any] = dict(
        kind="train",
        problem="tsp",
        seed=3,
        backend="torch",
        config={"size": 6},
        metrics={"training_s": 2.0},
        artifacts={},
    )
    args.update(overrides)
    return write_run(path, **args)


def test_standard_training_and_evaluation_at_arbitrary_path(tmp_path):
    output = tmp_path / "my-custom-output"
    _json(
        output / "config.json",
        {
            "schema": "neuro-co-training/v1",
            "backend": "jax",
            "training": {"problem": "cvrp", "size": 20, "seed": 42},
        },
    )
    _json(
        output / "metrics.json",
        {
            "best_reward": -5,
            "best_step": 2,
            "timing": {"training_s": 1.5, "total_s": 2.0},
            "history": [{"step": 2, "loss": 0.5}],
        },
    )
    _json(
        output / "evaluation.json",
        {"greedy_tour_length": 5.1, "first_call_s": 0.3, "costs": [5.0, 5.2]},
    )
    frame = collect(output)
    assert set(frame["kind"]) == {"train", "eval"}
    assert set(frame["backend"]) == {"jax"}
    assert set(frame["size"]) == {20}
    assert set(frame["seed"]) == {42}
    assert "training_s" in set(frame["metric"])
    assert frame[frame.metric == "greedy_tour_length"].value.tolist() == [5.1]
    assert "costs" not in set(frame["metric"])
    assert len(frame) == len(collect(tmp_path))


def test_manifest_precedence_and_relative_scope(tmp_path):
    first = tmp_path / "experiments" / "first" / "runs" / "weird-name"
    second = tmp_path / "experiments" / "second" / "runs" / "weird-name"
    run = _manifest(first)
    _manifest(second, metrics={"training_s": 99.0})
    _json(
        first / "config.json",
        {"backend": "torch", "training": {"problem": "tsp", "seed": 3, "size": 6}},
    )
    _json(first / "metrics.json", {"best_reward": -6, "timing": {"training_s": 100.0}})
    frame = collect(first.parent.parent)
    assert len(frame) == 1
    assert frame.value.tolist() == [2.0]
    assert frame.run_id.tolist() == [run["run_id"]]
    assert frame.schema.tolist() == [RUN_SCHEMA]


def test_train_manifest_retains_test_evaluation(tmp_path):
    _manifest(tmp_path)
    _json(tmp_path / "evaluation.json", {"greedy_tour_length": 6.0, "warm_call_s": 0.2})
    frame = collect(tmp_path)
    assert set(frame.kind) == {"train", "eval"}
    assert frame.run_id.nunique() == 1


def test_failed_run_remains_visible_without_fabricated_metric(tmp_path):
    _manifest(tmp_path, status="failed", metrics={}, error="interrupted")
    frame = collect(tmp_path)
    assert frame.status.tolist() == ["failed"]
    assert frame.metric.tolist() == [""]
    assert frame.value.isna().all()


def test_explanation_method_and_legacy_deduplication(tmp_path):
    output = tmp_path / "tsp" / "explain_seed3_gradient"
    _manifest(
        output,
        kind="explain",
        config={"size": 6, "method": "gradient"},
        metrics={"deletion.mean_flip_rate": 0.5},
    )
    _json(
        output / "explanation.json",
        {
            "faithfulness": {
                "deletion": {"mean_flip_rate": 0.5},
                "sufficiency": {"mean_keep_rate": 0.75},
            }
        },
    )
    frame = collect(tmp_path)
    assert len(frame) == 2
    assert set(frame.method) == {"gradient"}
    assert set(frame.metric) == {"deletion.mean_flip_rate", "sufficiency.mean_keep_rate"}


def test_historical_energy_and_probe_sidecars(tmp_path):
    output = tmp_path / "cvrp" / "train_seed4"
    _json(output / "energy_train.json", {"energy_wh": 0.2, "duration_s": 5})
    _json(
        output / "probes" / "probes.json",
        {"probes": [{"concept": "capacity", "layer": -1, "val_acc": 0.8}]},
    )
    frame = collect(tmp_path)
    assert len(frame) == 3
    assert set(frame.kind) == {"train", "probe"}
    assert set(frame.seed) == {4}


def test_corrupt_and_nonfinite_artifacts_are_skipped(tmp_path, caplog):
    _json(
        tmp_path / "tsp" / "train_seed1" / "energy_train.json",
        {"energy_wh": float("nan"), "duration_s": float("inf")},
    )
    _json(tmp_path / "broken" / "run.json", [])
    _json(tmp_path / "invalid" / "run.json", {"schema": RUN_SCHEMA, "metrics": {}})
    frame = collect(tmp_path)
    assert frame.empty
    assert "skipping" in caplog.text


def test_invalid_probe_fields_do_not_abort_other_results(tmp_path):
    output = tmp_path / "cvrp" / "train_seed0"
    _json(
        output / "probes" / "probes.json",
        {"probes": [{"layer": "bad", "val_acc": 0.8}, {"layer": 0, "val_acc": 0.9}]},
    )
    assert collect(tmp_path).value.tolist() == [0.9]


def test_arbitrary_modern_run_keeps_probe_sidecar(tmp_path):
    output = tmp_path / "arbitrary-run"
    _manifest(output)
    _json(
        output / "probes" / "probes.json",
        {"probes": [{"concept": "c", "layer": 0, "val_acc": 0.8}]},
    )
    frame = collect(output)
    assert set(frame.kind) == {"train", "probe"}
    assert set(frame.seed) == {3}
    assert frame.run_id.nunique() == 1


def test_malformed_manifest_status_and_huge_numbers_do_not_crash(tmp_path):
    data = _manifest(tmp_path / "bad")
    data["status"] = []
    _json(tmp_path / "bad" / "run.json", data)
    _json(tmp_path / "tsp" / "train_seed0" / "energy_train.json", {"energy_wh": 10**500})
    assert collect(tmp_path).empty


def test_figures_do_not_average_failed_run_metrics(tmp_path, monkeypatch):
    from neuro_co.cli import figures

    _manifest(tmp_path / "good", metrics={"training_s": 2})
    _manifest(tmp_path / "failed", status="failed", metrics={"training_s": 100})
    observed = []

    def panel(frame, out):
        observed.extend(frame.value.tolist())
        return None

    for name in (
        "plot_probe_accuracy_heatmap",
        "plot_attribution_quality",
        "plot_sanity_pass_rate",
        "plot_cross_seed_stability",
    ):
        monkeypatch.setattr(figures, name, panel)
    figures.render_all(collect(tmp_path), tmp_path / "figures")
    assert observed == [2.0] * 4


def test_modern_probe_stage_reads_referenced_granular_metrics(tmp_path):
    output = tmp_path / "campaign" / "runs" / "tsp" / "probe_seed3"
    _manifest(output, kind="probe", metrics={}, artifacts={"probes": "probes.json"})
    _json(
        output / "probes.json",
        {
            "probes": [
                {"concept": "near", "layer": 0, "val_acc": 0.8},
                {"concept": "near", "layer": -1, "val_acc": 0.9},
            ]
        },
    )
    frame = collect(tmp_path / "campaign")
    assert len(frame) == 2
    assert frame.kind.tolist() == ["probe", "probe"]
    assert set(frame.layer) == {0, -1}
    assert set(frame.concept) == {"near"}
    assert frame.value.tolist() == [0.8, 0.9]
    assert set(frame.seed) == {3}
    assert frame.run_id.nunique() == 1


def test_probe_reference_does_not_collect_another_campaign(tmp_path, caplog):
    output = tmp_path / "first"
    other = tmp_path / "second" / "probes.json"
    _manifest(output, kind="probe", metrics={}, artifacts={"probes": other})
    _json(other, {"probes": [{"concept": "near", "layer": 0, "val_acc": 0.8}]})
    frame = collect(output)
    assert len(frame) == 1
    assert frame.metric.tolist() == [""]
    assert "outside collection root" in caplog.text
