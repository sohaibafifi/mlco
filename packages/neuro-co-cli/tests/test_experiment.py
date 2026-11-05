"""Complete experiments, validated plans, isolation and interruption recovery."""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from neuro_co.cli import experiment
from neuro_co.cli.stage_runners import _nearest_neighbour


def recipe(tmp_path, name="sample"):
    payload = {
        "name": name,
        "parallel": 1,
        "problems": {
            "tsp": {
                "seeds": [0],
                "train": {
                    "size": 6,
                    "epochs": 1,
                    "steps_per_epoch": 1,
                    "batch_size": 2,
                    "eval_batch_size": 4,
                    "hidden_dim": 8,
                    "num_layers": 1,
                    "num_heads": 2,
                    "n_starts": 2,
                    "device": "cpu",
                },
                "baseline": {},
                "eval": {},
                "explain": {
                    "methods": ["gradient"],
                    "num_instances": 2,
                    "top_k": 2,
                    "max_steps": 1,
                },
            }
        },
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path, payload


def test_complete_experiment_has_paired_data_and_real_explanation(tmp_path):
    source, _ = recipe(tmp_path)
    output = tmp_path / "complete"
    assert experiment.run_experiment(source, out_dir=output) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert {item["status"] for item in manifest["stages"].values()} == {"complete"}
    train = output / "runs/tsp/train_seed0"
    checkpoint = train / "best.pt"
    assert checkpoint.is_file()
    evaluation = json.loads((output / "runs/tsp/eval_seed0/evaluation.json").read_text())
    baseline = json.loads((output / "runs/tsp/baseline_seed0/baseline.json").read_text())
    training_evaluation = json.loads((train / "evaluation.json").read_text())
    assert (
        evaluation["test_sha256"] == baseline["test_sha256"] == training_evaluation["test_sha256"]
    )
    assert len(evaluation["costs"]) == len(baseline["costs"]) == 4
    assert evaluation["costs"] == pytest.approx(training_evaluation["costs"])
    explanation = json.loads(
        (output / "runs/tsp/explain_seed0_gradient/explanation.json").read_text()
    )
    assert Path(explanation["ckpt_path"]).resolve() == checkpoint.resolve()
    assert (output / "datasets/tsp_seed0.npz").is_file()
    assert (output / "results.parquet").stat().st_size > 0
    assert "nearest_neighbour" in (output / "results.csv").read_text()
    assert "<table" in (output / "report.html").read_text()
    timestamp = checkpoint.stat().st_mtime_ns
    assert experiment.run_experiment(source, out_dir=output, resume=True) == 0
    assert checkpoint.stat().st_mtime_ns == timestamp
    # Corruption must be rejected before changing a completed campaign receipt.
    manifest_bytes = (output / "manifest.json").read_bytes()
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.unlink()
    with pytest.raises(ValueError, match="missing required artifact"):
        experiment.run_experiment(source, out_dir=output, resume=True)
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    checkpoint.write_bytes(checkpoint_bytes)
    dataset = output / "datasets/tsp_seed0.npz"
    dataset_bytes = dataset.read_bytes()
    dataset.write_bytes(b"corrupted dataset")
    with pytest.raises(ValueError, match="dataset is missing or corrupted"):
        experiment.run_experiment(source, out_dir=output, resume=True)
    assert (output / "manifest.json").read_bytes() == manifest_bytes
    dataset.write_bytes(dataset_bytes)
    with pytest.raises(FileExistsError):
        experiment.run_experiment(source, out_dir=output)


def test_failed_stage_is_recorded_and_retried_without_retraining(tmp_path, monkeypatch):
    source, _ = recipe(tmp_path)
    output = tmp_path / "retry"
    real_execute = experiment._execute_stage

    def fail_eval(stage, directory, resume):
        if stage.kind == "eval":
            raise RuntimeError("simulated interruption")
        return real_execute(stage, directory, resume)

    monkeypatch.setattr(experiment, "_execute_stage", fail_eval)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        experiment.run_experiment(source, out_dir=output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["stages"]["tsp/eval_seed0"]["status"] == "failed"
    assert manifest["stages"]["tsp/explain_seed0_gradient"]["status"] == "pending"
    checkpoint = output / "runs/tsp/train_seed0/best.pt"
    timestamp = checkpoint.stat().st_mtime_ns
    with pytest.raises(ValueError, match="retry-failed"):
        experiment.run_experiment(source, out_dir=output, resume=True)
    monkeypatch.setattr(experiment, "_execute_stage", real_execute)
    assert experiment.run_experiment(source, out_dir=output, retry_failed=True) == 0
    assert checkpoint.stat().st_mtime_ns == timestamp
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stages"]["tsp/train_seed0"]["attempts"] == 1
    assert manifest["stages"]["tsp/eval_seed0"]["attempts"] == 2


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda raw: raw.update(parallel=2), "parallel: 1"),
        (lambda raw: raw.update(name="../escape"), "safe directory"),
        (lambda raw: raw["problems"]["tsp"]["train"].update(overrides=[]), "unknown options"),
        (lambda raw: raw["problems"]["tsp"]["eval"].update(ignored=True), "unknown options"),
        (
            lambda raw: raw["problems"]["tsp"]["explain"].update(methods=["deeplift"]),
            "explain.methods",
        ),
        (lambda raw: raw["problems"]["tsp"].update(seeds=[0, 0]), "duplicate seeds"),
    ],
)
def test_invalid_recipe_is_rejected_before_writes(tmp_path, mutation, message):
    source, payload = recipe(tmp_path)
    mutation(payload)
    source.write_text(yaml.safe_dump(payload))
    output = tmp_path / "invalid"
    with pytest.raises(SystemExit, match=message):
        experiment.run_experiment(source, out_dir=output)
    assert not output.exists()


def test_dry_run_creates_no_outputs_and_changed_recipe_cannot_resume(tmp_path):
    source, payload = recipe(tmp_path)
    output = tmp_path / "dry"
    assert experiment.run_experiment(source, out_dir=output, dry_run=True) == 0
    assert not output.exists()
    output.mkdir()
    (output / "manifest.json").write_text(
        json.dumps({"schema": "neuro-co-experiment/v1", "recipe": {}})
    )
    with pytest.raises(ValueError, match="changed recipe"):
        experiment.run_experiment(source, out_dir=output, resume=True)
    assert payload["name"] == "sample"


def test_nearest_neighbour_routes_preserve_customers_and_capacity():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    demand = np.array([0, 3, 3, 2])
    routes, cost = _nearest_neighbour(coords, demand, capacity=5)
    assert sorted(node for route in routes for node in route) == [1, 2, 3]
    assert all(sum(demand[node] for node in route) <= 5 for route in routes)
    recomputed = sum(
        np.linalg.norm(coords[a] - coords[b])
        for route in routes
        for a, b in zip([0, *route], [*route, 0], strict=True)
    )
    assert cost == pytest.approx(recomputed)
