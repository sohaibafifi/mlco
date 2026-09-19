"""Keep the Torch/JAX training and evaluation protocols comparable."""

import json

import numpy as np
import pytest

from neuro_co.cli.training import (
    TrainingConfig,
    batch,
    evaluate_saved_run,
    initial_weights,
    make_trainer,
    summarize,
    train,
)


@pytest.mark.parametrize("problem", ["tsp", "cvrp"])
def test_shared_protocol_and_checkpoint_selection(tmp_path, problem):
    pytest.importorskip("jax")
    config = TrainingConfig(
        problem=problem,
        size=5,
        epochs=2,
        steps_per_epoch=1,
        batch_size=2,
        eval_batch_size=3,
        n_starts=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        capacity=15,
        max_demand=7,
    )
    weights = initial_weights(config)
    test_data = batch(config, "test")
    initial_costs = [
        make_trainer(config, backend, "cpu", weights).evaluate(test_data)
        for backend in ("torch", "jax")
    ]
    np.testing.assert_allclose(*initial_costs, rtol=1e-5)
    assert not np.array_equal(
        batch(config, "train", 0)["coords"], batch(config, "train", 1)["coords"]
    )
    assert not np.array_equal(batch(config, "validation")["coords"], test_data["coords"])
    runs = []
    for backend in ("torch", "jax"):
        directory = tmp_path / backend / problem
        result = train(config, backend, "cpu", directory)
        runs.append(result)
        assert result["timing"]["training_s"] > result["timing"]["first_train_step_s"] > 0
        assert result["timing"]["warm_train_step_mean_s"] > 0
        assert len(result["history"]) == 2
        saved = json.loads((directory / "evaluation.json").read_text())
        replayed = evaluate_saved_run(directory / saved["checkpoint"])
        np.testing.assert_array_equal(saved["costs"], replayed["costs"])
        assert replayed["step"] == result["best_step"]
        selected = (directory / "evaluation.json").read_text()
        latest = "latest.pt" if backend == "torch" else "checkpoint.npz"
        evaluate_saved_run(directory / latest)
        assert (directory / "evaluation.json").read_text() == selected
        with pytest.raises(FileExistsError):
            train(config, backend, "cpu", directory)
    assert runs[0]["protocol"] == runs[1]["protocol"]
    summary = summarize(tmp_path)
    assert summary["matched_protocol"][problem]
    assert summary["timing_comparable"][problem]
    metrics = tmp_path / "jax" / problem / "metrics.json"
    altered = json.loads(metrics.read_text())
    altered["args"]["capacity"] += 1
    metrics.write_text(json.dumps(altered))
    assert not summarize(tmp_path)["matched_protocol"][problem]
