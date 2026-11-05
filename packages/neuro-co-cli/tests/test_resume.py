"""CLI recovery preserves optimizer progress, validation selection and run identity."""

import hashlib
import json
from dataclasses import replace
from typing import Any

import pytest
import torch

from neuro_co.cli import runners, training
from neuro_co.cli.runners import TrainArgs, train_run
from neuro_co.cli.training import TrainingConfig, train


def _config():
    return TrainingConfig(
        size=5,
        epochs=2,
        steps_per_epoch=3,
        batch_size=2,
        eval_batch_size=3,
        n_starts=2,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )


def _tree_digest(path):
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def test_shared_resume_after_mid_epoch_interruption_matches_uninterrupted(tmp_path, monkeypatch):
    config = _config()
    reference = tmp_path / "reference"
    expected = train(config, "torch", "cpu", reference)
    interrupted = tmp_path / "interrupted"
    original = training.make_trainer

    def faulty(*args, **kwargs):
        trainer = original(*args, **kwargs)
        update = trainer.train_step
        calls = 0

        def step(data):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError("simulated interruption")
            return update(data)

        trainer.train_step = step
        return trainer

    monkeypatch.setattr(training, "make_trainer", faulty)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(config, "torch", "cpu", interrupted, checkpoint_every_steps=1)
    failed = json.loads((interrupted / "run.json").read_text())
    assert failed["status"] == "failed"
    assert torch.load(interrupted / "latest.pt", weights_only=True)["step"] == 2
    monkeypatch.setattr(training, "make_trainer", original)
    actual = train(config, "torch", "cpu", interrupted, resume_from="latest.pt")
    complete = json.loads((interrupted / "run.json").read_text())
    assert complete["status"] == "complete"
    assert complete["run_id"] == failed["run_id"]
    from neuro_co.cli.results import collect

    collected = collect(interrupted)
    train_metrics = set(collected.loc[collected["kind"] == "train", "metric"])
    assert {
        "duration_s",
        "total_s",
        "setup_s",
        "first_train_step_s",
        "best_reward",
        "best_step",
    } <= train_metrics
    assert actual["protocol"] == expected["protocol"]
    assert actual["best_step"] == expected["best_step"]
    assert [x["step"] for x in actual["history"]] == [3, 6]
    for actual_epoch, expected_epoch in zip(actual["history"], expected["history"], strict=True):
        assert {k: v for k, v in actual_epoch.items() if k != "training_s"} == {
            k: v for k, v in expected_epoch.items() if k != "training_s"
        }
    for filename in ("latest.pt", "best.pt"):
        a = torch.load(interrupted / filename, weights_only=True)
        e = torch.load(reference / filename, weights_only=True)
        for key, value in a["model"].items():
            torch.testing.assert_close(value, e["model"][key], rtol=0, atol=0)
    assert (
        json.loads((interrupted / "evaluation.json").read_text())["costs"]
        == json.loads((reference / "evaluation.json").read_text())["costs"]
    )


def test_shared_resume_rejects_mismatch_without_changing_files(tmp_path):
    config = replace(_config(), epochs=1, steps_per_epoch=1)
    train(config, "torch", "cpu", tmp_path)
    before = _tree_digest(tmp_path)
    with pytest.raises(ValueError, match="configuration"):
        train(
            replace(config, seed=config.seed + 1), "torch", "cpu", tmp_path, resume_from="latest.pt"
        )
    assert _tree_digest(tmp_path) == before
    with pytest.raises(ValueError, match="lacks resumable"):
        train(config, "torch", "cpu", tmp_path, resume_from="best.pt")
    assert _tree_digest(tmp_path) == before


def test_completed_target_performs_no_extra_updates_and_can_be_extended(tmp_path, monkeypatch):
    config = replace(_config(), epochs=1, steps_per_epoch=1)
    train(config, "torch", "cpu", tmp_path)
    original = training.make_trainer

    def no_updates(*args, **kwargs):
        trainer = original(*args, **kwargs)
        trainer.train_step = lambda data: pytest.fail("completed target retrained")
        return trainer

    monkeypatch.setattr(training, "make_trainer", no_updates)
    result = train(config, "torch", "cpu", tmp_path, resume_from="latest.pt")
    assert len(result["history"]) == 1
    monkeypatch.setattr(training, "make_trainer", original)
    result = train(replace(config, epochs=2), "torch", "cpu", tmp_path, resume_from="latest.pt")
    assert [x["step"] for x in result["history"]] == [1, 2]
    assert len(list(tmp_path.glob("best-step-*.pt"))) == 1


def test_jax_resume_rejected_before_creating_output(tmp_path):
    output = tmp_path / "jax"
    with pytest.raises(ValueError, match="Torch backend"):
        train(_config(), "jax", "cpu", output, resume_from="checkpoint.npz")
    assert not output.exists()


def test_generic_torch_resume_preserves_reinforce_baseline_and_history(tmp_path, monkeypatch):
    common: dict[str, Any] = dict(
        problem="tsp",
        algo="reinforce",
        size=5,
        epochs=2,
        steps_per_epoch=2,
        batch_size=2,
        eval_batch_size=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    reference = tmp_path / "reference"
    train_run(TrainArgs(**common, out_dir=str(reference)))
    output = tmp_path / "interrupted"
    original = runners.make_algo

    def faulty(*args, **kwargs):
        algo = original(*args, **kwargs)
        update = algo.train_step
        calls = 0

        def step(rng):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return update(rng)

        algo.train_step = step
        return algo

    monkeypatch.setattr(runners, "make_algo", faulty)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_run(TrainArgs(**common, out_dir=str(output), checkpoint_every_steps=1))
    monkeypatch.setattr(runners, "make_algo", original)
    actual = train_run(TrainArgs(**common, out_dir=str(output), resume="latest.pt"))
    assert [item["step"] for item in actual["history"]] == [2, 4]
    a = torch.load(output / "latest.pt", weights_only=True)
    e = torch.load(reference / "latest.pt", weights_only=True)
    assert a["step"] == e["step"] == 4
    for key, value in a["algo_state"].items():
        torch.testing.assert_close(value, e["algo_state"][key], rtol=0, atol=0)
    assert actual["history"] == json.loads((reference / "metrics.json").read_text())["history"]
    from neuro_co.cli.results import collect

    collected = collect(output)
    assert {"duration_s", "setup_s", "first_train_step_s", "best_reward", "best_step"} <= set(
        collected.loc[collected["kind"] == "train", "metric"]
    )
    before = _tree_digest(output)
    with pytest.raises(FileExistsError):
        train_run(TrainArgs(**common, out_dir=str(output)))
    assert _tree_digest(output) == before
    altered = {**common, "seed": 123}
    with pytest.raises(ValueError, match="configuration"):
        train_run(TrainArgs(**altered, out_dir=str(output), resume="latest.pt"))
    assert _tree_digest(output) == before


def test_cli_forwards_resume_and_checkpoint_interval(monkeypatch):
    from neuro_co.cli import main as cli

    observed = []
    monkeypatch.setattr(cli, "train_run", lambda args: observed.append(args))
    cli.main(
        ["train", "--problem", "tsp", "--resume", "latest.pt", "--checkpoint-every-steps", "7"]
    )
    assert observed[0].resume == "latest.pt"
    assert observed[0].checkpoint_every_steps == 7


def test_generic_resume_survives_failure_between_best_and_latest(tmp_path, monkeypatch):
    common: dict[str, Any] = dict(
        problem="tsp",
        algo="reinforce",
        size=5,
        epochs=2,
        steps_per_epoch=2,
        batch_size=2,
        eval_batch_size=3,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )
    make_algo = runners.make_algo

    def predictable_selection(*args, **kwargs):
        algo = make_algo(*args, **kwargs)
        evaluate = algo.eval_step

        def step(rng):
            metrics = evaluate(rng)
            return {**metrics, "eval_reward": float(algo._step)}

        algo.eval_step = step
        return algo

    monkeypatch.setattr(runners, "make_algo", predictable_selection)
    reference = tmp_path / "reference"
    train_run(TrainArgs(**common, out_dir=str(reference)))
    output = tmp_path / "interrupted"
    atomic_save = runners.atomic_save

    def fail_latest(payload, path):
        if path.name == "latest.pt" and payload["step"] == 4:
            raise OSError("interrupted latest replacement")
        return atomic_save(payload, path)

    monkeypatch.setattr(runners, "atomic_save", fail_latest)
    with pytest.raises(OSError, match="latest replacement"):
        train_run(TrainArgs(**common, out_dir=str(output)))
    durable = torch.load(output / "latest.pt", weights_only=True)
    assert durable["step"] == 2
    assert durable["run_state"]["best_checkpoint"] == "best-step-2.pt"
    assert (output / "best-step-2.pt").is_file()
    assert torch.load(output / "best.pt", weights_only=True)["step"] == 4
    monkeypatch.setattr(runners, "atomic_save", atomic_save)
    actual = train_run(TrainArgs(**common, out_dir=str(output), resume="latest.pt"))
    expected = json.loads((reference / "metrics.json").read_text())
    assert actual["history"] == expected["history"]
    selected = torch.load(output / "best.pt", weights_only=True)
    for name, value in torch.load(reference / "best.pt", weights_only=True)["model"].items():
        torch.testing.assert_close(selected["model"][name], value, rtol=0, atol=0)
    assert [path.name for path in output.glob("best-step-*.pt")] == ["best-step-4.pt"]
