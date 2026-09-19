"""JAX training checkpoints preserve optimizer and random state across resume."""

from dataclasses import replace

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from neuro_co.core.jax_backend.train import (  # noqa: E402
    Config,
    checkpoint_metadata,
    main,
    train,
)


def small_config(**overrides):
    return replace(
        Config(
            steps=2,
            size=5,
            batch_size=2,
            n_starts=3,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
            eval_batch_size=2,
        ),
        **overrides,
    )


def test_resume_matches_uninterrupted_training(tmp_path):
    config = small_config()
    uninterrupted = train(config, tmp_path / "full")
    resumed = train(replace(config, steps=1), tmp_path / "resumed")
    copied_output = tmp_path / "continued"
    copied = train(config, copied_output, resume=resumed)
    main(["--resume", str(resumed), "--steps", "2"])
    with np.load(uninterrupted, allow_pickle=False) as expected:
        with np.load(resumed, allow_pickle=False) as actual:
            assert expected.files == actual.files
            for name in expected.files:
                np.testing.assert_array_equal(expected[name], actual[name])
    assert checkpoint_metadata(resumed)["step"] == 2
    with np.load(uninterrupted, allow_pickle=False) as expected:
        with np.load(copied, allow_pickle=False) as actual:
            for name in expected.files:
                np.testing.assert_array_equal(expected[name], actual[name])


def test_tsp_training_writes_finite_metrics(tmp_path):
    checkpoint = train(small_config(problem="tsp", steps=1), tmp_path)
    assert checkpoint_metadata(checkpoint)["config"]["problem"] == "tsp"
    assert (tmp_path / "evaluation.json").is_file()


def test_existing_run_requires_resume(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("")
    with pytest.raises(FileExistsError, match="--resume"):
        train(small_config(), tmp_path)


def test_pomo_training_rejects_degenerate_starts():
    with pytest.raises(ValueError, match="at least two"):
        small_config(n_starts=1)
    with pytest.raises(ValueError, match="must not exceed"):
        small_config(problem="tsp", n_starts=5)


@pytest.mark.parametrize(
    "artifact", ["metrics.jsonl", "checkpoint.npz", "config.json", "evaluation.json"]
)
def test_resume_cannot_overwrite_another_run(tmp_path, artifact):
    output = tmp_path / "existing"
    output.mkdir()
    existing = output / artifact
    original = b"existing run artifact"
    existing.write_bytes(original)
    resume = tmp_path / "different" / "checkpoint.npz"
    with pytest.raises(FileExistsError, match="new --output"):
        train(small_config(), output, resume=resume)
    assert existing.read_bytes() == original
    assert list(output.iterdir()) == [existing]


def test_nonfinite_evaluation_is_rejected(tmp_path, monkeypatch):
    from neuro_co.core.jax_backend.pomo import JaxPOMO

    monkeypatch.setattr(
        JaxPOMO,
        "greedy_rollout",
        lambda self, params, problem_state: jax.numpy.asarray([float("nan")]),
    )
    with pytest.raises(FloatingPointError, match="greedy evaluation"):
        train(small_config(steps=1), tmp_path)
    assert not (tmp_path / "evaluation.json").exists()
