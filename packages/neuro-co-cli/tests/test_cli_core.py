"""neuroco CLI on core: train/eval/explain roundtrip."""

import pytest

from neuro_co.cli import experiment
from neuro_co.cli.main import main


def test_train_eval_explain_roundtrip(tmp_path) -> None:
    run = str(tmp_path / "run")
    common = ["--size", "6", "--hidden-dim", "16", "--num-layers", "1", "--num-heads", "2"]
    main(
        [
            "train",
            "--problem",
            "tsp",
            "--epochs",
            "1",
            "--steps-per-epoch",
            "2",
            "--batch-size",
            "16",
            "--out-dir",
            run,
            *common,
        ]
    )
    assert (tmp_path / "run" / "best.pt").exists()
    assert (tmp_path / "run" / "metrics.json").exists()

    main(["eval", "--problem", "tsp", "--size", "6", "--ckpt-path", run + "/best.pt"])

    exp = str(tmp_path / "exp")
    main(
        [
            "explain",
            "--problem",
            "tsp",
            "--size",
            "6",
            "--ckpt-path",
            run + "/best.pt",
            "--num-instances",
            "4",
            "--top-k",
            "3",
            "--out-dir",
            exp,
        ]
    )
    assert (tmp_path / "exp" / "explanation.json").exists()


def test_make_test_set(tmp_path) -> None:
    from neuro_co.cli.make_test_set import make_test_set

    out = make_test_set("tsp", num_instances=8, size=6, out_path=str(tmp_path / "ts.pt"))
    assert out.exists()


def test_figures_command_writes_png_from_results(tmp_path) -> None:
    pytest.importorskip("pandas")
    pytest.importorskip("matplotlib")
    source = tmp_path / "results.csv"
    source.write_text(
        "problem,kind,method,seed,layer,concept,metric,value\n"
        "cvrp,probe,,1,0,high_demand,val_acc,0.8\n"
        "cvrp,probe,,2,0,high_demand,val_acc,0.9\n",
        encoding="utf-8",
    )
    output = tmp_path / "figures"
    with pytest.raises(SystemExit) as result:
        main(["figures", str(source), "--out", str(output), "--ext", "png"])
    assert result.value.code == 0
    assert (output / "probe_accuracy_heatmap.png").read_bytes().startswith(b"\x89PNG\r\n")
    assert (output / "cross_seed_stability.png").read_bytes().startswith(b"\x89PNG\r\n")


def test_factory_all_problems() -> None:
    from neuro_co.core.factory import ENV_BUILDERS, make_env, make_model

    for name in ENV_BUILDERS:
        env = make_env(name, size=6) if name != "fjsp" else make_env(name, size=3)
        assert make_model(env, hidden_dim=8, num_layers=1, num_heads=2) is not None


def test_experiment_entrypoint_routes_only_to_generic_executor(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("name: generic\n", encoding="utf-8")
    observed = []
    monkeypatch.setattr(
        experiment,
        "run_experiment",
        lambda path: observed.append(path) or 7,
    )

    assert experiment.main_run_cli([str(recipe)]) == 7
    assert observed == [recipe]


def test_experiment_entrypoint_rejects_package_specific_flags(tmp_path) -> None:
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("name: generic\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        experiment.main_run_cli(["--dry-run", str(recipe)])
