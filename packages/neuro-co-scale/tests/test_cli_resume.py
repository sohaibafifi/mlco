"""An interrupted CLI run must retain the same schedule and data stream."""

import torch

from neuro_co.scale.cli import main


def test_cli_resumes_exact_training_state(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        common = [
            "train",
            "--size",
            "12",
            "--batch",
            "2",
            "--starts",
            "2",
            "--hidden-dim",
            "16",
            "--layers",
            "1",
            "--heads",
            "4",
            "--max-customers",
            "4",
            "--capacity",
            "30",
            "--steps",
            "3",
            "--lr-warmup",
            "1",
            "--seed",
            "42",
            "--neighbor-span",
            "1",
            "--reanchor",
            "--checkpoint",
            "--checkpoint-chunk",
            "4",
            "--save-every",
            "1",
            "--log-every",
            "0",
        ]
        full = tmp_path / "full.pt"
        resumed = tmp_path / "resumed.pt"
        assert main([*common, "--out", str(full)]) == 0
        assert main([*common, "--out", str(resumed), "--max-steps-this-run", "1"]) == 0
        partial = torch.load(resumed, weights_only=False)
        assert partial["training_state"]["completed_step"] == 1
        assert main([*common, "--out", str(resumed), "--resume", str(resumed)]) == 0
        a = torch.load(full, weights_only=False)
        b = torch.load(resumed, weights_only=False)
        assert a["config"] == b["config"]
        assert a["training_state"]["history"] == b["training_state"]["history"]
        assert b["training_state"]["completed_step"] == 3
        for k in a["state_dict"]:
            assert torch.equal(a["state_dict"][k], b["state_dict"][k]), k
    finally:
        torch.set_num_threads(old_threads)
