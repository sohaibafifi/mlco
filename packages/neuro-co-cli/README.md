# neuro-co-cli

The `neuroco` command trains, evaluates, and explains core policies.

```bash
uv run --no-sync neuroco train --problem cvrp --algo pomo --size 20 \
  --epochs 1 --steps-per-epoch 2 --batch-size 8 --out-dir outputs/cvrp
uv run --no-sync neuroco eval --problem cvrp --algo pomo --size 20 \
  --ckpt-path outputs/cvrp/best.pt
uv run --no-sync neuroco explain --problem cvrp --size 20 \
  --ckpt-path outputs/cvrp/best.pt --out-dir outputs/cvrp-explain
```

Training writes `best.pt`, `latest.pt`, and `metrics.json` under `--out-dir`.
Use matching problem, size, and algorithm settings when evaluating.
`neuroco probe`, `results`, `stats`, `figures`, `experiment`, and `budget` have their own
`--help` pages. Energy reports use the separate `neuroco-aet-report` command.

See [training](../../docs/training.md) for backbones and JAX.
