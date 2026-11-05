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

`neuroco solve instance.vrp --ckpt outputs/cvrp/best.pt --out solution.json`
solves supplied data and exports a common result with independent routing checks.
Without a checkpoint, the default is the nearest-neighbour reference heuristic.
See [solving instances](../../docs/solving.md) for formats, status, and solver options.
