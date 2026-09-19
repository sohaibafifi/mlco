# Scaling

`neuroco-scale` partitions CVRP or CVRPTW instances and trains hierarchical
policies that select clusters and customers. Its `cluster_local` encoder limits
attention to clusters; the `am` encoder uses dense attention.

## Partition and train

```bash
uv run --no-sync neuroco-scale partition-smoke --method morton --size 50 --max-customers 10
uv run --no-sync neuroco-scale train --size 20 --max-customers 10 \
  --steps 2 --batch 2 --starts 2 --hidden-dim 32 --layers 1 --heads 4 \
  --device cpu --out outputs/scale/smoke.pt
uv run --no-sync neuroco-scale eval --ckpt outputs/scale/smoke.pt --eval-instances 4
```

Partition methods include sweep, capacity-aware sweep, grid, Morton order,
Morton refinement, and feature-aware clustering. Feature-aware clustering uses
a dense compatibility matrix. Morton partitioning uses sorting and bounded
clusters, but total training cost also depends on the encoder and decoding loop.

`--neighbor-span` allows routes to cross neighboring Morton clusters, and
`--reanchor` selects another cluster when a route exhausts its neighborhood.
`--checkpoint` reduces training memory through gradient checkpointing.

## Resume training

`--resume` restores the model, optimizer, scheduler, and random states.
`--init-ckpt` loads model weights to start a new training run.
`--steps` sets the total schedule; `--max-steps-this-run` limits the current
invocation without changing that schedule. Resume with the same model, sampler,
and training settings; the runner checks the configuration and source hashes.

```bash
uv run --no-sync neuroco-scale train --size 20 --max-customers 10 \
  --steps 10 --max-steps-this-run 2 --batch 2 --starts 2 \
  --hidden-dim 32 --layers 1 --heads 4 --device cpu \
  --out outputs/scale/resume.pt
uv run --no-sync neuroco-scale train --resume outputs/scale/resume.pt \
  --size 20 --max-customers 10 --steps 10 --batch 2 --starts 2 \
  --hidden-dim 32 --layers 1 --heads 4 --device cpu \
  --out outputs/scale/resume.pt
```

## Evaluation

`eval` supports synthetic instances and CVRPLIB directories containing `.vrp`
files. Use `--instances-dir` to select a directory; matching `.sol` files supply
reference costs. `--pyvrp-time` adds an optional PyVRP runtime budget per instance.
Use held-out instances and report feasibility, solution cost, runtime, and the
full configuration before drawing conclusions about scale or quality.

Run `uv run --no-sync neuroco-scale <command> --help` for the complete option list.
