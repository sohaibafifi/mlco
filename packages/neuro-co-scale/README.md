# neuro-co-scale

Partitioning, hierarchical policies, and evaluation tools for large CVRP and
CVRPTW instances. The package includes synthetic and CVRPLIB loaders, route
recomposition, repair, and classical partition baselines.

```bash
uv run --no-sync neuroco-scale partition-smoke --size 50 --max-customers 10
uv run --no-sync neuroco-scale train --size 20 --max-customers 10 \
  --steps 2 --batch 2 --starts 2 --hidden-dim 32 --layers 1 --heads 4 \
  --device cpu --out outputs/scale/smoke.pt
uv run --no-sync neuroco-scale eval --ckpt outputs/scale/smoke.pt --eval-instances 4
```

The default `cluster_local` encoder operates within bounded clusters. `am` uses
a dense attention encoder. Partition methods include sweep, capacity-aware
sweep, grid, Morton order, Morton refinement, and feature-aware clustering.
Feature-aware clustering builds a dense compatibility matrix, so its memory
cost limits the instance sizes it can handle.

See [scaling](../../docs/scale.md) for checkpoint resume and evaluation.
