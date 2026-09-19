# Training

## Torch and JAX

Use one command and one set of parameters for both backends:

```bash
uv sync --all-packages --extra jax
bash scripts/run_all.sh
```

Defaults: TSP and CVRP, size 20, AM with 128 hidden units, 3 layers and 8 heads,
POMO, 10 epochs of 100 updates, batch size 128, Adam at `1e-4`, weight decay
`1e-6`, gradient clipping at 1, fp32, seed 42, and CPU. POMO uses up to 20 valid
starts. CVRP capacity is 50 and customer demands range from 1 to 9.

Both backends use identical initial weights and freshly generated training
batches. Validation and test sets each contain 512 identical instances across
backends, with separate seeds and streams. Each epoch validates on the same set;
the best checkpoint is evaluated greedily on the separate test set. Framework
random samplers differ, so sampled actions and learned weights need not match.

Set parameters once to apply them to both backends:

```bash
EPOCHS=10 STEPS=100 BATCH=128 SIZE=20 N_STARTS=19 \
  HIDDEN_DIM=128 NUM_LAYERS=3 NUM_HEADS=8 DEVICE=cpu \
  OUT_ROOT=outputs/comparison-1 bash scripts/run_all.sh
```

`STEPS` means updates per epoch. Other options include `EVAL_BATCH`, `LR`,
`SEED`, `EVAL_SEED`, `TEST_SEED`, `CAPACITY`, `MAX_DEMAND`, `OPTIMIZER`,
`WEIGHT_DECAY`, and `GRAD_CLIP`. Use `DEVICE=cuda` only when both frameworks have
CUDA support. `BACKENDS=torch` or `BACKENDS=jax` selects a single backend.

Runs go to `outputs/matched/<backend>/<problem>` by default. Existing runs are
preserved; choose a new `OUT_ROOT` when retraining. Progress shows steps, elapsed
time, and training ETA on stderr. Each run saves its configuration, checkpoints,
metrics, test costs, data hashes, and timings. `comparison.csv` and
`comparison.json` collect the results and indicate whether settings and hardware
match. Training time includes the first JAX compilation; warm update time excludes
the first update. Validation and test inference are timed separately. Standalone
evaluation measures a first call and a warm call in a fresh process.

Reevaluate selected checkpoints without training:

```bash
SKIP_TRAIN=1 OUT_ROOT=outputs/comparison-1 bash scripts/run_all.sh
```

For a single run, change only `--backend`:

```bash
uv run --no-sync neuroco train --backend torch --problem tsp --out-dir outputs/tsp-torch
uv run --no-sync neuroco train --backend jax --problem tsp --out-dir outputs/tsp-jax
```

`scripts/train_jax.py` delegates to the same CLI and accepts the same training
parameters. Its legacy `--steps N` alias means one epoch of N updates. Legacy
native JAX runs remain available through `python -m neuro_co.core.jax_backend.train`;
they use the older protocol and should not be mixed into this comparison.

## Other Torch problems and models

Torch also supports `atsp`, `cvrptw`, `op`, `pdp`, `mtsp`, and `fjsp`:

```bash
BACKENDS=torch PROBLEMS="tsp atsp cvrp cvrptw op pdp mtsp fjsp" \
  OUT_ROOT=outputs/torch-problems bash scripts/run_all.sh
```

`FJSP_SIZE` sets the number of jobs, default 10. The Torch CLI also accepts
`--algo reinforce|ppo` and `--backbone gnn|matnet|mamba`; these have no matching
JAX training path. Backbone compatibility depends on the problem representation.
Install the core `gnn` or `mamba` extra before using those models.

```bash
uv sync --all-packages --extra mamba
uv run --no-sync neuroco train --problem tsp --backbone mamba --algo reinforce \
  --epochs 1 --steps-per-epoch 2 --batch-size 8 --size 20 \
  --hidden-dim 32 --num-layers 1 --device cpu --out-dir outputs/mamba
```

## Exporting solutions

`neuro_co.core.inference.greedy_rollout_actions(model, env, state)` returns Torch
actions without retaining a full trace. JAX provides
`JaxPOMO.greedy_rollout_actions(params, state)`, which supports `jax.jit`.
Both accept an initially unfinished environment state and return integer arrays
with shape `(batch, steps)`. Completed rows use `-1` padding; Torch trims trailing
columns once the batch finishes. Add the initial city when exporting a TSP tour.
CVRP actions include depot returns, so split the sequence at each return.
