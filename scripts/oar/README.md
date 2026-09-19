# OAR scripts

`train.oar` submits 12 tasks: `cvrptw`, `op`, `pdp`, and `fjsp`, each with three
seeds. `eval.oar` submits one evaluation task per problem and runs CAX
adjudication over the saved checkpoints.

These are cluster templates. Install the workspace, including `neuro-co-core`
and `neuro-co-problems`, in the job environment before submission. The scripts
use `uv run --no-sync` and leave dependency installation to that setup step.
Review host constraints, modules, the uv path, resources, and working directory.
Match evaluation size and algorithm settings to the training run.

```bash
oarsub -S ./train.oar
oarsub -S ./eval.oar
```

Run evaluation after the training array has completed. Outputs use
`outputs/<problem>/train_seed<S>/` relative to the job's working directory and
contain `best.pt`, `latest.pt`, and `metrics.json`. These scripts do not produce
paired classical-baseline or AET measurement bundles.
