# POMO memory and inference options

`POMOConfig.microbatch_size` limits the number of instances processed by each
training forward/backward pass. `batch_size` remains the logical batch size.
For example, `POMOConfig(batch_size=256, microbatch_size=32)` accumulates eight
weighted gradients before one optimizer, scheduler and EMA update. A smaller
last microbatch is weighted by its instance count. When `train_step` receives
an explicit state, that state's batch size determines these weights.

The default `None` retains a single full-batch pass. Microbatching preserves the
per-instance POMO baseline, but batch-normalization statistics and the order of
random action draws can change. It is not a promise of identical trajectories
or numerical equivalence to a full-batch forward pass.

For inference, `eval_augment` accepts 1, 2, 4 or 8 dihedral transforms. The sets
are nested and include the original instance. Larger sets retain the best
solution per instance at an additional inference cost.
