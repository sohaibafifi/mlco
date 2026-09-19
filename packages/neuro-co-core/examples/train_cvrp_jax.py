"""Train a JAX attention policy on CVRP, with metrics and resumable checkpoints.

    python packages/neuro-co-core/examples/train_cvrp_jax.py --steps 100

Use --problem tsp to train on TSP. See --help for model and training options.
"""

from neuro_co.core.jax_backend.train import main

if __name__ == "__main__":
    main()
