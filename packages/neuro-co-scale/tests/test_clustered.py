import numpy as np

from neuro_co.scale.datasets import sample_clustered_instances


def _mean_nn_distance(coords: np.ndarray) -> float:
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(d.min(axis=1).mean())


def test_clustered_instances_are_valid_and_clustered() -> None:
    instances = sample_clustered_instances(size=200, num_instances=3, seed=0, capacity=30.0)

    assert len(instances) == 3
    for inst in instances:
        assert inst.num_customers == 200
        assert inst.coords.shape == (201, 2)
        assert inst.demand[0] == 0  # depot
        assert inst.demand[1:].max() <= inst.capacity  # rollout requires this
        assert (inst.coords >= 0).all() and (inst.coords <= 1).all()

    # Clustered customers sit much closer together than uniform ones.
    rng = np.random.default_rng(0)
    clustered_nn = _mean_nn_distance(instances[0].coords[1:])
    uniform_nn = _mean_nn_distance(rng.random((200, 2)))
    assert clustered_nn < 0.5 * uniform_nn


def test_clustered_is_deterministic_per_seed() -> None:
    a = sample_clustered_instances(size=50, num_instances=1, seed=7)[0]
    b = sample_clustered_instances(size=50, num_instances=1, seed=7)[0]
    assert np.array_equal(a.coords, b.coords)
    assert np.array_equal(a.demand, b.demand)
