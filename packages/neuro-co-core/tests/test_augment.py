"""Instance-augmentation tests."""

import torch

from neuro_co.core.algos.pomo import POMO, POMOConfig
from neuro_co.core.augment import N_DIHEDRAL, augment_state, best_over_aug, dihedral8
from neuro_co.core.envs.tsp import TSPEnv
from neuro_co.core.models import AttentionModel


def _pairwise_dist(coords: torch.Tensor) -> torch.Tensor:
    return torch.cdist(coords, coords)


def test_dihedral8_shape() -> None:
    coords = torch.rand(3, 7, 2)
    out = dihedral8(coords)
    assert out.shape == (8 * 3, 7, 2)


def test_dihedral8_preserves_distances() -> None:
    """All 8 transforms isometric -> pairwise distances invariant."""
    coords = torch.rand(2, 6, 2)
    out = dihedral8(coords).view(8, 2, 6, 2)
    base = _pairwise_dist(coords)  # (2, 6, 6)
    for k in range(8):
        assert torch.allclose(_pairwise_dist(out[k]), base, atol=1e-5)


def test_dihedral8_stays_in_unit_square() -> None:
    coords = torch.rand(4, 5, 2)
    out = dihedral8(coords)
    assert (out >= 0).all() and (out <= 1).all()


def test_augment_state_tsp() -> None:
    env = TSPEnv(size=6)
    s = env.reset(3)
    s8 = augment_state(s, N_DIHEDRAL)
    assert s8.coords.shape == (24, 6, 2)
    # visited/current tiled (block layout): first 3 rows == original batch.
    assert torch.equal(s8.visited[:3], s.visited)


def test_augment_state_noop() -> None:
    env = TSPEnv(size=6)
    s = env.reset(3)
    assert augment_state(s, 1) is s


def test_best_over_aug() -> None:
    # reward aug-major (n_aug=2, batch=3): block0 then block1.
    reward = torch.tensor([1.0, 2.0, 3.0, 5.0, 1.0, 9.0])
    best = best_over_aug(reward, n_aug=2, batch=3)
    assert torch.equal(best, torch.tensor([5.0, 2.0, 9.0]))


def test_pomo_eval_augment_not_worse() -> None:
    """x8 augmentation keeps the best -> tour length <= non-augmented."""
    env = TSPEnv(size=8)
    model = AttentionModel(in_dim=env.encoder_in_dim, hidden_dim=16, num_layers=1, num_heads=2)
    base = POMO(
        model, env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=64, eval_augment=1)
    )
    aug = POMO(model, env, POMOConfig(batch_size=4, n_starts=3, eval_batch_size=64, eval_augment=8))
    # Same weights (aug shares model object).
    rng1 = torch.Generator().manual_seed(7)
    rng2 = torch.Generator().manual_seed(7)
    t_base = base.eval_step(rng1)["eval_tour_length"]
    t_aug = aug.eval_step(rng2)["eval_tour_length"]
    assert t_aug <= t_base + 1e-4
