import pytest
import torch

from neuro_co.scale.hierarchical import (
    LinearClusterSelector,
    active_cluster_mask,
    aggregate_cluster_embeddings,
    build_cluster_index,
    cluster_log_prob,
    gather_selected_cluster_action_set,
    mask_cluster_logits,
    scatter_local_logits,
    select_cluster,
)
from neuro_co.scale.types import Partition


def _partition() -> Partition:
    return Partition(clusters=((1, 3), (2, 4)), method="unit")


def test_build_cluster_index_maps_global_customers() -> None:
    index = build_cluster_index(_partition(), num_nodes=5)

    assert index.num_clusters == 2
    assert index.max_cluster_size == 2
    assert index.max_local_actions == 3
    assert index.customer_to_cluster == (-1, 0, 1, 0, 1)
    assert index.cluster_of(3) == 0
    assert index.cluster_of(4) == 1


def test_aggregate_cluster_embeddings_mean_pools_customers() -> None:
    index = build_cluster_index(_partition(), num_nodes=5)
    node_embs = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        ]
    )

    cluster_embs = aggregate_cluster_embeddings(node_embs, index)

    assert cluster_embs.shape == (1, 2, 2)
    assert torch.allclose(cluster_embs[0, 0], torch.tensor([2.0, 20.0]))
    assert torch.allclose(cluster_embs[0, 1], torch.tensor([3.0, 30.0]))


def test_active_cluster_mask_checks_remaining_customers() -> None:
    index = build_cluster_index(_partition(), num_nodes=5)
    remaining = torch.tensor(
        [
            [False, False, True, False, False],
            [False, False, False, True, False],
        ]
    )

    active = active_cluster_mask(remaining, index)

    assert active.tolist() == [[False, True], [True, False]]


def test_mask_cluster_logits_masks_inactive_clusters() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    active = torch.tensor([[True, False, True]])

    masked = mask_cluster_logits(logits, active)

    assert masked[0, 0] == 1.0
    assert masked[0, 1] < -1.0e8
    assert masked[0, 2] == 3.0


def test_linear_cluster_selector_scores_and_masks_clusters() -> None:
    selector = LinearClusterSelector(hidden_dim=2, bias=False)
    with torch.no_grad():
        selector.score.weight.copy_(torch.tensor([[1.0, 0.5]]))
    cluster_embs = torch.tensor([[[1.0, 2.0], [4.0, 0.0], [0.0, 1.0]]])
    active = torch.tensor([[True, False, True]])

    logits = selector(cluster_embs, active)

    assert logits.shape == (1, 3)
    assert torch.allclose(logits[0, 0], torch.tensor(2.0))
    assert logits[0, 1] < -1.0e8
    assert torch.allclose(logits[0, 2], torch.tensor(0.5))


def test_select_cluster_greedy_and_log_prob() -> None:
    logits = torch.tensor([[0.0, 3.0, 1.0], [5.0, 1.0, 0.0]])
    active = torch.tensor([[True, True, True], [False, True, True]])

    selected = select_cluster(logits, active, mode="greedy")
    logp = cluster_log_prob(logits, selected, active)

    assert selected.tolist() == [1, 1]
    assert logp.shape == (2,)
    assert torch.isfinite(logp).all()


def test_gather_selected_cluster_action_set_slices_depot_and_selected_cluster() -> None:
    index = build_cluster_index(_partition(), num_nodes=5)
    node_embs = torch.arange(2 * 5 * 2, dtype=torch.float32).reshape(2, 5, 2)
    selected = torch.tensor([1, 0])
    global_mask = torch.tensor(
        [
            [True, True, False, True, True],
            [True, True, True, False, True],
        ]
    )

    action_set = gather_selected_cluster_action_set(
        node_embs,
        index,
        selected,
        global_action_mask=global_mask,
    )

    assert action_set.global_indices.tolist() == [[0, 2, 4], [0, 1, 3]]
    assert action_set.mask.tolist() == [[True, False, True], [True, True, False]]
    assert torch.equal(action_set.node_embs[0, 1], node_embs[0, 2])
    assert torch.equal(action_set.node_embs[1, 2], node_embs[1, 3])


def test_scatter_local_logits_returns_global_logits() -> None:
    index = build_cluster_index(_partition(), num_nodes=5)
    node_embs = torch.zeros(1, 5, 2)
    action_set = gather_selected_cluster_action_set(node_embs, index, torch.tensor([1]))
    local_logits = torch.tensor([[0.5, 1.0, 2.0]])

    global_logits = scatter_local_logits(local_logits, action_set, num_nodes=5)

    assert global_logits.shape == (1, 5)
    assert global_logits[0, 0] == 0.5
    assert global_logits[0, 2] == 1.0
    assert global_logits[0, 4] == 2.0
    assert global_logits[0, 1] < -1.0e8
    assert global_logits[0, 3] < -1.0e8


def test_scatter_local_logits_preserves_depot_with_padded_clusters() -> None:
    # Cluster 0 has size 1, so its action set is padded; padded slots default to
    # global index 0 (depot) and must not clobber the real depot logit.
    index = build_cluster_index(Partition(clusters=((1,), (2, 3, 4)), method="unit"), num_nodes=5)
    node_embs = torch.zeros(1, 5, 2)
    action_set = gather_selected_cluster_action_set(node_embs, index, torch.tensor([0]))

    assert action_set.global_indices.tolist() == [[0, 1, 0, 0]]
    local_logits = torch.tensor([[0.5, 1.5, -3.0, -4.0]])
    global_logits = scatter_local_logits(local_logits, action_set, num_nodes=5)

    assert global_logits.shape == (1, 5)
    assert global_logits[0, 0] == 0.5  # depot logit survives the padding
    assert global_logits[0, 1] == 1.5
    assert global_logits[0, 2] < -1.0e8
    assert global_logits[0, 3] < -1.0e8
    assert global_logits[0, 4] < -1.0e8


def test_build_cluster_index_rejects_invalid_partition() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        build_cluster_index(Partition(clusters=((1,), (1,)), method="bad"), num_nodes=3)
    with pytest.raises(ValueError, match="missing"):
        build_cluster_index(Partition(clusters=((1,),), method="bad"), num_nodes=3)
    with pytest.raises(ValueError, match="depot"):
        build_cluster_index(Partition(clusters=((0, 1),), method="bad"), num_nodes=2)


def test_select_cluster_rejects_missing_active_cluster() -> None:
    logits = torch.tensor([[1.0, 2.0]])
    active = torch.tensor([[False, False]])

    with pytest.raises(ValueError, match="active cluster"):
        select_cluster(logits, active)
