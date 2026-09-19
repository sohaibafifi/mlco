"""Export policy actions without retaining an analysis trace."""

import torch
from jaxtyping import Int
from torch import Tensor

from .env import Env, get_dynamic_decoder_context
from .models.policy import ConstructivePolicy
from .state import State


@torch.no_grad()
def greedy_rollout_actions(
    model: ConstructivePolicy,
    env: Env,
    state: State,
) -> Int[Tensor, "b t"]:
    """Decode initially unfinished episodes and return their selected actions.

    Rows contain ``-1`` after completion. Columns stop when the whole batch
    finishes. TSP omits the initial city and implicit closing edge; CVRP includes
    every selected depot return. The caller supplies any initial route prefix.

    Inference uses evaluation mode and restores each module's previous mode.
    Raise ``RuntimeError`` if any episode exceeds the environment's step bound.
    """
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        features = env.build_features(state)
        node_embs, graph_emb = model.encode(features)
        decoder_cache = model.precompute_decoder_cache(node_embs)
        batch_size = features.shape[0]
        max_steps = env.max_steps(state)
        actions = torch.full((batch_size, max_steps), -1, dtype=torch.long, device=features.device)
        done_acc = torch.zeros(batch_size, dtype=torch.bool, device=features.device)

        for step in range(max_steps):
            mask = env.action_mask(state)
            # Finished rows still pass through the batched decoder.
            mask = mask | done_acc.unsqueeze(1)
            first_idx, current_idx = env.decoder_context(state)
            logits = model.decode_step(
                node_embs,
                graph_emb,
                first_idx,
                current_idx,
                mask,
                dynamic_context=get_dynamic_decoder_context(env, state),
                decoder_cache=decoder_cache,
            )
            action = logits.masked_fill(~mask, -torch.inf).argmax(dim=-1)
            actions[:, step] = action.masked_fill(done_acc, -1)
            state, _, done = env.step(state, action.masked_fill(done_acc, 0))
            done_acc = done_acc | done
            if bool(done_acc.all()):
                return actions[:, : step + 1]

        raise RuntimeError(f"Greedy rollout did not finish within {max_steps} steps")
    finally:
        for module, training in modes:
            module.training = training


__all__ = ["greedy_rollout_actions"]
