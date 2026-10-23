"""Warm-start models while allowing an explicit nine-feature anchor-goal ablation."""
from __future__ import annotations

import torch


def _insert_zero_columns(weight: torch.Tensor, at: int, count: int) -> torch.Tensor:
    return torch.cat([weight[:, :at], weight.new_zeros(weight.shape[0], count), weight[:, at:]], dim=1)


def warm_start_models(wm, policy, critic, checkpoint: dict) -> None:
    old_goal = checkpoint['config']['policy'].get('goal_dim', 680)
    new_goal = policy.cfg.goal_dim
    extra = new_goal - old_goal
    if extra not in (0, 9):
        raise ValueError('warm-start supports unchanged goals or nine additional anchor features')
    actor_state = dict(checkpoint['policy'])
    critic_state = dict(checkpoint['critic'])
    if extra:
        actor_key = 'encoders.motion.mlp.net.0.weight'
        actor_state[actor_key] = _insert_zero_columns(actor_state[actor_key], old_goal, extra)
        actor_state['encoders.motion.scale'] = policy.state_dict()['encoders.motion.scale']
        critic_key = 'mlp.net.0.weight'
        # Critic input order is canonical state | goal | context | root errors.
        state_dim = critic_state[critic_key].shape[1] - old_goal - 33 - 4
        critic_state[critic_key] = _insert_zero_columns(critic_state[critic_key], state_dim + old_goal, extra)
        critic_state['goal_scale'] = critic.state_dict()['goal_scale']
    wm.load_state_dict(checkpoint['wm'])
    policy.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
