from dataclasses import replace

import torch
from torch import nn

from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
from supermimic.training.checkpoints import warm_start_models


def test_anchor_expansion_preserves_initial_actor_and_critic_outputs():
    cfg = PolicyCfg(encoder_hidden=(16,), decoder_hidden=(16,))
    old_policy = Policy(cfg)
    old_critic = Critic(CriticCfg(hidden=(16,)), pcfg=cfg)
    old_wm = nn.Linear(2, 2)
    ck = {'wm':old_wm.state_dict(), 'policy':old_policy.state_dict(),
          'critic':old_critic.state_dict(), 'config':{'policy':{'goal_dim':680}}}
    new_cfg = replace(cfg, goal_dim=689)
    policy = Policy(new_cfg)
    critic = Critic(CriticCfg(hidden=(16,)), pcfg=new_cfg)
    wm = nn.Linear(2, 2)
    warm_start_models(wm, policy, critic, ck)
    obs = torch.randn(4, 930)
    goal = torch.randn(4, 680)
    extra = torch.randn(4, 9)
    feat, xi, error = torch.randn(4,71), torch.randn(4,33), torch.randn(4,4)
    torch.testing.assert_close(policy(obs,torch.cat([goal,extra],-1)),old_policy(obs,goal))
    torch.testing.assert_close(critic(feat,torch.cat([goal,extra],-1),xi,error),old_critic(feat,goal,xi,error))
    torch.testing.assert_close(wm.weight,old_wm.weight)
