"""Check whether fixed-reference control can observe a robot-only global drift."""
import json
from pathlib import Path

import torch
from supermimic.data.motion_convert import load_g1_model
from supermimic.envs import make_env, SuperMimicEnvCfgArgs
from supermimic.envs.state_io import StateIO
from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.kinematics.rotations import quat_from_yaw, quat_mul, quat_apply
from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major, anchor_relative_obs
from supermimic.world_model.wm_trainer import reference_at


def main():
    env = make_env(SuperMimicEnvCfgArgs(npz_files=['data/npz/lafan1/walk1_subject1.npz'], num_envs=1, obs_noise=False, reset_noise=False, domain_randomization=False, record_transitions=False), 'cuda:0')
    try:
        env.reset()
        cmd = env.command_manager.get_term('motion')
        robot = env.scene['robot']
        q0 = robot.data.default_joint_pos[0].clone()
        h = AnalyticHeads(G1Kinematics(load_g1_model(), device='cuda:0'), TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0].clone(), q0)
        s = StateIO(env).read()
        shifted = s.clone()
        shifted[:, :2] += torch.tensor([[1., .5]], device='cuda')
        yaw = quat_from_yaw(torch.tensor([.8], device='cuda'))
        shifted[:, 3:7] = quat_mul(yaw, s[:, 3:7])
        shifted[:, 36:39] = quat_apply(yaw, s[:, 36:39])
        xi = torch.zeros(1,33,device='cuda'); xi[:,0]=1
        prev = torch.zeros(1,29,device='cuda')
        o1 = analytic_observation(s,prev,xi,q0)
        o2 = analytic_observation(shifted,prev,xi,q0)
        goal = cmd.goal
        policy,_ = load_policy_from_checkpoint('logs/lafan1_strict_s0_20261023/ckpt_latest.pt','cuda:0')
        with torch.no_grad():
            a1 = policy(flatten_history_term_major(o1[None].expand(10,-1,-1)),goal)
            a2 = policy(flatten_history_term_major(o2[None].expand(10,-1,-1)),goal)
            ref = reference_at(cmd.library,cmd.frame)
            r1 = h.reward(s,a1,prev,ref); r2 = h.reward(shifted,a2,prev,ref)
            b1,b2 = h.robot_bodies(s),h.robot_bodies(shifted)
            i=h.anchor_index
            rel1 = anchor_relative_obs(b1['pos'][:,i],b1['quat'][:,i],ref.body_pos_w[:,i],ref.body_quat_w[:,i])
            rel2 = anchor_relative_obs(b2['pos'][:,i],b2['quat'][:,i],ref.body_pos_w[:,i],ref.body_quat_w[:,i])
        out={'robot_only_translation_m':[1.,.5], 'robot_only_yaw_rad':.8,
             'proprio_max_abs_difference':float((o1-o2).abs().max()),
             'action_max_abs_difference':float((a1-a2).abs().max()),
             'anchor_goal_max_abs_difference':float((rel1-rel2).abs().max()),
             'root_rewards_before':{k:float(r1[k]) for k in ('motion_global_root_pos','motion_global_root_ori')},
             'root_rewards_after':{k:float(r2[k]) for k in ('motion_global_root_pos','motion_global_root_ori')}}
        assert out['action_max_abs_difference'] < 1e-5
        assert out['anchor_goal_max_abs_difference'] > .1
        Path('../../artifacts/policy_drift_observability.json').write_text(json.dumps(out,indent=2))
        print(json.dumps(out,indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
