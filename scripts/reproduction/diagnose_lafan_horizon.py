"""Separate 10-second prefix diagnostics from full-clip LAFAN1 evaluation."""
import json
from pathlib import Path
import torch
from supermimic.envs import make_env, SuperMimicEnvCfgArgs
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import load_policy_from_checkpoint


def main():
    files=discover_npz(Path('data/npz/lafan1'))
    policy,_=load_policy_from_checkpoint('logs/lafan1_strict_s0_20261023/ckpt_latest.pt','cuda:0')
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[str(p) for p in files],num_envs=len(files),obs_noise=False,reset_noise=False,domain_randomization=False,action_noise_std=0.,record_transitions=False,episode_length_s=10.,sampling_mode='start'),'cuda:0')
    try:
        cmd=env.command_manager.get_term('motion')
        cmd.forced_frame[:]=cmd.library.clip_start
        obs,_=env.reset()
        alive=torch.ones(len(files),dtype=torch.bool,device='cuda')
        failed=torch.zeros_like(alive)
        sums=torch.zeros(len(files),device='cuda')
        with torch.no_grad():
            for _ in range(500):
                obs,r,term,tout,_=env.step(policy(obs['policy'],obs['goal']))
                sums+=torch.where(alive,r,0.)
                failed|=alive&term
                alive&=~(term|tout)
        scores=sums/(.1*500)
        result={'protocol':'diagnostic first 10 seconds, not confirmed author LAFAN protocol','mean_norm_return':float(scores.mean()),'failed':int(failed.sum()),'clips':len(files),'max_steps':500,'per_clip':dict(zip([p.stem for p in files],scores.tolist()))}
        Path('../../artifacts/lafan_prefix10s_diagnostic.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    finally:
        env.close()


if __name__=='__main__':main()
