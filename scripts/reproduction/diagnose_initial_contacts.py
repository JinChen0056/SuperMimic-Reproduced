"""Audit initial-state contact counts for failures and matched random successes."""
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from supermimic.envs import make_env, SuperMimicEnvCfgArgs


def main():
    pack=torch.load('data/library/bones_seed_compact.pt',map_location='cpu',mmap=True,weights_only=False)
    result=json.loads(Path('logs/bones_remote_final/benchmark_full.json').read_text())['clips']
    assert pack['names']==[r['name'] for r in result]
    failed=np.array([r['failed'] for r in result])
    selection=np.concatenate([np.where(failed)[0],np.random.default_rng(0).choice(np.where(~failed)[0],705,replace=False)])
    qpos=pack['ref'][torch.tensor([pack['starts'][i] for i in selection]),:36].float().numpy()
    env=make_env(SuperMimicEnvCfgArgs(npz_files=['data/npz/lafan1/walk1_subject1.npz'],num_envs=1,record_transitions=False,domain_randomization=False),'cuda:0')
    try:
        model=env.sim.mj_model
        origin=env.scene.env_origins[0].cpu().numpy()
        soft=env.scene['robot'].data.soft_joint_pos_limits[0].cpu().numpy()
        data=mujoco.MjData(model)
        rows=[]
        for j,i in enumerate(selection):
            data.qpos[:]=qpos[j];data.qpos[:3]+=origin
            data.qpos[7:]=np.clip(data.qpos[7:],soft[:,0],soft[:,1])
            data.qvel[:]=0.
            mujoco.mj_forward(model,data)
            rows.append({'name':pack['names'][i],'failed':bool(failed[i]),'ncon':data.ncon,'nefc':data.nefc})
        out={}
        for is_failure,label in [(True,'failed705'),(False,'success705')]:
            group=[r for r in rows if r['failed']==is_failure]
            out[label]={'max_ncon':max(r['ncon'] for r in group),'above_ncon35':sum(r['ncon']>35 for r in group),'max_nefc':max(r['nefc'] for r in group),'above_nefc250':sum(r['nefc']>250 for r in group)}
        out['scope']='CPU forward of initial poses, not trajectory overflow measurement. nconmax is a shared pool estimate; per-world ncon>35 alone is NOT an overflow. njmax is per world.'
        Path('../../artifacts/initial_contact_audit.json').write_text(json.dumps({'summary':out,'clips':rows},indent=2))
        print(json.dumps(out,indent=2))
    finally:
        env.close()


if __name__=='__main__':main()
