"""Equal-update warm-start ablation of reference-relative anchor goals."""
from pathlib import Path
import json
import subprocess

import tyro
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer
from supermimic.eval.benchmark import load_policy_from_checkpoint, run_benchmark, save_results
from supermimic.envs.mdp.motion_library import discover_npz


def main(checkpoint: Path, run_name: str, anchor: bool = False, iterations: int = 5000):
    cfg = TrainConfig(run_name=run_name, warm_start=checkpoint, seed=0)
    cfg.env.anchor_obs = anchor
    cfg.schedule.stop_at_iteration = iterations
    cfg.schedule.max_hours = 1.0
    cfg.schedule.checkpoint_interval = 2000
    out = cfg.log_root / run_name
    if (out / 'config.json').exists():
        raise FileExistsError(f'run already exists: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out/'provenance.json').write_text(json.dumps({'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), 'checkpoint':str(checkpoint), 'anchor_goal':anchor,'seed':0,'planned_env_steps':iterations,'optimizer_state':'fresh for both variants','lr_schedule_total_steps':320000},indent=2))
    trainer = Trainer(cfg)
    try:
        trainer.run()
    finally:
        trainer.env.close()
    policy,_ = load_policy_from_checkpoint(out/'ckpt_latest.pt','cuda:0')
    results = run_benchmark(policy,[str(f) for f in discover_npz(cfg.data.npz_dir)],num_envs=40,anchor_obs=anchor)
    save_results(results,out/'benchmark_final.json',protocol={'anchor_goal':anchor,'reset_joint_limits':cfg.env.reset_joint_limits,'warm_start':str(checkpoint),'seed':0,'env_steps':iterations})


if __name__ == '__main__':
    tyro.cli(main)
