"""Small diagnostic run: real simulator, WM fit, BPTT, critic and checkpoint."""
from supermimic.config import TrainConfig
from supermimic.training.loop import Trainer

cfg = TrainConfig(run_name="local_smoke_s0", seed=0)
cfg.data.clips = ["walk1_subject1", "dance1_subject1", "jumps1_subject1"]
cfg.env.num_envs = 32
cfg.env.replay_capacity = 128
cfg.world_model.hidden = (128, 128)
cfg.policy.encoder_hidden = (128, 64)
cfg.policy.decoder_hidden = (128, 64)
cfg.wm_trainer.batch_size = 32
cfg.wm_trainer.batches_per_update = 1
cfg.wm_trainer.compile = None
cfg.policy_trainer.batch_size = 32
cfg.policy_trainer.horizon = 8
cfg.policy_trainer.compile = None
cfg.schedule.total_iterations = 60
cfg.schedule.warmup_iterations = 20
cfg.schedule.log_interval = 20
cfg.schedule.checkpoint_interval = 60
cfg.schedule.max_hours = 0.25
trainer = Trainer(cfg)
try:
    trainer.run()
    assert trainer.num_updates > 0
finally:
    trainer.env.close()
