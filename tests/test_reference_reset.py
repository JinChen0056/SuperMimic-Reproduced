from types import SimpleNamespace

import pytest
import torch

from supermimic.envs.mdp.commands import MultiClipMotionCommand


@pytest.mark.parametrize('mode,expected', [('soft', 0.9), ('hard', 0.95)])
def test_reference_reset_can_preserve_valid_pose_outside_soft_limits(mode, expected):
    class Robot:
        data = SimpleNamespace(
            soft_joint_pos_limits=torch.tensor([[[-0.9, 0.9]]]).expand(1, 29, 2),
            joint_pos_limits=torch.tensor([[[-1.0, 1.0]]]).expand(1, 29, 2),
        )
        def write_joint_state_to_sim(self, pos, vel, **kwargs):
            self.written = pos
        def write_root_state_to_sim(self, *args, **kwargs):
            pass
        def reset(self, **kwargs):
            pass
    robot = Robot()
    cmd = SimpleNamespace(robot=robot, cfg=SimpleNamespace(reset_joint_limits=mode))
    root = torch.zeros(1, 3)
    MultiClipMotionCommand._write_reference_state_to_sim(
        cmd, torch.tensor([0]), root, torch.tensor([[1., 0., 0., 0.]]),
        root, root, torch.full((1, 29), .95), torch.zeros(1, 29))
    torch.testing.assert_close(robot.written, torch.full((1, 29), expected))
