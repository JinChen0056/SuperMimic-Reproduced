"""Transition labels must stay aligned with the actions that produced them."""
import pytest
import torch

from supermimic.data.replay_buffer import ReplayBuffer


@pytest.mark.parametrize('terminal_row_label', [0.0, 1.0])
def test_window_uses_done_of_executed_transition(terminal_row_label):
    buf = ReplayBuffer(1, 16, 71, 29, 33, 'cpu')
    ids = torch.tensor([0])
    # s0 --a0, alive--> s1 --a1, terminal--> s2. The terminal
    # row has no executed action, so its bookkeeping label is not a target.
    for i, (done, terminal) in enumerate([(0.0, False), (1.0, False), (terminal_row_label, True)]):
        buf.add(ids, s=torch.full((1, 71), float(i)), a=torch.zeros(1, 29),
                a_exec=torch.full((1, 29), float(i)), xi=torch.zeros(1, 33),
                r=torch.tensor([float(i)]), done=torch.tensor([done]),
                frame=torch.tensor([i]), clip=torch.tensor([0]),
                terminal=torch.tensor([terminal]), ep_start=torch.tensor([i == 0]))
    batch = buf.sample_windows(8, 2)
    torch.testing.assert_close(batch.done, torch.tensor([[0.0] * 8, [1.0] * 8]))
    torch.testing.assert_close(batch.states[:, 0, 0], torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(batch.a_exec[:, 0, 0], torch.tensor([0.0, 1.0]))
