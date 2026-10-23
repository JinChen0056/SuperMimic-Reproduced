import pytest

from supermimic.eval.benchmark import ClipResult, normalized_return, summarize


def test_early_failure_cannot_score_perfect_tracking():
    assert normalized_return(1.0, frames=101) == pytest.approx(0.1)
    assert normalized_return(10.0, frames=101) == pytest.approx(1.0)


def test_aggregate_weights_full_clips_including_failed_remainder():
    results = [
        ClipResult("short_failure", 101, 10, 0.1, True, 10),
        ClipResult("long_success", 201, 200, 1.0, False, -1),
    ]
    assert summarize(results)["step_weighted_norm_return"] == pytest.approx(0.7)


def test_single_frame_clip_rejected():
    with pytest.raises(ValueError):
        normalized_return(0.0, frames=1)


def test_benchmark_ignores_steps_after_reference_horizon(monkeypatch):
    from types import SimpleNamespace
    import torch
    import supermimic.eval.benchmark as benchmark

    class Env:
        def __init__(self):
            self.t = 0
            self.closed = False
            self.lib = SimpleNamespace(
                num_clips=1,
                clip_start=torch.tensor([0]),
                clip_len=torch.tensor([4]),
                clips=[SimpleNamespace(name="perfect", length=4)],
            )
            self.cmd = SimpleNamespace(library=self.lib, forced_frame=torch.tensor([-1]))
            self.command_manager = SimpleNamespace(get_term=lambda _: self.cmd)
            self.reward_manager = SimpleNamespace(active_terms=["tracking"], term_rewards=torch.tensor([[0.1]]))

        def reset(self):
            return {"policy": torch.zeros(1, 1), "goal": torch.zeros(1, 1)}, {}

        def step(self, action):
            self.t += 1
            obs, _ = self.reset()
            # Extra final physics step falls, after the three valid transitions.
            return obs, torch.tensor([0.1]), torch.tensor([self.t == 5]), torch.tensor([False]), {}

        def close(self):
            self.closed = True

    env = Env()
    monkeypatch.setattr(benchmark, "make_env", lambda *a, **kw: env)
    result = benchmark.run_benchmark(lambda *a: torch.zeros(1, 29), ["dummy"], device="cpu", verbose=False)[0]
    assert result.norm_return == pytest.approx(1.0)
    assert result.steps == 3
    assert not result.failed
    assert env.closed
