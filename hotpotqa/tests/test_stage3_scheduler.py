"""Stage-3 tests: scheduler logic + runner integration with per-step gating."""

from types import SimpleNamespace

from src.runner import HotPotQARun
from src.scheduler import CostAwareScheduler


class _FakeEnv:
    def __init__(self):
        self.normal_trajectory_dict = {
            "prompt": "",
            "observations": [],
            "thoughts": [],
            "actions": [],
            "time_taken": [],
        }
        self.sim_trajectory_dict = {
            "prompt": "",
            "observations": [],
            "thoughts": [],
            "actions": [],
            "time_taken": [],
        }

    def reset(self, idx=None):
        return f"Question: fake-{idx}"

    def update_traj_dict_records(self, thought, action, observation, time, sim=False):
        target = self.sim_trajectory_dict if sim else self.normal_trajectory_dict
        target["thoughts"].append(thought)
        target["actions"].append(action)
        target["observations"].append(observation)
        target["time_taken"].append(time)


def test_scheduler_rejects_non_eligible_action():
    scheduler = CostAwareScheduler(threshold=0.05)
    should_speculate, reason, benefit = scheduler.should_speculate("lookup")
    assert should_speculate is False
    assert reason == "not_eligible"
    assert benefit is None


def test_scheduler_threshold_decision_changes_with_threshold():
    # With defaults: p_hat=0.5 and util=1/(1+0.1)=0.909..., benefit~0.303
    scheduler_easy = CostAwareScheduler(threshold=0.2)
    scheduler_hard = CostAwareScheduler(threshold=0.4)

    assert scheduler_easy.should_speculate("search")[0] is True
    assert scheduler_hard.should_speculate("search")[0] is False


def test_scheduler_records_search_only_and_updates_histories():
    scheduler = CostAwareScheduler(threshold=0.05, window_size=3)
    scheduler.record_step("lookup", tool_latency=0.01, speculated=True, hit=True, speculator_latency=0.02)
    assert len(scheduler.api_latency_history["lookup"]) == 0

    scheduler.record_step("search", tool_latency=1.0, speculated=True, hit=True, speculator_latency=0.1)
    scheduler.record_step("search", tool_latency=1.2, speculated=True, hit=False, speculator_latency=0.2)

    assert list(scheduler.hit_history["search"]) == [1, 0]
    assert list(scheduler.api_latency_history["search"]) == [1.0, 1.2]
    assert list(scheduler.spec_latency_history["search"]) == [0.1, 0.2]


def test_runner_scheduler_policy_gates_per_step(monkeypatch):
    """Runner should respect scheduler decisions per step and persist step-level decisions."""
    monkeypatch.setattr(HotPotQARun, "_get_env", lambda self: _FakeEnv())
    runner = HotPotQARun(
        model_name="openai/gpt-4",
        guess_model_name="openai/gpt-5-nano",
        to_print_output=False,
        spec_policy="scheduler",
        scheduler_threshold=0.05,
    )
    monkeypatch.setattr(runner, "log", lambda *args, **kwargs: None)

    class _FakeScheduler:
        def __init__(self):
            self.calls = 0
            self.record_calls = []

        def should_speculate(self, action_type):
            self.calls += 1
            if self.calls == 1:
                return False, "low_benefit", 0.01
            return True, None, 0.42

        def record_step(self, action_type, tool_latency, speculated, hit=None, speculator_latency=None):
            self.record_calls.append((action_type, tool_latency, speculated, hit, speculator_latency))

    fake_scheduler = _FakeScheduler()
    runner.scheduler = fake_scheduler

    def fake_generate(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        return "normal-thought", [f"Search[item-{i}]"]

    class _FakeSpeculator:
        def __init__(self):
            self.feedback = []

        def reset_episode(self):
            return None

        def predict_actions(self, step_index, running_prompt, num_actions, max_retries):
            return SimpleNamespace(
                thought="sim-thought",
                actions=[f"Search[item-{step_index}]", "Lookup[foo]"],
                n_calls=1,
                n_badcalls=0,
            )

        def predict_observation(self, action, max_retries):
            return SimpleNamespace(observation="sim-obs", latency_s=0.05)

        def record_feedback(self, action, real_observation, predicted_observation):
            self.feedback.append((action, real_observation, predicted_observation))

    call_state = {"normal_calls": 0}

    def fake_step(self, env, action):
        call_state["normal_calls"] += 1
        done = call_state["normal_calls"] >= 2
        return "obs", 0, done, {"em": 1, "f1": 1}, 0.20

    monkeypatch.setattr(HotPotQARun, "generate_thought_actions", fake_generate)
    monkeypatch.setattr(HotPotQARun, "step", fake_step)
    runner.speculator = _FakeSpeculator()

    info = runner.webthink(idx=0, prompt="", to_print=False, n=3, simulate=True)

    assert len(info["step_records"]) == 2

    first = info["step_records"][0]
    assert first["speculated"] is False
    assert first["skip_reason"] == "low_benefit"
    assert first["hit"] is None
    assert first["speculator_latency"] is None
    assert first["expected_benefit"] == 0.01

    second = info["step_records"][1]
    assert second["speculated"] is True
    assert second["skip_reason"] is None
    assert second["hit"] is True
    assert second["speculator_latency"] == 0.05
    assert second["expected_benefit"] == 0.42

    assert len(fake_scheduler.record_calls) == 2
    assert fake_scheduler.record_calls[0][2] is False
    assert fake_scheduler.record_calls[1][2] is True
