"""Stage-2 tests: step_record structure and None semantics.

These tests avoid external API calls by stubbing:
- environment
- thought/action generation
- step execution timing/outputs
"""

from src.runner import HotPotQARun


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


def _build_runner(monkeypatch):
    monkeypatch.setattr(HotPotQARun, "_get_env", lambda self: _FakeEnv())
    return HotPotQARun(
        model_name="openai/gpt-4",
        guess_model_name="openai/gpt-5-nano",
        to_print_output=False,
        spec_policy="always",
    )


def test_webthink_records_none_fields_when_not_simulating(monkeypatch):
    """When simulate=False, each step_record should keep hit/speculator_latency as None."""
    runner = _build_runner(monkeypatch)

    def fake_generate(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        return "normal-thought", ["Search[Alan Turing]"]

    def fake_step(self, env, action, simulate=False):
        if simulate:
            return "sim-obs", 0, False, {}, 0.05
        return "obs", 0, True, {"em": 1, "f1": 1}, 0.20

    monkeypatch.setattr(HotPotQARun, "generate_thought_actions", fake_generate)
    monkeypatch.setattr(HotPotQARun, "step", fake_step)

    info = runner.webthink(idx=0, prompt="", to_print=False, n=2, simulate=False)

    assert "step_records" in info
    assert len(info["step_records"]) == 1
    record = info["step_records"][0]
    assert record["action_type"] == "search"
    assert record["eligible_for_speculation"] is True
    assert record["speculated"] is False
    assert record["skip_reason"] == "policy_disabled"
    assert record["hit"] is None
    assert record["speculator_latency"] is None
    assert record["tool_latency"] == 0.20


def test_webthink_records_hit_and_latency_when_simulating(monkeypatch):
    """When simulate=True, step_record should include hit=True and measured speculator latency."""
    runner = _build_runner(monkeypatch)

    def fake_generate(self, i, running_prompt, n_calls_badcalls, num_actions=1, max_retries=1):
        if num_actions == 1:
            return "normal-thought", ["Search[Alan Turing]"]
        return "sim-thought", ["Search[Alan Turing]", "Lookup[foo]"]

    def fake_step(self, env, action, simulate=False):
        if simulate:
            return "sim-obs", 0, False, {}, 0.05
        return "obs", 0, True, {"em": 1, "f1": 1}, 0.20

    monkeypatch.setattr(HotPotQARun, "generate_thought_actions", fake_generate)
    monkeypatch.setattr(HotPotQARun, "step", fake_step)

    info = runner.webthink(idx=0, prompt="", to_print=False, n=2, simulate=True)

    assert "step_records" in info
    assert len(info["step_records"]) == 1
    record = info["step_records"][0]
    assert record["speculated"] is True
    assert record["skip_reason"] is None
    assert record["hit"] is True
    assert record["speculator_latency"] == 0.05
    assert record["tool_latency"] == 0.20


def test_get_action_type_handles_supported_actions(monkeypatch):
    """get_action_type should parse action prefixes reliably."""
    runner = _build_runner(monkeypatch)
    assert runner.get_action_type("Search[abc]") == "search"
    assert runner.get_action_type("Lookup[x]") == "lookup"
    assert runner.get_action_type("Finish[y]") == "finish"
    assert runner.get_action_type("NoBracketAction") is None
