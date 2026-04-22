"""Stage-4 tests: search-only speculation metrics and N/A (None) rules."""

from src.runner import HotPotQARun
from src.metrics import Metrics
from src.utils import Utils


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
        self.write_called = False

    def write(self):
        self.write_called = True


def _make_runner(monkeypatch, spec_policy):
    monkeypatch.setattr(HotPotQARun, "_get_env", lambda self: _FakeEnv())
    runner = HotPotQARun(
        model_name="openai/gpt-4",
        guess_model_name="openai/gpt-5-nano",
        to_print_output=False,
        spec_policy=spec_policy,
    )
    monkeypatch.setattr(runner, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(Utils, "delete_file", lambda path: None)
    monkeypatch.setattr(Utils, "delete_dir", lambda path, nested=False: None)
    monkeypatch.setattr(Utils, "read_json", lambda path: {"webthink_simple6": ""})

    saved = []

    def fake_save_json(obj, path, delete_prev_file=False):
        saved.append((path, obj))

    monkeypatch.setattr(Utils, "save_json", fake_save_json)
    return runner, saved


def test_speculation_metrics_computation_search_only():
    records = [
        # eligible + speculated + hit
        {"eligible_for_speculation": True, "speculated": True, "hit": True},
        # eligible + speculated + miss
        {"eligible_for_speculation": True, "speculated": True, "hit": False},
        # eligible + not speculated
        {"eligible_for_speculation": True, "speculated": False, "hit": None},
        # not eligible (lookup/finish), should be ignored from denominator
        {"eligible_for_speculation": False, "speculated": False, "hit": None},
    ]
    metric = Metrics.get_speculation_metrics_from_step_records(records)
    assert metric["eligible_steps"] == 3
    assert metric["speculated_steps"] == 2
    assert metric["hit_steps"] == 1
    assert metric["conditional_hit_rate"] == 0.5
    assert metric["coverage"] == 2 / 3
    assert metric["overall_hit_rate"] == 1 / 3


def test_speculation_metrics_return_none_on_zero_denominator():
    # No eligible search steps.
    records = [
        {"eligible_for_speculation": False, "speculated": False, "hit": None},
    ]
    metric = Metrics.get_speculation_metrics_from_step_records(records)
    assert metric["eligible_steps"] == 0
    assert metric["speculated_steps"] == 0
    assert metric["hit_steps"] == 0
    assert metric["conditional_hit_rate"] is None
    assert metric["coverage"] is None
    assert metric["overall_hit_rate"] is None


def test_runner_run_merges_stage4_metrics_for_simulate_policies(monkeypatch):
    runner, saved = _make_runner(monkeypatch, "always")

    def fake_webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        return {
            "em": 1,
            "f1": 1,
            "step_records": [
                {
                    "eligible_for_speculation": True,
                    "speculated": True,
                    "hit": True,
                },
                {
                    "eligible_for_speculation": True,
                    "speculated": False,
                    "hit": None,
                },
            ],
        }

    monkeypatch.setattr(HotPotQARun, "webthink", fake_webthink)
    monkeypatch.setattr(Metrics, "get_action_metrics", lambda *args, **kwargs: {"general": 1.0})

    runner.run(idxs_override=[0], seed=42)

    metrics_payloads = [obj for path, obj in saved if path.endswith("metrics.json")]
    assert metrics_payloads
    payload = metrics_payloads[-1]
    assert payload["general"] == 1.0
    assert payload["conditional_hit_rate"] == 1.0
    assert payload["coverage"] == 0.5
    assert payload["overall_hit_rate"] == 0.5
