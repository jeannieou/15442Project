"""Stage-1 tests for speculation policy wiring.

These tests validate policy routing only:
- never: disables simulate path and skips speculation metrics.
- always: enables simulate path and computes speculation metrics.
- scheduler: currently shares the simulate path with always (before scheduler gating is implemented).
"""

from src.runner import HotPotQARun
from src.utils import Utils
from src.metrics import Metrics


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
    """Create a runner with all external side effects patched out."""
    monkeypatch.setattr(HotPotQARun, "_get_env", lambda self: _FakeEnv())
    runner = HotPotQARun(
        model_name="openai/gpt-4",
        guess_model_name="openai/gpt-5-nano",
        to_print_output=False,
        spec_policy=spec_policy,
    )

    # Avoid writing logs/files in tests.
    monkeypatch.setattr(runner, "log", lambda *args, **kwargs: None)
    monkeypatch.setattr(Utils, "delete_file", lambda path: None)
    monkeypatch.setattr(Utils, "delete_dir", lambda path, nested=False: None)
    monkeypatch.setattr(Utils, "read_json", lambda path: {"webthink_simple6": ""})

    saved = []

    def fake_save_json(obj, path, delete_prev_file=False):
        saved.append((path, obj))

    monkeypatch.setattr(Utils, "save_json", fake_save_json)
    return runner, saved


def test_never_policy_disables_simulation_and_skips_spec_metrics(monkeypatch):
    """Policy=never should call webthink(simulate=False) and never compute spec metrics."""
    runner, saved = _make_runner(monkeypatch, "never")
    sim_flags = []

    def fake_webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        sim_flags.append(simulate)
        return {"em": 1, "f1": 1}

    monkeypatch.setattr(HotPotQARun, "webthink", fake_webthink)
    monkeypatch.setattr(
        Metrics,
        "get_action_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called in never mode")),
    )

    runner.run(idxs_override=[0], seed=42)

    assert sim_flags == [False]
    metrics_payloads = [obj for path, obj in saved if path.endswith("metrics.json")]
    assert metrics_payloads
    assert metrics_payloads[-1]["policy"] == "never"


def test_always_policy_enables_simulation_and_computes_metrics(monkeypatch):
    """Policy=always should call webthink(simulate=True) and compute spec metrics."""
    runner, saved = _make_runner(monkeypatch, "always")
    sim_flags = []
    metric_calls = {"n": 0}

    def fake_webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        sim_flags.append(simulate)
        return {"em": 1, "f1": 1}

    def fake_metrics(*args, **kwargs):
        metric_calls["n"] += 1
        return {"general": 1.0}

    monkeypatch.setattr(HotPotQARun, "webthink", fake_webthink)
    monkeypatch.setattr(Metrics, "get_action_metrics", fake_metrics)

    runner.run(idxs_override=[0], seed=42)

    assert sim_flags == [True]
    assert metric_calls["n"] == 1
    metrics_payloads = [obj for path, obj in saved if path.endswith("metrics.json")]
    assert metrics_payloads[-1] == {"general": 1.0}


def test_scheduler_policy_currently_uses_simulate_path(monkeypatch):
    """Policy=scheduler currently shares always-path simulation until gating is added."""
    runner, _ = _make_runner(monkeypatch, "scheduler")
    sim_flags = []
    metric_calls = {"n": 0}

    def fake_webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        sim_flags.append(simulate)
        return {"em": 1, "f1": 1}

    def fake_metrics(*args, **kwargs):
        metric_calls["n"] += 1
        return {"general": 0.5}

    monkeypatch.setattr(HotPotQARun, "webthink", fake_webthink)
    monkeypatch.setattr(Metrics, "get_action_metrics", fake_metrics)

    runner.run(idxs_override=[0], seed=42)

    assert sim_flags == [True]
    assert metric_calls["n"] == 1


def test_safe_mean_filters_none_values():
    """safe_mean should ignore None values and return None for empty-valid input."""
    assert Utils.safe_mean([1.0, None, 3.0]) == 2.0
    assert Utils.safe_mean([None, None]) is None
    assert Utils.safe_mean([]) is None


def test_run_persists_step_records_with_none_fields(monkeypatch):
    """run() should persist step_records exactly, including None-valued fields."""
    runner, saved = _make_runner(monkeypatch, "never")

    def fake_webthink(self, idx=None, prompt=None, to_print=True, n=8, simulate=False):
        return {
            "em": 1,
            "f1": 1,
            "step_records": [
                {
                    "step_idx": 1,
                    "action_type": "search",
                    "eligible_for_speculation": True,
                    "speculated": False,
                    "skip_reason": "policy_disabled",
                    "hit": None,
                    "tool_latency": 0.25,
                    "speculator_latency": None,
                }
            ],
        }

    monkeypatch.setattr(HotPotQARun, "webthink", fake_webthink)
    monkeypatch.setattr(
        Metrics,
        "get_action_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("never mode should skip metrics")),
    )

    runner.run(idxs_override=[0], seed=42)

    records_payloads = [obj for path, obj in saved if path.endswith("step_records.json")]
    assert records_payloads, "expected step_records.json to be saved"
    record = records_payloads[-1][0]
    assert record["hit"] is None
    assert record["speculator_latency"] is None
