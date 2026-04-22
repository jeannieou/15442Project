import math
import os
from datetime import datetime

from src import analysis


def test_load_split_indices_supports_list_payload(monkeypatch):
    monkeypatch.setattr(analysis.Utils, "read_json", lambda path: [1, 2, 3])
    assert analysis.load_split_indices("samples.json", "validation") == [1, 2, 3]


def test_load_split_indices_supports_dict_payload(monkeypatch):
    payload = {"all": [1, 2, 3], "validation": [1, 2], "test": [3]}
    monkeypatch.setattr(analysis.Utils, "read_json", lambda path: payload)
    assert analysis.load_split_indices("samples.json", "validation") == [1, 2]


def test_summarize_metric_rows_uses_count_based_rates():
    rows = [
        {"eligible_steps": 4, "speculated_steps": 2, "hit_steps": 1, "general": 0.2},
        {"eligible_steps": 2, "speculated_steps": 1, "hit_steps": 1, "general": 0.8},
    ]
    summary = analysis.summarize_metric_rows(rows)
    assert summary["eligible_steps"] == 6
    assert summary["speculated_steps"] == 3
    assert summary["hit_steps"] == 2
    assert math.isclose(summary["coverage"], 3 / 6)
    assert math.isclose(summary["conditional_hit_rate"], 2 / 3)
    assert math.isclose(summary["overall_hit_rate"], 2 / 6)
    assert math.isclose(summary["general"], 0.5)


def test_pick_best_threshold_prefers_higher_hit_then_lower_cost():
    rows = [
        {"threshold": 0.05, "overall_hit_rate": 0.4, "speculated_steps": 8},
        {"threshold": 0.10, "overall_hit_rate": 0.4, "speculated_steps": 6},
        {"threshold": 0.15, "overall_hit_rate": 0.35, "speculated_steps": 5},
    ]
    best = analysis.pick_best_threshold(rows)
    assert best["threshold"] == 0.10


def test_summarize_across_seeds_outputs_mean_std():
    seed_rows = [
        {"overall_hit_rate": 0.4, "speculated_steps": 10},
        {"overall_hit_rate": 0.6, "speculated_steps": 14},
    ]
    agg = analysis.summarize_across_seeds(seed_rows)
    assert agg["n_seeds"] == 2
    assert math.isclose(agg["overall_hit_rate_mean"], 0.5)
    assert math.isclose(agg["speculated_steps_mean"], 12.0)
    assert agg["overall_hit_rate_std"] > 0


def test_resolve_idx_file_supports_seed_template():
    assert analysis.resolve_idx_file("samples_{seed}.json", 42) == "samples_42.json"
    assert analysis.resolve_idx_file("samples.json", 42) == "samples.json"


def test_make_output_subdir_includes_threshold_for_scheduler():
    out = analysis.make_output_subdir("scheduler", "validation", 42, threshold=0.15)
    assert "analysis" in out
    assert "seed_42" in out
    assert "validation" in out
    assert "scheduler" in out
    assert "thr_0p15" in out


def test_build_default_run_tag_format():
    tag = analysis.build_default_run_tag(
        ["gemini-2.5-flash"],
        ["gemini-2.5-flash-lite"],
        now=datetime(2026, 4, 21, 13, 7),
    )
    assert tag == "gemini-2.5-flash__gemini-2.5-flash-lite__0421_1307"


def test_resolve_artifact_paths_uses_run_root_and_tag():
    paths = analysis.resolve_artifact_paths(
        model_names=["a/model"],
        guess_model_names=["b/spec"],
        run_tag="demo_tag",
        runs_root="run_metrics/analysis/runs",
    )
    expected_root = os.path.normpath("run_metrics/analysis/runs/demo_tag")
    assert os.path.normpath(paths["cache_dir"]) == os.path.join(expected_root, "cache")
    assert os.path.normpath(paths["checkpoint_path"]) == os.path.join(expected_root, "checkpoint.json")
    assert os.path.normpath(paths["save_path"]) == os.path.join(expected_root, "summary.json")
    assert os.path.normpath(paths["log_path"]) == os.path.join(expected_root, "analysis.log")


def test_parse_rate_limit_wait_seconds_from_ms_message():
    err = RuntimeError("Rate limit reached. Please try again in 156ms.")
    wait_s = analysis.parse_rate_limit_wait_seconds(err)
    assert wait_s >= 0.1
    assert wait_s < 1.0


def test_run_policy_once_retries_rate_limit_and_resumes(monkeypatch):
    class FakeRateLimitError(Exception):
        pass

    calls = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.base_traj_path = "fake_path"

        def run(self, skip_done=False, idxs_override=None, seed=None):
            calls.append(skip_done)
            if len(calls) == 1:
                raise FakeRateLimitError("Rate limit reached. Please try again in 100ms.")

    monkeypatch.setattr(analysis, "HotPotQARun", FakeRunner)
    monkeypatch.setattr(analysis, "OpenAIRateLimitError", FakeRateLimitError)
    monkeypatch.setattr(analysis.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        analysis,
        "read_metrics_for_indices",
        lambda base_traj_path, idxs: [{"eligible_steps": 1, "speculated_steps": 1, "hit_steps": 1}],
    )

    out = analysis.run_policy_once(
        model_name="openai/gpt-4o",
        guess_model_name="openai/gpt-4.1-nano",
        spec_policy="scheduler",
        idxs=[1],
        seed=42,
        split_name="validation",
        threshold=0.1,
        to_print_output=False,
        max_rate_limit_retries=2,
    )

    assert calls == [True, True]
    assert out["overall_hit_rate"] == 1.0


def test_run_policy_once_uses_cache_when_available(monkeypatch, tmp_path):
    calls = {"run": 0}

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.base_traj_path = "fake_path"

        def run(self, skip_done=False, idxs_override=None, seed=None):
            calls["run"] += 1

    monkeypatch.setattr(analysis, "HotPotQARun", FakeRunner)
    monkeypatch.setattr(
        analysis,
        "read_metrics_for_indices",
        lambda base_traj_path, idxs: [{"eligible_steps": 2, "speculated_steps": 1, "hit_steps": 1}],
    )

    cache_dir = str(tmp_path / "cache")
    args = dict(
        model_name="openai/gpt-4o",
        guess_model_name="openai/gpt-4.1-nano",
        spec_policy="scheduler",
        idxs=[1],
        seed=42,
        split_name="validation",
        threshold=0.15,
        to_print_output=False,
        cache_dir=cache_dir,
        use_cache=True,
    )
    out1 = analysis.run_policy_once(**args)
    out2 = analysis.run_policy_once(**args)
    assert calls["run"] == 1
    assert out1 == out2


def test_run_policy_once_retries_incomplete_passes(monkeypatch):
    calls = {"run": 0}
    reads = {"n": 0}

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            self.base_traj_path = "fake_path"

        def run(self, skip_done=False, idxs_override=None, seed=None):
            calls["run"] += 1

    def fake_read(base_traj_path, idxs):
        reads["n"] += 1
        # First pass incomplete, second pass complete.
        if reads["n"] == 1:
            return []
        return [{"eligible_steps": 2, "speculated_steps": 1, "hit_steps": 1}]

    monkeypatch.setattr(analysis, "HotPotQARun", FakeRunner)
    monkeypatch.setattr(analysis, "read_metrics_for_indices", fake_read)
    monkeypatch.setattr(analysis, "missing_metric_indices", lambda base_traj_path, idxs: idxs)
    monkeypatch.setattr(analysis.time, "sleep", lambda s: None)

    out = analysis.run_policy_once(
        model_name="openai/gpt-4o",
        guess_model_name="openai/gpt-4.1-nano",
        spec_policy="scheduler",
        idxs=[1],
        seed=42,
        split_name="validation",
        threshold=0.1,
        to_print_output=False,
        max_incomplete_passes=2,
        use_cache=False,
    )
    assert calls["run"] == 2
    assert out["n_samples"] == 1
