"""Stage-5 tests for fixed sample splits from --idx-file."""

import pytest

import run as run_module


def test_resolve_idxs_override_supports_list_payload(monkeypatch):
    monkeypatch.setattr(run_module.Utils, "read_json", lambda path: [3, 7, 9])
    assert run_module.resolve_idxs_override("samples.json", split="validation") == [3, 7, 9]


def test_resolve_idxs_override_supports_named_splits(monkeypatch):
    payload = {"all": [1, 2, 3], "validation": [1, 2], "test": [3]}
    monkeypatch.setattr(run_module.Utils, "read_json", lambda path: payload)

    assert run_module.resolve_idxs_override("samples.json", split="all") == [1, 2, 3]
    assert run_module.resolve_idxs_override("samples.json", split="validation") == [1, 2]
    assert run_module.resolve_idxs_override("samples.json", split="test") == [3]


def test_resolve_idxs_override_missing_split_raises(monkeypatch):
    monkeypatch.setattr(run_module.Utils, "read_json", lambda path: {"all": [1, 2]})
    with pytest.raises(ValueError, match="Requested split 'validation' not found"):
        run_module.resolve_idxs_override("samples.json", split="validation")


def test_main_passes_selected_split_to_runner(monkeypatch):
    calls = {"run": []}

    class _FakeRunner:
        def __init__(self, *args, **kwargs):
            self.base_traj_path = "./dummy_trajs"

        def run(self, skip_done=False, idxs_override=None, seed=None):
            calls["run"].append(
                {"skip_done": skip_done, "idxs_override": idxs_override, "seed": seed}
            )

    monkeypatch.setattr(run_module, "HotPotQARun", _FakeRunner)
    monkeypatch.setattr(run_module.Utils, "cleanup_trajs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_module.Utils,
        "read_json",
        lambda path: {"all": [10, 11], "validation": [10], "test": [11]},
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--idx-file",
            "samples.json",
            "--split",
            "validation",
            "--seed",
            "42",
        ],
    )
    run_module.main()

    assert calls["run"] == [
        {"skip_done": True, "idxs_override": [10], "seed": 42},
    ]
