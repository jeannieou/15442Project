import argparse
import math
import os
import re
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from itertools import product
from os.path import join
from statistics import mean, stdev

from . import constants
from .runner import HotPotQARun
from .utils import Utils

try:
    from openai import RateLimitError as OpenAIRateLimitError
except Exception:  # pragma: no cover - defensive import
    OpenAIRateLimitError = None


DEFAULT_THRESHOLDS = [
    0.05, 0.15,        #  loose range
    0.25,        #  stricter range
]


class TeeWriter:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def parse_csv_floats(text):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_csv_strings(text):
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_csv_ints(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def sanitize_tag_component(value):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value))


def threshold_tag(threshold):
    return str(threshold).replace(".", "p")


def make_output_subdir(spec_policy, split_name, seed, threshold=None):
    parts = ["analysis", f"seed_{seed}", sanitize_tag_component(split_name), sanitize_tag_component(spec_policy)]
    if spec_policy == "scheduler" and threshold is not None:
        parts.append(f"thr_{threshold_tag(threshold)}")
    return join(*parts)


def make_cache_path(cache_dir, model_name, guess_model_name, spec_policy, split_name, seed, threshold=None):
    file_name = sanitize_tag_component(spec_policy)
    if spec_policy == "scheduler" and threshold is not None:
        file_name += f"_thr_{threshold_tag(threshold)}"
    file_name += ".json"
    return join(
        cache_dir,
        sanitize_tag_component(model_name),
        sanitize_tag_component(guess_model_name),
        f"seed_{seed}",
        sanitize_tag_component(split_name),
        file_name,
    )


def build_default_run_tag(model_names, guess_model_names, now=None):
    now = now or datetime.now()
    ts = now.strftime("%m%d_%H%M")
    actor = sanitize_tag_component(model_names[0]) if model_names else "actor"
    spec = sanitize_tag_component(guess_model_names[0]) if guess_model_names else "spec"
    if len(model_names) > 1:
        actor += f"_plus{len(model_names)-1}"
    if len(guess_model_names) > 1:
        spec += f"_plus{len(guess_model_names)-1}"
    return f"{actor}__{spec}__{ts}"


def resolve_artifact_paths(
    model_names,
    guess_model_names,
    run_tag=None,
    runs_root=join(".", "run_metrics", "analysis", "runs"),
    cache_dir=None,
    checkpoint_path=None,
    save_path=None,
    log_path=None,
):
    resolved_tag = run_tag or build_default_run_tag(model_names, guess_model_names)
    run_dir = join(runs_root, resolved_tag)
    return {
        "run_tag": resolved_tag,
        "run_dir": run_dir,
        "cache_dir": cache_dir or join(run_dir, "cache"),
        "checkpoint_path": checkpoint_path or join(run_dir, "checkpoint.json"),
        "save_path": save_path or join(run_dir, "summary.json"),
        "log_path": log_path or join(run_dir, "analysis.log"),
        "meta_path": join(run_dir, "run_meta.json"),
    }


def resolve_idx_file(idx_file, seed):
    if "{seed}" in idx_file:
        return idx_file.format(seed=seed)
    return idx_file


def load_split_indices(idx_file, split):
    payload = Utils.read_json(idx_file)
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        raise ValueError("idx-file must contain a list or dict payload")
    if split not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise ValueError(f"Split '{split}' not found in idx-file. Available: {available}")
    return payload[split]


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def summarize_metric_rows(rows):
    if not rows:
        return {
            "n_samples": 0,
            "conditional_hit_rate": None,
            "coverage": None,
            "overall_hit_rate": None,
            "eligible_steps": 0,
            "speculated_steps": 0,
            "hit_steps": 0,
            "general": None,
            "Search": None,
            "em": None,
            "f1": None,
        }

    eligible_steps = sum(r.get("eligible_steps", 0) or 0 for r in rows)
    speculated_steps = sum(r.get("speculated_steps", 0) or 0 for r in rows)
    hit_steps = sum(r.get("hit_steps", 0) or 0 for r in rows)

    def _mean_key(key):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        return Utils.safe_mean(vals)

    summary = {
        "n_samples": len(rows),
        "conditional_hit_rate": safe_ratio(hit_steps, speculated_steps),
        "coverage": safe_ratio(speculated_steps, eligible_steps),
        "overall_hit_rate": safe_ratio(hit_steps, eligible_steps),
        "eligible_steps": eligible_steps,
        "speculated_steps": speculated_steps,
        "hit_steps": hit_steps,
        "general": _mean_key("general"),
        "Search": _mean_key("Search"),
        "em": _mean_key("em"),
        "f1": _mean_key("f1"),
    }
    return summary


def summarize_across_seeds(seed_summaries):
    output = {"n_seeds": len(seed_summaries)}
    if not seed_summaries:
        return output

    keys = set()
    for row in seed_summaries:
        keys.update(row.keys())

    for key in sorted(keys):
        vals = [r.get(key) for r in seed_summaries if isinstance(r.get(key), (int, float))]
        if not vals:
            continue
        output[f"{key}_mean"] = mean(vals)
        output[f"{key}_std"] = stdev(vals) if len(vals) > 1 else 0.0
    return output


def read_metrics_for_indices(base_traj_path, idxs):
    rows = []
    for idx in idxs:
        metric_path = join(base_traj_path, str(idx), "metrics.json")
        if not os.path.exists(metric_path):
            continue
        rows.append(Utils.read_json(metric_path))
    return rows


def missing_metric_indices(base_traj_path, idxs):
    missing = []
    for idx in idxs:
        metric_path = join(base_traj_path, str(idx), "metrics.json")
        if not os.path.exists(metric_path):
            missing.append(idx)
    return missing


def parse_rate_limit_wait_seconds(exc, default_seconds=2.0):
    msg = str(exc)

    ms_match = re.search(r"try again in\s*([0-9]+)\s*ms", msg, flags=re.IGNORECASE)
    if ms_match:
        return max(float(ms_match.group(1)) / 1000.0, 0.1)

    s_match = re.search(r"try again in\s*([0-9]*\.?[0-9]+)\s*s", msg, flags=re.IGNORECASE)
    if s_match:
        return max(float(s_match.group(1)), 0.1)

    return default_seconds


def run_policy_once(
    model_name,
    guess_model_name,
    spec_policy,
    idxs,
    seed,
    split_name,
    threshold=None,
    to_print_output=False,
    max_rate_limit_retries=30,
    max_incomplete_passes=5,
    cache_dir=None,
    use_cache=True,
):
    cache_path = None
    if cache_dir is not None:
        cache_path = make_cache_path(
            cache_dir=cache_dir,
            model_name=model_name,
            guess_model_name=guess_model_name,
            spec_policy=spec_policy,
            split_name=split_name,
            seed=seed,
            threshold=threshold,
        )
        if use_cache and os.path.exists(cache_path):
            return Utils.read_json(cache_path)

    runner = HotPotQARun(
        model_name=model_name,
        guess_model_name=guess_model_name,
        to_print_output=to_print_output,
        spec_policy=spec_policy,
        scheduler_threshold=threshold,
        output_subdir=make_output_subdir(
            spec_policy=spec_policy, split_name=split_name, seed=seed, threshold=threshold
        ),
    )
    retries = 0
    # Resume from already-finished samples when rerunning analysis after interruptions.
    skip_done = True
    incomplete_pass = 0
    rows = []
    while True:
        while True:
            try:
                runner.run(skip_done=skip_done, idxs_override=idxs, seed=seed)
                break
            except Exception as exc:
                if OpenAIRateLimitError is not None and isinstance(exc, OpenAIRateLimitError):
                    retries += 1
                    if retries > max_rate_limit_retries:
                        raise
                    wait_s = parse_rate_limit_wait_seconds(exc)
                    print(
                        f"[analysis] rate limited for policy={spec_policy}, seed={seed}; "
                        f"retry {retries}/{max_rate_limit_retries} in {wait_s:.2f}s"
                    )
                    time.sleep(wait_s)
                    continue
                raise

        rows = read_metrics_for_indices(runner.base_traj_path, idxs)
        if len(rows) >= len(idxs):
            break

        incomplete_pass += 1
        missing = missing_metric_indices(runner.base_traj_path, idxs)
        if incomplete_pass > max_incomplete_passes:
            raise RuntimeError(
                f"Incomplete run for policy={spec_policy}, seed={seed}. "
                f"Missing {len(missing)}/{len(idxs)} metrics after {max_incomplete_passes} retries. "
                f"Missing idx sample (first 10): {missing[:10]}"
            )
        print(
            f"[analysis] incomplete pass for policy={spec_policy}, seed={seed}: "
            f"{len(missing)}/{len(idxs)} missing metrics; retry pass "
            f"{incomplete_pass}/{max_incomplete_passes}"
        )
        time.sleep(2.0)

    summary = summarize_metric_rows(rows)
    if cache_path is not None and use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        Utils.save_json(summary, cache_path)
    return summary


def pick_best_threshold(threshold_results, primary_key="overall_hit_rate", cost_key="speculated_steps"):
    """Pick threshold by:
    1) higher primary metric
    2) lower cost metric
    3) smaller threshold
    """
    best = None
    for row in threshold_results:
        score = row.get(primary_key)
        cost = row.get(cost_key)
        if score is None:
            score = -math.inf
        if cost is None:
            cost = math.inf
        candidate = (score, -cost, -row["threshold"])
        if best is None or candidate > best["rank"]:
            best = {"rank": candidate, **row}
    return best


def evaluate_model_pair(
    model_name,
    guess_model_name,
    seeds,
    idx_file_pattern,
    thresholds,
    validation_split="validation",
    test_split="test",
    to_print_output=False,
    cache_dir=None,
    use_cache=True,
):
    threshold_results = []
    for threshold in thresholds:
        per_seed = []
        for seed in seeds:
            idx_file = resolve_idx_file(idx_file_pattern, seed)
            idxs = load_split_indices(idx_file, validation_split)
            summary = run_policy_once(
                model_name=model_name,
                guess_model_name=guess_model_name,
                spec_policy="scheduler",
                idxs=idxs,
                seed=seed,
                split_name=validation_split,
                threshold=threshold,
                to_print_output=to_print_output,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            per_seed.append(summary)
        agg = summarize_across_seeds(per_seed)
        threshold_results.append(
            {
                "threshold": threshold,
                "validation_per_seed": per_seed,
                "validation_aggregate": agg,
                "overall_hit_rate": agg.get("overall_hit_rate_mean"),
                "speculated_steps": agg.get("speculated_steps_mean"),
            }
        )

    best = pick_best_threshold(threshold_results)
    best_threshold = best["threshold"]

    test_policy_results = {}
    for policy in ["never", "always", "scheduler"]:
        per_seed = []
        for seed in seeds:
            idx_file = resolve_idx_file(idx_file_pattern, seed)
            idxs = load_split_indices(idx_file, test_split)
            summary = run_policy_once(
                model_name=model_name,
                guess_model_name=guess_model_name,
                spec_policy=policy,
                idxs=idxs,
                seed=seed,
                split_name=test_split,
                threshold=best_threshold if policy == "scheduler" else None,
                to_print_output=to_print_output,
                cache_dir=cache_dir,
                use_cache=use_cache,
            )
            per_seed.append(summary)
        test_policy_results[policy] = {
            "per_seed": per_seed,
            "aggregate": summarize_across_seeds(per_seed),
        }

    return {
        "model_name": model_name,
        "guess_model_name": guess_model_name,
        "seeds": seeds,
        "best_threshold": best_threshold,
        "threshold_scan": threshold_results,
        "test_results": test_policy_results,
    }


def run_stage6_analysis(
    model_names,
    guess_model_names,
    seeds,
    idx_file_pattern,
    thresholds,
    validation_split="validation",
    test_split="test",
    to_print_output=False,
    cache_dir=None,
    use_cache=True,
    checkpoint_path=None,
):
    report = {"pairs": []}
    for model_name, guess_model_name in product(model_names, guess_model_names):
        result = evaluate_model_pair(
            model_name=model_name,
            guess_model_name=guess_model_name,
            seeds=seeds,
            idx_file_pattern=idx_file_pattern,
            thresholds=thresholds,
            validation_split=validation_split,
            test_split=test_split,
            to_print_output=to_print_output,
            cache_dir=cache_dir,
            use_cache=use_cache,
        )
        report["pairs"].append(result)
        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            Utils.save_json(report, checkpoint_path)
    return report


def main():
    parser = argparse.ArgumentParser(description="Stage-6 analysis: threshold sweep + multi-seed summary")
    parser.add_argument("--idx-file", required=True, help="Sample config file path (supports {seed} template)")
    parser.add_argument(
        "--model-names",
        default=constants.openrouter_model_name,
        help="Comma-separated actor model names",
    )
    parser.add_argument(
        "--guess-model-names",
        default=constants.openrouter_guess_model_name,
        help="Comma-separated speculator model names",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(x) for x in DEFAULT_THRESHOLDS),
        help="Comma-separated threshold values",
    )
    parser.add_argument(
        "--seeds",
        default=str(constants.random_seed),
        help="Comma-separated seeds",
    )
    parser.add_argument("--validation-split", default="validation", choices=["all", "validation", "test"])
    parser.add_argument("--test-split", default="test", choices=["all", "validation", "test"])
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Per-task summary cache directory for resume/reuse",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing task cache",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Incremental checkpoint report written after each model pair",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Where to save final analysis report JSON",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Reusable run tag. If omitted, auto-generated as actor__spec__MMDD_HHMM",
    )
    parser.add_argument(
        "--runs-root",
        default=join(".", "run_metrics", "analysis", "runs"),
        help="Root directory for auto-generated run artifacts",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        help="Analysis process log path (defaults to <runs-root>/<run-tag>/analysis.log)",
    )
    parser.add_argument("--print-output", action="store_true", help="Print detailed per-step runner logs")
    args = parser.parse_args()

    model_names = parse_csv_strings(args.model_names)
    guess_model_names = parse_csv_strings(args.guess_model_names)
    thresholds = parse_csv_floats(args.thresholds)
    seeds = parse_csv_ints(args.seeds)

    artifacts = resolve_artifact_paths(
        model_names=model_names,
        guess_model_names=guess_model_names,
        run_tag=args.run_tag,
        runs_root=args.runs_root,
        cache_dir=args.cache_dir,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        log_path=args.log_path,
    )
    for path_key in ["cache_dir", "checkpoint_path", "save_path", "log_path", "meta_path"]:
        path = artifacts[path_key]
        dir_path = path if path_key == "cache_dir" else os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    run_meta = {
        "run_tag": artifacts["run_tag"],
        "created_at": datetime.now().isoformat(),
        "model_names": model_names,
        "guess_model_names": guess_model_names,
        "seeds": seeds,
        "thresholds": thresholds,
        "idx_file_pattern": args.idx_file,
        "validation_split": args.validation_split,
        "test_split": args.test_split,
        "paths": artifacts,
    }
    Utils.save_json(run_meta, artifacts["meta_path"])

    print(f"[analysis] run_tag={artifacts['run_tag']}")
    print(f"[analysis] log={artifacts['log_path']}")
    print(f"[analysis] cache={artifacts['cache_dir']}")
    print(f"[analysis] checkpoint={artifacts['checkpoint_path']}")
    print(f"[analysis] summary={artifacts['save_path']}")

    with open(artifacts["log_path"], "a", encoding="utf-8") as log_f:
        tee_out = TeeWriter(sys.stdout, log_f)
        tee_err = TeeWriter(sys.stderr, log_f)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            report = run_stage6_analysis(
                model_names=model_names,
                guess_model_names=guess_model_names,
                seeds=seeds,
                idx_file_pattern=args.idx_file,
                thresholds=thresholds,
                validation_split=args.validation_split,
                test_split=args.test_split,
                to_print_output=args.print_output,
                cache_dir=artifacts["cache_dir"],
                use_cache=not args.no_cache,
                checkpoint_path=artifacts["checkpoint_path"],
            )
            Utils.save_json(report, artifacts["save_path"])
            print(f"Saved stage-6 analysis report to: {artifacts['save_path']}")


if __name__ == "__main__":
    main()
