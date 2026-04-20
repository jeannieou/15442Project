# Cost-Aware Speculation Scheduler

This README documents how to run and analyze the cost-aware scheduler pipeline in `hotpotqa/`.

## 1) Environment Variables

Set these before running.

### Required for recommended setup (DeepSeek actor + Gemini speculator)

```bash
export OPENAI_API_KEY="<YOUR_DEEPSEEK_API_KEY>"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export GEMINI_API_KEY="<YOUR_GEMINI_API_KEY>"
```

Notes:
- In this setup, `OPENAI_API_KEY` is your **DeepSeek** key (via OpenAI-compatible endpoint).
- `OPENAI_BASE_URL` points to DeepSeek API base.
- `GEMINI_API_KEY` is used for direct Gemini calls when model name starts with `gemini...`.

### Optional / alternative

```bash
export OPENROUTER_API_KEY="<YOUR_OPENROUTER_API_KEY>"
```

Only needed when using OpenRouter-routed model names (for example `google/gemini-...`).

## 2) Recommended Model Choice

- Main agent (Actor): `deepseek-chat` (or your DeepSeek 3.2 model ID if different)
  - Rationale: strong capability, usually cheaper than frontier OpenAI models.
- Speculator: `gemini-2.5-flash-lite`
  - Rationale: fast and often cheapest/free-tier friendly for speculative branches.

Important routing rule in this codebase:
- `gemini-...` -> direct Gemini SDK (`GEMINI_API_KEY`)
- `google/gemini-...` -> OpenRouter route (`OPENROUTER_API_KEY`)

## 3) Quick Start

### 3.1 Generate a 20-question split file

```bash
python -c "import json,random; from src import constants as c; seed=42; idxs=list(range(c.num)); random.Random(seed).shuffle(idxs); idxs=idxs[:20]; sp=int(0.8*len(idxs)); payload={'all':idxs,'validation':idxs[:sp],'test':idxs[sp:],'seed':seed}; open('samples20.json','w',encoding='utf-8').write(json.dumps(payload,indent=2)); print(payload['validation'][:3], '...')"
```

### 3.2 Run one policy manually (`run.py`)

```bash
python run.py \
  --modelname deepseek-chat \
  --guessmodelname gemini-2.5-flash-lite \
  --spec-policy scheduler \
  --threshold 0.05 \
  --idx-file samples20.json \
  --split validation \
  --seed 42
```

### 3.3 Run stage-6 analysis (`src.analysis`)

```bash
python -m src.analysis \
  --idx-file samples20.json \
  --seeds 42 \
  --thresholds 0.05,0.15 \
  --model-names deepseek-chat \
  --guess-model-names gemini-2.5-flash-lite \
  --print-output
```

## 4) Command-Line Arguments

## 4.1 `run.py` arguments

Run entry:
```bash
python run.py [options]
```

Flags:
- `--getmetric`: compute and print average metrics from saved trajectories.
- `--getmetric2`: compute and print cumulative metrics.
- `--savemetrics`: save computed metrics to disk.
- `--graph`: plot agent-time comparison graphs.
- `--graph2`: plot top-1 vs top-3 comparison graphs.
- `--graph3`: plot detailed metric comparison graphs.
- `--noprint`: suppress console output during run.
- `--norun`: skip running experiment loop (kept for legacy behavior; default flow runs loop unless overridden in code path).
- `--modelname <str>`: actor model name. Default from `src/constants.py`.
- `--guessmodelname <str>`: speculator model name. Default from `src/constants.py`.
- `--spec-policy {never,always,scheduler}`: speculation policy selector.
- `--threshold <float>`: scheduler threshold.
- `--output-subdir <str>`: optional subdirectory under trajectory path to avoid overwrite.
- `--seed <int>`: shuffle seed.
- `--idx-file <path>`: sample indices JSON file.
  - Supports list payload: `[1,2,3]`
  - Supports dict payload: `{"all":[...],"validation":[...],"test":[...]}`
- `--split {all,validation,test}`: pick which split from `--idx-file` dict payload.
- `--idx <int>`: run single question index.
- `--cleanuptrajs`: cleanup incomplete trajectory directories.

## 4.2 `python -m src.analysis` arguments

Run entry:
```bash
python -m src.analysis [options]
```

Core:
- `--idx-file <path>` (required): sample config file. Supports `{seed}` template in path.
- `--model-names <csv>`: actor models, comma-separated.
- `--guess-model-names <csv>`: speculator models, comma-separated.
- `--thresholds <csv-floats>`: threshold sweep values.
- `--seeds <csv-ints>`: seeds to evaluate.
- `--validation-split {all,validation,test}`: split used for threshold selection.
- `--test-split {all,validation,test}`: split used for final reporting.

Caching / resume:
- `--cache-dir <path>`: task-level summary cache root. If omitted, auto-generated.
- `--no-cache`: disable cache read/write.
- `--checkpoint-path <path>`: incremental checkpoint JSON path. If omitted, auto-generated.
- `--save-path <path>`: final summary JSON path. If omitted, auto-generated.
- `--run-tag <str>`: stable run ID for resuming exactly the same run.
- `--runs-root <path>`: root directory used when auto-generating artifact paths.
- `--log-path <path>`: analysis process log path. If omitted, auto-generated.
- `--print-output`: print detailed runner logs (also appended to log file).

## 5) Auto Artifacts and Naming

If you do not pass explicit cache/checkpoint/save/log paths, analysis auto-generates:

- `run_tag = <actor_model>__<spec_model>__MMDD_HHMM`
- under `run_metrics/analysis/runs/<run_tag>/`:
  - `cache/`
  - `checkpoint.json`
  - `summary.json`
  - `analysis.log`
  - `run_meta.json`

This keeps CLI short and reproducible.

To resume an interrupted run reliably, rerun with the same `--run-tag`.

## 6) How Resume/Skip Works

There are two layers:

1. Task cache reuse:
   - If cached summary exists for `(model, guess_model, seed, split, policy, threshold)`, analysis skips re-running that task.

2. Runner-level skip:
   - `runner.run(skip_done=True)` checks whether each sample already has `log.txt` under the target trajectory directory.
   - Existing samples are skipped (`[SKIP] Index ... already done`).

Additionally, analysis checks missing `metrics.json` and retries incomplete passes before accepting a summary.

## 7) Output Files You’ll Read Most

- Final report:
  - `run_metrics/analysis/runs/<run_tag>/summary.json`
- Incremental checkpoint:
  - `run_metrics/analysis/runs/<run_tag>/checkpoint.json`
- Full run log:
  - `run_metrics/analysis/runs/<run_tag>/analysis.log`
- Per-question trajectories:
  - `run_metrics/agent_<actor>_top<k>/trajs_<spec>/analysis/seed_<seed>/<split>/<policy>[/thr_xxx]/<idx>/`
  - Contains `log.txt`, `metrics.json`, `step_records.json`, `normalobs.json`, `simobs.json`.

## 8) What Was Changed for Cost Scheduler

Implemented and tested:

1. Policy wiring:
   - `never / always / scheduler` in `run.py` and `src/runner.py`.

2. Step record schema:
   - Unified `step_records` with explicit `None` semantics for skipped speculation.

3. Scheduler core:
   - `src/scheduler.py` with smoothed hit rate and threshold gate.

4. Metrics rewrite:
   - Search-only `conditional_hit_rate / coverage / overall_hit_rate` and zero-denominator `N/A` behavior.

5. Fixed split execution:
   - `--idx-file + --split` support in `run.py`.

6. Stage-6 analysis:
   - Threshold sweep + multi-seed aggregation.
   - Auto-retry for rate limits.
   - Incomplete-pass retry checks.
   - Auto artifact path generation and run metadata.
   - Cache/checkpoint/log orchestration.

7. Path robustness and runtime stability:
   - Absolute path handling for data/prompts.
   - Windows console Unicode-safe logging fallback.

## 9) Minimal Recommended Commands

### One clean analysis run (short CLI, auto paths)

```bash
python -m src.analysis \
  --idx-file samples20.json \
  --seeds 42 \
  --thresholds 0.05,0.15 \
  --model-names deepseek-chat \
  --guess-model-names gemini-2.5-flash-lite \
  --print-output
```

### Resume same run explicitly

```bash
python -m src.analysis \
  --idx-file samples20.json \
  --seeds 42 \
  --thresholds 0.05,0.15 \
  --model-names deepseek-chat \
  --guess-model-names gemini-2.5-flash-lite \
  --run-tag "<the_previous_run_tag>" \
  --print-output
```

