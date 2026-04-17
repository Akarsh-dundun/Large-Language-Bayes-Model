# `trial.py` usage guide

This is the user-facing driver on the `anant` branch. It runs the full
Large Language Bayes stacking pipeline — generate (or load) NumPyro
programs, run posterior inference, compute uniform / BMA / LOO-stacking
weights, and save every metric — for one or many (task, LLM) pairs.

## TL;DR

```bash
# Live-LLM run against a local Ollama
python trial.py --task tasks/coin_flip.json \
                --llm-config llm_configs/qwen25_coder.json \
                --n-models 10

# Load pre-generated programs from scratch (no LLM calls)
python trial.py --task tasks/tornado_counts_plains.json \
                --llm-config llm_configs/qwen25_coder.json \
                --n-models 100 \
                --preload-codes-dir /scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results/tornado_counts_plains/qwen25_coder/codes

# Sweep every cell with codes on scratch
LLB_PAPER_RESULTS_ROOT=/scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results \
python trial.py --sweep-paper --n-models 100
```

Output lands in `./experiment_results_anant/` (gitignored).

---

## How it's wired

Three pieces, all on the `anant` branch:

| File | Purpose |
|---|---|
| `tasks/<name>.json` | Problem text, training `data`, optional `test_data`, list of `targets`, and metadata. One file per task. |
| `llm_configs/<name>.json` | `name`, `api_url`, `api_key`, `api_model`, `llm_timeout`, `llm_max_retries`, `llm_retry_backoff`. One file per LLM. |
| `trial.py` | Reads one task + one LLM config, calls `llb.infer(...)`, writes `results_<task>_<llm>_n<N>.json`. |

Codes and samples from the two-stage paper sweep are **not** in the repo.
They live on Unity scratch at
`/scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results/<task>/<llm>/`.
See `experiments_progress.md` for the per-cell progress snapshot and
file schemas.

### Off-cluster: using the pre-generated codes zip

Collaborators without access to Unity scratch can run the full
`trial.py` pipeline from a downloadable archive of the codes. Ask
Eddie for the Google Drive link to `paper_results_codes.zip`
(≈230 MB, 41,806 programs across 5 populated cells:
hurricane_eal_counties × {qwen25_coder, gemma4_e4b, llama32} and
tornado_counts_plains × {qwen25_coder, gemma4_e4b}).

```bash
# 1. Download paper_results_codes.zip from the shared Drive link.
# 2. Unpack anywhere — the archive preserves paper_results/<task>/<llm>/codes/.
unzip paper_results_codes.zip -d ~/llb-codes

# 3a. Point --paper-results-root at the unpacked tree (auto-preload picks
#     up the right codes/ dir for each task+llm pair).
python trial.py \
  --task tasks/hurricane_eal_counties.json \
  --llm-config llm_configs/qwen25_coder.json \
  --n-models 100 \
  --paper-results-root ~/llb-codes/paper_results

# 3b. Or pass --preload-codes-dir explicitly for a single cell.
python trial.py \
  --task tasks/tornado_counts_plains.json \
  --llm-config llm_configs/qwen25_coder.json \
  --n-models 100 \
  --preload-codes-dir ~/llb-codes/paper_results/tornado_counts_plains/qwen25_coder/codes

# 3c. Sweep every populated cell from the archive.
python trial.py --sweep-paper --n-models 100 \
  --paper-results-root ~/llb-codes/paper_results
```

Equivalent to setting `LLB_PAPER_RESULTS_ROOT=~/llb-codes/paper_results`
in your shell so you can omit `--paper-results-root` on every call.

No Ollama / GPU required in this mode — inference is pure CPU NumPyro
on pre-generated programs.

---

## CLI reference

```
python trial.py [--task TASK.json] [--llm-config LLM.json]
                [--n-models 10,20,50]
                [--preload-codes-dir DIR]
                [--paper-results-root ROOT]
                [--sweep-paper]
                [--mcmc-warmup 500] [--mcmc-samples 1000]
                [--loo-warmup 50]   [--loo-samples 100]
                [--verbose]
```

- `--task` / `--llm-config` — required **unless** `--sweep-paper`.
- `--n-models` — comma-separated sweep of pool sizes. Default `10,20`. One
  output file per pool size.
- `--preload-codes-dir` — if given, skip LLM generation and load every
  `code_*.code.json` under this directory (uses the `canonical_code` field).
  If omitted but `<paper-results-root>/<task_stem>/<llm_name>/codes/` exists
  and has ≥1 code file, that directory is used automatically.
- `--paper-results-root` — where to look for codes when auto-deriving the
  preload dir or running `--sweep-paper`. Defaults to the scratch path
  above; can also be overridden via `LLB_PAPER_RESULTS_ROOT`.
- `--sweep-paper` — enumerate every `(task, llm)` pair with ≥1 code under
  `paper-results-root`, process each. Useful once Stage A has populated
  many cells.
- `--mcmc-*` / `--loo-*` — NUTS knobs forwarded to `llb.infer`. Defaults
  match the paper config; shrink them for smoke runs.
- `--verbose` — forwards to `llb.infer` (prints rich tables during
  stacking optimisation and LOO diagnostics).

---

## Output format

Per (task, llm, n_models), `trial.py` writes:

```
experiment_results_anant/results_<task>_<llm>_n<N>.json
```

Keys:
- `success` — `true` if inference produced ≥1 valid model.
- `task`, `llm_name`, `n_models`, `preload_codes_dir`, `timestamp`.
- `metrics` — flat dict with counts, timings, three epistemic variances
  (uniform / BMA / LOO), three weight vectors, entropy/ESS per scheme,
  `l1_distance_loo_bma`, posterior means, `final_loo_objective`,
  `log_marginal_per_model`.
- `full_result` — serialised form of the `llb.infer` return value
  (posteriors, diagnostics, everything).
- `model_codes` — the actual NumPyro programs that made it through
  inference.

After each run a combined `all_results.json` is rewritten.

---

## Common recipes

### 1. Reproduce hurricane × qwen25_coder at K=100, no LLM

```bash
python trial.py \
  --task tasks/hurricane_eal_counties.json \
  --llm-config llm_configs/qwen25_coder.json \
  --n-models 100 \
  --preload-codes-dir /scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results/hurricane_eal_counties/qwen25_coder/codes
```

### 2. K-sweep on the same cell

```bash
python trial.py \
  --task tasks/hurricane_eal_counties.json \
  --llm-config llm_configs/qwen25_coder.json \
  --n-models 10,20,50,100,200,500
```

(Preload dir auto-derived from the default scratch root.)

### 3. Sweep every populated cell

```bash
python trial.py --sweep-paper --n-models 100
```

Today that hits 4 cells (hurricane × {qwen25_coder, gemma4_e4b, llama32},
tornado × qwen25_coder).

### 4. Fast smoke to confirm the pipeline runs

```bash
python trial.py \
  --task tasks/tornado_counts_plains.json \
  --llm-config llm_configs/qwen25_coder.json \
  --n-models 10 \
  --mcmc-warmup 20 --mcmc-samples 40 \
  --loo-warmup 10 --loo-samples 20
```

Expect under two minutes on CPU. Some generated programs will fail
inference (LLM output is genuinely buggy in a few percent of cases) —
you need ≥1 survivor for the run to count as `success: true`.

### 5. Live LLM run (requires Ollama on `localhost:11434`)

```bash
ollama pull qwen2.5-coder:latest
python trial.py --task tasks/coin_flip.json \
                --llm-config llm_configs/qwen25_coder.json \
                --n-models 10
```

Omit `--preload-codes-dir` and leave `LLB_PAPER_RESULTS_ROOT` unset (or
point it at an empty directory) so auto-derivation doesn't pick up a
stale cell.

---

## Gotchas

- **All 3 codes failed inference.** If your `--n-models` is tiny and you
  get `LLM produced 0 valid models…`, try raising `--n-models`; the first
  few lexicographic SHAs in a real cell can all be bad code.
- **LOO is the hot path.** With `--n-models 20` and default LOO knobs
  this can take 15+ minutes per cell on CPU. Use `--loo-warmup 10
  --loo-samples 20` for smoke runs; use full knobs for paper numbers.
- **Auto-preload.** If a codes dir exists under
  `<paper-results-root>/<task_stem>/<llm_name>/codes`, it is used silently.
  Pass an explicit `--preload-codes-dir /dev/null`-like empty dir to
  force live LLM mode, or point `--paper-results-root` somewhere empty.
- **Output overwrites.** A second run with the same `(task, llm, n_models)`
  overwrites the results file.
- **`n_models` semantics differ between modes.** In live-LLM mode it is
  the *requested* number of generations (actual valid count is lower due
  to failures). In preload mode it is the *cap* on how many pre-generated
  codes to load (after sorting by sha).

---

## Branch layout

- `main` — original LLB (do not develop here).
- `model_gen` — two-stage GPU + CPU pipeline (`generate_codes.py`,
  `sample_from_codes.py`, `evaluate.py`, `scripts/`, `slurm/`). This is
  what populates `codes/` on scratch.
- `anant` — this branch. Consumes codes via `trial.py`. Single source of
  truth for the paper's weights and epistemic-variance numbers.

To run a new generation sweep, check out `model_gen` and follow
`docs/launch_checklist.md` there. Come back to `anant` to consume.

---

## Extending to a new (task, llm) cell

1. Write `tasks/<new_task>.json` with `text`, `data`, `targets`.
2. On `model_gen`, submit `scripts/submit_paper_experiments.sh --only task=<new_task>,llm=<existing_llm>`. Wait for Stage A to finish (codes under `<scratch>/<new_task>/<llm>/codes/`).
3. Back on `anant`:
   ```bash
   python trial.py --task tasks/<new_task>.json \
                   --llm-config llm_configs/<llm>.json \
                   --n-models 100
   ```

The preload dir is picked up automatically.
