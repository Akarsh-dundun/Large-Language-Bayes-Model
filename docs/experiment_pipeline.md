# Experiment pipeline tutorial

This document describes the infrastructure used to run the BMA versus stacking
experiments for the paper draft in
[Improving-the-Epistemic-Uncertainty-of-Large-Language-Bayes/main.tex](../Improving-the-Epistemic-Uncertainty-of-Large-Language-Bayes/main.tex).

The pipeline now runs in **two cluster stages** so the GPU is only touched
during LLM generation and CPU partitions handle the bulk of the work
(MCMC, IWAE, true-LOO, test scoring). For the design narrative see
[stacking_experiments_overview.md](../stacking_experiments_overview.md). For
the launch checklist see [launch_checklist.md](launch_checklist.md).

## Two-stage pipeline overview

```
tasks/<name>.json                                  # problem description
llm_configs/<name>.json                            # how to reach an LLM
        |
        v
generate_codes.py  <--- gpu-preempt array (Stage A, one Ollama per task)
        |
        v
experiment_results/paper/<task>/<llm>/codes/
        code_<sha16>.code.json                     # AST-canonicalized, deduped
        _index.jsonl
        _failures/
        |
        v
sample_from_codes.py  <--- cpu-preempt array (Stage B, NUTS+IWAE+LOO+test)
        |
        v
experiment_results/paper/<task>/<llm>/samples/
        sample_<sha16>.npz                         # target samples, loo_log_liks,
        sample_<sha16>.meta.json                   # test_log_liks, MCMC diagnostics
        |
        v
scripts/aggregate_paper_results.py
        |
        v
experiment_results/paper/summary.csv               # paper-ready scalars
                            generation_summary.csv # per cell counts and timings
                            test_scores.csv        # held-out predictive density
                            loo_matrices/          # bootstrap-ready raw LOO
                            cumulative_weight_curves/
                            top_models/
```

The legacy single-stage [generate_and_sample.py](../generate_and_sample.py) is
preserved as a thin local wrapper used by the smoke tests in
`tests/test_generate_and_sample_and_evaluate.py`. It writes the older
`model_<index>.{npz,meta.json}` layout. Production paper runs use Stage A
plus Stage B.

`trial.py` is kept only as an end-to-end sanity-check driver that calls
`llb.infer` directly. It is not part of the paper loop.

## Stage A: code generation on GPU

[generate_codes.py](../generate_codes.py) is the Stage A driver. Per slot in
the shard's range:

1. Call `generate_one_with_full_diagnostics` on the LLM (up to 4 attempts with
   feedback on failure).
2. AST-canonicalize the resulting code via `ast.unparse(ast.parse(...))`,
   take the first 16 hex chars of the sha256 as the hash.
3. Atomically claim `codes/code_<sha>.code.json` via `os.O_EXCL`. Duplicates
   are recorded in `codes/_index.jsonl` without overwriting.
4. Failures land in `codes/_failures/failure_<slot>.json` for forensic
   inspection.

Each shard polls `len(codes/code_*.code.json)` and exits as soon as the
cell-wide distinct target (default 10,000) is reached or the shard's slot
range is exhausted (default 150 slots, default 100 distinct-valid target).

Sbatch wrapper: [slurm/gen_codes_gpu.sbatch](../slurm/gen_codes_gpu.sbatch).
Partition `gpu-preempt`, `--gres=gpu:1`, 2 CPUs, 16 GB RAM, 3 hour wall,
`--array=0-99`, `--requeue`.

## Stage B: posterior inference on CPU

[sample_from_codes.py](../sample_from_codes.py) is the Stage B driver. Each
array task picks `hashes[i::n]` from `codes/`, then for each hash:

1. Skip if `samples/sample_<sha>.meta.json` already has status `ok` (resume).
2. Run NUTS (extra fields capture divergences).
3. Estimate the IWAE log marginal bound.
4. Run true LOO (parallelisable across data points via
   `--loo-workers > 1`).
5. For prediction tasks, compute per-test-point log p(x_test_i | x_train, m)
   from posterior draws (`test_log_liks`).
6. Extract NUTS divergences, per-site r-hat, n-eff and store under
   `meta.mcmc.diagnostics`.

Sbatch wrapper: [slurm/sample_cpu.sbatch](../slurm/sample_cpu.sbatch).
Partition `cpu-preempt`, no GPU, 16 CPUs, 32 GB RAM, 6 hour wall,
`--array=0-99`, `--requeue`. Top-up variant for late fills:
[slurm/sample_cpu_topup.sbatch](../slurm/sample_cpu_topup.sbatch).

## Submission

[scripts/submit_paper_experiments.sh](../scripts/submit_paper_experiments.sh)
takes `--all` or one or more `--only task=...,llm=...` filters, plus a
`--smoke` mode for a 5-valid-target sanity run. Each cell's Stage B is
submitted with `--dependency=afterany` on its Stage A.

## Aggregation

[scripts/aggregate_paper_results.py](../scripts/aggregate_paper_results.py)
walks `experiment_results/paper/<task>/<llm>/`, runs `evaluate.py` per cell
across a shared K grid, and writes the reviewer-ready CSVs and per-cell
artifacts described in the launch checklist.

## Reviewer cheat sheet

| Question | Where on disk |
|---|---|
| Show the LLM response for model X | `codes/code_<sha>.code.json` |
| Git commit, JAX/NumPyro versions, host | `_manifest/shard_<uuid>.json` |
| Top-K NumPyro programs by stacking weight | `top_models/<task>_<llm>/` |
| NUTS divergences and r-hat per model | `samples/sample_<sha>.meta.json` under `mcmc.diagnostics` |
| Test-set predictive density per scheme | `test_scores.csv` |
| Per-cell raw LOO matrix for bootstrap | `loo_matrices/<task>_<llm>.npz` |
| Cumulative weight curve | `cumulative_weight_curves/<task>_<llm>.csv` |
| Re-evaluate at a different K | `evaluate.py --ks ...` against the same `samples/` dir |



## Stage 1, task specs

A task is a JSON file with this shape.

```json
{
  "name": "coin_flip",
  "text": "I have a bunch of coin flips. What's the bias?",
  "data": {"flips": [0, 1, 0, 1, 1, 0]},
  "test_data": null,
  "targets": ["true_bias"],
  "true_latents": null,
  "task_type": "estimation",
  "metadata": {"internet_seen": true, "notes": "Original LLB coin example."}
}
```

Fields used by the pipeline today.

- `text`, `data`, `targets` feed the LLM prompt and NumPyro inference.
- `task_type` is informational, used by `evaluate.py` to decide which metrics
  apply (estimation tasks report epistemic variance on latents, prediction
  tasks would add held-out log likelihood on `test_data`).
- `test_data` and `true_latents` are placeholders for new tasks. They are unused
  for the coin task.

Adding a new task is one file. Nothing else needs to change.

## Stage 2, LLM configs

Each file in `llm_configs/` captures everything about one generator.

```json
{
  "name": "gemma4_e4b",
  "api_url": "http://127.0.0.1:11434/api/generate",
  "api_key": null,
  "api_model": "gemma4:e4b",
  "llm_timeout": null,
  "llm_max_retries": 2,
  "llm_retry_backoff": 2.0
}
```

For Ollama on a Slurm node, the sbatch script overrides `api_url` at runtime so
it points at the per-task local Ollama port. You do not need to edit the JSON.

## Stage 3, generation plus PPL sampling

### What happens per model index

`generate_and_sample.py` processes a contiguous range of global model indices.
For each index `i`:

1. Call the LLM with seed `base_seed + i` to get a NumPyro code string. The
   prompt uses the same `build_messages` logic as the rest of the package.
2. Validate the code (regex for duplicate site names and missing goal names).
3. Run NUTS on the task data.
4. Estimate the IWAE log marginal likelihood bound.
5. Estimate the leave-one-out log predictive density vector (true LOO by
   default, which refits NUTS per held-out point).
6. Write `model_{i:06d}.npz` and `model_{i:06d}.meta.json` atomically using a
   tmp file plus `os.replace`.

A model is considered done when its sibling `.meta.json` exists with a
populated `status`. A restart of the same index reads that file and returns
immediately, which is what makes the pipeline resilient to Slurm preemption.

### Artifact layout

Each model produces two files in the output directory.

`model_{i:06d}.npz` contains numeric arrays.

- `target__<name>` per target, shape `(mcmc_num_samples, ...)`.
- `log_marginal_bound`, a scalar float.
- `loo_log_liks`, shape `(n_datapoints,)`.
- `post__<site>` per posterior site, only when `--save-full-posterior` is set.

`model_{i:06d}.meta.json` contains everything that is not a numpy array.

- `status` in `{ok, partial, syntax_error, compile_error, inference_error,
  missing_targets, generation_failed, inference_partial, unhandled_exception}`.
- The raw `code` string returned by the LLM.
- Timings (`generation_seconds`, `mcmc_seconds`, `log_marginal_seconds`,
  `loo_seconds`).
- Nested `log_marginal` and `loo` blocks with their own status and reason
  fields, which let `evaluate.py` keep the partial models that have useful
  marginal likelihoods but failed LOO, and vice versa.

### Running locally, one model

Useful for debugging a new task.

```bash
uv run python -m generate_and_sample \
  --task tasks/coin_flip.json \
  --llm-config llm_configs/gemma4.json \
  --output-dir sample_cache/coin_flip/gemma4_e4b \
  --shard-start 0 --shard-count 1 --seed 42
```

After it finishes, inspect `sample_cache/coin_flip/gemma4_e4b/model_000000.meta.json`
to see the code that was generated and the timing breakdown.

### Running on Slurm

`slurm/gen_and_sample_coin.sbatch` reads everything it needs from environment
variables, so one sbatch works for every (task, LLM) pair.

```bash
TASK_PATH=tasks/coin_flip.json \
LLM_CONFIG_PATH=llm_configs/gemma4.json \
OLLAMA_MODEL=gemma4:e4b \
OUTPUT_DIR=sample_cache/coin_flip/gemma4_e4b \
SHARD_SIZE=100 \
sbatch slurm/gen_and_sample_coin.sbatch
```

The default array is `0-99` with `SHARD_SIZE=100`, producing 10000 model
indices. Each task starts its own Ollama on port `11434 + SLURM_ARRAY_TASK_ID`,
pulls the model tag, and invokes `generate_and_sample.py`. The script has
`#SBATCH --requeue`, and the resume-on-existing-meta logic inside the Python
driver means preempted tasks pick up exactly where they stopped.

To change MCMC or LOO sizes without editing the script:

```bash
MCMC_NUM_WARMUP=200 MCMC_NUM_SAMPLES=500 \
LOO_NUM_WARMUP=20 LOO_NUM_SAMPLES=50 \
USE_TRUE_LOO=0 \
TASK_PATH=tasks/coin_flip.json \
LLM_CONFIG_PATH=llm_configs/llama32.json \
OLLAMA_MODEL=llama3.2:latest \
OUTPUT_DIR=sample_cache/coin_flip/llama32 \
sbatch slurm/gen_and_sample_coin.sbatch
```

`USE_TRUE_LOO=0` switches to the PSIS-LOO approximation from
[llb/mcmc_log.py](../llb/mcmc_log.py), which is much cheaper when true LOO is
the bottleneck.

### Reading the output

```bash
# count artifacts by status
jq -r '.status' sample_cache/coin_flip/gemma4_e4b/model_*.meta.json | sort | uniq -c

# look at the code the LLM produced for model 0
jq -r '.code' sample_cache/coin_flip/gemma4_e4b/model_000000.meta.json

# what took the longest
jq -r '[.index, .timings.loo_seconds] | @tsv' \
  sample_cache/coin_flip/gemma4_e4b/model_*.meta.json \
  | sort -k2 -n | tail -10
```

## Stage 4, evaluation

`evaluate.py` is the only module you rerun when you change a weighting idea,
sweep K, or add a new metric. It does not call the LLM and it does not run
NUTS.

```bash
uv run python -m evaluate \
  --task tasks/coin_flip.json \
  --sample-dir sample_cache/coin_flip/gemma4_e4b \
  --llm-name gemma4_e4b \
  --ks 10 20 50 100 500 1000 5000 10000 \
  --output-jsonl experiment_results/metrics.jsonl \
  --output-full-json experiment_results/coin_flip_gemma4_full.json
```

What it computes per K.

- `uniform`, `bma`, `stacking` weight vectors. BMA uses a numerically stable
  softmax over `log_marginal_bound`. Stacking uses SLSQP with analytical
  gradient against the mean-over-datapoints log stacking score from the paper.
- Weight statistics per scheme: entropy, effective sample size, max weight,
  `l1_bma_stacking`, `kl_bma_given_stacking`, `kl_stacking_given_bma`.
- Per target: posterior mean and Bessel-corrected between-model variance under
  each weighting.
- `stacking_objective`, the value of the paper's optimization problem at the
  solution.
- `log_marginal_distribution` summary (min, max, mean, std across models),
  which is what the NOTES section calls out as a tail-behavior diagnostic.

Each K produces one row in the JSONL file and one entry in the full JSON dump.
If `--ks` is omitted, a log-spaced default sweep is used, clipped to the number
of usable artifacts.

Reading the JSONL in pandas:

```python
import pandas as pd
rows = pd.read_json("experiment_results/metrics.jsonl", lines=True)
print(rows[["task", "llm", "k_requested", "n_bma_finite", "n_loo_valid"]])
```

## Tests

`tests/test_generate_and_sample_and_evaluate.py` exercises the full pipeline
without needing a live LLM. It mocks `LLMClient.generate` and drives
`generate_and_sample.py` and `evaluate.py` end to end on the coin task with
small MCMC sizes.

```bash
uv run python -m pytest tests/test_generate_and_sample_and_evaluate.py -q
```

Nine tests, roughly 30 seconds on CPU. Cover atomic writes, resume-on-existing,
each failure mode (syntax, compile, missing targets), the CLI round trip, and
the stacking optimizer on dominant and mixture cases.

## Adding a new task

1. Create `tasks/your_task.json` following the coin schema. Include `test_data`
   if you want held-out log likelihood in evaluation, and `true_latents` if the
   task is synthetic and you want coverage diagnostics.
2. Kick off generation. Set `OUTPUT_DIR=sample_cache/your_task/<llm_name>` so
   the artifact directory mirrors the task and LLM names.
3. Evaluate with `evaluate.py`. No code changes required.

## Adding a new LLM

1. Create `llm_configs/your_llm.json`. The minimum fields are `name`, `api_url`,
   `api_model`. Set `api_key` if the endpoint is not Ollama.
2. Point the sbatch at it via `LLM_CONFIG_PATH` and set `OLLAMA_MODEL` to the
   tag to pull. For non-Ollama endpoints you can skip the Ollama-specific
   environment variables and run `generate_and_sample.py` directly instead of
   through `gen_and_sample_coin.sbatch`.

## Performance notes

True LOO dominates wall time. Each held-out point refits NUTS. For the coin
task with six observations and the defaults (500 warmup, 1000 samples, 50 warmup
and 100 samples for LOO, 25 inner IWAE samples), per-model wall time on CPU is
30 to 60 seconds. GPU nodes are substantially faster for NUTS but true LOO still
scales linearly with the number of observations.

When the observation count grows, either raise the Slurm time limit, drop
`USE_TRUE_LOO=0` to use PSIS-LOO, or reduce `LOO_NUM_SAMPLES`. The switch is a
single environment variable and does not invalidate existing artifacts written
with the other setting, since the LOO method used is recorded in the meta file
under `loo.use_true_loo`.
