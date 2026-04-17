# Paper experiments overview

This document is the single narrative description of the experiments backing
the paper **Stacking-Based Model Weighting and Epistemic Uncertainty in Large
Language Bayes**. It exists so that a new collaborator, a reviewer, or a future
version of us can read one page and understand what we ran, why, how, and where
the resulting artifacts live on disk.

For the operational recipe (exact commands, sbatch invocations, launch
checklist) see [experiment_pipeline.md](experiment_pipeline.md) and
[launch_checklist.md](launch_checklist.md). For the paper text see
[main.tex](../Improving-the-Epistemic-Uncertainty-of-Large-Language-Bayes/main.tex).

## Contents

1. [Research question in one paragraph](#1-research-question-in-one-paragraph)
2. [What counts as one "experiment"](#2-what-counts-as-one-experiment)
3. [Tasks](#3-tasks)
4. [Language models](#4-language-models)
5. [Pipeline, end to end](#5-pipeline-end-to-end)
6. [What gets saved, and where](#6-what-gets-saved-and-where)
7. [Metrics](#7-metrics)
8. [Cluster layout](#8-cluster-layout)
9. [Time and cost budget](#9-time-and-cost-budget)
10. [Reproducibility and reviewer access](#10-reproducibility-and-reviewer-access)
11. [Known risks and mitigations](#11-known-risks-and-mitigations)
12. [Glossary](#12-glossary)

---

## 1. Research question in one paragraph

Large Language Bayes (LLB) turns a text problem description into a set of
NumPyro programs by sampling from a language model, does Bayesian inference in
each program, then averages the posteriors across programs. The default
weighting scheme in LLB is Bayesian model averaging (BMA), which asymptotically
concentrates all the weight on whichever candidate model has the highest
evidence. In the M-open setting (where none of the candidate models is
correct) this collapse wastes the diversity of the candidate set. Our paper
asks whether replacing BMA with **leave-one-out (LOO) stacking weights**
yields better predictive performance and a more calibrated sense of
epistemic uncertainty. The experiments provide the empirical answer.

## 2. What counts as one "experiment"

One **cell** of the experimental grid is a (task, LLM) pair. For each cell we
sample 10,000 distinct valid NumPyro programs from the LLM, run posterior
inference in each, and then compute three different weighted mixtures:
uniform, BMA, and LOO stacking. We compare those three weightings on the
same downstream metrics.

We run five tasks and three LLMs, giving **15 cells total**. Every cell is
self-contained on disk; evaluation is always done within a cell, never across
them.

## 3. Tasks

Every task is a JSON file in [tasks/](../tasks/) with six slots: a natural
language `text` description, training `data`, optional held-out `test_data`,
a list of `targets` (names of latent variables the reviewer should be able to
inspect), a `task_type`, and a `metadata` block with full provenance.

Five real tasks plus the original coin flip from the LLB paper.

### 3.1 The five FEMA National Risk Index tasks

All five come from the [FEMA National Risk Index v1.20](https://www.fema.gov/about/openfema/data-sets/national-risk-index-data)
county-level feature service. Counties are sampled with a fixed seed per task
so any of them can be reproduced with `uv run python -m tasks.build_fema_nri_tasks`.
The five tasks are engineered to exercise three different data regimes:
heavy-tailed continuous losses (useful for the paper's Pareto-tail
discussion), integer count data, and small positive rates.

| Task file | Type | Data variable | N train | N test | Target latent |
|---|---|---|---|---|---|
| `hurricane_eal_counties.json` | prediction | `losses_musd` (USD millions, Hurricane EAL) | 20 | 10 | `median_loss` |
| `wildfire_eal_west.json` | prediction | `losses_musd` (USD millions, Wildfire EAL) | 20 | 10 | `median_loss` |
| `inland_flood_eal.json` | prediction | `losses_musd` (USD millions, Inland Flood EAL) | 20 | 10 | `median_loss` |
| `tornado_counts_plains.json` | estimation | `events` (integer count, recorded tornadoes) | 20 | 0 | `rate` |
| `earthquake_frequency_west.json` | estimation | `frequency_per_year` (events/yr) | 20 | 0 | `mean_frequency` |

Why these attributes:

- Three heavy-tailed continuous prediction tasks force the LLM to pick among
  LogNormal, Gamma, Weibull, Pareto, and half-Normal families. Small n plus
  heavy tails means no single parametric family is right, which is the exact
  M-open regime the paper wants to study.
- Tornado counts are a clean integer count task where Poisson vs Negative
  Binomial vs ZIP vs Gamma-Poisson are all defensible candidates.
- Earthquake frequency is a tiny-positive-rate task with Gamma vs LogNormal
  vs Exponential as the obvious candidates.

Why n=20: LOO refits NUTS per held-out point, so LOO cost scales linearly in
n. Cutting from 30 to 20 saves a third of the inference time with no loss of
signal at this sample size.

Target latent names (`median_loss`, `rate`, `mean_frequency`) were picked to
be well-defined across every plausible model family. In particular
`median_loss` remains finite for Pareto with shape parameter <= 1 where the
mean diverges.

### 3.2 The coin-flip canary

`tasks/coin_flip.json` is the original LLB coin task. We keep it in the
regression test suite as a fast canary but it is not part of the paper's
main result.

## 4. Language models

Three LLMs, all run locally via Ollama on a private port per Slurm array task.
The three were chosen to span a useful quality range:

| Config | Model tag | Approx. size | Role |
|---|---|---|---|
| `llm_configs/qwen25_coder.json` | `qwen2.5-coder:latest` (~7B Q4) | 4.7 GB | Strongest coder; expected highest valid-generation rate |
| `llm_configs/gemma4.json` | `gemma4:e4b` | 3.5 GB | Mid-tier general model; expected mid generation quality |
| `llm_configs/llama32.json` | `llama3.2:latest` (3B) | 2.0 GB | Small general model; expected lowest generation quality |

Having three models with different generation quality is itself a finding:
we expect the paper to discuss how BMA vs stacking behave when the candidate
pool has a high rate of bad programs (llama3.2) versus a high rate of good
programs (qwen2.5-coder).

Model prompts come from [llb/model_generator.py](../llb/model_generator.py)
`build_messages`, which shows the LLM a few in-context example programs
before the actual task text. The prompt is saved verbatim alongside every
generated code, so nothing about the conditioning is lost.

## 5. Pipeline, end to end

```mermaid
flowchart LR
  spec[Task JSON] --> sA[Stage A<br/>GPU + Ollama]
  sA -->|code_SHA.code.json| codes[(codes/ directory)]
  codes --> sB[Stage B<br/>CPU only]
  sB -->|sample_SHA.npz + meta| samples[(samples/ directory)]
  samples --> agg[Aggregator]
  agg --> summary[summary.csv, test_scores.csv,<br/>top_models/, loo_matrices/, ...]
  summary --> paper[Paper tables and figures]
```

### 5.1 Stage A: code generation on GPU

- One Ollama server per Slurm array task on a private port, loading one of
  the three LLMs.
- For each slot index, `generate_models_with_diagnostics` calls the LLM with
  up to 4 attempts, re-prompting when the returned program fails to parse or
  fails to declare the goal latent site.
- On success, the program is **AST-canonicalized** via
  `ast.unparse(ast.parse(raw_code))` and hashed. The first 16 hex characters
  of the `sha256` are the program's identifier.
- We try to atomically claim `codes/code_<sha>.code.json` via
  `open("x")`. If the file already exists, the code is a duplicate of an
  earlier generation; we append a row to `codes/_index.jsonl` and move on.
- On failure, a one-line JSON goes to `codes/_failures/` for later forensic
  inspection by a reviewer.
- Every ~10 slots the shard counts the number of files in `codes/` and
  exits once the cell-wide target (10,000 distinct valid programs) is
  reached.

The reason this stage is GPU-only: the only expensive thing happening is LLM
sampling, which is what GPUs are for. MCMC and LOO do not happen here.

### 5.2 Stage B: inference on CPU

- Enumerates `codes/code_*.code.json` for a cell.
- Array tasks partition the sorted hash list: task id `i` out of `n`
  processes `hashes[i::n]`.
- For each hash, if `samples/sample_<sha>.meta.json` exists with status `ok`
  we skip (idempotent resume).
- Otherwise run in order:
  1. **NUTS** with 500 warmup steps and 1000 samples to draw from the posterior.
  2. **IWAE log marginal bound** with 5 inner and 80 outer samples.
  3. **True leave-one-out**: for each of the n_train held-out points,
     run NUTS on the remaining n_train - 1 points, then estimate
     log p(x_i | x_{-i}, m) via variational importance weighting.
     Vectorized over the n_train indices with
     `eqx.filter_vmap`.
  4. For prediction tasks, **held-out test scoring**: for each of the n_test
     points, compute log p(x_test_j | x_train, m) under the same VI
     machinery. Stored as `test_log_liks`.
- NUTS diagnostics (number of divergences, per-site r-hat, per-site n_eff)
  are extracted from the MCMC object and saved in `meta.json` under
  `mcmc.diagnostics`.

The reason this stage is CPU-only: NumPyro and JAX run perfectly on CPU for
our tiny n=20 problems, the CPU queue is far faster than the GPU queue,
and avoiding GPU bring-up time (~60 seconds per node) and waste amortizes
across thousands of programs.

### 5.3 Aggregation

[scripts/aggregate_paper_results.py](../scripts/aggregate_paper_results.py)
walks every `experiment_results/paper/<task>/<llm>/` directory and produces:

- `summary.csv`: one row per (task, llm, K) with posterior means, epistemic
  variances, ESS, entropy, L1 and KL between BMA and stacking, stacking
  objective, held-out log predictive density, and bootstrap 5th and 95th
  percentiles.
- `generation_summary.csv`: per cell attempts, distinct valid count,
  duplicates, failure histogram, mean per-stage timings.
- `test_scores.csv`: per prediction cell, held-out log predictive density
  under uniform vs BMA vs stacking at each K.
- `loo_matrices/<task>_<llm>.npz`: the (n_train, K) matrix of LOO log
  likelihoods for bootstrap analysis.
- `cumulative_weight_curves/<task>_<llm>.csv`: rank vs cumulative weight,
  directly plottable as a Lorenz curve.
- `top_models/<task>_<llm>/`: the 10 highest-weighted programs under each
  weighting scheme, as copyable `.py` files named with their weight.

Everything the paper reports is either a cell of one of those files or a
simple function of them.

## 6. What gets saved, and where

```
experiment_results/paper/
  <task>/<llm>/
    codes/
      code_<sha16>.code.json        # raw response, prompt, raw code, canonical
                                    #   code, seed, per-attempt diagnostics
      _index.jsonl                  # one line per attempt, including duplicates
      _failures/                    # per-failure diagnostics for reviewer
    samples/
      sample_<sha16>.npz            # target posterior samples, loo_log_liks,
                                    #   test_log_liks, log_marginal_bound,
                                    #   optional full posterior per site
      sample_<sha16>.meta.json      # status, reasons, timings, MCMC/IWAE/LOO
                                    #   diagnostics, NUTS divergences/r-hat/n-eff
    top_models/                     # populated by aggregator
    _manifest/
      shard_<uuid>.json             # git commit, Python/JAX/NumPyro versions,
                                    #   host, UTC start/end, CLI args, file hashes
  _submissions/
    <timestamp>.txt                 # Slurm job ids submitted in this sweep
  aggregate_metrics.jsonl           # per (task, llm, K) evaluate.py rows
  summary.csv                       # paper-ready table
  generation_summary.csv
  test_scores.csv
  loo_matrices/
  cumulative_weight_curves/
  top_models/
```

A few properties worth highlighting:

- Hash-addressed filenames mean dedup and resume are a single filesystem
  operation. No external database.
- Every artifact is atomic (`tmp` then `os.replace`) so preemption cannot
  leave half-written files behind.
- Every artifact includes enough breadcrumbs (git commit, versions,
  prompt, raw response) to reconstruct how it came to be, without needing
  the shell history.

## 7. Metrics

### 7.1 Weighted mixtures

Given K programs `m_1, ..., m_K` and weights `w`, the paper's predictive
distribution is `p(z | x) = sum_k w_k p(z | x, m_k)`. We compute three
weight vectors per cell:

- **Uniform**: `w_k = 1/K`. The trivial baseline.
- **BMA**: `w_k ∝ exp(log p(x | m_k))` where the IWAE log marginal bound
  stands in for the intractable log marginal. Softmax over the K log
  bounds.
- **Stacking**: SLSQP solve on the simplex that maximises the leave-one-out
  log predictive density of the mixture.

### 7.2 Predictive performance

For prediction tasks (hurricane, wildfire, inland flood), we report the
mean held-out log predictive density on the 10 test points:

```
log_pred_test(w) = (1/n_test) * sum_j log [ sum_k w_k * p(x_test_j | x_train, m_k) ]
```

under each of uniform, BMA, stacking. This is the primary "does stacking
predict better?" metric.

For estimation tasks (tornado, earthquake) we report only the training LOO
objective and the target posterior mean plus epistemic variance.

### 7.3 Epistemic uncertainty

Following [main.tex](../Improving-the-Epistemic-Uncertainty-of-Large-Language-Bayes/main.tex)
Section 4.1, we use the Bessel-corrected weighted between-model variance
of the target posterior mean:

```
V_epi = sum_k w_k (mu_k - mu_bar)^2 / (1 - sum_k w_k^2)
```

where `mu_k = E[z | x, m_k]` is the posterior mean under program k.
Reported under each of the three weight vectors.

### 7.4 Weight concentration diagnostics

- **Effective sample size**: `ESS(w) = 1 / sum_k w_k^2`. Quantifies collapse.
- **Entropy**: `H(w) = -sum_k w_k log w_k`.
- **Max weight**: biggest single contribution.
- **L1 distance between BMA and stacking**: `sum_k |w_k^{BMA} - w_k^{stack}|`.
- **KL(BMA || stacking)** and **KL(stacking || BMA)**.

### 7.5 K sweep

All metrics are computed at K in `{10, 20, 50, 100, 200, 500, 1000, 2000,
5000, 10000}` so the paper can show how the comparison changes with the
pool size. This is a standard curve plot.

### 7.6 Bootstrap bands

To convert every point estimate into an interval, we resample the n_train
training points with replacement B=200 times, recompute stacking weights and
every downstream scalar, and report the 5th and 95th percentiles alongside
the point estimate.

## 8. Cluster layout

Unity cluster, three partitions.

### 8.1 Stage A on `gpu-preempt`

- `--gres=gpu:1` with no type constraint. The smallest preempt GPU is an
  A16 with 15 GB VRAM; all three LLMs fit comfortably.
- `--cpus-per-task=2`, `--mem=16G`, `--time=03:00:00`.
- `--array=0-99` per cell. Each array task owns 150 slot indices
  (`SHARD_START = SLURM_ARRAY_TASK_ID * 150`), so the cell has a hard cap
  of 15,000 attempts and a target of 10,000 valid distinct codes.
- `--requeue` on. Resume is via the hash registry in `codes/`.

### 8.2 Stage B on `cpu-preempt`

- No GPU. `--cpus-per-task=16`, `--mem=32G`, `--time=06:00:00`.
- `--array=0-99`. Each array task handles `hashes[i::100]`, which is about
  100 hashes at ~3 minutes each with vectorized LOO.
- Fallback to `cpu` if the preempt queue is contested on a given day.
- Verified against the [Unity CPU summary](https://docs.unity.rc.umass.edu/documentation/cluster_specs/cpu_summary/):
  both partitions have ample nodes with >= 7 GB per core.

### 8.3 Pipelining between stages

Stage B is submitted with
`sbatch --dependency=afterany:<stage_A_jobid>`, so it starts as soon as
Stage A's jobs finish (successful or preempted). Because Stage B enumerates
`codes/` at launch, any codes Stage A adds after Stage B starts are picked up
by `slurm/sample_cpu_topup.sbatch` in a top-up pass, not by the initial pass.

### 8.4 Submission orchestration

[scripts/submit_paper_experiments.sh](../scripts/submit_paper_experiments.sh)
handles the 15-cell matrix:

```
bash scripts/submit_paper_experiments.sh --all
bash scripts/submit_paper_experiments.sh --only task=hurricane_eal_counties,llm=qwen25_coder
bash scripts/submit_paper_experiments.sh --only task=tornado_counts_plains,llm=gemma4_e4b --smoke
```

[scripts/watch_progress.sh](../scripts/watch_progress.sh) prints a table of
every cell's distinct-code count, pending samples, and last-updated timestamp
once a minute for live monitoring.

## 9. Time and cost budget

Rough estimates from the A16 smoke run, extrapolated to the full sweep.

| Stage | Per model | Per cell wall (100 parallel tasks) | Across 15 cells serialized | With 3 cells concurrent |
|---|---|---|---|---|
| A (GPU) | ~60 s | ~1.7 h | ~25 h | ~8 h |
| B (CPU, LOO vmap) | ~90 s | ~1.5 h | ~23 h | ~8 h |

Total sweep wall clock is roughly one to two days if we can keep 3 cells
concurrent across both stages. Preemption can extend this but
the pipeline resumes without data loss.

Disk is about 60 GB under `experiment_results/paper/` with full posterior
samples enabled. Scratch has plenty of room; no special cleanup needed.

## 10. Reproducibility and reviewer access

Every scalar in the paper can be traced back to a saved artifact. The
table below is the reviewer's cheat sheet.

| Reviewer question | Artifact |
|---|---|
| How many attempts vs distinct vs inference-ok per cell? | `generation_summary.csv` |
| Show me the LLM response for a specific model. | `codes/code_<sha>.code.json` |
| Git commit and library versions? | `_manifest/shard_<uuid>.json` |
| Top 10 by stacking weight with code. | `top_models/<task>_<llm>/` |
| NUTS divergences, r-hat, n_eff per model. | `samples/sample_<sha>.meta.json` under `mcmc.diagnostics` |
| Held-out test log predictive density? | `test_scores.csv` and `samples/*.npz` `test_log_liks` |
| Re-evaluate at a different K. | `uv run python -m evaluate --ks ...` |
| Bootstrap CI on stacking vs BMA. | Already in `summary.csv`. |
| Per-point LOO matrix. | `loo_matrices/<task>_<llm>.npz` |
| Cumulative weight curve. | `cumulative_weight_curves/<task>_<llm>.csv` |
| Re-dedup under a different hash policy. | Raw responses in `codes/*.code.json`, rehash offline. |
| Swap true-LOO for PSIS-LOO. | Same `loo_matrices`, apply PSIS offline. |
| Why was model X's LOO so negative? | Open its `meta.json`, read the reason; open its `.code.json` for the full code. |

A short demo notebook lives in `analysis.py` / `explore_models.ipynb`
showing how to load any of these files and reproduce any table or figure.

## 11. Known risks and mitigations

- **Stage A stall**: Ollama dies or the GPU gets evicted. Watch script
  surfaces this as `last_code_age_s` climbing past ~10 minutes. Resubmit the
  cell with `--only`. Hash-addressed artifacts mean no work is lost.
- **Stage B OOM**: unlikely at n=20, but possible if a generated model draws
  thousands of latents. Mitigation: 32 GB per task, `--requeue`, and a
  per-model exception handler in `process_index` that writes an
  `inference_error` meta instead of crashing the whole shard.
- **Cell comes up short**: if fewer than 10,000 distinct codes are produced
  (because the LLM is too diverse or shards ran out of indices), submit a
  supplementary Stage A wave with a higher `SHARD_MAX_INDICES` and then
  `sample_cpu_topup.sbatch` for the new hashes.
- **Solver slow at K=10000**: `_solve_stacking` uses SLSQP over the simplex.
  [tests/test_evaluate_large_k.py](../tests/test_evaluate_large_k.py)
  checks that it finishes in under 60 seconds before we rely on it.
- **Semantic duplicates**: AST canonicalization collapses only cosmetic
  differences. Two programs that differ only in variable names count as
  distinct. If a reviewer complains, we re-dedup offline with any policy
  they prefer (raw responses are preserved).
- **Divergences in NUTS for some generated models**: expected; we record the
  divergence count and let the reviewer filter.
- **LLM cache on a node**: first call to Ollama pulls the model (~60 s). We
  can avoid paying this 100 times per cell with a per-node prelude job that
  pulls the model once and touches a sentinel file.

## 12. Glossary

- **BMA**: Bayesian model averaging. Weights programs by marginal
  likelihood.
- **Cell**: one (task, LLM) pair. 15 cells total.
- **Distinct valid code**: a generated program that parses, declares the
  required goal sites, compiles, and is not an AST-canonical duplicate of
  an earlier program in the same cell.
- **ESS**: effective sample size of a weight vector.
- **IWAE bound**: importance-weighted autoencoder lower bound on the log
  marginal likelihood.
- **LOO**: leave-one-out. We use **true** LOO (refits NUTS per held-out
  point) as opposed to PSIS-LOO.
- **M-open**: setting where the true data-generating model is not in the
  candidate set. LLB always operates here.
- **Stage A**: GPU-only LLM code generation.
- **Stage B**: CPU-only posterior inference and scoring.
- **Stacking**: weight selection that maximises mixture LOO log predictive
  density over the simplex.
- **Target**: the name of a latent or deterministic site the program must
  declare (for example `median_loss`), used to produce reviewer-readable
  posterior samples of that quantity.
