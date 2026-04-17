# Launch checklist for the paper sweep

Operational recipe for kicking off the two-stage paper sweep on Unity. Run
through the pre-flight section first, then a smoke run, then the full launch.

For the design narrative see [stacking_experiments_overview.md](../stacking_experiments_overview.md).
For the per-stage architecture see [experiment_pipeline.md](experiment_pipeline.md).

## Pre-flight (5 to 10 minutes)

1. Tests green:

   ```
   uv run pytest tests/ -q --ignore=tests/test_gemma4_live.py --ignore=tests/test_llama32_live.py
   ```

   Should report ~93 passed.

2. FEMA task JSONs match the committed builder:

   ```
   uv run python -m tasks.build_fema_nri_tasks
   git status tasks/
   ```

   No diffs expected. If anything changed, decide whether to commit before
   launching.

3. Ollama binary and model blobs ready on scratch:

   ```
   ls /scratch3/workspace/edmondcunnin_umass_edu-siple/ollama/bin/ollama
   ls /scratch3/workspace/edmondcunnin_umass_edu-siple/ollama_models/manifests/registry.ollama.ai/library/
   # expect: gemma4 llama3.2 qwen2.5-coder
   ```

4. Repo is at the commit you want recorded in the run manifest:

   ```
   git rev-parse HEAD
   git status --short
   ```

   The run manifest captures both, so any uncommitted change will be
   visible to a reviewer.

5. Disk free for ~60 GB under `experiment_results/paper/`:

   ```
   df -h /home/edmondcunnin_umass_edu/Large-Language-Bayes-Model/
   ```

## One-cell smoke (10 to 15 minutes wall)

Before launching the 15-cell matrix, validate one cell end to end on the
cluster.

```
bash scripts/submit_paper_experiments.sh \
  --only task=tornado_counts_plains,llm=qwen25_coder \
  --smoke
```

This sets `cell_valid_target=5`, `shard_max_indices=20`, `array=0-0`. Stage A
should finish in a few minutes, Stage B finishes ~5 to 10 minutes after.

Inspect:

```
bash scripts/watch_progress.sh experiment_results/paper_smoke 0
ls experiment_results/paper_smoke/tornado_counts_plains/qwen25_coder/codes/
ls experiment_results/paper_smoke/tornado_counts_plains/qwen25_coder/samples/
uv run python -m scripts.aggregate_paper_results --root experiment_results/paper_smoke
cat experiment_results/paper_smoke/summary.csv
```

Sanity criteria for the smoke:
- At least 5 `code_*.code.json` files exist.
- At least 3 of the corresponding `sample_*.meta.json` files have `status=ok`.
- `summary.csv` has at least one row per K, with finite `stacking_objective`
  and finite per-target `posterior_mean_*` columns.
- The `_manifest/` directory has at least two JSONs (one per stage).

If anything fails, see Troubleshooting below.

## Full launch (~ 1 to 2 days wall, fully unattended)

```
bash scripts/submit_paper_experiments.sh --all
tmux new -s watch
bash scripts/watch_progress.sh experiment_results/paper 60
```

This submits 15 Stage A jobs (gpu-preempt, array 0-99) and 15 Stage B jobs
(cpu-preempt, array 0-99) with `--dependency=afterany`, then prints a
per-cell live table every 60 seconds.

Monitoring rules of thumb:
- Healthy `last_code_age_s` is in the tens of seconds while Stage A is live.
- A cell stalled at the same `codes_distinct` count for more than 600 seconds
  while other cells progress means Ollama died on its node; resubmit the cell
  with `--only`.
- Once `codes_distinct == 10000`, Stage A for that cell will exit and Stage B
  will start (already queued).
- Stage B's `samples_ok` should grow steadily toward 10,000 per cell.

When all cells show `codes_distinct == 10000` and `samples_pending == 0`,
move to aggregation.

## Aggregation (~10 minutes)

```
uv run python -m scripts.aggregate_paper_results --root experiment_results/paper
```

Produces, under `experiment_results/paper/`:

- `summary.csv` (one row per task by llm by K)
- `generation_summary.csv` (per cell counts, timings, divergences)
- `test_scores.csv` (held-out predictive density per prediction cell)
- `aggregate_metrics.jsonl` (raw per-K rows from evaluate.py)
- `loo_matrices/` (per cell raw LOO log-lik matrices for bootstrap)
- `cumulative_weight_curves/` (per cell rank vs cumulative weight CSVs)
- `top_models/` (top 10 by BMA and by stacking weight, as `.code.py` files)

These power every paper table and figure.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Stage A writes zero codes for a cell, all `syntax_error` | LLM not following the goal-name instruction | Inspect a `_failures/failure_*.json`; usually a smaller LLM. Switch to a stronger one or revise prompts. |
| Stage A `last_code_age_s` keeps climbing for one cell only | Ollama died on its array task | `scancel <stage_A_jobid>`, then `submit_paper_experiments.sh --only task=X,llm=Y` |
| Stage B fails en masse with `inference_error` | A NumPyro generated model has a runtime bug | Open a few `sample_<sha>.meta.json` of status `inference_error`, read the traceback. Often safe to ignore one bad code among 10,000. |
| Stage B finishes but evaluate cannot find artifacts | `--root` mismatch or aggregator looking in the wrong tree | `scripts/aggregate_paper_results.py --root experiment_results/paper` |
| Cell came up short on distinct codes (e.g. 9,213 instead of 10,000) | Many slot indices produced duplicates | Submit `slurm/sample_cpu_topup.sbatch` for the partial Stage B (already idempotent). For Stage A, increase `SHARD_MAX_INDICES` and resubmit the cell. |
| `summary.csv` shows `n_loo_valid` much smaller than `n_bma_finite` | LOO failed for many models (often unstable variational fit) | Use a longer LOO warmup (`LOO_NUM_WARMUP=100`) and resubmit Stage B for that cell. |
| Cluster reports throttling or denied submissions | Concurrent task cap hit | Submit cells in waves with `--only` rather than `--all`. |

## Aborting

The submission log records every job id:

```
cat experiment_results/paper/_submissions/*.txt
```

Cancel a single cell:

```
scancel <stage_A_jobid> <stage_B_jobid>
```

Cancel everything from a launch:

```
awk -F, 'NR>1 {print $4 " " $5}' experiment_results/paper/_submissions/<timestamp>.txt | xargs scancel
```
