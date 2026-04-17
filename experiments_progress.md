# Stacking-paper experiment artifacts — what, where, progress

This doc is the concise index for the pre-generated NumPyro programs that
back the paper. It complements `stacking_experiments_overview.md` (narrative
+ pipeline design) by telling you where the bytes actually live and how
far along each (task, LLM) cell is.

The pipeline that produced these artifacts lives on the `model_gen` branch
(two-stage GPU + CPU sweep). The `anant` branch consumes them via a
simpler end-to-end script (`trial.py`) — see section "How anant consumes
this" below.

## Storage location

Everything is on **Unity scratch**, not in the repo:

```
/scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results/
  <task>/<llm>/
    codes/                  # Stage A output (GPU, Ollama)
      code_<sha16>.code.json
      _index.jsonl
      _failures/
    samples/                # Stage B output (CPU, NumPyro)
      sample_<sha16>.npz
      sample_<sha16>.meta.json
    _manifest/
      shard_<uuid>.json
  _submissions/
    <timestamp>.txt         # slurm job ids per sweep
```

Tasks: `hurricane_eal_counties`, `wildfire_eal_west`, `inland_flood_eal`,
`tornado_counts_plains`, `earthquake_frequency_west`.
LLMs: `qwen25_coder`, `gemma4_e4b`, `llama32`.

## File schemas

### `codes/code_<sha16>.code.json`
| key | meaning |
|---|---|
| `sha` | first 16 hex of sha256 over the AST-canonical code |
| `raw_code` | code as extracted from the LLM response (with imports fixed) |
| `canonical_code` | `ast.unparse(ast.parse(raw_code))` — **use this field** |
| `raw_llm_response` | verbatim model output (incl. THOUGHT + fenced block) |
| `prompt_messages` | full chat messages sent to the LLM |
| `first_seed` / `first_slot` | RNG seed and shard slot that first produced the code |
| `generation_seconds` | wall-clock for the LLM call |
| `generation_diagnostics` | `n_attempts` + per-attempt status/reason |

### `samples/sample_<sha16>.npz`
- posterior draws per target site
- `loo_log_liks` — shape `(n_train,)`, per-held-out-point log predictive
- `test_log_liks` — shape `(n_test,)`, only on prediction tasks
- `log_marginal_bound` — scalar IWAE bound

### `samples/sample_<sha16>.meta.json`
- `status` (`ok` / `inference_error` / …), `reason`
- stage timings
- `mcmc.diagnostics` — divergences, per-site r-hat, n_eff

## Progress (snapshot 2026-04-17)

| Task | LLM | codes | samples (ok) |
|---|---|---:|---:|
| hurricane_eal_counties | qwen25_coder | 10,000 | 7,910 |
| hurricane_eal_counties | gemma4_e4b | 9,864 | 0 |
| hurricane_eal_counties | llama32 | 9,945 | 0 |
| tornado_counts_plains | qwen25_coder | 6,347 | 0 |
| wildfire_eal_west × any | — | 0 | 0 |
| inland_flood_eal × any | — | 0 | 0 |
| earthquake_frequency_west × any | — | 0 | 0 |
| tornado_counts_plains × gemma4_e4b / llama32 | — | 0 | 0 |

Stage A is effectively done for hurricane across all three LLMs and
partially done for tornado×qwen25_coder. The rest of the 15-cell grid has
not been scheduled yet.

## How `anant/trial.py` consumes this

`anant`'s pipeline ignores `samples/` entirely and recomputes NUTS + IWAE
+ LOO on the loaded codes (single source of truth for weights).

```python
for path in sorted(Path(codes_dir).glob("code_*.code.json")):
    with open(path) as f:
        record = json.load(f)
    code_strings.append(record["canonical_code"])
# hand code_strings to llb.infer as preloaded model_codes
```

See `trial.py --preload-codes-dir <codes_dir>` on `anant` for the full
driver; no LLM calls are made on that path.

## Where more cells come from

On `model_gen`:

- `scripts/submit_paper_experiments.sh --all` launches the full 15-cell sweep.
- `scripts/watch_progress.sh` prints a live table of per-cell counts.
- `scripts/aggregate_paper_results.py` walks scratch and produces `summary.csv`
  etc. for the paper — but `anant`'s `trial.py` is the authoritative source
  for the paper's downstream numbers.
