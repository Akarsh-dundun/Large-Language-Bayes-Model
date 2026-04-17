"""Driver for the LLB stacking experiments.

This script replaces the old hard-coded coin-flip trial with a task- and
LLM-parameterised runner that can either (a) hit a live LLM via Ollama or
(b) consume pre-generated NumPyro programs produced on the ``model_gen``
branch (Stage A of the two-stage paper pipeline). See
``experiments_progress.md`` for where those artifacts live.

Examples
--------

Preload mode (no LLM calls, uses codes already on scratch)::

    python trial.py --task tasks/tornado_counts_plains.json \\
                    --llm-config llm_configs/qwen25_coder.json \\
                    --n-models 5 \\
                    --preload-codes-dir /scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results/tornado_counts_plains/qwen25_coder/codes

Live LLM mode (original behaviour)::

    python trial.py --task tasks/coin_flip.json \\
                    --llm-config llm_configs/qwen25_coder.json \\
                    --n-models 10

Sweep every (task, LLM) cell that has codes on scratch::

    LLB_PAPER_RESULTS_ROOT=/scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results \\
    python trial.py --sweep-paper --n-models 100
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import llb


DEFAULT_SCRATCH_ROOT = "/scratch3/workspace/edmondcunnin_umass_edu-siple/paper_results"
RESULTS_DIR = Path("experiment_results_anant")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--task", type=Path, help="Path to a task JSON (tasks/*.json)")
    p.add_argument("--llm-config", type=Path, help="Path to an LLM config (llm_configs/*.json)")
    p.add_argument(
        "--n-models",
        type=str,
        default="10,20",
        help="Comma-separated model-count sweep (e.g. '10,50,200').",
    )
    p.add_argument(
        "--preload-codes-dir",
        type=Path,
        default=None,
        help="If set, load pre-generated codes from this directory instead of calling the LLM.",
    )
    p.add_argument(
        "--paper-results-root",
        type=Path,
        default=Path(os.environ.get("LLB_PAPER_RESULTS_ROOT", DEFAULT_SCRATCH_ROOT)),
        help=(
            "Root of the paper_results tree on scratch. Used to auto-derive "
            "--preload-codes-dir (<root>/<task_stem>/<llm_name>/codes) when not given, "
            "and to enumerate cells in --sweep-paper."
        ),
    )
    p.add_argument(
        "--sweep-paper",
        action="store_true",
        help=(
            "Iterate every (task, llm) cell under --paper-results-root that has "
            "at least one code_*.code.json file. Requires --paper-results-root."
        ),
    )
    p.add_argument("--mcmc-warmup", type=int, default=500)
    p.add_argument("--mcmc-samples", type=int, default=1000)
    p.add_argument("--loo-warmup", type=int, default=50)
    p.add_argument("--loo-samples", type=int, default=100)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def load_task(path: Path) -> dict:
    with open(path) as f:
        task = json.load(f)
    missing = [k for k in ("text", "data", "targets") if k not in task]
    if missing:
        raise ValueError(f"task {path} missing required keys: {missing}")
    return task


def load_llm_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def serialize_result(result):
    """Convert numpy arrays to JSON-safe python types."""
    def _convert(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, dict):
            return {k: _convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_convert(x) for x in v]
        return v

    return {k: _convert(v) for k, v in result.items()}


def extract_metrics(result, target, llm_name, n_models_req, elapsed):
    diag = result["diagnostics"]
    w_uni = np.asarray(result["weights_uniform"])
    w_bma = np.asarray(result["weights_bma"])
    w_loo = np.asarray(result["weights_loo"])

    def _entropy(w):
        return float(-np.sum(w * np.log(w + 1e-10)))

    def _ess(w):
        return float(1.0 / np.sum(w ** 2))

    return {
        "llm_name": llm_name,
        "n_models_requested": int(n_models_req),
        "elapsed_time_seconds": float(elapsed),
        "n_models_generated": int(diag.get("generated_models", 0)),
        "n_models_deduplicated": int(diag.get("deduplicated_models", 0)),
        "n_models_invalid_syntax": int(diag.get("invalid_models_syntax_or_parsing", 0)),
        "n_models_generation_failures": int(diag.get("generation_request_failures", 0)),
        "n_models_missing_targets": int(diag.get("missing_targets_failures", 0)),
        "n_models_compile_failures": int(diag.get("compile_failures", 0)),
        "n_models_inference_failures": int(diag.get("inference_failures", 0)),
        "n_models_shape_mismatch": int(diag.get("shape_mismatch_drops", 0)),
        "n_models_nonfinite_log_bound": int(diag.get("nonfinite_log_bound_drops", 0)),
        "n_models_valid_final": int(diag.get("valid_models_final", 0)),
        "valid_model_rate": diag.get("valid_models_final", 0) / max(1, n_models_req),
        "epistemic_var_uniform": float(result["epistemic_uncertainty_uniform"][target]),
        "epistemic_var_bma": float(result["epistemic_uncertainty_bma"][target]),
        "epistemic_var_loo": float(result["epistemic_uncertainty_loo"][target]),
        "weights_uniform": w_uni.tolist(),
        "weights_bma": w_bma.tolist(),
        "weights_loo": w_loo.tolist(),
        "entropy_uniform": _entropy(w_uni),
        "entropy_bma": _entropy(w_bma),
        "entropy_loo": _entropy(w_loo),
        "ess_uniform": _ess(w_uni),
        "ess_bma": _ess(w_bma),
        "ess_loo": _ess(w_loo),
        "l1_distance_loo_bma": float(np.sum(np.abs(w_loo - w_bma))),
        "posterior_mean_uniform": float(np.mean(result["posterior_flat"][target])),
        "posterior_mean_bma": float(np.mean(result["posterior_bma"][target])),
        "posterior_mean_loo": float(np.mean(result["posterior_loo"][target])),
        "final_loo_objective": float(result["final_loo_objective"]),
        "log_marginal_per_model": result["log_marginal_per_model"],
    }


def run_one(task: dict, task_name: str, llm_cfg: dict, n_models: int, args, outer_bar=None) -> dict:
    primary_target = task["targets"][0] if task["targets"] else None

    preload_dir = args.preload_codes_dir
    if preload_dir is None and not args.sweep_paper:
        candidate = args.paper_results_root / task_name / llm_cfg["name"] / "codes"
        if candidate.is_dir() and any(candidate.glob("code_*.code.json")):
            preload_dir = candidate

    label = f"{task_name}/{llm_cfg['name']} n={n_models} [{'preload' if preload_dir else 'live-llm'}]"
    if outer_bar is not None:
        outer_bar.set_description(label)
        outer_bar.write(f"\n=== {label} ===" + (f" preload={preload_dir}" if preload_dir else ""))
    else:
        print("\n" + "=" * 80)
        print(f"Running: {label}")
        if preload_dir:
            print(f"  preload: {preload_dir}")
        print("=" * 80)

    start = time.time()
    try:
        result = llb.infer(
            task["text"],
            task["data"],
            task["targets"],
            api_url=llm_cfg.get("api_url"),
            api_key=llm_cfg.get("api_key"),
            api_model=llm_cfg.get("api_model"),
            n_models=n_models,
            verbose=args.verbose,
            mcmc_num_warmup=args.mcmc_warmup,
            mcmc_num_samples=args.mcmc_samples,
            loo_num_warmup=args.loo_warmup,
            loo_num_samples=args.loo_samples,
            use_true_loo=True,
            preloaded_codes_dir=str(preload_dir) if preload_dir else None,
        )
        elapsed = time.time() - start
        metrics = extract_metrics(result, primary_target, llm_cfg["name"], n_models, elapsed)
        return {
            "success": True,
            "task": task_name,
            "llm_name": llm_cfg["name"],
            "n_models": n_models,
            "preload_codes_dir": str(preload_dir) if preload_dir else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "full_result": serialize_result(result),
            "model_codes": result.get("model_codes", []),
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n[FAIL] {task_name}/{llm_cfg['name']} n={n_models}: {e}")
        return {
            "success": False,
            "task": task_name,
            "llm_name": llm_cfg["name"],
            "n_models": n_models,
            "preload_codes_dir": str(preload_dir) if preload_dir else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "error": str(e),
            "elapsed_time_seconds": elapsed,
        }


def discover_sweep_cells(root: Path):
    cells = []
    if not root.is_dir():
        return cells
    task_dir = Path("tasks")
    llm_dir = Path("llm_configs")
    task_map = {p.stem: p for p in task_dir.glob("*.json")}
    llm_map = {p.stem: p for p in llm_dir.glob("*.json")}
    for task_name in sorted(task_map):
        for llm_name in sorted(llm_map):
            codes = root / task_name / llm_name / "codes"
            if codes.is_dir() and any(codes.glob("code_*.code.json")):
                cells.append((task_name, task_map[task_name], llm_name, llm_map[llm_name], codes))
    return cells


def main():
    args = parse_args()
    n_models_list = [int(n) for n in str(args.n_models).split(",") if n.strip()]
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    jobs = []  # list of (task_dict, task_name, llm_cfg)
    if args.sweep_paper:
        cells = discover_sweep_cells(args.paper_results_root)
        if not cells:
            raise SystemExit(f"No populated cells under {args.paper_results_root}")
        print(f"Sweep-paper: {len(cells)} (task, llm) cells")
        for task_name, task_path, llm_name, llm_path, codes_dir in cells:
            task = load_task(task_path)
            llm_cfg = load_llm_config(llm_path)
            jobs.append((task, task_name, llm_cfg))
    else:
        if not args.task or not args.llm_config:
            raise SystemExit("Provide --task and --llm-config, or use --sweep-paper")
        task = load_task(args.task)
        task_name = args.task.stem
        llm_cfg = load_llm_config(args.llm_config)
        jobs.append((task, task_name, llm_cfg))

    all_results = []
    total_runs = len(jobs) * len(n_models_list)
    outer_bar = tqdm(
        total=total_runs,
        desc="runs",
        unit="run",
        dynamic_ncols=True,
        position=0,
    )
    for task, task_name, llm_cfg in jobs:
        for n_models in n_models_list:
            result = run_one(task, task_name, llm_cfg, n_models, args, outer_bar=outer_bar)
            all_results.append(result)
            out_path = RESULTS_DIR / f"results_{task_name}_{llm_cfg['name']}_n{n_models}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            outer_bar.write(f"  saved: {out_path}")
            outer_bar.update(1)
    outer_bar.close()

    summary_path = RESULTS_DIR / "all_results.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll done. Combined: {summary_path}")

    # Text summary
    print("\n" + "=" * 120)
    print(f"{'task':<30} {'llm':<14} {'n':<5} {'valid':<6} {'Epi_Uni':<10} {'Epi_BMA':<10} {'Epi_LOO':<10} {'L1':<7} {'t(s)':<7}")
    print("-" * 120)
    for r in all_results:
        if r["success"]:
            m = r["metrics"]
            print(f"{r['task']:<30} {r['llm_name']:<14} {m['n_models_requested']:<5} "
                  f"{m['n_models_valid_final']:<6} "
                  f"{m['epistemic_var_uniform']:<10.4g} {m['epistemic_var_bma']:<10.4g} "
                  f"{m['epistemic_var_loo']:<10.4g} {m['l1_distance_loo_bma']:<7.3f} "
                  f"{m['elapsed_time_seconds']:<7.1f}")
        else:
            print(f"{r['task']:<30} {r['llm_name']:<14} {r['n_models']:<5} FAILED: {r['error']}")


if __name__ == "__main__":
    main()
