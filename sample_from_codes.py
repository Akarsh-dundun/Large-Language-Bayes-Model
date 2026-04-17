"""Stage B: posterior inference and scoring for codes produced by Stage A.

For each ``codes/code_<sha>.code.json`` in this cell, runs NUTS, the IWAE
log marginal bound, true leave-one-out, and (for prediction tasks) held-out
test predictive density, then writes ``samples/sample_<sha>.npz`` and
``samples/sample_<sha>.meta.json`` next door.

This script is CPU-only by design. It does not require a GPU and does not
talk to the LLM. Re-running on the same cell skips any hash whose
sample meta.json already exists with status ok.

Usage:

  uv run python -m sample_from_codes \\
    --task tasks/hurricane_eal_counties.json \\
    --cell-dir experiment_results/paper/hurricane_eal_counties/qwen25_coder \\
    --array-task-id 0 --array-task-count 100 \\
    --mcmc-num-warmup 500 --mcmc-num-samples 1000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llb.codes import (
  CodePayload,
  list_code_hashes,
  load_code_payload,
  sample_meta_path,
  sample_npz_path,
  samples_dir,
)
from llb.io_utils import (
  atomic_savez,
  atomic_write_json,
  sanitize_site_key,
)
from llb.mcmc_log import (
  estimate_log_marginal_iw,
  estimate_loo_log_likelihoods,
  estimate_loo_log_likelihoods_parallel,
  estimate_test_log_likelihoods,
  run_inference,
)
from llb.run_manifest import write_run_manifest


@dataclass
class SampleConfig:
  mcmc_num_warmup: int = 500
  mcmc_num_samples: int = 1000
  log_marginal_num_inner: int = 5
  log_marginal_num_outer: int = 80
  loo_num_inner: int = 25
  loo_num_warmup: int = 50
  loo_num_samples: int = 100
  use_true_loo: bool = True
  save_full_posterior: bool = True
  loo_workers: int = 1


def _load_json(path):
  with open(path) as f:
    return json.load(f)


def process_hash(sha, cell_dir, task, cfg, base_seed=42, verbose_loo=False):
  """Run inference for one code hash. Idempotent: skip if meta exists with ok status."""
  meta_path = sample_meta_path(cell_dir, sha)
  npz_path = sample_npz_path(cell_dir, sha)

  if meta_path.exists():
    try:
      with open(meta_path) as f:
        existing = json.load(f)
      if existing.get("status") == "ok":
        return "skip_ok"
    except Exception:
      pass

  payload = CodePayload.from_dict(load_code_payload(cell_dir, sha))
  data = task["data"]
  test_data = task.get("test_data")
  targets = task.get("targets")

  # Per-hash seed derived from sha so re-runs are reproducible.
  rng_seed = base_seed + (int(sha[:8], 16) % 1_000_000)

  t0 = time.time()
  try:
    infer_out = run_inference(
      code=payload.canonical_code,
      data=data,
      targets=targets,
      num_warmup=cfg.mcmc_num_warmup,
      num_samples=cfg.mcmc_num_samples,
      rng_seed=rng_seed,
    )
  except Exception as exc:
    msg = str(exc)
    status = "compile_error" if msg.startswith("compile_error:") else "inference_error"
    _save_failure(meta_path, sha, status, msg, base_seed=rng_seed,
                  timings={"inference_seconds": time.time() - t0})
    return status
  mcmc_time = time.time() - t0

  missing = infer_out.get("missing_targets") or []
  if missing:
    _save_failure(
      meta_path, sha, "missing_targets",
      reason=f"missing targets: {', '.join(missing)}",
      base_seed=rng_seed,
      timings={"inference_seconds": mcmc_time},
      extra={"available_sites": infer_out["available_sites"]},
    )
    return "missing_targets"

  model = infer_out["model"]
  samples = infer_out["samples"]
  target_samples = infer_out["target_samples"]
  mcmc_diagnostics = infer_out.get("mcmc_diagnostics", {})

  t1 = time.time()
  marginal_status = "ok"
  marginal_reason = None
  try:
    log_bound = float(estimate_log_marginal_iw(
      model=model, data=data, posterior_samples=samples,
      num_inner=cfg.log_marginal_num_inner,
      num_outer=cfg.log_marginal_num_outer,
      rng_seed=rng_seed + 10_000,
    ))
  except Exception as exc:
    log_bound = float("nan")
    marginal_status = "failed"
    marginal_reason = f"{type(exc).__name__}: {exc}"
  marginal_time = time.time() - t1

  t2 = time.time()
  loo_status = "ok"
  loo_reason = None
  loo_diag = None
  try:
    if cfg.use_true_loo and cfg.loo_workers and cfg.loo_workers > 1:
      loo_result = estimate_loo_log_likelihoods_parallel(
        code=payload.canonical_code,
        data=data,
        posterior_samples=samples,
        num_inner=cfg.loo_num_inner,
        num_warmup=cfg.loo_num_warmup,
        num_samples=cfg.loo_num_samples,
        rng_seed=rng_seed + 20_000,
        n_workers=cfg.loo_workers,
        return_diagnostics=True,
      )
    else:
      loo_result = estimate_loo_log_likelihoods(
        model=model, data=data, posterior_samples=samples,
        num_inner=cfg.loo_num_inner,
        num_warmup=cfg.loo_num_warmup,
        num_samples=cfg.loo_num_samples,
        rng_seed=rng_seed + 20_000,
        use_true_loo=cfg.use_true_loo,
        return_diagnostics=True,
        verbose=verbose_loo,
      )
    loo_log_liks = np.asarray(loo_result["loo_log_liks"], dtype=np.float64)
    loo_diag = _serialize_loo_diagnostics(loo_result.get("diagnostics"))
  except Exception as exc:
    loo_log_liks = None
    loo_status = "failed"
    loo_reason = f"{type(exc).__name__}: {exc}"
  loo_time = time.time() - t2

  t3 = time.time()
  test_status = "skipped"
  test_reason = None
  test_log_liks = None
  test_diag = None
  if test_data is not None:
    try:
      test_log_liks, test_diag = estimate_test_log_likelihoods(
        model=model, train_data=data, test_data=test_data,
        posterior_samples=samples, rng_seed=rng_seed + 30_000,
      )
      test_log_liks = np.asarray(test_log_liks, dtype=np.float64)
      test_status = "ok"
    except Exception as exc:
      test_status = "failed"
      test_reason = f"{type(exc).__name__}: {exc}"
  test_time = time.time() - t3

  arrays = {}
  for name, arr in target_samples.items():
    arrays[sanitize_site_key("target", name)] = np.asarray(arr, dtype=np.float64)
  if cfg.save_full_posterior:
    for site, arr in samples.items():
      arrays[sanitize_site_key("post", site)] = np.asarray(arr)
  arrays["log_marginal_bound"] = np.asarray(log_bound, dtype=np.float64)
  if loo_log_liks is not None:
    arrays["loo_log_liks"] = loo_log_liks.astype(np.float64)
  if test_log_liks is not None:
    arrays["test_log_liks"] = test_log_liks.astype(np.float64)

  atomic_savez(npz_path, **arrays)

  status_map = {("ok", "ok"): "ok"}
  base_status = "ok"
  if loo_status != "ok" and marginal_status != "ok":
    base_status = "inference_partial"
  elif loo_status != "ok" or marginal_status != "ok":
    base_status = "partial"

  meta = {
    "sha": sha,
    "status": base_status,
    "reason": None,
    "rng_seed": int(rng_seed),
    "n_train": int(loo_log_liks.shape[0]) if loo_log_liks is not None else None,
    "n_test": int(test_log_liks.shape[0]) if test_log_liks is not None else (0 if test_data is None else None),
    "targets": list(target_samples.keys()),
    "available_sites": infer_out["available_sites"],
    "log_marginal": {
      "status": marginal_status, "reason": marginal_reason,
      "value": log_bound if np.isfinite(log_bound) else None,
    },
    "loo": {
      "status": loo_status, "reason": loo_reason,
      "use_true_loo": bool(cfg.use_true_loo),
      "diagnostics": loo_diag,
    },
    "test_scoring": {
      "status": test_status, "reason": test_reason,
      "diagnostics": test_diag,
    },
    "mcmc": {
      "num_warmup": int(cfg.mcmc_num_warmup),
      "num_samples": int(cfg.mcmc_num_samples),
      "diagnostics": mcmc_diagnostics,
    },
    "timings": {
      "mcmc_seconds": mcmc_time,
      "log_marginal_seconds": marginal_time,
      "loo_seconds": loo_time,
      "test_scoring_seconds": test_time,
    },
  }
  atomic_write_json(meta_path, meta)
  return base_status


def _save_failure(meta_path, sha, status, reason, base_seed, timings=None, extra=None):
  meta = {
    "sha": sha,
    "status": status,
    "reason": reason,
    "rng_seed": int(base_seed),
    "timings": timings or {},
  }
  if extra:
    meta.update(extra)
  atomic_write_json(meta_path, meta)


def _serialize_loo_diagnostics(diag):
  if not isinstance(diag, dict):
    return None
  out = {}
  for k, v in diag.items():
    if k == "elbo_histories":
      out[k] = [list(h) for h in v]
    elif isinstance(v, (int, float, str, bool)) or v is None:
      out[k] = v
    elif isinstance(v, (list, tuple)):
      out[k] = list(v)
    else:
      out[k] = str(v)
  return out


def run_shard(
  task_path,
  cell_dir,
  array_task_id,
  array_task_count,
  cfg,
  seed=42,
  verbose_loo=False,
  cli_args=None,
):
  cell_dir = Path(cell_dir)
  samples_dir(cell_dir).mkdir(parents=True, exist_ok=True)

  task = _load_json(task_path)
  hashes = list_code_hashes(cell_dir)
  if array_task_count > 1:
    my_hashes = hashes[array_task_id::array_task_count]
  else:
    my_hashes = hashes

  write_run_manifest(
    cell_dir=cell_dir,
    stage="sample_from_codes",
    cli_args=cli_args or {},
    extras={
      "task_path": str(task_path),
      "array_task_id": int(array_task_id),
      "array_task_count": int(array_task_count),
      "n_hashes_total": len(hashes),
      "n_hashes_this_shard": len(my_hashes),
      "seed": int(seed),
    },
  )

  shard_t0 = time.time()
  counts = {}
  for sha in my_hashes:
    try:
      status = process_hash(sha, cell_dir, task, cfg, base_seed=seed, verbose_loo=verbose_loo)
    except Exception:
      traceback.print_exc()
      status = "unhandled_exception"
      try:
        _save_failure(
          sample_meta_path(cell_dir, sha), sha, status,
          reason=traceback.format_exc(), base_seed=seed,
        )
      except Exception:
        pass
    counts[status] = counts.get(status, 0) + 1
    print(f"[sample_from_codes] sha={sha} status={status}", flush=True)

  shard_elapsed = time.time() - shard_t0
  print(
    f"[sample_from_codes] shard complete in {shard_elapsed:.1f}s; counts={counts}",
    flush=True,
  )
  return counts


def _build_parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", required=True)
  p.add_argument("--cell-dir", required=True)
  p.add_argument("--array-task-id", type=int, default=0)
  p.add_argument("--array-task-count", type=int, default=1)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--verbose-loo", action="store_true",
                 help="Print per-datapoint LOO progress (noisy across many models)")

  p.add_argument("--mcmc-num-warmup", type=int, default=500)
  p.add_argument("--mcmc-num-samples", type=int, default=1000)
  p.add_argument("--log-marginal-num-inner", type=int, default=5)
  p.add_argument("--log-marginal-num-outer", type=int, default=80)
  p.add_argument("--loo-num-inner", type=int, default=25)
  p.add_argument("--loo-num-warmup", type=int, default=50)
  p.add_argument("--loo-num-samples", type=int, default=100)
  p.add_argument("--use-true-loo", dest="use_true_loo", action="store_true", default=True)
  p.add_argument("--no-true-loo", dest="use_true_loo", action="store_false")
  p.add_argument("--save-full-posterior", dest="save_full_posterior",
                 action="store_true", default=True)
  p.add_argument("--no-save-full-posterior", dest="save_full_posterior",
                 action="store_false")
  p.add_argument("--loo-workers", type=int, default=1,
                 help="Process pool workers for true-LOO. 1 = serial in-process.")
  return p


def main(argv=None):
  args = _build_parser().parse_args(argv)
  cfg = SampleConfig(
    mcmc_num_warmup=args.mcmc_num_warmup,
    mcmc_num_samples=args.mcmc_num_samples,
    log_marginal_num_inner=args.log_marginal_num_inner,
    log_marginal_num_outer=args.log_marginal_num_outer,
    loo_num_inner=args.loo_num_inner,
    loo_num_warmup=args.loo_num_warmup,
    loo_num_samples=args.loo_num_samples,
    use_true_loo=bool(args.use_true_loo),
    save_full_posterior=bool(args.save_full_posterior),
    loo_workers=int(args.loo_workers),
  )
  return run_shard(
    task_path=args.task,
    cell_dir=args.cell_dir,
    array_task_id=args.array_task_id,
    array_task_count=args.array_task_count,
    cfg=cfg,
    seed=args.seed,
    verbose_loo=args.verbose_loo,
    cli_args=vars(args),
  )


if __name__ == "__main__":
  sys.exit(0 if main() is not None else 1)
