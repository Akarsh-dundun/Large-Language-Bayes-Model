"""Per-model LLM generation and PPL sampling driver.

For each global model index in a shard, this script:
  1. Calls the LLM once (with bounded retries) to get a NumPyro model code string.
  2. Runs NUTS to produce posterior samples.
  3. Estimates the IWAE log marginal likelihood bound.
  4. Estimates the leave-one-out log predictive density vector.
  5. Writes one .npz sample artifact and one .meta.json status artifact atomically.

Artifacts are written per model so the job is resumable under Slurm preemption and
so downstream evaluation can slice any prefix of the model pool without re-running
inference. A model is considered done when its sibling .meta.json exists with a
populated ``status`` field.
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llb.llm import LLMClient
from llb.mcmc_log import (
  estimate_log_marginal_iw,
  estimate_loo_log_likelihoods,
  run_inference,
)
from llb.model_generator import generate_models_with_diagnostics


def _load_json(path):
  with open(path) as f:
    return json.load(f)


def _atomic_write_bytes(path, payload):
  """Write payload bytes to path via a tmp file and os.replace."""
  path = Path(path)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "wb") as f:
    f.write(payload)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def _atomic_write_text(path, text):
  _atomic_write_bytes(path, text.encode("utf-8"))


def _atomic_savez(path, **arrays):
  """np.savez_compressed with atomic replace.

  np.savez_compressed appends ``.npz`` to string/path arguments, so we hand it
  an open file handle to keep the temp name exactly as written.
  """
  path = Path(path)
  tmp = path.with_suffix(path.suffix + ".tmp")
  with open(tmp, "wb") as f:
    np.savez_compressed(f, **arrays)
    f.flush()
    os.fsync(f.fileno())
  os.replace(tmp, path)


def _sanitize_site_key(prefix, name):
  """Keys in np.savez must survive round-trip, so escape any slashes."""
  safe = name.replace("/", "__").replace("\\", "__")
  return f"{prefix}__{safe}"


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
  save_full_posterior: bool = False


def _generate_one_code(llm, text, data, targets, global_slot, base_seed):
  """Generate a single candidate code string for the given slot index.

  Returns a tuple (code_or_none, diagnostics_dict). When generation fails for all
  attempts, ``code_or_none`` is None and the diagnostic includes a failure reason.
  """
  codes, diag = generate_models_with_diagnostics(
    llm=llm,
    text=text,
    data=data,
    targets=targets,
    n_models=1,
    base_seed=base_seed,
    slot_offset=global_slot,
  )
  if codes:
    return codes[0], diag
  return None, diag


def _save_failure_meta(meta_path, status, index, seed, reason, code=None, extra=None):
  meta = {
    "index": int(index),
    "seed": int(seed),
    "status": status,
    "reason": reason,
    "code": code,
  }
  if extra:
    meta.update(extra)
  _atomic_write_text(meta_path, json.dumps(meta, indent=2))


def _serialize_loo_diagnostics(diag):
  """Strip non-JSON-safe entries from the LOO diagnostics dict."""
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


def process_index(
  index,
  task,
  llm,
  output_dir,
  base_seed,
  cfg,
  force=False,
):
  """Run the full generation+PPL pipeline for one global model index.

  Writes ``model_{index:06d}.npz`` and ``model_{index:06d}.meta.json`` atomically.
  Returns the status string written.
  """
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  stem = f"model_{index:06d}"
  meta_path = output_dir / f"{stem}.meta.json"
  npz_path = output_dir / f"{stem}.npz"
  seed = base_seed + index

  if meta_path.exists() and not force:
    existing = _load_json(meta_path)
    if existing.get("status"):
      return existing["status"]

  text = task["text"]
  data = task["data"]
  targets = task.get("targets")

  t0 = time.time()

  code, gen_diag = _generate_one_code(
    llm=llm,
    text=text,
    data=data,
    targets=targets,
    global_slot=index,
    base_seed=base_seed,
  )
  gen_time = time.time() - t0

  if code is None:
    failures = gen_diag.get("generation_failures", [])
    reason = failures[0][1] if failures else "generation_request_error: unknown"
    status = "syntax_error" if reason.startswith("parsing_error:") else "generation_failed"
    _save_failure_meta(
      meta_path,
      status=status,
      index=index,
      seed=seed,
      reason=reason,
      code=None,
      extra={"timings": {"generation_seconds": gen_time}},
    )
    return status

  t1 = time.time()
  try:
    infer_out = run_inference(
      code=code,
      data=data,
      targets=targets,
      num_warmup=cfg.mcmc_num_warmup,
      num_samples=cfg.mcmc_num_samples,
      rng_seed=seed,
    )
  except Exception as exc:
    msg = str(exc)
    status = "compile_error" if msg.startswith("compile_error:") else "inference_error"
    _save_failure_meta(
      meta_path,
      status=status,
      index=index,
      seed=seed,
      reason=msg,
      code=code,
      extra={
        "timings": {
          "generation_seconds": gen_time,
          "mcmc_seconds": time.time() - t1,
        },
      },
    )
    return status
  mcmc_time = time.time() - t1

  missing = infer_out["missing_targets"] if targets is not None else []
  if missing:
    _save_failure_meta(
      meta_path,
      status="missing_targets",
      index=index,
      seed=seed,
      reason=f"missing targets: {', '.join(missing)}",
      code=code,
      extra={
        "available_sites": infer_out["available_sites"],
        "timings": {
          "generation_seconds": gen_time,
          "mcmc_seconds": mcmc_time,
        },
      },
    )
    return "missing_targets"

  model = infer_out["model"]
  samples = infer_out["samples"]
  target_samples = infer_out["target_samples"]

  t2 = time.time()
  marginal_status = "ok"
  marginal_reason = None
  try:
    log_bound = float(
      estimate_log_marginal_iw(
        model=model,
        data=data,
        posterior_samples=samples,
        num_inner=cfg.log_marginal_num_inner,
        num_outer=cfg.log_marginal_num_outer,
        rng_seed=seed + 10_000,
      )
    )
  except Exception as exc:
    log_bound = float("nan")
    marginal_status = "failed"
    marginal_reason = f"{type(exc).__name__}: {exc}"
  marginal_time = time.time() - t2

  t3 = time.time()
  loo_status = "ok"
  loo_reason = None
  loo_diag = None
  try:
    loo_result = estimate_loo_log_likelihoods(
      model=model,
      data=data,
      posterior_samples=samples,
      num_inner=cfg.loo_num_inner,
      num_warmup=cfg.loo_num_warmup,
      num_samples=cfg.loo_num_samples,
      rng_seed=seed + 20_000,
      use_true_loo=cfg.use_true_loo,
      return_diagnostics=True,
    )
    loo_log_liks = np.asarray(loo_result["loo_log_liks"], dtype=np.float64)
    loo_diag = _serialize_loo_diagnostics(loo_result.get("diagnostics"))
  except Exception as exc:
    loo_log_liks = None
    loo_status = "failed"
    loo_reason = f"{type(exc).__name__}: {exc}"
  loo_time = time.time() - t3

  arrays = {}
  for name, arr in target_samples.items():
    key = _sanitize_site_key("target", name)
    arrays[key] = np.asarray(arr, dtype=np.float64)

  if cfg.save_full_posterior:
    for site, arr in samples.items():
      arrays[_sanitize_site_key("post", site)] = np.asarray(arr)

  arrays["log_marginal_bound"] = np.asarray(log_bound, dtype=np.float64)
  if loo_log_liks is not None:
    arrays["loo_log_liks"] = loo_log_liks.astype(np.float64)

  _atomic_savez(npz_path, **arrays)

  if loo_status != "ok" and marginal_status != "ok":
    status = "inference_partial"
  elif loo_status != "ok" or marginal_status != "ok":
    status = "partial"
  else:
    status = "ok"

  meta = {
    "index": int(index),
    "seed": int(seed),
    "status": status,
    "reason": None,
    "code": code,
    "targets": list(target_samples.keys()),
    "available_sites": infer_out["available_sites"],
    "n_datapoints": int(loo_log_liks.shape[0]) if loo_log_liks is not None else None,
    "log_marginal": {
      "status": marginal_status,
      "reason": marginal_reason,
      "value": log_bound if np.isfinite(log_bound) else None,
    },
    "loo": {
      "status": loo_status,
      "reason": loo_reason,
      "use_true_loo": bool(cfg.use_true_loo),
      "diagnostics": loo_diag,
    },
    "timings": {
      "generation_seconds": gen_time,
      "mcmc_seconds": mcmc_time,
      "log_marginal_seconds": marginal_time,
      "loo_seconds": loo_time,
    },
    "mcmc": {
      "num_warmup": int(cfg.mcmc_num_warmup),
      "num_samples": int(cfg.mcmc_num_samples),
    },
  }
  _atomic_write_text(meta_path, json.dumps(meta, indent=2))
  return status


def build_llm(llm_config_dict):
  kwargs = {
    "api_url": llm_config_dict["api_url"],
    "api_key": llm_config_dict.get("api_key"),
    "model": llm_config_dict["api_model"],
    "max_retries": int(llm_config_dict.get("llm_max_retries", 2)),
    "retry_backoff": float(llm_config_dict.get("llm_retry_backoff", 2.0)),
  }
  timeout = llm_config_dict.get("llm_timeout")
  if timeout is not None:
    kwargs["timeout"] = int(timeout)
  if "temperature" in llm_config_dict:
    kwargs["temperature"] = float(llm_config_dict["temperature"])
  return LLMClient(**kwargs)


def run_shard(task_path, llm_config_path, output_dir, shard_start, shard_count, seed, cfg, force=False, api_url_override=None):
  task = _load_json(task_path)
  llm_config = _load_json(llm_config_path)
  if api_url_override:
    llm_config["api_url"] = api_url_override
  llm = build_llm(llm_config)

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  counts = {}
  for offset in range(shard_count):
    idx = shard_start + offset
    try:
      status = process_index(
        index=idx,
        task=task,
        llm=llm,
        output_dir=output_dir,
        base_seed=seed,
        cfg=cfg,
        force=force,
      )
    except Exception:
      traceback.print_exc()
      status = "unhandled_exception"
      try:
        meta_path = output_dir / f"model_{idx:06d}.meta.json"
        _save_failure_meta(
          meta_path,
          status=status,
          index=idx,
          seed=seed + idx,
          reason=traceback.format_exc(),
        )
      except Exception:
        pass
    counts[status] = counts.get(status, 0) + 1
    print(f"[shard] idx={idx} status={status}", flush=True)

  print(f"[shard] complete counts={counts}", flush=True)
  return counts


def _build_parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", required=True, help="Path to a task spec JSON")
  p.add_argument("--llm-config", required=True, help="Path to an LLM config JSON")
  p.add_argument("--output-dir", required=True, help="Directory for per-model artifacts")
  p.add_argument("--shard-start", type=int, required=True, help="First global model index for this task")
  p.add_argument("--shard-count", type=int, required=True, help="Number of model indices to process")
  p.add_argument("--seed", type=int, default=42, help="Base seed; per-model seed = seed + index")
  p.add_argument("--api-url", default=None, help="Override the LLM config api_url (for per-task Ollama ports)")
  p.add_argument("--force", action="store_true", help="Reprocess even if an artifact already exists")

  p.add_argument("--mcmc-num-warmup", type=int, default=500)
  p.add_argument("--mcmc-num-samples", type=int, default=1000)
  p.add_argument("--log-marginal-num-inner", type=int, default=5)
  p.add_argument("--log-marginal-num-outer", type=int, default=80)
  p.add_argument("--loo-num-inner", type=int, default=25)
  p.add_argument("--loo-num-warmup", type=int, default=50)
  p.add_argument("--loo-num-samples", type=int, default=100)
  p.add_argument("--use-true-loo", dest="use_true_loo", action="store_true", default=True)
  p.add_argument("--no-true-loo", dest="use_true_loo", action="store_false")
  p.add_argument("--save-full-posterior", action="store_true", help="Also persist every MCMC site, not just targets")
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
  )
  run_shard(
    task_path=args.task,
    llm_config_path=args.llm_config,
    output_dir=args.output_dir,
    shard_start=args.shard_start,
    shard_count=args.shard_count,
    seed=args.seed,
    cfg=cfg,
    force=args.force,
    api_url_override=args.api_url,
  )


if __name__ == "__main__":
  sys.exit(main())
