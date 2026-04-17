"""Stage A: per-slot LLM code generation with hash-based dedup.

For each slot index in this shard's range, calls the LLM (with internal
retries) to obtain a NumPyro program. Successful programs are
AST-canonicalized, hashed (sha256, first 16 hex chars), and atomically
claimed via O_EXCL on ``codes/code_<sha>.code.json``. Duplicates are
recorded in ``codes/_index.jsonl`` without overwriting the original.
Failures land in ``codes/_failures/`` for later inspection.

This script is GPU-bound: the only expensive thing it does is call Ollama.
It does not run NUTS, IWAE, or LOO. See ``sample_from_codes.py`` for
Stage B.

Usage:

  uv run python -m generate_codes \\
    --task tasks/hurricane_eal_counties.json \\
    --llm-config llm_configs/qwen25_coder.json \\
    --cell-dir experiment_results/paper/hurricane_eal_counties/qwen25_coder \\
    --shard-start 0 --shard-max-indices 150 \\
    --shard-valid-target 100 --cell-valid-target 10000 \\
    --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from llb.codes import (
  CODES_SUBDIR,
  append_code_index,
  canonicalize_code,
  claim_code,
  code_hash,
  code_path,
  codes_dir,
  count_distinct_codes,
  failures_dir,
  write_failure,
)
from llb.io_utils import atomic_write_json
from llb.llm import LLMClient
from llb.model_generator import generate_one_with_full_diagnostics
from llb.run_manifest import write_run_manifest


def _load_json(path):
  with open(path) as f:
    return json.load(f)


def build_llm(llm_config_dict, api_url_override=None):
  cfg = dict(llm_config_dict)
  if api_url_override:
    cfg["api_url"] = api_url_override
  kwargs = {
    "api_url": cfg["api_url"],
    "api_key": cfg.get("api_key"),
    "model": cfg["api_model"],
    "max_retries": int(cfg.get("llm_max_retries", 2)),
    "retry_backoff": float(cfg.get("llm_retry_backoff", 2.0)),
  }
  timeout = cfg.get("llm_timeout")
  if timeout is not None:
    kwargs["timeout"] = int(timeout)
  if "temperature" in cfg:
    kwargs["temperature"] = float(cfg["temperature"])
  return LLMClient(**kwargs)


def process_slot(slot, task, llm, cell_dir, base_seed):
  """Call LLM for one slot and either claim a new hash or record duplicate/failure.

  Returns one of: ``"new"``, ``"duplicate"``, ``"syntax_error"``,
  ``"generation_failed"``.
  """
  text = task["text"]
  data = task["data"]
  targets = task.get("targets")

  t0 = time.time()
  result = generate_one_with_full_diagnostics(
    llm=llm,
    text=text,
    data=data,
    targets=targets,
    slot=slot,
    base_seed=base_seed,
  )
  elapsed = time.time() - t0

  if result["code"] is None:
    record = {
      "slot": int(slot),
      "final_status": result["final_status"],
      "final_reason": result["final_reason"],
      "attempts": result["attempts"],
      "messages_used": result["messages_used"],
      "raw_llm_response": result["raw_llm_response"],
      "generation_seconds": elapsed,
      "base_seed": int(base_seed),
    }
    write_failure(cell_dir, slot, record)
    append_code_index(cell_dir, {
      "slot": int(slot),
      "sha": None,
      "status": result["final_status"],
      "reason": result["final_reason"],
      "generation_seconds": elapsed,
    })
    return result["final_status"]

  raw_code = result["code"]
  try:
    canonical = canonicalize_code(raw_code)
  except SyntaxError as exc:
    # extract_model_code may have produced something that does not parse
    # despite passing the regex-based goal check. Treat as syntax error.
    record = {
      "slot": int(slot),
      "final_status": "syntax_error",
      "final_reason": f"ast_parse_error: {exc}",
      "raw_code": raw_code,
      "raw_llm_response": result["raw_llm_response"],
      "messages_used": result["messages_used"],
      "attempts": result["attempts"],
      "generation_seconds": elapsed,
      "base_seed": int(base_seed),
    }
    write_failure(cell_dir, slot, record)
    append_code_index(cell_dir, {
      "slot": int(slot),
      "sha": None,
      "status": "syntax_error",
      "reason": f"ast_parse_error: {exc}",
      "generation_seconds": elapsed,
    })
    return "syntax_error"

  sha = code_hash(canonical)

  payload = {
    "sha": sha,
    "raw_code": raw_code,
    "canonical_code": canonical,
    "raw_llm_response": result["raw_llm_response"],
    "prompt_messages": result["messages_used"],
    "first_seed": int(base_seed),
    "first_slot": int(slot),
    "generation_seconds": elapsed,
    "generation_diagnostics": {
      "n_attempts": len(result["attempts"]),
      "attempts": result["attempts"],
    },
  }

  claimed = claim_code(cell_dir, sha, payload)
  append_code_index(cell_dir, {
    "slot": int(slot),
    "sha": sha,
    "status": "new" if claimed else "duplicate",
    "generation_seconds": elapsed,
    "n_attempts": len(result["attempts"]),
  })
  return "new" if claimed else "duplicate"


def run_shard(
  task_path,
  llm_config_path,
  cell_dir,
  shard_start,
  shard_max_indices,
  shard_valid_target,
  cell_valid_target,
  seed,
  api_url_override=None,
  poll_interval_slots=10,
  cli_args=None,
  llm=None,
):
  cell_dir = Path(cell_dir)
  codes_dir(cell_dir).mkdir(parents=True, exist_ok=True)
  failures_dir(cell_dir).mkdir(parents=True, exist_ok=True)

  task = _load_json(task_path)
  if llm is None:
    llm_config = _load_json(llm_config_path)
    llm = build_llm(llm_config, api_url_override=api_url_override)

  write_run_manifest(
    cell_dir=cell_dir,
    stage="generate_codes",
    cli_args=cli_args or {},
    extras={
      "task_path": str(task_path),
      "llm_config_path": str(llm_config_path),
      "shard_start": int(shard_start),
      "shard_max_indices": int(shard_max_indices),
      "shard_valid_target": int(shard_valid_target),
      "cell_valid_target": int(cell_valid_target),
      "seed": int(seed),
    },
  )

  shard_end = shard_start + shard_max_indices
  counts = {"new": 0, "duplicate": 0, "syntax_error": 0, "generation_failed": 0}
  shard_t0 = time.time()

  # Skip slots we already processed (their hash is in the index already).
  done_slots = _read_done_slots(cell_dir, shard_start, shard_end)

  for slot in range(shard_start, shard_end):
    if slot in done_slots:
      continue

    distinct = count_distinct_codes(cell_dir)
    if distinct >= cell_valid_target:
      print(
        f"[generate_codes] cell target reached: distinct={distinct} >= "
        f"{cell_valid_target}; stopping shard at slot {slot}",
        flush=True,
      )
      break

    status = process_slot(slot, task, llm, cell_dir, base_seed=seed)
    counts[status] = counts.get(status, 0) + 1

    if status == "new" and counts["new"] >= shard_valid_target:
      print(
        f"[generate_codes] shard valid target reached: new={counts['new']} >= "
        f"{shard_valid_target}; stopping shard at slot {slot+1}",
        flush=True,
      )
      break

    if (slot - shard_start) % poll_interval_slots == 0:
      print(
        f"[generate_codes] slot={slot} status={status} "
        f"new={counts['new']} dup={counts['duplicate']} "
        f"synerr={counts['syntax_error']} genfail={counts['generation_failed']} "
        f"distinct_in_cell={distinct + counts['new']}",
        flush=True,
      )

  shard_elapsed = time.time() - shard_t0
  print(
    f"[generate_codes] shard complete in {shard_elapsed:.1f}s; counts={counts}",
    flush=True,
  )
  return counts


def _read_done_slots(cell_dir, shard_start, shard_end) -> set:
  """Slots whose result already lives in ``codes/_index.jsonl``."""
  index_path = codes_dir(cell_dir) / "_index.jsonl"
  if not index_path.exists():
    return set()
  done = set()
  with open(index_path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
      except Exception:
        continue
      slot = int(rec.get("slot", -1))
      if shard_start <= slot < shard_end:
        done.add(slot)
  return done


def _build_parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", required=True)
  p.add_argument("--llm-config", required=True)
  p.add_argument("--cell-dir", required=True)
  p.add_argument("--shard-start", type=int, required=True)
  p.add_argument("--shard-max-indices", type=int, default=150,
                 help="Hard cap on how many slot indices this shard touches")
  p.add_argument("--shard-valid-target", type=int, default=100,
                 help="Stop the shard once this many NEW (non-duplicate) hashes were claimed")
  p.add_argument("--cell-valid-target", type=int, default=10000,
                 help="Stop the shard once the cell has this many distinct codes total")
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--api-url", default=None,
                 help="Override the LLM api_url for the per-task Ollama port")
  return p


def main(argv=None):
  args = _build_parser().parse_args(argv)
  return run_shard(
    task_path=args.task,
    llm_config_path=args.llm_config,
    cell_dir=args.cell_dir,
    shard_start=args.shard_start,
    shard_max_indices=args.shard_max_indices,
    shard_valid_target=args.shard_valid_target,
    cell_valid_target=args.cell_valid_target,
    seed=args.seed,
    api_url_override=args.api_url,
    cli_args=vars(args),
  )


if __name__ == "__main__":
  sys.exit(0 if main() is not None else 1)
