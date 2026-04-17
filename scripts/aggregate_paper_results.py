"""Roll up the paper sweep into reviewer-ready summaries.

Walks ``<root>/<task>/<llm>/`` and produces:

  <root>/aggregate_metrics.jsonl          # per-K rows from evaluate.py
  <root>/summary.csv                      # paper-ready scalars per cell, K
  <root>/generation_summary.csv           # per-cell attempts, valid, failures, timings
  <root>/test_scores.csv                  # per prediction cell, per K
  <root>/loo_matrices/<task>_<llm>.npz    # (n_train, K) raw LOO log-liks
  <root>/cumulative_weight_curves/<task>_<llm>.csv  # rank vs cumulative weight
  <root>/top_models/<task>_<llm>/         # 10 highest-weighted programs each scheme

Run:

  uv run python -m scripts.aggregate_paper_results --root experiment_results/paper

A reviewer can answer most questions about the sweep from these files alone.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import evaluate as ev


DEFAULT_KS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def _list_cells(root):
  root = Path(root)
  out = []
  for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
    for llm_dir in sorted(p for p in task_dir.iterdir() if p.is_dir() and not p.name.startswith("_")):
      out.append((task_dir.name, llm_dir.name, llm_dir))
  return out


def _load_task(repo_root, task_name):
  path = Path(repo_root) / "tasks" / f"{task_name}.json"
  if not path.exists():
    return {}
  with open(path) as f:
    return json.load(f)


def summarize_generation(cell_dir):
  cell_dir = Path(cell_dir)
  codes_dir = cell_dir / "codes"
  failures_dir = codes_dir / "_failures"
  index_path = codes_dir / "_index.jsonl"

  codes_count = len(list(codes_dir.glob("code_*.code.json")))
  failure_count = len(list(failures_dir.glob("failure_*.json"))) if failures_dir.exists() else 0
  attempts = 0
  duplicates = 0
  total_gen_seconds = 0.0
  status_counter = Counter()
  if index_path.exists():
    with open(index_path) as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          rec = json.loads(line)
        except Exception:
          continue
        attempts += 1
        status_counter[rec.get("status", "unknown")] += 1
        if rec.get("status") == "duplicate":
          duplicates += 1
        gs = rec.get("generation_seconds")
        if isinstance(gs, (int, float)):
          total_gen_seconds += float(gs)

  samples_dir = cell_dir / "samples"
  sample_status = Counter()
  total_mcmc = 0.0
  total_loo = 0.0
  total_iwae = 0.0
  total_test = 0.0
  n_div_total = 0
  n_div_models = 0
  if samples_dir.exists():
    for meta_path in samples_dir.glob("sample_*.meta.json"):
      try:
        with open(meta_path) as f:
          meta = json.load(f)
      except Exception:
        sample_status["read_error"] += 1
        continue
      sample_status[meta.get("status", "unknown")] += 1
      timings = meta.get("timings") or {}
      total_mcmc += float(timings.get("mcmc_seconds", 0.0) or 0.0)
      total_loo += float(timings.get("loo_seconds", 0.0) or 0.0)
      total_iwae += float(timings.get("log_marginal_seconds", 0.0) or 0.0)
      total_test += float(timings.get("test_scoring_seconds", 0.0) or 0.0)
      diags = ((meta.get("mcmc") or {}).get("diagnostics") or {})
      d = diags.get("num_divergences")
      if isinstance(d, int):
        n_div_total += d
        n_div_models += 1

  return {
    "codes_count": codes_count,
    "failure_count": failure_count,
    "attempts": attempts,
    "duplicates": duplicates,
    "total_generation_seconds": total_gen_seconds,
    "code_status_counts": dict(status_counter),
    "sample_status_counts": dict(sample_status),
    "total_mcmc_seconds": total_mcmc,
    "total_loo_seconds": total_loo,
    "total_iwae_seconds": total_iwae,
    "total_test_seconds": total_test,
    "num_divergences_sum": int(n_div_total),
    "num_models_with_divergence_field": int(n_div_models),
  }


def write_loo_matrix(cell_dir, out_path, target_names):
  artifacts = ev.load_artifacts(cell_dir / "samples", target_names)
  if not artifacts:
    return None
  loo_lengths = [a.loo_log_liks.size for a in artifacts]
  n_data = max(loo_lengths) if loo_lengths else 0
  usable = [a for a in artifacts if a.loo_log_liks.size == n_data and n_data > 0]
  if not usable or n_data == 0:
    return None
  matrix = np.column_stack([a.loo_log_liks for a in usable])  # (n_data, K)
  shas = np.array([a.sha or str(a.index) for a in usable])
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(out_path, loo=matrix, shas=shas)
  return matrix.shape


def write_cumulative_weight_curve(metrics, out_path):
  weights = metrics.get("weights") or {}
  bma = np.asarray(weights.get("bma", []), dtype=np.float64)
  stacking = np.asarray(weights.get("stacking", []), dtype=np.float64)
  if bma.size == 0:
    return
  bma_sorted = -np.sort(-bma)
  st_sorted = -np.sort(-stacking)
  rows = []
  for rank in range(bma.size):
    rows.append({
      "rank": rank + 1,
      "cumulative_bma": float(np.sum(bma_sorted[:rank + 1])),
      "cumulative_stacking": float(np.sum(st_sorted[:rank + 1])),
    })
  out_path.parent.mkdir(parents=True, exist_ok=True)
  with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["rank", "cumulative_bma", "cumulative_stacking"])
    writer.writeheader()
    writer.writerows(rows)


def dump_top_models(cell_dir, metrics, out_dir, top_n=10):
  weights = metrics.get("weights") or {}
  bma = np.asarray(weights.get("bma", []), dtype=np.float64)
  stacking = np.asarray(weights.get("stacking", []), dtype=np.float64)
  if bma.size == 0:
    return

  artifacts = ev.load_artifacts(cell_dir / "samples", target_names=[])
  if len(artifacts) != bma.size:
    # Length mismatch: K_used was a strict prefix of artifacts; align.
    artifacts = artifacts[:bma.size]

  out_dir.mkdir(parents=True, exist_ok=True)

  for label, w in [("bma", bma), ("stacking", stacking)]:
    order = np.argsort(-w)
    for rank in range(min(top_n, len(order))):
      idx = int(order[rank])
      art = artifacts[idx]
      sha = art.sha or f"idx{art.index:06d}"
      code_payload_path = cell_dir / "codes" / f"code_{sha}.code.json"
      if not code_payload_path.exists():
        continue
      try:
        with open(code_payload_path) as f:
          payload = json.load(f)
        code = payload.get("canonical_code") or payload.get("raw_code", "")
      except Exception:
        continue
      weight_str = f"{w[idx]:.4f}".replace(".", "p")
      filename = f"{label}_rank_{rank+1:02d}_w_{weight_str}_{sha}.code.py"
      with open(out_dir / filename, "w") as f:
        f.write(f"# task={cell_dir.parent.name} llm={cell_dir.name}\n")
        f.write(f"# scheme={label} rank={rank+1} weight={float(w[idx]):.6f} sha={sha}\n")
        f.write(code)
        f.write("\n")


def aggregate_cell(root, repo_root, task_name, llm_name, cell_dir, ks, jsonl_path):
  task = _load_task(repo_root, task_name)
  targets = task.get("targets") or []

  artifacts = ev.load_artifacts(cell_dir / "samples", targets)
  if not artifacts:
    print(f"  [{task_name}/{llm_name}] no artifacts; skipping evaluate")
    return None

  n_avail = len(artifacts)
  use_ks = [k for k in ks if k <= n_avail]
  if n_avail > 0 and (not use_ks or use_ks[-1] != n_avail):
    use_ks.append(n_avail)
  use_ks = sorted(set(use_ks))

  rows = []
  per_k_metrics = {}
  for k in use_ks:
    metrics = ev.evaluate_for_k(artifacts, k, targets)
    if metrics is None:
      continue
    per_k_metrics[k] = metrics

    flat = _flatten_metrics(task_name, llm_name, k, metrics)
    rows.append(flat)
    with open(jsonl_path, "a") as f:
      f.write(json.dumps({
        "task": task_name, "llm": llm_name, **metrics,
      }) + "\n")

  largest_k = max(per_k_metrics.keys()) if per_k_metrics else None
  if largest_k is not None:
    last_metrics = per_k_metrics[largest_k]
    write_cumulative_weight_curve(
      last_metrics,
      root / "cumulative_weight_curves" / f"{task_name}_{llm_name}.csv",
    )
    dump_top_models(
      cell_dir, last_metrics,
      root / "top_models" / f"{task_name}_{llm_name}",
    )
    write_loo_matrix(
      cell_dir,
      root / "loo_matrices" / f"{task_name}_{llm_name}.npz",
      targets,
    )

  return rows


def _flatten_metrics(task_name, llm_name, k, metrics):
  ws = metrics.get("weight_stats") or {}
  test = metrics.get("test_set") or {}
  boot = metrics.get("bootstrap") or {}
  out = {
    "task": task_name,
    "llm": llm_name,
    "k_requested": metrics.get("k_requested"),
    "k_used": metrics.get("k_used"),
    "n_bma_finite": metrics.get("n_bma_finite"),
    "n_loo_valid": metrics.get("n_loo_valid"),
    "n_datapoints": metrics.get("n_datapoints"),
    "n_test": test.get("n_test"),
    "n_models_with_test": test.get("n_models_with_test"),
    "stacking_objective": metrics.get("stacking_objective"),
    "ess_uniform": ws.get("ess_uniform"),
    "ess_bma": ws.get("ess_bma"),
    "ess_stacking": ws.get("ess_stacking"),
    "entropy_bma": ws.get("entropy_bma"),
    "entropy_stacking": ws.get("entropy_stacking"),
    "max_bma": ws.get("max_bma"),
    "max_stacking": ws.get("max_stacking"),
    "l1_bma_stacking": ws.get("l1_bma_stacking"),
    "kl_bma_given_stacking": ws.get("kl_bma_given_stacking"),
    "kl_stacking_given_bma": ws.get("kl_stacking_given_bma"),
    "log_predictive_uniform": test.get("log_predictive_uniform"),
    "log_predictive_bma": test.get("log_predictive_bma"),
    "log_predictive_stacking": test.get("log_predictive_stacking"),
    "boot_n": boot.get("n_bootstrap"),
    "boot_ess_stacking_p05": boot.get("ess_stacking_p05"),
    "boot_ess_stacking_p50": boot.get("ess_stacking_p50"),
    "boot_ess_stacking_p95": boot.get("ess_stacking_p95"),
    "boot_max_stacking_p05": boot.get("max_stacking_p05"),
    "boot_max_stacking_p50": boot.get("max_stacking_p50"),
    "boot_max_stacking_p95": boot.get("max_stacking_p95"),
  }
  for tname, tdict in (metrics.get("targets") or {}).items():
    if not tdict:
      continue
    out[f"target_{tname}_mean_uniform"] = tdict.get("posterior_mean_uniform")
    out[f"target_{tname}_mean_bma"] = tdict.get("posterior_mean_bma")
    out[f"target_{tname}_mean_stacking"] = tdict.get("posterior_mean_stacking")
    out[f"target_{tname}_var_uniform"] = tdict.get("epistemic_var_uniform")
    out[f"target_{tname}_var_bma"] = tdict.get("epistemic_var_bma")
    out[f"target_{tname}_var_stacking"] = tdict.get("epistemic_var_stacking")
  return out


def write_csv(path, rows, fieldnames=None):
  path.parent.mkdir(parents=True, exist_ok=True)
  if not rows:
    return
  if fieldnames is None:
    seen = []
    for r in rows:
      for k in r.keys():
        if k not in seen:
          seen.append(k)
    fieldnames = seen
  with open(path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in rows:
      writer.writerow({k: r.get(k) for k in fieldnames})


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--root", required=True, help="experiment_results/paper or similar")
  p.add_argument("--repo-root", default=".", help="Root of the repo (for tasks/)")
  p.add_argument("--ks", type=int, nargs="*", default=DEFAULT_KS)
  args = p.parse_args(argv)

  root = Path(args.root).resolve()
  repo_root = Path(args.repo_root).resolve()
  cells = _list_cells(root)
  print(f"Found {len(cells)} cells under {root}")

  jsonl_path = root / "aggregate_metrics.jsonl"
  if jsonl_path.exists():
    jsonl_path.unlink()

  summary_rows = []
  gen_rows = []
  test_rows = []

  for task_name, llm_name, cell_dir in cells:
    print(f"--- {task_name} / {llm_name} ---")
    gen = summarize_generation(cell_dir)
    gen_row = {
      "task": task_name,
      "llm": llm_name,
      "codes_distinct": gen["codes_count"],
      "attempts": gen["attempts"],
      "duplicates": gen["duplicates"],
      "failures": gen["failure_count"],
      "total_generation_seconds": gen["total_generation_seconds"],
      "total_mcmc_seconds": gen["total_mcmc_seconds"],
      "total_iwae_seconds": gen["total_iwae_seconds"],
      "total_loo_seconds": gen["total_loo_seconds"],
      "total_test_seconds": gen["total_test_seconds"],
      "num_divergences_sum": gen["num_divergences_sum"],
      "num_models_with_divergence_field": gen["num_models_with_divergence_field"],
    }
    for k, v in gen["sample_status_counts"].items():
      gen_row[f"samples_{k}"] = v
    for k, v in gen["code_status_counts"].items():
      gen_row[f"codes_{k}"] = v
    gen_rows.append(gen_row)

    cell_rows = aggregate_cell(
      root=root, repo_root=repo_root,
      task_name=task_name, llm_name=llm_name,
      cell_dir=cell_dir, ks=args.ks, jsonl_path=jsonl_path,
    )
    if not cell_rows:
      continue
    summary_rows.extend(cell_rows)
    for r in cell_rows:
      if r.get("n_test") and r.get("n_models_with_test"):
        test_rows.append({
          "task": r["task"],
          "llm": r["llm"],
          "k_used": r["k_used"],
          "n_test": r["n_test"],
          "n_models_with_test": r["n_models_with_test"],
          "log_predictive_uniform": r.get("log_predictive_uniform"),
          "log_predictive_bma": r.get("log_predictive_bma"),
          "log_predictive_stacking": r.get("log_predictive_stacking"),
        })

  write_csv(root / "summary.csv", summary_rows)
  write_csv(root / "generation_summary.csv", gen_rows)
  write_csv(root / "test_scores.csv", test_rows)

  print()
  print(f"Wrote summary.csv with {len(summary_rows)} rows")
  print(f"Wrote generation_summary.csv with {len(gen_rows)} rows")
  print(f"Wrote test_scores.csv with {len(test_rows)} rows")
  print(f"Aggregate metrics jsonl: {jsonl_path}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
