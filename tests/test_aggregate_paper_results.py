"""Aggregator end-to-end: build a fake two-cell tree and check outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import scripts.aggregate_paper_results as agg
from llb.codes import canonicalize_code, claim_code, code_hash
from llb.io_utils import atomic_savez, atomic_write_json


CODE_TEMPLATES = [
  """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["y"], dtype=jnp.float32)
    mu = numpyro.sample("mu", dist.Normal({mu0}, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    numpyro.deterministic("median_loss", jnp.exp(mu))
    numpyro.sample("obs", dist.LogNormal(mu, sigma), obs=y)
""",
]


def _seed_cell(cell_dir: Path, n_models=3, n_train=20, n_test=5, with_test=True):
  cell_dir.mkdir(parents=True, exist_ok=True)
  rng = np.random.default_rng(0)
  for k in range(n_models):
    raw = CODE_TEMPLATES[0].format(mu0=k * 0.5)
    canonical = canonicalize_code(raw)
    sha = code_hash(canonical)
    payload = {
      "sha": sha,
      "raw_code": raw,
      "canonical_code": canonical,
      "raw_llm_response": "",
      "prompt_messages": [],
      "first_seed": 0,
      "first_slot": k,
      "generation_seconds": 1.0,
      "generation_diagnostics": {"n_attempts": 1, "attempts": []},
    }
    claim_code(cell_dir, sha, payload)

    target_arr = rng.normal(size=200).astype(np.float64)
    loo = rng.normal(loc=-3.0 + k, scale=1.0, size=n_train).astype(np.float64)
    test = rng.normal(loc=-3.0 + k, scale=1.0, size=n_test).astype(np.float64) if with_test else np.zeros(0)
    arrays = {
      "target__median_loss": target_arr,
      "log_marginal_bound": np.float64(-100.0 - k),
      "loo_log_liks": loo,
    }
    if with_test:
      arrays["test_log_liks"] = test
    atomic_savez(cell_dir / "samples" / f"sample_{sha}.npz", **arrays)
    meta = {
      "sha": sha,
      "status": "ok",
      "reason": None,
      "rng_seed": 1,
      "n_train": n_train,
      "n_test": n_test if with_test else 0,
      "targets": ["median_loss"],
      "available_sites": ["mu", "sigma", "median_loss"],
      "log_marginal": {"status": "ok", "reason": None, "value": -100.0 - k},
      "loo": {"status": "ok", "reason": None, "use_true_loo": True, "diagnostics": None},
      "test_scoring": {"status": "ok" if with_test else "skipped", "reason": None, "diagnostics": None},
      "mcmc": {"num_warmup": 100, "num_samples": 200, "diagnostics": {"num_divergences": k}},
      "timings": {"mcmc_seconds": 1.0, "log_marginal_seconds": 0.5, "loo_seconds": 2.0, "test_scoring_seconds": 0.5},
    }
    atomic_write_json(cell_dir / "samples" / f"sample_{sha}.meta.json", meta)


def _write_task(repo_root: Path, task_name: str, n_train: int, n_test: int):
  task = {
    "name": task_name,
    "text": "fake",
    "data": {"y": list(np.zeros(n_train).tolist())},
    "test_data": {"y": list(np.zeros(n_test).tolist())} if n_test else None,
    "targets": ["median_loss"],
    "true_latents": None,
    "task_type": "prediction" if n_test else "estimation",
    "metadata": {},
  }
  d = repo_root / "tasks"
  d.mkdir(parents=True, exist_ok=True)
  (d / f"{task_name}.json").write_text(json.dumps(task))


def test_aggregator_produces_all_outputs(tmp_path):
  root = tmp_path / "experiment_results" / "paper"
  _seed_cell(root / "task_a" / "qwen25_coder", n_models=3, n_test=5, with_test=True)
  _seed_cell(root / "task_b" / "gemma4_e4b", n_models=3, n_test=0, with_test=False)
  _write_task(tmp_path, "task_a", n_train=20, n_test=5)
  _write_task(tmp_path, "task_b", n_train=20, n_test=0)

  rc = agg.main([
    "--root", str(root),
    "--repo-root", str(tmp_path),
    "--ks", "2", "3",
  ])
  assert rc == 0

  summary = list(csv.DictReader(open(root / "summary.csv")))
  assert len(summary) >= 2
  cols = set(summary[0].keys())
  assert "stacking_objective" in cols
  assert "log_predictive_uniform" in cols
  assert "boot_ess_stacking_p50" in cols

  gen = list(csv.DictReader(open(root / "generation_summary.csv")))
  assert len(gen) == 2
  for row in gen:
    assert "codes_distinct" in row

  # Test scores only for cells with test_log_liks (task_a).
  test_rows = list(csv.DictReader(open(root / "test_scores.csv")))
  task_a_rows = [r for r in test_rows if r["task"] == "task_a"]
  assert len(task_a_rows) >= 1
  for r in task_a_rows:
    assert r["log_predictive_uniform"] not in (None, "", "None")

  # LOO matrix and cumulative weight curve and top-models all present.
  assert (root / "loo_matrices" / "task_a_qwen25_coder.npz").exists()
  cwc = root / "cumulative_weight_curves" / "task_a_qwen25_coder.csv"
  assert cwc.exists()
  cwc_rows = list(csv.DictReader(open(cwc)))
  assert len(cwc_rows) == 3
  top_dir = root / "top_models" / "task_a_qwen25_coder"
  assert top_dir.is_dir()
  files = list(top_dir.glob("*.code.py"))
  assert len(files) >= 2  # at least bma and stacking top files
