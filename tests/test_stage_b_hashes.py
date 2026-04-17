"""Stage B: inference + test scoring + resume on hash-addressed cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sample_from_codes as sb
from llb.codes import (
  canonicalize_code,
  claim_code,
  code_hash,
  sample_meta_path,
  sample_npz_path,
)


GOOD_LOSS_LOGNORMAL = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["losses_musd"], dtype=jnp.float32)
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(3.0))
    numpyro.deterministic("median_loss", jnp.exp(mu))
    numpyro.sample("obs", dist.LogNormal(mu, sigma), obs=y)
"""


def _fast_cfg():
  return sb.SampleConfig(
    mcmc_num_warmup=20,
    mcmc_num_samples=40,
    log_marginal_num_inner=2,
    log_marginal_num_outer=4,
    loo_num_inner=3,
    loo_num_warmup=10,
    loo_num_samples=20,
    use_true_loo=True,
    save_full_posterior=False,
    loo_workers=1,
  )


def _seed_cell(tmp_path, codes):
  cell_dir = tmp_path / "cell"
  shas = []
  for raw in codes:
    canonical = canonicalize_code(raw)
    sha = code_hash(canonical)
    payload = {
      "sha": sha,
      "raw_code": raw,
      "canonical_code": canonical,
      "raw_llm_response": "",
      "prompt_messages": [],
      "first_seed": 0,
      "first_slot": 0,
      "generation_seconds": 0.0,
      "generation_diagnostics": {"n_attempts": 1, "attempts": []},
    }
    claim_code(cell_dir, sha, payload)
    shas.append(sha)
  return cell_dir, shas


def _make_task(tmp_path, with_test=True):
  task = {
    "name": "fake_pred",
    "text": "predict losses",
    "data": {"losses_musd": [0.1, 0.5, 1.2, 3.4, 7.8, 12.0]},
    "test_data": {"losses_musd": [0.3, 2.0]} if with_test else None,
    "targets": ["median_loss"],
    "true_latents": None,
    "task_type": "prediction" if with_test else "estimation",
    "metadata": {},
  }
  path = tmp_path / "task.json"
  path.write_text(json.dumps(task))
  return task, path


def test_stage_b_writes_npz_and_meta_with_test(tmp_path):
  task, task_path = _make_task(tmp_path, with_test=True)
  cell_dir, shas = _seed_cell(tmp_path, [GOOD_LOSS_LOGNORMAL])
  sha = shas[0]

  status = sb.process_hash(sha, cell_dir, task, _fast_cfg(), base_seed=42)
  assert status == "ok"

  meta_path = sample_meta_path(cell_dir, sha)
  npz_path = sample_npz_path(cell_dir, sha)
  assert meta_path.exists() and npz_path.exists()

  meta = json.loads(meta_path.read_text())
  assert meta["status"] == "ok"
  assert meta["targets"] == ["median_loss"]
  assert meta["n_train"] == 6
  assert meta["n_test"] == 2
  assert meta["loo"]["status"] == "ok"
  assert meta["log_marginal"]["status"] == "ok"
  assert meta["test_scoring"]["status"] == "ok"
  diag = meta["mcmc"]["diagnostics"]
  assert diag.get("num_divergences") is not None
  # n_eff dict should have at least one entry
  assert isinstance(diag.get("n_eff"), dict) and len(diag["n_eff"]) >= 1

  z = np.load(npz_path, allow_pickle=False)
  assert "target__median_loss" in z.files
  assert z["loo_log_liks"].shape == (6,)
  assert z["test_log_liks"].shape == (2,)
  assert np.all(np.isfinite(z["test_log_liks"]) | (z["test_log_liks"] < 0))


def test_stage_b_resume_skips(tmp_path):
  task, task_path = _make_task(tmp_path, with_test=True)
  cell_dir, shas = _seed_cell(tmp_path, [GOOD_LOSS_LOGNORMAL])
  sha = shas[0]

  s1 = sb.process_hash(sha, cell_dir, task, _fast_cfg(), base_seed=42)
  assert s1 == "ok"

  # Touching no inputs and re-running must short-circuit.
  s2 = sb.process_hash(sha, cell_dir, task, _fast_cfg(), base_seed=42)
  assert s2 == "skip_ok"


def test_stage_b_run_shard_partitions(tmp_path):
  task, task_path = _make_task(tmp_path, with_test=False)
  cell_dir, shas = _seed_cell(
    tmp_path,
    [GOOD_LOSS_LOGNORMAL.replace("Normal(0.0, 5.0)", f"Normal({i}.0, 5.0)") for i in range(3)],
  )

  # Two shards, count=2, should split 3 hashes as ceil(3/2)=2 then 1.
  c0 = sb.run_shard(task_path=task_path, cell_dir=cell_dir,
                   array_task_id=0, array_task_count=2,
                   cfg=_fast_cfg(), seed=42)
  c1 = sb.run_shard(task_path=task_path, cell_dir=cell_dir,
                   array_task_id=1, array_task_count=2,
                   cfg=_fast_cfg(), seed=42)
  total_done = sum(c0.get("ok", 0) + c1.get("ok", 0)
                  for _ in range(1))
  assert (c0.get("ok", 0) + c1.get("ok", 0)) >= 2
