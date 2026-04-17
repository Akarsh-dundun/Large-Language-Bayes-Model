"""End-to-end smoke tests for every FEMA NRI task.

For each task spec we supply a hand-written NumPyro model that matches the
task's data shape and target site name, mock ``LLMClient.generate`` to
return the code, and run ``generate_and_sample.process_index``. This
catches problems where a task's data shape or target name breaks the
MCMC plus IWAE plus LOO pipeline without requiring a live Ollama server.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import generate_and_sample as gs
import evaluate as ev
from tasks.build_fema_nri_tasks import TASKS_DIR, TASK_SPECS


def _fenced(code: str) -> str:
  return f"```python\n{code.strip()}\n```"


def _fast_cfg() -> gs.SampleConfig:
  return gs.SampleConfig(
    mcmc_num_warmup=20,
    mcmc_num_samples=40,
    log_marginal_num_inner=2,
    log_marginal_num_outer=4,
    loo_num_inner=3,
    loo_num_warmup=10,
    loo_num_samples=20,
    use_true_loo=True,
    save_full_posterior=False,
  )


LOGNORMAL_LOSSES = """
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

POISSON_COUNTS = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
  y = jnp.asarray(data["events"], dtype=jnp.int32)
  rate = numpyro.sample("rate", dist.Gamma(2.0, 0.05))
  numpyro.sample("obs", dist.Poisson(rate), obs=y)
"""

LOGNORMAL_FREQUENCY = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
  y = jnp.asarray(data["frequency_per_year"], dtype=jnp.float32)
  mu = numpyro.sample("mu", dist.Normal(-3.0, 3.0))
  sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
  numpyro.deterministic("mean_frequency", jnp.exp(mu + 0.5 * sigma ** 2))
  numpyro.sample("obs", dist.LogNormal(mu, sigma), obs=y)
"""

GAMMA_LOSSES = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
  y = jnp.asarray(data["losses_musd"], dtype=jnp.float32)
  shape = numpyro.sample("shape", dist.HalfNormal(5.0))
  rate = numpyro.sample("rate", dist.HalfNormal(5.0))
  # median of Gamma lacks a closed form; use mean / 0.693 as a simple surrogate
  numpyro.deterministic("median_loss", shape / rate * 0.6931)
  numpyro.sample("obs", dist.Gamma(shape + 0.1, rate + 0.1), obs=y)
"""


CODE_FOR_TASK = {
  "hurricane_eal_counties": LOGNORMAL_LOSSES,
  "tornado_counts_plains": POISSON_COUNTS,
  "earthquake_frequency_west": LOGNORMAL_FREQUENCY,
  "wildfire_eal_west": LOGNORMAL_LOSSES,
  "inland_flood_eal": LOGNORMAL_LOSSES,
}

SPECS_BY_NAME = {s.name: s for s in TASK_SPECS}


@pytest.fixture
def llm(tmp_path):
  cfg = {
    "name": "mock",
    "api_url": "http://127.0.0.1:0/api/generate",
    "api_key": None,
    "api_model": "mock:1",
    "llm_timeout": 5,
    "llm_max_retries": 0,
    "llm_retry_backoff": 0.1,
  }
  return gs.build_llm(cfg)


@pytest.mark.parametrize("task_name", list(CODE_FOR_TASK.keys()))
def test_process_index_runs_on_task(task_name, llm, tmp_path):
  spec = SPECS_BY_NAME[task_name]
  with open(TASKS_DIR / f"{spec.name}.json") as f:
    task = json.load(f)
  out = tmp_path / "out"
  out.mkdir()

  code = CODE_FOR_TASK[task_name]
  with patch.object(llm, "generate", return_value=_fenced(code)):
    status = gs.process_index(
      index=0, task=task, llm=llm, output_dir=out,
      base_seed=0, cfg=_fast_cfg(),
    )

  assert status == "ok", (
    f"{task_name}: process_index returned {status}. "
    f"Meta: {(out / 'model_000000.meta.json').read_text()}"
  )

  meta = json.loads((out / "model_000000.meta.json").read_text())
  assert meta["status"] == "ok"
  assert meta["targets"] == list(spec.targets)
  assert meta["n_datapoints"] == spec.n_train
  assert meta["log_marginal"]["status"] == "ok"
  assert meta["loo"]["status"] == "ok"

  z = np.load(out / "model_000000.npz", allow_pickle=False)
  for t in spec.targets:
    key = f"target__{t}"
    assert key in z.files, f"{task_name}: missing {key} in npz"
  assert z["loo_log_liks"].shape == (spec.n_train,)
  assert np.isfinite(float(z["log_marginal_bound"]))


def test_evaluate_end_to_end_on_hurricane(llm, tmp_path):
  """Generate 3 models for the hurricane task and make sure evaluate.py produces metrics."""
  spec = SPECS_BY_NAME["hurricane_eal_counties"]
  task_path = TASKS_DIR / f"{spec.name}.json"
  with open(task_path) as f:
    task = json.load(f)
  out = tmp_path / "samples"
  out.mkdir()

  def fake_generate(*args, **kwargs):
    seed = kwargs.get("seed", 0)
    code = LOGNORMAL_LOSSES if seed % 2 == 0 else GAMMA_LOSSES
    return _fenced(code)

  with patch.object(llm, "generate", side_effect=fake_generate):
    for idx in range(3):
      gs.process_index(
        index=idx, task=task, llm=llm, output_dir=out,
        base_seed=200, cfg=_fast_cfg(),
      )

  jsonl_path = tmp_path / "metrics.jsonl"
  full_path = tmp_path / "metrics_full.json"
  rc = ev.main([
    "--task", str(task_path),
    "--sample-dir", str(out),
    "--llm-name", "mock",
    "--ks", "2", "3",
    "--output-jsonl", str(jsonl_path),
    "--output-full-json", str(full_path),
  ])
  assert rc == 0
  rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
  assert len(rows) == 2
  for r in rows:
    assert r["task"] == "hurricane_eal_counties"
    assert r["n_datapoints"] == spec.n_train
    assert r["n_loo_valid"] > 0
    assert np.isfinite(r["stacking_objective"])
    med = r["targets"]["median_loss"]
    assert med["posterior_mean_uniform"] > 0


def test_evaluate_end_to_end_on_tornado(llm, tmp_path):
  """Tornado task runs through evaluate.py too (estimation, no test split)."""
  spec = SPECS_BY_NAME["tornado_counts_plains"]
  task_path = TASKS_DIR / f"{spec.name}.json"
  with open(task_path) as f:
    task = json.load(f)
  out = tmp_path / "samples"
  out.mkdir()

  with patch.object(llm, "generate", return_value=_fenced(POISSON_COUNTS)):
    for idx in range(3):
      gs.process_index(
        index=idx, task=task, llm=llm, output_dir=out,
        base_seed=300, cfg=_fast_cfg(),
      )

  jsonl_path = tmp_path / "metrics.jsonl"
  rc = ev.main([
    "--task", str(task_path),
    "--sample-dir", str(out),
    "--llm-name", "mock",
    "--ks", "3",
    "--output-jsonl", str(jsonl_path),
  ])
  assert rc == 0
  rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
  assert len(rows) == 1
  r = rows[0]
  assert r["task"] == "tornado_counts_plains"
  assert r["n_datapoints"] == spec.n_train
  assert r["n_loo_valid"] > 0
  rate = r["targets"]["rate"]
  assert rate["posterior_mean_uniform"] > 0
