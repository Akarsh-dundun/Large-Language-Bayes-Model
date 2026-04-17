"""End-to-end smoke test for generate_and_sample.py and evaluate.py.

Uses a mocked LLMClient so the test does not require a live Ollama endpoint.
Runs a small shard against the coin_flip task, verifies artifacts, and then
runs evaluate.py against them.
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import generate_and_sample as gs
import evaluate as ev


COIN_MODEL_CODE = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
  flips = jnp.asarray(data["flips"])
  true_bias = numpyro.sample("true_bias", dist.Beta(1.0, 1.0))
  numpyro.sample("obs", dist.Bernoulli(probs=true_bias), obs=flips)
"""

COIN_MODEL_CODE_SKEWED = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
  flips = jnp.asarray(data["flips"])
  true_bias = numpyro.sample("true_bias", dist.Beta(2.0, 5.0))
  numpyro.sample("obs", dist.Bernoulli(probs=true_bias), obs=flips)
"""


def _fenced(code):
  return f"```python\n{code.strip()}\n```"


@pytest.fixture
def coin_task(tmp_path):
  task = {
    "name": "coin_flip",
    "text": "coin task",
    "data": {"flips": [0, 1, 0, 1, 1, 0]},
    "test_data": None,
    "targets": ["true_bias"],
    "true_latents": None,
    "task_type": "estimation",
    "metadata": {"internet_seen": True},
  }
  path = tmp_path / "task.json"
  path.write_text(json.dumps(task))
  return task, path


@pytest.fixture
def llm_config_path(tmp_path):
  cfg = {
    "name": "mock",
    "api_url": "http://127.0.0.1:0/api/generate",
    "api_key": None,
    "api_model": "mock:1",
    "llm_timeout": 5,
    "llm_max_retries": 0,
    "llm_retry_backoff": 0.1,
  }
  path = tmp_path / "llm.json"
  path.write_text(json.dumps(cfg))
  return path


def _fast_cfg():
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


def test_process_index_writes_artifacts(coin_task, llm_config_path, tmp_path):
  """Single model index produces a well-formed .npz + .meta.json."""
  task, _ = coin_task
  out = tmp_path / "out"
  out.mkdir()

  llm = gs.build_llm(json.loads(llm_config_path.read_text()))
  with patch.object(llm, "generate", return_value=_fenced(COIN_MODEL_CODE)):
    status = gs.process_index(
      index=0, task=task, llm=llm, output_dir=out,
      base_seed=42, cfg=_fast_cfg(),
    )

  assert status == "ok"
  meta_path = out / "model_000000.meta.json"
  npz_path = out / "model_000000.npz"
  assert meta_path.exists()
  assert npz_path.exists()

  meta = json.loads(meta_path.read_text())
  assert meta["status"] == "ok"
  assert meta["targets"] == ["true_bias"]
  assert meta["log_marginal"]["status"] == "ok"
  assert meta["loo"]["status"] == "ok"
  assert meta["n_datapoints"] == 6

  z = np.load(npz_path, allow_pickle=False)
  assert "target__true_bias" in z.files
  assert "log_marginal_bound" in z.files
  assert "loo_log_liks" in z.files
  assert z["loo_log_liks"].shape == (6,)
  assert z["target__true_bias"].shape[0] == 40
  assert np.isfinite(float(z["log_marginal_bound"]))


def test_resume_skips_existing_meta(coin_task, llm_config_path, tmp_path):
  """A second call with the same index is skipped when meta already exists."""
  task, _ = coin_task
  out = tmp_path / "out"
  out.mkdir()

  llm = gs.build_llm(json.loads(llm_config_path.read_text()))
  with patch.object(llm, "generate", return_value=_fenced(COIN_MODEL_CODE)) as mock_gen:
    status1 = gs.process_index(
      index=3, task=task, llm=llm, output_dir=out,
      base_seed=42, cfg=_fast_cfg(),
    )
    calls_first = mock_gen.call_count
    status2 = gs.process_index(
      index=3, task=task, llm=llm, output_dir=out,
      base_seed=42, cfg=_fast_cfg(),
    )
    calls_second = mock_gen.call_count

  assert status1 == status2 == "ok"
  assert calls_second == calls_first, "resume must not call the LLM again"


def test_generation_failure_records_meta(coin_task, llm_config_path, tmp_path):
  """If the LLM never returns parseable code, we still write a meta.json."""
  task, _ = coin_task
  out = tmp_path / "out"
  out.mkdir()

  llm = gs.build_llm(json.loads(llm_config_path.read_text()))
  with patch.object(llm, "generate", return_value="not a code block at all"):
    status = gs.process_index(
      index=7, task=task, llm=llm, output_dir=out,
      base_seed=42, cfg=_fast_cfg(),
    )

  meta_path = out / "model_000007.meta.json"
  npz_path = out / "model_000007.npz"
  assert meta_path.exists()
  assert not npz_path.exists()
  meta = json.loads(meta_path.read_text())
  assert status in {"syntax_error", "generation_failed"}
  assert meta["status"] == status
  assert meta["code"] is None


def test_compile_error_records_meta(coin_task, llm_config_path, tmp_path):
  """Code that exec-raises after passing goal-name validation yields compile_error."""
  task, _ = coin_task
  out = tmp_path / "out"
  out.mkdir()

  bad_code = (
    "import numpyro\n"
    "import numpyro.distributions as dist\n"
    "raise RuntimeError('module-level boom')\n"
    "def model(data):\n"
    "  numpyro.sample('true_bias', dist.Beta(1.0, 1.0))\n"
  )
  llm = gs.build_llm(json.loads(llm_config_path.read_text()))
  with patch.object(llm, "generate", return_value=_fenced(bad_code)):
    status = gs.process_index(
      index=11, task=task, llm=llm, output_dir=out,
      base_seed=42, cfg=_fast_cfg(),
    )

  meta = json.loads((out / "model_000011.meta.json").read_text())
  assert status == "compile_error", f"expected compile_error, got {status}: {meta.get('reason')}"
  assert meta["status"] == "compile_error"
  assert "boom" in meta["reason"]
  assert meta["code"].startswith("import numpyro")


def test_missing_goal_records_syntax_error(coin_task, llm_config_path, tmp_path):
  """Code that never declares the goal site is caught as syntax_error at generation."""
  task, _ = coin_task
  out = tmp_path / "out"
  out.mkdir()

  bad_code = (
    "import numpyro\n"
    "import numpyro.distributions as dist\n"
    "def model(data):\n"
    "  numpyro.sample('not_the_target', dist.Beta(1.0, 1.0))\n"
  )
  llm = gs.build_llm(json.loads(llm_config_path.read_text()))
  with patch.object(llm, "generate", return_value=_fenced(bad_code)):
    status = gs.process_index(
      index=12, task=task, llm=llm, output_dir=out,
      base_seed=42, cfg=_fast_cfg(),
    )

  meta = json.loads((out / "model_000012.meta.json").read_text())
  assert status == "syntax_error"
  assert meta["status"] == "syntax_error"
  assert "missing goal names" in meta["reason"]


def test_evaluate_end_to_end(coin_task, llm_config_path, tmp_path):
  """Generate a handful of models, then evaluate.py produces metrics."""
  task, task_path = coin_task
  out = tmp_path / "samples"
  out.mkdir()

  llm = gs.build_llm(json.loads(llm_config_path.read_text()))

  def fake_generate(*args, **kwargs):
    seed = kwargs.get("seed")
    if seed is not None and seed % 2 == 0:
      return _fenced(COIN_MODEL_CODE)
    return _fenced(COIN_MODEL_CODE_SKEWED)

  with patch.object(llm, "generate", side_effect=fake_generate):
    for idx in range(4):
      gs.process_index(
        index=idx, task=task, llm=llm, output_dir=out,
        base_seed=100, cfg=_fast_cfg(),
      )

  jsonl_path = tmp_path / "metrics.jsonl"
  full_path = tmp_path / "metrics_full.json"

  rc = ev.main([
    "--task", str(task_path),
    "--sample-dir", str(out),
    "--llm-name", "mock",
    "--ks", "2", "4",
    "--output-jsonl", str(jsonl_path),
    "--output-full-json", str(full_path),
  ])
  assert rc == 0
  assert jsonl_path.exists()
  rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
  assert len(rows) == 2
  ks = [r["k_requested"] for r in rows]
  assert ks == [2, 4]

  for r in rows:
    assert r["task"] == "coin_flip"
    assert r["llm"] == "mock"
    assert r["k_used"] == r["k_requested"]
    assert r["n_datapoints"] == 6
    stats = r["weight_stats"]
    assert stats["entropy_uniform"] > 0
    assert 0.0 <= stats["l1_bma_stacking"] <= 2.0 + 1e-9
    t = r["targets"]["true_bias"]
    assert 0.0 <= t["posterior_mean_uniform"] <= 1.0
    assert 0.0 <= t["posterior_mean_bma"] <= 1.0
    assert 0.0 <= t["posterior_mean_stacking"] <= 1.0
    assert t["epistemic_var_uniform"] >= 0.0

  full = json.loads(full_path.read_text())
  assert "per_k" in full
  assert set(full["per_k"].keys()) == {"2", "4"}
  per_k4 = full["per_k"]["4"]
  assert len(per_k4["weights"]["stacking"]) == 4
  assert np.isclose(sum(per_k4["weights"]["stacking"]), 1.0, atol=1e-6)
  assert np.isclose(sum(per_k4["weights"]["bma"]), 1.0, atol=1e-6)


def test_solve_stacking_on_simple_matrix():
  """Stacking optimizer returns a probability simplex and assigns dominant mass to the best model."""
  n_data = 10
  loo = np.zeros((n_data, 3))
  loo[:, 0] = -1.0
  loo[:, 1] = -3.0
  loo[:, 2] = -5.0
  w = ev._solve_stacking(loo)
  assert w.shape == (3,)
  assert np.all(w >= 0.0)
  assert np.isclose(np.sum(w), 1.0)
  assert w[0] > 0.9
  assert w[1] < 0.05
  assert w[2] < 0.05


def test_solve_stacking_prefers_diverse_mix():
  """When two models dominate on different datapoints, stacking should mix them."""
  n_data = 20
  loo = np.full((n_data, 3), -10.0)
  loo[:10, 0] = -0.1
  loo[10:, 1] = -0.1
  w = ev._solve_stacking(loo)
  assert np.isclose(np.sum(w), 1.0)
  assert w[0] > 0.3 and w[1] > 0.3
  assert w[2] < 0.05


def test_softmax_robust_to_nonfinite():
  """_softmax_from_logs falls back to uniform when all entries are non-finite."""
  import math
  w = ev._softmax_from_logs([math.inf, math.nan, -math.inf])
  assert np.isclose(np.sum(w), 1.0)
  assert w.size == 3
