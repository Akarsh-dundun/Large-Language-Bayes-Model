"""Stage A dedup, resume, and target-respect tests with a mocked LLM."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import generate_codes as gc
from llb.codes import (
  canonicalize_code,
  code_hash,
  code_path,
  count_distinct_codes,
)


GOOD_CODE_A = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["events"], dtype=jnp.int32)
    rate = numpyro.sample("rate", dist.Gamma(2.0, 0.05))
    numpyro.sample("obs", dist.Poisson(rate), obs=y)
"""

# Same code as A, just with extra comments and whitespace; canonical hash must match.
GOOD_CODE_A_VARIANT = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

# A different comment
def model(data):
    y = jnp.asarray(data["events"], dtype=jnp.int32)
    rate = numpyro.sample("rate", dist.Gamma(2.0, 0.05))   # rate prior
    numpyro.sample("obs", dist.Poisson(rate), obs=y)
"""

GOOD_CODE_B = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["events"], dtype=jnp.int32)
    rate = numpyro.sample("rate", dist.Exponential(0.05))
    numpyro.sample("obs", dist.Poisson(rate), obs=y)
"""

BAD_CODE_NO_GOAL = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["events"], dtype=jnp.int32)
    not_rate = numpyro.sample("not_rate", dist.Gamma(2.0, 0.05))
    numpyro.sample("obs", dist.Poisson(not_rate), obs=y)
"""


def _fenced(code):
  return f"```python\n{code.strip()}\n```"


def _make_task(tmp_path):
  task = {
    "name": "fake",
    "text": "fake task",
    "data": {"events": [1, 2, 3, 4, 5]},
    "test_data": None,
    "targets": ["rate"],
    "true_latents": None,
    "task_type": "estimation",
    "metadata": {},
  }
  path = tmp_path / "task.json"
  path.write_text(json.dumps(task))
  return task, path


def _make_llm_config(tmp_path):
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


def test_canonical_hash_collapses_comments():
  ha = code_hash(canonicalize_code(GOOD_CODE_A))
  hb = code_hash(canonicalize_code(GOOD_CODE_A_VARIANT))
  assert ha == hb, "AST canonicalization must collapse comments and whitespace"
  assert ha != code_hash(canonicalize_code(GOOD_CODE_B))


def _make_llm(tmp_path):
  return gc.build_llm(json.loads(_make_llm_config(tmp_path).read_text()))


def test_dedup_via_o_excl(tmp_path):
  task, task_path = _make_task(tmp_path)
  cell_dir = tmp_path / "cell"
  llm = _make_llm(tmp_path)

  responses = [_fenced(GOOD_CODE_A), _fenced(GOOD_CODE_A_VARIANT), _fenced(GOOD_CODE_B)]
  with patch.object(llm, "generate", side_effect=responses * 10):
    counts = gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=0, shard_max_indices=3,
      shard_valid_target=10, cell_valid_target=10, seed=42, llm=llm,
    )

  assert counts["new"] == 2, "two distinct hashes expected, the second variant must dedup"
  assert counts["duplicate"] == 1
  assert count_distinct_codes(cell_dir) == 2


def test_shard_valid_target_respected(tmp_path):
  task, task_path = _make_task(tmp_path)
  cell_dir = tmp_path / "cell"
  llm = _make_llm(tmp_path)

  with patch.object(llm, "generate", return_value=_fenced(GOOD_CODE_A)):
    counts = gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=0, shard_max_indices=20,
      shard_valid_target=1, cell_valid_target=1000, seed=42, llm=llm,
    )

  assert counts["new"] == 1, "shard should stop as soon as one new hash is claimed"


def test_cell_valid_target_respected(tmp_path):
  task, task_path = _make_task(tmp_path)
  cell_dir = tmp_path / "cell"
  llm = _make_llm(tmp_path)

  with patch.object(llm, "generate", return_value=_fenced(GOOD_CODE_A)):
    gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=0, shard_max_indices=1,
      shard_valid_target=1, cell_valid_target=1000, seed=42, llm=llm,
    )

  assert count_distinct_codes(cell_dir) == 1

  with patch.object(llm, "generate", side_effect=AssertionError("LLM should not be called")):
    counts2 = gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=10, shard_max_indices=5,
      shard_valid_target=10, cell_valid_target=1, seed=42, llm=llm,
    )

  assert counts2.get("new", 0) == 0
  assert count_distinct_codes(cell_dir) == 1


def test_resume_skips_done_slots(tmp_path):
  task, task_path = _make_task(tmp_path)
  cell_dir = tmp_path / "cell"
  llm = _make_llm(tmp_path)

  with patch.object(llm, "generate", return_value=_fenced(GOOD_CODE_A)):
    gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=0, shard_max_indices=2,
      shard_valid_target=10, cell_valid_target=10, seed=42, llm=llm,
    )

  with patch.object(llm, "generate", side_effect=AssertionError("should not call LLM on resume")) as mock2:
    gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=0, shard_max_indices=2,
      shard_valid_target=10, cell_valid_target=10, seed=42, llm=llm,
    )
    assert mock2.call_count == 0


def test_failure_recorded_for_missing_goal(tmp_path):
  task, task_path = _make_task(tmp_path)
  cell_dir = tmp_path / "cell"
  llm = _make_llm(tmp_path)

  with patch.object(llm, "generate", return_value=_fenced(BAD_CODE_NO_GOAL)):
    counts = gc.run_shard(
      task_path=task_path, llm_config_path=None, cell_dir=cell_dir,
      shard_start=0, shard_max_indices=1,
      shard_valid_target=1, cell_valid_target=1, seed=42, llm=llm,
    )
  assert counts.get("syntax_error", 0) == 1
  assert count_distinct_codes(cell_dir) == 0
  failures = list((cell_dir / "codes" / "_failures").glob("failure_*.json"))
  assert len(failures) == 1
