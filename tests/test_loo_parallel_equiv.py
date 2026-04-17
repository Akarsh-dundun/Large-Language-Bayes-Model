"""Sequential vs multi-process true-LOO numerically agree up to MC noise."""
from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from llb.mcmc_log import (
  estimate_loo_log_likelihoods,
  estimate_loo_log_likelihoods_parallel,
  run_inference,
)


CODE = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["y"], dtype=jnp.float32)
    mu = numpyro.sample("mu", dist.Normal(0.0, 5.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(2.0))
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
"""


def test_loo_parallel_matches_serial_up_to_mc_noise():
  data = {"y": [0.1, 0.5, -0.2, 1.1, 0.7, -0.4, 0.9, 0.0]}
  infer = run_inference(CODE, data, targets=None,
                        num_warmup=50, num_samples=100, rng_seed=0)

  serial = estimate_loo_log_likelihoods(
    model=infer["model"], data=data,
    posterior_samples=infer["samples"],
    num_inner=8, num_warmup=20, num_samples=40,
    rng_seed=123, use_true_loo=True, return_diagnostics=False, verbose=False,
  )

  parallel = estimate_loo_log_likelihoods_parallel(
    code=CODE, data=data,
    posterior_samples=infer["samples"],
    num_inner=8, num_warmup=20, num_samples=40,
    rng_seed=123, n_workers=2, return_diagnostics=False,
  )

  assert serial.shape == parallel.shape == (8,)
  # Both estimators are noisy MC bounds, so we only check they are in the same
  # ballpark on average.
  diff = np.abs(serial - parallel)
  finite = diff[np.isfinite(diff)]
  assert finite.size > 0
  assert np.median(finite) < 5.0, (
    f"serial vs parallel LOO disagree by median {np.median(finite):.2f}; "
    f"serial={serial}, parallel={parallel}"
  )


def test_parallel_falls_back_to_serial_when_one_worker():
  data = {"y": [0.1, 0.5, -0.2, 1.1]}
  infer = run_inference(CODE, data, targets=None,
                        num_warmup=20, num_samples=40, rng_seed=0)
  result = estimate_loo_log_likelihoods_parallel(
    code=CODE, data=data,
    posterior_samples=infer["samples"],
    num_inner=4, num_warmup=10, num_samples=20,
    rng_seed=7, n_workers=1, return_diagnostics=True,
  )
  assert isinstance(result, dict)
  assert "loo_log_liks" in result
  assert result["loo_log_liks"].shape == (4,)
