"""Sanity checks on held-out test scoring.

A single-parameter Normal model with known posterior mean lets us sanity-check
that ``estimate_test_log_likelihoods`` gives a value close to the expected log
predictive density on a held-out point.
"""
from __future__ import annotations

import numpy as np
import pytest

from llb.mcmc_log import (
  estimate_test_log_likelihoods,
  run_inference,
)


CODE = """
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(data):
    y = jnp.asarray(data["y"], dtype=jnp.float32)
    mu = numpyro.sample("mu", dist.Normal(0.0, 10.0))
    numpyro.sample("obs", dist.Normal(mu, 1.0), obs=y)
"""


def test_estimate_test_log_likelihoods_basic():
  train = {"y": [0.1, 0.2, -0.1, 0.0, 0.3, -0.2, 0.1, 0.0, 0.05, -0.05]}
  test = {"y": [0.0, 5.0]}  # in-distribution and out-of-distribution

  out = run_inference(CODE, train, targets=None,
                      num_warmup=200, num_samples=400, rng_seed=0)
  test_ll, diag = estimate_test_log_likelihoods(
    model=out["model"], train_data=train, test_data=test,
    posterior_samples=out["samples"], rng_seed=0,
  )

  assert test_ll.shape == (2,)
  assert diag.get("method") == "test_predictive_lme"
  assert np.all(np.isfinite(test_ll))
  # In-distribution point should score much higher than the outlier.
  assert test_ll[0] > test_ll[1] + 5.0


def test_estimate_test_log_likelihoods_skipped_when_test_none():
  train = {"y": [0.1, 0.2]}
  out = run_inference(CODE, train, targets=None,
                      num_warmup=20, num_samples=40, rng_seed=0)
  test_ll, diag = estimate_test_log_likelihoods(
    model=out["model"], train_data=train, test_data=None,
    posterior_samples=out["samples"],
  )
  assert test_ll.shape == (0,)
  assert diag.get("method") == "skipped"
