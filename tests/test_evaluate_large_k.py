"""Stacking solver scales to K=10000 within a reasonable time budget."""
from __future__ import annotations

import time

import numpy as np
import pytest

import evaluate as ev


def test_solve_stacking_at_k10000_finishes_within_budget():
  rng = np.random.default_rng(0)
  n_data = 20
  k = 10_000
  # Synthetic LOO matrix: most models around -3, a few around -1.
  loo = rng.normal(loc=-3.0, scale=1.0, size=(n_data, k))
  loo[:, :50] = rng.normal(loc=-1.0, scale=0.5, size=(n_data, 50))

  t0 = time.time()
  w = ev._solve_stacking(loo)
  elapsed = time.time() - t0

  assert w.shape == (k,)
  assert np.all(w >= -1e-9)
  assert abs(np.sum(w) - 1.0) < 1e-6
  assert elapsed < 120.0, f"K=10000 stacking took {elapsed:.1f}s, expected < 120s"


def test_bootstrap_stacking_returns_simplex():
  rng = np.random.default_rng(1)
  loo = rng.normal(size=(20, 50))
  boot = ev._bootstrap_stacking_weights(loo, n_bootstrap=10, seed=0)
  assert boot.shape == (10, 50)
  for row in boot:
    assert abs(row.sum() - 1.0) < 1e-6
    assert np.all(row >= -1e-9)


def test_test_log_predictive_handles_empty():
  empty = np.zeros((0, 0))
  assert np.isnan(ev._test_log_predictive(np.array([]), empty))
