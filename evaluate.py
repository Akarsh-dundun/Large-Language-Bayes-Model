"""Evaluation driver for cached per-model PPL artifacts.

Globs the artifacts produced by ``generate_and_sample.py``, sweeps over K, and
computes uniform, BMA, and stacking weightings together with paper metrics:
epistemic variance per target, posterior means, weight entropy and ESS, the L1
distance and KL between weight schemes, and the final stacking objective.

Does not run MCMC. The expensive steps are assumed to already live on disk.
"""

import argparse
import glob
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


def _load_json(path):
  with open(path) as f:
    return json.load(f)


def _append_jsonl(path, record):
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  with open(path, "a") as f:
    f.write(json.dumps(record) + "\n")


def _softmax_from_logs(log_values):
  """Numerically stable softmax over a 1D log-score vector."""
  log_values = np.asarray(log_values, dtype=np.float64)
  finite = np.isfinite(log_values)
  if not np.any(finite):
    n = log_values.size
    return np.ones(n) / n
  shifted = np.where(finite, log_values - np.max(log_values[finite]), -np.inf)
  exps = np.where(finite, np.exp(shifted), 0.0)
  total = np.sum(exps)
  if total <= 0:
    n = log_values.size
    return np.ones(n) / n
  return exps / total


def _solve_stacking(loo_matrix, max_iter=2000, tol=1e-7):
  """Stacking weights on the simplex via exponentiated gradient (EG).

  Maximises ``mean_i log sum_k w_k exp(loo[i,k])`` subject to ``w >= 0`` and
  ``sum_k w_k = 1``. EG scales as O(n*K) per iteration, which keeps the
  solver tractable at K=10,000 unlike SLSQP (whose dense constraint Jacobian
  becomes prohibitive for K > a few hundred).

  ``loo_matrix`` has shape (n_datapoints, n_models).
  """
  loo_matrix = np.asarray(loo_matrix, dtype=np.float64)
  n_data, n_models = loo_matrix.shape
  if n_models == 0:
    return np.zeros(0, dtype=np.float64)
  if n_models == 1:
    return np.array([1.0])

  row_max = np.max(loo_matrix, axis=1, keepdims=True)
  exp_vals = np.exp(loo_matrix - row_max)  # (n_data, K)

  w = np.ones(n_models) / n_models
  prev_obj = -np.inf
  step = 1.0
  for it in range(max_iter):
    weighted = exp_vals @ w  # (n_data,)
    weighted = np.clip(weighted, 1e-300, None)
    grad = (exp_vals / weighted[:, None]).mean(axis=0)  # (K,)
    obj = float(np.mean(np.log(weighted) + row_max.squeeze(1)))

    if not np.isfinite(obj):
      break
    # EG update with normalization. Step size schedule: 1/sqrt(it+1)
    # plus a backtrack-style shrink if the objective decreased.
    log_w_new = np.log(np.clip(w, 1e-300, None)) + step * (grad - grad.max())
    log_w_new -= log_w_new.max()
    w_new = np.exp(log_w_new)
    s = w_new.sum()
    if s <= 0 or not np.isfinite(s):
      break
    w_new = w_new / s

    weighted_new = exp_vals @ w_new
    weighted_new = np.clip(weighted_new, 1e-300, None)
    obj_new = float(np.mean(np.log(weighted_new) + row_max.squeeze(1)))

    if obj_new < obj - 1e-9:
      step *= 0.5
      if step < 1e-9:
        break
      continue

    rel_imp = abs(obj_new - prev_obj) / max(1.0, abs(prev_obj))
    prev_obj = obj_new
    w = w_new
    if rel_imp < tol and it > 50:
      break

  w = np.maximum(w, 0.0)
  total = w.sum()
  if total <= 0:
    return np.ones(n_models) / n_models
  return w / total


def _stacking_objective(loo_matrix, weights):
  """Mean-over-datapoints log stacking score used as the paper objective."""
  if loo_matrix.shape[1] == 0:
    return float("nan")
  row_max = np.max(loo_matrix, axis=1)
  shifted = loo_matrix - row_max[:, None]
  weighted = np.exp(shifted) @ weights
  weighted = np.clip(weighted, 1e-300, None)
  return float(np.mean(row_max + np.log(weighted)))


def _weight_entropy(w):
  w = np.asarray(w, dtype=np.float64)
  safe = np.clip(w, 1e-12, None)
  return float(-np.sum(w * np.log(safe)))


def _ess(w):
  w = np.asarray(w, dtype=np.float64)
  return float(1.0 / np.sum(w * w))


def _kl(p, q):
  p = np.asarray(p, dtype=np.float64)
  q = np.asarray(q, dtype=np.float64)
  mask = p > 0
  if not np.any(mask):
    return 0.0
  return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(np.clip(q[mask], 1e-300, None)))))


@dataclass
class ModelArtifact:
  index: int
  status: str
  log_marginal: float
  loo_log_liks: np.ndarray  # may be zero-length if missing
  test_log_liks: np.ndarray  # may be zero-length if missing
  target_samples: dict      # name -> 1D array of per-sample means (scalar targets only for now)
  meta: dict
  sha: str = ""


def _load_one(meta_path, npz_path, target_names, index_hint=None):
  """Load a single per-model artifact. Returns None if unusable for weighting."""
  meta = _load_json(meta_path)
  status = meta.get("status")
  if status not in ("ok", "partial"):
    return None
  if not npz_path.exists():
    return None
  z = np.load(npz_path, allow_pickle=False)

  log_marginal_info = meta.get("log_marginal") or {}
  value = log_marginal_info.get("value")
  log_marginal = float(value) if value is not None and np.isfinite(value) else float("nan")

  loo_info = meta.get("loo") or {}
  if loo_info.get("status") == "ok" and "loo_log_liks" in z.files:
    loo = np.asarray(z["loo_log_liks"], dtype=np.float64)
  else:
    loo = np.zeros(0, dtype=np.float64)

  test_info = meta.get("test_scoring") or {}
  if test_info.get("status") == "ok" and "test_log_liks" in z.files:
    test_ll = np.asarray(z["test_log_liks"], dtype=np.float64)
  else:
    test_ll = np.zeros(0, dtype=np.float64)

  target_samples = {}
  for name in target_names:
    key = f"target__{name.replace('/', '__').replace(chr(92), '__')}"
    if key in z.files:
      target_samples[name] = np.asarray(z[key], dtype=np.float64)

  if "index" in meta:
    index = int(meta["index"])
  elif index_hint is not None:
    index = int(index_hint)
  else:
    index = 0

  return ModelArtifact(
    index=index,
    status=status,
    log_marginal=log_marginal,
    loo_log_liks=loo,
    test_log_liks=test_ll,
    target_samples=target_samples,
    meta=meta,
    sha=str(meta.get("sha", "")),
  )


def load_artifacts(sample_dir, target_names):
  """Load all per-model artifacts. Supports both legacy ``model_*.meta.json``
  and hash-addressed ``sample_*.meta.json`` layouts.
  """
  sample_dir = Path(sample_dir)
  metas = sorted(sample_dir.glob("model_*.meta.json"))
  hashed = False
  if not metas:
    metas = sorted(sample_dir.glob("sample_*.meta.json"))
    hashed = True
  artifacts = []
  for i, meta_path in enumerate(metas):
    npz_path = meta_path.with_suffix("").with_suffix(".npz")
    art = _load_one(meta_path, npz_path, target_names, index_hint=i if hashed else None)
    if art is not None:
      artifacts.append(art)
  if hashed:
    artifacts.sort(key=lambda a: a.sha)
  else:
    artifacts.sort(key=lambda a: a.index)
  return artifacts


def _weighted_between_model_variance(weights, mu_per_model):
  """Bessel-corrected weighted between-model variance. mu_per_model is 1D len K."""
  mu = np.asarray(mu_per_model, dtype=np.float64)
  w = np.asarray(weights, dtype=np.float64)
  mean = float(np.sum(w * mu))
  S = float(np.sum(w * (mu - mean) ** 2))
  C = 1.0 - float(np.sum(w * w))
  if C <= 0:
    return 0.0, mean
  return S / C, mean


def _target_mean_per_model(artifact, target):
  arr = artifact.target_samples.get(target)
  if arr is None:
    return float("nan")
  return float(np.mean(arr))


def _test_log_predictive(weights, test_log_liks_matrix):
  """Mean-over-test-points held-out log predictive density for a weighted mixture.

  Args:
    weights: (K,) probability vector summing to 1.
    test_log_liks_matrix: (n_test, K) per-(test_point, model) log p(x_test_i | x_train, m).
  Returns the per-test mean of log sum_k w_k p(x_test_i | x_train, m_k).
  """
  if test_log_liks_matrix.size == 0 or test_log_liks_matrix.shape[1] == 0:
    return float("nan")
  row_max = np.max(test_log_liks_matrix, axis=1)
  shifted = test_log_liks_matrix - row_max[:, None]
  weighted = np.exp(shifted) @ weights
  weighted = np.clip(weighted, 1e-300, None)
  return float(np.mean(row_max + np.log(weighted)))


def _renorm_weights(w, mask):
  """Slice w[mask] and renormalize to sum to 1, falling back to uniform."""
  w_sub = np.asarray(w, dtype=np.float64)[mask]
  s = float(np.sum(w_sub))
  if s <= 0:
    n = w_sub.size
    return np.ones(n) / n if n > 0 else w_sub
  return w_sub / s


def _bootstrap_stacking_weights(loo_matrix, n_bootstrap=200, seed=0):
  """Bootstrap stacking weights by resampling rows (training points) with replacement.

  Returns (B, K) array of weights for each bootstrap.
  """
  rng = np.random.default_rng(seed)
  n_data, n_models = loo_matrix.shape
  if n_models <= 0 or n_data <= 0:
    return np.zeros((0, n_models))
  out = np.empty((n_bootstrap, n_models), dtype=np.float64)
  for b in range(n_bootstrap):
    idx = rng.integers(0, n_data, size=n_data)
    out[b] = _solve_stacking(loo_matrix[idx])
  return out


def evaluate_for_k(artifacts, k, targets):
  """Slice to first K usable models and compute all weightings + metrics."""
  pool = artifacts[:k]
  if len(pool) == 0:
    return None

  log_bounds = np.array([a.log_marginal for a in pool], dtype=np.float64)
  finite_marginal = np.isfinite(log_bounds)
  n_bma = int(np.sum(finite_marginal))

  loo_lengths = [a.loo_log_liks.size for a in pool]
  n_data = 0
  for length in loo_lengths:
    if length > 0:
      n_data = length
      break
  loo_mask = np.array([length == n_data and length > 0 for length in loo_lengths], dtype=bool)
  n_loo = int(np.sum(loo_mask))

  k_eff = len(pool)
  uniform = np.ones(k_eff) / k_eff

  bma = np.zeros(k_eff)
  if n_bma > 0:
    bma_sub = _softmax_from_logs(log_bounds[finite_marginal])
    bma[finite_marginal] = bma_sub
  else:
    bma[:] = uniform

  stacking = np.zeros(k_eff)
  stacking_objective = float("nan")
  if n_loo >= 2:
    loo_matrix = np.column_stack([pool[i].loo_log_liks for i in range(k_eff) if loo_mask[i]])
    w_sub = _solve_stacking(loo_matrix)
    stacking[loo_mask] = w_sub
    total = np.sum(stacking)
    if total > 0:
      stacking = stacking / total
    stacking_objective = _stacking_objective(loo_matrix, w_sub)
  elif n_loo == 1:
    j = int(np.where(loo_mask)[0][0])
    stacking[j] = 1.0
  else:
    stacking[:] = uniform

  # Test-set predictive density when test_log_liks are available.
  test_log_liks_lengths = [a.test_log_liks.size for a in pool]
  n_test = 0
  for length in test_log_liks_lengths:
    if length > 0:
      n_test = length
      break
  test_mask = np.array(
    [length == n_test and length > 0 for length in test_log_liks_lengths], dtype=bool
  )
  n_test_models = int(np.sum(test_mask))

  test_metrics = {
    "n_test": int(n_test),
    "n_models_with_test": n_test_models,
  }
  if n_test_models >= 1 and n_test > 0:
    test_matrix = np.column_stack(
      [pool[i].test_log_liks for i in range(k_eff) if test_mask[i]]
    )
    w_uni = _renorm_weights(uniform, test_mask)
    w_bma = _renorm_weights(bma, test_mask)
    w_st = _renorm_weights(stacking, test_mask)
    test_metrics.update({
      "log_predictive_uniform": _test_log_predictive(w_uni, test_matrix),
      "log_predictive_bma": _test_log_predictive(w_bma, test_matrix),
      "log_predictive_stacking": _test_log_predictive(w_st, test_matrix),
    })
  else:
    test_metrics.update({
      "log_predictive_uniform": None,
      "log_predictive_bma": None,
      "log_predictive_stacking": None,
    })

  # Bootstrap stacking objective + weight ESS spread by resampling training points.
  bootstrap = {"n_bootstrap": 0}
  if n_loo >= 2:
    loo_matrix = np.column_stack([pool[i].loo_log_liks for i in range(k_eff) if loo_mask[i]])
    boot_weights = _bootstrap_stacking_weights(loo_matrix, n_bootstrap=200, seed=k)
    if boot_weights.size > 0:
      ess_boot = 1.0 / np.sum(boot_weights * boot_weights, axis=1)
      max_boot = np.max(boot_weights, axis=1)
      bootstrap = {
        "n_bootstrap": int(boot_weights.shape[0]),
        "ess_stacking_p05": float(np.percentile(ess_boot, 5)),
        "ess_stacking_p50": float(np.percentile(ess_boot, 50)),
        "ess_stacking_p95": float(np.percentile(ess_boot, 95)),
        "max_stacking_p05": float(np.percentile(max_boot, 5)),
        "max_stacking_p50": float(np.percentile(max_boot, 50)),
        "max_stacking_p95": float(np.percentile(max_boot, 95)),
      }

  metrics = {
    "k_requested": int(k),
    "k_used": int(k_eff),
    "n_bma_finite": n_bma,
    "n_loo_valid": n_loo,
    "n_datapoints": int(n_data),
    "test_set": test_metrics,
    "bootstrap": bootstrap,
    "weights": {
      "uniform": uniform.tolist(),
      "bma": bma.tolist(),
      "stacking": stacking.tolist(),
    },
    "weight_stats": {
      "entropy_uniform": _weight_entropy(uniform),
      "entropy_bma": _weight_entropy(bma),
      "entropy_stacking": _weight_entropy(stacking),
      "ess_uniform": _ess(uniform),
      "ess_bma": _ess(bma),
      "ess_stacking": _ess(stacking),
      "max_uniform": float(np.max(uniform)),
      "max_bma": float(np.max(bma)),
      "max_stacking": float(np.max(stacking)),
      "l1_bma_stacking": float(np.sum(np.abs(bma - stacking))),
      "kl_bma_given_stacking": _kl(bma, stacking),
      "kl_stacking_given_bma": _kl(stacking, bma),
    },
    "stacking_objective": stacking_objective,
    "log_marginal_distribution": {
      "min": float(np.min(log_bounds[finite_marginal])) if n_bma > 0 else None,
      "max": float(np.max(log_bounds[finite_marginal])) if n_bma > 0 else None,
      "mean": float(np.mean(log_bounds[finite_marginal])) if n_bma > 0 else None,
      "std": float(np.std(log_bounds[finite_marginal])) if n_bma > 0 else None,
    },
    "targets": {},
  }

  for target in targets:
    mu_per_model = np.array(
      [_target_mean_per_model(a, target) for a in pool], dtype=np.float64
    )
    keep = np.isfinite(mu_per_model)
    if not np.any(keep):
      metrics["targets"][target] = None
      continue

    mu_k = mu_per_model[keep]

    def _renorm(w, mask):
      w_sub = w[mask]
      s = float(np.sum(w_sub))
      if s <= 0:
        n = w_sub.size
        return np.ones(n) / n if n > 0 else w_sub
      return w_sub / s

    w_u = _renorm(uniform, keep)
    w_b = _renorm(bma, keep)
    w_s = _renorm(stacking, keep)

    var_u, mean_u = _weighted_between_model_variance(w_u, mu_k)
    var_b, mean_b = _weighted_between_model_variance(w_b, mu_k)
    var_s, mean_s = _weighted_between_model_variance(w_s, mu_k)

    metrics["targets"][target] = {
      "posterior_mean_uniform": mean_u,
      "posterior_mean_bma": mean_b,
      "posterior_mean_stacking": mean_s,
      "epistemic_var_uniform": var_u,
      "epistemic_var_bma": var_b,
      "epistemic_var_stacking": var_s,
      "n_models_with_target": int(np.sum(keep)),
    }

  return metrics


def _default_ks(n_available):
  """Return the plan's sweep list intersected with what's actually on disk."""
  candidates = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
  ks = [k for k in candidates if k <= n_available]
  if n_available > 0 and (not ks or ks[-1] != n_available):
    ks.append(n_available)
  return sorted(set(ks))


def _build_parser():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--task", required=True, help="Task spec JSON (for targets and name)")
  p.add_argument("--sample-dir", required=True, help="Directory with model_*.npz+meta.json artifacts")
  p.add_argument("--llm-name", required=True, help="Label for the LLM in result rows")
  p.add_argument(
    "--ks", type=int, nargs="*", default=None,
    help="K values to sweep; defaults to a log-spaced list up to n_available",
  )
  p.add_argument(
    "--output-jsonl", default="experiment_results/metrics.jsonl",
    help="Append one row per (task, llm, K) to this JSONL file",
  )
  p.add_argument(
    "--output-full-json", default=None,
    help="If set, also dump the full per-K metrics (including weights) to this JSON",
  )
  return p


def main(argv=None):
  args = _build_parser().parse_args(argv)
  task = _load_json(args.task)
  targets = task.get("targets") or []
  artifacts = load_artifacts(args.sample_dir, targets)
  n_available = len(artifacts)
  print(f"Loaded {n_available} usable artifacts from {args.sample_dir}")

  if n_available == 0:
    print("No artifacts to evaluate.")
    return 0

  ks = args.ks if args.ks else _default_ks(n_available)
  print(f"Sweeping K values: {ks}")

  run_id = time.strftime("%Y-%m-%dT%H-%M-%S")
  full_by_k = {}
  for k in ks:
    t0 = time.time()
    metrics = evaluate_for_k(artifacts, k, targets)
    dt = time.time() - t0
    if metrics is None:
      continue
    row = {
      "run_id": run_id,
      "task": task.get("name"),
      "llm": args.llm_name,
      "sample_dir": str(args.sample_dir),
      "k_requested": metrics["k_requested"],
      "k_used": metrics["k_used"],
      "n_bma_finite": metrics["n_bma_finite"],
      "n_loo_valid": metrics["n_loo_valid"],
      "n_datapoints": metrics["n_datapoints"],
      "stacking_objective": metrics["stacking_objective"],
      "weight_stats": metrics["weight_stats"],
      "log_marginal_distribution": metrics["log_marginal_distribution"],
      "targets": metrics["targets"],
      "eval_seconds": dt,
    }
    _append_jsonl(args.output_jsonl, row)
    full_by_k[k] = metrics
    stats = metrics["weight_stats"]
    print(
      f"K={k:>6}  bma_fin={metrics['n_bma_finite']:>5}  loo_ok={metrics['n_loo_valid']:>5}  "
      f"ess_bma={stats['ess_bma']:7.2f}  ess_stack={stats['ess_stacking']:7.2f}  "
      f"l1={stats['l1_bma_stacking']:.4f}  obj={metrics['stacking_objective']:.4f}  "
      f"dt={dt:.1f}s"
    )

  if args.output_full_json is not None:
    Path(args.output_full_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_full_json, "w") as f:
      json.dump(
        {
          "run_id": run_id,
          "task": task.get("name"),
          "llm": args.llm_name,
          "sample_dir": str(args.sample_dir),
          "per_k": {str(k): v for k, v in full_by_k.items()},
        },
        f,
        indent=2,
      )
    print(f"Wrote {args.output_full_json}")
  print(f"Appended rows to {args.output_jsonl}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
