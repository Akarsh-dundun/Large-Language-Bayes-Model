import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace, substitute
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

try:
    import arviz as az
    import xarray as xr
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False


def estimate_loo_log_likelihoods(
    model,
    data,
    posterior_samples,
    num_inner=25,
    num_warmup=100,
    num_samples=200,
    rng_seed=0,
    min_std=1e-4,
    fallback_log_bound=-1e12,
    use_true_loo=True,
    return_diagnostics=False,
    verbose=True,
):
    """
    Compute leave-one-out log predictive densities.

    Args:
        use_true_loo: If True, use true LOO-ELBO (expensive).
                      If False, use PSIS-LOO approximation (fast).
        return_diagnostics: If True, return dict with diagnostics.
        verbose: If True, print per-datapoint progress; if False, run silently.

    Returns:
        If return_diagnostics=False: np.array of LOO log likelihoods
        If return_diagnostics=True: dict with 'loo_log_liks' and 'diagnostics'
    """
    if use_true_loo:
        result = _estimate_loo_true(
            model, data, posterior_samples,
            num_inner, num_warmup, num_samples,
            rng_seed, min_std, fallback_log_bound,
            verbose=verbose,
        )
    else:
        result = _estimate_loo_psis(
            model, data, posterior_samples,
            rng_seed, fallback_log_bound
        )

    if return_diagnostics:
        return result
    else:
        return result['loo_log_liks'] if isinstance(result, dict) else result


def estimate_loo_log_likelihoods_parallel(
    code,
    data,
    posterior_samples,
    num_inner=25,
    num_warmup=100,
    num_samples=200,
    rng_seed=0,
    min_std=1e-4,
    fallback_log_bound=-1e12,
    n_workers=4,
    return_diagnostics=False,
):
    """Parallel true-LOO across held-out indices using a process pool.

    JAX-traced numpyro models do not always pickle; each worker re-execs the
    code string to get a fresh model. This is safe and reproducible because
    every worker also re-derives its per-index PRNG seed from rng_seed.

    Falls back to the serial in-process loop when ``n_workers <= 1``.
    """
    n_datapoints = _get_num_datapoints(data)
    if n_workers is None or n_workers <= 1 or n_datapoints <= 1:
        env = {}
        exec(code, env)
        model = env["model"]
        return estimate_loo_log_likelihoods(
            model=model,
            data=data,
            posterior_samples=posterior_samples,
            num_inner=num_inner,
            num_warmup=num_warmup,
            num_samples=num_samples,
            rng_seed=rng_seed,
            min_std=min_std,
            fallback_log_bound=fallback_log_bound,
            use_true_loo=True,
            return_diagnostics=return_diagnostics,
            verbose=False,
        )

    import multiprocessing as mp

    # Posterior samples often contain jax DeviceArrays; coerce to numpy for
    # cheap pickling across worker processes.
    posterior_for_worker = {name: np.asarray(arr) for name, arr in posterior_samples.items()}
    data_for_worker = _to_picklable(data)

    n_workers = min(int(n_workers), n_datapoints)
    args = [
        (
            i, code, data_for_worker, posterior_for_worker,
            num_inner, num_warmup, num_samples,
            int(rng_seed) + i, float(min_std), float(fallback_log_bound),
        )
        for i in range(n_datapoints)
    ]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_loo_worker, args)

    loo_log_liks = np.array([r["loo_log_lik"] for r in results], dtype=np.float64)
    elbo_histories = [r["elbo_history"] for r in results]

    out = {
        "loo_log_liks": loo_log_liks,
        "diagnostics": {
            "method": "true_loo_elbo_parallel",
            "elbo_histories": elbo_histories,
            "n_datapoints": n_datapoints,
            "num_inner": num_inner,
            "num_warmup": num_warmup,
            "num_samples": num_samples,
            "n_workers": n_workers,
        },
    }
    return out if return_diagnostics else loo_log_liks


def _loo_worker(args):
    """Compute LOO log lik for one held-out index in a fresh worker process."""
    (
        i, code, data, posterior_samples,
        num_inner, num_warmup, num_samples,
        rng_seed, min_std, fallback_log_bound,
    ) = args
    try:
        env = {}
        exec(code, env)
        model = env["model"]
        loo_data = _create_loo_dataset(data, i)

        kernel = NUTS(model)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(rng_seed), data=loo_data)
        loo_posterior_samples = mcmc.get_samples(group_by_chain=False)

        means = {}
        stds = {}
        for name, values in loo_posterior_samples.items():
            arr = np.asarray(values, dtype=np.float64)
            mean, std = _finite_mean_std_axis0(arr, min_std=min_std)
            means[name] = mean
            stds[name] = std

        rng = np.random.default_rng(rng_seed + 1_000_000)
        elbo_estimates = []
        for _ in range(num_inner):
            z = {}
            log_q = 0.0
            for name in loo_posterior_samples:
                sample = rng.normal(loc=means[name], scale=stds[name])
                z[name] = jnp.asarray(sample)
                centered = (sample - means[name]) / stds[name]
                log_q += float(
                    -0.5 * np.sum(
                        centered * centered
                        + np.log(2.0 * np.pi)
                        + 2.0 * np.log(stds[name])
                    )
                )
            try:
                log_lik_i = _compute_pointwise_log_likelihood(model, data, z, i)
                log_prior_loo, _ = log_density(model, (), {"data": loo_data}, z)
                elbo_term = log_lik_i + float(log_prior_loo) - log_q
                if np.isfinite(elbo_term):
                    elbo_estimates.append(elbo_term)
            except Exception:
                continue

        if elbo_estimates:
            return {"loo_log_lik": float(np.mean(elbo_estimates)),
                    "elbo_history": list(elbo_estimates)}
        return {"loo_log_lik": fallback_log_bound, "elbo_history": []}
    except Exception:
        return {"loo_log_lik": fallback_log_bound, "elbo_history": []}


def _to_picklable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (jnp.ndarray,)):
            out[k] = np.asarray(v)
        elif isinstance(v, np.ndarray):
            out[k] = v
        elif isinstance(v, list):
            out[k] = list(v)
        else:
            out[k] = v
    return out


def _estimate_loo_true(
    model, data, posterior_samples,
    num_inner, num_warmup, num_samples,
    rng_seed, min_std, fallback_log_bound,
    verbose=True,
):
    """
    True LOO-ELBO implementation (Algorithm 2).
    Returns dict with loo_log_liks and diagnostics.
    """
    n_datapoints = _get_num_datapoints(data)
    loo_log_liks = []
    elbo_histories = []  # Track ELBO convergence for each datapoint

    if verbose:
        print(f"Computing TRUE LOO-ELBO for {n_datapoints} datapoints (this will take a while)...")

    for i in range(n_datapoints):
        if verbose:
            print(f"  LOO for datapoint {i+1}/{n_datapoints}...", end=" ")
        
        try:
            # Step 1: Create LOO dataset
            loo_data = _create_loo_dataset(data, i)
            
            # Step 2: Run MCMC on LOO data
            kernel = NUTS(model)
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                progress_bar=False
            )
            mcmc.run(jax.random.PRNGKey(rng_seed + i), data=loo_data)
            loo_posterior_samples = mcmc.get_samples(group_by_chain=False)
            
            # Step 3: Fit proposal q(z)
            means = {}
            stds = {}
            for name, values in loo_posterior_samples.items():
                arr = np.asarray(values, dtype=np.float64)
                mean, std = _finite_mean_std_axis0(arr, min_std=min_std)
                means[name] = mean
                stds[name] = std
            
            # Step 4: Compute ELBO with tracking
            rng = np.random.default_rng(rng_seed + 1000 + i)
            elbo_estimates = []
            
            for iter_idx in range(num_inner):
                z = {}
                log_q = 0.0
                
                # Sample z ~ q(z)
                for name in loo_posterior_samples:
                    sample = rng.normal(loc=means[name], scale=stds[name])
                    z[name] = jnp.asarray(sample)
                    
                    # Compute log q(z)
                    centered = (sample - means[name]) / stds[name]
                    log_q += float(
                        -0.5 * np.sum(
                            centered * centered
                            + np.log(2.0 * np.pi)
                            + 2.0 * np.log(stds[name])
                        )
                    )
                
                try:
                    # log p(x_i | z)
                    log_lik_i = _compute_pointwise_log_likelihood(model, data, z, i)
                    
                    # log p(z | x_{-i})
                    log_prior_loo, _ = log_density(model, (), {"data": loo_data}, z)
                    
                    # ELBO = log p(x_i, z | x_{-i}) - log q(z)
                    log_p_joint = log_lik_i + float(log_prior_loo)
                    elbo_term = log_p_joint - log_q
                    
                    if np.isfinite(elbo_term):
                        elbo_estimates.append(elbo_term)
                        
                except Exception:
                    continue
            
            # Step 5: Average ELBO
            if len(elbo_estimates) > 0:
                elbo_i = np.mean(elbo_estimates)
                loo_log_liks.append(elbo_i)
                elbo_histories.append(elbo_estimates)
                if verbose:
                    print(f"ELBO = {elbo_i:.4f} (±{np.std(elbo_estimates):.4f})")
            else:
                loo_log_liks.append(fallback_log_bound)
                elbo_histories.append([])
                if verbose:
                    print("FAILED (using fallback)")

        except Exception as e:
            if verbose:
                print(f"ERROR: {e}")
            loo_log_liks.append(fallback_log_bound)
            elbo_histories.append([])
    
    return {
        'loo_log_liks': np.array(loo_log_liks, dtype=np.float64),
        'diagnostics': {
            'method': 'true_loo_elbo',
            'elbo_histories': elbo_histories,
            'n_datapoints': n_datapoints,
            'num_inner': num_inner,
            'num_warmup': num_warmup,
            'num_samples': num_samples,
        }
    }


def _estimate_loo_psis(model, data, posterior_samples, rng_seed, fallback_log_bound):
    """
    PSIS-LOO approximation using arviz.
    Returns dict with loo_log_liks and diagnostics.
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError(
            "arviz is required for PSIS-LOO. Install with: pip install arviz\n"
            "Or use use_true_loo=True for the exact (but slower) method."
        )
    
    n_datapoints = _get_num_datapoints(data)
    n_samples = len(next(iter(posterior_samples.values())))
    
    print(f"Computing PSIS-LOO for {n_datapoints} datapoints...")
    
    # Compute pointwise log likelihoods
    log_lik_matrix = np.zeros((n_samples, n_datapoints))
    
    for s in range(n_samples):
        z = {name: jnp.asarray(posterior_samples[name][s]) 
             for name in posterior_samples}
        
        for i in range(n_datapoints):
            try:
                log_lik_i = _compute_pointwise_log_likelihood(model, data, z, i)
                if np.isfinite(log_lik_i) and log_lik_i > -1e10:
                    log_lik_matrix[s, i] = log_lik_i
                else:
                    log_lik_matrix[s, i] = fallback_log_bound
            except Exception:
                log_lik_matrix[s, i] = fallback_log_bound
    
    # Convert to xarray for arviz
    log_likelihood = xr.DataArray(
        log_lik_matrix,
        dims=["draw", "obs_id"],
        coords={"draw": np.arange(n_samples), "obs_id": np.arange(n_datapoints)}
    )
    
    # Compute PSIS-LOO
    try:
        loo_result = az.loo(log_likelihood, pointwise=True)
        loo_log_liks = loo_result.loo_i.values
        
        pareto_k = loo_result.pareto_k.values if hasattr(loo_result, 'pareto_k') else None
        warning = loo_result.warning if hasattr(loo_result, 'warning') else False
        
        print(f"  PSIS-LOO complete. Warning: {warning}")
        if pareto_k is not None:
            print(f"  Pareto k range: [{pareto_k.min():.3f}, {pareto_k.max():.3f}]")
            n_bad = np.sum(pareto_k > 0.7)
            if n_bad > 0:
                print(f"  ⚠️  {n_bad}/{n_datapoints} points have k > 0.7 (unreliable)")
        
        return {
            'loo_log_liks': np.array(loo_log_liks, dtype=np.float64),
            'diagnostics': {
                'method': 'psis_loo',
                'pareto_k': pareto_k,
                'warning': warning,
                'n_samples': n_samples,
                'n_datapoints': n_datapoints,
            }
        }
        
    except Exception as e:
        print(f"PSIS-LOO failed: {e}, using fallback")
        return {
            'loo_log_liks': np.full(n_datapoints, fallback_log_bound, dtype=np.float64),
            'diagnostics': {
                'method': 'psis_loo_failed',
                'error': str(e),
            }
        }


def _create_loo_dataset(data, leave_out_idx):
    """Create dataset with index leave_out_idx removed."""
    loo_data = {}
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                loo_data[key] = np.concatenate([arr[:leave_out_idx], arr[leave_out_idx+1:]])
            else:
                loo_data[key] = value
        else:
            loo_data[key] = value
    return loo_data


def _get_num_datapoints(data):
    """Extract number of datapoints from data dict."""
    if not isinstance(data, dict):
        return 1
    
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                return int(arr.shape[0])
    
    return 1


def _compute_pointwise_log_likelihood(model, data, z, idx):
    """Compute log p(x_idx | z, m) for a single datapoint."""
    try:
        conditioned_model = substitute(model, z)
        model_trace = trace(seed(conditioned_model, jax.random.PRNGKey(0))).get_trace(data=data)
        
        total_log_lik = 0.0
        found_obs_for_idx = False
        
        for site_name, site in model_trace.items():
            if site.get("type") != "sample":
                continue
            if not site.get("is_observed", False):
                continue
            
            obs_value = site["value"]
            fn = site["fn"]
            
            if isinstance(obs_value, (jnp.ndarray, np.ndarray)):
                obs_array = jnp.asarray(obs_value)
                
                if obs_array.ndim >= 1 and obs_array.shape[0] > 1:
                    if idx < obs_array.shape[0]:
                        log_prob_i = fn.log_prob(obs_array[idx])
                        total_log_lik += float(jnp.sum(log_prob_i))
                        found_obs_for_idx = True
                elif obs_array.ndim >= 1 and obs_array.shape[0] == 1:
                    if idx == 0:
                        log_prob = fn.log_prob(obs_array[0])
                        total_log_lik += float(jnp.sum(log_prob))
                        found_obs_for_idx = True
                else:
                    if idx == 0:
                        log_prob = fn.log_prob(obs_array)
                        total_log_lik += float(jnp.sum(log_prob))
                        found_obs_for_idx = True
            else:
                if idx == 0:
                    log_prob = fn.log_prob(obs_value)
                    total_log_lik += float(log_prob)
                    found_obs_for_idx = True
        
        if not found_obs_for_idx:
            return -1e12
        
        return total_log_lik
        
    except Exception as e:
        return -1e12


def run_inference(code, data, targets=None, num_warmup=500, num_samples=1000, rng_seed=0):
    env = {}
    try:
        exec(code, env)
    except Exception as exc:
        raise ValueError(f"compile_error: {exc}") from exc

    if "model" not in env or not callable(env["model"]):
        raise ValueError("compile_error: generated code does not define callable model(data)")

    model = env["model"]

    discrete_sites = _find_unobserved_discrete_sites(model=model, data=data, rng_seed=rng_seed)
    if discrete_sites:
        names = ", ".join(discrete_sites[:8])
        suffix = "" if len(discrete_sites) <= 8 else f", ... (+{len(discrete_sites) - 8} more)"
        raise ValueError(
            "inference_error: Model has unobserved discrete latent site(s) not supported by this pipeline: "
            f"{names}{suffix}. Use continuous latent variables or mark discrete structure explicitly."
        )

    _validate_model_support(model, data, rng_seed)

    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    try:
        mcmc.run(jax.random.PRNGKey(rng_seed), data=data, extra_fields=("diverging",))
    except Exception as exc:
        msg = str(exc)
        if "TracerIntegerConversionError" in msg or "__index__() method was called on traced array" in msg:
            raise ValueError(
                "inference_error: Generated model used a traced value where a Python int is required "
                "(for example range(sampled_value) or list indexing with sampled/jnp values). "
                "Use loop bounds from static data fields instead."
            ) from exc
        raise ValueError(f"inference_error: {exc}") from exc

    samples = mcmc.get_samples(group_by_chain=False)
    available = sorted(samples.keys())

    diagnostics = _extract_mcmc_diagnostics(mcmc, samples)

    if targets is None:
        selected_targets = available
    elif isinstance(targets, str):
        selected_targets = [targets]
    else:
        selected_targets = []
        for target in targets:
            if isinstance(target, str):
                selected_targets.append(target)
            elif isinstance(target, set):
                for item in sorted(target):
                    if not isinstance(item, str):
                        raise TypeError("target names must be strings")
                    selected_targets.append(item)
            else:
                raise TypeError("target names must be strings")

    present_targets = [name for name in selected_targets if name in samples]
    missing_targets = [name for name in selected_targets if name not in samples]
    target_samples = {name: np.asarray(samples[name]).tolist() for name in present_targets}
    return {
        "model": model,
        "samples": {name: np.asarray(value) for name, value in samples.items()},
        "target_samples": target_samples,
        "available_sites": available,
        "missing_targets": missing_targets,
        "mcmc_diagnostics": diagnostics,
    }


def _extract_mcmc_diagnostics(mcmc, samples):
    """Return divergence count, per-site r_hat (chain count permitting), and per-site n_eff."""
    out = {
        "num_divergences": None,
        "num_samples": None,
        "num_chains": None,
        "r_hat": {},
        "n_eff": {},
    }
    try:
        extras = mcmc.get_extra_fields(group_by_chain=False)
        if "diverging" in extras:
            div = np.asarray(extras["diverging"])
            out["num_divergences"] = int(div.sum())
    except Exception:
        pass

    try:
        first_value = next(iter(samples.values()))
        out["num_samples"] = int(np.asarray(first_value).shape[0])
    except StopIteration:
        pass

    try:
        out["num_chains"] = int(getattr(mcmc, "num_chains", 1))
    except Exception:
        pass

    try:
        from numpyro.diagnostics import effective_sample_size, gelman_rubin
        # Both helpers expect (num_chains, num_samples, *event_shape).
        try:
            grouped = mcmc.get_samples(group_by_chain=True)
        except Exception:
            grouped = None
        for name, arr in (grouped or {}).items():
            arr_np = np.asarray(arr)
            if arr_np.ndim < 2:
                continue
            try:
                ess = effective_sample_size(arr_np)
                out["n_eff"][name] = _summarize_diag(ess)
            except Exception:
                pass
            if arr_np.shape[0] > 1:
                try:
                    rhat = gelman_rubin(arr_np)
                    out["r_hat"][name] = _summarize_diag(rhat)
                except Exception:
                    pass
    except Exception:
        pass

    return out


def _summarize_diag(arr):
    """Return min/median/max for any-shape diagnostic array."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(np.min(a)),
        "median": float(np.median(a)),
        "max": float(np.max(a)),
    }


def estimate_test_log_likelihoods(
    model,
    train_data,
    test_data,
    posterior_samples,
    rng_seed=0,
    fallback_log_bound=-1e12,
):
    """Posterior-predictive log p(x_test_i | x_train, m) for each test point.

    Uses a Monte Carlo estimate over the posterior samples already fitted on
    ``train_data``. For each test point we evaluate the per-point log
    likelihood under the model conditioned on a posterior draw, then take the
    log-mean-exp across draws.

    Returns ``(np.ndarray of shape (n_test,), diagnostics dict)``.
    """
    if test_data is None:
        return np.zeros(0, dtype=np.float64), {"method": "skipped"}

    n_test = _get_num_datapoints(test_data)
    if n_test <= 0:
        return np.zeros(0, dtype=np.float64), {"method": "empty_test_data"}

    # Build a combined dataset {train + test, shape (n_train + n_test, ...)}.
    combined = _concatenate_train_test(train_data, test_data)
    n_train = _get_num_datapoints(train_data)

    sample_iter = list(zip(*[
        (name, np.asarray(values)) for name, values in posterior_samples.items()
    ])) if posterior_samples else []

    n_draws = 0
    if posterior_samples:
        first = next(iter(posterior_samples.values()))
        n_draws = int(np.asarray(first).shape[0])

    if n_draws == 0:
        return np.full(n_test, fallback_log_bound, dtype=np.float64), {
            "method": "test_predictive",
            "n_draws": 0,
            "n_test": n_test,
            "reason": "no posterior samples",
        }

    # Per (test_point, draw) log-likelihood matrix.
    log_lik = np.full((n_draws, n_test), fallback_log_bound, dtype=np.float64)
    posterior_arrays = {name: np.asarray(values) for name, values in posterior_samples.items()}

    for s in range(n_draws):
        z = {name: jnp.asarray(arr[s]) for name, arr in posterior_arrays.items()}
        for j in range(n_test):
            global_idx = n_train + j
            try:
                ll = _compute_pointwise_log_likelihood(model, combined, z, global_idx)
                if np.isfinite(ll):
                    log_lik[s, j] = ll
            except Exception:
                pass

    test_log_liks = np.empty(n_test, dtype=np.float64)
    for j in range(n_test):
        col = log_lik[:, j]
        finite = col[np.isfinite(col) & (col > fallback_log_bound + 1.0)]
        if finite.size == 0:
            test_log_liks[j] = fallback_log_bound
        else:
            m = float(np.max(finite))
            test_log_liks[j] = m + math.log(np.mean(np.exp(finite - m)))

    diagnostics = {
        "method": "test_predictive_lme",
        "n_draws": n_draws,
        "n_test": n_test,
    }
    return test_log_liks, diagnostics


def _concatenate_train_test(train_data, test_data):
    """Concatenate train and test arrays along axis 0 for shared keys."""
    if not isinstance(test_data, dict):
        return train_data
    out = {}
    for key, train_val in train_data.items():
        test_val = test_data.get(key)
        if test_val is None:
            out[key] = train_val
            continue
        if isinstance(train_val, (list, np.ndarray)) and isinstance(test_val, (list, np.ndarray)):
            t_arr = np.asarray(train_val)
            te_arr = np.asarray(test_val)
            if t_arr.ndim >= 1 and te_arr.ndim >= 1:
                out[key] = np.concatenate([t_arr, te_arr])
                continue
        out[key] = train_val
    # Carry over keys present only in test_data (rare).
    for key, val in test_data.items():
        if key not in out:
            out[key] = val
    return out


def estimate_log_marginal_iw(
    model,
    data,
    posterior_samples,
    num_inner=25,
    num_outer=1000,
    rng_seed=0,
    min_std=1e-4,
    fallback_log_bound=-1e12,
):
    means = {}
    stds = {}
    for name, values in posterior_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        mean, std = _finite_mean_std_axis0(arr, min_std=min_std)
        means[name] = mean
        stds[name] = std

    rng = np.random.default_rng(rng_seed)
    outer_vals = []

    for _ in range(num_outer):
        log_ws = []
        for _ in range(num_inner):
            z = {}
            log_q = 0.0
            for name in posterior_samples:
                sample = rng.normal(loc=means[name], scale=stds[name])
                z[name] = jnp.asarray(sample)

                centered = (sample - means[name]) / stds[name]
                log_q += float(
                    -0.5
                    * np.sum(
                        centered * centered
                        + np.log(2.0 * np.pi)
                        + 2.0 * np.log(stds[name])
                    )
                )

            try:
                log_joint, _ = log_density(model, (), {"data": data}, z)
                log_w = float(log_joint) - log_q
            except Exception:
                continue

            if not np.isfinite(log_w):
                continue
            log_ws.append(log_w)

        if len(log_ws) > 0:
            outer_vals.append(_logmeanexp(log_ws))

    if len(outer_vals) == 0:
        return float(fallback_log_bound)

    finite_outer = np.asarray(outer_vals, dtype=np.float64)
    finite_outer = finite_outer[np.isfinite(finite_outer)]
    if finite_outer.size == 0:
        return float(fallback_log_bound)
    return float(np.mean(finite_outer))


def _logmeanexp(values):
    vals = np.asarray(values, dtype=np.float64)
    m = vals.max()
    return float(m + math.log(np.mean(np.exp(vals - m))))


def _validate_model_support(model, data, rng_seed):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            model_trace = trace(seed(model, jax.random.PRNGKey(rng_seed))).get_trace(data=data)
        except Exception as exc:
            raise ValueError(f"inference_error: model failed during validation trace: {exc}") from exc

    out_of_support = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning) and "Out-of-support" in str(w.message)
    ]
    if out_of_support:
        raise ValueError(
            "inference_error: Model has out-of-support sample site(s) — likely a probability "
            "sampled from an unbounded distribution (e.g. Normal instead of Beta). "
            f"Details: {'; '.join(out_of_support[:3])}"
        )


def _find_unobserved_discrete_sites(model, data, rng_seed):
    model_trace = trace(seed(model, jax.random.PRNGKey(rng_seed))).get_trace(data=data)
    names = []
    for name, site in model_trace.items():
        if site.get("type") != "sample":
            continue
        if site.get("is_observed", False):
            continue
        fn = site.get("fn")
        if bool(getattr(fn, "has_enumerate_support", False)):
            names.append(name)
    return names


def _finite_mean_std_axis0(arr, min_std=1e-4):
    if arr.ndim == 0:
        if np.isfinite(arr):
            return arr, np.asarray(min_std, dtype=np.float64)
        return np.asarray(0.0, dtype=np.float64), np.asarray(min_std, dtype=np.float64)

    finite = np.isfinite(arr)
    count = np.sum(finite, axis=0)

    safe_den = np.maximum(count, 1)
    finite_vals = np.where(finite, arr, 0.0)
    mean = np.sum(finite_vals, axis=0) / safe_den

    centered = np.where(finite, arr - mean, 0.0)
    var = np.sum(centered * centered, axis=0) / safe_den
    std = np.sqrt(var)

    no_finite = count <= 0
    mean = np.where(no_finite, 0.0, mean)
    std = np.where(no_finite, float(min_std), std)
    std = np.maximum(std, float(min_std))

    return mean, std