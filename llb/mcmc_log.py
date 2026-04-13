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
):
    """
    Compute leave-one-out log predictive densities.
    
    Args:
        use_true_loo: If True, use true LOO-ELBO (expensive). 
                      If False, use PSIS-LOO approximation (fast).
        return_diagnostics: If True, return dict with diagnostics.
    
    Returns:
        If return_diagnostics=False: np.array of LOO log likelihoods
        If return_diagnostics=True: dict with 'loo_log_liks' and 'diagnostics'
    """
    if use_true_loo:
        result = _estimate_loo_true(
            model, data, posterior_samples, 
            num_inner, num_warmup, num_samples, 
            rng_seed, min_std, fallback_log_bound
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


def _estimate_loo_true(
    model, data, posterior_samples,
    num_inner, num_warmup, num_samples,
    rng_seed, min_std, fallback_log_bound
):
    """
    True LOO-ELBO implementation (Algorithm 2).
    Returns dict with loo_log_liks and diagnostics.
    """
    n_datapoints = _get_num_datapoints(data)
    loo_log_liks = []
    elbo_histories = []  # Track ELBO convergence for each datapoint
    
    print(f"Computing TRUE LOO-ELBO for {n_datapoints} datapoints (this will take a while)...")
    
    for i in range(n_datapoints):
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
                print(f"ELBO = {elbo_i:.4f} (±{np.std(elbo_estimates):.4f})")
            else:
                loo_log_liks.append(fallback_log_bound)
                elbo_histories.append([])
                print(f"FAILED (using fallback)")
                
        except Exception as e:
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
        mcmc.run(jax.random.PRNGKey(rng_seed), data=data)
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
    }


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