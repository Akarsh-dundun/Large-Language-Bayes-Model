import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, NUTS
from numpyro.infer.util import log_density

def estimate_loo_log_likelihoods(
    model,
    data,
    posterior_samples,
    num_inner=25,
    rng_seed=0,
    min_std=1e-4,
    fallback_log_bound=-1e12,
):
    """
    Compute leave-one-out log predictive densities for each datapoint.
    """
    n_datapoints = _get_num_datapoints(data)
    
    # Fit proposal distribution
    means = {}
    stds = {}
    for name, values in posterior_samples.items():
        arr = np.asarray(values, dtype=np.float64)
        mean, std = _finite_mean_std_axis0(arr, min_std=min_std)
        means[name] = mean
        stds[name] = std
    
    rng = np.random.default_rng(rng_seed)
    loo_log_liks = []
    
    # For each datapoint
    for i in range(n_datapoints):
        log_pred_estimates = []
        
        # Monte Carlo estimation
        for _ in range(num_inner):
            z = {}
            
            # Sample from posterior
            for name in posterior_samples:
                sample = rng.normal(loc=means[name], scale=stds[name])
                z[name] = jnp.asarray(sample)
            
            try:
                log_lik_i = _compute_pointwise_log_likelihood(model, data, z, i)
                
                # Only accept finite, non-fallback values
                if np.isfinite(log_lik_i) and log_lik_i > -1e10:
                    log_pred_estimates.append(log_lik_i)
                    
            except Exception:
                continue
        
        # Aggregate estimates
        if len(log_pred_estimates) > 0:
            loo_i = _logmeanexp(log_pred_estimates)
            loo_log_liks.append(loo_i)
        else:
            # All iterations failed - use fallback
            loo_log_liks.append(fallback_log_bound)
    
    return np.array(loo_log_liks, dtype=np.float64)

def _get_num_datapoints(data):
    """Extract number of datapoints from data dict."""
    if not isinstance(data, dict):
        return 1
    
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                return int(arr.shape[0])
    
    # Fallback: assume 1 datapoint
    return 1

def _compute_pointwise_log_likelihood(model, data, z, idx):
    """
    Compute log p(x_idx | z, m) for a single datapoint.
    """
    from numpyro.handlers import substitute
    
    try:
        # Condition model on latent variables
        conditioned_model = substitute(model, z)
        model_trace = trace(seed(conditioned_model, jax.random.PRNGKey(0))).get_trace(data=data)
        
        total_log_lik = 0.0
        found_obs_for_idx = False
        
        # Extract pointwise log likelihood
        for site_name, site in model_trace.items():
            if site.get("type") != "sample":
                continue
            if not site.get("is_observed", False):
                continue
            
            obs_value = site["value"]
            fn = site["fn"]
            
            # Handle vectorized observations
            if isinstance(obs_value, (jnp.ndarray, np.ndarray)):
                obs_array = jnp.asarray(obs_value)
                
                if obs_array.ndim >= 1 and obs_array.shape[0] > 1:
                    # Multiple datapoints - extract idx-th observation
                    if idx < obs_array.shape[0]:
                        log_prob_i = fn.log_prob(obs_array[idx])
                        total_log_lik += float(jnp.sum(log_prob_i))
                        found_obs_for_idx = True
                elif obs_array.ndim >= 1 and obs_array.shape[0] == 1:
                    # Single datapoint stored as array
                    if idx == 0:
                        log_prob = fn.log_prob(obs_array[0])
                        total_log_lik += float(jnp.sum(log_prob))
                        found_obs_for_idx = True
                else:
                    # Scalar array
                    if idx == 0:
                        log_prob = fn.log_prob(obs_array)
                        total_log_lik += float(jnp.sum(log_prob))
                        found_obs_for_idx = True
            else:
                # Scalar observation (Python scalar)
                if idx == 0:
                    log_prob = fn.log_prob(obs_value)
                    total_log_lik += float(log_prob)
                    found_obs_for_idx = True
        
        # If we didn't find an observation for this idx, return fallback
        if not found_obs_for_idx:
            return -1e12
        
        return total_log_lik
        
    except Exception as e:
        return -1e12    
def _create_loo_dataset(data, leave_out_idx):
    """Create dataset with index leave_out_idx removed."""
    loo_data = {}
    for key, value in data.items():
        if isinstance(value, (list, np.ndarray)):
            arr = np.asarray(value)
            if arr.ndim >= 1 and arr.shape[0] > 1:
                # Remove the i-th element
                loo_data[key] = np.concatenate([arr[:leave_out_idx], arr[leave_out_idx+1:]])
            else:
                loo_data[key] = value
        else:
            loo_data[key] = value
    return loo_data



def run_inference(code, data, targets=None, num_warmup=500, num_samples=1000, rng_seed=0):
    env = {}
    try:
        exec(code, env)
    except Exception as exc:
        raise ValueError(f"compile_error: {exc}") from exc

    if "model" not in env or not callable(env["model"]):
        raise ValueError("compile_error: generated code does not define callable model(data)")

    model = env["model"]

    # Reject models with unobserved discrete latent variables; these can trigger
    # automatic enumeration warnings and unstable behavior for generic NUTS usage.
    discrete_sites = _find_unobserved_discrete_sites(model=model, data=data, rng_seed=rng_seed)
    if discrete_sites:
        names = ", ".join(discrete_sites[:8])
        suffix = "" if len(discrete_sites) <= 8 else f", ... (+{len(discrete_sites) - 8} more)"
        raise ValueError(
            "inference_error: Model has unobserved discrete latent site(s) not supported by this pipeline: "
            f"{names}{suffix}. Use continuous latent variables or mark discrete structure explicitly."
        )

    # Cheap validation: one forward trace to catch out-of-support parameters before
    # spending warmup time on a broken model.
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
    """Run one forward trace and raise if any site has out-of-support values."""
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
