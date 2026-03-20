import numpy as np

from .mcmc_log import estimate_log_marginal_iw, run_inference
from .llm import LLMClient
from .model_generator import generate_models


def infer(
    text,
    data,
    targets=None,
    api_url=None,
    api_key=None,
    api_model=None,
    n_models=8,
    mcmc_num_warmup=200,
    mcmc_num_samples=400,
    random_seed=None,
    llm_timeout=300,
    llm_max_retries=2,
    llm_retry_backoff=2.0,
    log_marginal_num_inner=5,
    log_marginal_num_outer=80,
    verbose=False,
    auto_print_result=True,
):
    base_seed = int(random_seed) if random_seed is not None else int(np.random.SeedSequence().generate_state(1)[0])

    llm = LLMClient(
        api_url=api_url,
        api_key=api_key,
        model=api_model,
        timeout=llm_timeout,
        max_retries=llm_max_retries,
        retry_backoff=llm_retry_backoff,
    )
    model_codes = generate_models(llm, text=text, data=data, targets=targets, n_models=n_models)

    valid = []
    failed_models = []
    auto_targets = None

    for idx, code in enumerate(model_codes):
        try:
            infer_out = run_inference(
                code=code,
                data=data,
                targets=targets,
                num_warmup=mcmc_num_warmup,
                num_samples=mcmc_num_samples,
                rng_seed=base_seed + idx,
            )
            log_bound = estimate_log_marginal_iw(
                model=infer_out["model"],
                data=data,
                posterior_samples=infer_out["samples"],
                num_inner=log_marginal_num_inner,
                num_outer=log_marginal_num_outer,
                rng_seed=base_seed + 10_000 + idx,
            )
        except Exception as exc:
            failed_models.append((idx, str(exc)))
            continue

        if targets is not None and infer_out["missing_targets"]:
            failed_models.append(
                (idx, f"missing targets: {', '.join(infer_out['missing_targets'])}")
            )
            continue

        valid.append(
            {
                "code": code,
                "target_samples": infer_out["target_samples"],
                "available_sites": infer_out["available_sites"],
                "log_marginal_bound": log_bound,
            }
        )

        if targets is None:
            site_set = set(infer_out["target_samples"].keys())
            auto_targets = site_set if auto_targets is None else (auto_targets & site_set)

    if not valid:
        detail = "; ".join(f"model {i}: {msg}" for i, msg in failed_models[:3])
        if len(failed_models) > 3:
            detail += f"; ... and {len(failed_models) - 3} more"
        raise RuntimeError(f"No valid model could be inferred. {detail}")

    if failed_models and verbose:
        print(f"Skipped {len(failed_models)} invalid model(s) during inference.")

    final_targets = list(targets) if targets is not None else sorted(auto_targets or [])
    if not final_targets:
        raise RuntimeError("No target variables are available in valid inferred models.")
    log_bounds = np.array([v["log_marginal_bound"] for v in valid], dtype=np.float64)
    
    finite_mask = np.isfinite(log_bounds)
    valid_filtered = [v for v, keep in zip(valid, finite_mask) if keep]
    log_bounds_filtered = log_bounds[finite_mask]
    
    if len(valid_filtered) > 0:
        valid = valid_filtered
        weights = _softmax_from_logs(log_bounds_filtered)
    else:
        weights = np.ones(len(valid), dtype=np.float64) / len(valid)

    per_model_target_samples = [
        {target: v["target_samples"][target] for target in final_targets}
        for v in valid
    ]

    draws_per_model = len(per_model_target_samples[0][final_targets[0]])
    total_draws = draws_per_model * len(per_model_target_samples)
    posterior = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=weights,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed),
    )

    
    _print_posterior_summary(posterior, final_targets)
    return posterior


def _print_posterior_summary(posterior, targets):
    for target in targets:
        values = posterior.get(target, [])
        arr = np.asarray(values, dtype=np.float64)
        print(f"{target}: {arr[:10].tolist()}")


def _softmax_from_logs(log_values):
    shifted = log_values - np.max(log_values)
    unnorm = np.exp(shifted)
    return unnorm / np.sum(unnorm)


def _resample_weighted_samples(per_model_samples, targets, model_weights, total_draws, rng):
    out = {target: [] for target in targets}
    model_choices = rng.choice(len(per_model_samples), size=total_draws, p=model_weights)

    for m_idx in model_choices:
        samples_m = per_model_samples[m_idx]
        max_len = len(samples_m[targets[0]])
        s_idx = int(rng.integers(low=0, high=max_len))
        for target in targets:
            out[target].append(samples_m[target][s_idx])

    return out
