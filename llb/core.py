import numpy as np

from .mcmc_log import estimate_log_marginal_iw, run_inference
from .llm import LLMClient
from .model_generator import generate_models_with_diagnostics


class NoValidModelsError(RuntimeError):
    """Raised when no valid generated models remain for aggregation."""


def infer(
    text,
    data,
    targets=None,
    api_url=None,
    api_key=None,
    api_model=None,
    n_models=2,
    mcmc_num_warmup=50,
    mcmc_num_samples=100,
    random_seed=None,
    llm_timeout=None,
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
    model_codes, gen_diag = generate_models_with_diagnostics(
        llm,
        text=text,
        data=data,
        targets=targets,
        n_models=n_models,
    )
    generated_models = len(model_codes)
    model_codes, deduplicated_models = _dedupe_model_codes(model_codes)

    diagnostics = {
        "requested_models": int(n_models),
        "generated_models": int(generated_models),
        "deduplicated_models": int(deduplicated_models),
        "invalid_models_syntax_or_parsing": int(gen_diag.get("invalid_syntax_parsing_count", 0)),
        "generation_request_failures": int(gen_diag.get("generation_request_failures", 0)),
        "missing_targets_failures": 0,
        "compile_failures": 0,
        "inference_failures": 0,
        "nonfinite_log_bound_drops": 0,
        "valid_models_final": 0,
    }

    def _evaluate_candidates(codes, start_index):
        valid_local = []
        failed_local = []
        auto_targets_local = None

        for local_idx, code in enumerate(codes):
            idx = start_index + local_idx
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
                msg = str(exc)
                if msg.startswith("compile_error:"):
                    diagnostics["compile_failures"] += 1
                    failed_local.append((idx, msg))
                else:
                    diagnostics["inference_failures"] += 1
                    failed_local.append((idx, msg if msg.startswith("inference_error:") else f"inference_error: {msg}"))
                continue

            if targets is not None and infer_out["missing_targets"]:
                diagnostics["missing_targets_failures"] += 1
                failed_local.append(
                    (idx, f"missing targets: {', '.join(infer_out['missing_targets'])}")
                )
                continue

            valid_local.append(
                {
                    "code": code,
                    "target_samples": infer_out["target_samples"],
                    "available_sites": infer_out["available_sites"],
                    "log_marginal_bound": log_bound,
                }
            )

            if targets is None:
                site_set = set(infer_out["target_samples"].keys())
                auto_targets_local = site_set if auto_targets_local is None else (auto_targets_local & site_set)

        return valid_local, failed_local, auto_targets_local

    valid, failed_models, auto_targets = _evaluate_candidates(model_codes, start_index=0)

    if not valid:
        extra_goal = max(0, 6 - int(n_models))
        if extra_goal > 0:
            extra_codes_raw, extra_gen_diag = generate_models_with_diagnostics(
                llm,
                text=text,
                data=data,
                targets=targets,
                n_models=extra_goal,
            )
            extra_generated_models = len(extra_codes_raw)
            extra_codes, extra_deduplicated = _dedupe_model_codes(extra_codes_raw)
            diagnostics["requested_models"] += int(extra_goal)
            diagnostics["generated_models"] += int(extra_generated_models)
            diagnostics["deduplicated_models"] += int(extra_deduplicated)
            diagnostics["invalid_models_syntax_or_parsing"] += int(extra_gen_diag.get("invalid_syntax_parsing_count", 0))
            diagnostics["generation_request_failures"] += int(extra_gen_diag.get("generation_request_failures", 0))
            extra_valid, extra_failed, extra_auto_targets = _evaluate_candidates(
                extra_codes,
                start_index=len(model_codes),
            )
            valid.extend(extra_valid)
            failed_models.extend(extra_failed)
            if targets is None:
                if auto_targets is None:
                    auto_targets = extra_auto_targets
                elif extra_auto_targets is not None:
                    auto_targets = auto_targets & extra_auto_targets

    if not valid:
        raise NoValidModelsError(
            "LLM produced 0 valid models out of "
            f"{diagnostics['generated_models']} generated. Cannot perform inference."
        )

    if failed_models and verbose:
        print(f"Skipped {len(failed_models)} invalid model(s) during inference.")

    final_targets = list(targets) if targets is not None else sorted(auto_targets or [])
    if not final_targets:
        raise RuntimeError("No target variables are available in valid inferred models.")
    log_bounds = np.array([v["log_marginal_bound"] for v in valid], dtype=np.float64)

    finite_mask = np.isfinite(log_bounds)
    dropped_nonfinite = int(np.size(finite_mask) - int(np.sum(finite_mask)))
    diagnostics["nonfinite_log_bound_drops"] = dropped_nonfinite
    valid_filtered = [v for v, keep in zip(valid, finite_mask) if keep]
    log_bounds_filtered = log_bounds[finite_mask]

    if len(valid_filtered) > 0:
        valid = valid_filtered
        weights = _softmax_from_logs(log_bounds_filtered)
    else:
        raise NoValidModelsError(
            "LLM produced 0 valid models out of "
            f"{diagnostics['generated_models']} generated. Cannot perform inference."
        )

    diagnostics["valid_models_final"] = int(len(valid))

    per_model_target_samples = [
        {target: v["target_samples"][target] for target in final_targets}
        for v in valid
    ]

    if auto_print_result:
        print(f"Number of requested models: {diagnostics['requested_models']}")
        print(f"Number of generated models: {diagnostics['generated_models']}")
        print(f"Number of deduplicated models: {diagnostics['deduplicated_models']}")
        print(f"Number of invalid models (syntax/parsing): {diagnostics['invalid_models_syntax_or_parsing']}")
        print(f"Number of generation request failures (timeout/network/API): {diagnostics['generation_request_failures']}")
        print(f"Number of models missing required targets: {diagnostics['missing_targets_failures']}")
        print(f"Number of models that failed to compile: {diagnostics['compile_failures']}")
        print(f"Number of models that failed during inference: {diagnostics['inference_failures']}")
        print(f"Number of models dropped due to non-finite log bound: {diagnostics['nonfinite_log_bound_drops']}")
        print(f"Number of valid models used in final aggregation: {diagnostics['valid_models_final']}")
        for target in final_targets:
            per_model_samples = [
                np.asarray(model_samples[target], dtype=np.float64)
                for model_samples in per_model_target_samples
            ]
            _print_model_averaging_summary(
                samples=per_model_samples,
                weights=weights,
                target_name=target,
            )

    draws_per_model = len(per_model_target_samples[0][final_targets[0]])
    total_draws = draws_per_model * len(per_model_target_samples)
    posterior_weighted = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=weights,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed),
    )

    flat_weights = np.ones(len(per_model_target_samples), dtype=np.float64) / len(per_model_target_samples)
    posterior_flat = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=flat_weights,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed + 1),
    )

    if auto_print_result:
        _print_posterior_summary(posterior_weighted, final_targets)
        _print_weighted_flat_first10(posterior_weighted, posterior_flat, final_targets)
    return posterior_weighted


def _print_posterior_summary(posterior, targets):
    for target in targets:
        values = posterior.get(target, [])
        arr = np.asarray(values, dtype=np.float64)
        print(f"weighted first_10 ({target}): {arr[:10].tolist()}")
        print(f"weighted mean ({target}): {float(arr.mean()):.6f}")


def _print_weighted_flat_first10(posterior_weighted, posterior_flat, targets):
    print("--- Weighted vs Flat (first 10) ---")
    for target in targets:
        weighted = np.asarray(posterior_weighted.get(target, []), dtype=np.float64)
        flat = np.asarray(posterior_flat.get(target, []), dtype=np.float64)
        print(f"target: {target}")
        print(f"weighted first_10: {weighted[:10].tolist()}")
        print(f"flat first_10: {flat[:10].tolist()}")


def _print_model_averaging_summary(samples, weights, target_name):
    if len(samples) == 0:
        print("--- Model Averaging Summary ---")
        print("Number of models: 0")
        print()
        return

    mu_per_model = np.array([float(np.mean(s)) for s in samples], dtype=np.float64)
    flat_weights = np.ones(len(mu_per_model), dtype=np.float64) / len(mu_per_model)

    mu_flat = float(np.mean(mu_per_model))
    mu_weighted = float(np.sum(np.asarray(weights, dtype=np.float64) * mu_per_model))
    diff = mu_weighted - mu_flat

    print("--- Model Averaging Summary ---")
    print(f"Target: {target_name}")
    print(f"Number of models: {len(mu_per_model)}")
    print()
    print(f"Flat mean prediction: {mu_flat:.4f}")
    print(f"Weighted mean prediction: {mu_weighted:.4f}")
    print()
    print(f"Difference (weighted - flat): {diff:.4f}")
    print()
    print("Top 5 models by weight:")

    ranked = np.argsort(-np.asarray(weights, dtype=np.float64))
    for rank_idx in ranked[:5]:
        w = float(weights[rank_idx])
        mu_i = float(mu_per_model[rank_idx])
        print(f"model={int(rank_idx)}, weight={w:.6f}, mu_i={mu_i:.6f}")

    print("2 least-weighted models:")
    least_ranked = np.argsort(np.asarray(weights, dtype=np.float64))
    for rank_idx in least_ranked[:2]:
        w = float(weights[rank_idx])
        mu_i = float(mu_per_model[rank_idx])
        print(f"model={int(rank_idx)}, weight={w:.6f}, mu_i={mu_i:.6f}")
    print()


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


def _dedupe_model_codes(codes):
    seen = set()
    out = []
    dropped = 0
    for code in codes:
        key = _normalize_code_for_hash(code)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        out.append(code)
    return out, dropped


def _normalize_code_for_hash(code):
    lines = []
    for raw in str(code).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)
