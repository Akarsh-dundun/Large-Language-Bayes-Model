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
        "shape_mismatch_drops": 0,
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
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

    if failed_models and verbose:
        print(f"Skipped {len(failed_models)} invalid model(s) during inference.")

    final_targets = list(targets) if targets is not None else sorted(auto_targets or [])
    if not final_targets:
        raise RuntimeError("No target variables are available in valid inferred models.")

    valid, shape_drops = _filter_models_by_target_shape(valid, final_targets)
    diagnostics["shape_mismatch_drops"] = int(shape_drops)
    if len(valid) == 0:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

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
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

    diagnostics["valid_models_final"] = int(len(valid))

    per_model_target_samples = []
    valid_after_shape = []
    kept_weights = []
    for m_idx, model_info in enumerate(valid):
        ok, payload = _normalize_target_sample_map(
            target_samples=model_info["target_samples"],
            targets=final_targets,
        )
        if not ok:
            diagnostics["shape_mismatch_drops"] += 1
            continue
        per_model_target_samples.append(payload)
        valid_after_shape.append(model_info)
        kept_weights.append(float(weights[m_idx]))

    if len(per_model_target_samples) == 0:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

    valid = valid_after_shape
    weights = np.asarray(kept_weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    diagnostics["valid_models_final"] = int(len(valid))

    report_targets = _resolve_report_targets(per_model_target_samples, final_targets)

    if auto_print_result:
        print(f"Number of requested models: {diagnostics['requested_models']}")
        print(f"Number of generated models: {diagnostics['generated_models']}")
        print(f"Number of deduplicated models: {diagnostics['deduplicated_models']}")
        print(f"Number of invalid models (syntax/parsing): {diagnostics['invalid_models_syntax_or_parsing']}")
        print(f"Number of generation request failures (timeout/network/API): {diagnostics['generation_request_failures']}")
        print(f"Number of models missing required targets: {diagnostics['missing_targets_failures']}")
        print(f"Number of models that failed to compile: {diagnostics['compile_failures']}")
        print(f"Number of models that failed during inference: {diagnostics['inference_failures']}")
        print(f"Number of models dropped due to target shape mismatch: {diagnostics['shape_mismatch_drops']}")
        print(f"Number of models dropped due to non-finite log bound: {diagnostics['nonfinite_log_bound_drops']}")
        print(f"Number of valid models used in final aggregation: {diagnostics['valid_models_final']}")
        for target in report_targets:
            per_model_samples = [
                np.asarray(model_samples[target], dtype=np.float64)
                for model_samples in per_model_target_samples
            ]
            _print_model_averaging_summary(
                samples=per_model_samples,
                weights=weights,
                target_name=target,
            )
        # Removed compact model averaging summary printout as requested

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
        returned_targets = _resolve_report_targets([posterior_weighted], report_targets)
        _print_posterior_summary(posterior_weighted, returned_targets)
        _print_weighted_flat_first10(posterior_weighted, posterior_flat, returned_targets)
    return posterior_weighted


def _print_posterior_summary(posterior, targets):
    for target in targets:
        arr = np.asarray(posterior.get(target, []), dtype=np.float64)
        mean_value = _target_mean(arr)
        print("--- Weighted Posterior Summary ---")
        print(f"Target: {target}")
        _print_mean_summary("weighted", mean_value)
        print()


def _print_weighted_flat_first10(posterior_weighted, posterior_flat, targets):
    for target in targets:
        weighted = np.asarray(posterior_weighted.get(target, []), dtype=np.float64)
        flat = np.asarray(posterior_flat.get(target, []), dtype=np.float64)
        weighted_mean = _target_mean(weighted)
        flat_mean = _target_mean(flat)
        print("--- Weighted vs Flat Target Summary ---")
        print(f"Target: {target}")
        _print_mean_summary("flat", flat_mean)
        _print_mean_summary("weighted", weighted_mean)
        if np.asarray(weighted_mean).ndim == 0 and np.asarray(flat_mean).ndim == 0:
            print(f"difference (weighted - flat): {float(weighted_mean) - float(flat_mean):.6f}")
        else:
            diff = np.asarray(weighted_mean, dtype=np.float64) - np.asarray(flat_mean, dtype=np.float64)
            _print_array_preview("difference (weighted - flat)", diff)
        print()


def _print_model_averaging_summary(samples, weights, target_name):
    if len(samples) == 0:
        print("--- Model Averaging Summary ---")
        print("Number of models: 0")
        print()
        return

    mu_per_model = [np.asarray(_target_mean(np.asarray(s, dtype=np.float64)), dtype=np.float64) for s in samples]

    # Some generated models may emit the same target name with different shapes.
    # Summarize the dominant compatible shape instead of crashing on np.stack.
    shape_groups = {}
    for idx, mu in enumerate(mu_per_model):
        key = tuple(mu.shape)
        shape_groups.setdefault(key, []).append(idx)

    dominant_shape = max(shape_groups, key=lambda k: len(shape_groups[k]))
    keep_idx = shape_groups[dominant_shape]
    dropped_idx = [i for i in range(len(mu_per_model)) if i not in keep_idx]

    mu_stack = np.stack([mu_per_model[i] for i in keep_idx], axis=0)
    kept_weights = np.asarray([weights[i] for i in keep_idx], dtype=np.float64)
    kept_weights = kept_weights / np.sum(kept_weights)

    mu_flat = np.mean(mu_stack, axis=0)
    mu_weighted = np.tensordot(kept_weights, mu_stack, axes=(0, 0))
    diff = np.asarray(mu_weighted, dtype=np.float64) - np.asarray(mu_flat, dtype=np.float64)

    print("--- Model Averaging Summary ---")
    print(f"Target: {target_name}")
    print(f"Number of models: {len(mu_per_model)}")
    if dropped_idx:
        shape_counts = {str(k): len(v) for k, v in shape_groups.items()}
        print(f"Shape mismatch detected for target '{target_name}'; using dominant shape {dominant_shape}.")
        print(f"Shape counts: {shape_counts}")
        print(f"Dropped models for summary due to shape mismatch: {len(dropped_idx)}")
    _print_mean_summary("flat", mu_flat)
    _print_mean_summary("weighted", mu_weighted)
    if np.asarray(diff).ndim == 0:
        print(f"Difference (weighted - flat): {float(diff):.6f}")
    else:
        _print_array_preview("Difference (weighted - flat)", diff)
    print()
    print(f"Top 5 models by weight for target '{target_name}':")

    ranked_local = np.argsort(-kept_weights)
    for local_idx in ranked_local[:5]:
        rank_idx = keep_idx[int(local_idx)]
        w = float(weights[rank_idx])
        mu_i = np.asarray(mu_per_model[rank_idx], dtype=np.float64)
        if mu_i.ndim == 0:
            print(f"model={int(rank_idx)}, weight={w:.6f}, mu_i={float(mu_i):.6f}")
        else:
            preview = mu_i.reshape(-1)[:10].tolist()
            print(
                f"model={int(rank_idx)}, weight={w:.6f}, "
                f"mu_i_shape={tuple(mu_i.shape)}, mu_i_first10={preview}"
            )

    print(f"2 least-weighted models for target '{target_name}':")
    least_ranked_local = np.argsort(kept_weights)
    for local_idx in least_ranked_local[:2]:
        rank_idx = keep_idx[int(local_idx)]
        w = float(weights[rank_idx])
        mu_i = np.asarray(mu_per_model[rank_idx], dtype=np.float64)
        if mu_i.ndim == 0:
            print(f"model={int(rank_idx)}, weight={w:.6f}, mu_i={float(mu_i):.6f}")
        else:
            preview = mu_i.reshape(-1)[:10].tolist()
            print(
                f"model={int(rank_idx)}, weight={w:.6f}, "
                f"mu_i_shape={tuple(mu_i.shape)}, mu_i_first10={preview}"
            )
    print()


def _target_mean(arr):
    if arr.ndim == 0:
        return arr
    if arr.shape[0] == 0:
        return np.nan
    return np.mean(arr, axis=0)


def _print_mean_summary(label, mean_value):
    mean_arr = np.asarray(mean_value, dtype=np.float64)
    if mean_arr.ndim == 0:
        print(f"{label} mean prediction: {float(mean_arr):.6f}")
        return
    print(f"{label} mean prediction shape: {tuple(mean_arr.shape)}")
    _print_array_preview(f"{label} mean prediction", mean_arr)


def _print_array_preview(label, arr):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    if flat.size <= 10:
        print(f"{label} full: {flat.tolist()}")
    else:
        print(f"{label} first_10: {flat[:10].tolist()}")


    # Removed _print_compact_model_averaging_summary as requested


def _softmax_from_logs(log_values):
    shifted = log_values - np.max(log_values)
    unnorm = np.exp(shifted)
    return unnorm / np.sum(unnorm)


def _resample_weighted_samples(per_model_samples, targets, model_weights, total_draws, rng):
    out = {target: [] for target in targets}
    model_choices = rng.choice(len(per_model_samples), size=total_draws, p=model_weights)

    for m_idx in model_choices:
        samples_m = per_model_samples[m_idx]
        if any(target not in samples_m for target in targets):
            continue
        lengths = [len(samples_m[target]) for target in targets]
        min_len = int(min(lengths)) if lengths else 0
        if min_len <= 0:
            continue
        s_idx = int(rng.integers(low=0, high=min_len))
        for target in targets:
            out[target].append(samples_m[target][s_idx])

    return out


def _build_no_valid_models_message(diagnostics):
    return (
        "LLM produced 0 valid models out of "
        f"{diagnostics['generated_models']} generated. Cannot perform inference. "
        f"requested={diagnostics['requested_models']}, "
        f"deduplicated={diagnostics['deduplicated_models']}, "
        f"invalid_syntax_or_parsing={diagnostics['invalid_models_syntax_or_parsing']}, "
        f"generation_request_failures={diagnostics['generation_request_failures']}, "
        f"missing_targets={diagnostics['missing_targets_failures']}, "
        f"compile_failures={diagnostics['compile_failures']}, "
        f"inference_failures={diagnostics['inference_failures']}, "
        f"shape_mismatch_drops={diagnostics['shape_mismatch_drops']}, "
        f"nonfinite_log_bound_drops={diagnostics['nonfinite_log_bound_drops']}."
    )


def _filter_models_by_target_shape(valid_models, targets):
    if len(valid_models) <= 1:
        return valid_models, 0

    profiles = []
    for model in valid_models:
        target_samples = model.get("target_samples", {})
        try:
            draw_lengths = []
            event_shapes = []
            for target in targets:
                arr = np.asarray(target_samples[target])
                if arr.ndim < 1 or arr.shape[0] <= 0:
                    raise ValueError("empty target samples")
                draw_lengths.append(int(arr.shape[0]))
                event_shapes.append(tuple(arr.shape[1:]))
            if len(set(draw_lengths)) != 1:
                profiles.append(None)
            else:
                profiles.append(tuple(event_shapes))
        except Exception:
            profiles.append(None)

    valid_profiles = [p for p in profiles if p is not None]
    if not valid_profiles:
        return [], len(valid_models)

    counts = {}
    for p in valid_profiles:
        counts[p] = counts.get(p, 0) + 1
    dominant_profile = max(counts, key=counts.get)

    kept = []
    dropped = 0
    for model, profile in zip(valid_models, profiles):
        if profile == dominant_profile:
            kept.append(model)
        else:
            dropped += 1

    return kept, dropped


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


def _normalize_target_sample_map(target_samples, targets):
    arrays = {}
    for target in targets:
        if target not in target_samples:
            return False, f"missing target '{target}'"
        arr = np.asarray(target_samples[target], dtype=np.float64)
        if arr.ndim == 0:
            return False, f"target '{target}' has scalar sample container"
        arrays[target] = arr

    first_dims = [arrays[target].shape[0] for target in targets]
    n_ref = int(max(first_dims))

    for target in targets:
        arr = arrays[target]
        if arr.shape[0] == n_ref:
            continue
        if arr.ndim >= 2 and arr.shape[1] == n_ref:
            arrays[target] = np.swapaxes(arr, 0, 1)

    first_dims_after = [arrays[target].shape[0] for target in targets]
    if len(set(first_dims_after)) != 1:
        return False, f"inconsistent sample axis lengths across targets: {first_dims_after}"

    normalized = {target: arrays[target] for target in targets}
    return True, normalized


def _resolve_report_targets(sample_maps, fallback_targets):
    if not sample_maps:
        return list(fallback_targets)

    returned = []
    seen = set()

    for target in fallback_targets:
        if all(target in sm for sm in sample_maps):
            returned.append(target)
            seen.add(target)

    for target in sample_maps[0].keys():
        if target in seen:
            continue
        if all(target in sm for sm in sample_maps):
            returned.append(target)
            seen.add(target)

    return returned
