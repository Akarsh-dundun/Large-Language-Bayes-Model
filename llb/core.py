import numpy as np
from scipy.optimize import minimize
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

from .mcmc_log import estimate_log_marginal_iw, run_inference, estimate_loo_log_likelihoods, _get_num_datapoints
from .llm import LLMClient
from .model_generator import generate_models_with_diagnostics

# Initialize rich console
console = Console()


class NoValidModelsError(RuntimeError):
    """Raised when no valid generated models remain for aggregation."""


def _solve_stacking_optimization(loo_log_liks_matrix, verbose=False):
    """
    Solve the stacking optimization problem with detailed debugging.
    """
    n_datapoints, n_models = loo_log_liks_matrix.shape

    if n_models == 1:
        return np.array([1.0])

    # Add rich panel header
    console.rule("[bold cyan]Stacking Optimization[/bold cyan]", style="cyan")
    print(f"n_datapoints: {n_datapoints}, n_models: {n_models}")

    def objective(w):
        """Negative of the stacking objective (for minimization)."""
        log_sum = np.zeros(n_datapoints)
        for i in range(n_datapoints):
            max_val = np.max(loo_log_liks_matrix[i, :])
            log_sum[i] = max_val + np.log(
                np.sum(w * np.exp(loo_log_liks_matrix[i, :] - max_val))
            )
        return -np.mean(log_sum)

    def gradient(w):
        """Gradient of the negative objective."""
        grad = np.zeros(n_models)
        for i in range(n_datapoints):
            max_val = np.max(loo_log_liks_matrix[i, :])
            exp_vals = np.exp(loo_log_liks_matrix[i, :] - max_val)
            weighted_sum = np.sum(w * exp_vals)
            grad += exp_vals / weighted_sum
        return -grad / n_datapoints

    constraints = {
        'type': 'eq',
        'fun': lambda w: np.sum(w) - 1.0,
        'jac': lambda w: np.ones(n_models)
    }

    bounds = [(0.0, 1.0) for _ in range(n_models)]
    w0 = np.ones(n_models) / n_models

    console.print(f"Initial weights (uniform): [yellow]{w0[:5]}[/yellow]... (showing first 5)")
    console.print(f"Initial objective value: [cyan]{objective(w0):.10f}[/cyan]")
    initial_grad = gradient(w0)
    console.print(f"Initial gradient norm: [cyan]{np.linalg.norm(initial_grad):.10f}[/cyan]")
    console.print(f"Initial gradient (first 5): [yellow]{initial_grad[:5]}[/yellow]")

    if np.allclose(initial_grad, 0, atol=1e-8):
        console.print("\n[yellow]⚠️  GRADIENT IS ZERO at uniform weights![/yellow]")
        print("This means uniform weights are locally optimal.")
        print("This happens when all models have identical LOO performance.")
        console.rule(style="cyan")
        return w0

    # Optimize
    result = minimize(
        objective,
        w0,
        method='SLSQP',
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': verbose}
    )

    print("\nOptimization completed:")
    console.print(f"  Success: [green]{result.success}[/green]" if result.success else f"  Success: [red]{result.success}[/red]")
    console.print(f"  Message: [dim]{result.message}[/dim]")
    print(f"  Iterations: {result.nit}")
    console.print(f"  Final objective: [cyan]{result.fun:.10f}[/cyan]")
    console.print(f"  Objective improvement: [green]{objective(w0) - result.fun:.10f}[/green]")

    weights = result.x
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)

    console.print(f"  Final weights (first 5): [yellow]{weights[:5]}[/yellow]")
    console.print(f"  Weight range: [[cyan]{weights.min():.6f}[/cyan], [cyan]{weights.max():.6f}[/cyan]]")
    console.print(f"  Weight std: [cyan]{np.std(weights):.6f}[/cyan]")

    if np.allclose(weights, w0, atol=1e-6):
        console.print("\n[yellow]⚠️  WARNING: Weights did not change from initial uniform![/yellow]")
        print("Likely cause: All models perform identically on LOO.")
    else:
        console.print(f"\n[green]✓ Weights changed from uniform[/green] (max change: {np.max(np.abs(weights - w0)):.6f})")

    console.rule(style="cyan")

    return weights


def _solve_stacking_optimization_simple(loo_log_liks_matrix):
    """
    Simpler version using just scipy without gradient.
    """
    n_datapoints, n_models = loo_log_liks_matrix.shape

    if n_models == 1:
        return np.array([1.0])

    def objective(w):
        """Negative of the stacking objective."""
        log_sum = np.zeros(n_datapoints)
        for i in range(n_datapoints):
            max_val = np.max(loo_log_liks_matrix[i, :])
            log_sum[i] = max_val + np.log(
                np.sum(w * np.exp(loo_log_liks_matrix[i, :] - max_val))
            )
        return -np.mean(log_sum)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    w0 = np.ones(n_models) / n_models

    result = minimize(
        objective,
        w0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-9}
    )

    weights = np.maximum(result.x, 0.0)
    return weights / np.sum(weights)


def infer(
    text,
    data,
    targets=None,
    model_codes=None,
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
    loo_num_inner=25,
    loo_num_warmup=50,      # NEW
    loo_num_samples=100,    # NEW
    use_true_loo=True,      # NEW FLAG
    verbose=False,
    auto_print_result=True,
):
    base_seed = int(random_seed) if random_seed is not None else int(np.random.SeedSequence().generate_state(1)[0])

    pre_supplied = model_codes is not None
    if pre_supplied:
        supplied_codes = list(model_codes)
        generated_models = len(supplied_codes)
        all_generated_codes = supplied_codes.copy()
        supplied_codes, deduplicated_models = _dedupe_model_codes(supplied_codes)
        gen_diag = {}
    else:
        llm_kwargs = {
            "api_url": api_url,
            "api_key": api_key,
            "model": api_model,
            "max_retries": llm_max_retries,
            "retry_backoff": llm_retry_backoff,
        }
        if llm_timeout is not None:
            llm_kwargs["timeout"] = llm_timeout
        llm = LLMClient(**llm_kwargs)
        supplied_codes, gen_diag = generate_models_with_diagnostics(
            llm,
            text=text,
            data=data,
            targets=targets,
            n_models=n_models,
        )
        generated_models = len(supplied_codes)
        all_generated_codes = supplied_codes.copy()
        supplied_codes, deduplicated_models = _dedupe_model_codes(supplied_codes)

    model_codes = supplied_codes

    diagnostics = {
        "requested_models": int(n_models),
        "generated_models": int(generated_models),
        "deduplicated_models": int(deduplicated_models),
        "invalid_models_syntax_or_parsing": int(gen_diag.get("invalid_syntax_parsing_count", 0)),
        "generation_request_failures": int(gen_diag.get("generation_request_failures", 0)),
        "first_failure_reason": _first_request_failure_reason(gen_diag),
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

                # Build model_info dict - COMPUTE BOTH
                model_info = {
                    "code": code,
                    "target_samples": infer_out["target_samples"],
                    "available_sites": infer_out["available_sites"],
                }

                # Always compute marginal likelihood
                try:
                    log_bound = estimate_log_marginal_iw(
                        model=infer_out["model"],
                        data=data,
                        posterior_samples=infer_out["samples"],
                        num_inner=log_marginal_num_inner,
                        num_outer=log_marginal_num_outer,
                        rng_seed=base_seed + 10_000 + idx,
                    )
                    model_info["log_marginal_bound"] = log_bound
                except Exception as e:
                    if verbose:
                        console.print(f"[yellow]Model {idx}: Marginal likelihood failed ({e})[/yellow]")
                    model_info["log_marginal_bound"] = -1e12

                # Always compute LOO with diagnostics
                try:
                    loo_result = estimate_loo_log_likelihoods(
                        model=infer_out["model"],
                        data=data,
                        posterior_samples=infer_out["samples"],
                        num_inner=loo_num_inner,
                        num_warmup=loo_num_warmup,
                        num_samples=loo_num_samples,
                        rng_seed=base_seed + 20_000 + idx,
                        use_true_loo=use_true_loo,
                        return_diagnostics=True,  # Get diagnostics
                    )

                    if isinstance(loo_result, dict):
                        model_info["loo_log_liks"] = loo_result['loo_log_liks']
                        model_info["loo_diagnostics"] = loo_result['diagnostics']
                    else:
                        model_info["loo_log_liks"] = loo_result
                        model_info["loo_diagnostics"] = None

                except Exception as loo_exc:
                    if verbose:
                        console.print(f"[yellow]Model {idx}: LOO failed ({loo_exc})[/yellow]")
                    n_data = _get_num_datapoints(data)
                    model_info["loo_log_liks"] = np.full(n_data, -1e12, dtype=np.float64)
                    model_info["loo_diagnostics"] = {'method': 'failed', 'error': str(loo_exc)}

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

            valid_local.append(model_info)

            if targets is None:
                site_set = set(infer_out["target_samples"].keys())
                auto_targets_local = site_set if auto_targets_local is None else (auto_targets_local & site_set)

        return valid_local, failed_local, auto_targets_local

    valid, failed_models, auto_targets = _evaluate_candidates(model_codes, start_index=0)

    if not valid and not pre_supplied:
        extra_goal = int(n_models)
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
            if diagnostics["first_failure_reason"] is None:
                diagnostics["first_failure_reason"] = _first_request_failure_reason(extra_gen_diag)
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

    # ========================================
    # METHOD 1: Marginal Likelihood (BMA-style)
    # ========================================
    log_bounds = np.array([v["log_marginal_bound"] for v in valid], dtype=np.float64)
    finite_mask = np.isfinite(log_bounds)
    log_bounds_filtered = log_bounds[finite_mask]

    if len(log_bounds_filtered) > 0:
        weights_bma = _softmax_from_logs(log_bounds_filtered)
        weights_bma_full = np.zeros(len(valid))
        weights_bma_full[finite_mask] = weights_bma
    else:
        console.print("[red]All marginal likelihood bounds are non-finite![/red]")
        weights_bma_full = np.ones(len(valid)) / len(valid)

    # ========================================
    # METHOD 2: LOO Stacking
    # ========================================
    loo_matrix_raw = np.column_stack([v["loo_log_liks"] for v in valid])

    # Filter out models where LOO completely failed
    fallback_threshold = 0.5
    valid_models_mask = []

    for j in range(loo_matrix_raw.shape[1]):
        col = loo_matrix_raw[:, j]
        num_fallback = np.sum(col < -1e10)
        frac_fallback = num_fallback / len(col)
        valid_models_mask.append(frac_fallback < fallback_threshold)

    valid_models_mask = np.array(valid_models_mask)
    loo_matrix = loo_matrix_raw[:, valid_models_mask]

    num_dropped_loo = int(np.sum(~valid_models_mask))
    if num_dropped_loo > 0:
        console.print(f"[yellow]LOO: Dropped {num_dropped_loo} models due to computation failures[/yellow]")

    if loo_matrix.shape[1] > 0:
        # Diagnostic
        console.rule("[bold magenta]LOO Matrix Diagnostic[/bold magenta]", style="magenta")
        console.print(f"Shape: [cyan]{loo_matrix.shape}[/cyan]")
        print(f"\nFirst 3 rows (first 3 datapoints):")
        print(loo_matrix[:3, :])
        print(f"\nVariance per datapoint (across models):")
        for i in range(min(6, loo_matrix.shape[0])):
            variance = np.var(loo_matrix[i, :])
            range_val = np.max(loo_matrix[i, :]) - np.min(loo_matrix[i, :])
            console.print(f"  Point {i}: var=[yellow]{variance:.8f}[/yellow], range=[green]{range_val:.8f}[/green]")
        console.print(f"\nOverall variance: [yellow]{np.var(loo_matrix):.8f}[/yellow]")
        console.print(f"Overall range: [green]{np.max(loo_matrix) - np.min(loo_matrix):.8f}[/green]")

        # Check if all columns are identical
        first_col = loo_matrix[:, 0]
        all_same = True
        for j in range(1, loo_matrix.shape[1]):
            if not np.allclose(loo_matrix[:, j], first_col, rtol=1e-5):
                all_same = False
                break

        if all_same:
            console.print("\n[red]⚠️  WARNING: ALL MODELS HAVE IDENTICAL LOO VALUES![/red]")
            print("This means the LOO computation is broken, not the optimization.")
        else:
            console.print(f"\n[green]✓ Models have different LOO values[/green]")

        console.rule(style="magenta")

        # Optimize
        weights_loo_subset = _solve_stacking_optimization(loo_matrix, verbose=verbose)

        # Map back to full valid set
        weights_loo_full = np.zeros(len(valid))
        weights_loo_full[valid_models_mask] = weights_loo_subset
        weights_loo_full = weights_loo_full / np.sum(weights_loo_full)  # Renormalize
    else:
        console.print("[red]All models failed LOO computation![/red]")
        weights_loo_full = np.ones(len(valid)) / len(valid)

    # ========================================
    # Compute final LOO objective value
    # ========================================
    final_loo_objective = 0.0
    if loo_matrix.shape[1] > 0:
        for i in range(loo_matrix.shape[0]):
            max_val = np.max(loo_matrix[i, :])
            log_sum_i = max_val + np.log(
                np.sum(weights_loo_full[valid_models_mask] * np.exp(loo_matrix[i, :] - max_val))
            )
            final_loo_objective += log_sum_i
        final_loo_objective /= loo_matrix.shape[0]

    # ========================================
    # COMPARISON TABLE
    # ========================================
    console.rule("[bold cyan]Weight Comparison: BMA vs LOO Stacking[/bold cyan]", style="cyan")

    comparison_table = Table(
        title="Model Weights Comparison",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    comparison_table.add_column("Model", style="dim", justify="right")
    comparison_table.add_column("BMA Weight", style="yellow", justify="right")
    comparison_table.add_column("LOO Weight", style="green", justify="right")
    comparison_table.add_column("Difference", style="magenta", justify="right")
    comparison_table.add_column("Log Marginal", style="cyan", justify="right")

    for i in range(len(valid)):
        w_bma = weights_bma_full[i]
        w_loo = weights_loo_full[i]
        diff = w_loo - w_bma
        log_marg = log_bounds[i]

        # Color code difference
        if abs(diff) > 0.1:
            diff_str = f"[bold magenta]{diff:+.6f}[/bold magenta]"
        elif abs(diff) > 0.01:
            diff_str = f"[magenta]{diff:+.6f}[/magenta]"
        else:
            diff_str = f"[dim]{diff:+.6f}[/dim]"

        comparison_table.add_row(
            str(i),
            f"{w_bma:.6f}",
            f"{w_loo:.6f}",
            diff_str,
            f"{log_marg:.2f}" if np.isfinite(log_marg) else "—"
        )

    console.print(comparison_table)

    # Summary statistics
    console.print(f"\n[bold]Weight Statistics:[/bold]")
    console.print(f"  BMA entropy: [yellow]{-np.sum(weights_bma_full * np.log(weights_bma_full + 1e-10)):.4f}[/yellow]")
    console.print(f"  LOO entropy: [green]{-np.sum(weights_loo_full * np.log(weights_loo_full + 1e-10)):.4f}[/green]")
    console.print(f"  BMA max weight: [yellow]{weights_bma_full.max():.6f}[/yellow]")
    console.print(f"  LOO max weight: [green]{weights_loo_full.max():.6f}[/green]")
    console.print(f"  L1 distance: [magenta]{np.sum(np.abs(weights_loo_full - weights_bma_full)):.6f}[/magenta]")
    console.print(f"  Final LOO objective: [green]{final_loo_objective:.4f}[/green]")
    console.rule(style="cyan")

    # ========================================
    # Compute posteriors for BOTH methods
    # ========================================
    diagnostics["valid_models_final"] = int(len(valid))

    per_model_target_samples = []
    valid_after_shape = []
    kept_weights_bma = []
    kept_weights_loo = []

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
        kept_weights_bma.append(float(weights_bma_full[m_idx]))
        kept_weights_loo.append(float(weights_loo_full[m_idx]))

    if len(per_model_target_samples) == 0:
        diagnostics["valid_models_final"] = 0
        raise NoValidModelsError(_build_no_valid_models_message(diagnostics))

    valid = valid_after_shape
    weights_bma = np.asarray(kept_weights_bma, dtype=np.float64)
    weights_bma = weights_bma / np.sum(weights_bma)
    weights_loo = np.asarray(kept_weights_loo, dtype=np.float64)
    weights_loo = weights_loo / np.sum(weights_loo)
    flat_weights = np.ones(len(per_model_target_samples)) / len(per_model_target_samples)
    diagnostics["valid_models_final"] = int(len(valid))

    report_targets = _resolve_report_targets(per_model_target_samples, final_targets)

    # Compute epistemic uncertainty for BOTH
    epistemic_uncertainty_bma = {}
    epistemic_uncertainty_loo = {}
    epistemic_uncertainty_uniform = {}

    for target in final_targets:
        mu_per_model = []
        for model_samples in per_model_target_samples:
            arr = np.asarray(model_samples[target], dtype=np.float64)
            mu_per_model.append(_target_mean(arr))

        mu_stack = np.stack(
            [np.asarray(mu, dtype=np.float64) for mu in mu_per_model],
            axis=0,
        )

        # BMA epistemic uncertainty
        mu_bma = np.tensordot(weights_bma, mu_stack, axes=(0, 0))
        S_bma = np.tensordot(weights_bma, (mu_stack - mu_bma) ** 2, axes=(0, 0))
        C_bma = 1.0 - float(np.sum(weights_bma ** 2))
        epistemic_uncertainty_bma[target] = S_bma / C_bma if C_bma > 0 else np.zeros_like(mu_bma)

        # LOO epistemic uncertainty
        mu_loo = np.tensordot(weights_loo, mu_stack, axes=(0, 0))
        S_loo = np.tensordot(weights_loo, (mu_stack - mu_loo) ** 2, axes=(0, 0))
        C_loo = 1.0 - float(np.sum(weights_loo ** 2))
        epistemic_uncertainty_loo[target] = S_loo / C_loo if C_loo > 0 else np.zeros_like(mu_loo)

        mu_uniform = np.tensordot(flat_weights, mu_stack, axes=(0, 0))
        S_uniform = np.tensordot(flat_weights, (mu_stack - mu_uniform) ** 2, axes=(0, 0))
        C_uniform = 1.0 - float(np.sum(flat_weights ** 2))
        epistemic_uncertainty_uniform[target] = S_uniform / C_uniform if C_uniform > 0 else np.zeros_like(mu_uniform)

    # ========================================
    # PRINT DIAGNOSTICS
    # ========================================
    if auto_print_result:
        console.rule("[bold blue]Inference Diagnostics[/bold blue]", style="blue")
        console.print(f"Number of requested models: [cyan]{diagnostics['requested_models']}[/cyan]")
        console.print(f"Number of generated models: [cyan]{diagnostics['generated_models']}[/cyan]")
        console.print(f"Number of deduplicated models: [yellow]{diagnostics['deduplicated_models']}[/yellow]" if diagnostics['deduplicated_models'] > 0 else f"Number of deduplicated models: [dim]{diagnostics['deduplicated_models']}[/dim]")
        console.print(f"Number of invalid models (syntax/parsing): [yellow]{diagnostics['invalid_models_syntax_or_parsing']}[/yellow]" if diagnostics['invalid_models_syntax_or_parsing'] > 0 else f"Number of invalid models (syntax/parsing): [dim]{diagnostics['invalid_models_syntax_or_parsing']}[/dim]")
        console.print(f"Number of generation request failures: [red]{diagnostics['generation_request_failures']}[/red]" if diagnostics['generation_request_failures'] > 0 else f"Number of generation request failures: [dim]{diagnostics['generation_request_failures']}[/dim]")
        console.print(f"Number of models missing required targets: [yellow]{diagnostics['missing_targets_failures']}[/yellow]" if diagnostics['missing_targets_failures'] > 0 else f"Number of models missing required targets: [dim]{diagnostics['missing_targets_failures']}[/dim]")
        console.print(f"Number of models that failed to compile: [red]{diagnostics['compile_failures']}[/red]" if diagnostics['compile_failures'] > 0 else f"Number of models that failed to compile: [dim]{diagnostics['compile_failures']}[/dim]")
        console.print(f"Number of models that failed during inference: [red]{diagnostics['inference_failures']}[/red]" if diagnostics['inference_failures'] > 0 else f"Number of models that failed during inference: [dim]{diagnostics['inference_failures']}[/dim]")
        console.print(f"Number of models dropped due to target shape mismatch: [yellow]{diagnostics['shape_mismatch_drops']}[/yellow]" if diagnostics['shape_mismatch_drops'] > 0 else f"Number of models dropped due to target shape mismatch: [dim]{diagnostics['shape_mismatch_drops']}[/dim]")
        console.print(f"Number of valid models used in final aggregation: [bold green]{diagnostics['valid_models_final']}[/bold green]")

        # Print both weight summaries
        for target in report_targets:
            per_model_samples = [
                np.asarray(model_samples[target], dtype=np.float64)
                for model_samples in per_model_target_samples
            ]
            console.rule(f"[bold green]Model Averaging: {target}[/bold green]", style="green")
            _print_dual_model_averaging_summary(
                samples=per_model_samples,
                weights_bma=weights_bma,
                weights_loo=weights_loo,
                target_name=target,
            )

        # Epistemic uncertainty comparison
        console.rule("[bold magenta]Epistemic Uncertainty Comparison[/bold magenta]", style="magenta")

        epi_table = Table(
            title="Between-Model Variance (Unbiased)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta"
        )
        epi_table.add_column("Target", style="cyan")
        epi_table.add_column("BMA Epistemic Var", style="yellow", justify="right")
        epi_table.add_column("LOO Epistemic Var", style="green", justify="right")
        epi_table.add_column("Difference", style="magenta", justify="right")

        for target in final_targets:
            epi_bma = np.asarray(epistemic_uncertainty_bma[target], dtype=np.float64)
            epi_loo = np.asarray(epistemic_uncertainty_loo[target], dtype=np.float64)

            if epi_bma.ndim == 0 and epi_loo.ndim == 0:
                diff = float(epi_loo) - float(epi_bma)
                epi_table.add_row(
                    target,
                    f"{float(epi_bma):.6f}",
                    f"{float(epi_loo):.6f}",
                    f"{diff:+.6f}"
                )
            else:
                epi_table.add_row(
                    target,
                    f"shape: {epi_bma.shape}",
                    f"shape: {epi_loo.shape}",
                    "—"
                )

        console.print(epi_table)
        console.print()

    # Compute posteriors
    draws_per_model = len(per_model_target_samples[0][final_targets[0]])
    total_draws = draws_per_model * len(per_model_target_samples)

    posterior_bma = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=weights_bma,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed),
    )

    posterior_loo = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=weights_loo,
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed + 1),
    )

    posterior_flat = _resample_weighted_samples(
        per_model_target_samples,
        final_targets,
        model_weights=np.ones(len(per_model_target_samples)) / len(per_model_target_samples),
        total_draws=total_draws,
        rng=np.random.default_rng(base_seed + 2),
    )

    if auto_print_result:
        returned_targets = _resolve_report_targets([posterior_loo], report_targets)
        _print_posterior_comparison(posterior_bma, posterior_loo, posterior_flat, returned_targets)

    return {
        "posterior_bma": posterior_bma,
        "posterior_loo": posterior_loo,
        "posterior_flat": posterior_flat,
        "epistemic_uncertainty_bma": epistemic_uncertainty_bma,
        "epistemic_uncertainty_loo": epistemic_uncertainty_loo,
        "epistemic_uncertainty_uniform": epistemic_uncertainty_uniform,
        "weights_bma": weights_bma,
        "weights_loo": weights_loo,
        "weights_uniform": flat_weights,
        "diagnostics": diagnostics,
        "final_loo_objective": float(final_loo_objective),
        "log_marginal_per_model": log_bounds.tolist(),
        "loo_diagnostics_per_model": [v.get("loo_diagnostics") for v in valid],
        "model_codes": all_generated_codes,
    }

def _print_dual_model_averaging_summary(samples, weights_bma, weights_loo, target_name):
    """Print comparison of BMA vs LOO weights."""
    if len(samples) == 0:
        print("Number of models: 0")
        return

    mu_per_model = [np.asarray(_target_mean(np.asarray(s, dtype=np.float64)), dtype=np.float64) for s in samples]

    # Handle shape mismatches
    shape_groups = {}
    for idx, mu in enumerate(mu_per_model):
        key = tuple(mu.shape)
        shape_groups.setdefault(key, []).append(idx)

    dominant_shape = max(shape_groups, key=lambda k: len(shape_groups[k]))
    keep_idx = shape_groups[dominant_shape]
    dropped_idx = [i for i in range(len(mu_per_model)) if i not in keep_idx]

    mu_stack = np.stack([mu_per_model[i] for i in keep_idx], axis=0)
    kept_weights_bma = np.asarray([weights_bma[i] for i in keep_idx], dtype=np.float64)
    kept_weights_bma = kept_weights_bma / np.sum(kept_weights_bma)
    kept_weights_loo = np.asarray([weights_loo[i] for i in keep_idx], dtype=np.float64)
    kept_weights_loo = kept_weights_loo / np.sum(kept_weights_loo)

    mu_flat = np.mean(mu_stack, axis=0)
    mu_bma = np.tensordot(kept_weights_bma, mu_stack, axes=(0, 0))
    mu_loo = np.tensordot(kept_weights_loo, mu_stack, axes=(0, 0))

    console.print(f"Number of models: {len(mu_per_model)}")
    if dropped_idx:
        shape_counts = {str(k): len(v) for k, v in shape_groups.items()}
        console.print(f"[yellow]Shape mismatch detected; using dominant shape {dominant_shape}.[/yellow]")
        print(f"Shape counts: {shape_counts}")

    if mu_flat.ndim == 0:
        console.print(f"Flat mean: [dim]{float(mu_flat):.6f}[/dim]")
        console.print(f"BMA mean: [yellow]{float(mu_bma):.6f}[/yellow]")
        console.print(f"LOO mean: [green]{float(mu_loo):.6f}[/green]")
        diff = float(mu_loo) - float(mu_bma)
        console.print(f"Difference (LOO - BMA): [magenta]{diff:+.6f}[/magenta]")
    else:
        console.print(f"Flat mean shape: {mu_flat.shape}")
        console.print(f"BMA mean shape: {mu_bma.shape}")
        console.print(f"LOO mean shape: {mu_loo.shape}")

    print()

    # Top models table
    console.print("[bold]Top 5 models by LOO weight:[/bold]")
    ranked_loo = np.argsort(-kept_weights_loo)
    for local_idx in ranked_loo[:5]:
        rank_idx = keep_idx[int(local_idx)]
        w_bma = float(weights_bma[rank_idx])
        w_loo = float(weights_loo[rank_idx])
        mu_i = np.asarray(mu_per_model[rank_idx], dtype=np.float64)

        if mu_i.ndim == 0:
            console.print(f"  model={rank_idx}, BMA=[yellow]{w_bma:.6f}[/yellow], LOO=[green]{w_loo:.6f}[/green], mu_i={float(mu_i):.6f}")
        else:
            preview = mu_i.reshape(-1)[:3].tolist()
            console.print(f"  model={rank_idx}, BMA=[yellow]{w_bma:.6f}[/yellow], LOO=[green]{w_loo:.6f}[/green], mu_i={preview}...")

    print()


def _print_posterior_comparison(posterior_bma, posterior_loo, posterior_flat, targets):
    """Compare posterior means across all three methods."""
    console.rule("[bold blue]Posterior Mean Comparison[/bold blue]", style="blue")

    comp_table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
    comp_table.add_column("Target", style="cyan")
    comp_table.add_column("Flat", style="dim", justify="right")
    comp_table.add_column("BMA", style="yellow", justify="right")
    comp_table.add_column("LOO", style="green", justify="right")
    comp_table.add_column("LOO - BMA", style="magenta", justify="right")

    for target in targets:
        flat_arr = np.asarray(posterior_flat.get(target, []), dtype=np.float64)
        bma_arr = np.asarray(posterior_bma.get(target, []), dtype=np.float64)
        loo_arr = np.asarray(posterior_loo.get(target, []), dtype=np.float64)

        flat_mean = _target_mean(flat_arr)
        bma_mean = _target_mean(bma_arr)
        loo_mean = _target_mean(loo_arr)

        if np.asarray(flat_mean).ndim == 0:
            diff = float(loo_mean) - float(bma_mean)

            # Color code based on magnitude
            if abs(diff) > 0.01:
                diff_str = f"[bold magenta]{diff:+.6f}[/bold magenta]"
            elif abs(diff) > 0.001:
                diff_str = f"[magenta]{diff:+.6f}[/magenta]"
            else:
                diff_str = f"[dim]{diff:+.6f}[/dim]"

            comp_table.add_row(
                target,
                f"{float(flat_mean):.6f}",
                f"{float(bma_mean):.6f}",
                f"{float(loo_mean):.6f}",
                diff_str
            )
        else:
            comp_table.add_row(
                target,
                f"shape: {np.asarray(flat_mean).shape}",
                f"shape: {np.asarray(bma_mean).shape}",
                f"shape: {np.asarray(loo_mean).shape}",
                "—"
            )

    console.print(comp_table)
    console.print()


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
    msg = (
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
    reason = diagnostics.get("first_failure_reason")
    if reason:
        msg += f" First failure reason: {reason}"
    return msg


def _first_request_failure_reason(gen_diag):
    failures = gen_diag.get("generation_failures", []) if isinstance(gen_diag, dict) else []
    for _slot_idx, reason in failures:
        if isinstance(reason, str) and not reason.startswith("parsing_error:"):
            return reason
    return None


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

    # Try to repair simple transposition errors: (event_dim, n_samples) -> (n_samples, event_dim)
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