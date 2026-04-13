import llb
import json
import time
from pathlib import Path
import numpy as np

# Experiment configuration
text = "I have a bunch of coin flips. What's the bias?"
data = {"flips": [0, 1, 0, 1, 1, 0]}
targets = ["true_bias"]

EXPERIMENTS = [
    {
        "name": "qwen2.5-coder",
        "api_url": "http://localhost:11434/api/generate",
        "api_key": None,
        "api_model": "qwen2.5-coder:latest",
    },
    {
        "name": "deepseek-coder",  # CHANGE THIS TO YOUR SECOND MODEL
        "api_url": "http://localhost:11434/api/generate",
        "api_key": None,
        "api_model": "deepseek-coder:6.7b",  # CHANGE THIS
    },
]

N_MODELS_LIST = [10, 20, 50, 100]
MCMC_WARMUP = 500
MCMC_SAMPLES = 1000

# Create results directory
results_dir = Path("experiment_results")
results_dir.mkdir(exist_ok=True)

def serialize_result(result):
    """Convert numpy arrays to lists for JSON serialization."""
    serialized = {}
    for key, value in result.items():
        if isinstance(value, dict):
            # Handle nested dicts (epistemic_uncertainty_*, diagnostics)
            serialized[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.float32, np.float64, np.int64)) else v
                for k, v in value.items()
            }
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.float32, np.float64, np.int64, np.int32)):
            serialized[key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
        else:
            serialized[key] = value
    return serialized

def run_experiment(llm_config, n_models):
    """Run a single experiment and return results."""
    print(f"\n{'='*80}")
    print(f"Running: {llm_config['name']} with {n_models} models")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    try:
        result = llb.infer(
            text,
            data,
            targets,
            api_url=llm_config["api_url"],
            api_key=llm_config["api_key"],
            api_model=llm_config["api_model"],
            n_models=n_models,
            verbose=True,
            mcmc_num_warmup=MCMC_WARMUP,
            mcmc_num_samples=MCMC_SAMPLES,
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract diagnostics
        diag = result["diagnostics"]
        
        # Extract key metrics
        metrics = {
            "llm_name": llm_config["name"],
            "n_models_requested": n_models,
            "elapsed_time_seconds": elapsed_time,
            
            # Model generation diagnostics
            "n_models_generated": diag["generated_models"],
            "n_models_deduplicated": diag["deduplicated_models"],
            "n_models_invalid_syntax": diag["invalid_models_syntax_or_parsing"],
            "n_models_generation_failures": diag["generation_request_failures"],
            "n_models_missing_targets": diag["missing_targets_failures"],
            "n_models_compile_failures": diag["compile_failures"],
            "n_models_inference_failures": diag["inference_failures"],
            "n_models_shape_mismatch": diag["shape_mismatch_drops"],
            "n_models_nonfinite_log_bound": diag["nonfinite_log_bound_drops"],
            "n_models_valid_final": diag["valid_models_final"],
            
            # Success rate
            "valid_model_rate": diag["valid_models_final"] / n_models if n_models > 0 else 0,
            
            # Epistemic uncertainties
            "epistemic_var_uniform": float(result["epistemic_uncertainty_uniform"]["true_bias"]),
            "epistemic_var_bma": float(result["epistemic_uncertainty_bma"]["true_bias"]),
            "epistemic_var_loo": float(result["epistemic_uncertainty_loo"]["true_bias"]),
            
            # Weights
            "weights_uniform": result["weights_uniform"].tolist(),
            "weights_bma": result["weights_bma"].tolist(),
            "weights_loo": result["weights_loo"].tolist(),
            
            # Weight statistics
            "entropy_uniform": float(-np.sum(result["weights_uniform"] * np.log(result["weights_uniform"] + 1e-10))),
            "entropy_bma": float(-np.sum(result["weights_bma"] * np.log(result["weights_bma"] + 1e-10))),
            "entropy_loo": float(-np.sum(result["weights_loo"] * np.log(result["weights_loo"] + 1e-10))),
            
            "ess_uniform": float(1 / np.sum(result["weights_uniform"] ** 2)),
            "ess_bma": float(1 / np.sum(result["weights_bma"] ** 2)),
            "ess_loo": float(1 / np.sum(result["weights_loo"] ** 2)),
            
            "l1_distance_loo_bma": float(np.sum(np.abs(result["weights_loo"] - result["weights_bma"]))),
            
            # Posterior means
            "posterior_mean_uniform": float(np.mean(result["posterior_flat"]["true_bias"])),
            "posterior_mean_bma": float(np.mean(result["posterior_bma"]["true_bias"])),
            "posterior_mean_loo": float(np.mean(result["posterior_loo"]["true_bias"])),
        }
        
        print(f"\n✓ Completed in {elapsed_time:.1f}s")
        print(f"  Models: Requested={n_models}, Generated={diag['generated_models']}, Valid={diag['valid_models_final']} ({metrics['valid_model_rate']:.1%})")
        print(f"  Failures: Syntax={diag['invalid_models_syntax_or_parsing']}, Compile={diag['compile_failures']}, Inference={diag['inference_failures']}")
        print(f"  Epistemic Var: Uniform={metrics['epistemic_var_uniform']:.6f}, BMA={metrics['epistemic_var_bma']:.6f}, LOO={metrics['epistemic_var_loo']:.6f}")
        print(f"  ESS: Uniform={metrics['ess_uniform']:.2f}, BMA={metrics['ess_bma']:.2f}, LOO={metrics['ess_loo']:.2f}")
        
        return {
            "success": True,
            "metrics": metrics,
            "full_result": serialize_result(result),
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ Failed after {elapsed_time:.1f}s: {e}")
        
        return {
            "success": False,
            "error": str(e),
            "elapsed_time_seconds": elapsed_time,
        }

# Main experiment loop
all_results = []

for llm_config in EXPERIMENTS:
    for n_models in N_MODELS_LIST:
        # Run experiment
        result = run_experiment(llm_config, n_models)
        
        # Add metadata
        result["llm_name"] = llm_config["name"]
        result["n_models"] = n_models
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Store result
        all_results.append(result)
        
        # Save intermediate results after each run
        output_file = results_dir / f"results_{llm_config['name']}_n{n_models}.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to: {output_file}")

# Save combined results
combined_file = results_dir / "all_results.json"
with open(combined_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*80}")
print(f"All experiments complete! Results saved to: {combined_file}")
print(f"{'='*80}")

# Print summary table
print("\n" + "="*120)
print("SUMMARY TABLE")
print("="*120)
print(f"{'LLM':<20} {'Req':<5} {'Gen':<5} {'Valid':<5} {'Rate':<7} {'Time(s)':<8} {'Epi_Uni':<10} {'Epi_BMA':<10} {'Epi_LOO':<10} {'L1':<8}")
print("-"*120)

for r in all_results:
    if r["success"]:
        m = r["metrics"]
        print(f"{r['llm_name']:<20} {m['n_models_requested']:<5} {m['n_models_generated']:<5} {m['n_models_valid_final']:<5} "
              f"{m['valid_model_rate']:<7.1%} {m['elapsed_time_seconds']:<8.1f} "
              f"{m['epistemic_var_uniform']:<10.6f} {m['epistemic_var_bma']:<10.6f} "
              f"{m['epistemic_var_loo']:<10.6f} {m['l1_distance_loo_bma']:<8.4f}")
    else:
        print(f"{r['llm_name']:<20} {r['n_models']:<5} FAILED: {r['error']}")

print("\n" + "="*120)
print("FAILURE BREAKDOWN")
print("="*120)
print(f"{'LLM':<20} {'N':<5} {'Syntax':<8} {'GenFail':<8} {'Compile':<8} {'Infer':<8} {'Shape':<8} {'Dedup':<8}")
print("-"*120)

for r in all_results:
    if r["success"]:
        m = r["metrics"]
        print(f"{r['llm_name']:<20} {m['n_models_requested']:<5} "
              f"{m['n_models_invalid_syntax']:<8} {m['n_models_generation_failures']:<8} "
              f"{m['n_models_compile_failures']:<8} {m['n_models_inference_failures']:<8} "
              f"{m['n_models_shape_mismatch']:<8} {m['n_models_deduplicated']:<8}")