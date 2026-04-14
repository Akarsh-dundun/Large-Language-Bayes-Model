import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Load results
results_dir = Path("experiment_results")
with open(results_dir / "all_results.json") as f:
    all_results = json.load(f)

def plot_marginal_likelihood_distribution(results_dir):
    """
    Plot the distribution of log marginal likelihoods log p(x|m) across models.
    Shows which models have strong evidence vs weak evidence.
    """
    print("\n" + "="*80)
    print("PLOTTING MARGINAL LIKELIHOOD DISTRIBUTIONS")
    print("="*80)
    
    with open(results_dir / "all_results.json") as f:
        all_results = json.load(f)
    
    # Create figure with subplots for each experiment
    n_experiments = sum(1 for r in all_results if r["success"])
    n_cols = 2
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_experiments > 1 else [axes]
    
    plot_idx = 0
    
    for r in all_results:
        if not r["success"]:
            continue
        
        # Extract log marginal likelihoods
        if "full_result" in r:
            log_marginals = r["full_result"]["log_marginal_per_model"]
        else:
            log_marginals = r["metrics"]["log_marginal_per_model"]
        
        # Filter out fallback values
        log_marginals = np.array([lm for lm in log_marginals if lm > -1e10])
        
        if len(log_marginals) == 0:
            print(f"No valid log marginals for {r['llm_name']}, n={r['n_models']}")
            plot_idx += 1
            continue
        
        ax = axes[plot_idx]
        
        # Histogram
        n, bins, patches = ax.hist(
            log_marginals, 
            bins=20, 
            alpha=0.7, 
            color='steelblue',
            edgecolor='black',
            linewidth=1.2
        )
        
        # Add vertical line for mean
        mean_log_marginal = np.mean(log_marginals)
        ax.axvline(mean_log_marginal, color='red', linestyle='--', linewidth=2.5,
                   label=f'Mean: {mean_log_marginal:.2f}')
        
        # Add vertical line for max
        max_log_marginal = np.max(log_marginals)
        ax.axvline(max_log_marginal, color='green', linestyle='--', linewidth=2.5,
                   label=f'Max: {max_log_marginal:.2f}')
        
        # Styling
        ax.set_xlabel('log p(x | m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Number of Models', fontsize=11, fontweight='bold')
        ax.set_title(
            f"{r['llm_name']}, N={r['n_models']} (Valid={len(log_marginals)})",
            fontsize=12,
            fontweight='bold'
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics text box
        stats_text = (
            f"Range: [{np.min(log_marginals):.2f}, {np.max(log_marginals):.2f}]\n"
            f"Std: {np.std(log_marginals):.2f}\n"
            f"Median: {np.median(log_marginals):.2f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = results_dir / "marginal_likelihood_distributions.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()
    
    # Also create a comparison plot showing all experiments together
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for r in all_results:
        if not r["success"]:
            continue
        
        if "full_result" in r:
            log_marginals = r["full_result"]["log_marginal_per_model"]
        else:
            log_marginals = r["metrics"]["log_marginal_per_model"]
        
        log_marginals = np.array([lm for lm in log_marginals if lm > -1e10])
        
        if len(log_marginals) == 0:
            continue
        
        # Kernel density estimate
        from scipy.stats import gaussian_kde
        
        if len(log_marginals) > 1:
            kde = gaussian_kde(log_marginals)
            x_range = np.linspace(log_marginals.min(), log_marginals.max(), 200)
            density = kde(x_range)
            
            label = f"{r['llm_name']}, N={r['n_models']}"
            ax.plot(x_range, density, linewidth=2.5, label=label, alpha=0.8)
            ax.fill_between(x_range, density, alpha=0.2)
    
    ax.set_xlabel('log p(x | m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Marginal Likelihood Distribution Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    output_path = results_dir / "marginal_likelihood_comparison.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved comparison to: {output_path}")
    plt.close()


def plot_bma_vs_loo_weights_scatter(results_dir):
    """
    Scatter plot comparing BMA weights vs LOO weights for each model.
    Shows how the two weighting schemes differ at the individual model level.
    """
    print("\n" + "="*80)
    print("PLOTTING BMA vs LOO WEIGHT SCATTER")
    print("="*80)
    
    with open(results_dir / "all_results.json") as f:
        all_results = json.load(f)
    
    # Create figure with subplots for each experiment
    n_experiments = sum(1 for r in all_results if r["success"])
    n_cols = 2
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_experiments > 1 else [axes]
    
    plot_idx = 0
    
    for r in all_results:
        if not r["success"]:
            continue
        
        # Extract weights
        if "full_result" in r:
            weights_bma = np.array(r["full_result"]["weights_bma"])
            weights_loo = np.array(r["full_result"]["weights_loo"])
        else:
            weights_bma = np.array(r["metrics"]["weights_bma"])
            weights_loo = np.array(r["metrics"]["weights_loo"])
        
        ax = axes[plot_idx]
        
        # Scatter plot
        ax.scatter(weights_bma, weights_loo, s=100, alpha=0.6, 
                  c=np.arange(len(weights_bma)), cmap='viridis',
                  edgecolors='black', linewidth=1.5)
        
        # Add diagonal line (perfect agreement)
        max_weight = max(weights_bma.max(), weights_loo.max())
        ax.plot([0, max_weight], [0, max_weight], 'r--', linewidth=2,
               label='Perfect Agreement', alpha=0.7)
        
        # Add colorbar
        scatter = ax.scatter(weights_bma, weights_loo, s=100, alpha=0.6,
                           c=np.arange(len(weights_bma)), cmap='viridis',
                           edgecolors='black', linewidth=1.5)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Model Index', fontsize=10)
        
        # Styling
        ax.set_xlabel('BMA Weight', fontsize=11, fontweight='bold')
        ax.set_ylabel('LOO Weight', fontsize=11, fontweight='bold')
        ax.set_title(
            f"{r['llm_name']}, N={r['n_models']}",
            fontsize=12,
            fontweight='bold'
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics
        l1_distance = r["metrics"]["l1_distance_loo_bma"]
        correlation = np.corrcoef(weights_bma, weights_loo)[0, 1]
        
        stats_text = (
            f"L1 Distance: {l1_distance:.4f}\n"
            f"Correlation: {correlation:.4f}"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    output_path = results_dir / "bma_vs_loo_weights_scatter.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def plot_weight_distributions(results_dir):
    """
    Plot the distribution of BMA and LOO weights side-by-side.
    Shows how concentrated or dispersed the weights are.
    """
    print("\n" + "="*80)
    print("PLOTTING WEIGHT DISTRIBUTIONS")
    print("="*80)
    
    with open(results_dir / "all_results.json") as f:
        all_results = json.load(f)
    
    # Create figure with subplots for each experiment (2 columns: BMA and LOO)
    n_experiments = sum(1 for r in all_results if r["success"])
    
    fig, axes = plt.subplots(n_experiments, 2, figsize=(12, 4 * n_experiments))
    if n_experiments == 1:
        axes = axes.reshape(1, -1)
    
    for exp_idx, r in enumerate([r for r in all_results if r["success"]]):
        # Extract weights
        if "full_result" in r:
            weights_bma = np.array(r["full_result"]["weights_bma"])
            weights_loo = np.array(r["full_result"]["weights_loo"])
        else:
            weights_bma = np.array(r["metrics"]["weights_bma"])
            weights_loo = np.array(r["metrics"]["weights_loo"])
        
        # Sort for better visualization
        weights_bma_sorted = np.sort(weights_bma)[::-1]  # Descending
        weights_loo_sorted = np.sort(weights_loo)[::-1]
        
        model_indices = np.arange(len(weights_bma))
        
        # BMA plot
        ax_bma = axes[exp_idx, 0]
        bars_bma = ax_bma.bar(model_indices, weights_bma_sorted, 
                             color='gold', edgecolor='black', linewidth=1.2,
                             alpha=0.8)
        
        # Color top 3 bars differently
        for i in range(min(3, len(bars_bma))):
            bars_bma[i].set_color('orange')
        
        ax_bma.set_xlabel('Model Rank', fontsize=11, fontweight='bold')
        ax_bma.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax_bma.set_title(
            f"BMA Weights - {r['llm_name']}, N={r['n_models']}",
            fontsize=12,
            fontweight='bold'
        )
        ax_bma.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add entropy and ESS
        entropy_bma = r["metrics"]["entropy_bma"]
        ess_bma = r["metrics"]["ess_bma"]
        ax_bma.text(
            0.98, 0.98,
            f"Entropy: {entropy_bma:.3f}\nESS: {ess_bma:.2f}",
            transform=ax_bma.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # LOO plot
        ax_loo = axes[exp_idx, 1]
        bars_loo = ax_loo.bar(model_indices, weights_loo_sorted,
                             color='lightblue', edgecolor='black', linewidth=1.2,
                             alpha=0.8)
        
        # Color top 3 bars differently
        for i in range(min(3, len(bars_loo))):
            bars_loo[i].set_color('steelblue')
        
        ax_loo.set_xlabel('Model Rank', fontsize=11, fontweight='bold')
        ax_loo.set_ylabel('Weight', fontsize=11, fontweight='bold')
        ax_loo.set_title(
            f"LOO Weights - {r['llm_name']}, N={r['n_models']}",
            fontsize=12,
            fontweight='bold'
        )
        ax_loo.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # Add entropy and ESS
        entropy_loo = r["metrics"]["entropy_loo"]
        ess_loo = r["metrics"]["ess_loo"]
        ax_loo.text(
            0.98, 0.98,
            f"Entropy: {entropy_loo:.3f}\nESS: {ess_loo:.2f}",
            transform=ax_loo.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.tight_layout()
    output_path = results_dir / "weight_distributions.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()

# Extract successful results into DataFrame
data = []  # ← UNCOMMENT THIS LINE
for r in all_results:
    if r["success"]:
        m = r["metrics"]
        data.append({
            "LLM": r["llm_name"],
            "N_Requested": r["n_models"],
            "N_Generated": m["n_models_generated"],
            "N_Valid": m["n_models_valid_final"],
            "Valid_Rate": m["valid_model_rate"],
            "Time(s)": m["elapsed_time_seconds"],
            
            # Failures
            "Syntax_Fail": m["n_models_invalid_syntax"],
            "Gen_Fail": m["n_models_generation_failures"],
            "Compile_Fail": m["n_models_compile_failures"],
            "Inference_Fail": m["n_models_inference_failures"],
            "Dedup": m["n_models_deduplicated"],
            
            # Metrics
            "Epi_Uniform": m["epistemic_var_uniform"],
            "Epi_BMA": m["epistemic_var_bma"],
            "Epi_LOO": m["epistemic_var_loo"],
            "Entropy_BMA": m["entropy_bma"],
            "Entropy_LOO": m["entropy_loo"],
            "ESS_BMA": m["ess_bma"],
            "ESS_LOO": m["ess_loo"],
            "L1_Distance": m["l1_distance_loo_bma"],
        })

df = pd.DataFrame(data)
print(df.to_string())

# Save to CSV
df.to_csv(results_dir / "summary_table.csv", index=False)
print(f"\nSaved summary to: {results_dir / 'summary_table.csv'}")

# Create comprehensive plots
fig, axes = plt.subplots(3, 2, figsize=(14, 14))

# 1. Valid model rate
for llm in df["LLM"].unique():
    llm_data = df[df["LLM"] == llm]
    axes[0, 0].plot(llm_data["N_Requested"], llm_data["Valid_Rate"] * 100, 'o-', label=llm, linewidth=2)

axes[0, 0].set_xlabel("Models Requested", fontsize=11)
axes[0, 0].set_ylabel("Valid Model Rate (%)", fontsize=11)
axes[0, 0].set_title("Model Generation Success Rate", fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim([0, 105])

# 2. Epistemic uncertainty comparison
for llm in df["LLM"].unique():
    llm_data = df[df["LLM"] == llm]
    axes[0, 1].plot(llm_data["N_Valid"], llm_data["Epi_Uniform"], 'o-', label=f"{llm} (Uniform)", alpha=0.6)
    axes[0, 1].plot(llm_data["N_Valid"], llm_data["Epi_BMA"], 's-', label=f"{llm} (BMA)", linewidth=2)
    axes[0, 1].plot(llm_data["N_Valid"], llm_data["Epi_LOO"], '^-', label=f"{llm} (LOO)", linewidth=2)

axes[0, 1].set_xlabel("Valid Models", fontsize=11)
axes[0, 1].set_ylabel("Epistemic Variance", fontsize=11)
axes[0, 1].set_title("Epistemic Uncertainty vs Valid Models", fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 3. L1 distance (weight divergence)
for llm in df["LLM"].unique():
    llm_data = df[df["LLM"] == llm]
    axes[1, 0].plot(llm_data["N_Valid"], llm_data["L1_Distance"], 'o-', label=llm, linewidth=2)

axes[1, 0].set_xlabel("Valid Models", fontsize=11)
axes[1, 0].set_ylabel("L1 Distance (LOO - BMA)", fontsize=11)
axes[1, 0].set_title("Weight Divergence", fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. ESS comparison
for llm in df["LLM"].unique():
    llm_data = df[df["LLM"] == llm]
    axes[1, 1].plot(llm_data["N_Valid"], llm_data["ESS_BMA"], 's-', label=f"{llm} (BMA)", linewidth=2)
    axes[1, 1].plot(llm_data["N_Valid"], llm_data["ESS_LOO"], '^-', label=f"{llm} (LOO)", linewidth=2)

axes[1, 1].set_xlabel("Valid Models", fontsize=11)
axes[1, 1].set_ylabel("Effective Sample Size", fontsize=11)
axes[1, 1].set_title("ESS: BMA vs LOO", fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5. Failure breakdown (stacked bar)
llms = df["LLM"].unique()
x = range(len(df))
width = 0.35

failure_types = ["Syntax_Fail", "Gen_Fail", "Compile_Fail", "Inference_Fail", "Dedup"]
colors = ['#ff6b6b', '#ee5a6f', '#c44569', '#6c5ce7', '#a29bfe']

bottom = [0] * len(df)
for i, fail_type in enumerate(failure_types):
    axes[2, 0].bar(x, df[fail_type], width, label=fail_type.replace("_", " "), 
                   bottom=bottom, color=colors[i], alpha=0.8)
    bottom = [b + f for b, f in zip(bottom, df[fail_type])]

axes[2, 0].set_xlabel("Experiment Index", fontsize=11)
axes[2, 0].set_ylabel("Number of Failures", fontsize=11)
axes[2, 0].set_title("Model Failure Breakdown", fontsize=12, fontweight='bold')
axes[2, 0].legend(fontsize=9, loc='upper left')
axes[2, 0].grid(True, alpha=0.3, axis='y')

# 6. Runtime
for llm in df["LLM"].unique():
    llm_data = df[df["LLM"] == llm]
    axes[2, 1].plot(llm_data["N_Requested"], llm_data["Time(s)"], 'o-', label=llm, linewidth=2)

axes[2, 1].set_xlabel("Models Requested", fontsize=11)
axes[2, 1].set_ylabel("Runtime (seconds)", fontsize=11)
axes[2, 1].set_title("Computational Cost", fontsize=12, fontweight='bold')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(results_dir / "experiment_plots.png", dpi=200, bbox_inches='tight')
print(f"Saved plots to: {results_dir / 'experiment_plots.png'}")
plt.show()

# Print detailed statistics
print("\n" + "="*80)
print("DETAILED STATISTICS")
print("="*80)

for llm in df["LLM"].unique():
    llm_data = df[df["LLM"] == llm]
    print(f"\n{llm}:")
    print(f"  Average valid rate: {llm_data['Valid_Rate'].mean():.1%}")
    print(f"  Total failures: {llm_data['Syntax_Fail'].sum() + llm_data['Gen_Fail'].sum() + llm_data['Compile_Fail'].sum() + llm_data['Inference_Fail'].sum()}")
    print(f"  Most common failure: ", end="")
    
    failure_cols = ['Syntax_Fail', 'Gen_Fail', 'Compile_Fail', 'Inference_Fail']
    max_fail = max([(col, llm_data[col].sum()) for col in failure_cols], key=lambda x: x[1])
    print(f"{max_fail[0]} ({max_fail[1]} total)")


# ============================================================================
# ELBO CONVERGENCE PLOTS
# ============================================================================
def plot_elbo_convergence(results_file, output_dir):
    """Plot ELBO convergence for each datapoint across models."""
    print("\n" + "="*80)
    print("PLOTTING ELBO CONVERGENCE")
    print("="*80)
    
    with open(results_file) as f:
        result = json.load(f)
    
    # Check if this is a single result or full_result
    if "full_result" in result:
        result = result["full_result"]
    
    loo_diagnostics = result.get("loo_diagnostics_per_model", [])
    
    if not loo_diagnostics:
        print("No LOO diagnostics found (using PSIS-LOO or feature disabled)")
        return
    
    # Filter for models with true LOO ELBO
    models_with_elbo = []
    for model_idx, diag in enumerate(loo_diagnostics):
        if diag and diag.get('method') == 'true_loo_elbo':
            models_with_elbo.append((model_idx, diag))
    
    if not models_with_elbo:
        print("No models with ELBO histories found (using PSIS-LOO)")
        return
    
    print(f"Found {len(models_with_elbo)} models with ELBO training histories")
    
    # Plot for each model (limit to first 5 for readability)
    for model_idx, diag in models_with_elbo[:5]:
        elbo_histories = diag['elbo_histories']
        n_datapoints = len(elbo_histories)
        
        if n_datapoints == 0:
            continue
        
        # Determine grid size
        n_rows = (n_datapoints + 1) // 2
        n_cols = min(2, n_datapoints)
        
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(12, 3 * n_rows),
            squeeze=False
        )
        axes = axes.flatten()
        
        for i, elbo_hist in enumerate(elbo_histories):
            if len(elbo_hist) == 0:
                axes[i].text(0.5, 0.5, 'Failed', ha='center', va='center', fontsize=14)
                axes[i].set_title(f'Datapoint {i}', fontweight='bold')
                axes[i].set_xticks([])
                axes[i].set_yticks([])
                continue
            
            # Plot raw ELBO samples
            axes[i].plot(elbo_hist, 'o', alpha=0.4, markersize=4, label='ELBO samples')
            
            # Plot running mean
            running_mean = np.cumsum(elbo_hist) / (np.arange(len(elbo_hist)) + 1)
            axes[i].plot(running_mean, 'r-', linewidth=2.5, label='Running mean')
            
            # Final estimate line
            final_val = np.mean(elbo_hist)
            axes[i].axhline(final_val, color='green', linestyle='--', linewidth=2,
                           label=f'Final: {final_val:.3f}')
            
            # Styling
            axes[i].set_xlabel('Monte Carlo Iteration', fontsize=10)
            axes[i].set_ylabel('ELBO Value', fontsize=10)
            axes[i].set_title(f'Datapoint {i} (n={len(elbo_hist)} samples)', fontweight='bold')
            axes[i].legend(fontsize=8, loc='best')
            axes[i].grid(True, alpha=0.3, linestyle='--')
            
            # Add variance annotation
            std_val = np.std(elbo_hist)
            axes[i].text(0.02, 0.98, f'σ = {std_val:.4f}', 
                        transform=axes[i].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide extra subplots
        for i in range(n_datapoints, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(f'ELBO Convergence - Model {model_idx}', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        output_path = output_dir / f"elbo_convergence_model_{model_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {output_path.name}")
        plt.close()
    
    # Summary plot: ELBO variance across datapoints
    print("\nCreating ELBO variance summary plot...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model_idx, diag in models_with_elbo[:10]:  # Limit to 10 models
        elbo_histories = diag['elbo_histories']
        stds = [np.std(hist) if len(hist) > 0 else 0 for hist in elbo_histories]
        
        ax.plot(stds, 'o-', label=f'Model {model_idx}', alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Datapoint Index', fontsize=12)
    ax.set_ylabel('ELBO Standard Deviation', fontsize=12)
    ax.set_title('ELBO Estimation Variance Across Datapoints', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "elbo_variance_summary.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path.name}")
    plt.close()


# Generate ELBO plots for all experiments
print("\n" + "="*80)
print("GENERATING ELBO CONVERGENCE PLOTS")
print("="*80)

for r in all_results:
    if r["success"]:
        llm_name = r["llm_name"]
        n_models = r["n_models"]
        result_file = results_dir / f"results_{llm_name}_n{n_models}.json"
        
        if result_file.exists():
            print(f"\nProcessing: {result_file.name}")
            plot_elbo_convergence(result_file, results_dir)

# ADD THESE NEW PLOTTING FUNCTIONS
print("\n" + "="*80)
print("GENERATING ADDITIONAL ANALYSIS PLOTS")
print("="*80)

plot_marginal_likelihood_distribution(results_dir)
plot_bma_vs_loo_weights_scatter(results_dir)
plot_weight_distributions(results_dir)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)