import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("experiment_results")
with open(results_dir / "all_results.json") as f:
    all_results = json.load(f)

# Extract successful results into DataFrame
data = []
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