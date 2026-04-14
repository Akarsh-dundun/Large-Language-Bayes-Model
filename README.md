# Large Language Bayes - LOO-ELBO Stacking

Implementation of true LOO-ELBO stacking for Bayesian model aggregation with LLM-generated models.

## Setup

### 1. Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install Ollama and Download Model
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download qwen2.5-coder 32B
ollama pull qwen2.5-coder:32b
```

## Running Experiments

### 1. Run Experiments
```bash
python trial.py
```

**Expected runtime:** ~2-4 hours (depending on your machine)

This will:
- Generate models using qwen2.5-coder
- Run MCMC inference on each model
- Compute BMA and LOO weights
- Save results to `experiment_results/`

### 2. Generate Plots
```bash
python analysis.py
```

This creates visualizations in `experiment_results/`:
- `experiment_plots.png` - Main 6-panel figure
- `marginal_likelihood_distributions.png` - log p(x|m) distributions
- `bma_vs_loo_weights_scatter.png` - Weight comparison
- `weight_distributions.png` - Weight bar charts
- `elbo_convergence_model_*.png` - ELBO diagnostics
- `summary_table.csv` - All metrics

## Results

All outputs are saved to `experiment_results/`:
- JSON files with detailed results
- PNG plots for the paper
- CSV summary table

---
