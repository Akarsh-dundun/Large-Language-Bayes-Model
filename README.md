# Large-Language-Bayes-Model

Large-Language-Bayes-Model is a lightweight pipeline for Bayesian inference with LLM-generated NumPyro models.

The project does the following:
- Takes a natural-language problem description and user data.
- Prompts an LLM to generate candidate NumPyro models.
- Runs MCMC inference for each valid model.
- Scores models with an approximate log marginal likelihood.
- Builds a weighted posterior by resampling from model posteriors.

## Installation

```bash
pip install -e .
```

Or install from GitHub:

```bash
pip install git+https://github.com/HoangHiepCS/Large-Language-Bayes-Model.git
```

## Quick Start (OpenAI-Compatible API)

```python
import llb

text = "I have a bunch of coin flips. What is the true bias of the coin?"
data = {"flips": [0, 0, 1, 0, 1, 0, 0]}
targets = ["true"]

posterior_samples = llb.infer(
	text=text,
	data=data,
	targets=targets,
	api_url="https://api.openai.com/v1/chat/completions",
	api_key="YOUR_API_KEY",
	api_model="gpt-4.1-mini",
	llm_timeout=300,
)
```

Expected output shape:

```python
{"true": [0.61, 0.41, 0.58, ...]}
```

## Local Model Example (LM Studio / Ollama OpenAI-Compat)

```python
posterior_samples = llb.infer(
	text=text,
	data=data,
	targets=targets,
	api_url="http://127.0.0.1:1234/v1/chat/completions",
	api_key="not-needed",
	api_model="qwen/qwen3-4b",
	llm_timeout=600,
	llm_max_retries=3,
)
```

## Paper Alignment Notes

- In-context examples are supplied in a chat-template style (`user`/`assistant` turns), not embedded as one long system prompt.
- Each assistant example contains explicit `OUTPUT` blocks with `THOUGHT` and `MODEL`, matching the paper-style structure.
- Binary prediction goals are encouraged as deterministic outputs (e.g., predicted probability) to avoid unstable discrete latent targets.
- The implementation targets OpenAI-compatible HTTP APIs (OpenAI, LM Studio, Ollama OpenAI mode, OpenRouter).

### Differences From Paper (Explicit)

- Runtime safety checks skip invalid generated models rather than failing the entire run immediately.
- Local LLM usage may require larger timeouts (`llm_timeout=300` to `600`) depending on model size/hardware.
- Final posterior is a weighted mixture over valid generated models.

## Demos and Docs

- Docs: [docs/quickstart.md](docs/quickstart.md)
- Coin demo notebook: [notebooks/coin_demo.ipynb](notebooks/coin_demo.ipynb)
- Rain demo notebook: [notebooks/rain_demo.ipynb](notebooks/rain_demo.ipynb)
- Local model demo notebook: [notebooks/local_ollama_demo.ipynb](notebooks/local_ollama_demo.ipynb)
