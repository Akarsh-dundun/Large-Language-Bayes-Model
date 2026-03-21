# Large-Language-Bayes-Model

Large-Language-Bayes-Model (LLB) is a package that performs Bayesian inference by combining:
- natural-language problem descriptions,
- LLM-generated NumPyro models,
- MCMC posterior inference, and
- model-weighted posterior aggregation.

## Installation

```bash
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/HoangHiepCS/Large-Language-Bayes-Model.git"
python -c "import llb; print(llb)"
```

## Quick Start (Hosted OpenAI-Compatible API)

```python
import llb

text = (
        "I have a coin. I've flipped it a bunch of times. I'm wondering what the true bias of the coin is. "
        "I just got the coin from the US mint, so I'm almost completely sure that it's a standard US penny."
)
data = {"num_flips": 20, "num_heads": 14}
targets = ["bias"]

API_KEY = "paste_your_openai_api_key_here"
API_MODEL = "gpt-4.1-mini"

posterior = llb.infer(
        text,
        data,
        targets,
        api_url="https://api.openai.com/v1/responses",
        api_key=API_KEY,
        api_model=API_MODEL,
)

for target in targets:
        print(f"posterior draws for {target}: {len(posterior[target])}")
        print(f"first 10 draws for {target}: {posterior[target][:10]}")
```

## Quick Start (Ollama Local)

```python
import llb

text = "I have a bunch of coin flips. What's the bias?"
data = {"flips": [0, 1, 0, 1, 1, 0]}
targets = ["true_bias"]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:latest"  # or whatever model you have set up in Ollama

posterior = llb.infer(
        text,
        data,
        targets,
        api_url=OLLAMA_URL,
        api_key=None,
        api_model=OLLAMA_MODEL,
        llm_timeout=600,
        llm_max_retries=3,
)
```

## `llb.infer` Main Parameters

- `text`: natural-language problem statement.
- `data`: observed data dictionary passed to generated models.
- `targets`: list of variable names to return.
- `api_url`: LLM endpoint URL.
- `api_key`: API key, if required by provider.
- `api_model`: provider model name.
- `n_models`: number of model candidates to generate (default `2`).
- `mcmc_num_warmup`: MCMC warmup steps (default `50`).
- `mcmc_num_samples`: posterior samples per model (default `100`).
- `llm_timeout`: HTTP timeout seconds for each LLM call (default `None`, which means no timeout).
- `llm_max_retries`: retries for transient timeout/network failures (default `2`).

## Local Timeout Troubleshooting

If you see:

`ReadTimeoutError: HTTPConnectionPool(host='127.0.0.1', port=1234): Read timed out.`

common causes and fixes are:

1. Local model generation is slower than your timeout.
Set `llm_timeout=600` and `llm_max_retries=3`.

2. Endpoint mismatch.
If you use Ollama native API, use `http://localhost:11434/api/generate`.
If you use OpenAI-compatible local servers, use their `/v1/chat/completions` URL.

3. Model too heavy for current hardware.
Try a smaller/faster model first.

## Notes

- `targets` should be a list of strings, for example `["bias"]`.
- Access posterior samples by string key, for example `posterior[targets[0]]`.
- For extended setup instructions, also see `docs/quickstart.md`.
