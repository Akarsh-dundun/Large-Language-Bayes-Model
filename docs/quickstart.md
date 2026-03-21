# Quickstart

## 1) Installation Guide

```bash
python -m pip install "git+https://github.com/HoangHiepCS/Large-Language-Bayes-Model.git"
python -c "import llb; print(llb)"
```

## 2) Run the coin example

```python
import llb

text = (
    "I have a coin. I've flipped it a bunch of times. I'm wondering what the true bias of the coin is. "
    "I just got the coin from the US mint, so I'm almost completely sure that it's a standard US penny."
)
data = {"num_flips": 20, "num_heads": 14}
targets = ["bias"]

API_KEY = 'paste_your_openai_api_key_here'
API_MODEL = "gpt-4.1-mini"

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/responses",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models=2,
    mcmc_num_warmup=50,
    mcmc_num_samples=100,
)
)
```
