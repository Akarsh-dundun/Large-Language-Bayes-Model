import os

import llb

text = "I have a bunch of coin flips. What's the bias?"
data = {"flips": [0, 1, 0, 1, 1, 0]}
targets = ["true_bias"]

API_KEY = os.environ.get("OPENAI_API_KEY")
API_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/responses",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models = 16,
    mcmc_num_warmup=50,
    mcmc_num_samples=100
)
