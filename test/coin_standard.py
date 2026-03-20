import os

import llb

text = (
    "I have a coin. I've flipped it a bunch of times. I'm wondering what the true bias of the coin is. "
    "I just got the coin from the US mint, so I'm almost completely sure that it's a standard US penny."
)
data = {"num_flips": 20, "num_heads": 14}
targets = ["bias"]

API_KEY = os.environ.get("OPENAI_API_KEY")
API_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

posterior = llb.infer(
    text,
    data,
    targets,
    api_url="https://api.openai.com/v1/responses",
    api_key=API_KEY,
    api_model=API_MODEL,
    n_models=8,
    mcmc_num_warmup=50,
    mcmc_num_samples=100,
)

print(f"posterior draws for bias: {len(posterior)}")
