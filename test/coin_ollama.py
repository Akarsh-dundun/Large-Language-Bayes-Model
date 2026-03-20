from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import llb

text = "I have a bunch of coin flips. What's the bias?"
data = {"flips": [0, 1, 0, 1, 1, 0]}
targets = ["true_bias"]

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:latest")

posterior = llb.infer(
    text,
    data,
    targets,
    api_url=OLLAMA_URL,
    api_key=None,
    api_model=OLLAMA_MODEL,
    n_models=8,
    mcmc_num_warmup=50,
    mcmc_num_samples=100,
)
