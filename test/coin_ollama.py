
import llb

text = "I have a bunch of coin flips. What's the bias?"
data = {"flips": [0, 1, 0, 1, 1, 0]}
targets = ["true_bias"]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3:latest" # or whatever model you have set up in Ollama

posterior = llb.infer(
    text,
    data,
    targets,
    api_url=OLLAMA_URL,
    api_key=None,
    api_model=OLLAMA_MODEL,
)
