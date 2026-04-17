"""Live integration tests for Llama 3.2 via Ollama.

These require a running Ollama server with llama3.2:latest pulled.
Run with:  uv run python -m pytest tests/test_llama32_live.py -m live -v
Skip with: uv run python -m pytest -m "not live"
"""

import pytest
from llb.llm import LLMClient
from llb.model_generator import build_messages, extract_model_code

OLLAMA_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.2:latest"


@pytest.mark.live
def test_llama32_simple_generation():
  llm = LLMClient(api_url=OLLAMA_URL, model=LLAMA_MODEL, timeout=120)
  response = llm.generate("Say hello in one sentence.")
  assert isinstance(response, str)
  assert len(response.strip()) > 0


@pytest.mark.live
def test_llama32_model_code_generation():
  llm = LLMClient(api_url=OLLAMA_URL, model=LLAMA_MODEL, timeout=300)
  messages = build_messages(
    text="I have coin flips. What is the bias?",
    data={"flips": [0, 1, 0, 1, 1]},
    targets=["true_bias"],
  )
  raw = llm.generate(messages, seed=42)
  code = extract_model_code(raw)
  assert "def model(" in code
