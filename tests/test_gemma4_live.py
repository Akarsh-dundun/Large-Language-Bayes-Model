"""Live integration tests for Gemma 4 via Ollama.

These require a running Ollama server with gemma4:e4b pulled.
Run with:  uv run python -m pytest tests/test_gemma4_live.py -m live -v
Skip with: uv run python -m pytest -m "not live"
"""

import pytest
from llb.llm import LLMClient
from llb.model_generator import build_messages, extract_model_code

OLLAMA_URL = "http://localhost:11434/api/generate"
GEMMA4_MODEL = "gemma4:e4b"


@pytest.mark.live
def test_gemma4_simple_generation():
  llm = LLMClient(api_url=OLLAMA_URL, model=GEMMA4_MODEL, timeout=120)
  response = llm.generate("Say hello in one sentence.")
  assert isinstance(response, str)
  assert len(response.strip()) > 0


@pytest.mark.live
def test_gemma4_model_code_generation():
  llm = LLMClient(api_url=OLLAMA_URL, model=GEMMA4_MODEL, timeout=300)
  messages = build_messages(
    text="I have coin flips. What is the bias?",
    data={"flips": [0, 1, 0, 1, 1]},
    targets=["true_bias"],
  )
  raw = llm.generate(messages, seed=42)
  code = extract_model_code(raw)
  assert "def model(" in code
