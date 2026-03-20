# Quickstart

## 1) Installation Guide

### Option A: Install from GitHub (recommended for reviewers)

```bash
python3 -m venv ~/venvs/llb
source ~/venvs/llb/bin/activate
python -m pip install --upgrade pip
python -m pip install "git+https://github.com/HoangHiepCS/Large-Language-Bayes-Model.git"
python -c "import llb; print(llb)"
```

### Option B: Install from local clone (editable mode for development)

```bash
git clone https://github.com/HoangHiepCS/Large-Language-Bayes-Model.git
cd Large-Language-Bayes-Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -c "import llb; print(llb)"
```

### macOS/Homebrew note (PEP 668)

If you see `externally-managed-environment`, install inside a virtual environment as above.
Avoid installing into the system Python.

Last-resort (not recommended):

```bash
python3 -m pip install --break-system-packages --user "git+https://github.com/HoangHiepCS/Large-Language-Bayes-Model.git"
```

## 2) Environment Variables

For OpenAI-compatible hosted APIs:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

For local servers (optional):

```bash
export LOCAL_LLM_URL="http://127.0.0.1:1234/v1/chat/completions"
export LOCAL_LLM_MODEL="qwen/qwen3-4b"
```

## 3) Run a basic example

```python
import llb
import os

text = "I have a bunch of coin flips. What's the true bias?"
data = {"flips": [0, 1, 0, 1, 1, 0, 0]}
targets = ["true"]

posterior = llb.infer(
    text=text,
    data=data,
    targets=targets,
    api_url="https://api.openai.com/v1/chat/completions",
        api_key=os.environ.get("OPENAI_API_KEY"),
        api_model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        llm_timeout=300,
)

print(posterior["true"][:10])
```

## 4) Local model (OpenAI-compatible server)

```python
import os

posterior = llb.infer(
    text=text,
    data=data,
    targets=targets,
        api_url=os.environ.get("LOCAL_LLM_URL", "http://127.0.0.1:1234/v1/chat/completions"),
    api_key="not-needed",
        api_model=os.environ.get("LOCAL_LLM_MODEL", "qwen/qwen3-4b"),
    llm_timeout=600,
    llm_max_retries=3,
)
```

## 5) Troubleshooting

- `ModuleNotFoundError: No module named 'llb'`
    - Activate the same virtual environment where you installed the package.
- `externally-managed-environment`
    - Use `python3 -m venv ...` and install inside that venv.
- `Read timed out`
    - Increase `llm_timeout` (for local models, 300-600 is common).
    - Use a smaller/faster local model.
- Connection refused on local URL
    - Start your local server first (LM Studio/Ollama).

## 6) What pyproject.toml does

`pyproject.toml` is the package definition for this project. It tells `pip`:

- How to build/install the package (`setuptools.build_meta`).
- Which Python version is required (`>=3.10`).
- Which dependencies to install (`numpy`, `requests`, `jax`, `jaxlib`, `numpyro`).
- Which package directory to expose (`llb`).

Without `pyproject.toml`, `pip install -e .` and GitHub installs are brittle or may fail on modern Python packaging tooling.