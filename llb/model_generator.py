import re

from .examples import PAPER_CHAT_EXAMPLES


def build_messages(text, data, targets):
    goal = targets if targets is not None else []
    messages = [
        {
            "role": "system",
            "content": (
                "You are a probabilistic programmer translating paper-style model descriptions into NumPyro. "
                "Return only Python code inside one fenced ```python block."
            ),
        },
    ]

    for i, ex in enumerate(PAPER_CHAT_EXAMPLES, start=1):
        messages.append(
            {
                "role": "user",
                "content": f"Example {i}\n{ex['input']}",
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": ex["output"],
            }
        )

    messages.append(
        {
            "role": "user",
            "content": (
                "Now solve this new task using the same style.\n\n"
                "REQUIREMENTS:\n"
                "1. Define exactly one function: def model(data):\n"
                "2. Include needed imports in the code block.\n"
                "3. Use unique site names across numpyro.sample / numpyro.deterministic / plates.\n"
                "4. Respect provided goal names exactly.\n"
                "5. Prefer direct translation fidelity to the paper style.\n"
                "6. For binary predictions, prefer numpyro.deterministic(name, probability) for goal outputs.\n"
                "   Do NOT define the same goal name both as numpyro.sample(...) and numpyro.deterministic(...).\n"
                "7. Do NOT introduce unobserved discrete latent sample sites (e.g., Bernoulli/Categorical without obs=...).\n"
                "8. Do NOT use sampled values or jax/jnp traced values as Python integers\n"
                "   (for example range(sampled_k), list[sampled_idx], int(sampled_x), or shape values that are traced).\n"
                "   Use static loop bounds from data fields like data['num_items'].\n"
                "9. Return only code in a single python fenced block.\n\n"
                "INPUT\n"
                f"PROBLEM\n{text}\n\n"
                f"DATA\n{data}\n\n"
                f"GOAL\n{goal}\n\n"
                "Return an OUTPUT with THOUGHT and MODEL."
            ),
        }
    )
    return messages


def extract_model_code(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError(
            "LLM returned empty or non-text output; check API URL/provider format and response parsing."
        )

    text = raw_text.strip()

    block_matches = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in block_matches:
        candidate = block.strip()
        if "def model(" in candidate:
            code = candidate
            # Ensure imports are present
            return _add_imports_if_needed(code)

    idx = text.find("def model(")
    if idx != -1:
        code = text[idx:].strip()
        return _add_imports_if_needed(code)

    lines = text.split('\n')
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith('def ') and 'model' in stripped:
            code = '\n'.join(lines[i:]).strip()
            return _add_imports_if_needed(code)

    return _add_imports_if_needed(text)


def _add_imports_if_needed(code):
    """Add required imports if they're not already present."""
    lines = code.split('\n')
    
    # Check what's already imported
    has_numpyro = any('import numpyro' in line for line in lines[:20])
    has_dist = any('numpyro.distributions' in line or 'import dist' in line for line in lines[:20])
    has_jnp = any('import jax.numpy as jnp' in line for line in lines[:30])
    has_np = any('import numpy as np' in line for line in lines[:30])

    uses_jnp = re.search(r'\bjnp\s*\.', code) is not None
    uses_np = re.search(r'\bnp\s*\.', code) is not None
    
    imports_needed = []
    if not has_numpyro:
        imports_needed.append("import numpyro")
    if not has_dist:
        imports_needed.append("import numpyro.distributions as dist")
    if uses_jnp and not has_jnp:
        imports_needed.append("import jax.numpy as jnp")
    if uses_np and not has_np:
        imports_needed.append("import numpy as np")
    
    if imports_needed:
        # Find where to insert imports (before the first def or at the start)
        def_idx = next((i for i, line in enumerate(lines) if line.strip().startswith('def ')), 0)
        for imp in reversed(imports_needed):
            lines.insert(def_idx, imp)
    
    return '\n'.join(lines)


def generate_models(llm, text, data, targets, n_models):
    models = []
    max_attempts_per_model = 4
    for _ in range(n_models):
        last_reason = None
        code = None
        for _attempt in range(max_attempts_per_model):
            messages = build_messages(text, data, targets)
            if last_reason is not None:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Previous output was invalid: "
                            f"{last_reason}. Regenerate with unique numpyro site names."
                        ),
                    }
                )

            candidate = extract_model_code(llm.generate(messages))
            duplicate_names = _duplicate_site_names(candidate)
            if duplicate_names:
                last_reason = f"duplicate site names: {', '.join(duplicate_names)}"
                continue

            code = candidate
            break

        if code is not None:
            models.append(code)

    return models


def _duplicate_site_names(code):
    pattern = r'numpyro\.(?:sample|deterministic)\(\s*["\']([^"\']+)["\']'
    seen = set()
    dup = set()
    for name in re.findall(pattern, code):
        if name in seen:
            dup.add(name)
        else:
            seen.add(name)
    return sorted(dup)
