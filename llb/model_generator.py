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
                "   For every name in GOAL, the model must create that exact variable name\n"
                "   (via numpyro.sample or numpyro.deterministic) so it appears in posterior samples.\n"
                "   Do not omit any GOAL variable.\n"
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


MAX_ATTEMPTS_PER_MODEL = 4


def generate_models(llm, text, data, targets, n_models, base_seed=None):
    models, _diag = generate_models_with_diagnostics(
        llm=llm,
        text=text,
        data=data,
        targets=targets,
        n_models=n_models,
        base_seed=base_seed,
    )
    return models


def generate_one_with_full_diagnostics(
    llm, text, data, targets,
    slot, base_seed=None, max_attempts=MAX_ATTEMPTS_PER_MODEL,
):
    """Single-slot LLM generation that records every retry verbatim.

    Returns a dict with::

        {
          "slot": int,
          "code": Optional[str],          # successful code or None
          "messages_used": list,          # prompt that produced the success or last attempt
          "raw_llm_response": str,        # raw text of the success or last attempt
          "attempts": [
            {
              "attempt": int,
              "seed": int,
              "raw_response": str,
              "status": "ok" | "generation_request_error" | "parsing_error",
              "reason": Optional[str],
            },
            ...
          ],
          "final_status": "ok" | "syntax_error" | "generation_failed",
          "final_reason": Optional[str],
        }

    No internal printing. The caller decides how to log.
    """
    attempts = []
    last_reason = None
    code = None
    last_messages = None
    last_raw = ""

    for attempt in range(max_attempts):
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
        last_messages = messages

        if base_seed is None:
            call_seed = None
        else:
            call_seed = int(base_seed) + slot * max_attempts + attempt

        try:
            raw = llm.generate(messages, seed=call_seed)
        except Exception as exc:
            last_reason = f"generation_request_error: {exc}"
            attempts.append({
                "attempt": attempt,
                "seed": call_seed,
                "raw_response": "",
                "status": "generation_request_error",
                "reason": last_reason,
            })
            continue

        last_raw = raw if isinstance(raw, str) else ""

        try:
            candidate = extract_model_code(raw)
        except Exception as exc:
            last_reason = f"parsing_error: {exc}"
            attempts.append({
                "attempt": attempt,
                "seed": call_seed,
                "raw_response": last_raw,
                "status": "parsing_error",
                "reason": last_reason,
            })
            continue

        duplicate_names = _duplicate_site_names(candidate)
        if duplicate_names:
            last_reason = f"parsing_error: duplicate site names: {', '.join(duplicate_names)}"
            attempts.append({
                "attempt": attempt,
                "seed": call_seed,
                "raw_response": last_raw,
                "status": "parsing_error",
                "reason": last_reason,
            })
            continue

        missing_goal_names = _missing_goal_names(candidate, targets)
        if missing_goal_names:
            last_reason = (
                "parsing_error: missing goal names: "
                + ", ".join(missing_goal_names)
            )
            attempts.append({
                "attempt": attempt,
                "seed": call_seed,
                "raw_response": last_raw,
                "status": "parsing_error",
                "reason": last_reason,
            })
            continue

        code = candidate
        attempts.append({
            "attempt": attempt,
            "seed": call_seed,
            "raw_response": last_raw,
            "status": "ok",
            "reason": None,
        })
        break

    if code is not None:
        final_status = "ok"
        final_reason = None
    elif last_reason is not None and last_reason.startswith("parsing_error:"):
        final_status = "syntax_error"
        final_reason = last_reason
    else:
        final_status = "generation_failed"
        final_reason = last_reason or "generation_request_error: unknown"

    return {
        "slot": int(slot),
        "code": code,
        "messages_used": last_messages or [],
        "raw_llm_response": last_raw,
        "attempts": attempts,
        "final_status": final_status,
        "final_reason": final_reason,
    }


def generate_models_with_diagnostics(
    llm, text, data, targets, n_models,
    base_seed=None, slot_offset=0, max_attempts_per_model=MAX_ATTEMPTS_PER_MODEL,
):
    """Backward-compatible wrapper that produces n_models codes via repeated
    calls to ``generate_one_with_full_diagnostics``.
    """
    models = []
    failures = []
    syntax_parsing_failures = 0
    request_failures = 0
    for slot_idx in range(n_models):
        global_slot = slot_offset + slot_idx
        result = generate_one_with_full_diagnostics(
            llm=llm,
            text=text,
            data=data,
            targets=targets,
            slot=global_slot,
            base_seed=base_seed,
            max_attempts=max_attempts_per_model,
        )
        if result["code"] is not None:
            models.append(result["code"])
        else:
            reason = result["final_reason"] or "generation_request_error: unknown"
            failures.append((global_slot, reason))
            if reason.startswith("parsing_error:"):
                syntax_parsing_failures += 1
            else:
                request_failures += 1

    diagnostics = {
        "requested_models": int(n_models),
        "generated_models": int(len(models)),
        "generation_failures": failures,
        "invalid_generation_count": int(len(failures)),
        "invalid_syntax_parsing_count": int(syntax_parsing_failures),
        "generation_request_failures": int(request_failures),
    }
    return models, diagnostics


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


def _missing_goal_names(code, targets):
    if not targets:
        return []
    declared = set(re.findall(r'numpyro\.(?:sample|deterministic)\(\s*["\']([^"\']+)["\']', code))
    missing = []
    for name in targets:
        if isinstance(name, str) and name not in declared:
            missing.append(name)
    return missing
