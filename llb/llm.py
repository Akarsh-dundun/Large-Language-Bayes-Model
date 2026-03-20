import json
import time

import requests

class LLMClient:
    def __init__(
        self,
        api_url,
        api_key=None,
        model=None,
        provider="auto",
        extra_headers=None,
        timeout=120,
        max_retries=2,
        retry_backoff=2.0,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.provider = provider
        self.extra_headers = extra_headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def generate(self, prompt_or_messages):
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = self._build_payload(prompt_or_messages)
        attempts = int(self.max_retries) + 1
        last_exc = None
        for attempt in range(attempts):
            try:
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                payload = response.json()
                text = self._extract_text(payload)
                if not isinstance(text, str) or not text.strip():
                    keys = sorted(payload.keys()) if isinstance(payload, dict) else [type(payload).__name__]
                    raise RuntimeError(
                        "Unable to parse text from LLM response. "
                        f"provider={self._resolved_provider()} api_url={self.api_url} payload_keys={keys}"
                    )
                return text
            except (requests.ReadTimeout, requests.ConnectTimeout) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                raise RuntimeError(
                    "LLM request timed out. Increase llm_timeout or use a faster/smaller local model. "
                    f"api_url={self.api_url} timeout={self.timeout} attempts={attempts}"
                ) from exc
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(self.retry_backoff * (attempt + 1))
                    continue
                raise

        raise RuntimeError("LLM request failed after retries") from last_exc

    def _build_payload(self, prompt_or_messages):
        mode = self._resolved_provider()
        prompt = _flatten_prompt(prompt_or_messages)
        messages = _coerce_messages(prompt_or_messages)

        if mode == "openai_responses":
            input_value = messages if messages is not None else prompt
            return {
                "model": self.model or "gpt-4.1-mini",
                "input": input_value,
            }

        if mode == "openai_chat":
            chat_messages = messages if messages is not None else [{"role": "user", "content": prompt}]
            return {
                "model": self.model or "gpt-4.1-mini",
                "messages": chat_messages,
            }

        if mode == "ollama_generate":
            payload = {
                "prompt": prompt,
                "stream": False,
            }
            if self.model:
                payload["model"] = self.model
            return payload

        payload = {"prompt": prompt}
        if self.model:
            payload["model"] = self.model
        return payload

    def _resolved_provider(self):
        if self.provider != "auto":
            return self.provider

        url = self.api_url.lower()
        if "/v1/responses" in url:
            return "openai_responses"
        if "/v1/chat/completions" in url:
            return "openai_chat"
        if "/api/generate" in url:
            return "ollama_generate"
        return "generic_prompt"

    def _extract_text(self, payload):
        if not isinstance(payload, dict):
            return None

        if payload.get("output_text"):
            return payload["output_text"]

        if "output" in payload:
            texts = []
            for item in payload["output"]:
                for content in item.get("content", []):
                    if content.get("type") == "output_text" and content.get("text"):
                        texts.append(content["text"])
            if texts:
                return "\n".join(texts)

        if "choices" in payload and payload["choices"]:
            msg = payload["choices"][0].get("message", {})
            content = msg.get("content")
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and item.get("text"):
                        parts.append(item["text"])
                if parts:
                    return "\n".join(parts)

        if payload.get("response"):
            return payload["response"]
        if payload.get("text"):
            return payload["text"]

        if payload.get("data") and isinstance(payload["data"], str):
            try:
                nested = json.loads(payload["data"])
                return self._extract_text(nested)
            except json.JSONDecodeError:
                return None

        return None


def _coerce_messages(prompt_or_messages):
    if isinstance(prompt_or_messages, list):
        return prompt_or_messages
    return None


def _flatten_prompt(prompt_or_messages):
    if isinstance(prompt_or_messages, str):
        return prompt_or_messages
    if isinstance(prompt_or_messages, list):
        parts = []
        for msg in prompt_or_messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(f"{role.upper()}: {content}")
        return "\n\n".join(parts)
    return str(prompt_or_messages)
