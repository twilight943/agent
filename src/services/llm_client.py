from __future__ import annotations

import json
import re
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    pass


class LLMClient:
    """Small synchronous client for an OpenAI-compatible chat-completions API."""

    def __init__(self, config: dict[str, Any], temperature: float = 0.1) -> None:
        self.enabled = bool(config.get("enabled", True))
        self.base_url = str(config.get("base_url", "")).rstrip("/")
        self.api_key = str(config.get("api_key", "EMPTY"))
        self.model = str(config.get("model", ""))
        self.timeout = float(config.get("timeout_seconds", 60))
        self.temperature = float(temperature)

    @property
    def configured(self) -> bool:
        return self.enabled and bool(self.base_url and self.model)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        if not self.configured:
            raise LLMUnavailableError("LLM is disabled or its base_url/model is not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        if response_format:
            payload["response_format"] = response_format
        response = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])

    def generate(self, prompt: str, system_prompt: str | None = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    def structured_output(self, messages: list[dict[str, str]], schema: type[T]) -> T:
        try:
            content = self.chat(messages, response_format={"type": "json_object"})
        except httpx.HTTPStatusError:
            content = self.chat(messages)
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise ValueError("LLM response did not contain a JSON object")
        payload = json.loads(match.group(0))
        if hasattr(schema, "model_validate"):
            return schema.model_validate(payload)
        return schema.parse_obj(payload)
