"""Small provider-neutral client for local Ollama and AICredits.

Secrets are read from environment variables only.  The client deliberately
does not log request headers or response bodies on errors.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    provider: str
    aicredits_api_key: str
    aicredits_base_url: str
    aicredits_model: str
    ollama_url: str
    ollama_model: str
    timeout_seconds: int

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "auto").strip().lower()
        api_key = os.getenv("AICREDITS_API_KEY", "").strip()
        if provider == "auto":
            provider = "aicredits" if api_key else "ollama"

        base_url = os.getenv(
            "AICREDITS_BASE_URL", "https://api.aicredits.in/v1"
        ).rstrip("/")
        if provider == "aicredits" and not base_url.endswith("/v1"):
            base_url += "/v1"

        return cls(
            provider=provider,
            aicredits_api_key=api_key,
            aicredits_base_url=base_url,
            aicredits_model=os.getenv(
                "AICREDITS_MODEL", "openai/gpt-4o-mini"
            ),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b"),
            timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        )


class LLMClient:
    """Use AICredits or Ollama through one simple chat interface."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()

    @classmethod
    def from_env(cls) -> "LLMClient":
        return cls(LLMConfig.from_env())

    @property
    def provider(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        if self.provider == "aicredits":
            return self.config.aicredits_model
        return self.config.ollama_model

    def is_configured(self) -> bool:
        if self.provider == "aicredits":
            return bool(self.config.aicredits_api_key)
        if self.provider == "ollama":
            return True
        return False

    def is_ready(self) -> bool:
        if not self.is_configured():
            return False
        if self.provider == "aicredits":
            return True
        try:
            response = requests.get(
                f"{self.config.ollama_url}/api/tags", timeout=3
            )
            return response.ok
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 768,
        json_mode: bool = False,
    ) -> str:
        if self.provider == "aicredits":
            return self._chat_aicredits(messages, temperature, max_tokens, json_mode)
        if self.provider == "ollama":
            return self._chat_ollama(messages, temperature, max_tokens, json_mode)
        raise RuntimeError(
            "Unsupported LLM_PROVIDER. Use 'aicredits', 'ollama', or 'auto'."
        )

    def _chat_aicredits(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        if not self.config.aicredits_api_key:
            raise RuntimeError("AICREDITS_API_KEY is not configured")

        payload: Dict[str, Any] = {
            "model": self.config.aicredits_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(
                f"{self.config.aicredits_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.aicredits_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"AICredits connection failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(f"AICredits request failed with HTTP {response.status_code}")
        try:
            return response.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("AICredits returned an unexpected response") from exc

    def _chat_ollama(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        prompt = "\n\n".join(
            f"{message['role'].upper()}: {message['content']}"
            for message in messages
        ) + "\n\nASSISTANT:"
        payload: Dict[str, Any] = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Ollama connection failed: {exc}") from exc

        if not response.ok:
            raise RuntimeError(f"Ollama request failed with HTTP {response.status_code}")
        try:
            return response.json().get("response", "").strip()
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Ollama returned an unexpected response") from exc


def extract_json(text: str) -> Any:
    """Parse JSON even when a model wraps it in a Markdown code fence."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    candidates = []
    for opening, closing in (("{", "}"), ("[", "]")):
        start = cleaned.find(opening)
        end = cleaned.rfind(closing)
        if start >= 0 and end > start:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError("The model response did not contain valid JSON")

