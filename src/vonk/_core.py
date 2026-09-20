"""Shared plumbing: config, response cleaning, retry policy.

Everything here is transport-agnostic so the sync and async clients can't drift.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

# vonk's chat models cap at 32768 tokens for input + output combined.
CONTEXT_LIMIT = 32768

_FENCE_RE = re.compile(r"^\s*```(?:json|javascript|python)?\s*|\s*```\s*$", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class VonkError(RuntimeError):
    """Any failure talking to vonk, including unparseable JSON replies."""


@dataclass(frozen=True)
class Config:
    base_url: str
    api_key: str
    chat_model: str
    embed_model: str
    image_model: str
    timeout: float

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build from VONK_* env vars. No key is ever baked in as a default."""
        cfg = {
            "base_url": os.getenv("VONK_BASE_URL", "http://vonk:4000/v1").rstrip("/"),
            "api_key": os.getenv("VONK_KEY", ""),
            "chat_model": os.getenv("VONK_MODEL", "code"),
            "embed_model": os.getenv("VONK_EMBED_MODEL", "embedding"),
            "image_model": os.getenv("VONK_IMAGE_MODEL", "image"),
            "timeout": float(os.getenv("VONK_TIMEOUT", "120")),
        }
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        if not cfg["api_key"]:
            raise VonkError(
                "No vonk API key. Set VONK_KEY in the environment, or pass api_key= explicitly."
            )
        return cls(**cfg)

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }


def clean(text: str) -> str:
    """Strip reasoning blocks and markdown fences models wrap around raw output.

    Local models ignore "no code fences" instructions often enough that every
    project here grew its own version of this; this is the shared one.
    """
    text = _THINK_RE.sub("", text or "")
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_RE.sub("", text)
    return text.strip()


def parse_json(text: str) -> dict | list:
    """Clean then parse. Falls back to the outermost {...} / [...] span."""
    cleaned = clean(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise VonkError(f"Model did not return JSON. Got: {cleaned[:200]!r}")


def chat_payload(
    prompt: str, *, system: str | None, model: str, max_tokens: int,
    temperature: float, json_mode: bool,
) -> dict:
    if max_tokens >= CONTEXT_LIMIT:
        raise VonkError(
            f"max_tokens={max_tokens} leaves no room for input; "
            f"vonk caps input+output at {CONTEXT_LIMIT}."
        )
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    return payload


def content_of(data: dict) -> str:
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as exc:
        raise VonkError(f"Unexpected chat response shape: {str(data)[:200]}") from exc
