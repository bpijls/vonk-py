"""Sync and async clients for vonk. Same surface, same behaviour."""
from __future__ import annotations

import httpx

from ._core import (
    Config,
    VonkError,
    chat_payload,
    clean,
    content_of,
    parse_json,
)


class _Base:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        image_model: str | None = None,
        timeout: float | None = None,
        retries: int = 1,
    ) -> None:
        self.config = Config.from_env(
            base_url=base_url,
            api_key=api_key,
            chat_model=chat_model,
            embed_model=embed_model,
            image_model=image_model,
            timeout=timeout,
        )
        self.retries = retries


class Vonk(_Base):
    """Blocking client. Holds one connection pool; reuse the instance."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = httpx.Client(
            base_url=self.config.base_url,
            headers=self.config.headers,
            timeout=self.config.timeout,
        )

    def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                res = self._client.post(path, json=payload)
                res.raise_for_status()
                return res.json()
            except Exception as exc:  # noqa: BLE001 - retry on anything transient
                last = exc
        raise VonkError(f"vonk request to {path} failed: {last}") from last

    def chat(
        self, prompt: str, *, system: str | None = None, model: str | None = None,
        max_tokens: int = 1024, temperature: float = 0.7,
    ) -> str:
        payload = chat_payload(
            prompt, system=system, model=model or self.config.chat_model,
            max_tokens=max_tokens, temperature=temperature, json_mode=False,
        )
        return clean(content_of(self._post("/chat/completions", payload)))

    def json(
        self, prompt: str, *, system: str | None = None, model: str | None = None,
        max_tokens: int = 1024, temperature: float = 0.0,
    ) -> dict | list:
        """Ask for JSON and return it parsed. Raises VonkError if unparseable."""
        payload = chat_payload(
            prompt, system=system, model=model or self.config.chat_model,
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
        return parse_json(content_of(self._post("/chat/completions", payload)))

    def embed(self, texts: str | list[str], *, model: str | None = None) -> list[list[float]]:
        one = isinstance(texts, str)
        payload = {"model": model or self.config.embed_model, "input": [texts] if one else texts}
        data = self._post("/embeddings", payload)
        try:
            return [row["embedding"] for row in data["data"]]
        except (KeyError, TypeError) as exc:
            raise VonkError(f"Unexpected embeddings response: {str(data)[:200]}") from exc

    def image(self, prompt: str, *, model: str | None = None, size: str = "1024x1024") -> str:
        """Returns a URL or base64 payload, whichever the backend sent."""
        payload = {"model": model or self.config.image_model, "prompt": prompt, "size": size}
        data = self._post("/images/generations", payload)
        try:
            item = data["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise VonkError(f"Unexpected image response: {str(data)[:200]}") from exc
        return item.get("url") or item.get("b64_json", "")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "Vonk":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class AsyncVonk(_Base):
    """Async twin of Vonk. Same methods, all awaitable."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            headers=self.config.headers,
            timeout=self.config.timeout,
        )

    async def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                res = await self._client.post(path, json=payload)
                res.raise_for_status()
                return res.json()
            except Exception as exc:  # noqa: BLE001
                last = exc
        raise VonkError(f"vonk request to {path} failed: {last}") from last

    async def chat(
        self, prompt: str, *, system: str | None = None, model: str | None = None,
        max_tokens: int = 1024, temperature: float = 0.7,
    ) -> str:
        payload = chat_payload(
            prompt, system=system, model=model or self.config.chat_model,
            max_tokens=max_tokens, temperature=temperature, json_mode=False,
        )
        return clean(content_of(await self._post("/chat/completions", payload)))

    async def json(
        self, prompt: str, *, system: str | None = None, model: str | None = None,
        max_tokens: int = 1024, temperature: float = 0.0,
    ) -> dict | list:
        payload = chat_payload(
            prompt, system=system, model=model or self.config.chat_model,
            max_tokens=max_tokens, temperature=temperature, json_mode=True,
        )
        return parse_json(content_of(await self._post("/chat/completions", payload)))

    async def embed(self, texts: str | list[str], *, model: str | None = None) -> list[list[float]]:
        one = isinstance(texts, str)
        payload = {"model": model or self.config.embed_model, "input": [texts] if one else texts}
        data = await self._post("/embeddings", payload)
        try:
            return [row["embedding"] for row in data["data"]]
        except (KeyError, TypeError) as exc:
            raise VonkError(f"Unexpected embeddings response: {str(data)[:200]}") from exc

    async def image(self, prompt: str, *, model: str | None = None, size: str = "1024x1024") -> str:
        payload = {"model": model or self.config.image_model, "prompt": prompt, "size": size}
        data = await self._post("/images/generations", payload)
        try:
            item = data["data"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise VonkError(f"Unexpected image response: {str(data)[:200]}") from exc
        return item.get("url") or item.get("b64_json", "")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncVonk":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
