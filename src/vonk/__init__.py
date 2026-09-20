"""Client for the local vonk LiteLLM gateway.

    from vonk import Vonk
    with Vonk() as v:
        print(v.chat("hello"))
        print(v.json('Return {"ok": true}'))
        print(len(v.embed("some text")[0]))

Reads VONK_BASE_URL / VONK_KEY / VONK_MODEL / VONK_EMBED_MODEL / VONK_IMAGE_MODEL.
"""
from ._core import CONTEXT_LIMIT, Config, VonkError, clean, parse_json
from .client import AsyncVonk, Vonk

__all__ = [
    "Vonk", "AsyncVonk", "VonkError", "Config",
    "clean", "parse_json", "CONTEXT_LIMIT",
]
