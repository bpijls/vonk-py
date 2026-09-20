# vonk

Tiny client for the local **vonk** LiteLLM gateway. Replaces the hand-rolled
OpenAI/httpx wrapper that had been copied into ~25 files across these projects,
each with its own retry policy, timeout and fence-stripping.

## Install

```
vonk @ git+https://github.com/bpijls/vonk-py@v0.1.0
```

## Use

```python
from vonk import Vonk

with Vonk() as v:
    v.chat("Summarise this", system="Be brief")     # -> str, cleaned
    v.json('Return {"concepts": [...]}')            # -> dict/list, parsed
    v.embed(["a", "b"])                             # -> list[list[float]]
    v.image("a ruined city")                        # -> url or b64
```

`AsyncVonk` has the identical surface with `await`; close it with `aclose()`.

## Config

Read from the environment, override per-call with keyword args:

| Var | Default |
|---|---|
| `VONK_BASE_URL` | `http://vonk:4000/v1` |
| `VONK_KEY` | **required** — no default, raises `VonkError` if unset |
| `VONK_MODEL` | `code` |
| `VONK_EMBED_MODEL` | `embedding` |
| `VONK_IMAGE_MODEL` | `image` |
| `VONK_TIMEOUT` | `120` |

## What it handles for you

- **Fences and `<think>` blocks.** Local models wrap output in ```` ```json ````
  and reasoning tags regardless of instructions. `clean()` strips both.
- **Unparseable JSON.** `json()` sets `response_format`, then falls back to the
  outermost `{...}`/`[...]` span before raising `VonkError`.
- **Transient failures.** One retry by default (`retries=`), then `VonkError`.
- **The 32K ceiling.** Input + output share 32768 tokens; a `max_tokens` at or
  above that is rejected up front rather than failing mid-call.
- **No baked-in key.** Missing `VONK_KEY` raises at construction.

## Tests

```bash
pip install -e ".[dev]" && pytest      # 18 tests, fully mocked, no vonk needed
```
