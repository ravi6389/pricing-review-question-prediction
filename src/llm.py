"""
Cached, provider-agnostic chat client.

WHY A CACHE AND NOT JUST temperature=0
    Even at temperature 0 a hosted model is not reproducible: serving stacks
    batch requests, change kernels, and get silently upgraded. So every call is
    stored on disk keyed by the SHA-256 of (model, messages, temperature). By
    default a cached call is REPLAYED and the network is never touched, which
    means a reviewer with no API key can reproduce every downstream artifact.

    Pass regenerate=True to force a real call (an authoring action -- it
    rewrites the cache, so the manifest hashes will change by design).

CONFIGURATION (environment variables, no secrets in the repo)
    LLM_PROVIDER      azure | openai | offline     (default: azure)
    LLM_MODEL         deployment or model name
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
    OPENAI_API_KEY, OPENAI_BASE_URL

    LLM_PROVIDER=offline uses the deterministic template fallback supplied by
    the caller. That exists so the whole pipeline can be exercised and tested
    without credits; the final dataset should be generated with a real model,
    and which mode produced a file is recorded in the run metadata.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).parent.parent
CACHE_DIR = ROOT / "cache"

# Load credentials from a .env file if one exists. Paths are explicit because
# scripts are launched with the repo root as the working directory, so a bare
# load_dotenv() would miss a .env sitting next to this file. Both locations are
# tried; neither is required (a reviewer replaying the cache needs no key).
# .env MUST be gitignored -- it holds live API credentials.
try:
    from dotenv import load_dotenv
    for _candidate in (ROOT / ".env", Path(__file__).parent / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate, override=False)
except ImportError:
    pass  # python-dotenv is optional; environment variables work on their own


def _key(model: str, messages: list, temperature: float) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature},
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedLLM:
    def __init__(self, cache_name: str, regenerate: bool = False):
        self.provider = os.environ.get("LLM_PROVIDER", "azure").lower()
        self.model = os.environ.get("LLM_MODEL", "gpt-4o")
        self.regenerate = regenerate
        self.cache_path = CACHE_DIR / f"{cache_name}.json"
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.cache = (
            json.loads(self.cache_path.read_text(encoding="utf-8"))
            if self.cache_path.exists() else {}
        )
        self.stats = {"replayed": 0, "generated": 0, "offline": 0}
        self._client = None

    # -- provider plumbing -------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self.provider == "azure":
            missing = [v for v in ("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY")
                       if not os.environ.get(v)]
            if missing:
                raise SystemExit(
                    f"\nMissing environment variable(s): {', '.join(missing)}\n\n"
                    "Set them before running, for example:\n"
                    "  set AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com\n"
                    "  set AZURE_OPENAI_API_KEY=<your key>\n"
                    "  set LLM_MODEL=<your deployment name>\n\n"
                    "Or run without any model at all:\n"
                    "  set LLM_PROVIDER=offline\n"
                )
            try:
                from openai import AzureOpenAI
            except ImportError:
                raise SystemExit("\nThe openai package is not installed:  pip install openai\n")
            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
                timeout=60.0,     # fail in a minute, do not hang on a bad endpoint
                max_retries=2,
            )
        elif self.provider == "openai":
            from openai import OpenAI
            if not os.environ.get("OPENAI_API_KEY"):
                raise SystemExit("\nMissing environment variable: OPENAI_API_KEY\n")
            self._client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL") or None,
                timeout=60.0,
                max_retries=2,
            )
        else:
            raise RuntimeError(f"no API client for provider {self.provider!r}")
        return self._client

    # -- public API --------------------------------------------------------

    def is_cached(self, messages: list, temperature: float = 0.7) -> bool:
        """Would this call be served from cache? Used to report work up front."""
        return _key(self.model, messages, temperature) in self.cache

    def chat(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 900,
        offline_fallback: Optional[Callable[[], str]] = None,
    ) -> str:
        """
        Return the assistant's text for `messages`.

        Resolution order:
          1. cache hit and not regenerating  -> replay
          2. provider == offline             -> offline_fallback()
          3. otherwise                       -> real API call, then cache it
        """
        k = _key(self.model, messages, temperature)

        if k in self.cache and not self.regenerate:
            self.stats["replayed"] += 1
            return self.cache[k]

        if self.provider == "offline":
            if offline_fallback is None:
                raise RuntimeError(
                    "LLM_PROVIDER=offline but this call site supplied no fallback"
                )
            text = offline_fallback()
            self.stats["offline"] += 1
            self.cache[k] = text
            return text

        resp = self._get_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=42,  # best-effort; hosted models do not honour this strictly
        )
        text = (resp.choices[0].message.content or "").strip()
        self.stats["generated"] += 1
        self.cache[k] = text
        return text

    def save(self) -> None:
        """Write the cache sorted, so the committed file is diffable and stable."""
        self.cache_path.write_text(
            json.dumps(self.cache, indent=1, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    def summary(self) -> dict:
        return {"provider": self.provider, "model": self.model, **self.stats}
