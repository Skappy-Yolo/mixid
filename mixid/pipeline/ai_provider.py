"""Provider-agnostic LLM completion layer (mirrors DJAgent's pattern).

Set AI_PROVIDER in .env to choose: gemini (free), groq (free), ollama
(local, free), anthropic (paid). All four take a `(system_prompt,
user_prompt) -> str` signature so the caller doesn't care which is
configured.

If no provider is reachable (no key, no Ollama, etc.) the re-ranker
gracefully degrades to picking the highest-fingerprint-score candidate
per segment — see reranker.py.
"""
from __future__ import annotations

import logging

import config


log = logging.getLogger(__name__)


class NoProviderConfigured(RuntimeError):
    """Raised when AI_PROVIDER is set but its credentials are missing."""


def complete(system_prompt: str, user_prompt: str) -> str:
    provider = (config.AI_PROVIDER or "").lower()
    if provider == "gemini":
        return _gemini(system_prompt, user_prompt)
    if provider == "groq":
        return _groq(system_prompt, user_prompt)
    if provider == "ollama":
        return _ollama(system_prompt, user_prompt)
    if provider == "anthropic":
        return _anthropic(system_prompt, user_prompt)
    raise NoProviderConfigured(
        f"AI_PROVIDER='{provider}' not recognized. Use gemini, groq, ollama, or anthropic."
    )


def _gemini(system_prompt: str, user_prompt: str) -> str:
    """Gemini via direct REST.

    Note: we deliberately avoid the google-genai SDK because it routes
    requests through a billing path that requires prepaid credits and
    fails with RESOURCE_EXHAUSTED even on free-tier keys. Raw REST hits
    the standard generativelanguage.googleapis.com endpoint and works
    on any free key.
    """
    if not config.GEMINI_API_KEY:
        raise NoProviderConfigured(
            "GEMINI_API_KEY not set. Get a free key at https://aistudio.google.com/app/apikey"
        )
    import requests

    model = config.AI_MODEL or "gemini-2.5-flash"
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }
    r = requests.post(
        url,
        headers={
            "x-goog-api-key": config.GEMINI_API_KEY,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
    data = r.json()
    # candidates[0].content.parts[0].text is the standard shape
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no candidates: {data}")
    parts = (candidates[0].get("content") or {}).get("parts") or []
    if not parts:
        raise RuntimeError(f"Gemini candidate had no parts: {candidates[0]}")
    return parts[0].get("text", "")


def _groq(system_prompt: str, user_prompt: str) -> str:
    if not config.GROQ_API_KEY:
        raise NoProviderConfigured(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com"
        )
    try:
        from groq import Groq
    except ImportError as e:
        raise ImportError("pip install groq") from e

    client = Groq(api_key=config.GROQ_API_KEY)
    model = config.AI_MODEL or "llama-3.3-70b-versatile"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=8192,
    )
    return resp.choices[0].message.content


def _ollama(system_prompt: str, user_prompt: str) -> str:
    import requests

    base = config.OLLAMA_BASE_URL or "http://localhost:11434"
    model = config.AI_MODEL or "llama3.1"
    r = requests.post(
        f"{base}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text[:200]}")
    return r.json()["message"]["content"]


def _anthropic(system_prompt: str, user_prompt: str) -> str:
    if not config.ANTHROPIC_API_KEY:
        raise NoProviderConfigured(
            "ANTHROPIC_API_KEY not set. Get a key at https://console.anthropic.com"
        )
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    model = config.AI_MODEL or "claude-opus-4-7"
    resp = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return resp.content[0].text
