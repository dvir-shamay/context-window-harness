"""LLM API client — multi-provider with retry logic.

Supports:
  - Azure OpenAI (GPT-4o, GPT-5.x)
  - Azure AI Foundry / MaaS (DeepSeek, Llama, Grok, Kimi, etc.)
  - GitHub Models (Claude, Gemini, etc.)
  - Local HTTP bridge (any OpenAI-compatible proxy)

All credentials are loaded from environment variables via python-dotenv.
See .env.example for the required variables.
"""
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class LLMResponse:
    """Response from an LLM API call."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    success: bool
    error: str = ""


def call_azure_openai(
    messages: list[dict],
    model: str = "gpt-4o",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> LLMResponse:
    """Call Azure OpenAI API."""
    key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        return LLMResponse("", model, "azure-openai", 0, 0, 0, 0, False, "Missing AZURE_OPENAI_KEY or AZURE_OPENAI_ENDPOINT")

    url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version=2024-06-01"
    # GPT-5.x uses max_completion_tokens instead of max_tokens
    if model.startswith("gpt-5"):
        payload = {"messages": messages, "max_completion_tokens": max_tokens, "temperature": temperature}
    else:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    body = json.dumps(payload).encode()

    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "api-key": key,
    })

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
        latency = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider="azure-openai",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency,
            success=True,
        )
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        err = e.read().decode()[:300]
        return LLMResponse("", model, "azure-openai", 0, 0, 0, latency, False, f"HTTP {e.code}: {err}")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return LLMResponse("", model, "azure-openai", 0, 0, 0, latency, False, str(e))


def call_azure_ai(
    messages: list[dict],
    model: str = "DeepSeek-V3.2",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> LLMResponse:
    """Call Azure AI Foundry (MaaS) API."""
    key = os.getenv("AZURE_AI_KEY")
    endpoint = os.getenv("AZURE_AI_ENDPOINT")
    if not key or not endpoint:
        return LLMResponse("", model, "azure-ai", 0, 0, 0, 0, False, "Missing AZURE_AI_KEY or AZURE_AI_ENDPOINT")

    if "/models/chat/completions" in endpoint:
        url = endpoint
    else:
        url = f"{endpoint}/models/chat/completions?api-version=2024-05-01-preview"

    # Reasoning models need more tokens for thinking
    effective_max = max_tokens * 3 if model in ("Kimi-K2.6", "grok-4-20-reasoning") else max_tokens
    effective_timeout = 300 if model in ("Kimi-K2.6", "grok-4-20-reasoning") else 180

    body = json.dumps({
        "messages": messages,
        "max_tokens": effective_max,
        "temperature": temperature,
        "model": model,
    }).encode()

    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            data = json.loads(resp.read())
        latency = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider="azure-ai",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency,
            success=True,
        )
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        err = e.read().decode()[:300]
        return LLMResponse("", model, "azure-ai", 0, 0, 0, latency, False, f"HTTP {e.code}: {err}")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return LLMResponse("", model, "azure-ai", 0, 0, 0, latency, False, str(e))


# ── Local Bridge Provider ──────────────────────────────────────────────────────

def call_local_bridge(
    messages: list[dict],
    model: str = "claude-opus-4-7",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> LLMResponse:
    """Call a local OpenAI-compatible HTTP bridge endpoint."""
    port = int(os.getenv("LOCAL_BRIDGE_PORT", "3456"))
    url = f"http://127.0.0.1:{port}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())
        latency = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        content = data["choices"][0]["message"].get("content", "")
        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider="local-bridge",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency,
            success=True,
        )
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        err = e.read().decode()[:300]
        return LLMResponse("", model, "local-bridge", 0, 0, 0, latency, False, f"HTTP {e.code}: {err}")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        hint = " (is the local bridge running?)" if "refused" in str(e).lower() else ""
        return LLMResponse("", model, "local-bridge", 0, 0, 0, latency, False, str(e) + hint)


# ── GitHub Models ──────────────────────────────────────────────────────────────

GITHUB_MODELS_ENDPOINT = "https://models.inference.ai.azure.com"

_GH_REASONING_MODELS = {"claude-opus-4-7", "claude-opus-4-6"}


def call_github_models(
    messages: list[dict],
    model: str = "claude-opus-4-7",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> LLMResponse:
    """Call GitHub Models inference endpoint (OpenAI-compatible)."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return LLMResponse("", model, "github-models", 0, 0, 0, 0, False, "Missing GITHUB_TOKEN in .env")

    url = f"{GITHUB_MODELS_ENDPOINT}/chat/completions"
    effective_max = max_tokens * 3 if model in _GH_REASONING_MODELS else max_tokens
    effective_timeout = 300 if model in _GH_REASONING_MODELS else 180

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": effective_max,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    })

    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=effective_timeout) as resp:
            data = json.loads(resp.read())
        latency = (time.perf_counter() - start) * 1000
        usage = data.get("usage", {})
        msg = data["choices"][0]["message"]
        content = msg.get("content") or msg.get("reasoning_content") or ""
        return LLMResponse(
            content=content,
            model=data.get("model", model),
            provider="github-models",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=latency,
            success=True,
        )
    except urllib.error.HTTPError as e:
        latency = (time.perf_counter() - start) * 1000
        err = e.read().decode()[:300]
        return LLMResponse("", model, "github-models", 0, 0, 0, latency, False, f"HTTP {e.code}: {err}")
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return LLMResponse("", model, "github-models", 0, 0, 0, latency, False, str(e))


# ── Dry-run provider (no API call) ────────────────────────────────────────────

def call_dry_run(
    messages: list[dict],
    model: str = "dry-run",
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> LLMResponse:
    """Dry-run provider — prints the prompt and returns an empty response.

    Useful for validating prompt construction and data flow without
    consuming API credits.
    """
    # Estimate token count (rough: 1 token ≈ 4 chars)
    total_chars = sum(len(m.get("content", "")) for m in messages)
    est_tokens = total_chars // 4
    print(f"  [dry-run] {len(messages)} messages, ~{est_tokens} estimated prompt tokens")
    print(f"  [dry-run] System: {messages[0]['content'][:80]}...")
    print(f"  [dry-run] User: {messages[-1]['content'][:120]}...")
    return LLMResponse(
        content="[]",  # empty JSON array — scorer will return zeros
        model=model,
        provider="dry-run",
        prompt_tokens=est_tokens,
        completion_tokens=0,
        total_tokens=est_tokens,
        latency_ms=0.0,
        success=True,
    )


# ── Unified entry point ───────────────────────────────────────────────────────

def call_llm(
    messages: list[dict],
    provider: str = "azure-ai",
    model: str = "DeepSeek-V3.2",
    max_tokens: int = 2000,
    temperature: float = 0.1,
    max_retries: int = 3,
) -> LLMResponse:
    """Unified entry point for LLM calls with retry on transient failures."""
    for attempt in range(max_retries):
        if provider == "azure-openai":
            resp = call_azure_openai(messages, model, max_tokens, temperature)
        elif provider == "azure-ai":
            resp = call_azure_ai(messages, model, max_tokens, temperature)
        elif provider == "github-models":
            resp = call_github_models(messages, model, max_tokens, temperature)
        elif provider == "local-bridge":
            resp = call_local_bridge(messages, model, max_tokens, temperature)
        elif provider == "dry-run":
            return call_dry_run(messages, model, max_tokens, temperature)
        else:
            return LLMResponse("", model, provider, 0, 0, 0, 0, False, f"Unknown provider: {provider}")

        if resp.success or attempt == max_retries - 1:
            return resp

        # Retry on transient errors
        err = resp.error.lower()
        transient = any(s in err for s in ("10054", "timed out", "429", "500", "502", "503", "504", "urlopen error", "winerror", "connection", "reset"))
        if not transient:
            return resp

        wait = 15 * (attempt + 1) if "429" in err else 5 * (attempt + 1)
        print(f"    [retry {attempt+1}/{max_retries}] {resp.error[:80]}... waiting {wait}s")
        time.sleep(wait)

    return resp
