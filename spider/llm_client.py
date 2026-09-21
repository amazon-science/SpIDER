# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin LLM client abstraction.

Supports:
  - Azure OpenAI Responses API  (provider="azure_responses")
  - Azure OpenAI native SDK     (provider="azure")
  - OpenAI                      (provider="openai")
  - vLLM / local                (provider="vllm", OpenAI-compatible)

All four are driven through the same `chat(messages) -> LLMResponse` interface,
so the pipeline code never branches on the provider. Transient failures (429,
timeouts, 5xx) are retried with exponential backoff; anything else propagates
to the caller, which records it as a Step-B call error.
"""

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Dict

import httpx
from openai import AzureOpenAI, OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError

logger = logging.getLogger(__name__)

# Retry/backoff defaults
MAX_RETRIES = 6
BASE_BACKOFF = 2.0      # seconds
MAX_BACKOFF = 60.0      # cap


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int          # API output_tokens (INCLUDES reasoning tokens)
    reasoning_tokens: int = 0        # subset of completion_tokens spent on reasoning


def build_client(cfg) -> "LLMClientBase":
    """Factory: build the right client from a SelectorConfig."""
    if cfg.provider == "azure_responses":
        return ResponsesAPIClient(
            base_url=cfg.api_base_url,
            api_key=cfg.api_key,
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    elif cfg.provider == "azure":
        return ChatClient(
            client=AzureOpenAI(
                azure_endpoint=cfg.azure_api_base,
                api_key=cfg.azure_api_key,
                api_version=cfg.azure_api_version,
            ),
            model=cfg.azure_deployment,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    elif cfg.provider == "vllm":
        return ChatClient(
            client=OpenAI(
                base_url=cfg.vllm_base_url,
                api_key=cfg.vllm_api_key or "EMPTY",
            ),
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )
    else:  # "openai"
        kwargs = {}
        if cfg.api_base_url:
            kwargs["base_url"] = cfg.api_base_url
        if cfg.api_key:
            kwargs["api_key"] = cfg.api_key
        return ChatClient(
            client=OpenAI(**kwargs),
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
        )


def _retry_after_seconds(exc) -> float:
    """Extract Retry-After header from an httpx or openai error, if present."""
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            ra = resp.headers.get("retry-after")
            if ra:
                return float(ra)
    except Exception:
        pass
    return 0.0


def _with_retry(call, tag: str = ""):
    """Call `call()` and retry on 429/timeout/connection/5xx with exponential backoff."""
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            return call()
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            last_exc = e
            wait = _retry_after_seconds(e)
            if wait <= 0:
                wait = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 1)
            logger.warning(f"{tag} {type(e).__name__} (attempt {attempt+1}/{MAX_RETRIES}), "
                            f"sleeping {wait:.1f}s")
            time.sleep(wait)
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429 or 500 <= code < 600:
                last_exc = e
                wait = _retry_after_seconds(e)
                if wait <= 0:
                    wait = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 1)
                logger.warning(f"{tag} HTTP {code} (attempt {attempt+1}/{MAX_RETRIES}), "
                                f"sleeping {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
        except APIError as e:
            # Catch-all for OpenAI SDK errors with status >= 500 / unknown transient
            status = getattr(e, "status_code", None)
            if status is not None and (status == 429 or 500 <= status < 600):
                last_exc = e
                wait = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt)) + random.uniform(0, 1)
                logger.warning(f"{tag} {type(e).__name__} status={status} "
                                f"(attempt {attempt+1}/{MAX_RETRIES}), sleeping {wait:.1f}s")
                time.sleep(wait)
            else:
                raise
    # Exhausted retries
    raise last_exc


class LLMClientBase:
    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        raise NotImplementedError


class ChatClient(LLMClientBase):
    """Standard OpenAI Chat Completions API client."""

    def __init__(self, client, model: str, temperature: float, max_tokens: int):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        response = _with_retry(_call, tag=f"[{self.model}]")
        # Chat Completions: completion_tokens includes reasoning; the subset is
        # under completion_tokens_details.reasoning_tokens (0 for non-reasoning
        # models / servers like vLLM that don't report it).
        details = getattr(response.usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", 0) or 0 if details else 0
        return LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            reasoning_tokens=reasoning,
        )


class ResponsesAPIClient(LLMClientBase):
    """Azure OpenAI Responses API client (for deployments that only support /responses)."""

    def __init__(self, base_url: str, api_key: str, model: str,
                 temperature: float, max_tokens: int):
        # Accept either the v1 root ("https://X.openai.azure.com/openai/v1")
        # — we append /responses — or the Foundry Target URI verbatim
        # ("https://X.cognitiveservices.azure.com/openai/responses?api-version=...").
        from urllib.parse import urlparse, urlunparse
        p = urlparse(base_url)
        path = p.path.rstrip("/")
        if not path.endswith("/responses"):
            path = path + "/responses"
        self.responses_url = urlunparse((p.scheme, p.netloc, path, p.params, p.query, p.fragment))
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.http = httpx.Client(timeout=120)

    def chat(self, messages: List[Dict[str, str]]) -> LLMResponse:
        def _call():
            r = self.http.post(
                self.responses_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": messages,
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
            r.raise_for_status()
            return r
        resp = _with_retry(_call, tag=f"[{self.model}]")
        data = resp.json()

        # Extract text from response output
        content = ""
        for item in data.get("output", []):
            if item.get("type") == "message":
                for c in item.get("content", []):
                    if c.get("type") == "output_text":
                        content += c.get("text", "")

        usage = data.get("usage", {})
        # Responses API: output_tokens INCLUDES reasoning; the reasoning subset
        # is under output_tokens_details.reasoning_tokens (0 / absent for
        # non-reasoning models).
        details = usage.get("output_tokens_details", {}) or {}
        return LLMResponse(
            content=content,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=details.get("reasoning_tokens", 0),
        )
