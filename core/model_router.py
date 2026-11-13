# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Aditi Jain (SmartWrapperOSS)

"""
core/model_router.py

Sends a prompt to whichever LLM provider is configured, and returns a
standard response shape (text + token counts + latency + cost).

This file does NOT know or care which workflow is calling it. It has no
concept of "summarization" or "tool use" — it just sends prompts and
returns text. That's deliberate: it's the one piece of infrastructure
every workflow shares, so it should stay as simple and generic as possible.

Adding a genuinely NEW provider protocol (not just a new model) requires
adding one new `_call_x` method and one line in `_dispatch`. But most
open-weight models (Llama, Mistral, Qwen, DeepSeek, ...) don't need that:
any local runtime (Ollama, vLLM, LM Studio, llama.cpp server,
text-generation-webui) or hosted OSS-model API (Together, Groq, Fireworks,
OpenRouter) that speaks the OpenAI chat-completions wire format is already
covered by the `openai_compatible` provider below — adding one of those
models is purely a `config.yaml` edit, no code change required.
"""

import time
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Raw response from a single LLM call."""
    model_id: str
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


# Published pricing per 1,000 tokens, in USD, for models with a known,
# stable list price. Update these as providers change their pricing — see
# the README disclaimer: these are estimates, always verify against your
# provider's billing dashboard.
#
# This table is only a fallback. Open-weight / self-hosted / third-party
# models have no single canonical price (a local Llama is free; the same
# weights served by a hosted API are not), so those should set
# `cost_per_1k_input` / `cost_per_1k_output` directly in config.yaml
# instead of being added here — see config/config.yaml.example.
COST_PER_1K_TOKENS = {
    "gpt-4o":            {"input": 0.0025,   "output": 0.010},
    "gpt-4o-mini":       {"input": 0.00015,  "output": 0.0006},
    "claude-sonnet-4-6": {"input": 0.003,    "output": 0.015},
    "gemini-3.5-flash":  {"input": 0.0015,   "output": 0.009},
    "muse-ember":        {"input": 0.0001,   "output": 0.0005},
    "muse-glimmer-30b":  {"input": 0.0003,   "output": 0.0012},
    "muse-spark-1.3":    {"input": 0.00125,  "output": 0.00425},
    "muse-wonder":       {"input": 0.0040,   "output": 0.0200},
}


class ModelRouter:
    """
    Usage:
        router = ModelRouter(config["models"])
        response = router.call("gpt-4o", "Summarize this: ...")
    """

    def __init__(self, model_configs: list):
        # model_configs looks like:
        # [{"id": "gpt-4o", "provider": "openai", "api_key": "..."}, ...]
        self.configs = {m["id"]: m for m in model_configs}

    def call(self, model_id: str, prompt: str) -> ModelResponse:
        """Send `prompt` to `model_id` and return the response."""
        if model_id not in self.configs:
            raise ValueError(
                f"Model '{model_id}' is not configured. "
                f"Available models: {list(self.configs.keys())}"
            )
        config = self.configs[model_id]
        return self._dispatch(model_id, config, prompt)

    def _dispatch(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        provider = config["provider"]
        if provider == "openai":
            return self._call_openai(model_id, config, prompt)
        elif provider == "anthropic":
            return self._call_anthropic(model_id, config, prompt)
        elif provider == "google":
            return self._call_google(model_id, config, prompt)
        elif provider == "openai_compatible":
            return self._call_openai_compatible(model_id, config, prompt)
        elif provider == "mistral":
            raise ValueError(
                f"Model '{model_id}' uses provider 'mistral', which has been "
                f"replaced by the more general 'openai_compatible' provider "
                f"(it covers Llama, Mistral, Qwen, DeepSeek, etc. via any "
                f"local runtime or hosted OSS-model API — see "
                f"config/config.yaml.example). Update config.yaml: set "
                f"provider: openai_compatible and base_url to your "
                f"endpoint's OpenAI-compatible root, e.g. "
                f"'http://localhost:11434/v1' for Ollama."
            )
        else:
            raise ValueError(f"Unknown provider: '{provider}'")

    def _call_openai(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import openai
        client = openai.OpenAI(api_key=config["api_key"])

        start = time.time()
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        latency_ms = (time.time() - start) * 1000

        usage = resp.usage
        cost = self._calc_cost(model_id, usage.prompt_tokens, usage.completion_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.choices[0].message.content,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_anthropic(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=config["api_key"])

        start = time.time()
        resp = client.messages.create(
            model=model_id,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = (time.time() - start) * 1000

        cost = self._calc_cost(model_id, resp.usage.input_tokens, resp.usage.output_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.content[0].text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_google(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        import google.generativeai as genai
        genai.configure(api_key=config["api_key"])
        model = genai.GenerativeModel(model_id)

        start = time.time()
        resp = model.generate_content(prompt)
        latency_ms = (time.time() - start) * 1000

        input_tokens = model.count_tokens(prompt).total_tokens
        output_tokens = model.count_tokens(resp.text).total_tokens
        cost = self._calc_cost(model_id, input_tokens, output_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _call_openai_compatible(self, model_id: str, config: dict, prompt: str) -> ModelResponse:
        """
        Calls any endpoint that implements the OpenAI chat-completions wire
        format. This single adapter covers local runtimes (Ollama's `/v1`
        endpoint, vLLM, LM Studio, llama.cpp server, text-generation-webui)
        AND hosted OSS-model APIs (Together, Groq, Fireworks, OpenRouter) —
        so Llama, Mistral, Qwen, DeepSeek, Muse, etc. are all just a
        config.yaml entry away, not a new adapter.
        """
        import os

        base_url = config.get("base_url")
        if not base_url:
            raise ValueError(
                f"Model '{model_id}' has provider 'openai_compatible' but no "
                f"'base_url' in config.yaml. Point it at the endpoint's "
                f"OpenAI-compatible root, e.g. 'http://localhost:11434/v1' "
                f"(Ollama), 'http://localhost:8000/v1' (vLLM / LM Studio), "
                f"or 'https://openrouter.ai/api/v1' (OpenRouter)."
            )

        raw_key = config.get("api_key", "not-needed")
        if isinstance(raw_key, str) and raw_key.startswith("$"):
            api_key = os.environ.get(raw_key[1:], "not-needed")
        elif config.get("api_key_env"):
            api_key = os.environ.get(config["api_key_env"], raw_key)
        else:
            api_key = raw_key

        api_model = config.get("api_model", model_id)
        extra_params = config.get("extra_params", {})
        timeout = config.get("timeout", 120)
        max_tokens = config.get("max_tokens")

        try:
            import openai
        except ModuleNotFoundError:
            openai = None

        if openai is None:
            import json
            import urllib.error
            import urllib.request

            url = base_url.rstrip("/") + "/chat/completions"
            body = {
                "model": api_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if max_tokens is not None:
                body["max_tokens"] = max_tokens
            if extra_params:
                body.update(extra_params)

            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
            )
            start = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"HTTP {e.code} from {url}: {err_body}") from e
            except urllib.error.URLError as e:
                raise ConnectionError(f"Could not reach '{model_id}' at {base_url}: {e}") from e
            latency_ms = (time.time() - start) * 1000

            if data.get("error"):
                raise RuntimeError(f"API error from {url}: {json.dumps(data['error'])[:500]}")

            text = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            cost = self._calc_cost(model_id, input_tokens, output_tokens, config)
            return ModelResponse(
                model_id=model_id,
                text=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=cost,
            )

        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=api_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                extra_body=extra_params if extra_params else None,
            )
        except openai.APIConnectionError as e:
            if ":11434" in base_url:
                hint = (
                    f"Is Ollama running? Try `ollama serve`, and make sure "
                    f"the model is pulled: `ollama pull {api_model}`."
                )
            else:
                hint = f"Is the server at {base_url} running and reachable?"
            raise ConnectionError(
                f"Could not reach '{model_id}' at {base_url}. {hint}"
            ) from e
        except openai.APITimeoutError as e:
            raise TimeoutError(
                f"Call to '{model_id}' at {base_url} timed out after "
                f"{timeout}s. Local models on modest "
                f"hardware can be slow — try raising `timeout` in "
                f"config.yaml for this model."
            ) from e
        latency_ms = (time.time() - start) * 1000

        usage = resp.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost = self._calc_cost(model_id, input_tokens, output_tokens, config)

        return ModelResponse(
            model_id=model_id,
            text=resp.choices[0].message.content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
        )

    def _calc_cost(self, model_id: str, input_tokens: int, output_tokens: int,
                    config: dict = None) -> float:
        # Config-supplied pricing always wins: the router can't know the
        # rates for every hosted endpoint or self-hosted deployment in
        # advance, so any model.yaml entry with explicit
        # cost_per_1k_input/output overrides the built-in table (and is
        # required for local models to correctly show $0, or for hosted
        # OSS-model APIs that DO charge to show a real cost).
        if config and ("cost_per_1k_input" in config or "cost_per_1k_output" in config):
            rates = {
                "input": config.get("cost_per_1k_input", 0.0),
                "output": config.get("cost_per_1k_output", 0.0),
            }
        else:
            rates = COST_PER_1K_TOKENS.get(model_id, {"input": 0.0, "output": 0.0})
        return (input_tokens / 1000 * rates["input"]) + (output_tokens / 1000 * rates["output"])
