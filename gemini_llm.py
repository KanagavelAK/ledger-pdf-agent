"""
gemini_llm.py - BACKUP model client: Google Gemini, for when Groq's free-tier rate limit is reached.

It has the same interface as harness.GroqLLM (complete() and pace()), so the agent loop, the budget
gate and the evidence check do not change at all: only the model behind them does. It calls Gemini's
OpenAI-compatible endpoint with Python's standard library (urllib), so there is nothing extra to install.

    key:    GEMINI_API_KEY in the local .env file or the terminal (create it in Google AI Studio)
    model:  GEMINI_MODEL environment variable, default gemini-3.7-flash
    switch: the model menu in the web UI (per question, no restart), or HARNESS_PROVIDER=gemini
            for the command line and run_tests.py --live
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
# Tried in order when a model answers 503 "experiencing high demand" (Google's servers, not our key).
GEMINI_FALLBACKS = [m for m in os.getenv("GEMINI_FALLBACKS", "gemini-3.7-flash,gemini-2.5-flash").split(",") if m]
OVERLOADED = (500, 503)  # server-side: another model may be free; other errors (400, 401, 429) are not hidden
TIMEOUT_S = 60  # Gemini thinks before it answers, so it gets longer than Groq's 30 s
# Free tier: 5 requests per minute per model (live error: "generate_content_free_tier_requests, limit: 5").
# Raise it with $env:GEMINI_RPM on a paid tier. One question needs 4-7 model calls.
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "5"))
_SENT: dict[str, list[float]] = {}  # model -> times of recent requests; module-level so it spans questions


class GeminiAPIError(Exception):
    """An HTTP error from Gemini, shaped like the Groq SDK's errors (.status_code, .body, .response.headers),
    so the harness's retry rules work unchanged: 429 and 5xx are retried, other 4xx fail fast."""

    def __init__(self, status_code: int, body: str, headers: dict):
        super().__init__(f"Error code: {status_code} - {body[:500]}")
        self.status_code, self.body = status_code, body
        headers = {k.lower(): v for k, v in headers.items()}
        m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', body)  # Gemini puts its wait in the body
        if m and "retry-after" not in headers:
            headers["retry-after"] = m.group(1)  # ...so harness.retry_delay() can read it like Groq's header
        self.response = type("Response", (), {"headers": headers})()


class GeminiLLM:
    def __init__(self, model: str = GEMINI_MODEL):
        self.key = os.environ.get("GEMINI_API_KEY")
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set: create a key in Google AI Studio, paste it into the "
                               ".env file next to harness.py (or set it in the terminal), then restart the app")
        self.model = model
        self.sent = _SENT  # shared by every GeminiLLM, because the server makes a new one per question

    def pace(self, messages: list[dict], tools: list[dict]) -> float:
        """Seconds to wait so the next request stays within GEMINI_RPM requests per minute (0 = send now).
        Gemini sends no rate-limit headers, so we count our own requests over the last 60 seconds."""
        recent = [t for t in self.sent.get(self.model, []) if time.time() - t < 60]
        self.sent[self.model] = recent
        if len(recent) < GEMINI_RPM:
            return 0.0
        return max(0.0, recent[-GEMINI_RPM] + 60 - time.time())  # until the oldest counted request is a minute old

    def complete(self, messages: list[dict], tools: list[dict], tool_choice: str = "auto") -> dict:
        """One model call. If the model is overloaded (503), the next model in GEMINI_FALLBACKS is tried;
        the reply says which model answered. (Google: thought signatures stay valid across models.)"""
        models = list(dict.fromkeys([self.model, *GEMINI_FALLBACKS]))  # in order, each model once
        for i, model in enumerate(models):
            try:
                data = self._post(model, messages, tools, tool_choice)
                break
            except GeminiAPIError as e:
                if e.status_code not in OVERLOADED or i == len(models) - 1:
                    raise  # not an overload, or no model left: the harness's retry rules take over
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return {
            "content": msg.get("content"),
            "tool_calls": [{
                "id": tc.get("id") or f"call_{i}",
                "name": tc["function"]["name"],
                "arguments": tc["function"].get("arguments") or "{}",
                # Anything else on the call (Gemini 3's thought signature lives in "extra_content") is kept,
                # because the harness must send it back exactly as received on the next request.
                "extra": {k: v for k, v in tc.items() if k not in ("id", "type", "function")},
            } for i, tc in enumerate(msg.get("tool_calls") or [])],
            "usage": {"in": usage.get("prompt_tokens", 0), "out": usage.get("completion_tokens", 0)},
            "model": model,  # which model actually answered (may be a fallback)
        }

    def _post(self, model: str, messages: list[dict], tools: list[dict], tool_choice: str) -> dict:
        # No temperature: Google's Gemini 3 guide strongly recommends the default (1.0); lower values
        # "may lead to unexpected behavior, such as looping". (Groq stays at 0.)
        body = {"model": model, "messages": messages, "tools": tools, "tool_choice": tool_choice}
        self.sent.setdefault(model, []).append(time.time())  # every request counts toward the per-minute limit
        req = urllib.request.Request(GEMINI_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.key}"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise GeminiAPIError(e.code, e.read().decode("utf-8", "replace"), dict(e.headers or {})) from None
