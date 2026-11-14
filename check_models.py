"""
check_models.py - which Groq chat models can this API key use, what is each one's tokens-per-minute (TPM)
limit, and does it call our tools correctly? Run once before choosing HARNESS_MODEL:

    python check_models.py

The model list comes live from Groq (GET /models), so it is never out of date. Each check is one small
request (~1.5k tokens: our system prompt + tool schemas). Limits come from the x-ratelimit-* headers
Groq sends back, so they are this key's real limits, not numbers from a web page.
"""

import sys

from groq import Groq

from harness import SYSTEM_PROMPT, tool_schemas

NOT_CHAT = ("whisper", "orpheus", "prompt-guard", "safeguard", "tts")  # speech / safety models: no tool use
PROBE = "Document: doc_id=abc123\nQuestion: On which pages is the word 'kernel' used?"


def check(client: Groq, model: str) -> str:
    raw = client.chat.completions.with_raw_response.create(
        model=model, temperature=0, tools=tool_schemas(), tool_choice="auto",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PROBE}])
    tpm = raw.headers.get("x-ratelimit-limit-tokens", "?")
    rpd = raw.headers.get("x-ratelimit-limit-requests", "?")
    calls = raw.parse().choices[0].message.tool_calls or []
    first = f"{calls[0].function.name}({calls[0].function.arguments})" if calls else "no tool call"
    good = bool(calls) and calls[0].function.name == "search_keyword"
    return f"{tpm:>8} {rpd:>8}  {'OK ' if good else 'BAD'} {first[:60]}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    client = Groq(max_retries=0)  # reads GROQ_API_KEY from the environment
    models = sorted(m.id for m in client.models.list().data if not any(x in m.id for x in NOT_CHAT))
    print(f"{'model':48} {'TPM':>8} {'req/day':>8}  first tool call (want search_keyword)")
    print("-" * 110)
    for model in models:
        try:
            print(f"{model:48} {check(client, model)}")
        except Exception as e:
            print(f"{model:48} error: {type(e).__name__}: {str(e)[:60]}")
    print("\nHigher TPM = fewer rate-limit waits. A hard question uses ~15-25k tokens in total.")
    print('Then compare answer quality:  $env:HARNESS_MODEL = "<model>";  python run_tests.py --live')


if __name__ == "__main__":
    main()
