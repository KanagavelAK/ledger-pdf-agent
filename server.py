"""
server.py - the backend: serves the chat UI and connects it to the agent.
Standard library only (http.server), so there is no web framework to install or explain.

    GET  /, /app.css, /app.js   the frontend (files in static/)
    POST /api/upload            body = the PDF bytes, header X-Filename      -> document summary (JSON)
    POST /api/ask               {"doc_id", "question", "history", "provider"} -> NDJSON stream: every trace
                                event as it happens (live call ledger), then one {"type": "result"} line
    GET  /api/trace/<run_id>    the question's full JSONL trace (deliverable: nothing hidden)
    GET  /api/health            backend reachable? each provider's model and whether its key is set (never the key)

Providers: "groq" (default) and "gemini" (backup, gemini_llm.py). The UI's model menu picks one per question.

Run:  python server.py        (needs GROQ_API_KEY in .env or the terminal)  then open http://127.0.0.1:8000
Hosted demo (e.g. Hugging Face Spaces): HOST=0.0.0.0, PORT=7860, and DEMO_ACCESS_CODE set, so only people
given the code can spend the API quota. Without DEMO_ACCESS_CODE nothing is locked (local use).
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import harness
from docstore import MAX_FILE_MB, STORE
from gemini_llm import GEMINI_MODEL

STATIC_DIR = Path(__file__).parent / "static"
STATIC_FILES = {  # a fixed list, so no request path can reach any other file
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}
TRACE_DIR = Path("traces/live_demo")  # traces of the live demo are a deliverable
LLM_FACTORY = None                    # None = harness.make_llm(provider); the tests plug in a ScriptedLLM
PROVIDERS = {  # name -> (model, environment variable holding its key); the UI's model menu lists these
    "groq": (harness.MODEL, "GROQ_API_KEY"),
    "gemini": (GEMINI_MODEL, "GEMINI_API_KEY"),  # backup when Groq's rate limit is reached
}
MAX_ASK_BYTES = 64 * 1024             # a question + short history is far smaller than this
ACCESS_CODE = os.getenv("DEMO_ACCESS_CODE", "")  # "" = open (local use); set on a public host to protect the quota


class Handler(BaseHTTPRequestHandler):
    server_version = "LedgerServer/1.0"

    # ---- routing ----

    def do_GET(self) -> None:
        if self.path in STATIC_FILES:
            name, content_type = STATIC_FILES[self.path]
            return self._send(200, (STATIC_DIR / name).read_bytes(), content_type)
        if self.path == "/api/health":  # the UI checks this on load, so a missing key shows before any question
            return self._json(200, {"ok": True, "budget": harness.TOOL_BUDGET, "default_provider": harness.PROVIDER,
                                    "access_code": bool(ACCESS_CODE),  # whether one is needed, never the code
                                    "providers": {name: {"model": model, "key_set": bool(os.environ.get(key))}
                                                  for name, (model, key) in PROVIDERS.items()}})
        if not self._allowed():
            return
        m = re.fullmatch(r"/api/trace/([\w-]+)", self.path)  # run ids are letters, digits and '-'
        if m and (TRACE_DIR / f"{m.group(1)}.jsonl").is_file():
            return self._send(200, (TRACE_DIR / f"{m.group(1)}.jsonl").read_bytes(), "application/x-ndjson",
                              {"Content-Disposition": f'attachment; filename="{m.group(1)}.jsonl"'})
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._allowed():
            return
        if self.path == "/api/upload":
            return self._upload()
        if self.path == "/api/ask":
            return self._ask()
        self._json(404, {"error": "not found"})

    # ---- endpoints ----

    def _upload(self) -> None:
        size = int(self.headers.get("Content-Length") or 0)
        if size > MAX_FILE_MB * 1024 * 1024:  # checked before reading the body
            return self._json(413, {"error": f"file is larger than {MAX_FILE_MB} MB"})
        filename = Path(unquote(self.headers.get("X-Filename") or "document.pdf")).name
        try:
            doc = STORE.add(filename, self.rfile.read(size))
        except ValueError as e:  # not a PDF, encrypted, too many pages
            return self._json(400, {"error": str(e)})
        self._json(200, {"doc_id": doc.doc_id, "title": doc.title, "filename": doc.filename,
                         "pages": doc.page_count, "headings": len(doc.headings),
                         "heading_source": doc.heading_source, "has_text": doc.has_text,
                         "parse_ms": doc.parse_ms, "budget": harness.TOOL_BUDGET})

    def _ask(self) -> None:
        try:
            size = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(min(size, MAX_ASK_BYTES)) or b"{}")
            doc = STORE.get(str(body.get("doc_id", "")))
            question = str(body.get("question", ""))
            history = [(str(q), str(a)) for q, a in body.get("history", [])][-harness.HISTORY_TURNS:]
            provider = str(body.get("provider") or harness.PROVIDER)  # the UI's model menu, per question
            if provider not in PROVIDERS:
                raise ValueError(f"unknown provider '{provider}' (use {' or '.join(PROVIDERS)})")
        except (ValueError, TypeError, AttributeError) as e:  # bad JSON, unknown doc/provider, bad history
            return self._json(400, {"error": str(e)})

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        calls = 0

        def stream(rec: dict) -> None:  # every trace event goes to the browser as one JSON line
            nonlocal calls
            if rec["event"] == "tool" and rec.get("tool") in harness.TOOLS:
                calls += 1
            self._line(rec)

        t0 = time.time()
        try:
            llm = LLM_FACTORY(provider) if LLM_FACTORY else harness.make_llm(provider)
            r = harness.run_agent(question, doc_id=doc.doc_id, doc_name=doc.filename, history=history,
                                  llm=llm, trace_dir=TRACE_DIR, on_event=stream)
            self._line({"type": "result", "status": r.status, "output": r.output, "error": r.error,
                        "run_id": r.run_id, "trace_url": f"/api/trace/{r.run_id}", "calls": calls,
                        "provider": provider, "model": getattr(llm, "model", None),
                        "elapsed_ms": int((time.time() - t0) * 1000)})
        except Exception as e:  # e.g. an API key missing: show it in the chat instead of a dead stream
            self._line({"type": "error", "error": f"{type(e).__name__}: {e}"})

    # ---- helpers ----

    def _allowed(self) -> bool:
        """Everything except the page and the health check needs the access code, when one is set.
        Answers 401 itself if not. compare_digest: the check takes the same time however much of the code is right."""
        given = self.headers.get("X-Access-Code") or ""
        if not ACCESS_CODE or hmac.compare_digest(given.encode(), ACCESS_CODE.encode()):
            return True
        self._json(401, {"error": "This demo needs an access code: enter it at the top of the page." if not given
                         else "Wrong access code."})
        return False

    def _line(self, rec: dict) -> None:
        self.wfile.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self.wfile.flush()

    def _send(self, code: int, data: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code: int, obj: dict) -> None:
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def log_message(self, fmt: str, *args) -> None:  # one short line per request
        print(f"[server] {self.address_string()} {fmt % args}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Defaults come from HOST / PORT when set (hosting platforms set them), else local-only on port 8000."""
    parser = argparse.ArgumentParser(description="Chat UI backend for the budgeted document agent")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"),  # 0.0.0.0 only inside a container
                        help="127.0.0.1 = this machine only (default); 0.0.0.0 = reachable from the network")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--trace-dir", default=str(TRACE_DIR))
    return parser.parse_args(argv)


def main() -> None:
    global TRACE_DIR
    args = parse_args()
    TRACE_DIR = Path(args.trace_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    shown = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Ledger is running: open http://{shown}:{args.port}   (traces -> {TRACE_DIR}/)")
    keys = ", ".join(f"{var} {'set' if os.environ.get(var) else 'MISSING'}" for _, var in PROVIDERS.values())
    print(f"API keys: {keys}   (from the terminal or {harness.ENV_FILE.name}; values never shown)")
    print(f"Access code: {'required' if ACCESS_CODE else 'off (open to anyone who can reach this server)'}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
