"use strict";
// Frontend for the budgeted document agent. It talks to server.py:
//   POST /api/upload  (PDF bytes)  -> document summary
//   POST /api/ask     (JSON)       -> NDJSON stream: trace events as they happen, then one {"type":"result"} line
// All text that comes from the document or the model is inserted with textContent, never innerHTML,
// so a PDF cannot inject HTML or scripts into the page.

const state = { doc: null, history: [], busy: false, shown: null };
const TOOL_NAMES = ["list_documents", "list_headings", "get_page", "search_keyword"];  // only these count
const $ = (id) => document.getElementById(id);

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

// ---------------------------------------------------------------- backend check (on page load)

const OPEN_FROM_SERVER = "Start the backend with  python server.py  and open http://127.0.0.1:8000";

// fetch() throws a TypeError ("Failed to fetch") when nothing answers: say what to do instead.
const friendly = (err) => (err instanceof TypeError ? `Cannot reach the backend. ${OPEN_FROM_SERVER}` : err.message);

async function checkBackend() {
  if (location.protocol === "file:") {  // opened as a file: there is no server behind the page
    showDocMessage(`This page was opened as a file. ${OPEN_FROM_SERVER}`, true);
    return;
  }
  try {
    state.health = await (await fetch("/api/health")).json();
    fillModelMenu(state.health);
    reportModel();
  } catch (err) {
    showDocMessage(friendly(err), true);
  }
}

// ---------------------------------------------------------------- model menu (Groq, or the Gemini backup)

const KEY_VARS = { groq: "GROQ_API_KEY", gemini: "GEMINI_API_KEY" };
const LABELS = { groq: "Groq", gemini: "Gemini (backup)" };

function fillModelMenu(h) {
  const menu = $("provider");
  menu.replaceChildren();
  for (const [name, p] of Object.entries(h.providers)) {
    const option = el("option", null, `${LABELS[name] || name} · ${p.model}${p.key_set ? "" : " (no key)"}`);
    option.value = name;
    option.disabled = !p.key_set;  // a provider without a key in the server's terminal cannot be picked
    menu.append(option);
  }
  let saved = null;
  try { saved = localStorage.getItem("provider"); } catch { /* storage blocked: use the default */ }
  const usable = Object.keys(h.providers).filter((n) => h.providers[n].key_set);
  menu.value = [saved, h.default_provider, usable[0]].find((n) => n && usable.includes(n)) || h.default_provider;
}

$("provider").addEventListener("change", () => {
  try { localStorage.setItem("provider", $("provider").value); } catch { /* not remembered: fine */ }
  reportModel();
});

function reportModel() {
  const h = state.health;
  if (!h) return;
  const name = $("provider").value;
  const p = h.providers[name];
  if (!p || !p.key_set) {
    showDocMessage(`${KEY_VARS[name] || "The API key"} is not set: questions will fail. Paste it into the .env ` +
      "file next to server.py (or set it in the server's terminal), then restart the server.", true);
  } else if (!state.doc) {
    showDocMessage(`Backend ready · ${LABELS[name] || name} · ${p.model} · ${h.budget} reads per question · ` +
      "upload a PDF to start");
  }
}

const geminiReady = () => Boolean(state.health?.providers?.gemini?.key_set) && $("provider").value !== "gemini";

// ---------------------------------------------------------------- upload

$("file").addEventListener("change", (e) => upload(e.target.files[0]));
document.addEventListener("dragover", (e) => { e.preventDefault(); $("empty")?.classList.add("over"); });
document.addEventListener("dragleave", () => $("empty")?.classList.remove("over"));
document.addEventListener("drop", (e) => {
  e.preventDefault();
  $("empty")?.classList.remove("over");
  if (e.dataTransfer.files[0]) upload(e.dataTransfer.files[0]);
});

async function upload(file) {
  if (!file || state.busy) return;
  showDocMessage(`Reading ${file.name} ...`);
  try {
    const res = await fetch("/api/upload", {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(file.name), "Content-Type": "application/pdf" },
      body: file,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    const hadConversation = state.doc !== null && $("thread").querySelector(".turn");
    state.doc = data;
    state.history = [];  // follow-ups never carry over to a different document
    renderDoc(data);
    if (hadConversation) $("thread").append(el("div", "divider", `New document: ${data.filename}`));
    if ($("empty")) {
      $("empty").querySelector(".drop-title").textContent = "Ready";
      $("empty-text").textContent = `Ask anything about ${data.filename}. Try a question whose answer spans ` +
        "several pages, one about a rule that was later changed, or one the document cannot answer.";
    }
    setBusy(false);
    $("q").placeholder = `Ask about ${data.filename}  (Enter to send, Shift+Enter for a new line)`;
    $("q").focus();
  } catch (err) {
    showDocMessage(`Could not read the PDF: ${friendly(err)}`, true);
  }
  $("file").value = "";
}

function renderDoc(d) {
  const strip = $("doc");
  strip.hidden = false;
  strip.replaceChildren(el("span", "docname", d.filename));
  const item = (value, label) => {
    const s = el("span");
    s.append(el("b", null, value), document.createTextNode(" " + label));
    strip.append(s);
  };
  item(String(d.pages), "pages");
  item(String(d.headings), `headings (${d.heading_source})`);
  item(`${d.parse_ms} ms`, "to index");
  item(String(d.budget), "reads per question");
  if (!d.has_text) strip.append(el("span", "warn", "No text layer found: probably a scanned PDF (no OCR)."));
  resetLedger();
}

function showDocMessage(msg, isError) {
  const strip = $("doc");
  strip.hidden = false;
  strip.replaceChildren(el("span", isError ? "warn" : null, msg));
}

// ---------------------------------------------------------------- ask

$("ask").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (q && state.doc && !state.busy) ask(q);
});
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask").requestSubmit(); }
});

async function ask(question) {
  setBusy(true);
  $("q").value = "";
  $("empty")?.remove();
  const run = { question, events: [], result: null };
  const turn = el("div", "turn");
  const pending = el("p", "pending", "planning");
  turn.append(el("p", "q", question), pending);
  $("thread").append(turn);
  scrollDown();
  showLedger(run);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: state.doc.doc_id, question, history: state.history.slice(-3),
                             provider: $("provider").value }),  // the model menu: Groq or the Gemini backup
    });
    if (!res.ok) {
      const d = await res.json().catch(() => ({}));
      throw new Error(d.error || res.statusText);
    }
    await readLines(res, (msg) => {
      if (msg.type === "result" || msg.type === "error") run.result = msg;
      else { run.events.push(msg); pending.textContent = progressText(run); }
      if (state.shown === run) renderLedger(run);
    });
  } catch (err) {
    run.result = { type: "error", error: friendly(err) };
  }
  pending.remove();
  turn.append(renderAnswer(run));
  if (run.result?.output) state.history.push([question, run.result.output.answer]);
  renderLedger(run);
  setBusy(false);
  scrollDown();
  $("q").focus();
}

// Reads an NDJSON response line by line as it arrives (this is what makes the ledger live).
async function readLines(res, onLine) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, i).trim();
      buf = buf.slice(i + 1);
      if (line) onLine(JSON.parse(line));
    }
  }
  if (buf.trim()) onLine(JSON.parse(buf));
}

function usedCalls(run) {  // the harness reports budget_left after every counted call
  const tools = run.events.filter((e) => e.event === "tool");
  return tools.length ? budget() - tools[tools.length - 1].budget_left : 0;
}
const budget = () => state.doc?.budget ?? 6;

function progressText(run) {
  const last = run.events[run.events.length - 1];
  if (last.event === "model_error") return last.rejected_tool ? "checking the answer" : "rate limited, waiting";
  if (last.event === "rate_wait") return `pacing for the rate limit (${last.wait_s} s)`;
  if (last.event === "forced_final") return "writing the answer";
  if (last.event === "verify_failed") return "evidence check failed, asking for exact quotes";
  return `reading · ${usedCalls(run)} of ${budget()} calls used`;
}

function setBusy(busy) {
  state.busy = busy;
  $("q").disabled = busy || !state.doc;
  $("send").disabled = busy || !state.doc;
  $("send").textContent = busy ? "Reading" : "Ask";
}

const scrollDown = () => { $("thread").scrollTop = $("thread").scrollHeight; };

// ---------------------------------------------------------------- answer card

function renderAnswer(run) {
  const r = run.result || { type: "error", error: "The connection closed before an answer arrived." };
  const out = r.output;
  let kind, label;
  if (r.type === "error") [kind, label] = ["failed", "Error"];
  else if (r.status === "rejected_input") [kind, label] = ["failed", "Question rejected"];
  else if (r.status !== "success") [kind, label] = ["failed", r.status.replace("_", " ")];
  else if (out.status === "answered") [kind, label] = ["answered", "Answered"];
  // Small talk ("thank you"): nothing read and no decline in the text, so a neutral stamp, not a red one.
  else if (r.calls === 0 && !/insufficient/i.test(out.answer)) [kind, label] = ["reply", "Reply"];
  else [kind, label] = ["declined", "Insufficient information"];

  const card = el("article", `answer ${kind}`);
  card.append(el("span", `stamp ${kind}`, label));
  card.append(el("p", "answer-text", out ? out.answer : explainError(r.error || "Something went wrong.")));
  if (out?.citations?.length) {
    const list = el("ol", "cites");
    for (const c of out.citations) {
      const li = el("li");
      li.append(el("span", "pg", `p.${c.page}`), el("span", "quote", `“${c.quote}”`));
      list.append(li);
    }
    card.append(list);
  }
  if (out?.steps_summary) card.append(el("p", "how", `How: ${out.steps_summary}`));

  const has = (name, test = () => true) => run.events.some((e) => e.event === name && test(e));
  if (has("final", (e) => e.downgraded)) {
    card.append(el("p", "note", "Its quotes could not be found on the pages it read, so it declined instead of guessing."));
  } else if (has("verify_failed") && has("verify_ok")) {  // "corrected" only if a later check passed
    card.append(el("p", "note", "A quote failed the evidence check and was corrected."));
  } else if (has("verify_failed")) {
    card.append(el("p", "note", "A quote failed the evidence check and could not be fixed in time."));
  }
  if (has("injection_flagged")) {
    card.append(el("p", "note", "Text in the document tried to give the agent instructions. It was flagged and ignored."));
  }

  const foot = el("footer");
  foot.append(el("span", null, `${usedCalls(run)} of ${budget()} reads`));
  if (r.elapsed_ms) foot.append(el("span", null, `${(r.elapsed_ms / 1000).toFixed(1)} s`));
  if (r.model) foot.append(el("span", null, r.model));
  const tokens = run.events.filter((e) => e.event === "model").reduce((sum, e) => sum + (e.tokens?.in || 0) + (e.tokens?.out || 0), 0);
  if (tokens) foot.append(el("span", null, `${(tokens / 1000).toFixed(1)}k tokens`));
  const show = el("button", "linkish", "show ledger");
  show.type = "button";
  show.addEventListener("click", () => showLedger(run));
  foot.append(show);
  if (r.trace_url) {
    const a = el("a", null, "trace");
    a.href = r.trace_url;
    foot.append(a);
  }
  card.append(foot);
  return card;
}

// Plain-language reason for a failed question; the raw error stays in the trace.
function explainError(text) {
  const other = $("provider").value === "gemini" ? "Groq" : "Gemini (backup)";
  if (/\b503\b|UNAVAILABLE|high demand|overloaded/i.test(text)) {
    return "The model's servers are overloaded right now (HTTP 503). This is on the provider's side, not your key " +
      `or your question. Ask again in a minute, or switch Model to ${other}.`;
  }
  if (/\b429\b|rate limit|RESOURCE_EXHAUSTED/i.test(text)) {
    return `Rate limit reached (HTTP 429): too many tokens this minute. Wait a minute or switch Model to ${other}.`;
  }
  if (/API_KEY|api key|\b401\b|\b403\b/i.test(text)) {
    return "The API key was missing or rejected. Set it in the terminal running server.py and restart it.";
  }
  return text.length > 300 ? text.slice(0, 300) + " ..." : text;
}

// ---------------------------------------------------------------- call ledger

function resetLedger() {
  state.shown = null;
  $("ledger-q").textContent = "No question yet";
  drawMeter(0);
  $("calls").replaceChildren();
  $("trace-link").hidden = true;
}

function showLedger(run) {
  state.shown = run;
  renderLedger(run);
}

function drawMeter(used) {
  const meter = $("meter");
  meter.replaceChildren();
  for (let i = 1; i <= budget(); i++) meter.append(el("div", i <= used ? "cell used" : "cell", String(i)));
  $("meter-text").textContent = `${used} of ${budget()} reads used`;
}

function renderLedger(run) {
  $("ledger-q").textContent = run.question;
  drawMeter(usedCalls(run));
  const list = $("calls");
  list.replaceChildren();
  let lastTool = null;
  for (const ev of run.events) {
    const row = ledgerRow(ev);
    if (ev.event === "injection_flagged" && lastTool) {
      lastTool.querySelector(".name").append(el("span", "badge flag", `flagged ${ev.lines}`));
      continue;
    }
    if (!row) continue;
    if (ev.event === "tool") lastTool = row;
    list.append(row);
  }
  const link = $("trace-link");
  link.hidden = !run.result?.trace_url;
  if (run.result?.trace_url) link.href = run.result.trace_url;
}

function row(cls, n, name, res) {
  const li = el("li", cls);
  li.append(el("span", "n", n));
  const body = el("div");
  body.append(el("div", "name", name));
  if (res) body.append(el("div", "res", res));
  li.append(body);
  return li;
}

function ledgerRow(ev) {
  switch (ev.event) {
    case "tool": {
      const counted = TOOL_NAMES.includes(ev.tool);  // an unknown tool never reaches a tool: not counted
      const n = String(budget() - ev.budget_left).padStart(2, "0");
      const li = row("", counted ? n : "-", `${ev.tool} ${describeArgs(ev)}`, describeResult(ev));
      if (ev.search_tier && ev.search_tier !== "none") li.querySelector(".name").append(el("span", "badge tier", ev.search_tier));
      return li;
    }
    case "repeat_blocked": return row("blocked", "-", `${ev.tool} ${describeArgs(ev)}`, "repeat blocked, not counted");
    case "budget_refused": return row("blocked", "-", `${ev.tool} ${describeArgs(ev)}`, "refused: budget used up");
    case "model": return row("info", "", `model turn · ${(ev.latency_ms / 1000).toFixed(1)} s` +
      (ev.model ? ` · ${ev.model}` : ""));  // shows which Gemini model answered after a fallback
    case "model_error":
      if (ev.rejected_tool) {  // the model called a tool that does not exist (gpt-oss: its answer wrapped in "json")
        return row("info", "", `model called non-existent tool "${ev.rejected_tool}"`,
          "API rejected it; the harness took the answer out of it and checks it like any answer (no budget used)");
      }
      if (ev.status_code === 429) {  // rate limit: the harness waits as long as the API says, then retries
        return row("info warn", "", ev.retry_in_s != null ? `rate limit: retrying in ${ev.retry_in_s} s` : "rate limit",
          geminiReady() ? "tokens per minute used up: switch Model to Gemini (backup) for the next question"
                        : "tokens per minute used up; a paid tier raises the limit");
      }
      if (ev.status_code === 503 || ev.status_code === 500) {  // provider overloaded (all fallback models tried)
        return row("info warn", "", ev.retry_in_s != null ? `provider overloaded (503): retrying in ${ev.retry_in_s} s`
          : "provider overloaded (503)", "the provider's servers are busy, not your key");
      }
      return row("info bad", "", ev.retry_in_s != null ? `API error, retrying in ${ev.retry_in_s} s` : "API error",
        (ev.error || "").slice(0, 120));
    case "rate_wait":
      return row("info warn", "", `pacing for the rate limit: waited ${ev.wait_s} s`,
        "waited before the call so it fits the per-minute limit (no error)" +
        (geminiReady() && ev.wait_s >= 10 ? "; long waits: switch Model to Gemini (backup)" : ""));
    case "forced_final": return row("info", "", `final answer forced (${ev.reason})`);
    case "bad_output": return row("info bad", "", "reply was not valid JSON, asked to fix it");
    case "verify_ok": return row("info ok", "", `quotes verified on p.${(ev.pages_read || []).join(", p.")}`);
    case "verify_failed": return row("info bad", "", "evidence check failed", (ev.problems || []).join("; "));
    case "input_rejected": return row("info bad", "", "question rejected", (ev.problems || []).join("; "));
    case "timeout": return row("info bad", "", "time limit reached");
    default: return null;  // start, final, end
  }
}

function describeArgs(ev) {
  let a;
  try { a = JSON.parse(ev.args); } catch { return ev.args || ""; }
  if (ev.tool === "search_keyword") return `"${a.keyword}"`;
  if (ev.tool === "get_page") return `p.${a.page_number}`;
  return "";
}

function describeResult(ev) {
  if (!ev.ok) return (ev.observation || "error").replace(/^ERROR:\s*/, "").slice(0, 140);
  if (ev.tool === "search_keyword") {
    const pages = JSON.parse(ev.observation || "[]");
    return pages.length ? `→ pages ${pages.join(", ")}` : "→ no pages";
  }
  if (ev.tool === "get_page") return `→ ${(ev.observation || "").split("\n")[0]}`;
  if (ev.tool === "list_headings") return "→ table of contents";
  return "";
}

resetLedger();
checkBackend();
