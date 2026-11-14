# Memo: Budgeted Document-Answering Agent ("Ledger")

**The Tech Crowd** · Kanagavel A K, Gurumurthy S · RAP Hackathon, 5 Oct 2026

## Architecture

`PDF upload → DocStore (parsed once) → 4 tools (+ call logger) ⇄ harness loop ⇄ LLM → evidence check → answer with citations, or "insufficient information" → JSONL trace + live UI ledger`

- **DocStore** (`docstore.py`, PyMuPDF) is the tools' backend: page text, headings and a keyword index. Only the 4 tools read it; the model never does.
- **Harness** (`harness.py`, plain Python, no framework): input check → model turn → budget gate → tool → untrusted-text wrapper → back to the model. The final JSON answer gets a format check and an evidence check.
- **LLM:** Groq `gpt-oss-120b` at temperature 0; a Gemini adapter with the same interface is a backup, switchable per question.
- **UI and server** (standard library): chat, a live budget meter, and a ledger of every call with the full trace.

## What we did and why

Principle: **reading costs budget, thinking is free, and every rule that matters is enforced in code, not left to the prompt.**

- **Budget gate.** Counts tool calls; a 7th call never reaches a tool. Repeated calls are blocked and not counted. When the budget is spent, time runs short (75 s), the model repeats itself twice, or the last turn arrives, the request goes out with `tool_choice="none"`, so the API itself forces the final answer.
- **Evidence check.** Every quote (≥ 15 characters) must appear on a page read with `get_page` *in this question*. A failing answer gets one fix attempt, then becomes "insufficient information". Guessing costs more than declining, so declining is the default when evidence is missing.
- **Untrusted text.** Page text is fenced as `<document_content>`, lines that address an AI are flagged, and invisible text (white or < 4 pt) is marked and can never be cited. Injection attempts in the question are rejected before the model sees them.
- **Search without embeddings, on any PDF.** A positional inverted index: a word-family phrase match (plural, -ed, -ing), then a typo-tolerant match; page numbers only, as specified. Headings come from bookmarks, else font size and bold, else numbering.
- **Prompt strategy.** Search the most specific term; an empty result means one synonym or the headings, then decline. Batch independent calls. For amendments, check the last hit: the later statement wins and both pages are cited; unresolved conflicts are reported, not chosen.

## How we tested

63 offline tests replay scripted model replies (budget, repeats, injection, hidden text, evidence, unusual layouts); live tests run against Groq. Our first live run passed 6 of 9 questions, and every failure became a test, then a fix. Examples: Groq rejected `tool_choice=null` at the end of the budget; a correct quote failed on a non-breaking hyphen; on an unseen PDF the model repeated one search until the step limit.

## Assumptions

Only the 4 document tools count toward the 6; the final JSON reply is the "+1". Each question starts with a fresh budget and an empty evidence ledger; history carries earlier questions and answers, never page text. Parsing at upload builds the tools' backend; the agent sees only what tool calls return.

## Known failure modes and intended fixes

| Problem | Intended fix |
|---|---|
| Scanned PDFs have no text layer (no OCR) | OCR at upload, inside the tools' backend |
| Table cells in separate text blocks come out one per line | PyMuPDF table detection |
| "Later statement wins" is guided by the prompt, not checked in code | If budget remains, search amendment terms before accepting |
| Strict quote matching can reject a correct paraphrase | Fuzzy matching with a high threshold, logged in the trace |
| Groq's free tier (8k tokens/min) makes hard questions slow | Paid tier, or the Gemini backup |
| Text hidden by background colour or behind images is not detected | Compare the text colour with the shapes drawn under it |
