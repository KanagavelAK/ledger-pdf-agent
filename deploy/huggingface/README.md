---
title: Ledger PDF Agent
emoji: 📒
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: Answers from any PDF in 6 reads, every claim quoted
---

# Ledger: a budgeted document-answering agent

Upload a PDF and ask questions. The agent reads the document through four tools and at most six reads per
question; every claim is quoted from a page it read and checked by code. If the document doesn't say, it
answers "insufficient information".

This demo is protected by an access code so its API quota isn't used up. Ask the authors for the code, or run
it yourself: the full source, design notes and tests are at
**https://github.com/KanagavelAK/ledger-pdf-agent**.

Built by The Tech Crowd (Kanagavel A K and Gurumurthy S) at the RAP hackathon, October 2026.
