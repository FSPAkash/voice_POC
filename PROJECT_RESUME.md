# DHL Collections AI — Voice POC

**Author:** Akash Patil
**Stack:** React 19 + TypeScript + Vite | Flask 3 + Python 3.12 | OpenAI Realtime (WebRTC)
**Deploy:** Docker on Render (Singapore), single container, self-keepalive
**Repo layout:** `backend/` (Flask API + prompts + mock SAP data) · `frontend/` (Vite SPA) · `Dockerfile` + `render.yaml` (deploy) · `scripts/` (pricing doc gen)

---

## 1. What This Is

Browser-based voice agent that simulates a **DHL Express India outbound B2B collections call**. Customer talks; AI calls back as "Yogesh" (or one of 10 personas) and walks the customer through overdue invoices, captures promise-to-pay, handles disputes, or escalates to a human. Built against the DHL provisional SoW (`Provisional_Requirement_Document_Apar.pdf`) — target volume 100K calls/month, 90–120s each, multilingual (Hinglish + 22 Indian languages).

POC goal: prove the call flow, grounded data, supervision, language switching, and per-call cost economics — without telephony or live SAP yet.

---

## 2. Architecture — Hybrid Voice Renderer + Deterministic Policy Brain

Key design decision: **OpenAI Realtime is a voice renderer, not the brain.** Reasons: lower hallucination, cheaper per-minute, SAP-groundable, maintainable as data grows.

```
Browser (React)  --WebRTC audio--  OpenAI Realtime (gpt-realtime-mini)
      |                                  |
      |  transcript text                  |  TTS only
      v                                  ^
Flask backend  -> language coach (gpt-4.1-mini)
              -> chat-turn policy engine (gpt-4.1) w/ JSON-mode, grounded SAP context
              -> supervisor reviewer (gpt-4.1-mini)
              -> summarizer (gpt-4.1-mini)
              -> mock SAP tools (get_customer, get_invoices, log_promise_to_pay, ...)
              -> cost ledger (per-million token pricing table)
```

- **Voice path:** WebRTC peer connection, mic in / audio out. `gpt-realtime-mini` default (with `gpt-realtime` toggle). Transcription via `gpt-4o-mini-transcribe`.
- **Brain path:** every customer turn -> backend `/api/turn/customer` -> language detection + grounded reply generation + tool invocation, single round-trip. Realtime API receives the **approved scripted text** to speak; never freelances dialog.
- **Defense layers in [backend/app.py](backend/app.py):** forbidden payment-method scrubber, invented-invoice-number scrubber, promise-date validator, tool allow-list, system-prompt-leak detector, language-compliance retry, deterministic regex-tree fallback if LLM JSON invalid.

---

## 3. Tech Stack

### Frontend ([frontend/package.json](frontend/package.json))
- React 19.2 (`useEffectEvent`, `useDeferredValue`, `startTransition` for concurrent rendering)
- TypeScript 6.0, Vite 8, ESLint 10
- Zero UI library — hand-rolled CSS (2,052 lines in [frontend/src/index.css](frontend/src/index.css)) using CSS variables, DHL brand palette (`--accent: #ffcc00`, `--accent-2: #d40511`)
- WebRTC `RTCPeerConnection` + data channel for realtime event stream
- localStorage persistence: call history, ambience volume, pending cost events (retry queue)

### Backend ([backend/requirements.txt](backend/requirements.txt))
- Flask 3.1 + flask-cors + Gunicorn
- OpenAI Python SDK 2.7
- File-backed state (`data/*.json`, `*.jsonl`) — no DB, intentional for POC
- 19 REST endpoints ([backend/app.py:3182-3645](backend/app.py))

### Infra
- Multi-stage [Dockerfile](Dockerfile): node:22 builds frontend → python:3.12 serves both
- [render.yaml](render.yaml) — free tier, Singapore, Docker runtime
- [backend/keep_alive.py](backend/keep_alive.py) — self-pings `/health` every 14 min to defeat Render free-tier sleep

---

## 4. Features Built

### Call Engine
- Outbound voice mode (WebRTC) **and** text-chat mode (no audio billing) — same backend turn engine
- Headless mode (audio-only UI for demos)
- Ambient call-center background sound, volume-faded based on speaking state, [frontend/public/sound/call_center_background.wav](frontend/public/sound/call_center_background.wav)
- Per-call elapsed timer, mic level meter, mute, barge-in (cancels agent reply on real customer speech, ignores cough/echo via 3-token substantive check)
- 10 voice personas mapped to gendered names (cedar→Yogesh m, marin→Priya f, etc.) so the AI never says a male name on a female voice

### Language Coach
- 24 supported languages (Hinglish + English + 22 Indian regional) — [backend/app.py:170-203](backend/app.py)
- Per-turn language detection on customer transcript; suggests switch when confidence high
- Transcript quality flag (good/suspect) drives retry/repair logic
- Live mid-stream language-compliance violation detection on the agent reply — discards and re-fetches approved script

### Supervisor
- Rule-based supervisor reviews every agent turn for grounding/policy violations
- Kanban board UI: New → Reviewing → Accepted → Dismissed ([frontend/src/components/SupervisorBoard.tsx](frontend/src/components/SupervisorBoard.tsx))
- Coaching hints feed back into next agent turn

### Tools (mock SAP) — [backend/app.py:225-348](backend/app.py)
`get_customer`, `get_invoices`, `log_promise_to_pay`, `log_already_paid`, `resend_invoice`, `log_dispute`, `update_contact`, `transfer_to_human`. All persist to `data/tool_actions.jsonl`.

### Cost Accounting
- Per-million-token price table for every model used (realtime audio in/out/cached, transcription, chat) — [backend/app.py:90-161](backend/app.py)
- Live USD ledger in UI, split by agent vs supervisor vs language_coach vs chat_agent
- Pending cost event queue with retry (survives backend hiccups)
- `gpt-realtime` GA vs `gpt-realtime-mini` toggle for cost A/B

### Call Wrap-up
- LLM summarizer produces structured disposition (Promise-to-pay / Already paid / Dispute / Escalation / etc.)
- Call history with per-call cost + token count, persisted to localStorage

### Auth
- Simple login screen ([frontend/src/components/Login.tsx](frontend/src/components/Login.tsx)) — POC-grade

---

## 5. UI Design

Layout: **3-pane operator console** (DHL brand-themed, light mode).

- **Topbar:** DHL logo, FleetSpeed co-brand, call state pill (Ready / Connecting / Live / Ending), elapsed timer, username + logout
- **Left:** customer card, invoice table, agent activity log (6-node graph: Caller / Chat / Lang Coach / Supervisor / Tool / Summarizer)
- **Center:** transcript stream (color-coded customer / assistant / system), big Start Call / End Call CTA, mic meter, mute, language selector, voice persona selector, realtime-model toggle
- **Right tabs:** Call · Wrap-up · Supervisor board
- **Bottom bar:** live cost (USD + tokens) split by agent role

Visual system:
- DHL yellow `#ffcc00` + red `#d40511` accents
- Pill-shaped status indicators ([frontend/src/components/StatusPill.tsx](frontend/src/components/StatusPill.tsx))
- 14px base, system font stack, generous whitespace, no shadows beyond `--shadow-1`
- Concurrent React (`useDeferredValue` on transcript) keeps streaming UI smooth under heavy delta load

---

## 6. Notable Engineering Decisions

- **Single round-trip per turn** — collapsed previous serial waterfall (lang coach → policy → speak) into `/api/turn/customer` to cut latency by ~40%
- **Barge-in heuristic** — VAD alone caused false cancels on coughs; require ≥3 substantive transcribed tokens before cancelling agent
- **STT hallucination filter** — Whisper regurgitates its own prompt vocab on silence; filter by marker phrases + vocab-density heuristic ([frontend/src/App.tsx:96-152](frontend/src/App.tsx))
- **Approved-script replay** — if the realtime model leaks system prompt or violates language lock, drop the draft and re-issue the last backend-approved script
- **Cost event idempotency** — events keyed by `session_id:event_id`, retry queue in localStorage survives reloads
- **Pricing model comparison doc** — [PRICING_COMPARISON.md](PRICING_COMPARISON.md) + [scripts/make_pricing_docx.py](scripts/make_pricing_docx.py) generate a Word doc for stakeholders comparing realtime vs chat-only architectures

---

## 7. Data / Knowledge Artifacts

- [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md) — full SoW digest (DHL business rules, branches, escalation matrix)
- [POC_TECHNICAL_PLAN.md](POC_TECHNICAL_PLAN.md) — architecture rationale
- [backend/prompts/agent.md](backend/prompts/agent.md) — 115-line collections playbook (call shape, anti-patterns, branch decision tree)
- [backend/prompts/supervisor.md](backend/prompts/supervisor.md), [language_coach.md](backend/prompts/language_coach.md), [call_summary.md](backend/prompts/call_summary.md)
- [backend/data/sap_mock.json](backend/data/sap_mock.json) — mock customer, invoices, payment methods, history
- [backend/data/GROUND_TRUTH.md](backend/data/GROUND_TRUTH.md) + xlsx — eval dataset
- [backend/tests/test_cost_accounting.py](backend/tests/test_cost_accounting.py) — cost-ledger unit tests

---

## 8. Skills Demonstrated

Full-stack TypeScript/React + Python/Flask · WebRTC + realtime streaming media · OpenAI Realtime API + tool use + structured JSON output · Prompt engineering + LLM guardrails (prompt-leak detection, language compliance, output validation) · Cost modelling for token-billed APIs · Docker multi-stage builds · Render deployment + free-tier keepalive · Hand-rolled CSS design system · React 19 concurrent features · Domain modelling for B2B collections workflow
