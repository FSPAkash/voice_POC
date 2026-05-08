# DHL Collections AI - POC Technical Plan (React + Flask + OpenAI Voice)

> Companion to `KNOWLEDGE_BASE.md`. Read that first for the DHL business rules and target behavior.
> This document now reflects the current implementation direction in this repo: **deterministic backend policy engine + OpenAI Realtime for voice rendering/transcription only**.

---

## 0. Current Build Summary

The architecture has intentionally moved away from "let the realtime model run the collections conversation."

Current implementation choices:
- Voice transport and speech output: `gpt-realtime` over browser WebRTC (renderer only)
- Realtime transcription: `gpt-4o-mini-transcribe` / configured realtime transcription model
- Live collections logic: **hybrid LLM turn engine** — `gpt-4.1-mini` driven by a strict system prompt with grounded SAP-mock context, JSON-structured output, and deterministic post-validation (forbidden payment-method scrubber, invented-invoice-number scrubber, promise-date validator, tool allow-list)
- Live language coach: deterministic backend heuristics
- Live supervisor: deterministic backend rule checks
- Wrap-up summary: deterministic backend summarizer
- Deterministic regex-tree engine retained as fallback when the LLM call fails or returns invalid JSON

Important design rule:
- `gpt-realtime` is now treated as a **voice renderer**.
- The backend decides **what to say**, **when to say it**, **which tool action to log**, and **which branch of the DHL flow applies**.

This is the core tradeoff:
- lower hallucination risk
- lower non-voice API cost
- easier SAP grounding
- easier long-term maintenance as DHL data grows

---

## 1. POC Goal

### 1.1 What the POC must prove
1. A browser-based "call" can run end-to-end against the DHL collections scenario in `KNOWLEDGE_BASE.md`.
2. The system can handle the required collections branches:
   - right contact / wrong contact
   - overdue invoice explanation
   - resolved-issue recall
   - already paid
   - invoice not received
   - dispute
   - cash-flow / approval-delay reasoning
   - promise-to-pay capture
   - escalation to human
3. All invoice and customer facts are grounded in backend data, currently `backend/data/sap_mock.json`, and later SAP.
4. The conversation rules stay stable even as invoice/customer/history data grows.
5. The UI still shows:
   - live transcript
   - active language state
   - tool/action log
   - disposition
   - supervisor findings board
   - call summary
   - estimated realtime voice cost

### 1.2 What the POC does not try to solve yet
- PSTN/telephony dial-out
- direct SAP integration
- enterprise auth, compliance, retention, and production observability
- high concurrency
- production-grade regional-language template coverage beyond the implemented paths

---

## 2. Architecture

### 2.1 High-level model

```text
+--------------------+        WebRTC (audio + data)         +-------------------------+
|  React (browser)   | <==================================> |  OpenAI Realtime API    |
|  - Mic capture     |                                       |  gpt-realtime           |
|  - Audio playback  |                                       |  voice renderer only    |
|  - Transcript UI   |                                       +-----------+-------------+
|  - Tool/action UI  |                                                   ^
|  - Supervisor UI   |                                                   |
+----------+---------+                                                   |
           |                                                             |
           | HTTP                                                        |
           v                                                             |
+-----------------------------------------------------------------------+|
| Flask backend                                                         ||
| - /api/session                 -> mint realtime client secret         ||
| - /api/turn/customer           -> unified per-turn endpoint           ||
|                                  (language coach + policy engine)     ||
| - /api/chat/turn               -> policy engine only (text mode)      ||
| - /api/language/detect         -> language routing only (legacy)      ||
| - /api/supervisor/evaluate     -> deterministic QA findings           ||
| - /api/call/summarize          -> deterministic wrap-up summary       ||
| - /api/tool/*                  -> mock SAP / disposition actions      ||
| - /api/metrics/costs*          -> estimated cost ledger               ||
| - /api/call/log                -> local call log                      ||
|                                                                       ||
| Ground truth sources:                                                  ||
| - backend/data/sap_mock.json (raw fixture)                            ||
| - backend/data/GROUND_TRUTH.md (auto-generated canonical doc          ||
|     prepended to every LLM turn — single source of truth for the      ||
|     LLM)                                                               ||
| - DHL collections rules from KNOWLEDGE_BASE.md encoded as code         ||
+-----------------------------------------------------------------------++
```

### 2.2 Key principle

There are two distinct layers:

1. Stable conversation policy
- DHL collections rules
- branch handling
- escalation criteria
- allowed payment methods
- promise-to-pay validation
- disposition rules

2. Dynamic business facts
- customer identity
- invoice list
- overdue days
- dispute history
- credit-note history
- email/contact records
- later: SAP-backed customer and billing data

This separation is the reason the architecture scales cleanly:
- data can grow without making the voice layer "smarter"
- policy can remain stable even as SAP adds more fields and records

### 2.3 Why this replaced the earlier design

The previous plan used:
- realtime model for live reasoning
- helper LLMs for language coaching
- helper LLMs for supervision
- helper LLMs for post-call summarization

That design created three problems:
- too many model hops per turn
- hallucination risk in the main conversation loop
- weak guarantees around SAP grounding and policy compliance

The current build fixes that by making the backend authoritative.

---

## 3. Realtime Role

### 3.1 What Realtime still does
- receive customer audio
- produce speech output
- provide low-latency barge-in capable audio session
- emit transcription events

### 3.2 What Realtime no longer does
- choose the next business action
- decide which DHL branch to follow
- decide whether to log PTP / dispute / escalation
- invent or retrieve invoice facts
- perform live collections reasoning

### 3.3 Runtime rule

The frontend now sends approved utterances to Realtime as exact speech instructions:
- "speak this exact line"
- do not improvise
- do not call tools
- do not continue the conversation on your own
- do not narrate or paraphrase these instructions back to the customer

That turns the voice model into a bounded renderer instead of a policy engine.

### 3.4 Anti-leak guard

The session-level prompt is phrased to forbid any meta-narration ("I can only repeat the approved response", etc.) because earlier builds saw the realtime model regurgitate its own system prompt as dialogue when it could not produce real conversation. To defend against any future drift, the frontend also runs `detectSystemPromptLeak` on every completed assistant turn (`response.done`). If the assistant output matches the leak pattern, the turn is suppressed from the visible transcript and the last approved line is re-issued (or a fresh approved reply is fetched). See `frontend/src/lib/realtime.ts`.

### 3.5 Barge-in policy

Realtime VAD fires `input_audio_buffer.speech_started` aggressively on coughs, breaths, and self-echo. The frontend deliberately does **not** cancel an active agent response on `speech_started` — that was truncating legitimate agent turns mid-sentence. Real barge-in is confirmed only when the customer's transcription event completes with substantive content (`>= 3` tokens), at which point the active agent response is cancelled and a new approved reply is generated.

---

## 4. Deterministic Backend Components

### 4.1 Policy engine (hybrid LLM + deterministic guards)

Implemented in `llm_collections_turn` in `backend/app.py`.

Per-turn flow:
1. Load the canonical `GROUND_TRUTH.md` (auto-generated from `sap_mock.json`, see §4.5) and prepend it to the LLM user prompt as a `CANONICAL GROUND TRUTH DOCUMENT` block.
2. Build a per-account GROUND-TRUTH context block from `sap_mock.json` (customer, invoices with full history, sanctioned payment methods, policy constants, human-transfer contact, collection notes) and append it after the doc.
3. Build a strict system prompt encoding the DHL collections policy as HARD RULES, including an explicit instruction that nothing outside those two blocks may be quoted.
4. Send transcript + grounded context to `gpt-4.1-mini` via the chat completions API with `response_format=json_object`.
5. Parse the structured response: `{intent, reply, language, tool_calls[]}`.
6. Run deterministic guards on the reply:
   - `scrub_forbidden_payment_methods` rewrites UPI/cheque/card/etc. mentions
   - `scrub_invented_invoice_numbers` replaces any `DHL\d+` token not present in the fixture
   - `reply_has_invented_amount` flags any `INR/₹/Rs.` amount ≥ 1000 that is not a valid per-invoice amount or the grand total. On detection the LLM is retried once with the explicit valid-amount set; if the retry still hallucinates, the turn falls through to the deterministic regex-tree fallback.
7. Run `validate_tool_args` per tool call: enforce tool allow-list, snap invoice numbers to known invoices, validate promise-to-pay dates against the 2-business-day window.
8. Execute approved tool calls and return the reply.

If the LLM call fails, returns non-JSON, returns an empty reply, or trips the invented-amount guard twice in a row, `run_chat_agent_turn` falls back to the legacy regex-tree `generate_collections_reply` so the call never breaks and the agent never speaks fabricated numbers.

Examples of supported backend decisions:
- if customer confirms identity on the opening turn, fetch invoices, state the DHL collections purpose immediately, and ask why payment is still pending
- if customer says "you called me" or asks what the call is about, restate the overdue-invoice reason for the call before asking for payment status
- if customer asks for payment options, return only:
  - DHL MyBill self-serve portal
  - Virtual Account Number bank transfer
- if customer gives a date, validate it against the 2-business-day rule
- if customer says "already paid", log the already-paid path
- if customer raises a dispute, log dispute and steer correctly
- if customer shows distress or safety risk, escalate immediately

### 4.2 Language coach

Now deterministic.

Responsibilities:
- detect explicit language requests
- detect likely plain English
- detect suspect transcript / prompt echo hallucination
- choose the suggested reply language
- update transcription hint language
- tell the UI whether the next turn should switch language

Current behavior:
- explicit customer language request wins
- plain English customer turn forces English reply, but is tracked as inferred language preference rather than an explicit switch request
- suspicious transcript causes "ask them to repeat" behavior
- no extra model call is required

### 4.3 Supervisor

Now deterministic.

Responsibilities:
- inspect the latest assistant turn
- compare it against transcript context and invoice history
- flag concrete defects

Current rule checks include:
- asked for account number when account is already preloaded
- claimed there were no resolved issues when invoice history says otherwise
- mentioned forbidden payment methods
- missed explicit English switch
- trailed off without a complete actionable turn

Output:
- structured issue objects
- persisted to local board/log storage
- surfaced in the existing Supervisor Board UI

### 4.4 Ground-truth document (`GROUND_TRUTH.md`)

The LLM in §4.1 receives a single canonical ground-truth document inline on every turn. The document covers:

1. The customer of record (account, company, contacts, phone, email, payment terms, language preferences).
2. Every outstanding invoice with full per-invoice tables (invoice number, type, amount, currency, invoice date, due date, overdue days, history).
3. The whitelist of allowed numeric values: every per-invoice amount, the grand total, every overdue-days count, every invoice/due date with multiple natural-language renderings.
4. Sanctioned payment methods + the explicit list of forbidden channels.
5. Policy constants (PTP window, monthly target, proof-of-payment email, allowed dispositions).
6. Human escalation contact.
7. Conversation flow rules from `KNOWLEDGE_BASE.md` §9–11.
8. Tone and abuse rules.
9. **Hard prohibitions** — explicit "MUST NOT" clauses naming every fabrication class observed (invented amounts, invented months/years, invented channels, invented contacts, rounding/blending).

`backend/data/GROUND_TRUTH.md` is **auto-generated** from `backend/data/sap_mock.json` by `backend/scripts/generate_ground_truth.py`. The generator is invoked:

- once at server startup via `ensure_state()` (regen is idempotent and cheap), and
- on demand via `python backend/scripts/generate_ground_truth.py`.

This means the document the LLM sees never drifts out of sync with the SAP fixture: editing `sap_mock.json` (or, post-PoC, refreshing SAP-backed data) automatically rebuilds the doc the next time the server starts. The `load_ground_truth_doc()` reader is `lru_cache`-d and its cache is cleared after every regen.

Defense-in-depth ordering against hallucination:

1. **Doc** — `GROUND_TRUTH.md` in user prompt is the primary signal.
2. **System rules** — explicit ban on rounding/blending/inventing.
3. **Scrubbers** — runtime regex on amounts and invoice numbers.
4. **Retry** — invented-amount detection forces an LLM correction with the valid set.
5. **Fallback** — deterministic regex-tree reply from `sap_mock.json` (which mirrors the doc).

### 4.5 Summarizer

Now deterministic.

Responsibilities:
- derive headline
- infer disposition from tool/action history
- extract customer requests
- extract agreements and follow-ups
- produce a structured wrap-up object for UI/logging

Important design note:
- summary is derived mostly from structured actions and transcript evidence
- not from a generative narrative model

This is the right shape for later SAP / CRM write-back.

---

## 5. Tech Stack

### 5.1 Frontend
- React 18
- Vite
- TypeScript
- Native WebRTC APIs

### 5.2 Backend
- Python 3.11+
- Flask
- Flask-CORS
- requests
- OpenAI Python SDK only for session setup and realtime-related API usage

### 5.3 External APIs
- OpenAI Realtime API:
  - browser WebRTC session
  - voice output
  - input transcription events

No live helper-model API calls are part of the current architecture.

---

## 6. Project Layout

The current repo is still intentionally simple and mostly centered in:

```text
DHL_POC/
|-- KNOWLEDGE_BASE.md
|-- POC_TECHNICAL_PLAN.md
|-- backend/
|   |-- app.py
|   |-- data/
|   |   |-- sap_mock.json
|   |   |-- GROUND_TRUTH.md          (auto-generated; LLM source of truth)
|   |   |-- call_log.jsonl
|   |   |-- tool_actions.jsonl
|   |   |-- supervisor_flags.jsonl
|   |   |-- supervisor_board.json
|   |   `-- cost_ledger.json
|   |-- scripts/
|   |   `-- generate_ground_truth.py (regenerates GROUND_TRUTH.md from sap_mock.json)
|   `-- prompts/
|       |-- agent.md
|       |-- supervisor.md
|       |-- language_coach.md
|       `-- call_summary.md
`-- frontend/
    `-- src/
        |-- App.tsx
        |-- types.ts
        |-- lib/
        |   |-- api.ts
        |   `-- realtime.ts
        `-- components/
```

Implementation note:
- The prompts remain in the repo for historical/reference value and for possible later experimentation.
- The active live logic is now code-driven.

---

## 7. Backend API Surface

### 7.1 `POST /api/session`

Purpose:
- mint short-lived browser client secret for Realtime

Current behavior:
- returns session token
- sets a minimal renderer-style realtime instruction set
- keeps `create_response: false` so the backend controls when speech happens

### 7.2 `POST /api/turn/customer` (unified voice-turn endpoint)

Purpose:
- single per-turn endpoint used by voice mode; collapses what was previously two sequential roundtrips (`/api/language/detect` followed by `/api/chat/turn`) into one.

Behavior:
1. Run the deterministic language coach inline on the new customer transcript and produce fresh `language_advice`.
2. Pass that fresh advice straight into `run_chat_agent_turn` (the policy engine) without an extra HTTP hop.
3. Return language advice and the approved next utterance together.

Input:
- transcript text (latest customer turn)
- current / preferred language ids
- recent transcript window
- account id, voice, full message history, coaching hints

Output:
- `advice` (same shape as `/api/language/detect`)
- `assistant_text`
- `tool_calls`
- `costs`
- `model`

Why this exists:
- Removes the client-side waterfall that added one full HTTP roundtrip of latency to every customer turn.
- Removes the "first reply after a language switch is in the old language" failure mode, because the chat turn always sees the freshest advice.

### 7.3 `POST /api/chat/turn`

Purpose:
- generate the next approved collections utterance deterministically; used directly by **text mode** and by the language-repair path in voice mode.

Input:
- transcript history
- account id
- current language advice

Output:
- `assistant_text`
- `tool_calls`
- `costs`
- model label: `deterministic-call-engine`

In voice mode, the unified `/api/turn/customer` endpoint wraps this internally; the standalone endpoint remains for text mode and follow-up after backend tool execution.

### 7.4 `POST /api/language/detect`

Purpose:
- deterministic language routing (kept for text mode and tool-followup paths; no longer the primary voice-turn entry point).

Output:
- detected language
- suggested reply language
- transcript quality
- confidence
- nudge

### 7.5 `POST /api/supervisor/evaluate`

Purpose:
- deterministic QA pass over the latest agent turn

Output:
- structured issues
- updated board
- unchanged cost ledger unless a priced helper model is reintroduced later

### 7.6 `POST /api/call/summarize`

Purpose:
- deterministic post-call wrap-up

Output:
- summary object
- costs

### 7.7 `POST /api/tool/*`

Mock backend actions still exposed as discrete endpoints:
- `get_customer`
- `get_invoices`
- `log_promise_to_pay`
- `log_already_paid`
- `resend_invoice`
- `log_dispute`
- `update_contact`
- `transfer_to_human`

These are still useful because they mimic later SAP / CRM side effects and keep the UI transparent about what happened.

### 7.8 `POST /api/call/log`

Purpose:
- append call artifact to local JSONL log

Stored fields:
- account number
- transcript
- tool calls
- summary
- disposition

### 7.9 Cost endpoints
- `GET /api/metrics/costs`
- `POST /api/metrics/costs/reset`
- `POST /api/metrics/costs/event`

Cost design in the new architecture:
- realtime voice and transcription still accumulate estimated spend
- deterministic helper components show zero additional model cost

---

## 8. Data Model

### 8.1 Current ground truth

Two paired files:

- **`backend/data/sap_mock.json`** — raw fixture; the editable source. Contains:
  - payment methods
  - proof-of-payment email
  - collections constants (PTP window, monthly target day, dispositions)
  - customer master fixture
  - invoice fixture list with history
  - human-transfer escalation contact

- **`backend/data/GROUND_TRUTH.md`** — auto-generated from `sap_mock.json` by `backend/scripts/generate_ground_truth.py`. This is the **only** document the LLM policy engine consults; it explicitly enumerates allowed amounts, dates, names, and forbidden fabrication classes. Never edited by hand.

The pair guarantees the LLM cannot drift from the underlying fixture: any change to `sap_mock.json` triggers a regenerate at next server start, and the regenerator is also runnable on demand.

### 8.2 SAP migration direction

The current file-backed structure is already shaped to transition into SAP-backed reads.

Later source-of-truth fields will come from SAP or adjacent systems:
- account master
- invoice records
- dispute notes
- credit-note history
- registered email/contact details
- customer payment terms

Architecture rule for that transition:
- only the data source changes
- the deterministic conversation policy remains the same

---

## 9. Frontend Flow

### 9.1 Session startup
1. Browser calls `/api/session`
2. Browser opens WebRTC session to Realtime
3. Browser sends `session.update`
4. Browser sends scripted opening response

### 9.2 Customer turn handling (voice mode)
1. Customer speaks
2. Realtime emits transcription event
3. Frontend appends customer transcript
4. Frontend calls `/api/turn/customer` (single roundtrip)
5. Backend runs the deterministic language coach and the policy engine in one request and returns:
   - fresh `language_advice`
   - approved next utterance
   - any tool calls executed
   - costs
6. Frontend updates transcription hint / language state from the returned advice
7. Frontend renders tool calls in UI
8. Frontend sends the exact utterance to Realtime for speech (verbatim TTS)
9. After speech completes, frontend runs `detectSystemPromptLeak` on the spoken transcript; if a leak is detected, the turn is suppressed and the last approved line is re-issued
10. Frontend calls `/api/supervisor/evaluate`

Text mode keeps the older `/api/language/detect` + `/api/chat/turn` sequence (latency is not user-visible in chat).

### 9.3 End of call
1. Frontend calls `/api/call/summarize`
2. Frontend shows summary block
3. Frontend logs the call via `/api/call/log`

---

## 10. UI Behavior

The visible UI roles remain, but their implementation meaning has changed:

- `Live Caller`
  - still the voice agent
  - but now reads backend-approved text

- `Language Coach`
  - now deterministic
  - no per-turn model spend

- `Supervisor`
  - now deterministic
  - flags backend-checkable errors

- `Policy Engine`
  - replaces the old freeform chat/assistant reasoning role

- `Summariser`
  - now deterministic

The current UI still shows:
- transcript
- tool/action feed
- language state
- supervisor board
- summary
- cost callouts

---

## 11. Cost Strategy

### 11.1 What still costs money live
- realtime speech session
- realtime transcription
- one `gpt-4.1-mini` chat-completion per customer turn for the collections policy engine

### 11.2 What no longer adds live model-call cost
- language coach (deterministic)
- supervisor (deterministic)
- summarizer (deterministic)

### 11.3 Why this matters

This was a direct project requirement after review of the earlier build:
- keep the useful roles
- remove extra live API cost

The new architecture does exactly that.

---

## 12. Demo Expectations

The best demo path is now:
1. start call
2. agent greets customer
3. customer asks about invoices / issues / payment options
4. backend policy engine drives the DHL path
5. UI shows exact tool actions and summary

Suggested flows to rehearse:
- happy path with promise-to-pay
- already-paid
- invoice not received
- dispute
- wrong contact
- English switch after Hinglish opening

---

## 13. Risks and Known Limits

### 13.1 Current strengths
- strong grounding against backend invoice data
- stable DHL flow enforcement
- lower hallucination risk
- lower non-voice API cost
- cleaner SAP migration path

### 13.2 Current limits
- deterministic language rendering is strongest for English/Hinglish/Hindi/Bengali right now
- frontend still carries some historical naming from the older architecture
- summary quality is structured and operational, not highly narrative
- no telephony integration yet

### 13.3 Future hardening path
- replace `sap_mock.json` with SAP adapter layer
- add explicit conversation-state object persisted per call
- extend regional-language response template packs
- add replay/regression tests for DHL branch coverage
- add telephony/SIP layer

---

## 14. Next Implementation Phase

The next clean technical step after this plan is:
1. formalize call state machine objects
2. separate backend policy code from `app.py` into modules
3. add SAP adapter interface
4. add branch-level replay tests from `KNOWLEDGE_BASE.md`

That will make the current implementation easier to maintain as the DHL scope grows.

---

## 15. Source References

- `KNOWLEDGE_BASE.md`
- `backend/app.py`
- `backend/data/sap_mock.json`
- `backend/data/GROUND_TRUTH.md` (auto-generated)
- `backend/scripts/generate_ground_truth.py`
- `frontend/src/App.tsx`
- `frontend/src/lib/realtime.ts`
- OpenAI Realtime WebRTC docs: https://developers.openai.com/api/docs/guides/realtime-webrtc
- OpenAI Realtime server-side controls docs: https://developers.openai.com/api/docs/guides/realtime-server-controls
