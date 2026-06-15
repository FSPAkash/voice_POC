# Conversational Voice Collections Agent — Solution Architecture

**Findability Sciences** &nbsp;|&nbsp; Prepared for the client

---

## Overview

An end-to-end voice agent that takes a live caller, understands what they say, decides
the next response against verified account data and business policy, then speaks it back —
on the web or over the phone.

The decision-making is powered entirely by **Findability Sciences Engines**. The
speech-to-text and text-to-speech layers only convert audio to and from text; they never
decide what the agent says.

---

## Flow at a glance

```
   Caller channels            Voice layer            Findability Sciences Engines           Voice layer        Caller channels
 ┌────────────────┐                                  (the decision brain)
 │  Web caller    │──audio──┐                ┌───────────────────────────────────┐
 └────────────────┘         │                │  1. Language Coach Engine         │
                            ▼                │  2. Policy Engine  ◄── facts ──┐  │
                     ┌────────────┐  turn    │  3. Supervisor Engine          │  │
                     │ STT Layer  │──text──► ┌┴────────────┐   ┌──────────────┴┐ │
                     │ speech→text│          │ Turn        │   │ Verified       │ │
 ┌────────────────┐  └────────────┘          │ assembly    │   │ account data   │ │
 │  Phone caller  │──audio──┘                └─────────────┘   └────────────────┘ │
 └────────────────┘                          │        One approved reply          │
        ▲                                     └──────────────────┬────────────────┘
        │                                                        │ approved reply only
        │                ┌────────────┐                          ▼
        └───audio out────│ TTS Layer  │◄─────────────────────────┘
                         │ text→speech│
                         └────────────┘
```

---

## Components

### Caller channels (audio in / out)

| Channel | What it is | Notes |
|---|---|---|
| **Web caller** | Browser voice session | Microphone capture with live barge-in support. Streams caller audio in. |
| **Phone caller** | Outbound call placed through a telephony gateway | Two-way audio stream. Customer speaks on a phone. |

### Voice layers

| Layer | Role | Important boundary |
|---|---|---|
| **Speech-to-Text (STT) Layer** | Converts both channels of caller audio into text. Returns live partial text and final transcripts. | **Speech in only — it does not decide the reply.** |
| **Turn assembly** (bridge) | Merges live speech fragments into one clean customer turn. Packages it with recent history, account context, and the active language, then hands it to the brain. | Plumbing between the STT Layer and the brain. |
| **Text-to-Speech (TTS) Layer** | Speaks numbers, dates, and names naturally, then renders the approved reply as audio back to the caller. | **Renders only — never chooses content.** |

### Findability Sciences Engines — the decision brain

This is where every decision is made. Three engines work together over a verified
knowledge base.

| Engine | Responsibility |
|---|---|
| **FS Engine 1 — Language Coach Engine** | Detects the caller's language and decides the reply language, so the agent mirrors how the customer speaks. |
| **FS Engine 2 — Policy Engine** | The core decision-maker. Chooses the next response from verified account facts and collections business rules, and triggers actions (promise to pay, callback, dispute, hand-off). |
| **FS Engine 3 — Supervisor Engine** | Reviews every turn for compliance and tone, and flags issues for the team. Guardrails ensure the agent only ever states verified, approved facts. |
| **Verified account data** (knowledge base) | Live customer, invoices, balances, payment methods, and notes — the only facts the agent is allowed to speak. |

**Output of the brain:** exactly one verified response string is released to be spoken.

---

## End-to-end sequence

1. A caller speaks on the **web** or on the **phone**; their audio streams in.
2. The **STT Layer** converts that audio into text (partial, then final).
3. **Turn assembly** merges the fragments into one customer turn and adds account
   context and the active language.
4. The **Findability Sciences Engines** decide the response:
   - the **Language Coach Engine** sets the reply language,
   - the **Policy Engine** chooses the next line from **verified account data** and
     business rules, and triggers any action,
   - the **Supervisor Engine** reviews the turn for compliance and tone.
5. The brain releases **one approved reply**.
6. The **TTS Layer** speaks only that approved reply back to the caller — in the browser
   or over the phone call.

---

## Key principles

- **The brain decides; the voice layers only translate.** The STT Layer turns speech into
  text and the TTS Layer turns approved text into speech. Neither one ever decides what
  the agent says.
- **Verified account data is the single source of truth.** The agent can only state facts
  that exist in the verified knowledge base.
- **Every turn is supervised.** The Supervisor Engine checks compliance and tone on each
  response before and as it is delivered.
- **One channel-agnostic decision path.** Web and phone callers flow through the same
  Findability Sciences Engines; only the audio transport differs.

---

*Summary: the STT Layer turns speech into text. The Findability Sciences Engines decide the
response from verified account data and business policy, with a Supervisor reviewing every
turn. The TTS Layer speaks only the already-approved reply back to the caller.*
