# TTS Replacement Research — Replace OpenAI Realtime with Indian-Accent TTS

## Context

Current architecture uses **OpenAI `gpt-realtime-mini` as a voice renderer only**. The reasoning, grounding, language detection, and reply generation happen earlier in a deterministic + chat-completion pipeline (`backend/app.py` ~line 880, `REALTIME_RENDERER_INSTRUCTIONS`). The realtime model is instructed to:

> "Speak only the exact approved reply supplied by the application. Never invent facts, never call tools, never continue the conversation on your own."

So we pay realtime audio rates (`$32 / 1M audio output tokens` on `gpt-realtime`, ~half on mini) for what is functionally TTS. Plus voices are American-English-flavoured (`cedar`, `marin`, etc.) — wrong accent for DHL India collections.

Goal: swap the realtime renderer for an India-localised TTS provider with voice cloning. Keep STT and the reasoning pipeline as-is (or swap STT separately later).

---

## Shortlist (recent web sources, 2026)

### 1. Sarvam AI — Bulbul v3 *(recommended primary)*

- **Indian focus:** purpose-built for 11 Indian languages, 25+ voices, native Hinglish code-switching mid-sentence, India-routed inference.
- **Latency:** sub-200 ms time-to-first-audio (claimed); independent measurements ~600 ms on older v2 — verify on v3 before commit.
- **Voice cloning:** supported on Bulbul v3 (custom voice, natural prosody preserved).
- **Streaming:** WebSocket + HTTP stream. REST for <=2500 chars.
- **Pricing:** ~Rs.15 per 10,000 chars (~$0.018 / 1k chars). Cheapest of the shortlist for Indic.
- **Risk:** smaller company; English-only voice quality less polished than ElevenLabs.
- **Fit:** highest because the customer is DHL India and the agent must handle Hindi/Hinglish/Bengali. Already covers 4 of our `RENDERABLE_LANGUAGE_IDS`.

### 2. Smallest.ai — Lightning v3.1

- **Speed leader:** sub-100 ms TTFA, geo-routed (Hyderabad + Oregon). Beats GPT-4o-mini-TTS in 76.2% blind A/B (their benchmark, take with salt).
- **Indian languages:** Hindi, Tamil, Telugu, Malayalam, Kannada, Marathi, Gujarati — solid but Hinglish code-switch claims less explicit than Sarvam.
- **Voice cloning:** 5-15 s clip => instant clone; 45+ min => professional clone. One clip works across all 15 languages.
- **Streaming:** HTTP, SSE, WebSocket.
- **Pricing:** ~$0.025 / 1k chars (Pro $9/mo plan).
- **Risk:** Hinglish quality claims thinner than Sarvam in third-party reviews.
- **Fit:** strongest if **latency** is the dominant constraint over Indic-language nuance.

### 3. Cartesia — Sonic 3 / 3.5

- **Speed:** 75-90 ms TTFA over WebSocket. Tied with Smallest on latency.
- **Indian languages:** 9 Indian locales as of 2026, including hi-IN. 94 new voices added across 17 locales in 2026.
- **Voice cloning:** 15 s clip => exact-fidelity clone, preserves accent + speaking style.
- **Streaming:** native WebSocket; 2026 added OpenAI-WebSocket-compatible mode (drop-in for realtime clients).
- **Pricing:** not surfaced in search; historically usage-based, broadly comparable to ElevenLabs.
- **Risk:** Hindi quality "exceptional" per their own copy, but no independent Indic benchmark surfaced. Cartesia's strength is English emotion + laughter, not Indic prosody.
- **Fit:** strongest if we want the smoothest realtime-WebSocket migration.

### 4. ElevenLabs — Multilingual v2 + Professional Voice Clone

- **Quality leader** for English; Hindi/Indic improved through 2026 but still rated below Sarvam on Indic MOS.
- **Voice cloning:** Professional clone is gold-standard; instant clone also available.
- **Latency:** ~200 ms only at higher tiers; self-serve sits >300 ms — borderline for voice agents.
- **Pricing:** $103-$206 / 1M chars premium — most expensive shortlisted.
- **Risk:** accent — clones preserve source accent, so an Indian-accent clone needs Indian-accent source audio. Without that the voice drifts neutral/American.
- **Fit:** best **only** if you bring a high-quality Indian voice actor recording (45+ min) for a Professional clone.

### 5. AI4Bharat IndicTTS / Bhasini *(open-source fallback)*

- Self-host; covers ~13 Indian languages.
- No hosted SLA, no built-in streaming pipeline — engineering cost is high.
- Worth knowing exists; not recommended for the POC timeline.

---

## Recommendation

**Primary: Sarvam Bulbul v3.** Native Hinglish code-switching is the killer feature — our pipeline already routes between English / Hinglish / Hindi / Bengali, and Sarvam handles all four natively with one voice. Cheapest of the shortlist.

**Backup: Cartesia Sonic 3.5** if Sarvam latency in practice exceeds 300 ms or if we want a drop-in OpenAI-WebSocket-compatible swap with minimum client refactor.

**Voice cloning path:** Record 30-60 minutes of a native Hindi/Hinglish speaker (DHL agent persona, both genders if needed). Submit to Sarvam custom voice or Cartesia clone. Validate code-switch quality on a held-out Hinglish script before lock-in.

---

## Cost comparison (rough, per 1k characters output)

| Provider | $ / 1k chars | Indic quality | TTFA | Voice clone |
|---|---|---|---|---|
| OpenAI gpt-realtime (current) | ~$0.50+ (audio tokens, est) | weak Indian accent | ~300 ms | no |
| Sarvam Bulbul v3 | ~$0.018 | best | ~200 ms | yes |
| Smallest Lightning v3.1 | ~$0.025 | good | <100 ms | yes (5-15 s) |
| Cartesia Sonic 3.5 | mid | good | 75-90 ms | yes (15 s) |
| ElevenLabs Multilingual | ~$0.10-0.20 | medium for Indic | ~200 ms (paid tier) | yes (best) |

OpenAI cost line is order-of-magnitude — realtime audio output tokens are not directly per-character, but a typical 200-char Hinglish reply runs ~30x more expensive than Sarvam at the same length.

---

## Migration sketch

Affected paths in [backend/app.py](backend/app.py):

- [app.py:33-65](backend/app.py#L33) — `REALTIME_MODEL`, voice constants, persona map.
- [app.py:225-345](backend/app.py#L225) — `REALTIME_TOOLS` (already unused by renderer; safe to keep or drop).
- [app.py:880-884](backend/app.py#L880) — `REALTIME_RENDERER_INSTRUCTIONS` (obsolete once we stop using realtime as renderer).
- [app.py:3241-3303](backend/app.py#L3241) — `/api/session` returns OpenAI client_secret for WebRTC. Replace with provider-specific session bootstrap (Sarvam WebSocket URL + token, or Cartesia equivalent).
- Frontend WebRTC peer connection currently negotiates with OpenAI realtime — must move to WebSocket-stream PCM/Opus from new provider and play via WebAudio (or Cartesia's OpenAI-compatible mode keeps WebRTC).
- STT decision: keep `gpt-4o-mini-transcribe` (cheap, decent for Indic) OR switch to Sarvam Saarika ASR for India-tuned STT. Independent of TTS choice.

No change needed to deterministic engine, ground-truth grounding, or `compose_agent_instructions` — those produce the text the TTS speaks.

---

## Sources

- [Sarvam Bulbul v3 blog](https://www.sarvam.ai/blogs/bulbul-v3)
- [Sarvam TTS API docs](https://www.sarvam.ai/apis/text-to-speech)
- [Smallest.ai Lightning v3 launch](https://smallest.ai/blog/introducing-lightning-v3)
- [Smallest.ai pricing](https://smallest.ai/pricing)
- [Cartesia Sonic 3](https://cartesia.ai/sonic)
- [Cartesia 2026 changelog (Indic voices, OpenAI WS mode)](https://docs.cartesia.ai/changelog/2026)
- [Cartesia Hindi page](https://cartesia.ai/languages/hindi)
- [ElevenLabs Indian-accent voices](https://elevenlabs.io/text-to-speech/indian-accent)
- [ElevenLabs Hindi](https://elevenlabs.io/text-to-speech/hindi)
- [Caller.Digital open-source Indic voice AI 2026](https://www.caller.digital/blog/open-source-voice-ai-india-sarvam-ai4bharat-bhasini-2026)
- [Fastest TTS APIs 2026 comparison](https://smallest.ai/blog/top-fastest-text-to-speech-apis-in-2026)
- [Voice TTS pricing April 2026](https://www.buildmvpfast.com/api-costs/ai-voice)
- [AI4Bharat Indic-TTS GitHub](https://github.com/AI4Bharat/Indic-TTS)
