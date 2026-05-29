# Sarvam Migration Plan — Replace OpenAI Realtime (STT + TTS) with Sarvam

## Scope locked

- **Full Sarvam:** Saarika ASR (STT) + Bulbul v3 TTS. OpenAI realtime ripped out.
- **Stock voices first:** built-in Sarvam speakers (`anushka`, `meera`, `abhilash`, `arvind`, etc.). Voice cloning deferred.
- **One PR.** App is broken until both backend and frontend land.

## Pre-flight

1. **Rotate Sarvam API key.** The key in chat is compromised. Generate a new one in Sarvam console.
2. Add to `backend/.env`:
   ```
   SARVAM_API_KEY=sk_xxx_new_key
   SARVAM_BASE_URL=https://api.sarvam.ai
   ```
3. `pip install sarvamai websockets` (use Sarvam Python SDK if available, else raw `httpx` + `websockets`).
4. Pick stock speakers per gender for `VOICE_PERSONAS`:
   - Female: `anushka`, `meera`, `pavithra`
   - Male: `abhilash`, `arvind`, `karun`
5. Confirm Sarvam endpoints from docs (may shift):
   - STT WS: `wss://api.sarvam.ai/speech-to-text-streaming/ws`
   - TTS WS: `wss://api.sarvam.ai/text-to-speech/ws`
   - TTS REST: `POST /text-to-speech`
6. Language code map:
   | App language_id | Sarvam code |
   |---|---|
   | `english` | `en-IN` |
   | `hinglish` | `hi-IN` |
   | `hindi` | `hi-IN` |
   | `bengali` | `bn-IN` |

---

## Backend changes ([backend/app.py](backend/app.py))

### Step 1 — Configuration

- Add at top of file:
  ```python
  SARVAM_API_KEY = os.environ.get("SARVAM_API_KEY")
  SARVAM_BASE_URL = os.environ.get("SARVAM_BASE_URL", "https://api.sarvam.ai")
  SARVAM_TTS_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
  SARVAM_STT_MODEL = os.environ.get("SARVAM_STT_MODEL", "saarika:v2.5")
  SARVAM_DEFAULT_FEMALE = os.environ.get("SARVAM_DEFAULT_FEMALE", "anushka")
  SARVAM_DEFAULT_MALE = os.environ.get("SARVAM_DEFAULT_MALE", "abhilash")
  ```
- Delete: `REALTIME_MODEL`, `REALTIME_TRANSCRIPTION_MODEL`, `DEFAULT_REALTIME_VOICE`, `SUPPORTED_REALTIME_MODELS` (lines 33-65).
- Replace `VOICE_PERSONAS` map (line 43) with Sarvam-speaker-keyed personas.
- Delete `REALTIME_TOOLS` (lines 225-345) — unused by renderer; chat-completion engine uses its own tool definitions.
- Delete `REALTIME_RENDERER_INSTRUCTIONS` (line 880).

### Step 2 — Pricing table

- In `PRICING` dict (~line 93), drop `gpt-realtime*` entries.
- Add Sarvam meter:
  ```python
  "sarvam-bulbul-v3": {
      "tts_per_million_chars": 18.0,  # ~Rs.15/10k chars → $0.018/1k → $18/1M
  },
  "sarvam-saarika-v2.5": {
      "stt_per_hour": 0.36,  # ~Rs.30/hr → $0.36
  },
  ```
- Update `realtime_cost_from_usage` (line 2026): rename to `tts_cost_from_chars(model, char_count)`. Drop audio-token math. Add `stt_cost_from_seconds(model, seconds)`.

### Step 3 — Replace `/api/session` (line 3241)

Old: returns OpenAI client_secret for WebRTC.
New: returns Sarvam config the frontend needs to open its own STT/TTS sockets.

```python
@app.post("/api/session")
def create_session():
    if not SARVAM_API_KEY:
        return error_json("SARVAM_API_KEY missing on backend.", 500)
    body = request.get_json(silent=True) or {}
    voice = str(body.get("voice") or SARVAM_DEFAULT_FEMALE)
    language_id = str(body.get("language_id") or DEFAULT_LANGUAGE_ID)
    return success_json({
        "session_id": uuid4().hex,
        "voice": voice,
        "language_id": language_id,
        "tts_ws_path": "/api/tts/stream",
        "stt_ws_path": "/api/stt/stream",
        "sample_rate": 16000,
        "tts_sample_rate": 24000,
    })
```

No client secret leaves the backend. Frontend talks only to **our** WebSocket; we proxy upstream to Sarvam with the API key server-side.

### Step 4 — New WebSocket `/api/tts/stream`

- Flask doesn't do WebSockets natively. Either:
  - Add `flask-sock` (lightweight, fits existing app), OR
  - Run a parallel `websockets`/`fastapi` process on another port.
- Recommend `flask-sock` to keep one process.
- Behaviour:
  1. Frontend opens WS, sends `{session_id, voice, language_id}` as first frame.
  2. Backend opens upstream WS to Sarvam with `Authorization: Bearer SARVAM_API_KEY`.
  3. Frontend sends `{type: "speak", text: "..."}` per utterance.
  4. Backend forwards to Sarvam, relays binary PCM frames back to frontend.
  5. Frontend can send `{type: "cancel"}` for barge-in — backend closes upstream and resets.
  6. On each utterance: tally chars, write a `usage_event` row for cost accounting.

### Step 5 — New WebSocket `/api/stt/stream`

- Same shape:
  1. Frontend sends `{session_id, language_id}` first frame.
  2. Backend opens upstream WS to Saarika.
  3. Frontend streams 16-kHz PCM mic chunks as binary frames.
  4. Backend forwards to Saarika, relays back transcripts as `{type: "partial"|"final", text}`.
  5. Track audio duration for STT cost accounting.

### Step 6 — Remove/rename realtime helpers

- `compose_agent_instructions` (line ~1893) — keep, but stop injecting realtime model param. It's used by the chat-completion engine that produces the approved reply.
- `build_transcription_config` — delete (was for OpenAI session creation).
- Snapshot payload at line ~3188: drop `realtime_model`, `supported_realtime_models`, `transcription_model`, `realtime_tools`, `realtime_voice`. Add `sarvam_voices: [{id, label, gender}]`, `sarvam_language_codes: {english:"en-IN", ...}`.

### Step 7 — Endpoints still in play

The deterministic engine and ground-truth grounding don't change. Confirm these still work end-to-end:
- `POST /api/render` (or wherever the chat-completion call lives ~line 2738) — produces approved reply text. Unchanged.
- `POST /api/tool/<tool_name>` — unchanged.
- `POST /api/turn` (language advice + supervisor) — unchanged.

---

## Frontend changes ([frontend/src/App.tsx](frontend/src/App.tsx), [frontend/src/lib/realtime.ts](frontend/src/lib/realtime.ts))

### Step 8 — Delete realtime primitives

In [App.tsx:1660-1748](frontend/src/App.tsx#L1660-L1748):
- Remove `RTCPeerConnection` setup, `getUserMedia`, SDP exchange, `oai-events` data channel.
- Remove `peerConnectionRef`, `dataChannelRef`.
- Remove the OpenAI SDP fetch to `api.openai.com/v1/realtime/calls`.

In [realtime.ts](frontend/src/lib/realtime.ts):
- Delete `buildSessionUpdate`, `buildScriptedResponse`, `buildGuidedResponse`, `buildLanguageRepairResponse`, `detectSystemPromptLeak`, `extractRealtimeText`, `extractRealtimeFunctionCalls`.
- Keep `buildOpeningText`, `detectLanguageComplianceIssue` — still useful (opening line text; language enforcement on the approved reply, not on a model output).
- Rename file to `voice.ts`.

### Step 9 — New `SarvamVoiceClient` class

`frontend/src/lib/sarvam.ts`:

```ts
export class SarvamVoiceClient {
  private sttWs: WebSocket | null = null
  private ttsWs: WebSocket | null = null
  private audioCtx: AudioContext | null = null
  private playbackQueue: AudioBuffer[] = []
  private currentSource: AudioBufferSourceNode | null = null
  private mediaStream: MediaStream | null = null
  private workletNode: AudioWorkletNode | null = null

  onPartialTranscript?: (text: string) => void
  onFinalTranscript?: (text: string) => void
  onPlaybackStart?: () => void
  onPlaybackEnd?: () => void

  async connect(session: SessionConfig): Promise<void> { /* open both WS */ }
  async startMic(): Promise<void> { /* getUserMedia + AudioWorklet → sttWs */ }
  speak(text: string, languageCode: string): void { /* ttsWs.send(...) */ }
  cancelSpeech(): void { /* ttsWs.send cancel + flush playback queue */ }
  disconnect(): void { /* close everything */ }
}
```

Two AudioWorklet processors needed:
- `mic-capture-worklet.js` — downsample 48k → 16k PCM, post `Int16Array` chunks back to main thread for WebSocket send.
- `tts-playback-worklet.js` — receive PCM frames from main, push to ring buffer, output to speakers.

### Step 10 — Wire into App.tsx

Replace the connection block at [App.tsx:1660](frontend/src/App.tsx#L1660):

```ts
const session = await createRealtimeSession({ voice, language_id })
const client = new SarvamVoiceClient()
sarvamClientRef.current = client

client.onFinalTranscript = (text) => {
  // existing customer-turn handler — pipe transcript to /api/turn etc.
  void handleCustomerTranscript(text)
}
client.onPlaybackEnd = () => {
  // mark agent turn done — existing logic
}

await client.connect(session)
await client.startMic()

// opening line
const openingLang = ...
const openingText = buildOpeningText(snapshot.customer, snapshot.agent_persona, openingLang)
client.speak(openingText, sarvamLangCode(selectedLanguageRef.current))
```

When the backend chat-completion engine returns an approved reply (existing flow), call `client.speak(reply, langCode)` instead of sending `response.create` over the data channel.

### Step 11 — Barge-in / interrupt

- Frontend VAD on mic input (use `@ricky0123/vad-web` or a simple energy threshold on the worklet side).
- On detected speech-start during agent playback: `client.cancelSpeech()` — closes TTS WS frame, flushes audio queue, stops `AudioBufferSourceNode`.

### Step 12 — UI bits

- Voice picker: swap dropdown options from OpenAI voices (`cedar`, `marin`) to Sarvam speakers. Pull from snapshot.sarvam_voices.
- Model picker: hide or repurpose (Sarvam has one TTS model, one STT model right now).
- Cost panel: rename "Realtime audio" line to "TTS chars" + "STT seconds". Update labels in the cost reducer that consumes `usage_event` rows.

---

## Testing checklist

1. **Smoke:** unit-test backend can open Sarvam WS, send 1 sentence, get audio back. Save to `.wav`, listen.
2. **STT:** record 10 s Hindi sample, send to `/api/stt/stream` from a `wscat` client, verify transcript.
3. **Cost meter:** synthetic 5-turn call → `usage_event` rows show correct chars + seconds, dollar math sane.
4. **End-to-end:** real browser call, English + Hinglish + Bengali turns, verify:
   - Mic captured at 16 kHz mono PCM.
   - Transcripts feed `/api/turn` correctly.
   - Approved reply spoken in correct language with Indian accent.
   - Barge-in cancels mid-utterance within ~200 ms.
   - Cost panel updates after each turn.
5. **Failure modes:**
   - Sarvam WS drop mid-call → frontend reconnects, doesn't lose mic.
   - Invalid character (emoji, math symbol) in approved reply → backend strips or escapes.
   - 401 on API key → backend returns clean error to frontend.

---

## Rollback

Keep the old realtime code path on a branch (`pre-sarvam`). If Sarvam quality/latency in production is unacceptable, revert the branch — no DB migrations or contract changes block rollback.

---

## File-by-file diff summary

| File | Change |
|---|---|
| [backend/app.py](backend/app.py) | Config rewrite, drop realtime constants, new TTS/STT WS routes, pricing table, `/api/session` replacement, snapshot payload update |
| `backend/requirements.txt` | Add `flask-sock`, `websockets`, optionally `sarvamai` SDK |
| `backend/.env` | Add `SARVAM_API_KEY`, `SARVAM_BASE_URL` |
| [frontend/src/lib/realtime.ts](frontend/src/lib/realtime.ts) | Rename → `voice.ts`. Delete realtime builders. Keep `buildOpeningText`, `detectLanguageComplianceIssue` |
| `frontend/src/lib/sarvam.ts` | **NEW** — `SarvamVoiceClient` class |
| `frontend/public/mic-capture-worklet.js` | **NEW** — 48k → 16k PCM downsampler |
| `frontend/public/tts-playback-worklet.js` | **NEW** — PCM ring buffer playback |
| [frontend/src/App.tsx](frontend/src/App.tsx) | Replace WebRTC setup with `SarvamVoiceClient`, swap voice picker, update cost labels |
| [frontend/src/types.ts](frontend/src/types.ts) | Drop `RealtimeTool`-related types unused after cleanup |

---

## Order of execution

1. Add Sarvam config + env reads.
2. Add pricing meter rewrite.
3. Add `flask-sock` and stub WS routes returning canned audio — verify wire format.
4. Wire real Sarvam upstream — verify single utterance round-trip from `curl`/`wscat`.
5. Replace `/api/session` and snapshot payload.
6. Build `SarvamVoiceClient` + two AudioWorklets in isolation (test page).
7. Swap App.tsx connection block.
8. Delete dead realtime code in `realtime.ts`.
9. End-to-end browser test.
10. Cost panel labels + cleanup.

Each step is a separately verifiable commit.

---

## Open questions before coding

1. Sarvam streaming WS exact protocol — confirm message shape from current docs (may have changed Q1 2026).
2. Sarvam pricing — confirm Rs.15/10k chars still current; some sources show enterprise tiers.
3. Saarika streaming latency on Indic languages — bench before committing UX expectations.
4. Whether barge-in needs frontend VAD or Sarvam STT has built-in endpointing we can reuse.
