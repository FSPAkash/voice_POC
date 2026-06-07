# ElevenLabs + OpenAI Speech Engine Demo

This example shows the cleanest "Eleven for voice, OpenAI for brain" stack:

1. Browser audio goes to **ElevenLabs Speech Engine**
2. ElevenLabs handles STT, TTS, turn-taking, and WebRTC audio
3. Your Python server receives the transcript
4. **OpenAI** decides what to say and when to call tools
5. Local tools return grounded DHL-style invoice data
6. ElevenLabs speaks the final answer back to the user

This is a browser demo, not PSTN telephony. The same backend pattern can later be attached to phone calls through ElevenAgents telephony or SIP trunking.

## Files

- `create_engine.py`
  Creates a Speech Engine resource that points ElevenLabs at your local websocket server.
- `server.py`
  Runs the Speech Engine websocket handler on port `3001`, serves the browser UI on port `3002`, and exposes `/api/token`.
- `data/mock_account.json`
  Small grounded DHL-style fixture used by the tool layer.
- `static/index.html`
  Tiny browser UI using the ElevenLabs JavaScript client.

## Prereqs

- Python 3.11+
- An ElevenLabs API key
- An OpenAI API key
- `ngrok` or another public tunnel

## Setup

1. Create a virtualenv and install dependencies.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in `ELEVENLABS_API_KEY` and `OPENAI_API_KEY`.

3. Start a public tunnel for the Speech Engine websocket server.

```powershell
ngrok http 3001
```

4. Copy the public HTTPS URL into `PUBLIC_WS_BASE_URL` in `.env`.

5. Create the Speech Engine resource.

```powershell
python create_engine.py
```

6. Copy the printed `seng_...` ID into `ELEVENLABS_SPEECH_ENGINE_ID` in `.env`.

## Run

```powershell
python server.py
```

Then open:

- `http://localhost:3002`

You can speak with the agent or type a message into the text box.

## What to test

Try:

- `Why are you calling me?`
- `What invoices are pending?`
- `I can pay on 2026-06-09`
- `Please transfer me to a person`

The agent prompt forces tool usage before quoting invoice numbers or amounts, so this example demonstrates a real "agentic" pattern rather than raw TTS.

## How this maps to a phone stack

For a real phone implementation, keep the same backend logic and swap the client/transport:

- **Fastest hosted path:** Exotel or Twilio -> ElevenAgents -> custom LLM endpoint -> your tools
- **Most custom path:** telephony stack -> ElevenLabs voice layer -> your orchestration server -> OpenAI

Because ElevenLabs documents SIP compatibility with **Exotel**, this repo's existing telephony direction is compatible with an ElevenLabs path:

- https://elevenlabs.io/docs/eleven-agents/phone-numbers/sip-trunking

## Notes

- Speech Engine requires a publicly reachable websocket URL, which is why `ngrok` is used here.
- This example keeps business logic local on purpose. That matches the strongest public enterprise pattern from Cisco, Convin, Regal, and likely Revolut's custom orchestration path.
