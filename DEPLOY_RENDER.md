# Render deployment

This repo is ready to deploy to Render as a single web service:

- Flask serves the API and the built Vite frontend from the same container.
- Render can build directly from the repo root using the included `Dockerfile`.
- The default health endpoint is `GET /health`.

## Recommended setup

1. Push this repo to GitHub.
2. In Render, choose `New +` -> `Blueprint`.
3. Select the repository and let Render pick up [render.yaml](./render.yaml).
4. When prompted, set `OPENAI_API_KEY` and `ELEVENLABS_API_KEY`.
5. Deploy.

After deploy, the app will be available on the Render web service URL and the frontend will call the backend on the same origin, so no extra frontend API URL setup is required.

## Environment variables

The blueprint now matches the live stack:

- OpenAI handles policy, supervisor, language-coach turns, **and STT** (realtime `gpt-4o-mini-transcribe`).
- ElevenLabs handles **TTS** (Flash v2.5) for the live voice loop.

You must provide these secrets in Render:

- `OPENAI_API_KEY`
- `ELEVENLABS_API_KEY`
- `ELEVENLABS_DEFAULT_FEMALE` (an ElevenLabs voice id)
- `ELEVENLABS_DEFAULT_MALE` (an ElevenLabs voice id)
- `EXOTEL_ACCOUNT_SID`
- `EXOTEL_API_KEY`
- `EXOTEL_API_TOKEN`
- `EXOTEL_CALLER_ID`

The blueprint already provides sane defaults for the rest. The most important ones are:

- `OPENAI_SUPERVISOR_MODEL`
- `OPENAI_LANGUAGE_COACH_MODEL`
- `OPENAI_CHAT_MODEL`
- `POLICY_ENGINE_MODE`
- `ELEVENLABS_BASE_URL`
- `ELEVENLABS_TTS_MODEL`
- `ELEVENLABS_TTS_SAMPLE_RATE_BROWSER`
- `ELEVENLABS_TTS_SAMPLE_RATE_PHONE`
- `ELEVENLABS_VOICE_STABILITY` / `_SIMILARITY` / `_STYLE` / `_SPEED` / `_SPEAKER_BOOST`
- `ELEVENLABS_USD_PER_MILLION_CHARS`
- `TTS_HUMANIZE`
- `OPENAI_STT_MODEL`
- `OPENAI_STT_SAMPLE_RATE`
- `OPENAI_REALTIME_STT_URL`
- `OPENAI_STT_USD_PER_MINUTE`
- `EXOTEL_API_BASE_URL`
- `EXOTEL_STREAM_SAMPLE_RATE`
- `DEMO_ACCOUNT_ID`

### Voice ids

ElevenLabs needs a concrete `voice_id` per persona. Set `ELEVENLABS_DEFAULT_FEMALE`
and `ELEVENLABS_DEFAULT_MALE` to voice ids from your ElevenLabs voice library. You
can override any single persona with `ELEVENLABS_VOICE_<KEY>` (e.g.
`ELEVENLABS_VOICE_PRIYA`). Until real ids are set the app boots but TTS will fail
on a live call.

### Before you push

1. Make sure `backend/.env` is not committed. Secrets should live only in Render env vars.
2. Commit [render.yaml](./render.yaml), [DEPLOY_RENDER.md](./DEPLOY_RENDER.md), and your code changes.
3. Confirm the service in Render has `OPENAI_API_KEY` and `ELEVENLABS_API_KEY`.
4. Set `ELEVENLABS_DEFAULT_FEMALE` and `ELEVENLABS_DEFAULT_MALE` to real ElevenLabs voice ids.
5. Set `EXOTEL_ACCOUNT_SID`, `EXOTEL_API_KEY`, `EXOTEL_API_TOKEN`, and `EXOTEL_CALLER_ID` in Render.
6. Make sure `RENDER_EXTERNAL_URL` exactly matches the public service URL, because Exotel uses it for the outbound status callback and live `wss://` media bridge.

### After redeploy

Smoke-test these before you call it done:

1. Open `/health` and make sure the service is healthy.
2. Start one live voice call and confirm the opening line speaks with the ElevenLabs voice.
3. Interrupt the agent once and confirm the call keeps listening.
4. Switch from Hindi to English and confirm the next turn actually changes language.
5. Open the pricing panel and confirm both `Voice (TTS + STT)` and `GPT Policy` show non-zero usage after a real turn.

## Exotel phone demo

This repo now supports Exotel as a pure telephony layer for the existing app:

- Exotel places the PSTN call.
- Exotel streams call audio to `wss://<your-render-url>/api/exotel/media`.
- The backend keeps using the same ElevenLabs TTS + OpenAI STT + GPT collections flow.
- Post-call summary, tool calls, and cost logging still run in the backend.

### Render setup for Exotel

After your GitHub push redeploys, set these env vars in the Render service:

- `EXOTEL_ACCOUNT_SID`
- `EXOTEL_API_KEY`
- `EXOTEL_API_TOKEN`
- `EXOTEL_CALLER_ID`
- `EXOTEL_API_BASE_URL`
Default: `https://api.in.exotel.com`
- `EXOTEL_STREAM_SAMPLE_RATE`
Default: `8000`
- `RENDER_EXTERNAL_URL`
Example: `https://dhl-poc.onrender.com`

### Start a real phone call

Once the service is live, trigger the outbound demo call with:

```bash
curl -X POST https://<your-render-url>/api/exotel/calls/start ^
  -H "Content-Type: application/json" ^
  -d "{\"to_number\":\"+919136152622\",\"language_id\":\"hinglish\",\"voice\":\"shubh\"}"
```

You can inspect the active phone call state at:

- `GET /api/exotel/calls/active`

### What to expect

- The backend resets the cost ledger when a phone demo call starts, just like the browser demo does when you click `Start Call`.
- Only one Exotel phone demo call is allowed at a time, because the current dashboard cost ledger is still single-session.
- If the call never reaches the media bridge, check the Render logs first:
  - outbound `Calls/connect` response
  - hits to `/api/exotel/status`
  - hits to `/api/exotel/media`

## Keep-alive (free plan)

Free Render web service sleeps after ~15 min of inactivity. The backend ships an in-process self-pinger at [backend/keep_alive.py](./backend/keep_alive.py) that hits its own `/health` every 14 min from a daemon thread. No extra service, no external infra.

### How it works

- `init_keep_alive()` runs at module import in [backend/app.py](./backend/app.py), so it boots under Gunicorn (which imports the module, never `__main__`).
- Gated on `RENDER` env var (auto-set by Render). Disabled locally.
- Pings `${RENDER_EXTERNAL_URL}/health` every 840 s after a 60 s boot delay.
- Daemon thread — dies with the process. If Flask crashes, Render restarts and the thread restarts with it.

### Setup (one time)

1. Push to GitHub, deploy the blueprint as described above.
2. After first deploy, copy the public URL (e.g. `https://dhl-poc.onrender.com`).
3. In Render -> service -> `Environment`, set `RENDER_EXTERNAL_URL` to that URL (no trailing slash).
4. Trigger a redeploy. In logs you should see:
   ```
   Keep-alive service initialized
   Keep-alive service thread started
   Keep-alive service started. Pinging https://.../health every 14.0 minutes
   Keep-alive ping successful at YYYY-MM-DD HH:MM:SS UTC
   ```

### Notes

- Single Gunicorn worker (`--workers 1 --threads 8` in [Dockerfile](./Dockerfile)) — keeps one pinger thread, fits the 512 MB free-tier RAM.
- Tunables live in [backend/keep_alive.py](./backend/keep_alive.py): `interval=840`, initial `time.sleep(60)`, request `timeout=10`.

### Alternatives (not needed but documented)

- **External monitor**: UptimeRobot / Better Stack / cron-job.org hitting `/health` every 5-10 min.
- **Upgrade to Starter ($7/mo)**: no spindown, keep-alive unneeded.

## Notes

- The backend stores demo state in local files under `backend/data`. On Render, that filesystem is ephemeral, so resets can happen on redeploy or restart.
- For demo use, that is usually fine because the app already has reset endpoints and fixture data.
