# Render deployment

This repo is ready to deploy to Render as a single web service:

- Flask serves the API and the built Vite frontend from the same container.
- Render can build directly from the repo root using the included `Dockerfile`.
- The default health endpoint is `GET /health`.

## Recommended setup

1. Push this repo to GitHub.
2. In Render, choose `New +` -> `Blueprint`.
3. Select the repository and let Render pick up [render.yaml](./render.yaml).
4. When prompted, set `OPENAI_API_KEY` and `SARVAM_API_KEY`.
5. Deploy.

After deploy, the app will be available on the Render web service URL and the frontend will call the backend on the same origin, so no extra frontend API URL setup is required.

## Environment variables

The blueprint now matches the live stack:

- OpenAI handles policy, supervisor, and language-coach turns.
- Sarvam handles TTS and STT for the live voice loop.

You must provide these secrets in Render:

- `OPENAI_API_KEY`
- `SARVAM_API_KEY`

The blueprint already provides sane defaults for the rest. The most important ones are:

- `OPENAI_SUPERVISOR_MODEL`
- `OPENAI_LANGUAGE_COACH_MODEL`
- `OPENAI_CHAT_MODEL`
- `POLICY_ENGINE_MODE`
- `SARVAM_BASE_URL`
- `SARVAM_TTS_MODEL`
- `SARVAM_STT_MODEL`
- `SARVAM_DEFAULT_MALE`
- `SARVAM_DEFAULT_FEMALE`
- `SARVAM_TTS_SAMPLE_RATE`
- `SARVAM_STT_SAMPLE_RATE`
- `SARVAM_TTS_PACE`
- `SARVAM_TTS_TEMPERATURE`
- `SARVAM_TTS_MIN_BUFFER_SIZE`
- `SARVAM_TTS_MAX_CHUNK_LENGTH`
- `SARVAM_TTS_OUTPUT_CODEC`
- `SARVAM_TTS_OUTPUT_BITRATE`
- `SARVAM_TTS_SEND_COMPLETION_EVENT`
- `SARVAM_STT_MODE`
- `SARVAM_TTS_DICT_ID`
- `SARVAM_INR_PER_USD`
- `SARVAM_BULBUL_V3_INR_PER_10K_CHARS`
- `SARVAM_BULBUL_V2_INR_PER_10K_CHARS`
- `SARVAM_STT_INR_PER_HOUR`
- `DEMO_ACCOUNT_ID`

### Important Sarvam migration note

If your old Render service was deployed before the Sarvam migration, it may still have stale env vars from the OpenAI realtime path such as:

- `OPENAI_REALTIME_MODEL`
- `OPENAI_REALTIME_TRANSCRIPTION_MODEL`
- `OPENAI_REALTIME_VOICE`

Those are no longer used by the backend. You can leave them in Render without breaking anything, but the clean setup is to remove them from the dashboard so the service config reflects the current speech stack.

### Pronunciation dictionary

If you created a Sarvam pronunciation dictionary for `DHL`, invoice numbers, `INR`, `NEFT`, or `MyBill`, set that ID in:

- `SARVAM_TTS_DICT_ID`

If you do not set it, deploy still works; you just lose the dictionary-based pronunciation boost.

### Before you push

1. Make sure `backend/.env` is not committed. Secrets should live only in Render env vars.
2. Commit [render.yaml](./render.yaml), [DEPLOY_RENDER.md](./DEPLOY_RENDER.md), and your Sarvam code changes.
3. Confirm the service in Render has both `OPENAI_API_KEY` and `SARVAM_API_KEY`.
4. If you are using a Sarvam dictionary locally, copy that dictionary ID into Render as `SARVAM_TTS_DICT_ID`.

### After redeploy

Smoke-test these before you call it done:

1. Open `/health` and make sure the service is healthy.
2. Start one live voice call and confirm the opening line speaks with the Sarvam voice.
3. Interrupt the agent once and confirm the call keeps listening.
4. Switch from Hindi to English and confirm the next turn actually changes language.
5. Open the pricing panel and confirm both `Sarvam Speech` and `GPT Policy` show non-zero usage after a real turn.

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
