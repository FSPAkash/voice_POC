# Render deployment

This repo is ready to deploy to Render as a single web service:

- Flask serves the API and the built Vite frontend from the same container.
- Render can build directly from the repo root using the included `Dockerfile`.
- The default health endpoint is `GET /health`.

## Recommended setup

1. Push this repo to GitHub.
2. In Render, choose `New +` -> `Blueprint`.
3. Select the repository and let Render pick up [render.yaml](./render.yaml).
4. When prompted, set `OPENAI_API_KEY`.
5. Deploy.

After deploy, the app will be available on the Render web service URL and the frontend will call the backend on the same origin, so no extra frontend API URL setup is required.

## Environment variables

The blueprint already defines safe defaults for the demo models and voice. You only need to provide:

- `OPENAI_API_KEY`

Optional variables you can override in Render later:

- `OPENAI_REALTIME_MODEL`
- `OPENAI_SUPERVISOR_MODEL`
- `OPENAI_LANGUAGE_COACH_MODEL`
- `OPENAI_CHAT_MODEL`
- `OPENAI_REALTIME_TRANSCRIPTION_MODEL`
- `OPENAI_REALTIME_VOICE`
- `DEMO_ACCOUNT_ID`

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
