# Keep-Alive Service for Render.com Free Tier

Self-pinging background service. Stops Render free tier from sleeping after 15 min idle. Pings own `/api/health` endpoint every 14 min from inside the Flask process.

## Why It Works

- Render free web services sleep after 15 min no traffic. Cold start ~30-60s.
- Internal thread sends HTTP GET to public URL every 14 min. Counts as traffic. Server stays warm.
- Self-ping (not external cron) so no extra infra needed. Tradeoff: if app crashes, pinger dies with it (acceptable — Render restarts).

## Architecture

```
Flask app (Gunicorn worker)
  |
  +-- module-level init_keep_alive() at import time
        |
        +-- KeepAliveService instance (singleton via global)
              |
              +-- daemon thread
                    |
                    +-- sleep 60s (let app boot)
                    +-- loop: GET {RENDER_EXTERNAL_URL}/api/health, sleep 840s
```

Daemon thread = dies with process. Module-level init = runs under Gunicorn (which imports app, doesn't run `__main__`).

## Files

Two files. One env var. One health endpoint.

### 1. `backend/keep_alive.py` (new file)

Full source. Drop in as-is.

```python
"""
Keep-alive service to prevent Render.com free tier from sleeping.
Pings the server every 14 minutes to keep it awake.
"""

import threading
import time
import requests
from datetime import datetime
import os


class KeepAliveService:
    def __init__(self, app_url=None, interval=840):
        """
        Args:
            app_url: URL to ping (e.g., 'https://your-app.onrender.com')
            interval: Ping interval in seconds (default: 840 = 14 minutes)
        """
        self.app_url = app_url or os.environ.get('RENDER_EXTERNAL_URL')
        self.interval = interval
        self.running = False
        self.thread = None

    def ping(self):
        if not self.app_url:
            print("Keep-alive: No URL configured, skipping ping")
            return False
        try:
            url = f"{self.app_url}/api/health"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print(f"Keep-alive ping successful at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                return True
            else:
                print(f"Keep-alive ping returned status {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"Keep-alive ping failed: {str(e)}")
            return False

    def _run(self):
        print(f"Keep-alive service started. Pinging {self.app_url} every {self.interval/60} minutes")
        # Wait before first ping so app finishes booting
        time.sleep(60)
        while self.running:
            self.ping()
            time.sleep(self.interval)

    def start(self):
        if self.running:
            print("Keep-alive service already running")
            return
        if not self.app_url:
            print("Keep-alive service not started: No URL configured")
            print("Set RENDER_EXTERNAL_URL environment variable to enable keep-alive")
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print("Keep-alive service thread started")

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Keep-alive service stopped")


# Global singleton
_keep_alive_service = None


def init_keep_alive(app_url=None, interval=840):
    global _keep_alive_service

    # Only on Render. RENDER=true is set automatically by Render platform.
    if not os.environ.get('RENDER'):
        print("Not running on Render, keep-alive service disabled")
        return None

    if _keep_alive_service is None:
        _keep_alive_service = KeepAliveService(app_url=app_url, interval=interval)
        _keep_alive_service.start()

    return _keep_alive_service


def get_keep_alive_service():
    return _keep_alive_service
```

### 2. `backend/app.py` (Flask entrypoint) — 3 additions

**A. Import (top of file, with try/except so local dev without `requests` still boots):**

```python
try:
    from keep_alive import init_keep_alive
    KEEP_ALIVE_AVAILABLE = True
except ImportError as e:
    print(f"Keep-alive service not available: {e}")
    KEEP_ALIVE_AVAILABLE = False
```

**B. Init at MODULE LEVEL (not inside `if __name__ == '__main__':`). Critical — Gunicorn imports the module, never runs `__main__`. Place after `app = Flask(...)` and config:**

```python
# Initialize keep-alive service for Render.com (must be at module level for Gunicorn)
if KEEP_ALIVE_AVAILABLE:
    try:
        init_keep_alive()
        print("Keep-alive service initialized")
    except Exception as e:
        print(f"Failed to initialize keep-alive: {e}")
```

**C. Health endpoint the pinger hits:**

```python
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })
```

Endpoint must be public (no auth). Cheap. Returns 200.

### 3. `backend/requirements.txt`

Need `requests`. Already there for most Flask apps:

```
requests>=2.28.0
```

## Environment Variables

Set in Render dashboard → Service → Environment:

| Var | Value | Source |
|---|---|---|
| `RENDER` | `true` | **Auto-set by Render.** Do not set manually. Gate that disables service in local dev. |
| `RENDER_EXTERNAL_URL` | `https://your-app.onrender.com` | **Manual.** Render does NOT auto-populate this for all plans. Set to public service URL. |

Without `RENDER_EXTERNAL_URL` the service logs warning and skips. Without `RENDER` it disables entirely (so local dev not affected).

## Tunables

- `interval=840` (14 min). Render sleeps at 15 min. Stay under. Don't go below ~600s (wasted requests).
- `time.sleep(60)` initial delay. Lets app finish boot before first ping. Increase if app boot is slow.
- `timeout=10` on requests. Short — health endpoint should respond instantly.

## Gunicorn Multi-Worker Caveat

If running Gunicorn with `--workers N > 1`, EACH worker imports the module and starts its own pinger thread. N pings every 14 min instead of 1. Not harmful but wasteful.

Mitigations:
- Run with `--workers 1 --threads N` on free tier (free tier has 512MB RAM anyway, single worker fits).
- OR move init to `gunicorn.conf.py` `post_fork` and gate to `worker.age == 0`.
- OR use `--preload` and set the singleton before fork (pinger thread won't survive fork on POSIX, so this needs an `on_starting` hook instead).

For free tier single-worker setups, ignore — module-level init is simplest.

## Verifying It Works

1. Deploy. Check Render logs for:
   ```
   Keep-alive service initialized
   Keep-alive service thread started
   Keep-alive service started. Pinging https://... every 14.0 minutes
   ```
2. ~1 min after boot:
   ```
   Keep-alive ping successful at YYYY-MM-DD HH:MM:SS
   ```
3. Every 14 min after, another success line.
4. Hit `https://your-app.onrender.com/api/health` manually. Should return JSON `{"status": "healthy", ...}` with no auth.

## Failure Modes

| Symptom | Cause | Fix |
|---|---|---|
| `No URL configured` | `RENDER_EXTERNAL_URL` unset | Add env var in Render dashboard |
| `Not running on Render, ... disabled` | `RENDER` env var missing | Expected locally. On Render it's auto-set; if missing, set `RENDER=true`. |
| `ping returned status 401/403` | Health endpoint behind auth | Make `/api/health` public (no `@jwt_required`) |
| `ping returned status 404` | Wrong path | Confirm route registered at `/api/health` |
| App still sleeps | Worker count > 1 may not be the issue; check ping logs are actually firing every 14m | Verify thread alive; reduce interval to 600s |
| `Connection refused` immediately after deploy | First ping fired before boot done | Increase initial `time.sleep(60)` to 120 |

## Porting Checklist

For new project:

- [ ] Copy `keep_alive.py` to backend dir
- [ ] Add `requests` to requirements
- [ ] Add try/except import in app entrypoint
- [ ] Add `init_keep_alive()` call at module level (NOT inside `if __name__ == '__main__'`)
- [ ] Add public `/api/health` route returning 200 JSON
- [ ] In Render dashboard: set `RENDER_EXTERNAL_URL` to service URL
- [ ] Confirm `RENDER=true` present (auto, but verify)
- [ ] Deploy. Watch logs for `Keep-alive ping successful`.
- [ ] If Gunicorn workers > 1, decide: accept duplicate pings or move init to gunicorn hooks.

## Non-Flask Adaptation

Logic is framework-agnostic. For FastAPI/Django/etc:
- Keep `keep_alive.py` unchanged.
- Call `init_keep_alive()` at module level wherever the ASGI/WSGI app object is created.
- Expose any public GET endpoint returning 200 — change `/api/health` in `ping()` if route differs.
