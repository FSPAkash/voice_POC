"""Self-ping keep-alive for Render free tier. Pings /health every 14 min."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime

import requests

PING_PATH = "/health"


class KeepAliveService:
    def __init__(self, app_url: str | None = None, interval: int = 840) -> None:
        self.app_url = (app_url or os.environ.get("RENDER_EXTERNAL_URL") or "").rstrip("/")
        self.interval = interval
        self.running = False
        self.thread: threading.Thread | None = None

    def ping(self) -> bool:
        if not self.app_url:
            print("Keep-alive: No URL configured, skipping ping", flush=True)
            return False
        url = f"{self.app_url}{PING_PATH}"
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "dhl-poc-keepalive/1.0"},
            )
        except requests.exceptions.RequestException as exc:
            print(f"Keep-alive ping failed: {exc}", flush=True)
            return False
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        if response.status_code == 200:
            print(f"Keep-alive ping successful at {stamp} UTC", flush=True)
            return True
        print(f"Keep-alive ping returned status {response.status_code} at {stamp} UTC", flush=True)
        return False

    def _run(self) -> None:
        print(
            f"Keep-alive service started. Pinging {self.app_url}{PING_PATH} every {self.interval / 60} minutes",
            flush=True,
        )
        time.sleep(60)
        while self.running:
            self.ping()
            time.sleep(self.interval)

    def start(self) -> None:
        if self.running:
            print("Keep-alive service already running", flush=True)
            return
        if not self.app_url:
            print(
                "Keep-alive service not started: set RENDER_EXTERNAL_URL to enable",
                flush=True,
            )
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True, name="keep-alive")
        self.thread.start()
        print("Keep-alive service thread started", flush=True)

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        print("Keep-alive service stopped", flush=True)


_keep_alive_service: KeepAliveService | None = None


def init_keep_alive(app_url: str | None = None, interval: int = 840) -> KeepAliveService | None:
    global _keep_alive_service

    if not os.environ.get("RENDER"):
        print("Not running on Render, keep-alive service disabled", flush=True)
        return None

    if _keep_alive_service is None:
        _keep_alive_service = KeepAliveService(app_url=app_url, interval=interval)
        _keep_alive_service.start()
    return _keep_alive_service


def get_keep_alive_service() -> KeepAliveService | None:
    return _keep_alive_service
