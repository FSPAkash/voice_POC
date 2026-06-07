import asyncio
import os
from urllib.parse import urlparse

from dotenv import load_dotenv
from elevenlabs import AsyncElevenLabs


def build_ws_url(public_base_url: str) -> str:
    parsed = urlparse(public_base_url.strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("PUBLIC_WS_BASE_URL must be a full URL such as https://abc.ngrok-free.app")
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/ws"


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    public_base_url = os.getenv("PUBLIC_WS_BASE_URL", "").strip()

    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is missing")
    if not public_base_url:
        raise RuntimeError("PUBLIC_WS_BASE_URL is missing")

    ws_url = build_ws_url(public_base_url)
    client = AsyncElevenLabs(api_key=api_key)

    engine = await client.speech_engine.create(
        name="DHL POC Speech Engine Demo",
        speech_engine={"ws_url": ws_url},
    )

    print("Speech Engine created.")
    print(f"ws_url: {ws_url}")
    print(f"engine_id: {engine.engine_id}")
    print("Copy the engine_id into ELEVENLABS_SPEECH_ENGINE_ID in .env")


if __name__ == "__main__":
    asyncio.run(main())
