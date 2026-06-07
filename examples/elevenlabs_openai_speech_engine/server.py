import asyncio
import json
import os
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, send_from_directory
from elevenlabs import AsyncElevenLabs, ElevenLabs
from openai import AsyncOpenAI


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DATA_FILE = BASE_DIR / "data" / "mock_account.json"
PROMISE_LOG_FILE = BASE_DIR / "promise_to_pay_log.jsonl"

SYSTEM_PROMPT = (
    "You are Yogesh, a DHL Express India payment follow-up caller in a live voice conversation. "
    "Sound like a real phone caller: short, warm, direct, and conversational. "
    "Never quote invoice numbers, amounts, due dates, or payment methods until you have called "
    "get_customer_context. If the customer gives a concrete payment date, call log_promise_to_pay. "
    "If the customer asks for a human or the issue is too complex, call transfer_to_human. "
    "Keep replies under three short sentences and ask at most one question at a time."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_context",
            "description": "Fetch the grounded customer and invoice context for this call before quoting facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Use the DHL account number. Default to DHL001 when unsure."
                    }
                },
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_promise_to_pay",
            "description": "Log a firm payment commitment when the customer gives a specific payment date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_id": {
                        "type": "string",
                        "description": "Account number for the payment promise."
                    },
                    "promise_date": {
                        "type": "string",
                        "description": "Payment date in YYYY-MM-DD format."
                    },
                    "notes": {
                        "type": "string",
                        "description": "Optional short note about the agreement."
                    }
                },
                "required": ["promise_date"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_to_human",
            "description": "Escalate the call to a human collections executive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short reason for the transfer."
                    }
                },
                "required": ["reason"],
                "additionalProperties": False
            },
        },
    },
]

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SPEECH_ENGINE_ID = os.getenv("ELEVENLABS_SPEECH_ENGINE_ID", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
ACCOUNT_ID = os.getenv("DHL_ACCOUNT_ID", "DHL001").strip() or "DHL001"

if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is missing")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is missing")
if not SPEECH_ENGINE_ID:
    raise RuntimeError("ELEVENLABS_SPEECH_ENGINE_ID is missing")

async_elevenlabs = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
sync_elevenlabs = ElevenLabs(api_key=ELEVENLABS_API_KEY)
openai = AsyncOpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


def load_fixture() -> dict[str, Any]:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_customer_context(account_id: str | None = None) -> dict[str, Any]:
    fixture = load_fixture()
    resolved_account_id = (account_id or fixture.get("default_account_id") or ACCOUNT_ID).strip()
    customer = deepcopy(fixture["customers"][resolved_account_id])
    invoices = deepcopy(fixture["invoices"][resolved_account_id])
    total_outstanding = sum(int(inv.get("amount", 0)) for inv in invoices)
    return {
        "ok": True,
        "account_id": resolved_account_id,
        "customer": customer,
        "invoices": invoices,
        "payment_methods": deepcopy(fixture.get("payment_methods", [])),
        "total_outstanding": total_outstanding,
    }


def log_promise_to_pay(account_id: str | None, promise_date: str, notes: str | None = None) -> dict[str, Any]:
    resolved_account_id = (account_id or ACCOUNT_ID).strip() or ACCOUNT_ID
    entry = {
        "timestamp": now_iso(),
        "account_id": resolved_account_id,
        "promise_date": promise_date,
        "notes": (notes or "").strip(),
    }
    with PROMISE_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")
    return {
        "ok": True,
        "message": f"Promise to pay recorded for {promise_date}.",
        "entry": entry,
    }


def transfer_to_human(reason: str) -> dict[str, Any]:
    fixture = load_fixture()
    customer = fixture["customers"][ACCOUNT_ID]
    transfer = deepcopy(customer["human_transfer"])
    return {
        "ok": True,
        "reason": reason,
        "transfer_contact": transfer,
        "client_message": (
            f"I will connect you to {transfer['name']}, {transfer['designation']}, "
            f"on {transfer['phone']}."
        ),
    }


TOOL_IMPLS = {
    "get_customer_context": lambda args: get_customer_context(args.get("account_id")),
    "log_promise_to_pay": lambda args: log_promise_to_pay(
        args.get("account_id"),
        str(args["promise_date"]),
        args.get("notes"),
    ),
    "transfer_to_human": lambda args: transfer_to_human(str(args["reason"])),
}


def transcript_to_messages(transcript: list[Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in transcript:
        role = "assistant" if getattr(item, "role", "user") == "agent" else "user"
        content = str(getattr(item, "content", "") or "").strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


async def complete_with_tools(transcript: list[Any]) -> str:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(transcript_to_messages(transcript))

    for _ in range(4):
        response = await openai.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0.2,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0].message

        tool_calls = choice.tool_calls or []
        if tool_calls:
            assistant_payload: dict[str, Any] = {"role": "assistant"}
            if choice.content:
                assistant_payload["content"] = choice.content
            assistant_payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]
            messages.append(assistant_payload)

            for tool_call in tool_calls:
                raw_arguments = tool_call.function.arguments or "{}"
                try:
                    args = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    args = {}
                impl = TOOL_IMPLS.get(tool_call.function.name)
                result = impl(args) if impl else {"ok": False, "error": "Unknown tool"}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=True),
                    }
                )
            continue

        text = (choice.content or "").strip()
        if text:
            return text

    return "I am sorry, I could not complete that just now. Let me connect you with the collections desk."


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/token")
def get_token():
    response = sync_elevenlabs.conversational_ai.conversations.get_webrtc_token(
        agent_id=SPEECH_ENGINE_ID,
    )
    return jsonify({"token": response.token})


@app.get("/api/promise-log")
def promise_log():
    if not PROMISE_LOG_FILE.exists():
        return jsonify({"entries": []})
    entries = [
        json.loads(line)
        for line in PROMISE_LOG_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return jsonify({"entries": entries[-20:]})


def on_init(conversation_id: str, session: Any) -> None:
    print(f"[init] conversation_id={conversation_id}")


async def on_transcript(transcript: list[Any], session: Any) -> None:
    reply = await complete_with_tools(transcript)
    print(f"[agent] {reply}")
    await session.send_response(reply)


def on_close(session: Any) -> None:
    print(f"[close] conversation_id={session.conversation_id}")


def on_error(err: Exception, session: Any) -> None:
    conversation_id = getattr(session, "conversation_id", "unknown")
    print(f"[error] conversation_id={conversation_id} error={err}")


def run_http_server() -> None:
    app.run(host="127.0.0.1", port=3002, debug=False, use_reloader=False)


async def main() -> None:
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    print("HTTP UI: http://localhost:3002")
    print("Speech Engine websocket: ws://localhost:3001/ws")
    print("Press Ctrl+C to stop.")

    engine = await async_elevenlabs.speech_engine.get(SPEECH_ENGINE_ID)
    await engine.serve(
        port=3001,
        path="/ws",
        debug=True,
        on_init=on_init,
        on_transcript=on_transcript,
        on_close=on_close,
        on_error=on_error,
    )


if __name__ == "__main__":
    asyncio.run(main())
