from __future__ import annotations

import json
import math
import os
import re
import uuid
from collections import deque
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlencode
from xml.etree import ElementTree
from zoneinfo import ZoneInfo

import base64
import struct
import threading
import time

import requests
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_sock import Sock
from openai import OpenAI

try:
    import websocket as ws_client  # websocket-client package; upstream Sarvam WS
    WEBSOCKET_CLIENT_AVAILABLE = True
except ImportError:
    WEBSOCKET_CLIENT_AVAILABLE = False
    ws_client = None  # type: ignore

try:
    from keep_alive import init_keep_alive
    KEEP_ALIVE_AVAILABLE = True
except ImportError as _keep_alive_import_error:
    print(f"Keep-alive service not available: {_keep_alive_import_error}", flush=True)
    KEEP_ALIVE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
# Create the data dir at import so the first call's wrap-up write cannot fail on
# a fresh deploy where backend/data/ does not exist yet.
try:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
SUPERVISOR_MODEL = os.environ.get("OPENAI_SUPERVISOR_MODEL", "gpt-4.1-mini")
LANGUAGE_COACH_MODEL = os.environ.get("OPENAI_LANGUAGE_COACH_MODEL", "gpt-4.1-mini")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1")
POLICY_ENGINE_MODE = os.environ.get("POLICY_ENGINE_MODE", "llm").strip().lower()


def _chat_kwargs(model: str, temperature: float) -> dict[str, Any]:
    """gpt-5 family rejects custom temperature (only default=1 allowed).
    Strip the param for those models, keep it for everything else."""
    if model.lower().startswith("gpt-5"):
        return {}
    return {"temperature": temperature}


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    raw = os.environ.get(name)
    try:
        value = float(raw) if raw is not None else default
    except (TypeError, ValueError):
        value = default
    if minimum is not None:
        value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _normalize_https_base_url(raw: str | None, default: str) -> str:
    value = (raw or default or "").strip()
    if not value:
        value = default
    if not re.match(r"^[a-z][a-z0-9+\-.]*://", value, re.IGNORECASE):
        value = f"https://{value.lstrip('/')}"
    return value.rstrip("/")


def _strip_matching_quotes(value: str | None) -> str:
    text = str(value or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text

# Voice layer — ElevenLabs (TTS) + Sarvam Saarika (STT). Sarvam STT is purpose-
# built for Indian languages (Marathi/Hindi/Bengali/Tamil/...) and outperforms
# OpenAI transcribe on Indic, so STT stays on Sarvam. The "brain" (policy engine,
# ground-truth grounding, chat-completion reply, supervisor, language coach) is
# unrelated to the voice vendor and stays untouched.
ELEVENLABS_API_KEY = _strip_matching_quotes(os.environ.get("ELEVENLABS_API_KEY", ""))
ELEVENLABS_BASE_URL = _normalize_https_base_url(
    _strip_matching_quotes(os.environ.get("ELEVENLABS_BASE_URL")),
    "https://api.elevenlabs.io",
)
ELEVENLABS_TTS_MODEL = (os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_v3") or "eleven_v3").strip()
# language_code values the model accepts. eleven_v3 natively supports all our
# Indic languages; older models only en/hi/ta. For codes NOT listed here we omit
# language_code and let the input script drive synthesis instead of 400ing.
ELEVENLABS_SUPPORTED_LANGUAGE_CODES = frozenset(
    code.strip().lower()
    for code in (
        os.environ.get(
            "ELEVENLABS_SUPPORTED_LANGUAGE_CODES",
            "en,hi,ta,mr,bn,te,gu,kn,ml,pa,or",
        )
        or "en,hi,ta,mr,bn,te,gu,kn,ml,pa,or"
    ).split(",")
    if code.strip()
)
# Per-persona voice ids. Placeholder defaults so the app boots; drop real
# ElevenLabs voice ids into backend/.env before a live call.
ELEVENLABS_DEFAULT_FEMALE = (os.environ.get("ELEVENLABS_DEFAULT_FEMALE", "21m00Tcm4TlvDq8ikWAM") or "").strip()
ELEVENLABS_DEFAULT_MALE = (os.environ.get("ELEVENLABS_DEFAULT_MALE", "TxGEqnHWrfWFTfGW9XjX") or "").strip()
# Map the app's persona voice keys (VOICE_PERSONAS) -> ElevenLabs voice ids.
# Each persona is individually env-overridable: ELEVENLABS_VOICE_<KEY>.
ELEVENLABS_TTS_SAMPLE_RATE_BROWSER = _env_int("ELEVENLABS_TTS_SAMPLE_RATE_BROWSER", 24000, minimum=8000, maximum=48000)
ELEVENLABS_TTS_SAMPLE_RATE_PHONE = _env_int("ELEVENLABS_TTS_SAMPLE_RATE_PHONE", 8000, minimum=8000, maximum=24000)
# ElevenLabs voice_settings overrides (Flash v2.5). Stability/similarity/style in
# 0..1, speed 0.7..1.2. Conservative human-leaning defaults.
ELEVENLABS_VOICE_STABILITY = _env_float("ELEVENLABS_VOICE_STABILITY", 0.5, minimum=0.0, maximum=1.0)
ELEVENLABS_VOICE_SIMILARITY = _env_float("ELEVENLABS_VOICE_SIMILARITY", 0.75, minimum=0.0, maximum=1.0)
ELEVENLABS_VOICE_STYLE = _env_float("ELEVENLABS_VOICE_STYLE", 0.0, minimum=0.0, maximum=1.0)
ELEVENLABS_VOICE_SPEED = _env_float("ELEVENLABS_VOICE_SPEED", 1.15, minimum=0.7, maximum=1.2)
ELEVENLABS_VOICE_SPEAKER_BOOST = _env_flag("ELEVENLABS_VOICE_SPEAKER_BOOST", True)
# Text normalization: "on" spells out numbers/dates in the target language (Hindi
# "3" -> "teen"); "auto" lets ElevenLabs decide; "off" disables. We pre-convert
# IDs/phones/amounts to English ourselves, so "on" handles the rest in-language.
ELEVENLABS_TEXT_NORMALIZATION = (os.environ.get("ELEVENLABS_TEXT_NORMALIZATION", "on") or "on").strip().lower()
if ELEVENLABS_TEXT_NORMALIZATION not in {"on", "auto", "off"}:
    ELEVENLABS_TEXT_NORMALIZATION = "on"
# Pronunciation dictionary (alias rules) so brand/payment terms + names are said
# consistently: DHL -> "D H L", NEFT/RTGS/UPI spelled, MyBill -> "My Bill", etc.
# Created via /v1/pronunciation-dictionaries/add-from-rules. Set both id+version.
ELEVENLABS_PRON_DICT_ID = (os.environ.get("ELEVENLABS_PRON_DICT_ID", "F6QAB5Afyd7NaPO4vXlb") or "").strip()
ELEVENLABS_PRON_DICT_VERSION_ID = (os.environ.get("ELEVENLABS_PRON_DICT_VERSION_ID", "sfnJqQWMZwZGEWGBbGBd") or "").strip()
# eleven_v3 audio tags steer delivery tone. For a collections agent we default to
# a calm, professional collector tone, and adapt per-turn from the reply's intent
# (empathetic for hardship, firm for delay/refusal). Set ELEVENLABS_AUDIO_TAGS=0
# to disable entirely (some voices over-act on tags).
ELEVENLABS_AUDIO_TAGS = _env_flag("ELEVENLABS_AUDIO_TAGS", True)
ELEVENLABS_TONE_DEFAULT = (os.environ.get("ELEVENLABS_TONE_DEFAULT", "professional") or "professional").strip()
ELEVENLABS_TONE_EMPATHETIC = (os.environ.get("ELEVENLABS_TONE_EMPATHETIC", "empathetic") or "empathetic").strip()
ELEVENLABS_TONE_FIRM = (os.environ.get("ELEVENLABS_TONE_FIRM", "firm") or "firm").strip()
# eleven_v3 only — older models would read the tag literally.
_AUDIO_TAG_MODELS = {"eleven_v3"}
# Models that REJECT previous_text/next_text (400 unsupported_model). eleven_v3
# does not support continuity context yet.
_NO_CONTINUITY_MODELS = {"eleven_v3"}
# Humanize spoken text (spell dates, group long numbers, light clause pauses) so
# the voice does not read IDs/dates mechanically. Vendor-neutral text prep.
TTS_HUMANIZE = _env_flag("TTS_HUMANIZE", True)

# Sarvam Saarika (STT) — purpose-built for Indian languages. Streaming WS proxy.
SARVAM_API_KEY = _strip_matching_quotes(os.environ.get("SARVAM_API_KEY", ""))
SARVAM_BASE_URL = _normalize_https_base_url(
    _strip_matching_quotes(os.environ.get("SARVAM_BASE_URL")),
    "https://api.sarvam.ai",
)
SARVAM_STT_MODEL = (os.environ.get("SARVAM_STT_MODEL", "saaras:v3") or "saaras:v3").strip()
SARVAM_STT_WS_URL = (
    os.environ.get("SARVAM_STT_WS_URL", "wss://api.sarvam.ai/speech-to-text/ws")
    or "wss://api.sarvam.ai/speech-to-text/ws"
).strip()
SARVAM_STT_SAMPLE_RATE = _env_int("SARVAM_STT_SAMPLE_RATE", 16000, minimum=8000, maximum=16000)
SARVAM_STT_MODE = (os.environ.get("SARVAM_STT_MODE", "codemix") or "codemix").strip().lower()
if SARVAM_STT_MODE not in {"transcribe", "translate", "verbatim", "translit", "codemix"}:
    SARVAM_STT_MODE = "codemix"

DEFAULT_REALTIME_VOICE = "ratan"  # persona key; resolved to a voice id via elevenlabs_voice_id()
# Aliases so legacy code paths that reference REALTIME_MODEL / REALTIME_TRANSCRIPTION_MODEL
# (cost ledger labels, snapshot payload) keep compiling.
REALTIME_MODEL = ELEVENLABS_TTS_MODEL
REALTIME_TRANSCRIPTION_MODEL = SARVAM_STT_MODEL
# Back-compat model-id alias.
SARVAM_TTS_MODEL = ELEVENLABS_TTS_MODEL

EXOTEL_ACCOUNT_SID = _strip_matching_quotes(os.environ.get("EXOTEL_ACCOUNT_SID", ""))
EXOTEL_API_KEY = _strip_matching_quotes(os.environ.get("EXOTEL_API_KEY", ""))
EXOTEL_API_TOKEN = _strip_matching_quotes(os.environ.get("EXOTEL_API_TOKEN", ""))
EXOTEL_API_BASE_URL = _normalize_https_base_url(
    _strip_matching_quotes(os.environ.get("EXOTEL_API_BASE_URL")),
    "https://api.in.exotel.com",
)
EXOTEL_CALLER_ID = _strip_matching_quotes(os.environ.get("EXOTEL_CALLER_ID", ""))
EXOTEL_STREAM_SAMPLE_RATE = 8000
EXOTEL_STREAM_PATH = (os.environ.get("EXOTEL_STREAM_PATH", "/api/exotel/media") or "/api/exotel/media").strip() or "/api/exotel/media"
PHONE_STT_MIN_SPEECH_SECONDS = _env_float("PHONE_STT_MIN_SPEECH_SECONDS", 0.15, minimum=0.05, maximum=1.0)
PHONE_STT_SILENCE_FLUSH_SECONDS = _env_float("PHONE_STT_SILENCE_FLUSH_SECONDS", 0.25, minimum=0.05, maximum=1.0)
PHONE_TURN_COMMIT_DELAY_SECONDS = _env_float("PHONE_TURN_COMMIT_DELAY_SECONDS", 0.1, minimum=0.0, maximum=1.0)
PHONE_SHORT_FRAGMENT_COMMIT_DELAY_SECONDS = _env_float("PHONE_SHORT_FRAGMENT_COMMIT_DELAY_SECONDS", 0.38, minimum=0.0, maximum=1.0)
PHONE_AMBIENCE_ENABLED = _env_flag("PHONE_AMBIENCE_ENABLED", True)
PHONE_AMBIENCE_IDLE_GAIN = _env_float("PHONE_AMBIENCE_IDLE_GAIN", 0.9, minimum=0.0, maximum=4.0)
PHONE_AMBIENCE_TTS_GAIN = _env_float("PHONE_AMBIENCE_TTS_GAIN", 0.15, minimum=0.0, maximum=4.0)
# Subtle bed: ~900 RMS idle is faint room tone, not a busy call-center wall.
# Measured 6836 RMS in the field was far too hot (target 2400 x idle 2.2).
PHONE_AMBIENCE_TARGET_RMS = _env_float("PHONE_AMBIENCE_TARGET_RMS", 900.0, minimum=200.0, maximum=6000.0)
PHONE_AMBIENCE_MAX_NORMALIZE_GAIN = _env_float("PHONE_AMBIENCE_MAX_NORMALIZE_GAIN", 4.0, minimum=1.0, maximum=12.0)
def _resolve_phone_ambience_file() -> Path:
    """Locate the call-center ambience WAV. Prefer an explicit override, then a
    backend-local copy (ships with the backend deploy), then the frontend asset
    (present only in a co-located dev checkout). The frontend path alone is NOT
    safe in production: a backend-only deploy has no frontend/ dir, so the file
    silently goes missing and the phone gets dead air instead of ambience."""
    override = (os.environ.get("PHONE_AMBIENCE_FILE", "") or "").strip()
    candidates = [Path(override)] if override else []
    candidates += [
        BASE_DIR / "assets" / "call_center_background.wav",
        BASE_DIR / "data" / "call_center_background.wav",
        BASE_DIR.parent / "frontend" / "public" / "sound" / "call_center_background.wav",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    # Return the backend-local path even if absent so logs point somewhere sane.
    return candidates[0] if override else (BASE_DIR / "assets" / "call_center_background.wav")


PHONE_AMBIENCE_FILE = _resolve_phone_ambience_file()
PHONE_GREETING_BARGE_IN_GRACE_SECONDS = _env_float("PHONE_GREETING_BARGE_IN_GRACE_SECONDS", 1.6, minimum=0.0, maximum=5.0)
PHONE_LANGUAGE_SWITCH_CONFIRM_WINDOW_SECONDS = _env_float(
    "PHONE_LANGUAGE_SWITCH_CONFIRM_WINDOW_SECONDS",
    8.0,
    minimum=1.0,
    maximum=20.0,
)
# Stickiness gate: a language SWITCH that rests on weak evidence (Sarvam STT
# confidence below this, or a switch with no script signal) must be corroborated
# by a second consistent turn before we flip the call language. Prevents a single
# misrecognized utterance from hijacking the whole call (the Tamil-misfire bug).
# This is pure local logic on results already received — it adds NO latency, it
# can only defer a low-confidence switch by one turn. Set to 0 to disable.
PHONE_LANGUAGE_SWITCH_MIN_CONFIDENCE = _env_float(
    "PHONE_LANGUAGE_SWITCH_MIN_CONFIDENCE",
    0.55,
    minimum=0.0,
    maximum=1.0,
)
# Switch reasons strong enough to bypass the corroboration gate even at low/absent
# confidence: clear native-script evidence, or the customer literally naming a
# language. These were never the misfire source.
PHONE_TRUSTED_SWITCH_REASONS = frozenset(
    {"strong_script_switch", "explicit_request", "plain_english"}
)

# Voice -> agent persona. Sarvam picks the voice; the agent prompt must use a
# matching name and pronouns so the customer never hears a male name on a female voice.
VOICE_PERSONAS: dict[str, dict[str, str]] = {
    "priya": {"name": "Priya", "gender": "female", "pronouns": "she/her"},
    "ishita": {"name": "Ishita", "gender": "female", "pronouns": "she/her"},
    "ritu": {"name": "Ritu", "gender": "female", "pronouns": "she/her"},
    "simran": {"name": "Simran", "gender": "female", "pronouns": "she/her"},
    "aditya": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "ashutosh": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "anand": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "shubh": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "ratan": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "mani": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    # Backward-compatible v2 voices still accepted if a local .env or stale
    # client sends them.
    "anushka": {"name": "Priya", "gender": "female", "pronouns": "she/her"},
    "abhilash": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "manisha": {"name": "Manisha", "gender": "female", "pronouns": "she/her"},
    "vidya": {"name": "Vidya", "gender": "female", "pronouns": "she/her"},
    "arya": {"name": "Arya", "gender": "female", "pronouns": "she/her"},
    "karun": {"name": "Karun", "gender": "male", "pronouns": "he/him"},
}
DEFAULT_PERSONA = {"name": "Yogesh", "gender": "male", "pronouns": "he/him"}

# Public catalogue for the frontend voice picker. Persona ids map to ElevenLabs
# voice ids via ELEVENLABS_VOICE_IDS below.
TTS_VOICES = [
    {"id": "priya", "label": "Priya (female, recommended)", "gender": "female"},
    {"id": "ishita", "label": "Ishita (female)", "gender": "female"},
    {"id": "ritu", "label": "Ritu (female)", "gender": "female"},
    {"id": "simran", "label": "Simran (female)", "gender": "female"},
    {"id": "anand", "label": "Anand (male, professional Hindi collections)", "gender": "male"},
    {"id": "aditya", "label": "Aditya (male, professional English collections)", "gender": "male"},
    {"id": "ashutosh", "label": "Ashutosh (male, Hindi)", "gender": "male"},
    {"id": "shubh", "label": "Shubh (male, Hindi)", "gender": "male"},
    {"id": "ratan", "label": "Ratan (male, authoritative Marathi/English)", "gender": "male"},
    {"id": "mani", "label": "Mani (male, broad coverage)", "gender": "male"},
]


def _resolve_elevenlabs_voice_ids() -> dict[str, str]:
    """Persona key -> ElevenLabs voice id. Default by gender to the
    ELEVENLABS_DEFAULT_FEMALE/MALE ids; allow a per-persona override via
    ELEVENLABS_VOICE_<KEY> (e.g. ELEVENLABS_VOICE_PRIYA=...)."""
    ids: dict[str, str] = {}
    for key, persona in VOICE_PERSONAS.items():
        default_id = (
            ELEVENLABS_DEFAULT_FEMALE if persona.get("gender") == "female" else ELEVENLABS_DEFAULT_MALE
        )
        override = (os.environ.get(f"ELEVENLABS_VOICE_{key.upper()}", "") or "").strip()
        ids[key] = override or default_id
    return ids


ELEVENLABS_VOICE_IDS = _resolve_elevenlabs_voice_ids()


def persona_for_voice(voice: str | None) -> dict[str, str]:
    return VOICE_PERSONAS.get((voice or DEFAULT_REALTIME_VOICE).lower(), DEFAULT_PERSONA)


def elevenlabs_voice_id(voice: str | None) -> str:
    """Resolve a persona key to its ElevenLabs voice id. ElevenLabs has one model
    and a single voice per persona, so there is no per-language voice swap."""
    requested = (voice or DEFAULT_REALTIME_VOICE).strip().lower()
    if requested not in VOICE_PERSONAS:
        requested = DEFAULT_REALTIME_VOICE
    gender = persona_for_voice(requested).get("gender", "male")
    fallback = ELEVENLABS_DEFAULT_FEMALE if gender == "female" else ELEVENLABS_DEFAULT_MALE
    return ELEVENLABS_VOICE_IDS.get(requested) or fallback


def elevenlabs_voice_settings() -> dict[str, Any]:
    return {
        "stability": ELEVENLABS_VOICE_STABILITY,
        "similarity_boost": ELEVENLABS_VOICE_SIMILARITY,
        "style": ELEVENLABS_VOICE_STYLE,
        "speed": ELEVENLABS_VOICE_SPEED,
        "use_speaker_boost": ELEVENLABS_VOICE_SPEAKER_BOOST,
    }


# App language_id -> BCP-47 code. Used by the language coach / brain to reason
# about the call language and as an optional ElevenLabs language_code hint. Not
# tied to any single voice vendor.
LANGUAGE_CODES: dict[str, str] = {
    "english": "en-IN",
    "hinglish": "hi-IN",
    "hindi": "hi-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "odia": "od-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
}


def language_code_for_id(language_id: str | None) -> str:
    return LANGUAGE_CODES.get((language_id or "hinglish").lower(), "hi-IN")


# ElevenLabs wants an ISO 639-1 code; derive it from the BCP-47 head.
def elevenlabs_language_code(language_id: str | None) -> str:
    return language_code_for_id(language_id).split("-")[0]


LANGUAGE_IDS_BY_CODE = {
    code: language_id
    for language_id, code in LANGUAGE_CODES.items()
    if language_id not in {"hinglish", "hindi"} or code not in {"hi-IN"}
}
LANGUAGE_IDS_BY_CODE.setdefault("bn-IN", "bengali")
LANGUAGE_IDS_BY_CODE.setdefault("mr-IN", "marathi")
LANGUAGE_IDS_BY_CODE.setdefault("ta-IN", "tamil")


# Back-compat aliases for call sites not yet renamed. The brain's language-switch
# logic and a few phone paths reference the old Sarvam-named helpers.
def sarvam_language_code(language_id: str | None) -> str:
    return language_code_for_id(language_id)


def sarvam_stt_language_code(language_id: str | None) -> str:
    """Sarvam STT language hint. hinglish -> "unknown" so Saarika auto-detects the
    English/Indic mix per utterance instead of forcing one language."""
    normalized = (language_id or DEFAULT_LANGUAGE_ID).strip().lower()
    if normalized == "hinglish":
        return "unknown"
    return language_code_for_id(normalized)


SARVAM_LANGUAGE_CODES = LANGUAGE_CODES
SARVAM_LANGUAGE_IDS_BY_CODE = LANGUAGE_IDS_BY_CODE


def default_language_advice(language_id: str | None = None) -> dict[str, Any]:
    normalized = supported_render_language_id(language_id or DEFAULT_LANGUAGE_ID)
    return {
        "detected_language_id": normalized,
        "suggested_language_id": normalized,
        "transcription_language_id": normalized,
        "transcript_quality": "good",
        "confidence": "high",
        "should_switch": False,
        "nudge": "Open in Hinglish and switch only when the customer clearly prefers another language.",
        "rationale": "Default call opening behavior.",
    }


def determine_disposition(tool_name: str) -> str | None:
    return {
        "log_promise_to_pay": "Promise to pay logged",
        "log_already_paid": "Already paid claimed",
        "resend_invoice": "Invoice resend requested",
        "log_dispute": "Dispute raised",
        "update_contact": "Alternate contact captured",
        "transfer_to_human": "Transferred to human",
    }.get(str(tool_name or "").strip())


def parse_iso_datetime(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def serialize_call_history_entry(entry: dict[str, Any]) -> dict[str, Any]:
    duration_sec = max(0, int(entry.get("duration_sec", 0) or 0))
    ended_at = parse_iso_datetime(entry.get("timestamp")) or utc_now()
    ended_at_ms = int(ended_at.timestamp() * 1000)
    started_at_ms = max(0, ended_at_ms - (duration_sec * 1000))
    costs = entry.get("costs") if isinstance(entry.get("costs"), dict) else {}
    mode = str(entry.get("mode") or "voice").strip().lower()
    if mode not in {"voice", "chat"}:
        mode = "voice"

    mode_cost_usd = entry.get("mode_cost_usd")
    mode_tokens = entry.get("mode_tokens")
    if mode == "chat":
        chat_costs = costs.get("chat_agent") if isinstance(costs.get("chat_agent"), dict) else {}
        if mode_cost_usd is None:
            mode_cost_usd = float(chat_costs.get("estimated_cost_usd", 0.0) or 0.0)
        if mode_tokens is None:
            mode_tokens = int(chat_costs.get("total_tokens", 0) or 0)
    else:
        agent_costs = costs.get("agent") if isinstance(costs.get("agent"), dict) else {}
        if mode_cost_usd is None:
            mode_cost_usd = float(agent_costs.get("estimated_cost_usd", 0.0) or 0.0)
        if mode_tokens is None:
            mode_tokens = int(agent_costs.get("total_tokens", 0) or 0)

    combined = costs.get("combined") if isinstance(costs.get("combined"), dict) else {}
    return {
        "id": str(entry.get("id") or f"call_{uuid.uuid4().hex[:10]}"),
        "startedAt": started_at_ms,
        "endedAt": ended_at_ms,
        "durationSec": duration_sec,
        "mode": mode,
        "disposition": str(entry.get("disposition") or "Call ended"),
        "costUsd": float(entry.get("cost_usd", combined.get("estimated_cost_usd", 0.0)) or 0.0),
        "totalTokens": int(entry.get("total_units", combined.get("total_tokens", 0)) or 0),
        "modeCostUsd": float(mode_cost_usd or 0.0),
        "modeTokens": int(mode_tokens or 0),
        "summary": entry.get("summary") if isinstance(entry.get("summary"), dict) else None,
    }


def load_call_history(limit: int = 50) -> list[dict[str, Any]]:
    entries = read_jsonl(CALL_LOG_FILE)
    if not entries:
        return []
    ordered = [serialize_call_history_entry(entry) for entry in reversed(entries[-limit:])]
    return ordered


def sanitize_phone_number(value: str, *, keep_plus: bool = True) -> str:
    stripped = re.sub(r"[^\d+]", "", str(value or "").strip())
    if keep_plus and stripped.startswith("+"):
        return f"+{re.sub(r'\\D', '', stripped[1:])}"
    return re.sub(r"\D", "", stripped)


def public_base_url() -> str:
    configured = (os.environ.get("RENDER_EXTERNAL_URL", "") or "").strip().rstrip("/")
    if configured:
        return configured
    return ""


def public_websocket_base_url() -> str:
    base_url = public_base_url()
    if not base_url:
        return ""
    if base_url.startswith("https://"):
        return f"wss://{base_url[len('https://'):]}"
    if base_url.startswith("http://"):
        return f"ws://{base_url[len('http://'):]}"
    return base_url


def exotel_enabled() -> bool:
    return bool(
        EXOTEL_ACCOUNT_SID
        and EXOTEL_API_KEY
        and EXOTEL_API_TOKEN
        and EXOTEL_CALLER_ID
        and public_websocket_base_url()
    )


def build_exotel_stream_url(session_id: str) -> str:
    base = public_websocket_base_url().rstrip("/")
    if not base:
        raise RuntimeError("RENDER_EXTERNAL_URL must be set for Exotel streaming.")
    params = urlencode({
        "session_id": session_id,
        "sample-rate": EXOTEL_STREAM_SAMPLE_RATE,
    })
    return f"{base}{EXOTEL_STREAM_PATH}?{params}"


def build_exotel_status_callback_url() -> str:
    base = public_base_url().rstrip("/")
    if not base:
        raise RuntimeError("RENDER_EXTERNAL_URL must be set for Exotel callbacks.")
    return f"{base}/api/exotel/status"


def build_exotel_connect_payload(
    *,
    to_number: str,
    caller_id: str,
    stream_url: str,
    status_callback_url: str,
) -> dict[str, str]:
    return {
        "From": sanitize_phone_number(to_number, keep_plus=True),
        "CallerId": sanitize_phone_number(caller_id, keep_plus=False),
        "StatusCallback": status_callback_url,
        "StatusCallbackMethod": "POST",
        "StatusCallbackEvents[]": "terminal",
        "StreamUrl": stream_url,
        "StreamType": "bidirectional",
        "StreamTimeout": "86400",
    }


def exotel_basic_auth_header() -> str:
    token = base64.b64encode(f"{EXOTEL_API_KEY}:{EXOTEL_API_TOKEN}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def language_id_for_sarvam_code(language_code: str) -> str | None:
    normalized = str(language_code or "").strip()
    if not normalized:
        return None
    if normalized == "bn-IN":
        return "bengali"
    if normalized == "mr-IN":
        return "marathi"
    if normalized == "ta-IN":
        return "tamil"
    return SARVAM_LANGUAGE_IDS_BY_CODE.get(normalized)


def build_opening_text(
    customer: dict[str, Any],
    persona: dict[str, Any] | None,
    opening_language_label: str = "Hinglish",
) -> str:
    contact = customer.get("contact_name") or "the accounts payable contact"
    agent_name = (persona or {}).get("name") or "the DHL collections specialist"
    lower_label = str(opening_language_label or "Hinglish").strip().lower()
    kolkata_now = datetime.now(ZoneInfo("Asia/Kolkata"))
    hour = kolkata_now.hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    is_female = (persona or {}).get("gender") == "female"
    if lower_label == "hinglish":
        return (
            f"{greeting}, मैं {agent_name} DHL Express India से बोल {'रही' if is_female else 'रहा'} हूँ। "
            f"क्या मैं {contact} से बात कर {'रही' if is_female else 'रहा'} हूँ?"
        )
    if lower_label == "hindi":
        return (
            f"नमस्कार, मैं {agent_name} DHL Express India से बोल {'रही' if is_female else 'रहा'} हूँ। "
            f"क्या मैं {contact} से बात कर {'रही' if is_female else 'रहा'} हूँ?"
        )
    if lower_label == "marathi":
        return (
            f"नमस्कार, मी {agent_name}, DHL Express India मधून बोलत आहे. "
            f"{contact} यांच्यासोबत मी बोलत आहे का?"
        )
    if lower_label == "tamil":
        return (
            f"வணக்கம், நான் {agent_name}, DHL Express India-லிருந்து பேசுகிறேன். "
            f"{contact} உடன் பேசுகிறேனா?"
        )
    if lower_label == "bengali":
        return f"{greeting}, ami {agent_name}, DHL Express India theke bolchi. Ami ki {contact}-er sathe kotha bolchi?"
    return f"{greeting}, this is {agent_name} from DHL Express India. Am I speaking with {contact}?"
DEFAULT_ACCOUNT_ID = os.environ.get("DEMO_ACCOUNT_ID", "DHL001")

OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

SAP_FILE = DATA_DIR / "sap_mock.json"
GROUND_TRUTH_FILE = DATA_DIR / "GROUND_TRUTH.md"
BOARD_FILE = DATA_DIR / "supervisor_board.json"
LEDGER_FILE = DATA_DIR / "cost_ledger.json"
CALL_LOG_FILE = DATA_DIR / "call_log.jsonl"
SUPERVISOR_FLAGS_FILE = DATA_DIR / "supervisor_flags.jsonl"
TOOL_LOG_FILE = DATA_DIR / "tool_actions.jsonl"

AGENT_PROMPT_FILE = PROMPTS_DIR / "agent.md"
SUPERVISOR_PROMPT_FILE = PROMPTS_DIR / "supervisor.md"
LANGUAGE_COACH_PROMPT_FILE = PROMPTS_DIR / "language_coach.md"
CALL_SUMMARY_PROMPT_FILE = PROMPTS_DIR / "call_summary.md"

HUMAN_AGENT = {
    "name": "Ms Sanorita",
    "phone": "09416340644",
    "team": "DHL Express India Collections",
}

# Per-million-unit USD prices. OpenAI text models are token-based; ElevenLabs
# bills TTS per character and Sarvam STT per second of audio, so we express those
# as synthetic "per million" rates so the existing ledger math stays uniform.
PRICE_TABLE_VERSION = "openai+elevenlabs+sarvam-stt-pricing-2026-06-07"
# ElevenLabs API LIST pricing (2026, undiscounted — track real ongoing cost, not
# the first-month 50% promo). 1 char = 1 credit for v2/v3.
#   Multilingual v2 / eleven_v3: $0.10 / 1K chars = $100 / 1M chars
#   Flash / Turbo:               $0.05 / 1K chars = $50  / 1M chars
# Source: https://elevenlabs.io/pricing/api
ELEVENLABS_USD_PER_MILLION_CHARS = _env_float("ELEVENLABS_USD_PER_MILLION_CHARS", 100.0, minimum=0.0)
ELEVENLABS_USD_PER_MILLION_CHARS_FLASH = _env_float("ELEVENLABS_USD_PER_MILLION_CHARS_FLASH", 50.0, minimum=0.0)
# Sarvam STT bills in INR per hour; convert to USD using a configurable rate.
SARVAM_INR_PER_USD = _env_float("SARVAM_INR_PER_USD", 96.13, minimum=1.0)
SARVAM_STT_INR_PER_HOUR = _env_float("SARVAM_STT_INR_PER_HOUR", 30.0, minimum=0.0)


def _sarvam_seconds_usd_per_million(inr_per_hour: float) -> float:
    return round(((float(inr_per_hour) / 3600.0) * 1_000_000.0) / SARVAM_INR_PER_USD, 6)


DEFAULT_PRICE_TABLE = {
    # ElevenLabs (TTS). We meter output characters in `text_output_tokens`
    # so the existing ledger keys keep working.
    "eleven_v3": {
        "text_output_per_million": ELEVENLABS_USD_PER_MILLION_CHARS,
    },
    "eleven_flash_v2_5": {
        "text_output_per_million": ELEVENLABS_USD_PER_MILLION_CHARS_FLASH,
    },
    "eleven_multilingual_v2": {
        "text_output_per_million": ELEVENLABS_USD_PER_MILLION_CHARS_FLASH,
    },
    # Sarvam Saarika (STT). We meter seconds of mic audio in `audio_input_tokens`.
    "saaras:v3": {
        "audio_input_per_million": _sarvam_seconds_usd_per_million(SARVAM_STT_INR_PER_HOUR),
    },
    "saarika:v2.5": {
        "audio_input_per_million": _sarvam_seconds_usd_per_million(SARVAM_STT_INR_PER_HOUR),
    },
    # Legacy OpenAI realtime / transcription entries retained for historical
    # call logs, tests, and any stale client payloads that still report them.
    # Mirrors the current official gpt-realtime-1.5 standard pricing. We keep
    # the generic key for historical logs that still emit `gpt-realtime`.
    "gpt-realtime": {
        "text_input_per_million": 4.0,
        "text_cached_input_per_million": 0.4,
        "text_output_per_million": 16.0,
        "audio_input_per_million": 32.0,
        "audio_cached_input_per_million": 0.4,
        "audio_output_per_million": 64.0,
    },
    "gpt-realtime-mini": {
        "text_input_per_million": 0.6,
        "text_cached_input_per_million": 0.06,
        "text_output_per_million": 2.4,
        "audio_input_per_million": 10.0,
        "audio_cached_input_per_million": 0.3,
        "audio_output_per_million": 20.0,
    },
    "gpt-4o-transcribe": {
        "audio_input_per_million": 6.0,
        "text_input_per_million": 2.5,
        "text_output_per_million": 10.0,
    },
    "gpt-4o-mini-transcribe": {
        "audio_input_per_million": 3.0,
        "text_input_per_million": 1.25,
        "text_output_per_million": 5.0,
    },
    "gpt-5.5": {
        "text_input_per_million": 5.0,
        "text_cached_input_per_million": 0.5,
        "text_output_per_million": 30.0,
    },
    "gpt-5.4": {
        "text_input_per_million": 2.5,
        "text_cached_input_per_million": 0.25,
        "text_output_per_million": 15.0,
    },
    "gpt-5.4-mini": {
        "text_input_per_million": 0.75,
        "text_cached_input_per_million": 0.075,
        "text_output_per_million": 4.5,
    },
    "gpt-5-mini": {
        "text_input_per_million": 0.25,
        "text_cached_input_per_million": 0.025,
        "text_output_per_million": 2.0,
    },
    "gpt-4.1-mini": {
        "text_input_per_million": 0.4,
        "text_cached_input_per_million": 0.1,
        "text_output_per_million": 1.6,
    },
    "gpt-4.1": {
        "text_input_per_million": 2.0,
        "text_cached_input_per_million": 0.5,
        "text_output_per_million": 8.0,
    },
}

MODEL_PRICE_ALIASES = {
    # OpenAI exposes both stable aliases and versioned/marketing names for these models.
    "gpt-realtime-1.5": "gpt-realtime",
    "gpt-4o-transcribe-latest": "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe-latest": "gpt-4o-mini-transcribe",
}

SUPPORTED_LANGUAGE_OPTIONS = [
    {"id": "hinglish", "label": "Hinglish", "agent_label": "Hinglish", "transcription_language": "en"},
    {"id": "english", "label": "English", "agent_label": "English", "transcription_language": "en"},
    {"id": "hindi", "label": "Hindi", "agent_label": "Hindi", "transcription_language": "hi"},
    {"id": "assamese", "label": "Assamese", "agent_label": "Assamese", "transcription_language": "as"},
    {"id": "bengali", "label": "Bengali", "agent_label": "Bengali", "transcription_language": "bn"},
    {"id": "bodo", "label": "Bodo", "agent_label": "Bodo", "transcription_language": None},
    {"id": "dogri", "label": "Dogri", "agent_label": "Dogri", "transcription_language": None},
    {"id": "gujarati", "label": "Gujarati", "agent_label": "Gujarati", "transcription_language": "gu"},
    {"id": "kannada", "label": "Kannada", "agent_label": "Kannada", "transcription_language": "kn"},
    {"id": "kashmiri", "label": "Kashmiri", "agent_label": "Kashmiri", "transcription_language": "ks"},
    {"id": "konkani", "label": "Konkani", "agent_label": "Konkani", "transcription_language": None},
    {"id": "maithili", "label": "Maithili", "agent_label": "Maithili", "transcription_language": None},
    {"id": "malayalam", "label": "Malayalam", "agent_label": "Malayalam", "transcription_language": "ml"},
    {"id": "marathi", "label": "Marathi", "agent_label": "Marathi", "transcription_language": "mr"},
    {"id": "meitei", "label": "Manipuri / Meitei", "agent_label": "Manipuri", "transcription_language": None},
    {"id": "nepali", "label": "Nepali", "agent_label": "Nepali", "transcription_language": "ne"},
    {"id": "odia", "label": "Odia", "agent_label": "Odia", "transcription_language": "or"},
    {"id": "punjabi", "label": "Punjabi", "agent_label": "Punjabi", "transcription_language": "pa"},
    {"id": "sanskrit", "label": "Sanskrit", "agent_label": "Sanskrit", "transcription_language": "sa"},
    {"id": "santali", "label": "Santali", "agent_label": "Santali", "transcription_language": None},
    {"id": "sindhi", "label": "Sindhi", "agent_label": "Sindhi", "transcription_language": "sd"},
    {"id": "tamil", "label": "Tamil", "agent_label": "Tamil", "transcription_language": "ta"},
    {"id": "telugu", "label": "Telugu", "agent_label": "Telugu", "transcription_language": "te"},
    {"id": "urdu", "label": "Urdu", "agent_label": "Urdu", "transcription_language": "ur"},
]
SUPPORTED_LANGUAGE_MAP = {item["id"]: item for item in SUPPORTED_LANGUAGE_OPTIONS}
_configured_default_language_id = (os.environ.get("DEFAULT_LANGUAGE_ID", "hinglish") or "hinglish").strip().lower()
DEFAULT_LANGUAGE_ID = (
    _configured_default_language_id
    if _configured_default_language_id in SUPPORTED_LANGUAGE_MAP
    else "hinglish"
)
LANGUAGE_REQUEST_ALIASES: dict[str, tuple[str, ...]] = {
    "english": (
        "english", "angrezi", "angreji", "inglish", "ingriz",
        "\u0907\u0902\u0917\u094d\u0932\u093f\u0936",  # inglish (Devanagari)
        "\u0905\u0902\u0917\u094d\u0930\u0947\u091c\u0940",  # angreji (Devanagari)
        "\u0b86\u0b99\u0bcd\u0b95\u0bbf\u0bb2\u0bae\u0bcd",  # aangilam (Tamil)
        "\u0987\u0982\u09b0\u09c7\u099c\u09bf",  # ingreji (Bengali)
    ),
    "hinglish": ("hinglish",),
    "hindi": (
        "hindi", "hindee", "hindhi",
        "\u0939\u093f\u0902\u0926\u0940",  # hindi (Devanagari)
        "\u0939\u093f\u0928\u094d\u0926\u0940",  # hindi (alt spelling)
        "\u0bb9\u0bbf\u0ba8\u0bcd\u0ba4\u0bbf",  # hindi (Tamil)
        "\u09b9\u09bf\u09a8\u09cd\u09a6\u09bf",  # hindi (Bengali)
    ),
    "bengali": (
        "bengali", "bangla", "bangali", "bengoli",
        "\u09ac\u09be\u0982\u09b2\u09be",  # bangla (Bengali)
        "\u09ac\u09be\u0999\u09be\u09b2\u09bf",  # bangali (Bengali)
        "\u092c\u0902\u0917\u093e\u0932\u0940",  # bangali (Devanagari)
        "\u092c\u093e\u0902\u0917\u094d\u0932\u093e",  # bangla (Devanagari)
    ),
    "marathi": (
        "marathi", "marati", "maraathi",
        "\u092e\u0930\u093e\u0920\u0940",  # marathi (Devanagari)
        "\u092e\u0930\u093e\u0920\u093f",  # marathi (alt)
    ),
    "tamil": (
        "tamil", "thamizh", "tamizh", "thamil",
        "\u0ba4\u0bae\u0bbf\u0bb4\u0bcd",  # tamizh (Tamil)
        "\u0ba4\u0bae\u0bbf\u0bb4",  # tamil (Tamil, no virama)
        "\u0924\u092e\u093f\u0933\u094d",  # tamil (Devanagari)
        "\u0924\u092e\u093f\u0933",  # tamil (Devanagari)
        "\u0924\u092e\u093f\u0932",  # tamil (Devanagari, l-variant)
    ),
}

# Supported scripts include Latin plus scripts used by Indian languages in the selector.
SUPPORTED_SCRIPT_RANGES = [
    (0x0041, 0x007A),  # Latin
    (0x00C0, 0x024F),  # Latin extended
    (0x0900, 0x097F),  # Devanagari
    (0x0980, 0x09FF),  # Bengali / Assamese
    (0x0A00, 0x0A7F),  # Gurmukhi
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0B00, 0x0B7F),  # Oriya
    (0x0B80, 0x0BFF),  # Tamil
    (0x0C00, 0x0C7F),  # Telugu
    (0x0C80, 0x0CFF),  # Kannada
    (0x0D00, 0x0D7F),  # Malayalam
    (0xABC0, 0xABFF),  # Meetei Mayek
    (0x1C50, 0x1C7F),  # Ol Chiki
    (0x0600, 0x06FF),  # Arabic
    (0x0750, 0x077F),  # Arabic Supplement
    (0x08A0, 0x08FF),  # Arabic Extended-A
]

# Legacy REALTIME_TOOLS removed. The chat-completion engine (run_chat_agent_turn)
# is the only path that fires tools now and uses LLM_TURN_TOOLS + TOOL_HANDLERS
# directly. Stub kept for any snapshot consumer still reading the key.
REALTIME_TOOLS: list[dict[str, Any]] = []


def utc_now() -> datetime:
    return datetime.now(UTC)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return deepcopy(default)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return deepcopy(default)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    # Ensure the parent dir exists on demand. On a fresh deploy backend/data/ may
    # not exist yet (it is only created in a reset/init path), which made every
    # phone-call wrap-up write fail and silently drop the call from history.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                entries.append(parsed)
    except OSError:
        return []
    return entries


def read_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def render_invoice_lines(invoices: list[dict[str, Any]]) -> str:
    lines = []
    for inv in invoices:
        history = "; ".join(inv.get("history", []) or []) or "no prior issues"
        lines.append(
            f"- {inv.get('invoice_no')} | {inv.get('invoice_type', 'invoice')} | "
            f"{inv.get('currency', 'INR')} {inv.get('amount')} | "
            f"invoice date {inv.get('invoice_date')} | due {inv.get('due_date')} | "
            f"{inv.get('overdue_days')} days overdue | history: {history}"
        )
    return "\n".join(lines)


def build_account_context_block(account_number: str) -> str:
    customer = get_customer(account_number) or {}
    invoices = get_invoices(account_number)
    if not customer:
        return ""

    contacts = ", ".join(
        filter(
            None,
            [customer.get("contact_name"), customer.get("alternate_contact_name")],
        )
    ) or "unknown"
    languages = ", ".join(customer.get("language_preferences", []) or []) or "Hinglish, English"
    notes = "\n".join(f"- {n}" for n in customer.get("collection_notes", []) or [])
    transfer = customer.get("human_transfer", {}) or HUMAN_AGENT
    total = customer_outstanding(invoices)
    invoice_block = render_invoice_lines(invoices) or "- (no invoices on file)"
    payment_methods = get_payment_methods()
    payment_block = render_payment_methods(payment_methods)
    constants = get_collections_constants()

    return (
        "\n\n# Known account context (already loaded — do NOT ask the customer for any of this)\n"
        f"- Account number: {customer.get('account_number', account_number)}\n"
        f"- Company: {customer.get('company_name')}\n"
        f"- Primary AP contact: {customer.get('contact_name')}"
        + (
            f" (backup: {customer.get('alternate_contact_name')})"
            if customer.get("alternate_contact_name")
            else ""
        )
        + "\n"
        f"- All known contact names on this account: {contacts}\n"
        f"- Registered email: {customer.get('registered_email')}\n"
        f"- Phone: {customer.get('phone')}\n"
        f"- Billing city: {customer.get('billing_city', 'unknown')}\n"
        f"- Payment terms: {customer.get('payment_terms', 'unknown')}\n"
        f"- Customer language preferences: {languages}\n"
        f"- Total outstanding: {customer.get('currency', 'INR')} {total} across {len(invoices)} invoices\n"
        f"- Human transfer target: {transfer.get('name', HUMAN_AGENT['name'])} "
        f"({transfer.get('phone', HUMAN_AGENT['phone'])})\n"
        f"- Internal notes:\n{notes or '- (none)'}\n"
        "- Overdue invoices:\n"
        f"{invoice_block}\n"
        "- Available payment methods (use ONLY these two — do NOT invent any other channel like UPI, cheque, debit/credit card, NEFT to a generic account, etc.):\n"
        f"{payment_block}\n"
        f"- Proof-of-payment email (for already-paid claims): {constants['proof_of_payment_email']}\n"
        f"- Promise-to-pay window: customer must commit to a date within {constants['promise_date_max_business_days']} business days. If the date is vague or further out, push back politely and ask for a date inside that window.\n"
        f"- Soft monthly collection target: try your best to secure payment before the {constants['monthly_collection_target_day']}th of every month.\n"
        f"- Allowed call dispositions (the ONLY values you may set when logging the outcome): {', '.join(constants['dispositions'])}.\n"
        "\nUsage rules for this context:\n"
        "- Treat the account number, company, contacts, and invoice list above as ground truth. They are the reason for this call.\n"
        "- Confirm identity by NAME (e.g. ask if you are speaking with the contact above), not by asking the customer for their account number or company name.\n"
        "- HARD RULE: Before stating ANY invoice number, amount, currency, due date, or overdue-days out loud you MUST have called get_invoices in this call. If you have not, call get_invoices first and wait for the result. NEVER invent or approximate any of those fields. Use ONLY values returned by the tool call or listed verbatim in this context block.\n"
        "- If you ever catch yourself about to say a number you did not pull from the tool result above, stop and call get_invoices instead. Numbers like 1200 / 15 days that are not in the ground-truth list are forbidden.\n"
        "- HARD RULE on past issues: When the customer asks about disputes, conflicts, or resolved issues for an invoice, you MUST consult the `history` field for that invoice in the context above (or call get_invoices). If history is non-empty, summarise it accurately (e.g. credit notes issued, disputes resolved, delayed shipments). NEVER say \"no resolved issues\" or \"no conflicts\" when the history list contains entries.\n"
        "- HARD RULE on payment methods: There are exactly TWO sanctioned payment channels — DHL MyBill self-serve portal, and Virtual Account Number bank transfer. When the customer asks how they can pay, what options/channels/methods are available, or where to send money, you MUST offer ONLY these two by their labels above. NEVER mention UPI, cheques, debit/credit cards, generic NEFT to other accounts, or any channel not in the list. If the customer asks for the specific Virtual Account Number, say you will share it from the collections desk after the call.\n"
        "- Never ask the customer for the account number.\n"
        "- The opening turn must NOT contain invoice numbers, amounts, or payment talk. Build rapport first as instructed in the main prompt.\n"
    )


def build_persona_block(voice: str | None) -> str:
    persona = persona_for_voice(voice)
    gender = persona["gender"]

    if gender == "female":
        hindi_rule = (
            "- This rule applies ONLY when you are actually speaking Hindi, Hinglish, Marathi, Punjabi, Gujarati or any "
            "language with gendered verb conjugation. It does NOT mean you should default to Hindi/Hinglish — language "
            "choice is governed by the # Language behaviour section and the per-turn language coach nudge.\n"
            "- When speaking such a language, you MUST use FEMININE verb forms for yourself. Examples: \"main kar raha hoon\" "
            "(not \"kar raha hoon\"), \"main bol raha hoon\", \"main madad karungi\" (not \"karunga\"). Never mix masculine "
            "and feminine forms inside a single turn.\n"
            "- In English, refer to yourself with she/her if needed.\n"
        )
    elif gender == "male":
        hindi_rule = (
            "- This rule applies ONLY when you are actually speaking Hindi, Hinglish, Marathi, Punjabi, Gujarati or any "
            "language with gendered verb conjugation. It does NOT mean you should default to Hindi/Hinglish — language "
            "choice is governed by the # Language behaviour section and the per-turn language coach nudge.\n"
            "- When speaking such a language, you MUST use MASCULINE verb forms for yourself. Examples: \"main kar raha hoon\" "
            "(not \"kar raha hoon\"), \"main bol raha hoon\", \"main madad karunga\" (not \"karungi\"). Never mix.\n"
            "- In English, refer to yourself with he/him if needed.\n"
        )
    else:
        hindi_rule = (
            "- In Hindi/Hinglish, prefer gender-neutral phrasings or stay in English when possible.\n"
        )

    return (
        "\n\n# Agent persona (matches the configured voice — do NOT override)\n"
        f"- Your name is {persona['name']}.\n"
        f"- Your gender for this call is {persona['gender']} ({persona['pronouns']}).\n"
        f"- Introduce yourself with this name only. If the customer addresses you by a different name, "
        f"politely correct them once: \"Actually, this is {persona['name']} from DHL Express India.\"\n"
        "- Never use a name or gender that does not match this persona block.\n"
        "- Stay consistent with this gender for the ENTIRE call. Do not flip between feminine and masculine "
        "verb forms within a single turn or across turns.\n"
        + hindi_rule
    )


def compose_agent_instructions(
    account_number: str | None = None,
    voice: str | None = None,
) -> str:
    base = read_prompt(AGENT_PROMPT_FILE)
    persona = build_persona_block(voice)
    context = build_account_context_block(account_number or DEFAULT_ACCOUNT_ID)
    return base + persona + context


def language_option(language_id: str | None) -> dict[str, Any]:
    normalized = str(language_id or DEFAULT_LANGUAGE_ID).strip().lower()
    return deepcopy(SUPPORTED_LANGUAGE_MAP.get(normalized, SUPPORTED_LANGUAGE_MAP[DEFAULT_LANGUAGE_ID]))


def text_contains_language_alias(text: str, aliases: tuple[str, ...]) -> bool:
    for alias in aliases:
        if alias.isascii():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return True
        elif alias in text:
            return True
    return False


def language_aliases(language_id: str) -> tuple[str, ...]:
    option = language_option(language_id)
    aliases = {
        language_id.casefold(),
        str(option.get("label") or "").casefold(),
        str(option.get("agent_label") or "").casefold(),
    }
    aliases.update(LANGUAGE_REQUEST_ALIASES.get(language_id, ()))
    return tuple(alias for alias in aliases if alias)


def explicit_language_request_language_id(transcript: str) -> str | None:
    text = re.sub(r"\s+", " ", transcript.casefold()).strip()
    if not text:
        return None

    if re.search(r"\bi do(?: not|n't) understand\b", text):
        for language_id in SUPPORTED_LANGUAGE_MAP:
            if language_id == "english":
                continue
            if text_contains_language_alias(text, language_aliases(language_id)):
                return "english"

    local_switch_verbs = (
        "\u092c\u094b\u0932",
        "\u092c\u094b\u0932\u093f",
        "\u092c\u094b\u0932\u093e",
        "\u0baa\u0bc7\u0b9a",
        "\u0baa\u0bc7\u0b9a\u0bc1",
    )
    for language_id in SUPPORTED_LANGUAGE_MAP:
        aliases = language_aliases(language_id)
        if text_contains_language_alias(text, aliases) and any(verb in text for verb in local_switch_verbs):
            return language_id

    command_patterns = (
        r"(?:speak|reply|respond|continue|talk|communicate|answer)\s+(?:to me\s+)?(?:in\s+)?{alias}",
        r"(?:switch(?:\s+back)?\s+to|back\s+to|use)\s+{alias}",
    )
    contextual_patterns = (
        r"(?:when you respond|right now|from now on|next response|next turn).{0,24}{alias}",
        r"{alias}\s+(?:mein|me)\b",
    )

    for language_id in SUPPORTED_LANGUAGE_MAP:
        aliases = language_aliases(language_id)
        if not text_contains_language_alias(text, aliases):
            continue
        for alias in aliases:
            alias_pattern = re.escape(alias) if not alias.isascii() else rf"\b{re.escape(alias)}\b"
            if any(re.search(pattern.replace("{alias}", alias_pattern), text) for pattern in command_patterns):
                return language_id
            if any(re.search(pattern.replace("{alias}", alias_pattern), text) for pattern in contextual_patterns):
                return language_id
    return None


def explicit_language_advice(
    requested_language_id: str,
    current_language_id: str | None,
    transcript_quality: str,
) -> dict[str, Any]:
    current = language_option(current_language_id)
    requested = language_option(requested_language_id)
    english_tail = (
        " Use zero Hindi, Hinglish, Bengali, or mixed-language filler."
        if requested["id"] == "english"
        else ""
    )
    bengali_tail = (
        " Do not first say you will switch later; your first words must already be in Bengali."
        if requested["id"] == "bengali"
        else ""
    )
    return {
        "detected_language_id": requested["id"],
        "suggested_language_id": requested["id"],
        "transcription_language_id": requested["id"],
        "transcript_quality": transcript_quality,
        "confidence": "high",
        "should_switch": requested["id"] != current["id"],
        "nudge": (
            f"The customer explicitly requested {requested['agent_label']}. "
            f"Your very next turn must be entirely in {requested['agent_label']}. "
            f"Do not promise to switch later; switch now.{english_tail}{bengali_tail}"
        ).strip(),
        "rationale": "Explicit language instruction from the customer overrides the default language flow.",
    }


def inferred_language_advice(
    requested_language_id: str,
    current_language_id: str | None,
    transcript_quality: str,
) -> dict[str, Any]:
    current = language_option(current_language_id)
    requested = language_option(requested_language_id)
    english_tail = (
        " Use zero Hindi, Hinglish, Bengali, or mixed-language filler."
        if requested["id"] == "english"
        else ""
    )
    return {
        "detected_language_id": requested["id"],
        "suggested_language_id": requested["id"],
        "transcription_language_id": requested["id"],
        "transcript_quality": transcript_quality,
        "confidence": "medium",
        "should_switch": requested["id"] != current["id"],
        "nudge": (
            f"The customer is speaking in {requested['agent_label']}. "
            f"Reply entirely in {requested['agent_label']} on your very next turn.{english_tail}"
        ).strip(),
        "rationale": "Language coach inferred the reply language from the customer's latest utterance.",
    }


def supported_languages_payload() -> list[dict[str, Any]]:
    return [deepcopy(item) for item in SUPPORTED_LANGUAGE_OPTIONS]


STT_PROMPT_VOCAB = (
    "DHL, DHL Express India, MyBill, Virtual Account Number, invoice, overdue, "
    "promise to pay, credit note, waybill, AP team, accounts payable, "
    "Hinglish, namaste, dhanyavaad, shukriya, theek hai, accha, paisa."
)


# Phrases that indicate the STT model echoed an instruction-style prompt back as
# fake "speech" on silence. Used to drop hallucinated user turns before they
# poison the agent / language coach.
STT_HALLUCINATION_MARKERS = (
    "transcribe faithfully",
    "do not hallucinate",
    "if audio is unclear",
    "[unclear]",
    "collections call. the agent",
    "primary mode starts in hinglish",
    "indian regional languages at any time",
    "prefer english text for english",
)


_SARVAM_HALLUCINATION_MARKERS = (
    "welcome back to my channel",
    "subscribe to my channel",
    "thanks for watching",
    "thank you for watching",
    "like and subscribe",
    "हिंदी समाचार",
    "नमस्कार दोस्तों",
    "हेलो दोस्तों",
    "today we will",
    "in this video",
    "i am your host",
    "host and i am here",
    "world is changing",
)

# Single-word filler that Saarika emits on near-silence or background noise.
_SARVAM_FILLER_SINGLE_WORDS = {
    "anyways", "anyway", "okay", "ok", "yeah", "yes", "no", "hmm", "mm",
    "uh", "um", "thanks", "thank", "hello", "hi", "bye",
    "अच्छा", "ठीक", "हाँ", "नहीं",
}


_STT_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_ECHO_OVERLAP_STOPWORDS = {
    "this",
    "that",
    "from",
    "with",
    "speaking",
    "hello",
    "good",
    "morning",
    "afternoon",
    "evening",
}


def is_stt_hallucination(text: str) -> bool:
    """Drop transcripts that look like Whisper/Saarika training-data filler
    rather than real customer speech. These show up when VAD flushes near-
    silent buffers (cough, breath, background chatter)."""
    if not text:
        return True
    lowered = text.casefold().strip().strip(".!?,").strip()
    if any(marker in lowered for marker in _SARVAM_HALLUCINATION_MARKERS):
        return True
    # Frontend gates single-word fillers per-context (e.g. drops them when
    # they arrive during agent playback, treats them as real otherwise).
    tokens = re.findall(r"\w+", lowered)
    # Extreme repetition (model echoing the same clause) — classic Whisper
    # behaviour on no-speech input.
    if len(tokens) >= 12:
        # If the most common 3-gram appears >=3 times, treat as loop.
        ngrams: dict[str, int] = {}
        for i in range(len(tokens) - 2):
            gram = " ".join(tokens[i : i + 3])
            ngrams[gram] = ngrams.get(gram, 0) + 1
        if ngrams and max(ngrams.values()) >= 3:
            return True
    return False


def is_likely_stt_hallucination(text: str) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    if any(marker in lowered for marker in STT_HALLUCINATION_MARKERS):
        return True
    # Echo of the vocabulary prompt itself.
    vocab_lower = STT_PROMPT_VOCAB.casefold()
    if lowered.strip() and lowered.strip() in vocab_lower:
        return True
    return False


def stt_word_tokens(text: str) -> list[str]:
    if not text:
        return []
    return _STT_TOKEN_RE.findall(normalize_whitespace(text).casefold())


_WEAK_OPENING_INTERRUPTION_PHRASES = {
    "hello",
    "hello ji",
    "hello sir",
    "hello madam",
    "hello maam",
    "hi",
    "yes",
    "yes sir",
    "yes ji",
    "yeah",
    "yep",
    "ok",
    "okay",
    "sir",
    "madam",
    "maam",
    "ji",
    "haan",
    "haan ji",
    "ji haan",
    "han",
    "hmm",
    "hmmm",
    "hmm ji",
    "boliye",
    "bolo",
    "suniye",
    "ji boliye",
}


def is_weak_opening_interruption(text: str) -> bool:
    tokens = stt_word_tokens(text)
    if not tokens:
        return True
    phrase = " ".join(tokens)
    if phrase in _WEAK_OPENING_INTERRUPTION_PHRASES:
        return True
    if len(tokens) == 1:
        token = tokens[0]
        return len(token) <= 3 or token in {"hello", "hi", "yes", "ji", "haan", "han", "hmm", "sir"}
    return False


def looks_like_agent_echo(transcript_text: str, assistant_text: str) -> bool:
    customer = normalize_whitespace(transcript_text).strip().casefold()
    assistant = normalize_whitespace(assistant_text).strip().casefold()
    if not customer or not assistant:
        return False
    if customer == assistant:
        return True
    if len(customer) >= 12 and customer in assistant:
        return True
    customer_tokens = stt_word_tokens(customer)
    assistant_tokens = stt_word_tokens(assistant)
    if len(customer_tokens) < 3 or len(assistant_tokens) < 3:
        return False
    informative_customer = [
        token for token in customer_tokens if len(token) >= 4 and token not in _ECHO_OVERLAP_STOPWORDS
    ]
    informative_assistant = {
        token for token in assistant_tokens if len(token) >= 4 and token not in _ECHO_OVERLAP_STOPWORDS
    }
    if len(informative_customer) < 3 or len(informative_assistant) < 3:
        return False
    shared = sum(1 for token in informative_customer if token in informative_assistant)
    return shared >= 3 and (shared / max(len(informative_customer), 1)) >= 0.75


def should_apply_language_switch_hint(text: str) -> bool:
    trimmed = normalize_whitespace(text).strip()
    if not trimmed:
        return False
    if explicit_language_request_language_id(trimmed):
        return True
    tokens = stt_word_tokens(trimmed)
    if len(tokens) >= 2:
        return True
    non_ascii_chars = sum(1 for char in trimmed if ord(char) > 127 and not char.isspace())
    return non_ascii_chars >= 4


def phone_turn_commit_delay_seconds(text: str) -> float:
    trimmed = normalize_whitespace(text).strip()
    if not trimmed:
        return PHONE_TURN_COMMIT_DELAY_SECONDS
    if explicit_language_request_language_id(trimmed):
        return PHONE_TURN_COMMIT_DELAY_SECONDS
    tokens = stt_word_tokens(trimmed)
    if len(tokens) <= 1:
        return max(PHONE_TURN_COMMIT_DELAY_SECONDS, PHONE_SHORT_FRAGMENT_COMMIT_DELAY_SECONDS)
    if len(tokens) <= 2 and not re.search(r"[.?!।]$", trimmed):
        return max(PHONE_TURN_COMMIT_DELAY_SECONDS, PHONE_SHORT_FRAGMENT_COMMIT_DELAY_SECONDS)
    return PHONE_TURN_COMMIT_DELAY_SECONDS


def alpha_character_count(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def phone_language_switch_signal(
    text: str,
    detected_code: str,
    current_language_id: str | None,
    preferred_language_id: str | None,
) -> dict[str, Any]:
    trimmed = normalize_whitespace(text).strip()
    current = supported_render_language_id(current_language_id or DEFAULT_LANGUAGE_ID)
    preferred = supported_render_language_id(preferred_language_id or current)
    if not trimmed:
        return {"action": "keep", "candidate_language_id": current, "reason": "empty"}

    explicit_language_id = explicit_language_request_language_id(trimmed)
    if explicit_language_id:
        explicit_language_id = supported_render_language_id(explicit_language_id)
        return {
            "action": "switch",
            "candidate_language_id": explicit_language_id,
            "reason": "explicit_request",
        }

    if is_plain_english(trimmed):
        return {
            "action": "switch" if current != "english" else "keep",
            "candidate_language_id": "english",
            "reason": "plain_english",
        }

    tokens = stt_word_tokens(trimmed)
    alpha_count = alpha_character_count(trimmed)
    has_script = has_indic_script(trimmed)
    detected_language_id = language_id_for_sarvam_code(detected_code)
    script_language_id = None
    if has_script:
        script_language_id = language_id_for_script(trimmed, current, preferred)

    candidate_language_id = script_language_id or detected_language_id or current
    normalized_candidate = supported_render_language_id(candidate_language_id)

    if normalized_candidate == current:
        return {
            "action": "keep",
            "candidate_language_id": normalized_candidate,
            "reason": "current_language",
        }

    if has_script and normalized_candidate != candidate_language_id:
        return {
            "action": "drop",
            "candidate_language_id": current,
            "reason": "unrenderable_script_candidate",
        }

    if has_script:
        # Token count is the reliable signal. alpha_character_count is biased
        # against Indic scripts — Devanagari words pack into far fewer codepoints
        # than Latin, so a clear 4-word reply like "हा जी आहे ना" has only ~5
        # alpha chars and was wrongly dropped as a short fragment. Lean on tokens;
        # keep a low alpha floor only to reject single stray glyphs.
        if len(tokens) >= 3:
            return {
                "action": "switch",
                "candidate_language_id": normalized_candidate,
                "reason": "strong_script_switch",
            }
        if len(tokens) >= 2 or alpha_count >= 4:
            return {
                "action": "tentative",
                "candidate_language_id": normalized_candidate,
                "reason": "tentative_script_switch",
            }
        return {
            "action": "drop",
            "candidate_language_id": normalized_candidate,
            "reason": "short_script_fragment",
        }

    if (
        detected_language_id
        and detected_language_id in RENDERABLE_LANGUAGE_IDS
        and normalized_candidate != current
        and len(tokens) >= 4
        and alpha_count >= 12
    ):
        return {
            "action": "tentative",
            "candidate_language_id": normalized_candidate,
            "reason": "detected_language_only",
        }

    return {
        "action": "keep",
        "candidate_language_id": current,
        "reason": "no_switch_signal",
    }


def extract_json_payload(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
    return {}


def char_in_ranges(char: str, ranges: list[tuple[int, int]]) -> bool:
    codepoint = ord(char)
    return any(start <= codepoint <= end for start, end in ranges)


HINGLISH_TOKENS = {
    "aap", "aapko", "aapke", "aapka", "aapki", "accha", "acha", "haan", "haanji", "hanji", "ji", "hoon", "hai",
    "main", "mein", "mera", "meri", "kar", "karta", "karti", "karunga", "karungi",
    "raha", "rahi", "rahe", "bilkul", "namaste", "theek", "thik", "kya", "kyun",
    "nahi", "nahin", "matlab", "samjha", "samjhi", "dheere", "din", "paisa",
    "paise", "rupee", "rupaye", "thoda", "bahut", "abhi", "phir", "kuch",
    "sahi", "galat", "lekin", "magar", "ya", "aur", "wala", "wali",
    "bata", "batao", "bataye", "bataiye", "bol", "bolo", "boliye", "samjhao",
    "arre", "arey", "ispe", "isme", "kya", "kyon", "ka", "ke", "ki", "sirf",
    "ab", "tak", "liye", "baar", "ek",
}
ROMANIZED_INDIC_TOKEN_RE = re.compile(
    r"\b(?:aap|aapko|aapke|aapka|aapki|main|mein|hoon|hain|karna|karke|karte|karti|karta|"
    r"kya|kyu|kyun|nahi|nahin|haan|haanji|hanji|namaste|theek|thik|accha|acha|raha|rahi|rahe|"
    r"baat|paisa|paise|abhi|phir|kuch|sahi|baad|pehle|liye|wala|wali|saath|baare|"
    r"din|dino|kal|aaj|kabhi|matlab|samjha|samjhi|bilkul|bata|batao|"
    r"bataye|bataiye|bolo|boliye|suno|dekho|hota|hoti|hone|honge|"
    r"hua|hui|huye|kisi|sakte|sakti|sakta|payenge|payega|"
    r"payegi|deti|deta|dete|leti|leta|lete|mera|meri|mere|tera|teri|tere|hamara|"
    r"hamari|hamare|shukriya|dhanyavaad|maaf|kripya|zaroor|bhej|jaldi|"
    r"arre|arey|ispe|isme)\b",
    re.IGNORECASE,
)

# High-frequency Marathi tokens that do NOT appear in standard Hindi. Used to
# split Marathi from Hindi when both share the Devanagari block. Keep these
# Marathi-exclusive: avoid words that are also common Hindi (e.g. \u0939\u0948, \u092e\u0947\u0902, \u0914\u0930).
MARATHI_SCRIPT_MARKERS = (
    # copula / existence (Marathi \u0906\u0939\u0947-family; Hindi uses \u0939\u0948/\u0939\u0948\u0902)
    "\u0906\u0939\u0947",          # aahe (is)
    "\u0906\u0939\u0947\u0924",    # aahet (are)
    "\u0906\u0939\u0947\u0938",    # aahes
    "\u0906\u0939\u0947\u0924\u093e",  # aaheta
    "\u0928\u093e\u0939\u0940",    # naahi (no/not; Hindi \u0928\u0939\u0940\u0902)
    "\u0928\u093e\u0939\u0940\u092f\u0947",  # naahiye
    "\u0939\u094b\u0924\u093e",    # hota (was)
    "\u0939\u094b\u0924\u0940",    # hoti
    "\u0939\u094b\u0924\u0947",    # hote
    "\u0939\u094b\u0924\u094d\u092f\u093e",  # hotya (were, fem pl) -- distinctly Marathi
    "\u0939\u094b\u0923\u093e\u0930",  # honar (will happen/be)
    "\u0939\u094b\u0908\u0932",    # hoil (will be)
    "\u091a\u093e\u0932\u0947\u0932",  # chalel (will do/work)
    # pronouns (Marathi-specific forms)
    "\u0924\u0941\u092e\u094d\u0939\u0940",  # tumhi (you, formal)
    "\u0906\u092a\u0923",          # aapan (we/you incl.)
    "\u0906\u092e\u094d\u0939\u0940",  # aamhi (we)
    "\u092e\u0940",                # mi (I; Hindi \u092e\u0948\u0902)
    # NOTE: \u0924\u094b removed -- it means "he/that" in Marathi but "then/so" in Hindi
    # and is very common in Hindi, causing false Marathi hits on Hindi turns.
    "\u0924\u094d\u092f\u093e\u0902\u0928\u0940",  # tyanni
    "\u092e\u0932\u093e",          # mala (to me)
    "\u0924\u0941\u092e\u094d\u0939\u093e\u0932\u093e",  # tumhala (to you)
    "\u0906\u092a\u0932\u094d\u092f\u093e",  # aaplya
    # common verbs / question words
    "\u0915\u093e\u092f",          # kaay (what; Hindi \u0915\u094d\u092f\u093e)
    "\u0915\u0938\u0947",          # kase (how)
    "\u0915\u0938\u0902",          # kasan
    "\u0915\u093f\u0924\u0940",    # kiti (how much; Hindi \u0915\u093f\u0924\u0928\u093e)
    "\u0915\u0941\u0920\u0947",    # kuthe (where; Hindi \u0915\u0939\u093e\u0901)
    "\u0915\u0947\u0935\u094d\u0939\u093e",  # kevha (when)
    "\u0938\u093e\u0902\u0917\u093e",  # sanga (tell)
    "\u0938\u093e\u0902\u0917\u093f\u0924\u0932\u0902",  # sangitla
    "\u0915\u0930\u0924\u094b",    # karto (do, masc)
    "\u0915\u0930\u0924\u0947",    # karte (do, fem)
    "\u0915\u0930\u0924\u094b\u092f",  # kartoy
    "\u0926\u0947\u0924\u094b",    # deto
    "\u092a\u093e\u0939\u093f\u091c\u0947",  # pahije (need/want)
    "\u092d\u0930\u0923\u093e",    # bharna (to pay/fill)
    "\u092d\u0930\u0924\u094b",    # bharto
    "\u091d\u093e\u0932\u0902",    # zhala (done)
    "\u091d\u093e\u0932\u0940",    # zhali
    "\u091c\u093e\u0938\u094d\u0924",  # jaast (more)
    # postpositions / connectives distinctive to Marathi
    "\u091a",                      # cha (emphatic particle)
    "\u0928\u0902\u0924\u0930",    # nantar (after)
    "\u092e\u094d\u0939\u0923\u091c\u0947",  # mhanje (meaning)
    "\u092e\u094d\u0939\u0923\u0942\u0928",  # mhanun (so/because)
    "\u0906\u0924\u093e",          # aata (now; Hindi \u0905\u092d\u0940 differs)
    "\u092a\u0923",                # pan (but)
    "\u0906\u0923\u093f",          # aani (and; Hindi \u0914\u0930)
    "\u092a\u0941\u0922\u091a\u094d\u092f\u093e",  # pudhchya (next) -- Marathi oblique
    "\u0906\u0920\u0935\u0921\u094d\u092f\u093e\u0924",  # aathvadyat (in the week)
    "\u092f\u093e\u091a\u094d\u092f\u093e\u0935\u0930\u0924\u0940",  # yachyavarti (on this)
    "\u0924\u0930",                # tar (then/if; Marathi connective)
    "\u092e\u0927\u094d\u092f\u0947",          # madhye (in; Hindi \u092e\u0947\u0902)
    "\u0915\u093f\u0902\u0935\u093e",        # kinva (or)
    "\u0926\u093f\u0935\u0938\u093e\u0924",  # divasat (in days)
)


# Markers short enough to appear as substrings inside unrelated Hindi words
# (e.g. "च" inside "अच्छा", "मी" inside "मीटिंग"). These must match as whole
# tokens only, never as substrings, to avoid mislabelling Hindi as Marathi.
_MARATHI_WHOLE_WORD_ONLY = frozenset(
    {m for m in MARATHI_SCRIPT_MARKERS if len(m) <= 3}
)


# High-frequency Hindi tokens that are NOT standard Marathi. Used to detect a
# switch BACK to Hindi while the active language is Marathi (both Devanagari).
# Without this, plain Hindi sticks to Marathi forever once Marathi is active.
HINDI_SCRIPT_MARKERS = (
    "है",            # hai (is; Marathi आहे)
    "हैं",           # hain (are)
    "हूँ",           # hoon
    "हूं",           # hoon (alt)
    "नहीं",          # nahin (no; Marathi नाही)
    "क्या",          # kya (what; Marathi काय)
    "क्यों",         # kyun (why)
    "और",            # aur (and; Marathi आणि)
    "रहा",           # raha
    "रही",           # rahi
    "रहे",           # rahe
    "मैं",           # main (I; Marathi मी)
    "आप",            # aap (you; Marathi तुम्ही)
    "आपको",          # aapko
    "करो",           # karo (do; Marathi करा)
    "बताओ",          # batao (tell; Marathi सांगा)
    "चलो",           # chalo (let's; Marathi चला)
    "हुआ",           # hua
    "गया",           # gaya
    "वापस",          # vaapas (back)
    "कब",            # kab (when; Marathi केव्हा)
    "कितना",         # kitna (how much; Marathi किती)
    "कहाँ",          # kahan (where; Marathi कुठे)
    "हूंगा",         # hoonga (will be, masc)
    "दूंगा",         # doonga (will give)
    "रुका",          # ruka (held/stopped)
    "क्यूं",         # kyun (alt)
    # NOTE: का / की / के / को / में / से / ठीक were removed. They are NOT
    # Hindi-exclusive — Marathi uses का (question particle), में/मध्ये, से, etc.
    # Keeping them mislabelled Marathi turns as Hindi (the "पुढच्या आठवड्यात ... का?"
    # bug) and forced wrong Hindi replies.
)
_HINDI_WHOLE_WORD_ONLY = frozenset({m for m in HINDI_SCRIPT_MARKERS if len(m) <= 3})


def _marker_hits(text: str, markers: tuple[str, ...], whole_word_only: frozenset[str]) -> int:
    lowered = normalize_whitespace(text).casefold()
    if not lowered:
        return 0
    tokens = set(re.split(r"[^ऀ-ॿa-z]+", lowered))
    tokens.discard("")
    hits = 0
    for marker in markers:
        if marker in whole_word_only:
            if marker in tokens:
                hits += 1
        elif marker in lowered:
            hits += 1
    return hits


def looks_like_marathi(text: str) -> bool:
    return _marker_hits(text, MARATHI_SCRIPT_MARKERS, _MARATHI_WHOLE_WORD_ONLY) > 0


def looks_like_hindi(text: str) -> bool:
    """True when Devanagari text carries Hindi-specific markers and is not
    Marathi. Used to break the Hindi/Marathi tie in language_id_for_script."""
    return (
        _marker_hits(text, HINDI_SCRIPT_MARKERS, _HINDI_WHOLE_WORD_ONLY) > 0
        and not looks_like_marathi(text)
    )

def has_indic_script(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0980 <= cp <= 0x09FF:
            return True
        if 0x0A00 <= cp <= 0x0DFF or 0x0E00 <= cp <= 0x0FFF:
            return True
    return False


def is_plain_english(text: str) -> bool:
    """Latin-only, no Hinglish tokens. Strong signal of pure English."""
    stripped = text.strip()
    if len(stripped) < 2:
        return False
    if has_indic_script(stripped):
        return False
    words = re.findall(r"[A-Za-z']+", stripped.lower())
    if not words:
        return False
    if len(words) < 3:
        return False
    if any(w in HINGLISH_TOKENS for w in words) or ROMANIZED_INDIC_TOKEN_RE.search(stripped):
        return False
    return True


def transcript_quality_signal(text: str) -> str:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "unclear"
    supported_letters = sum(1 for char in letters if char_in_ranges(char, SUPPORTED_SCRIPT_RANGES))
    supported_ratio = supported_letters / len(letters)
    if supported_ratio < 0.45:
        return "suspect"
    if len("".join(letters)) < 2:
        return "unclear"
    return "good"


def language_id_for_script(text: str, current_language_id: str | None, preferred_language_id: str | None) -> str:
    current = language_option(current_language_id)["id"]
    preferred = language_option(preferred_language_id)["id"]
    for char in text:
        codepoint = ord(char)
        if 0x0900 <= codepoint <= 0x097F:
            if looks_like_marathi(text):
                return "marathi"
            # Break the Hindi/Marathi tie BEFORE sticking to current: a Hindi
            # utterance while Marathi is active must switch back to Hindi, not be
            # absorbed as "still Marathi". This was the marathi->hindi stuck bug.
            if looks_like_hindi(text):
                return "hindi"
            if current in {"hindi", "marathi", "nepali", "konkani", "maithili", "sanskrit", "dogri", "bodo"}:
                return current
            if preferred in {"hindi", "marathi", "nepali", "konkani", "maithili", "sanskrit", "dogri", "bodo"}:
                return preferred
            return "hindi"
        if 0x0980 <= codepoint <= 0x09FF:
            if current in {"bengali", "assamese", "meitei"}:
                return current
            if preferred in {"bengali", "assamese", "meitei"}:
                return preferred
            return "bengali"
        if 0x0A00 <= codepoint <= 0x0A7F:
            return "punjabi"
        if 0x0A80 <= codepoint <= 0x0AFF:
            return "gujarati"
        if 0x0B00 <= codepoint <= 0x0B7F:
            return "odia"
        if 0x0B80 <= codepoint <= 0x0BFF:
            return "tamil"
        if 0x0C00 <= codepoint <= 0x0C7F:
            return "telugu"
        if 0x0C80 <= codepoint <= 0x0CFF:
            return "kannada"
        if 0x0D00 <= codepoint <= 0x0D7F:
            return "malayalam"
        if 0xABC0 <= codepoint <= 0xABFF:
            return "meitei"
        if 0x1C50 <= codepoint <= 0x1C7F:
            return "santali"
        if (
            0x0600 <= codepoint <= 0x06FF
            or 0x0750 <= codepoint <= 0x077F
            or 0x08A0 <= codepoint <= 0x08FF
        ):
            if current in {"urdu", "sindhi", "kashmiri"}:
                return current
            if preferred in {"urdu", "sindhi", "kashmiri"}:
                return preferred
            return "urdu"
    return preferred if preferred != DEFAULT_LANGUAGE_ID else current


def fallback_language_advice(
    transcript: str,
    current_language_id: str | None,
    preferred_language_id: str | None,
    transcript_quality: str,
) -> dict[str, Any]:
    current = language_option(current_language_id)
    preferred = language_option(preferred_language_id)
    if transcript_quality == "suspect":
        return {
            "detected_language_id": current["id"],
            "suggested_language_id": preferred["id"],
            "transcription_language_id": preferred["id"],
            "transcript_quality": "suspect",
            "confidence": "low",
            "should_switch": False,
            "nudge": (
                f"The last transcript may be wrong. Stay in {preferred['agent_label']}, apologize briefly, "
                "and ask the customer to repeat or name their preferred language before taking any action."
            ),
            "rationale": "Transcript used unsupported script or looked unreliable.",
        }

    suggested_language_id = language_id_for_script(transcript, current["id"], preferred["id"])
    suggested = language_option(suggested_language_id)
    should_switch = suggested["id"] != current["id"]
    return {
        "detected_language_id": suggested["id"],
        "suggested_language_id": suggested["id"],
        "transcription_language_id": suggested["id"],
        "transcript_quality": transcript_quality,
        "confidence": "medium",
        "should_switch": should_switch,
        "nudge": (
            f"Reply in {suggested['agent_label']} for the next turn."
            if should_switch
            else f"Continue in {current['agent_label']} and keep the turn compact."
        ),
        "rationale": "Fallback heuristics inferred the language from visible script and current call preference.",
    }


def normalize_language_advice(
    raw_advice: dict[str, Any],
    current_language_id: str | None,
    preferred_language_id: str | None,
    transcript_quality: str,
) -> dict[str, Any]:
    current = language_option(current_language_id)
    preferred = language_option(preferred_language_id)
    detected = language_option(raw_advice.get("detected_language_id") or current["id"])
    suggested = language_option(raw_advice.get("suggested_language_id") or preferred["id"] or current["id"])
    transcription_language = language_option(raw_advice.get("transcription_language_id") or suggested["id"])
    confidence = str(raw_advice.get("confidence", "medium")).lower()
    if confidence not in {"low", "medium", "high"}:
        confidence = "medium"
    quality = str(raw_advice.get("transcript_quality", transcript_quality)).lower()
    if quality not in {"good", "unclear", "suspect"}:
        quality = transcript_quality

    return {
        "detected_language_id": detected["id"],
        "suggested_language_id": suggested["id"],
        "transcription_language_id": transcription_language["id"],
        "transcript_quality": quality,
        "confidence": confidence,
        "should_switch": bool(raw_advice.get("should_switch", suggested["id"] != current["id"])),
        "nudge": str(raw_advice.get("nudge", "")).strip()
        or f"Reply in {suggested['agent_label']} for the next turn.",
        "rationale": str(raw_advice.get("rationale", "")).strip()
        or "Language coach did not provide a detailed rationale.",
    }


RENDERABLE_LANGUAGE_IDS = {"english", "hinglish", "hindi", "bengali", "marathi", "tamil"}
DETERMINISTIC_CHAT_MODEL = "deterministic-call-engine"
DETERMINISTIC_SUPERVISOR_MODEL = "deterministic-supervisor"
DETERMINISTIC_LANGUAGE_COACH_MODEL = "deterministic-language-coach"
MAX_PROCESSED_USAGE_EVENT_IDS = 4096
MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


def supported_render_language_id(language_id: str | None) -> str:
    candidate = language_option(language_id)["id"]
    return candidate if candidate in RENDERABLE_LANGUAGE_IDS else "english"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _coerce_confidence(value: Any) -> float | None:
    """Normalize a Sarvam confidence value to a 0..1 float, or None if absent /
    unparseable. Accepts numbers (0..1 or 0..100) and string labels."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        conf = float(value)
        if conf > 1.0:  # some APIs report 0..100
            conf = conf / 100.0
        return max(0.0, min(1.0, conf))
    label = str(value).strip().lower()
    return {"high": 0.95, "medium": 0.6, "low": 0.3}.get(label)


# Language codes that render in a native (non-Latin) script. en-IN is the only
# Latin-target code. hi-IN is shared by hindi (native) and hinglish (code-mix);
# the caller disambiguates via language_id. ElevenLabs Flash multilingual handles
# code-switch natively, so we no longer transliterate Latin runs.
_NATIVE_TTS_LANGUAGE_CODES = frozenset(
    code for code in LANGUAGE_CODES.values() if code != "en-IN"
)


def _spoken_iso_date_numeric(match: re.Match[str]) -> str:
    """ISO date -> 'day month year' as plain digits, e.g. 31 1 2026. Bulbul reads
    digits correctly in any language; this avoids the awful double-mangling that
    happens when English month/ordinal WORDS are later transliterated to Devanagari."""
    year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31):
        return match.group(0)
    return f"{day} {month} {year}"


_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _two_digit_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("-" + _ONES[ones] if ones else "")


def _indian_number_words(n: int) -> str:
    """Spell an integer in the Indian numbering system (lakh/crore) as words, so
    the voice says 'fifty-seven thousand nine hundred twenty' instead of mangling
    a digit string. Caps at crore for our invoice-sized amounts."""
    if n == 0:
        return "zero"
    parts: list[str] = []
    crore, n = divmod(n, 10_000_000)
    lakh, n = divmod(n, 100_000)
    thousand, n = divmod(n, 1000)
    hundred, rest = divmod(n, 100)
    if crore:
        parts.append(_indian_number_words(crore) + " crore")
    if lakh:
        parts.append(_two_digit_words(lakh) + " lakh")
    if thousand:
        parts.append(_two_digit_words(thousand) + " thousand")
    if hundred:
        parts.append(_ONES[hundred] + " hundred")
    if rest:
        parts.append(_two_digit_words(rest))
    # Join groups with commas, not spaces: the comma is a brief spoken pause that
    # keeps the voice from blurring compound words (e.g. "fifty-seven" misheard as
    # "seventy-seven") and machine-gunning the groups together.
    return ", ".join(parts)


# Map non-ASCII (Devanagari/Bengali/Tamil/etc.) digits to ASCII so number
# detection + word conversion works regardless of the surrounding script.
_DIGIT_TRANSLATION = {}
for _base in (0x0966, 0x09E6, 0x0BE6, 0x0C66, 0x0CE6, 0x0A66, 0x0AE6, 0x0B66, 0x0D66):  # Deva, Beng, Tamil, Telugu, Kannada, Gurmukhi, Gujarati, Oriya, Malayalam
    for _d in range(10):
        _DIGIT_TRANSLATION[_base + _d] = ord("0") + _d


def _to_ascii_digits(text: str) -> str:
    return text.translate(_DIGIT_TRANSLATION)


# Currency: ₹57,920 | Rs. 57920 | INR 57,920 | 57920 INR | 57,920 rupees/rupaye
_CURRENCY_RE = re.compile(
    r"(?:(?:₹|Rs\.?|INR|रुपये|रुपए|रुपया|रुपयांचे|रुपयांत|রুপি|ரூபாய்)\s*(?P<amount>\d[\d,]*)"
    r"|(?P<amount2>\d[\d,]*)\s*(?:INR|rupees|rupaye|rs|₹|रुपये|रुपए|रुपया|রুপি|ரூபாய்))",
    re.IGNORECASE,
)
# ISO date 2026-01-31 (reformatted to "day month year" digits for normalization).
_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def _currency_sub(match: re.Match[str]) -> str:
    amt = match.group("amount") or match.group("amount2") or ""
    digits = re.sub(r"[,\s]", "", amt)
    if not digits.isdigit():
        return match.group(0)
    value = int(digits)
    if value > 99_99_99_999:
        return match.group(0)
    # Keep the DIGITS (Indian-grouped for clear phrasing) + "Rupees" as the unit,
    # and let ElevenLabs text normalization speak the number in the sentence's
    # language. Our old English word-spelling ("fifty-seven thousand...") was
    # occasionally misheard by eleven_v3 as "seventy-seven" mid-sentence; digits
    # via normalization are read consistently.
    grouped = _indian_group_digits(value)
    return f"{grouped} Rupees"


def _indian_group_digits(value: int) -> str:
    """Format an integer with Indian thousands separators: 5792000 -> 57,92,000,
    57920 -> 57,920. The commas give a clearer spoken grouping."""
    s = str(value)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    # group head in pairs from the right (Indian system)
    parts = []
    while len(head) > 2:
        parts.insert(0, head[-2:])
        head = head[:-2]
    if head:
        parts.insert(0, head)
    return ",".join(parts) + "," + tail


_DIGIT_NAMES =("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def _spell_digits(digits: str) -> str:
    """Speak a digit run digit-by-digit: '123456' -> 'one two three four five six'.
    Used for invoice IDs / account numbers where grouped number words are wrong."""
    return " ".join(_DIGIT_NAMES[int(d)] for d in digits if d.isdigit())


# Invoice ID / reference: 2+ letters DIRECTLY joined to 3+ digits (e.g.
# "DHL123456"). Must be glued (no space) so we don't swallow "date 2026" or
# "invoices 3". Not followed by date punctuation. Spoken as letters + digits.
_ID_RE = re.compile(r"\b([A-Za-z]{2,})(\d{3,})(?![\d/\-])")
# Phone number: an optional +country code then a run of 8-13 digits (may contain
# spaces/dashes). Spoken digit-by-digit in English so it is NOT read as a giant
# currency-style number ("nine crore...").
_PHONE_RE = re.compile(r"(?<![\w])(\+?\d[\d\s\-]{7,}\d)(?![\w])")


def _id_sub(match: re.Match[str]) -> str:
    return f"{match.group(1)}, {_spell_digits(match.group(2))}"


def _phone_sub(match: re.Match[str]) -> str:
    raw = match.group(1)
    plus = raw.strip().startswith("+")
    digits = re.sub(r"\D", "", raw)
    # Only treat 8-13 digit runs as phone numbers; otherwise leave alone.
    if not (8 <= len(digits) <= 13):
        return match.group(0)
    spoken = _spell_digits(digits)
    return ("plus " + spoken) if plus else spoken


def humanize_spoken_text(text: str, language_code: str | None, language_id: str | None = None) -> str:
    """Make business numbers speak cleanly through eleven_v3.

    Spoken in ENGLISH regardless of sentence language (eleven_v3 mangles these in
    Indic scripts, and they are language-neutral business identifiers):
      - invoice IDs / references  -> "DHL, one two three four five six"
      - phone numbers             -> digit-by-digit ("zero nine four one ...")
      - currency amounts          -> English words + "Rupees" (never "INR"/₹)

    Left in the sentence's own language for ElevenLabs to normalize in-language
    (apply_text_normalization=on + language_code), so Hindi says "teen" not
    "three" and dates read naturally:
      - counts (number of invoices), overdue days, plain numbers
      - dates
    EXCEPTION: for an English sentence everything is English anyway, so convert
    counts/dates/numbers to English words too."""
    text = _to_ascii_digits(text)

    # We only hard-convert the three things ElevenLabs CANNOT infer or gets wrong:
    # phone numbers, invoice IDs, and currency (Indian lakh/crore + forced
    # "Rupees"). Everything else — plain counts ("3 invoices"), overdue days,
    # and DATES — is left as digits and handled by ElevenLabs text normalization
    # (apply_text_normalization=on + language_code), which speaks them correctly
    # in the sentence's language ("teen" in Hindi, "three" in English) without us
    # maintaining number/date spelling tables.

    # Reformat ISO date -> "day month year" digit order so normalization reads it
    # as a date, not three separate numbers. Stays numeric/in-language.
    text = _ISO_DATE_RE.sub(_spoken_iso_date_numeric, text)

    # Phones (long digit runs) -> English digit-by-digit. Before IDs/currency, and
    # after the date reformat so a date isn't mistaken for a phone number.
    text = _PHONE_RE.sub(_phone_sub, text)

    # Invoice IDs / references (DHL123456) -> letters + digit-by-digit.
    text = _ID_RE.sub(_id_sub, text)

    # Currency amounts -> English words + "Rupees" (always; Indian lakh/crore).
    text = _CURRENCY_RE.sub(_currency_sub, text)

    # Collapse an accidental "Rupees Rupees".
    text = re.sub(r"\bRupees(\s+(?:Rupees|rupees|rupaye|रुपये|रुपए|টাকা|ரூபாய்))+", "Rupees", text)

    return text


def prepare_tts_text(
    text: str,
    language_code: str | None,
    language_id: str | None = None,
) -> str:
    """Normalize spoken text for the voice without changing the visible transcript.

    Keeps the policy output intact while making business strings like invoice IDs
    and semicolon-separated lists sound less clipped. No transliteration —
    ElevenLabs Flash multilingual handles English/native code-switch natively.
    """
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return ""

    cleaned = cleaned.replace("•", ". ")
    cleaned = re.sub(r"\s*[;|]\s*", ". ", cleaned)

    if TTS_HUMANIZE:
        cleaned = humanize_spoken_text(cleaned, language_code, language_id)

    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    cleaned = re.sub(r"(?<!\d),(?=\S)", ", ", cleaned)
    cleaned = re.sub(r"([.!?])(?=\S)", r"\1 ", cleaned)

    # Any bare ₹ that survived (no following digits) -> the spoken word.
    cleaned = cleaned.replace("₹", " Rupees ")
    cleaned = re.sub(r"\bNo\.(?=\s*\d)", "number ", cleaned, flags=re.IGNORECASE)

    return normalize_whitespace(cleaned)


# Back-compat alias for call sites not yet renamed.
prepare_sarvam_tts_text = prepare_tts_text


def transcript_entries_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        if role == "user":
            role = "customer"
        text = normalize_whitespace(str(msg.get("text") or msg.get("content") or ""))
        if not text:
            continue
        entries.append({"role": role, "text": text})
    return entries


def collapse_trailing_customer_messages(
    messages: list[dict[str, Any]],
    merged_customer_text: str | None,
) -> list[dict[str, Any]]:
    """When the frontend buffers multiple STT fragments into one logical turn,
    replace the trailing customer fragments with the merged utterance before the
    policy engine inspects the transcript."""
    merged = normalize_whitespace(merged_customer_text or "")
    entries = transcript_entries_from_messages(messages)
    if not merged or not entries:
        return entries

    idx = len(entries)
    trailing: list[str] = []
    while idx > 0 and entries[idx - 1].get("role") == "customer":
        trailing.append(str(entries[idx - 1].get("text") or ""))
        idx -= 1
    if not trailing:
        return entries + [{"role": "customer", "text": merged}]

    joined = normalize_whitespace(" ".join(reversed(trailing)))
    if joined == merged or joined in merged or merged in joined:
        return entries[:idx] + [{"role": "customer", "text": merged}]
    return entries + [{"role": "customer", "text": merged}]


def last_entry(entries: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for entry in reversed(entries):
        if entry.get("role") == role:
            return entry
    return None


def assistant_has_stated_purpose(entries: list[dict[str, Any]]) -> bool:
    for entry in entries:
        if entry.get("role") != "assistant":
            continue
        text = normalize_whitespace(str(entry.get("text") or "")).lower()
        if not text:
            continue
        if (
            "credit account" in text
            or "outstanding" in text
            or "overdue" in text
            or "invoice" in text
            or "इनवॉइस" in text
            or "बकाया" in text
            or "payment pending" in text
        ):
            return True
    return False


def count_entries(entries: list[dict[str, Any]], role: str) -> int:
    return sum(1 for entry in entries if entry.get("role") == role)


def latest_tool_call(tool_calls: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for call in reversed(tool_calls):
        if call.get("name") == name:
            return call
    return None


def customer_display_name(customer: dict[str, Any]) -> str:
    contact = str(customer.get("contact_name") or "").strip()
    if contact:
        return contact
    return "there"


def agent_intro_text(language_id: str, voice: str | None) -> str:
    persona = persona_for_voice(voice)
    name = persona["name"]
    if language_id in {"hinglish", "hindi"}:
        verb = "बोल रही हूँ" if persona["gender"] == "female" else "बोल रहा हूँ"
        return f"मेरा नाम {name} है, मैं DHL Express India से {verb}."
    if language_id == "marathi":
        return f"\u092e\u093e\u091d\u0902 \u0928\u093e\u0935 {name} \u0906\u0939\u0947, \u092e\u0940 DHL Express India \u092e\u0927\u0942\u0928 \u092c\u094b\u0932\u0924 \u0906\u0939\u0947."
    if language_id == "tamil":
        return f"\u0ba8\u0bbe\u0ba9\u0bcd {name}, DHL Express India-\u0bb2\u0bbf\u0bb0\u0bc1\u0ba8\u0bcd\u0ba4\u0bc1 \u0baa\u0bc7\u0b9a\u0bc1\u0b95\u0bbf\u0bb1\u0bc7\u0ba9\u0bcd."
    if language_id == "bengali":
        return f"Amar naam {name}, ami DHL Express India theke bolchi."
    return f"My name is {name} and I am calling from DHL Express India."


def agent_calling_phrase(voice: str | None) -> str:
    return "call कर रही हूँ" if persona_for_voice(voice)["gender"] == "female" else "call कर रहा हूँ"


def reason_probe_text(language_id: str) -> str:
    if language_id in {"hinglish", "hindi"}:
        return "Payment अभी तक hold क्यों है, थोड़ा बता दीजिए. मैं note कर लेता हूँ."
    if language_id == "marathi":
        return "\u0915\u0943\u092a\u092f\u093e \u0935\u093f\u0932\u0902\u092c\u093e\u091a\u0902 \u0915\u093e\u0930\u0923 \u0938\u093e\u0902\u0917\u093e\u0932 \u0915\u093e, \u092e\u094d\u0939\u0923\u091c\u0947 \u092e\u0940 \u0924\u0947 \u0928\u0940\u091f \u0928\u094b\u0902\u0926\u0935\u0942 \u0936\u0915\u0947\u0928?"
    if language_id == "tamil":
        return "\u0baa\u0ba3\u0bae\u0bcd \u0ba4\u0bbe\u0bae\u0ba4\u0bae\u0bbe\u0ba9\u0ba4\u0bb1\u0bcd\u0b95\u0bbe\u0ba9 \u0b95\u0bbe\u0bb0\u0ba3\u0ba4\u0bcd\u0ba4\u0bc8\u0b9a\u0bcd \u0b9a\u0bca\u0bb2\u0bcd\u0bb2 \u0bae\u0bc1\u0b9f\u0bbf\u0baf\u0bc1\u0bae\u0bbe, \u0ba8\u0bbe\u0ba9\u0bcd \u0b9a\u0bb0\u0bbf\u0baf\u0bbe\u0b95 \u0baa\u0ba4\u0bbf\u0bb5\u0bc1 \u0b9a\u0bc6\u0baf\u0bcd\u0baf \u0bb5\u0bc7\u0ba3\u0bcd\u0b9f\u0bc1\u0bae\u0bcd."
    if language_id == "bengali":
        return "Payment deri hocche keno, seta ki ektu bolben jate ami thik bhabe note korte pari?"
    return "May I know the reason for the delay so that I can note it correctly?"


def payment_date_request_text(language_id: str) -> str:
    if language_id in {"hinglish", "hindi"}:
        return "आप payment कब तक release कर पाएँगे? Next 2 business days के अंदर एक clear date बता दीजिए."
    if language_id == "marathi":
        return "\u092a\u0941\u0922\u0940\u0932 2 business days \u092e\u0927\u094d\u092f\u0947 \u0928\u0947\u092e\u0915\u0940 payment date confirm \u0915\u0930\u0942 \u0936\u0915\u093e\u0932 \u0915\u093e?"
    if language_id == "tamil":
        return "\u0b85\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4 2 business days-\u0b95\u0bcd\u0b95\u0bc1\u0bb3\u0bcd \u0b92\u0bb0\u0bc1 exact payment date confirm \u0b9a\u0bc6\u0baf\u0bcd\u0baf \u0bae\u0bc1\u0b9f\u0bbf\u0baf\u0bc1\u0bae\u0bbe?"
    if language_id == "bengali":
        return "Apni ki agami 2 business days er moddhe ekta exact payment date confirm korte parben?"
    return "Could you confirm an exact payment date within the next 2 business days?"


def resolved_status_summary_text(invoices: list[dict[str, Any]], language_id: str) -> str:
    if not any(invoice.get("history") for invoice in invoices):
        return ""
    if language_id in {"hinglish", "hindi"}:
        return "जिन invoices पर पहले issues थे, वे अब resolve हो चुके हैं. अब सिर्फ payment clear होना बाकी है."
    if language_id == "marathi":
        return "\u091c\u094d\u092f\u093e invoices \u0935\u0930 \u0906\u0927\u0940 issues \u0939\u094b\u0924\u0947 \u0924\u0947 resolve \u091d\u093e\u0932\u0947 \u0906\u0939\u0947\u0924, \u092e\u094d\u0939\u0923\u0942\u0928 \u0906\u0924\u093e \u092b\u0915\u094d\u0924 payment pending \u0906\u0939\u0947."
    if language_id == "tamil":
        return "\u0bae\u0bc1\u0ba9\u0bcd\u0ba9\u0bb0\u0bcd \u0b87\u0bb0\u0bc1\u0ba8\u0bcd\u0ba4 issues resolve \u0b86\u0b95\u0bbf\u0bb5\u0bbf\u0b9f\u0bcd\u0b9f\u0ba9, \u0b85\u0ba4\u0ba9\u0bbe\u0bb2\u0bcd \u0b87\u0baa\u0bcd\u0baa\u0bcb\u0ba4\u0bc1 payment \u0bae\u0b9f\u0bcd\u0b9f\u0bc1\u0bae\u0bcd pending \u0b86\u0b95 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1."
    if language_id == "bengali":
        return "Je invoice-gulote age issue chhilo, segulo resolve hoye geche, tai ekhon sudhu payment pending."
    return "The earlier issues on these invoices have already been resolved, so the payments are simply pending now."


def payment_options_text(language_id: str) -> str:
    if language_id == "hinglish":
        return (
            "Payment के लिए दो options हैं: DHL MyBill self-serve portal, "
            "या Virtual Account Number bank transfer."
        )
    if language_id == "hindi":
        return (
            "Payment के लिए दो options हैं: DHL MyBill self-serve portal, "
            "या Virtual Account Number bank transfer."
        )
    if language_id == "marathi":
        return (
            "\u0924\u0941\u092e\u091a\u094d\u092f\u093e\u0938\u093e\u0920\u0940 \u0926\u094b\u0928 approved payment options \u0906\u0939\u0947\u0924: DHL MyBill self-serve portal, "
            "\u0915\u093f\u0902\u0935\u093e Virtual Account Number bank transfer."
        )
    if language_id == "tamil":
        return (
            "\u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bc1\u0b95\u0bcd\u0b95\u0bc1 \u0b87\u0bb0\u0ba3\u0bcd\u0b9f\u0bc1 approved payment options \u0bae\u0b9f\u0bcd\u0b9f\u0bc1\u0bae\u0bcd \u0b89\u0bb3\u0bcd\u0bb3\u0ba9: DHL MyBill self-serve portal, "
            "\u0b85\u0bb2\u0bcd\u0bb2\u0ba4\u0bc1 Virtual Account Number bank transfer."
        )
    if language_id == "bengali":
        return (
            "Apnar jonno duita approved payment option ache: DHL MyBill self-serve portal, "
            "ba Virtual Account Number bank transfer."
        )
    return (
        "You have two approved payment options: the DHL MyBill self-serve portal, "
        "or Virtual Account Number bank transfer."
    )


def format_currency(amount: int | float, currency: str = "INR") -> str:
    return f"{currency} {int(amount):,}"


def invoice_summary_line(invoice: dict[str, Any], language_id: str) -> str:
    amount = format_currency(invoice.get("amount", 0), invoice.get("currency", "INR"))
    overdue_days = int(invoice.get("overdue_days", 0) or 0)
    due_date = str(invoice.get("due_date") or "")
    if language_id == "hinglish":
        return (
            f"Invoice {invoice.get('invoice_no')} का amount {amount} है. "
            f"Due date {due_date} थी, और यह अब {overdue_days} दिन से overdue है."
        )
    if language_id == "hindi":
        return (
            f"Invoice {invoice.get('invoice_no')} का amount {amount} है. "
            f"Due date {due_date} थी, और यह अब {overdue_days} दिन से overdue है."
        )
    if language_id == "marathi":
        return (
            f"Invoice {invoice.get('invoice_no')} {amount} \u091a\u093e \u0906\u0939\u0947, "
            f"\u091c\u094b {overdue_days} \u0926\u093f\u0935\u0938 overdue \u0906\u0939\u0947 \u0906\u0923\u093f due date {due_date} \u0939\u094b\u0924\u0940."
        )
    if language_id == "tamil":
        return (
            f"Invoice {invoice.get('invoice_no')} {amount}, "
            f"\u0b87\u0ba4\u0bc1 {overdue_days} days overdue \u0b86\u0b95 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1, due date {due_date}."
        )
    if language_id == "bengali":
        return (
            f"Invoice {invoice.get('invoice_no')} {amount}, "
            f"eta {overdue_days} din overdue ebong due date chhilo {due_date}."
        )
    return (
        f"Invoice {invoice.get('invoice_no')} is for {amount}, "
        f"which is {overdue_days} days overdue and was due on {due_date}."
    )


def total_summary_text(customer: dict[str, Any], invoices: list[dict[str, Any]], language_id: str) -> str:
    total = format_currency(customer_outstanding(invoices), invoices[0].get("currency", "INR") if invoices else "INR")
    company = str(customer.get("company_name") or "your company")
    if language_id == "hinglish":
        return (
            f"{company} के DHL account पर total {total} outstanding है, "
            f"और {len(invoices)} invoices overdue चल रहे हैं."
        )
    if language_id == "hindi":
        return (
            f"{company} के DHL account पर total {total} outstanding है, "
            f"और {len(invoices)} invoices overdue चल रहे हैं."
        )
    if language_id == "marathi":
        return (
            f"\u092e\u0940 call \u0915\u0930\u0924 \u0906\u0939\u0947 \u0915\u093e\u0930\u0923 {company} \u091a\u094d\u092f\u093e DHL account \u0935\u0930 total {total} \u091a\u0902 outstanding \u0906\u0939\u0947 "
            f"\u0906\u0923\u093f {len(invoices)} invoices overdue \u0906\u0939\u0947\u0924."
        )
    if language_id == "tamil":
        return (
            f"\u0ba8\u0bbe\u0ba9\u0bcd call \u0b9a\u0bc6\u0baf\u0bcd\u0bb5\u0ba4\u0bb1\u0bcd\u0b95\u0bbe\u0ba9 \u0b95\u0bbe\u0bb0\u0ba3\u0bae\u0bcd {company} DHL account-\u0bb2\u0bcd total {total} outstanding \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1 "
            f"\u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd {len(invoices)} invoices overdue \u0b86\u0b95 \u0b89\u0bb3\u0bcd\u0bb3\u0ba9."
        )
    if language_id == "bengali":
        return (
            f"Ami call korchi karon {company}-er DHL account e total {total} outstanding ache "
            f"ebong {len(invoices)} ta invoice overdue."
        )
    return (
        f"The reason for my call is that {company} has a total outstanding of {total} "
        f"across {len(invoices)} overdue invoices."
    )


def bare_overdue_days_text(invoices: list[dict[str, Any]], language_id: str) -> str:
    """Just the overdue-days for each invoice — no amounts, no due dates, no recap.
    For a 'how many days overdue' question."""
    days = [int(inv.get("overdue_days", 0) or 0) for inv in invoices]
    if not days:
        return ""
    d = ", ".join(str(x) for x in days[:-1]) + (" and " if len(days) > 1 else "") + str(days[-1])
    # Use a simple "X, Y and Z days" list; the digits are spoken in-language by
    # ElevenLabs normalization.
    n = len(days)
    if language_id in {"hindi", "hinglish"}:
        parts = ", ".join(f"{x} दिन" for x in days)
        return f"तीनों invoices इतने दिन से overdue हैं: {parts}." if n == 3 else f"{parts} से overdue हैं."
    if language_id == "marathi":
        parts = ", ".join(f"{x} दिवस" for x in days)
        return f"तिन्ही invoices इतके दिवस overdue आहेत: {parts}."
    if language_id == "bengali":
        parts = ", ".join(f"{x} din" for x in days)
        return f"invoice-gulo eto din overdue: {parts}."
    if language_id == "tamil":
        parts = ", ".join(f"{x} days" for x in days)
        return f"invoices {parts} overdue ஆக உள்ளன."
    return f"The invoices are {d} days overdue respectively."


def bare_total_text(customer: dict[str, Any], invoices: list[dict[str, Any]], language_id: str) -> str:
    """Just the outstanding total + invoice count — no 'reason for call' preamble,
    no per-invoice breakdown. For a direct 'what is the total amount' question."""
    total = format_currency(customer_outstanding(invoices), invoices[0].get("currency", "INR") if invoices else "INR")
    n = len(invoices)
    texts = {
        "hinglish": f"Aapke DHL account par total {total} outstanding hai, {n} invoices overdue hain.",
        "hindi": f"आपके DHL account पर total {total} outstanding है, {n} invoices overdue हैं.",
        "marathi": f"तुमच्या DHL account वर एकूण {total} outstanding आहे, {n} invoices overdue आहेत.",
        "bengali": f"আপনার DHL account-এ মোট {total} outstanding আছে, {n} invoice overdue.",
        "tamil": f"உங்கள் DHL account-ல் மொத்தம் {total} outstanding உள்ளது, {n} invoices overdue.",
        "english": f"Your total outstanding is {total} across {n} overdue invoices.",
    }
    return texts.get(language_id, texts["english"])


def opening_purpose_text(
    customer: dict[str, Any],
    invoices: list[dict[str, Any]],
    language_id: str,
    voice: str | None,
) -> str:
    total_text = total_summary_text(customer, invoices, language_id)
    if language_id in {"hinglish", "hindi"}:
        return (
            f"जी, धन्यवाद. मैं आपके pending DHL invoices के बारे में {agent_calling_phrase(voice)}. "
            f"{total_text} {reason_probe_text(language_id)}"
        ).strip()
    if language_id == "marathi":
        return (
            f"\u0927\u0928\u094d\u092f\u0935\u093e\u0926. \u092e\u0940 \u0924\u0941\u092e\u091a\u094d\u092f\u093e pending DHL invoices \u092c\u0926\u094d\u0926\u0932 call \u0915\u0930\u0924 \u0906\u0939\u0947. "
            f"{total_text} {reason_probe_text(language_id)}"
        ).strip()
    if language_id == "tamil":
        return (
            f"\u0ba8\u0ba9\u0bcd\u0bb1\u0bbf. \u0ba8\u0bbe\u0ba9\u0bcd \u0b89\u0b99\u0bcd\u0b95\u0bb3\u0bcd pending DHL invoices \u0baa\u0bb1\u0bcd\u0bb1\u0bbf call \u0b9a\u0bc6\u0baf\u0bcd\u0b95\u0bbf\u0bb1\u0bc7\u0ba9\u0bcd. "
            f"{total_text} {reason_probe_text(language_id)}"
        ).strip()
    if language_id == "bengali":
        return (
            f"Dhonnobad confirm korar jonno. Ami apnar pending DHL invoice niye call korchi. "
            f"{total_text} {reason_probe_text(language_id)}"
        ).strip()
    return (
        f"Thank you for confirming. I am calling about your pending DHL invoices. "
        f"{total_text} {reason_probe_text(language_id)}"
    ).strip()


def resolved_history_text(invoices: list[dict[str, Any]], language_id: str) -> str:
    interesting = [invoice for invoice in invoices if invoice.get("history")]
    if not interesting:
        if language_id == "hindi":
            return "इन invoices पर पहले कोई active dispute नहीं था. अब payment ही pending है."
        if language_id == "hinglish":
            return "In invoices par pehle koi active dispute nahin tha. Ab payment hi pending hai."
        if language_id == "marathi":
            return "\u092f\u093e invoices \u0935\u0930 \u0915\u094b\u0923\u0924\u093e\u0939\u0940 prior dispute logged \u0928\u093e\u0939\u0940. \u0938\u0927\u094d\u092f\u093e \u092b\u0915\u094d\u0924 payment pending \u0906\u0939\u0947."
        if language_id == "tamil":
            return "\u0b87\u0ba8\u0bcd\u0ba4 invoices-\u0b95\u0bcd\u0b95\u0bc1 prior dispute \u0b8f\u0ba4\u0bc1\u0bae\u0bcd logged \u0b87\u0bb2\u0bcd\u0bb2\u0bc8. payment \u0bae\u0b9f\u0bcd\u0b9f\u0bc1\u0bae\u0bcd pending \u0b86\u0b95 \u0b89\u0bb3\u0bcd\u0bb3\u0ba4\u0bc1."
        if language_id == "bengali":
            return "Ei invoice-gulor upor kono prior dispute nei. Sudhu payment pending."
        return "There are no prior disputes on these invoices. Payment is simply pending."

    lines: list[str] = []
    for invoice in interesting[:2]:
        history = invoice.get("history") or []
        if language_id == "hindi":
            lines.append(
                f"{invoice.get('invoice_no')} पर जो पहले issue था, वो resolve हो चुका है और credit note भी issue हो गया था."
            )
        elif language_id == "hinglish":
            lines.append(
                f"{invoice.get('invoice_no')} par jo pehle issue tha, woh resolve ho chuka hai aur credit note bhi issue ho gaya tha."
            )
        elif language_id == "marathi":
            lines.append(
                f"{invoice.get('invoice_no')} \u0935\u0930 \u0906\u0927\u0940 issue \u0939\u094b\u0924\u093e, \u092a\u0923 \u0924\u094b resolve \u091d\u093e\u0932\u093e \u0906\u0939\u0947 \u0906\u0923\u093f credit note issue \u091d\u093e\u0932\u0947 \u0906\u0939\u0947."
            )
        elif language_id == "tamil":
            lines.append(
                f"{invoice.get('invoice_no')} \u0baa\u0bb1\u0bcd\u0bb1\u0bbf \u0bae\u0bc1\u0ba9\u0bcd\u0ba9\u0bb0\u0bcd issue \u0b87\u0bb0\u0bc1\u0ba8\u0bcd\u0ba4\u0ba4\u0bc1, \u0b86\u0ba9\u0bbe\u0bb2\u0bcd \u0b85\u0ba4\u0bc1 resolve \u0b86\u0b95\u0bbf\u0bb5\u0bbf\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1 \u0bae\u0bb1\u0bcd\u0bb1\u0bc1\u0bae\u0bcd credit note issue \u0b9a\u0bc6\u0baf\u0bcd\u0baf\u0baa\u0bcd\u0baa\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1."
            )
        elif language_id == "bengali":
            lines.append(
                f"{invoice.get('invoice_no')} niye age issue chhilo, kintu seta resolve hoye geche ebong credit note issue hoyeche."
            )
        else:
            lines.append(
                f"On {invoice.get('invoice_no')}, the earlier issue has already been resolved and the credit note was issued."
            )
        if any("confirmed receipt" in str(item).lower() for item in history):
            if language_id == "hindi":
                lines.append(f"{invoice.get('invoice_no')} के लिए आपकी side से credit note receipt भी confirm हो चुकी थी.")
            elif language_id == "hinglish":
                lines.append(f"Aapki side se credit note receipt bhi confirm ho chuki thi for {invoice.get('invoice_no')}.")
            elif language_id == "marathi":
                lines.append(f"{invoice.get('invoice_no')} sathi credit note receipt dekhil confirm \u091d\u093e\u0932\u0940 \u0939\u094b\u0924\u0940.")
            elif language_id == "tamil":
                lines.append(f"{invoice.get('invoice_no')} \u0b95\u0bcd\u0b95\u0bbe\u0ba9 credit note receipt-um confirm \u0b86\u0b95\u0bbf\u0bb5\u0bbf\u0b9f\u0bcd\u0b9f\u0ba4\u0bc1.")
            elif language_id == "bengali":
                lines.append(f"{invoice.get('invoice_no')} er credit note receipt-o confirm kora hoyechhilo.")
            else:
                lines.append(f"Receipt of the credit note was also confirmed for {invoice.get('invoice_no')}.")
    return " ".join(lines)


def count_business_days(start: datetime, end: datetime) -> int:
    if end.date() <= start.date():
        return 0
    days = 0
    cursor = start
    while cursor.date() < end.date():
        cursor = (cursor + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if cursor.weekday() < 5:
            days += 1
    return days


def parse_customer_date(text: str) -> tuple[str | None, datetime | None]:
    lowered = normalize_whitespace(text).lower()
    now = datetime.now(UTC)
    # English + Hindi/Marathi/Hinglish "today" and "tomorrow".
    if "today" in lowered or re.search(r"आज|aaj|आजच", lowered):
        return ("today", now)
    # "kal" is tomorrow in this collections context; परवा/parso = day after.
    if re.search(r"\bday after\b|परसों|parson|परवा|parva", lowered):
        return ("day after tomorrow", now + timedelta(days=2))
    if "tomorrow" in lowered or re.search(r"\bkal\b|कल|उद्या|udya|kaal", lowered):
        return ("tomorrow", now + timedelta(days=1))

    match = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+([a-zA-Z]+)(?:\s+(\d{2,4}))?\b",
        lowered,
    )
    if not match:
        return (None, None)
    day = int(match.group(1))
    month_name = match.group(2).lower()
    month = MONTH_NAME_TO_NUMBER.get(month_name)
    if not month:
        return (None, None)
    year_raw = match.group(3)
    year = int(year_raw) if year_raw else now.year
    if year < 100:
        year += 2000
    try:
        parsed = datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return (match.group(0), None)
    if parsed.date() < now.date() and not year_raw:
        try:
            parsed = datetime(now.year + 1, month, day, tzinfo=UTC)
        except ValueError:
            return (match.group(0), None)
    return (match.group(0), parsed)


def promise_date_is_within_window(target: datetime | None, business_days_limit: int) -> bool:
    if not target:
        return False
    now = datetime.now(UTC)
    return count_business_days(now, target) <= int(business_days_limit)


def invoice_mentioned_in_text(text: str, invoices: list[dict[str, Any]]) -> dict[str, Any] | None:
    lowered = text.lower()
    for invoice in invoices:
        if str(invoice.get("invoice_no", "")).lower() in lowered:
            return invoice
    if "duty" in lowered:
        return next((invoice for invoice in invoices if "duty" in str(invoice.get("invoice_type", "")).lower()), None)
    if "export" in lowered:
        return next((invoice for invoice in invoices if "export" in str(invoice.get("invoice_type", "")).lower()), None)
    if "import" in lowered:
        return next((invoice for invoice in invoices if "import" in str(invoice.get("invoice_type", "")).lower()), None)
    return invoices[0] if invoices else None


def analyze_customer_turn(text: str) -> dict[str, Any]:
    lowered = normalize_whitespace(text).lower()
    return {
        # Customer is busy / wants to be called back. Detected BEFORE acting on
        # is_affirmative so "Auntie is speaking but I'm in a meeting, call me in
        # 5 minutes" is handled as a callback request, not a yes to proceed.
        "call_back_later": bool(re.search(
            r"(?:\b(?:call (?:me )?(?:back|later|after)|call me in|in a meeting|i.?m busy|i am busy|busy right now|not a good time|can you call|reschedule|ring me later|later please)\b"
            r"|बाद में (?:कॉल|फोन|बात)|मीटिंग में|अभी busy|व्यस्त|बाद में call|थोड़ी देर बाद|मीटिंग चल रही|नंतर (?:कॉल|फोन)|मी busy|मीटिंग मध्ये|পরে (?:কল|ফোন)|মিটিং)",
            lowered,
        )),
        "is_affirmative": bool(re.search(
            r"(?:\b(?:yes|yeah|yep|yup|haan|ha|ji|speaking|that.s me|this is he|this is she|correct)\b|हाँ|हां|जी|मैं ही|मै ही|यही|बोल रहा हूँ|बोल रही हूँ|बात कर रहा हूँ|बात कर रही हूँ|तुम .*से बात कर रहे हो)",
            lowered,
        )),
        "why_calling": bool(re.search(
            r"(?:\b(?:you called me|why are you calling|what is this regarding|what is this about|what do you want|what.s this call|kis baare mein)\b|किस बात|किस बारे|क्यों कॉल|क्यों फोन|किसलिए|पेमेंट किस बात|ये कॉल किस बारे)",
            lowered,
        )),
        "payment_options": bool(re.search(
            r"(?:\b(?:payment option|payment method|options|how can i pay|how do i pay|how to pay|where do i pay)\b|पेमेंट ऑप्शन|पेमेंट कैसे|कैसे पेमेंट|कहाँ पेमेंट)",
            lowered,
        )),
        "invoice_copy": bool(re.search(
            r"(?:\b(?:invoice copy|send.*invoice|resend.*invoice|not received|didn.t receive|haven.t received|don.t have the invoice)\b|इनवॉइस नहीं मिला|इनवॉइस भेज|कॉपी भेज)",
            lowered,
        )),
        "resolved_issues": bool(re.search(
            r"(?:\b(?:resolved issue|resolved issues|conflict|dispute history|past dispute|"
            r"any (?:disputes?|issues?)(?: in the past| before| earlier| previously)?|"
            r"(?:were|was) there (?:any )?(?:disputes?|issues?)|"
            r"(?:disputes?|issues?) (?:in the past|before|earlier|previously)|"
            r"had (?:any )?(?:disputes?|issues?)|credit note)\b"
            r"|क्रेडिट नोट|पुराना विवाद|पुराना डिस्प्यूट|पहले वाला issue"
            # past-tense dispute QUESTIONS (asking, not raising): "dispute tha",
            # "koi dispute tha", "past mein dispute", "pehle dispute"
            r"|(?:कोई )?dispute था|dispute था क्या|डिस्प्यूट था|पास्ट में|past में|"
            r"पहले (?:कोई )?(?:dispute|डिस्प्यूट|issue|विवाद)|"
            r"(?:koi |kya koi )?dispute tha|past mein (?:koi )?dispute|pehle (?:koi )?dispute)",
            lowered,
        )),
        "one_at_a_time": bool(re.search(
            r"(?:\b(?:one (?:by one|at a time)|one invoice at a time|line at a time|line by line|slowly|slow down|too fast|one (?:request|thing) (?:at a time|line))\b|एक-एक करके|एक एक करके|लाइन by लाइन|धीरे|एक इनवॉइस)",
            lowered,
        )),
        "already_paid": bool(re.search(
            r"(?:\b(?:already paid|i paid|payment done|payment made|we paid|paid it|paid that|paid this"
            r"|proof of payment|whom (?:do|should) i (?:tell|send|inform)|where (?:do|should) i (?:send|share) (?:the )?(?:proof|receipt))\b"
            r"|पेमेंट कर दिया|भुगतान कर दिया|पहले ही पेमेंट|payment ho gaya|payment kar diya"
            # Marathi: "payment already झालं असेल तर कोणाला सांगायचं" (if already paid, whom to tell)
            r"|payment.*झाल|झालं असेल|झाला असेल|कोणाला सांगाय|कुणाला सांगाय|payment.*केल.*असेल|आधीच payment)",
            lowered,
        )),
        "dispute": bool(re.search(
            r"(?:\b(?:dispute|wrong charge|billing error|price mismatch|delayed shipment|incorrect amount)\b|डिस्प्यूट|गलत चार्ज|गलत amount|बिलिंग गलत|रेट गलत)",
            lowered,
        )),
        "wrong_contact": bool(re.search(
            r"(?:\b(?:not the right person|wrong person|not the right contact|wrong number)\b|गलत नंबर|गलत व्यक्ति|सही व्यक्ति नहीं)",
            lowered,
        )),
        "identity_confusion": bool(re.search(
            r"(?:\b(?:who is anthony|i am mark|i am not anthony|this is mark)\b|मैं एंथनी नहीं|मैं मार्क हूँ|मैं मार्क हूं|एंथनी नहीं)",
            lowered,
        )),
        # Customer asks ONLY for how many days overdue (no amounts/dates). Must be
        # caught before the generic "details" branch which dumps the full recap.
        "overdue_days_query": bool(re.search(
            r"(?:\b(?:how many days|how old|days overdue|how (?:long|many days) overdue|just (?:the )?days|only (?:the )?days)\b"
            r"|कितने दिन|कितने दिनों|दिन कितने|कितना पुराना|कितने din|din kitne|kitne din"
            r"|किती दिवस|दिवस किती)",
            lowered,
        )),
        "cash_flow": bool(re.search(
            r"(?:\b(?:cash flow|no funds|tight on cash|payment cycle|business problem|short on cash|liquidity)\b|पैसे नहीं|फंड नहीं|कैश फ्लो|cash नहीं)",
            lowered,
        )),
        "approval_pending": bool(re.search(
            r"(?:\b(?:approval|approver|po pending|purchase order|internal approval|waiting for approval)\b|अप्रूवल|मंजूरी|approval pending|po pending)",
            lowered,
        )),
        "discount": bool(re.search(r"(?:\b(?:discount|waive|waiver|reduce|reduction)\b|डिस्काउंट|कम कर|रिड्यूस)", lowered)),
        "asks_timeline": bool(re.search(
            r"(?:\b(?:timeline|when do i need to pay|what is my timeline|by when|deadline)\b|कब तक|किस तारीख तक|डेडलाइन)",
            lowered,
        )),
        "will_pay": bool(re.search(
            r"(?:\b(?:i will pay|we will pay|i can pay|we can pay|i.ll pay|we.ll pay|payment (?:will be|can be) released|arrange payment|release payment|pay soon|payment soon)\b|pay kar dunga|pay kar denge|payment kar dunga|payment kar denge|payment release kar dunga|payment release kar denge|कर दूंगा|कर देंगे|पेमेंट कर दूंगा|पेमेंट कर देंगे|भुगतान कर दूंगा|भुगतान कर देंगे)",
            lowered,
        )),
        "refusal": bool(re.search(
            r"(?:\b(?:don.t call me again|cannot pay|can.t pay|no commitment|refuse|won.t pay)\b|पेमेंट नहीं कर सकता|भुगतान नहीं कर सकता|नहीं दूँगा|नहीं दूंगा)",
            lowered,
        )),
        "human_request": bool(re.search(
            r"(?:\b(?:human|live agent|representative|collections executive|real person|talk to (?:a )?person)\b|किसी इंसान|real person|कलेक्शंस executive)",
            lowered,
        )),
        "safety": bool(re.search(r"(?:\b(?:kill myself|suicide|not safe|enemy|tried to kill|hurt myself)\b|आत्महत्या)", lowered)),
        "details": bool(re.search(
            r"(?:\b(?:details|what are the details|tell me more|elaborate|explain)\b|डिटेल्स|डिटेल|बताओ|और बताओ|समझाओ)",
            lowered,
        )),
        "count_invoices": bool(re.search(
            r"(?:\b(?:how many invoice|how many bill|number of invoice|count of invoice|how many are (?:there|outstanding|overdue|pending))\b|कितने इनवॉइस|कितने invoices)",
            lowered,
        )),
        "which_invoice": bool(re.search(
            r"(?:\b(?:which invoice|what invoice|invoice numbers?|list (?:the )?invoices?|all invoices)\b|कौन से इनवॉइस|कौनसे इनवॉइस|इनवॉइस बताओ|इनवॉइसेस बताओ)",
            lowered,
        )),
        "amount_query": bool(re.search(
            r"(?:\b(?:how much|what.s the amount|total amount|total outstanding|what do i owe|how much do i owe"
            r"|total kitna|amount kitna|kitna (?:amount|pending|outstanding|due|total)|kitna hai)\b"
            r"|कितना amount|amount कितना|कितना पेमेंट|कुल कितना|कुल अमाउंट|टोटल कितना|टोटल amount कितना|"
            r"amount कितना|कितना (?:बकाया|due|pending|outstanding)|एकूण किती|किती amount|किती रक्कम)",
            lowered,
        )),
        "repeat_request": bool(re.search(r"(?:\b(?:repeat|say again|come again|pardon|sorry,? what)\b|दुबारा|फिर से|क्या कहा)", lowered)),
        # Customer asks for the human agent / manager / supervisor's NAME or
        # CONTACT details (not a transfer demand). Must be caught before the
        # generic "details"/invoice branches or it recaps invoices instead.
        "agent_contact_request": bool(re.search(
            r"(?:\b(?:manager|supervisor|human agent|your agent|collections executive|contact (?:number|details|info)|phone number|whom (?:do|should) i (?:call|contact))\b"
            r"|मैनेजर|manager|supervisor|सुपरवाइज़र|आपका नाम|आपके manager|manager का (?:नाम|number|नंबर)|contact (?:details|number)|नंबर दे|number दे|किससे बात)",
            lowered,
        )),
    }


def build_tool_call_entry(name: str, args: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    log_tool_action(name, args, result)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "name": name,
        "args": args,
        "result": result,
        "timestamp": utc_now_iso(),
        "status": "completed" if result.get("ok") else "error",
    }


def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    handler = TOOL_HANDLERS.get(name)
    if not handler:
        return {"ok": False, "error": f"Unknown tool {name}"}
    return handler(args)


def ensure_invoice_tool(tool_calls: list[dict[str, Any]], account_number: str) -> list[dict[str, Any]]:
    if latest_tool_call(tool_calls, "get_invoices"):
        return []
    args = {"account_number": account_number}
    result = run_tool("get_invoices", args)
    return [build_tool_call_entry("get_invoices", args, result)]


FAST_DETERMINISTIC_SIGNAL_NAMES = {
    "why_calling",
    "resolved_issues",
    "one_at_a_time",
    "count_invoices",
    "which_invoice",
    "amount_query",
    "overdue_days_query",
    "details",
    "repeat_request",
    "asks_timeline",
    "will_pay",
    "already_paid",
    "dispute",
    "wrong_contact",
    "identity_confusion",
    "cash_flow",
    "approval_pending",
    "discount",
    "human_request",
    "agent_contact_request",
    "refusal",
    "call_back_later",
}


def looks_like_repeat_request(text: str) -> bool:
    lowered = normalize_whitespace(text).lower()
    if not lowered:
        return False
    return bool(re.search(
        r"(?:\b(?:repeat|say again|come again|pardon|sorry,? what|i do(?: not|n't) understand(?: it)?|did(?: not|n't) understand|not clear)\b|दुबारा|फिर से|क्या कहा|समझ नहीं|समझा नहीं|कुछ समझा नहीं|समझ में नहीं आया)",
        lowered,
    ))


def should_use_fast_deterministic_turn(messages: list[dict[str, Any]]) -> bool:
    entries = transcript_entries_from_messages(messages)
    latest_customer = last_entry(entries, "customer")
    if not latest_customer:
        return False

    customer_text = latest_customer["text"]
    signals = analyze_customer_turn(customer_text)
    assistant_turns = count_entries(entries, "assistant")
    raw_date, _ = parse_customer_date(customer_text)
    repeat_like = looks_like_repeat_request(customer_text)
    first_turn_fast = assistant_turns <= 1 and (
        signals["is_affirmative"]
        or signals["why_calling"]
        or signals["wrong_contact"]
        or signals["identity_confusion"]
        or repeat_like
    )

    if first_turn_fast:
        return True
    if raw_date:
        return True
    if repeat_like:
        return True
    return any(signals.get(name) for name in FAST_DETERMINISTIC_SIGNAL_NAMES)


def generate_collections_reply(
    messages: list[dict[str, Any]],
    account_number: str,
    voice: str | None,
    language_advice: dict[str, Any] | None = None,
    prior_tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], str]:
    entries = transcript_entries_from_messages(messages)
    customer = get_customer(account_number) or {}
    invoices = get_invoices(account_number)
    latest_customer = last_entry(entries, "customer")
    if not latest_customer:
        return ("", [], DETERMINISTIC_CHAT_MODEL)

    language_id = supported_render_language_id(
        (language_advice or {}).get("suggested_language_id") or DEFAULT_LANGUAGE_ID
    )
    tool_calls: list[dict[str, Any]] = []
    prior_tool_calls = prior_tool_calls or []
    customer_text = latest_customer["text"]
    signals = analyze_customer_turn(customer_text)
    signals["repeat_request"] = bool(signals.get("repeat_request") or looks_like_repeat_request(customer_text))
    constants = get_collections_constants()
    target_invoice = invoice_mentioned_in_text(customer_text, invoices) or (invoices[0] if invoices else {})
    purpose_already_stated = assistant_has_stated_purpose(entries)

    if count_entries(entries, "assistant") == 0:
        contact = customer_display_name(customer) or "the accounts payable contact"
        if language_id == "hindi":
            text = f"Good day, {agent_intro_text(language_id, voice)} क्या मैं {contact} से बात कर रहा हूँ?"
        elif language_id == "hinglish":
            text = f"Good day, {agent_intro_text(language_id, voice)} Kya main {contact} se baat kar raha hoon?"
        elif language_id == "marathi":
            text = f"Good day, {agent_intro_text(language_id, voice)} {contact} \u092f\u093e\u0902\u091a\u094d\u092f\u093e\u0938\u094b\u092c\u0924 \u092e\u0940 \u092c\u094b\u0932\u0924 \u0906\u0939\u0947 \u0915\u093e?"
        elif language_id == "tamil":
            text = f"Good day, {agent_intro_text(language_id, voice)} {contact} \u0b89\u0b9f\u0ba9\u0bcd \u0baa\u0bc7\u0b9a\u0bc1\u0b95\u0bbf\u0bb1\u0bc7\u0ba9\u0bbe?"
        elif language_id == "bengali":
            text = f"Good day, {agent_intro_text(language_id, voice)} Ami ki {contact}-er sathe kotha bolchi?"
        else:
            text = f"Good day, {agent_intro_text(language_id, voice)} Am I speaking with {contact}?"
        return (text, tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["safety"]:
        args = {
            "reason": "Customer expressed serious distress or safety concern during the collections call.",
            "customer_summary": customer_text,
        }
        result = run_tool("transfer_to_human", args)
        tool_calls.append(build_tool_call_entry("transfer_to_human", args, result))
        if language_id == "hindi":
            return (
                "मुझे बहुत अफ़सोस है यह सुन कर. आपकी safety सबसे important है, "
                "इसलिए मैं अभी इस call को तुरंत human team को escalate कर रहा हूँ.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                "Mujhe bahut afsos hai yeh sun kar. Aapki safety sabse important hai, "
                "isliye main abhi is call ko turant human team ko escalate kar raha hoon.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            "I am very sorry to hear that. Your safety matters most, so I am escalating this call to a human team immediately.",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    # Customer is busy / asked to be called back. Acknowledge and offer to
    # reschedule rather than pushing collections. Do NOT treat as affirmative.
    if signals["call_back_later"]:
        callback = {
            "english": "No problem, I won't take much of your time. When would be a good time to call you back about the pending DHL invoices?",
            "hinglish": "Koi baat nahi, main aapka zyada time nahi loonga. Pending DHL invoices ke liye main aapko dobara kab call karoon, jo aapke liye theek ho?",
            "hindi": "कोई बात नहीं, मैं आपका ज़्यादा समय नहीं लूँगा। Pending DHL invoices के लिए मैं आपको दोबारा कब call करूँ, जो आपके लिए ठीक हो?",
            "marathi": "काही हरकत नाही, मी जास्त वेळ घेणार नाही. Pending DHL invoices साठी मी तुम्हाला परत कधी call करू, जे तुम्हाला सोयीचं असेल?",
            "bengali": "কোনো সমস্যা নেই, আমি বেশি সময় নেব না। Pending DHL invoices-এর জন্য আপনাকে কখন আবার call করব, যেটা আপনার জন্য সুবিধাজনক?",
            "tamil": "பரவாயில்லை, நான் அதிக நேரம் எடுக்க மாட்டேன். Pending DHL invoices பற்றி உங்களை எப்போது மீண்டும் call செய்யலாம், உங்களுக்கு வசதியான நேரம் எது?",
        }
        message = callback.get(language_id, callback["english"])
        return (message, tool_calls, DETERMINISTIC_CHAT_MODEL)

    # Customer asked for the human agent / manager name + contact. Answer with the
    # details directly — do NOT recap invoices and do NOT trigger a transfer.
    # Checked before the invoice/details branches so "manager ka naam aur contact
    # details bata dijiye" never falls through to an invoice recap.
    if signals["agent_contact_request"]:
        transfer = customer.get("human_transfer") or HUMAN_AGENT
        agent_name = str(transfer.get("name", HUMAN_AGENT["name"]))
        agent_phone = str(transfer.get("phone", HUMAN_AGENT["phone"]))
        contact = {
            "english": f"Of course. My manager is {agent_name}, our Collections Executive, and the contact number is {agent_phone}. You can reach out on this number directly.",
            "hinglish": f"Ji bilkul. Mere manager ka naam {agent_name} hai, woh hamari Collections Executive hain, aur unka contact number {agent_phone} hai. Aap is number par direct call kar sakte hain.",
            "hindi": f"जी बिल्कुल. मेरे manager का नाम {agent_name} है, वे हमारी Collections Executive हैं, और उनका contact number {agent_phone} है. आप इस number पर direct call कर सकते हैं.",
            "marathi": f"हो नक्कीच. माझ्या manager चं नाव {agent_name} आहे, त्या आमच्या Collections Executive आहेत, आणि त्यांचा contact number {agent_phone} आहे. तुम्ही या number वर direct call करू शकता.",
            "bengali": f"অবশ্যই। আমার manager-এর নাম {agent_name}, তিনি আমাদের Collections Executive, এবং তাঁর contact number {agent_phone}. আপনি এই number-এ সরাসরি call করতে পারেন।",
            "tamil": f"கண்டிப்பாக. என் manager பெயர் {agent_name}, அவர் எங்கள் Collections Executive, அவரது contact number {agent_phone}. இந்த number-க்கு நேரடியாக call செய்யலாம்.",
        }
        return (contact.get(language_id, contact["english"]), tool_calls, DETERMINISTIC_CHAT_MODEL)

    if count_entries(entries, "assistant") <= 1 and signals["is_affirmative"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        return (opening_purpose_text(customer, invoices, language_id, voice), tool_calls, DETERMINISTIC_CHAT_MODEL)

    # Recovery guardrail: if the outbound reason for the call has not been
    # stated yet, do not skip straight to asking for a payment date. Recover by
    # stating the purpose or the invoice details first, depending on what the
    # customer asked.
    # A direct "what's the total amount" question gets a concise total ONLY — no
    # per-invoice breakdown, no reason-for-call preamble, no reason probe. Handled
    # before the recovery/recap branches so it isn't expanded into a full dump.
    if signals["amount_query"] and not (signals["overdue_days_query"] or signals["one_at_a_time"] or signals["which_invoice"] or signals["details"]):
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        return (bare_total_text(customer, invoices, language_id), tool_calls, DETERMINISTIC_CHAT_MODEL)

    # "How many days overdue" — just the days per invoice, no amounts/dates/recap.
    # Wins over amount_query and details (the customer explicitly wants only days).
    if signals["overdue_days_query"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        return (bare_overdue_days_text(invoices, language_id), tool_calls, DETERMINISTIC_CHAT_MODEL)

    # Specific actionable intents have their own dedicated branches below — the
    # recovery guardrail must NOT swallow them into a purpose dump just because
    # purpose-detection thinks the reason wasn't stated yet.
    _specific_intent = any(
        signals.get(k)
        for k in (
            "already_paid", "dispute", "resolved_issues", "invoice_copy", "cash_flow",
            "approval_pending", "will_pay", "asks_timeline", "discount", "payment_options",
            "agent_contact_request", "call_back_later", "refusal", "human_request", "safety",
            "overdue_days_query", "amount_query",
        )
    )
    if not purpose_already_stated and not _specific_intent and not (signals["wrong_contact"] or signals["identity_confusion"]):
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        if signals["one_at_a_time"] or signals["count_invoices"] or signals["which_invoice"] or signals["amount_query"] or signals["details"]:
            lines = " ".join(invoice_summary_line(inv, language_id) for inv in invoices)
            total = total_summary_text(customer, invoices, language_id)
            resolved = resolved_status_summary_text(invoices, language_id)
            return (
                f"{total} {lines} {resolved} {reason_probe_text(language_id)}".strip(),
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (opening_purpose_text(customer, invoices, language_id, voice), tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["wrong_contact"] or signals["identity_confusion"]:
        if language_id == "hindi":
            text = (
                "माफ़ कीजिए confusion के लिए. क्या आप मुझे accounts payable या payments handle करने वाले सही person से connect कर सकते हैं?"
            )
        elif language_id == "hinglish":
            text = (
                "Apologies for the confusion. Kya aap mujhe accounts payable ya payments handle karne wale sahi person se connect kar sakte hain?"
            )
        else:
            text = (
                "Apologies for the confusion. Could you please connect me to the person who handles accounts payable or payments for your company?"
            )
        return (text, tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["why_calling"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        parts = [
            total_summary_text(customer, invoices, language_id),
            resolved_history_text(invoices, language_id),
            invoice_summary_line(target_invoice, language_id) if target_invoice else "",
            (
                "That is why I am calling today, and I would like to understand why payment has not been made yet."
                if language_id == "english"
                else "इसी लिए call किया है. Payment अभी तक clear क्यों नहीं हुई, यह समझना था."
                if language_id == "hindi"
                else "Isi liye call kiya hai. Payment abhi tak clear kyun nahin hui, yeh samajhna tha."
            ),
        ]
        return (" ".join(part for part in parts if part), tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["resolved_issues"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        text = resolved_history_text(invoices, language_id)
        ask = (
            "With those issues resolved, may I ask what is holding the payment back now?"
            if language_id == "english"
            else "अब जब ये issues resolve हो चुके हैं, payment अभी तक hold क्यों है?"
            if language_id == "hindi"
            else "Ab jab yeh issues resolve ho chuke hain, payment abhi tak hold kyun hai?"
        )
        return (f"{text} {ask}", tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["one_at_a_time"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        line = invoice_summary_line(target_invoice, language_id) if target_invoice else ""
        if language_id == "hindi":
            return (
                f"ठीक है, एक-एक करके बताता हूँ. {line} इसके लिए payment कब तक release होगी?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                f"Theek hai, ek-ek karke batata hoon. {line} Iske liye payment kab tak release hogi?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "marathi":
            return (
                f"ठीक आहे, एक-एक करून सांगतो. {line} याची payment तारीख सांगू शकता का?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "bengali":
            return (
                f"ঠিক আছে, একটা একটা করে বলছি। {line} এর payment তারিখ জানাতে পারবেন?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "tamil":
            return (
                f"சரி, ஒவ்வொன்றாக சால்கிறேன். {line} இதற்கான payment தேதியை உறுதி செய்ய முடியுமா?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            f"Sure, one at a time. {line} Could you confirm a payment date for this invoice?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["count_invoices"] or signals["which_invoice"] or signals["amount_query"] or signals["details"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        lines = " ".join(invoice_summary_line(inv, language_id) for inv in invoices)
        total = total_summary_text(customer, invoices, language_id)
        resolved = resolved_status_summary_text(invoices, language_id)
        return (
            f"{total} {lines} {resolved} {reason_probe_text(language_id)}".strip(),
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["repeat_request"]:
        if language_id in {"hinglish", "hindi"}:
            return (
                (
                    "ठीक है, मैं आसान तरीके से दोबारा बताता हूँ. "
                    f"{total_summary_text(customer, invoices, language_id)} "
                    "क्या payment पर कोई approval या internal hold है?"
                ),
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            (
                "No problem, let me keep it simple. "
                f"{total_summary_text(customer, invoices, language_id)} "
                "Is there any approval or internal hold on the payment?"
            ),
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["asks_timeline"]:
        if language_id == "hindi":
            return (
                "ये invoices already overdue हैं. "
                "आप next 2 business days के अंदर एक clear payment date बता दीजिए.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                "Yeh invoices already overdue hain. "
                "Aap next 2 business days ke andar ek clear payment date bata dijiye.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            "As per the agreed terms, these invoices are already overdue. Could you confirm a specific payment date within the next 2 business days?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["discount"]:
        if language_id == "hindi":
            return (
                "Discount approve करने की authority मेरे पास नहीं है. "
                "लेकिन आप payment की एक clear date दे दें, तो मैं वो note कर लेता हूँ. "
                "आप कौनसी date दे सकते हैं?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                "Discount approve karne ka authority mere paas nahin hai. "
                "Lekin aap payment ki ek clear date de dein, toh main woh note kar leta hoon. "
                "Aap kaunsi date de sakte hain?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            "I do not have the authority to offer a discount. However, I can note the payment if you confirm a specific date. Could you share that date?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    # A date only counts as a payment PROMISE when the customer is actually
    # committing — not when "today"/"tomorrow"/a date appears inside a QUESTION
    # ("who were you trying to call today?") or a non-payment statement. Without
    # this guard, "today" in any sentence logged a bogus promise-to-pay.
    raw_date, parsed_date = parse_customer_date(customer_text)
    # Question detection for the promise guard. Use "?" and interrogative WORDS
    # (who/what/when/...), NOT auxiliaries like will/can/is — those appear in
    # commitments ("I WILL pay today") and would wrongly suppress real promises.
    _is_question = bool(
        "?" in customer_text
        or re.search(
            r"\b(?:who|what|when|where|why|how|which|whom|whose|kaun|kab|kahan|kaise|kitna|kitne)\b",
            customer_text.lower(),
        )
        or re.search(r"कौन|क्या|कब|कहाँ|कैसे|कितन|किसे|किसको|कोण|काय|केव्हा|कुठे|कोणाला|कुणाला", customer_text)
    )
    _payment_intent = bool(
        signals.get("will_pay")
        or signals.get("cash_flow")
        or re.search(
            r"\b(?:pay|payment|clear|release|transfer|settle|kar dunga|kar dungi|de dunga|de dungi)\b"
            r"|पेमेंट|भुगतान|कर दूँगा|कर दूंगा|दे दूँगा|भरतो|भरेन|करेन",
            customer_text.lower(),
        )
    )
    if raw_date and _payment_intent and not _is_question:
        if promise_date_is_within_window(parsed_date, int(constants["promise_date_max_business_days"])):
            args = {
                "account_number": account_number,
                "invoice_no": target_invoice.get("invoice_no"),
                "promise_date": raw_date,
                "notes": customer_text,
            }
            result = run_tool("log_promise_to_pay", args)
            tool_calls.append(build_tool_call_entry("log_promise_to_pay", args, result))
            recap = payment_options_text(language_id)
            if language_id == "hindi":
                return (
                    f"ठीक है, मैंने note कर लिया है कि payment {raw_date} तक release होगी. "
                    f"Please उस date तक payment clear कर दीजिए. {recap}",
                    tool_calls,
                    DETERMINISTIC_CHAT_MODEL,
                )
            if language_id == "hinglish":
                return (
                    f"Theek hai, maine note kar liya hai ki payment {raw_date} tak release hogi. "
                    f"Please us date tak payment clear kar dijiye. {recap}",
                    tool_calls,
                    DETERMINISTIC_CHAT_MODEL,
                )
            return (
                f"Thank you. I have noted that payment will be released by {raw_date}. "
                f"Please ensure it is made by then. {recap}",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hindi":
            return (
                f"{raw_date} थोड़ा ज़्यादा दूर लग रहा है. "
                "Next 2 business days के अंदर एक closer date बता दीजिए.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                f"{raw_date} thoda zyada door lag raha hai. "
                "Next 2 business days ke andar ek closer date bata dijiye.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            f"{raw_date} is a bit too far out. Could you confirm a specific date within the next 2 business days?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["will_pay"]:
        return (payment_date_request_text(language_id), tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["already_paid"]:
        args = {
            "invoice_no": target_invoice.get("invoice_no"),
            "reference_number": "",
            "paid_date": "",
        }
        result = run_tool("log_already_paid", args)
        tool_calls.append(build_tool_call_entry("log_already_paid", args, result))
        email = constants["proof_of_payment_email"]
        if language_id == "hindi":
            return (
                f"ठीक है, thank you. Transaction reference number और paid date share कर दीजिए. "
                f"Payment proof {email} पर email कर दीजिए, हम 24 hours के अंदर verify कर लेंगे.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                f"Theek hai, thank you. Transaction reference number aur paid date share kar dijiye. "
                f"Payment proof {email} par email kar dijiye, hum 24 hours ke andar verify kar lenge.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            f"Understood, thank you. Could you share the transaction reference number and paid date? "
            f"Please email the payment proof to {email}, and we will verify it within 24 hours.",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["invoice_copy"]:
        args = {
            "invoice_no": target_invoice.get("invoice_no"),
            "email": customer.get("registered_email"),
        }
        result = run_tool("resend_invoice", args)
        tool_calls.append(build_tool_call_entry("resend_invoice", args, result))
        if language_id == "hindi":
            return (
                f"बिल्कुल. आप पहले DHL MyBill portal पर registered email से login करके invoice देख सकते हैं. "
                f"अगर convenient हो, मैंने {customer.get('registered_email')} पर invoice resend भी trigger कर दिया है. "
                "Invoice मिलते ही please check करके payment arrange कर दीजिए.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                f"Bilkul. Aap pehle DHL MyBill portal par registered email se login karke invoice dekh sakte hain. "
                f"Agar convenient ho, maine {customer.get('registered_email')} par invoice resend bhi trigger kar diya hai. "
                "Invoice milte hi please check karke payment arrange kar dijiye.",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            f"Certainly. You can first check the invoice in the DHL MyBill portal using the registered email. "
            f"I have also triggered a resend to {customer.get('registered_email')}. "
            "Once you receive it, please review it and arrange the payment at the earliest.",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    # Only RAISING a new dispute reaches here. A past-tense question ("was there
    # a dispute on this?", "dispute tha kya?") sets resolved_issues, which is
    # handled earlier and answers from history — never logs a new dispute. Guard
    # again here in case ordering changes.
    if signals["dispute"] and not signals["resolved_issues"]:
        invoice_no = target_invoice.get("invoice_no")
        # Dedup: do not log the same invoice's dispute again if it was already
        # logged earlier in this call. The customer often re-states the dispute
        # across several turns; without this we logged duplicate disputes (and
        # repeated the identical reply) on every restatement.
        already_logged = any(
            isinstance(tc, dict)
            and tc.get("name") == "log_dispute"
            and (tc.get("args") or {}).get("invoice_no") == invoice_no
            for tc in (prior_tool_calls or [])
        ) or any(
            isinstance(tc, dict)
            and tc.get("name") == "log_dispute"
            and (tc.get("args") or {}).get("invoice_no") == invoice_no
            for tc in tool_calls
        )
        if not already_logged:
            args = {
                "invoice_no": invoice_no,
                "reason": customer_text,
                "undisputed_amount": None,
            }
            result = run_tool("log_dispute", args)
            tool_calls.append(build_tool_call_entry("log_dispute", args, result))
        ack = {
            "hinglish": "Samajh gaya. Main isko dispute ke taur par log kar raha hoon aur concerned team ko bhej diya jayega. Agar koi undisputed amount hai, kya aap woh meanwhile clear kar sakte hain?",
            "hindi": "समझ गया. मैं इसको dispute के तौर पर log कर रहा हूँ और concerned team को भेज दिया जाएगा. अगर कोई undisputed amount है, क्या आप वो meanwhile clear कर सकते हैं?",
            "marathi": "समजलं. मी हे dispute म्हणून log करतो आणि संबंधित team कडे पाठवतो. जर काही undisputed amount असेल, तर तुम्ही ती दरम्यान clear करू शकता का?",
            "bengali": "বুঝেছি। আমি এটি dispute হিসেবে log করছি এবং সংশ্লিষ্ট team-এর কাছে পাঠানো হবে। যদি কোনো undisputed amount থাকে, আপনি কি সেটি এর মধ্যে clear করতে পারবেন?",
            "tamil": "புரிகிறது. நான் இதை dispute-ஆக log செய்கிறேன், சம்பந்தப்பட்ட team-க்கு அனுப்பப்படும். ஏதேனும் undisputed amount இருந்தால், அதை இதற்கிடையில் clear செய்ய முடியுமா?",
        }
        message = ack.get(
            language_id,
            "Got it, thanks for flagging that. I've logged it as a dispute and it'll go to the concerned team. If there's any undisputed amount, could you clear that in the meantime?",
        )
        return (message, tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["payment_options"]:
        extra = (
            "If you need the specific Virtual Account Number, I can have the collections desk share it after the call."
            if language_id == "english"
            else "अगर आपको specific Virtual Account Number चाहिए, तो collections desk call के बाद share कर सकता है."
            if language_id == "hindi"
            else "Agar aapko specific Virtual Account Number chahiye, toh collections desk call ke baad share kar sakta hai."
        )
        return (f"{payment_options_text(language_id)} {extra}", tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["cash_flow"]:
        if language_id == "hindi":
            return (
                "समझ सकता हूँ कि cash flow tight हो सकता है. "
                "आप अभी partial payment कर सकते हैं, या full payment के लिए एक clear date दे सकते हैं?"
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        if language_id == "hinglish":
            return (
                "Samajh sakta hoon ki cash flow tight ho sakta hai. "
                "Aap abhi partial payment kar sakte hain, ya full payment ke liye ek clear date de sakte hain?"
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        return (
            "I understand cash flow can be tight. Could you make a partial payment now, or confirm a specific date for the full payment?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["approval_pending"]:
        if language_id == "hindi":
            return (
                "ठीक है. Approver का नाम और expected approval date बता दीजिए. "
                "Invoice already overdue है, इसको please priority दीजिए."
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        if language_id == "hinglish":
            return (
                "Theek hai. Approver ka naam aur expected approval date bata dijiye. "
                "Invoice already overdue hai, isko please priority dijiye."
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        return (
            "Understood. Could you confirm the approver name and the expected approval date? The invoice is already overdue, so I would request that this be prioritised.",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["refusal"] or signals["human_request"]:
        args = {
            "reason": "Customer did not provide a usable payment commitment and needs human follow-up.",
            "customer_summary": customer_text,
        }
        result = run_tool("transfer_to_human", args)
        tool_calls.append(build_tool_call_entry("transfer_to_human", args, result))
        if language_id in {"hinglish", "hindi"}:
            return (
                "Theek hai, main aapki position note kar raha hoon. Payment abhi bhi overdue hai, "
                "isliye yeh case main human collections executive ko follow-up ke liye transfer kar raha hoon."
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        return (
            "I respect your position. The payment remains overdue, so I am transferring this case to a human collections executive for follow-up.",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if language_id in {"hinglish", "hindi"}:
        return (
            reason_probe_text(language_id),
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )
    return (
        reason_probe_text(language_id),
        tool_calls,
        DETERMINISTIC_CHAT_MODEL,
    )


def detect_recent_language_request(transcript: list[dict[str, Any]]) -> str | None:
    for entry in reversed(transcript):
        if entry.get("role") != "customer":
            continue
        requested = explicit_language_request_language_id(str(entry.get("text") or ""))
        if requested:
            return requested
    return None


def deterministic_supervisor_review(payload: dict[str, Any]) -> list[dict[str, Any]]:
    transcript = payload.get("transcript") or []
    if not isinstance(transcript, list):
        return []
    invoices = payload.get("invoices") or []
    if not isinstance(invoices, list):
        invoices = []

    assistant_turn = None
    for entry in reversed(transcript):
        if isinstance(entry, dict) and entry.get("role") == "assistant":
            assistant_turn = str(entry.get("text") or "").strip()
            break
    if not assistant_turn:
        return []

    turn_number = int(payload.get("turn_number", 0) or 0)
    issues: list[dict[str, Any]] = []
    lowered = assistant_turn.lower()

    if re.search(r"\baccount number\b", lowered):
        issues.append(
            make_supervisor_issue(
                {
                    "title": "Asked for account number",
                    "category": "policy",
                    "severity": "high",
                    "evidence": "The agent asked the customer to confirm the account number even though the account is already preloaded.",
                    "suggested_fix": "Do not ask for the account number. Use the preloaded account context directly.",
                },
                turn_number,
            )
        )

    if ("no resolved issues" in lowered or "no conflicts" in lowered) and any(invoice.get("history") for invoice in invoices):
        issues.append(
            make_supervisor_issue(
                {
                    "title": "Ignored known invoice history",
                    "category": "reference",
                    "severity": "high",
                    "evidence": "The agent said there were no resolved issues or conflicts even though SAP invoice history includes prior resolved disputes and credit notes.",
                    "suggested_fix": "When asked about prior issues, summarise the known resolved history instead of saying none exist.",
                },
                turn_number,
            )
        )

    if re.search(r"\b(upi|cheque|credit card|debit card|generic neft)\b", lowered):
        issues.append(
            make_supervisor_issue(
                {
                    "title": "Mentioned forbidden payment method",
                    "category": "policy",
                    "severity": "high",
                    "evidence": "The agent mentioned a payment channel outside DHL MyBill or Virtual Account Number bank transfer.",
                    "suggested_fix": "Offer only DHL MyBill self-serve portal and Virtual Account Number bank transfer.",
                },
                turn_number,
            )
        )

    requested_language = detect_recent_language_request(transcript)
    if requested_language == "english" and not is_plain_english(assistant_turn):
        issues.append(
            make_supervisor_issue(
                {
                    "title": "Missed explicit English switch",
                    "category": "language",
                    "severity": "medium",
                    "evidence": "The customer asked for English, but the next assistant reply still contained mixed-language phrasing.",
                    "suggested_fix": "Make the very next reply 100% English when the customer explicitly requests English.",
                },
                turn_number,
            )
        )

    if re.search(r"\b(let me give you more info|one moment|let me check)\b", lowered) and assistant_turn.endswith(("check", "info", "moment")):
        issues.append(
            make_supervisor_issue(
                {
                    "title": "Trailed off without completing the thought",
                    "category": "other",
                    "severity": "low",
                    "evidence": "The agent ended the turn with an incomplete promise instead of a complete update or question.",
                    "suggested_fix": "Finish the same turn with the actual content or a clear actionable question.",
                },
                turn_number,
            )
        )

    return issues


def collect_customer_requests(transcript: list[dict[str, Any]]) -> list[str]:
    requests: list[str] = []
    seen: set[str] = set()
    for entry in transcript:
        if not isinstance(entry, dict) or entry.get("role") != "customer":
            continue
        text = normalize_whitespace(str(entry.get("text") or ""))
        lowered = text.lower()
        label = ""
        if "one by one" in lowered:
            label = "Asked for invoice details one invoice at a time."
        elif "payment option" in lowered or "how can i pay" in lowered or "what are my options" in lowered:
            label = "Asked for available payment options."
        elif "resolved" in lowered or "conflict" in lowered or "credit note" in lowered:
            label = "Asked about past disputes or resolved issues on the invoices."
        elif "not the right person" in lowered:
            label = "Said they were not the right contact."
        elif "discount" in lowered:
            label = "Asked whether an early payment discount was available."
        elif "invoice copy" in lowered or "not receive" in lowered:
            label = "Requested an invoice copy or said the invoice was not received."
        if label and label not in seen:
            seen.add(label)
            requests.append(label)
    return requests


def deterministic_call_summary(payload: dict[str, Any]) -> dict[str, Any]:
    transcript = payload.get("transcript") or []
    tool_calls = payload.get("tool_calls") or []
    customer = payload.get("customer") or {}
    if not isinstance(transcript, list):
        transcript = []
    if not isinstance(tool_calls, list):
        tool_calls = []

    disposition = "no-outcome"
    if latest_tool_call(tool_calls, "log_promise_to_pay"):
        disposition = "promise-to-pay"
    elif latest_tool_call(tool_calls, "log_already_paid"):
        disposition = "already-paid"
    elif latest_tool_call(tool_calls, "resend_invoice"):
        disposition = "invoice-resend"
    elif latest_tool_call(tool_calls, "log_dispute"):
        disposition = "dispute"
    elif latest_tool_call(tool_calls, "transfer_to_human"):
        disposition = "escalation"
    elif any("not the right person" in str(entry.get("text", "")).lower() for entry in transcript if isinstance(entry, dict)):
        disposition = "wrong-contact"

    customer_requests = collect_customer_requests(transcript)
    agreements: list[str] = []
    agent_commitments: list[str] = []
    follow_ups: list[str] = []
    key_decisions: list[str] = []
    risk_flags: list[str] = []

    ptp_call = latest_tool_call(tool_calls, "log_promise_to_pay")
    if ptp_call:
        promise_date = ptp_call.get("result", {}).get("promise_date") or ptp_call.get("args", {}).get("promise_date")
        agreements.append(f"Customer committed to make payment by {promise_date}.")
        key_decisions.append(f"Promise-to-pay date recorded for {promise_date}.")
    if latest_tool_call(tool_calls, "resend_invoice"):
        agent_commitments.append("Agent triggered an invoice resend to the registered email address.")
        follow_ups.append("Customer should review the resent invoice and arrange payment.")
    if latest_tool_call(tool_calls, "log_already_paid"):
        agreements.append("Customer said the payment has already been made.")
        follow_ups.append("Collections team should verify the proof of payment.")
    dispute_calls = [tc for tc in tool_calls if isinstance(tc, dict) and tc.get("name") == "log_dispute"]
    if dispute_calls:
        reasons_seen: list[str] = []
        invoices_seen: list[str] = []
        dispute_ids: list[str] = []
        for dc in dispute_calls:
            args = dc.get("args") or {}
            result = dc.get("result") or {}
            reason = normalize_whitespace(str(args.get("reason") or result.get("reason") or "")).strip()
            invoice_no = str(args.get("invoice_no") or result.get("invoice_no") or "").strip()
            dispute_id = str(result.get("dispute_id") or "").strip()
            if reason and reason not in reasons_seen:
                reasons_seen.append(reason)
            if invoice_no and invoice_no not in invoices_seen:
                invoices_seen.append(invoice_no)
            if dispute_id and dispute_id not in dispute_ids:
                dispute_ids.append(dispute_id)
        reason_text = "; ".join(reasons_seen) if reasons_seen else "no reason captured"
        invoice_text = ", ".join(invoices_seen) if invoices_seen else "unspecified invoice"
        id_text = f" [ids: {', '.join(dispute_ids)}]" if dispute_ids else ""
        plural = "Disputes" if len(dispute_calls) > 1 else "Dispute"
        key_decisions.append(
            f"{plural} logged on {invoice_text} (customer reason: \"{reason_text}\"){id_text}."
        )
        agreements.append(f"Customer raised a dispute on {invoice_text}: \"{reason_text}\".")
        follow_ups.append(
            f"Concerned team should review and resolve the logged dispute on {invoice_text} (reason: \"{reason_text}\")."
        )
        risk_flags.append(f"Open dispute pending team review on {invoice_text}.")
    transfer_call = latest_tool_call(tool_calls, "transfer_to_human")
    if transfer_call:
        transfer_reason = (
            transfer_call.get("args", {}).get("reason")
            or transfer_call.get("result", {}).get("reason")
            or "no usable payment commitment captured"
        )
        agent_commitments.append("Agent escalated the case to a human collections executive.")
        key_decisions.append(f"Call escalated to human collections executive (reason: {transfer_reason}).")
        follow_ups.append("Human collections should continue the case with full context.")
        risk_flags.append("Call required escalation to a human collections executive.")

    customer_mood = "unknown"
    sentiment = 0
    full_customer_text = " ".join(
        normalize_whitespace(str(entry.get("text") or ""))
        for entry in transcript
        if isinstance(entry, dict) and entry.get("role") == "customer"
    ).lower()
    if re.search(r"\b(kill myself|enemy|don.t call me again|angry|annoyed)\b", full_customer_text):
        customer_mood = "angry"
        sentiment = -2
        risk_flags.append("Customer showed serious distress or hostility during the call.")
    elif re.search(r"\b(not the right person|what\?|confused|who is anthony)\b", full_customer_text):
        customer_mood = "confused"
        sentiment = -1
    elif re.search(r"\b(sure|yes|okay|i can pay)\b", full_customer_text):
        customer_mood = "cooperative"
        sentiment = 1
    else:
        customer_mood = "calm"

    if disposition == "no-outcome":
        follow_ups.append("Collections team should obtain a firm payment commitment in the next follow-up.")
        risk_flags.append("No firm payment commitment was captured on the call.")
    if disposition == "wrong-contact":
        follow_ups.append("Collections team should reach the correct accounts payable contact.")
        risk_flags.append("The call did not reach the right payment contact.")

    headline_map = {
        "promise-to-pay": f"{customer.get('company_name', 'Customer')} committed to a payment date",
        "already-paid": "Customer claimed payment was already made",
        "invoice-resend": "Invoice resend was triggered and payment follow-up remains open",
        "dispute": "Customer raised a dispute that needs team follow-up",
        "wrong-contact": "Call reached the wrong contact for payment follow-up",
        "escalation": "Case was escalated to a human collections executive",
        "refusal": "Customer refused to commit to payment",
        "no-outcome": "Call ended without a firm payment commitment",
    }

    next_action = follow_ups[0] if follow_ups else "Review the transcript and continue the collections workflow."
    return {
        "headline": headline_map.get(disposition, "Collections call completed"),
        "customer_mood": customer_mood,
        "customer_sentiment_score": sentiment,
        "agent_tone_assessment": "Agent remained polite and procedural, and the next steps were kept tied to DHL collections policy.",
        "rapport_built": disposition not in {"wrong-contact"} and sentiment >= -1,
        "agreements": agreements,
        "customer_requests": customer_requests,
        "agent_commitments": agent_commitments,
        "follow_ups": follow_ups,
        "next_action": next_action,
        "key_decisions": key_decisions,
        "disposition": disposition,
        "risk_flags": risk_flags,
    }


def base_agent_ledger() -> dict[str, Any]:
    return {
        "model": REALTIME_MODEL,
        "events": 0,
        "response_usage": {
            "text_input_tokens": 0,
            "text_cached_input_tokens": 0,
            "text_output_tokens": 0,
            "audio_input_tokens": 0,
            "audio_cached_input_tokens": 0,
            "audio_output_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
        "transcription_usage": {
            "model": REALTIME_TRANSCRIPTION_MODEL,
            "audio_input_tokens": 0,
            "text_input_tokens": 0,
            "text_output_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def base_supervisor_ledger() -> dict[str, Any]:
    return {
        "model": DETERMINISTIC_SUPERVISOR_MODEL,
        "events": 0,
        "text_input_tokens": 0,
        "text_cached_input_tokens": 0,
        "text_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def base_language_coach_ledger() -> dict[str, Any]:
    return {
        "model": DETERMINISTIC_LANGUAGE_COACH_MODEL,
        "events": 0,
        "text_input_tokens": 0,
        "text_cached_input_tokens": 0,
        "text_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def base_chat_ledger() -> dict[str, Any]:
    return {
        "model": DETERMINISTIC_CHAT_MODEL,
        "events": 0,
        "text_input_tokens": 0,
        "text_cached_input_tokens": 0,
        "text_output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }


def default_board() -> dict[str, Any]:
    return {
        "columns": [
            {"id": "new", "title": "New", "issues": []},
            {"id": "reviewing", "title": "Reviewing", "issues": []},
            {"id": "accepted", "title": "Accepted", "issues": []},
            {"id": "dismissed", "title": "Dismissed", "issues": []},
        ],
        "updated_at": utc_now_iso(),
    }


def default_ledger(
    realtime_model: str | None = None,
    transcription_model: str | None = None,
) -> dict[str, Any]:
    agent = base_agent_ledger()
    if realtime_model:
        agent["model"] = str(realtime_model)
    if transcription_model:
        agent["transcription_usage"]["model"] = str(transcription_model)

    return {
        "agent": agent,
        "supervisor": base_supervisor_ledger(),
        "language_coach": base_language_coach_ledger(),
        "chat_agent": base_chat_ledger(),
        "processed_usage_event_ids": [],
        "session_id": f"cost_session_{uuid.uuid4().hex[:12]}",
        "updated_at": utc_now_iso(),
        "price_table_version": PRICE_TABLE_VERSION,
    }


def ensure_state() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    if not BOARD_FILE.exists():
        write_json(BOARD_FILE, default_board())
    if not LEDGER_FILE.exists():
        write_json(LEDGER_FILE, default_ledger())
    for path in (CALL_LOG_FILE, SUPERVISOR_FLAGS_FILE, TOOL_LOG_FILE):
        if not path.exists():
            path.write_text("", encoding="utf-8")
    # Regenerate the canonical GROUND_TRUTH.md from sap_mock.json on startup so
    # the doc the LLM sees never drifts out of sync with the underlying SAP
    # fixture. Auto-regen is idempotent and cheap.
    try:
        import importlib.util

        script_path = BASE_DIR / "scripts" / "generate_ground_truth.py"
        spec = importlib.util.spec_from_file_location("generate_ground_truth", script_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.regenerate_ground_truth_doc()
            load_ground_truth_doc.cache_clear()
    except Exception:  # noqa: BLE001
        # If regen fails for any reason, fall back to whatever doc is already
        # on disk. Never block server startup on this.
        pass


def load_price_table() -> dict[str, dict[str, float]]:
    price_table = deepcopy(DEFAULT_PRICE_TABLE)

    # Bulk JSON override.
    raw_override = os.environ.get("MODEL_PRICE_TABLE_JSON")
    if raw_override:
        try:
            override = json.loads(raw_override)
            for model_name, metrics in override.items():
                if not isinstance(metrics, dict):
                    continue
                bucket = price_table.setdefault(model_name, {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        bucket[key] = float(value)
        except json.JSONDecodeError:
            pass

    # Per-rate env overrides matching the format suggested in POC_TECHNICAL_PLAN.md.
    # Format: PRICE_<MODEL>_<RATE>_PER_1M, e.g. PRICE_GPT_REALTIME_AUDIO_INPUT_PER_1M.
    env_aliases = {
        "PRICE_GPT_REALTIME_TEXT_INPUT_PER_1M": ("gpt-realtime", "text_input_per_million"),
        "PRICE_GPT_REALTIME_TEXT_CACHED_INPUT_PER_1M": ("gpt-realtime", "text_cached_input_per_million"),
        "PRICE_GPT_REALTIME_TEXT_OUTPUT_PER_1M": ("gpt-realtime", "text_output_per_million"),
        "PRICE_GPT_REALTIME_AUDIO_INPUT_PER_1M": ("gpt-realtime", "audio_input_per_million"),
        "PRICE_GPT_REALTIME_AUDIO_CACHED_INPUT_PER_1M": ("gpt-realtime", "audio_cached_input_per_million"),
        "PRICE_GPT_REALTIME_AUDIO_OUTPUT_PER_1M": ("gpt-realtime", "audio_output_per_million"),
        "PRICE_GPT_4O_TRANSCRIBE_AUDIO_INPUT_PER_1M": ("gpt-4o-transcribe", "audio_input_per_million"),
        "PRICE_GPT_4O_TRANSCRIBE_TEXT_INPUT_PER_1M": ("gpt-4o-transcribe", "text_input_per_million"),
        "PRICE_GPT_4O_TRANSCRIBE_TEXT_OUTPUT_PER_1M": ("gpt-4o-transcribe", "text_output_per_million"),
        "PRICE_GPT_4O_MINI_TRANSCRIBE_AUDIO_INPUT_PER_1M": ("gpt-4o-mini-transcribe", "audio_input_per_million"),
        "PRICE_GPT_4O_MINI_TRANSCRIBE_TEXT_INPUT_PER_1M": ("gpt-4o-mini-transcribe", "text_input_per_million"),
        "PRICE_GPT_4O_MINI_TRANSCRIBE_TEXT_OUTPUT_PER_1M": ("gpt-4o-mini-transcribe", "text_output_per_million"),
        "PRICE_GPT_4_1_MINI_INPUT_PER_1M": ("gpt-4.1-mini", "text_input_per_million"),
        "PRICE_GPT_4_1_MINI_CACHED_INPUT_PER_1M": ("gpt-4.1-mini", "text_cached_input_per_million"),
        "PRICE_GPT_4_1_MINI_OUTPUT_PER_1M": ("gpt-4.1-mini", "text_output_per_million"),
    }
    for env_key, (model_name, rate_key) in env_aliases.items():
        raw = os.environ.get(env_key)
        if raw is None:
            continue
        try:
            price_table.setdefault(model_name, {})[rate_key] = float(raw)
        except ValueError:
            continue

    return price_table


COST_DEBUG = os.environ.get("COST_DEBUG", "").lower() in {"1", "true", "yes"}


def debug_cost(label: str, payload: Any) -> None:
    if not COST_DEBUG:
        return
    try:
        rendered = json.dumps(payload, default=str)[:1500]
    except Exception:
        rendered = str(payload)[:1500]
    print(f"[cost-debug] {label}: {rendered}", flush=True)


PRICE_TABLE = load_price_table()


def price_table_key_for_model(model: str) -> str:
    normalized = str(model or "").strip().lower()
    if not normalized:
        return normalized
    if normalized in PRICE_TABLE:
        return normalized

    for alias, canonical in MODEL_PRICE_ALIASES.items():
        if normalized == alias or normalized.startswith(f"{alias}-"):
            return canonical

    for candidate in sorted(PRICE_TABLE, key=len, reverse=True):
        if normalized == candidate or normalized.startswith(f"{candidate}-"):
            return candidate

    return normalized


def realtime_cost_from_usage(model: str, usage: dict[str, Any]) -> tuple[float, dict[str, int]]:
    pricing = PRICE_TABLE.get(price_table_key_for_model(model), {})
    # OpenAI sends either input_token_details (older) or input_tokens_details (newer GA).
    input_details = usage.get("input_tokens_details") or usage.get("input_token_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("output_token_details") or {}
    # Per-modality cached split now lives under cached_tokens_details on the realtime API.
    cached_details = (
        input_details.get("cached_tokens_details")
        or input_details.get("cached_token_details")
        or {}
    )

    audio_total = int(input_details.get("audio_tokens", 0) or 0)
    text_total = int(input_details.get("text_tokens", 0) or 0)
    audio_cached = int(cached_details.get("audio_tokens", 0) or 0)
    text_cached = int(cached_details.get("text_tokens", 0) or 0)
    cached_total = int(input_details.get("cached_tokens", 0) or 0)

    text_output = int(output_details.get("text_tokens", 0) or 0)
    audio_output = int(output_details.get("audio_tokens", 0) or 0)

    input_total = int(usage.get("input_tokens", 0) or 0)
    output_total = int(usage.get("output_tokens", 0) or 0)

    # Fall back to deriving text totals when only the rollup is provided.
    if not text_total and input_total and audio_total <= input_total:
        text_total = max(input_total - audio_total, 0)
    if not text_output and output_total and audio_output <= output_total:
        text_output = max(output_total - audio_output, 0)

    # Distribute cached_tokens rollup proportionally if per-modality cached was not provided.
    if cached_total and not (audio_cached or text_cached):
        denom = max(audio_total + text_total, 1)
        audio_cached = min(audio_total, round(cached_total * audio_total / denom))
        text_cached = min(text_total, max(cached_total - audio_cached, 0))

    # Clamp so cached never exceeds the modality total (defensive against API drift).
    audio_cached = max(0, min(audio_cached, audio_total))
    text_cached = max(0, min(text_cached, text_total))
    audio_uncached = max(audio_total - audio_cached, 0)
    text_uncached = max(text_total - text_cached, 0)

    total_cost = 0.0
    total_cost += text_uncached * pricing.get("text_input_per_million", 0.0) / 1_000_000
    total_cost += text_cached * pricing.get("text_cached_input_per_million", 0.0) / 1_000_000
    total_cost += audio_uncached * pricing.get("audio_input_per_million", 0.0) / 1_000_000
    total_cost += audio_cached * pricing.get("audio_cached_input_per_million", 0.0) / 1_000_000
    total_cost += text_output * pricing.get("text_output_per_million", 0.0) / 1_000_000
    total_cost += audio_output * pricing.get("audio_output_per_million", 0.0) / 1_000_000

    # Store *uncached* counters so totals never double-count cached subset.
    return total_cost, {
        "text_input_tokens": text_uncached,
        "text_cached_input_tokens": text_cached,
        "text_output_tokens": text_output,
        "audio_input_tokens": audio_uncached,
        "audio_cached_input_tokens": audio_cached,
        "audio_output_tokens": audio_output,
    }


def transcription_cost_from_usage(model: str, usage: dict[str, Any]) -> tuple[float, dict[str, int]]:
    pricing = PRICE_TABLE.get(price_table_key_for_model(model), {})
    input_details = usage.get("input_tokens_details") or usage.get("input_token_details") or {}
    audio_input = int(input_details.get("audio_tokens", 0) or 0)
    text_input = int(input_details.get("text_tokens", 0) or 0)
    if not text_input:
        # Older payloads expose only the rollup; back it out from the modality split.
        input_total = int(usage.get("input_tokens", 0) or 0)
        text_input = max(input_total - audio_input, 0)
    text_output = int(usage.get("output_tokens", 0) or 0)

    total_cost = 0.0
    total_cost += audio_input * pricing.get("audio_input_per_million", 0.0) / 1_000_000
    total_cost += text_input * pricing.get("text_input_per_million", 0.0) / 1_000_000
    total_cost += text_output * pricing.get("text_output_per_million", 0.0) / 1_000_000

    return total_cost, {
        "audio_input_tokens": audio_input,
        "text_input_tokens": text_input,
        "text_output_tokens": text_output,
    }


def text_cost_from_usage(model: str, usage: Any) -> tuple[float, dict[str, int]]:
    pricing = PRICE_TABLE.get(price_table_key_for_model(model), {})
    if hasattr(usage, "to_dict"):
        usage = usage.to_dict()
    elif not isinstance(usage, dict):
        usage = {}

    input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
    output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    input_details = (
        usage.get("input_tokens_details", {})
        or usage.get("input_token_details", {})
        or usage.get("prompt_tokens_details", {})
        or {}
    )
    cached_input = int(input_details.get("cached_tokens", 0) or 0)
    uncached_input = max(input_tokens - cached_input, 0)

    total_cost = 0.0
    total_cost += uncached_input * pricing.get("text_input_per_million", 0.0) / 1_000_000
    total_cost += cached_input * pricing.get("text_cached_input_per_million", 0.0) / 1_000_000
    total_cost += output_tokens * pricing.get("text_output_per_million", 0.0) / 1_000_000

    return total_cost, {
        "text_input_tokens": uncached_input,
        "text_cached_input_tokens": cached_input,
        "text_output_tokens": output_tokens,
    }


def tts_cost_from_chars(model: str, char_count: int) -> tuple[float, dict[str, int]]:
    """ElevenLabs bills per character of output text. We stash the count in
    `text_output_tokens` so the existing ledger summing logic keeps working."""
    chars = max(int(char_count or 0), 0)
    pricing = PRICE_TABLE.get(price_table_key_for_model(model), {})
    rate = float(pricing.get("text_output_per_million", 0.0))
    cost = chars * rate / 1_000_000
    return cost, {"text_output_tokens": chars}


def stt_cost_from_seconds(model: str, seconds: float) -> tuple[float, dict[str, int]]:
    """OpenAI STT bills per minute of mic audio; we meter whole seconds in
    `audio_input_tokens` to reuse the existing ledger schema."""
    secs = max(int(math.ceil(float(seconds or 0.0))), 0)
    pricing = PRICE_TABLE.get(price_table_key_for_model(model), {})
    rate = float(pricing.get("audio_input_per_million", 0.0))
    cost = secs * rate / 1_000_000
    return cost, {"audio_input_tokens": secs}


# Back-compat aliases for call sites not yet renamed.
sarvam_tts_cost_from_chars = tts_cost_from_chars
sarvam_stt_cost_from_seconds = stt_cost_from_seconds


def current_stack_pricing_snapshot(ledger: dict[str, Any] | None = None) -> dict[str, Any]:
    base = load_ledger() if ledger is None else ledger
    priced = ledger_with_combined(recompute_ledger_totals(base))
    agent = priced["agent"]
    chat_agent = priced.get("chat_agent") or base_chat_ledger()
    supervisor = priced["supervisor"]
    language_coach = priced["language_coach"]

    return {
        "price_table_version": PRICE_TABLE_VERSION,
        "stack": {
            "tts_model": ELEVENLABS_TTS_MODEL,
            "stt_model": SARVAM_STT_MODEL,
            "chat_model": CHAT_MODEL,
            "supervisor_model": SUPERVISOR_MODEL,
            "language_coach_model": LANGUAGE_COACH_MODEL,
        },
        "rates": {
            "elevenlabs": {
                "currency": "USD",
                "tts_usd_per_million_chars": {
                    ELEVENLABS_TTS_MODEL: ELEVENLABS_USD_PER_MILLION_CHARS,
                },
            },
            "sarvam_stt": {
                "currency": "INR",
                "inr_per_usd": SARVAM_INR_PER_USD,
                "stt_inr_per_hour": {
                    SARVAM_STT_MODEL: SARVAM_STT_INR_PER_HOUR,
                },
            },
            "openai": {
                CHAT_MODEL: PRICE_TABLE.get(price_table_key_for_model(CHAT_MODEL), {}),
                SUPERVISOR_MODEL: PRICE_TABLE.get(price_table_key_for_model(SUPERVISOR_MODEL), {}),
                LANGUAGE_COACH_MODEL: PRICE_TABLE.get(price_table_key_for_model(LANGUAGE_COACH_MODEL), {}),
            },
        },
        "metering": {
            "tts_field": "agent.response_usage.text_output_tokens",
            "stt_field": "agent.transcription_usage.audio_input_tokens",
            "chat_input_field": "chat_agent.text_input_tokens",
            "chat_cached_input_field": "chat_agent.text_cached_input_tokens",
            "chat_output_field": "chat_agent.text_output_tokens",
            "combined_units_field": "combined.total_tokens",
            "combined_cost_field": "combined.estimated_cost_usd",
        },
        "observed": {
            "session_id": priced.get("session_id", ""),
            "tts_chars": int(agent["response_usage"]["text_output_tokens"]),
            "tts_cost_usd": float(agent["response_usage"]["estimated_cost_usd"]),
            "stt_seconds": int(agent["transcription_usage"]["audio_input_tokens"]),
            "stt_cost_usd": float(agent["transcription_usage"]["estimated_cost_usd"]),
            "chat_input_tokens": int(chat_agent.get("text_input_tokens", 0)),
            "chat_cached_input_tokens": int(chat_agent.get("text_cached_input_tokens", 0)),
            "chat_output_tokens": int(chat_agent.get("text_output_tokens", 0)),
            "chat_cost_usd": float(chat_agent.get("estimated_cost_usd", 0.0)),
            "supervisor_cost_usd": float(supervisor["estimated_cost_usd"]),
            "language_coach_cost_usd": float(language_coach["estimated_cost_usd"]),
            "combined_units": int(priced["combined"]["total_tokens"]),
            "combined_cost_usd": float(priced["combined"]["estimated_cost_usd"]),
        },
        "notes": {
            "dashboard_ground_truth": (
                "The dashboard total is the source of truth for any completed call. "
                "Sarvam speech is metered on actual chars/seconds; OpenAI is metered on actual tokens."
            ),
            "deterministic_components": [
                "Supervisor review is deterministic in the current backend path unless it returns usage.",
                "Language coaching is deterministic in the current backend path unless it returns usage.",
                "Call summary is deterministic in the current backend path unless it returns usage.",
            ],
        },
    }


def load_sap_fixture() -> dict[str, Any]:
    return load_json(SAP_FILE, {})


def get_customer(account_number: str) -> dict[str, Any] | None:
    sap = load_sap_fixture()
    return sap.get("customers", {}).get(account_number)


def get_invoices(account_number: str) -> list[dict[str, Any]]:
    sap = load_sap_fixture()
    return sap.get("invoices", {}).get(account_number, [])


def customer_outstanding(invoices: list[dict[str, Any]]) -> int:
    return int(sum(invoice.get("amount", 0) for invoice in invoices))


def get_payment_methods() -> list[dict[str, Any]]:
    sap = load_sap_fixture()
    return sap.get("payment_methods", []) or []


def render_payment_methods(methods: list[dict[str, Any]]) -> str:
    if not methods:
        return "- (no payment methods on file)"
    return "\n".join(
        f"- {m.get('label', m.get('id', 'method'))}: {m.get('details', '')}" for m in methods
    )


def get_collections_constants() -> dict[str, Any]:
    sap = load_sap_fixture()
    return {
        "proof_of_payment_email": sap.get("proof_of_payment_email", ""),
        "monthly_collection_target_day": sap.get("monthly_collection_target_day", 25),
        "promise_date_max_business_days": sap.get("promise_date_max_business_days", 2),
        "dispositions": sap.get(
            "dispositions",
            ["refusal", "reason", "promise-to-pay", "dispute", "escalation"],
        ),
    }


def log_tool_action(tool_name: str, payload: dict[str, Any], result: dict[str, Any]) -> None:
    append_jsonl(
        TOOL_LOG_FILE,
        {
            "id": f"tool_{uuid.uuid4().hex[:10]}",
            "tool_name": tool_name,
            "payload": payload,
            "result": result,
            "timestamp": utc_now_iso(),
        },
    )


def update_board(issues: list[dict[str, Any]]) -> dict[str, Any]:
    board = load_json(BOARD_FILE, default_board())
    existing_fingerprints = {
        f"{issue.get('turn_number')}::{issue.get('category')}::{issue.get('title')}"
        for column in board.get("columns", [])
        for issue in column.get("issues", [])
    }
    new_column = next((column for column in board["columns"] if column["id"] == "new"), None)
    if not new_column:
        new_column = {"id": "new", "title": "New", "issues": []}
        board["columns"].insert(0, new_column)

    for issue in issues:
        fingerprint = f"{issue.get('turn_number')}::{issue.get('category')}::{issue.get('title')}"
        if fingerprint in existing_fingerprints:
            continue
        new_column["issues"].insert(0, issue)
        existing_fingerprints.add(fingerprint)

    board["updated_at"] = utc_now_iso()
    write_json(BOARD_FILE, board)
    return board


def load_board() -> dict[str, Any]:
    return load_json(BOARD_FILE, default_board())


def merge_missing_defaults(target: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    for key, value in defaults.items():
        if isinstance(value, dict):
            current = target.get(key)
            if not isinstance(current, dict):
                target[key] = deepcopy(value)
                continue
            merge_missing_defaults(current, value)
            continue
        target.setdefault(key, deepcopy(value))
    return target


def load_ledger() -> dict[str, Any]:
    ledger = load_json(LEDGER_FILE, default_ledger())
    return merge_missing_defaults(ledger, default_ledger())


def usage_event_already_recorded(ledger: dict[str, Any], event_id: str | None) -> bool:
    normalized = str(event_id or "").strip()
    if not normalized:
        return False
    return normalized in set(ledger.get("processed_usage_event_ids") or [])


def remember_usage_event(ledger: dict[str, Any], event_id: str | None) -> None:
    normalized = str(event_id or "").strip()
    if not normalized:
        return
    processed = ledger.setdefault("processed_usage_event_ids", [])
    if normalized in processed:
        return
    processed.append(normalized)
    if len(processed) > MAX_PROCESSED_USAGE_EVENT_IDS:
        del processed[:-MAX_PROCESSED_USAGE_EVENT_IDS]


def recompute_ledger_totals(ledger: dict[str, Any]) -> dict[str, Any]:
    normalized = merge_missing_defaults(deepcopy(ledger), default_ledger())
    agent_total = (
        float(normalized["agent"]["response_usage"]["estimated_cost_usd"])
        + float(normalized["agent"]["transcription_usage"]["estimated_cost_usd"])
    )
    normalized["agent"]["estimated_cost_usd"] = round(agent_total, 6)
    normalized["price_table_version"] = PRICE_TABLE_VERSION
    normalized["agent"]["total_tokens"] = (
        sum(
            int(value)
            for key, value in normalized["agent"]["response_usage"].items()
            if key.endswith("_tokens")
        )
        + int(normalized["agent"]["transcription_usage"]["audio_input_tokens"])
        + int(normalized["agent"]["transcription_usage"].get("text_input_tokens", 0))
        + int(normalized["agent"]["transcription_usage"]["text_output_tokens"])
    )
    normalized["supervisor"]["total_tokens"] = (
        int(normalized["supervisor"]["text_input_tokens"])
        + int(normalized["supervisor"]["text_cached_input_tokens"])
        + int(normalized["supervisor"]["text_output_tokens"])
    )
    normalized["language_coach"]["total_tokens"] = (
        int(normalized["language_coach"]["text_input_tokens"])
        + int(normalized["language_coach"]["text_cached_input_tokens"])
        + int(normalized["language_coach"]["text_output_tokens"])
    )
    chat_bucket = normalized.setdefault("chat_agent", base_chat_ledger())
    chat_bucket["total_tokens"] = (
        int(chat_bucket.get("text_input_tokens", 0))
        + int(chat_bucket.get("text_cached_input_tokens", 0))
        + int(chat_bucket.get("text_output_tokens", 0))
    )
    normalized["updated_at"] = utc_now_iso()
    return normalized


def save_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    ledger = recompute_ledger_totals(ledger)
    write_json(LEDGER_FILE, ledger)
    return ledger


def ledger_with_combined(ledger: dict[str, Any]) -> dict[str, Any]:
    chat_agent = ledger.get("chat_agent") or base_chat_ledger()
    combined_cost = round(
        float(ledger["agent"]["estimated_cost_usd"])
        + float(ledger["supervisor"]["estimated_cost_usd"])
        + float(ledger["language_coach"]["estimated_cost_usd"])
        + float(chat_agent.get("estimated_cost_usd", 0.0)),
        6,
    )
    combined_tokens = (
        int(ledger["agent"]["total_tokens"])
        + int(ledger["supervisor"]["total_tokens"])
        + int(ledger["language_coach"]["total_tokens"])
        + int(chat_agent.get("total_tokens", 0))
    )
    return {
        "agent": ledger["agent"],
        "supervisor": ledger["supervisor"],
        "language_coach": ledger["language_coach"],
        "chat_agent": chat_agent,
        "combined": {
            "total_tokens": combined_tokens,
            "estimated_cost_usd": combined_cost,
        },
        "updated_at": ledger["updated_at"],
        "session_id": ledger.get("session_id", ""),
        "price_table_version": PRICE_TABLE_VERSION,
        "price_table": PRICE_TABLE,
    }


def record_agent_response_usage(
    model: str,
    usage: dict[str, Any],
    event_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    ledger = load_ledger()
    if session_id and session_id != ledger.get("session_id"):
        debug_cost(
            "agent.response stale-session skipped",
            {"event_id": event_id, "event_session_id": session_id, "ledger_session_id": ledger.get("session_id")},
        )
        return ledger_with_combined(ledger)
    if usage_event_already_recorded(ledger, event_id):
        debug_cost("agent.response duplicate skipped", {"event_id": event_id})
        return ledger_with_combined(ledger)
    agent = ledger["agent"]
    agent["model"] = model or agent["model"]

    event_cost, token_map = realtime_cost_from_usage(agent["model"], usage)
    debug_cost(
        f"agent.response model={agent['model']}",
        {"raw_usage": usage, "computed_tokens": token_map, "event_cost_usd": event_cost},
    )
    bucket = agent["response_usage"]
    for key, value in token_map.items():
        bucket[key] = int(bucket.get(key, 0)) + int(value)
    bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]) + event_cost, 6)
    agent["events"] = int(agent.get("events", 0)) + 1
    remember_usage_event(ledger, event_id)

    return ledger_with_combined(save_ledger(ledger))


def record_agent_transcription_usage(
    usage: dict[str, Any],
    event_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    ledger = load_ledger()
    if session_id and session_id != ledger.get("session_id"):
        debug_cost(
            "agent.transcription stale-session skipped",
            {"event_id": event_id, "event_session_id": session_id, "ledger_session_id": ledger.get("session_id")},
        )
        return ledger_with_combined(ledger)
    if usage_event_already_recorded(ledger, event_id):
        debug_cost("agent.transcription duplicate skipped", {"event_id": event_id})
        return ledger_with_combined(ledger)
    agent = ledger["agent"]
    bucket = agent["transcription_usage"]

    event_cost, token_map = transcription_cost_from_usage(bucket["model"], usage)
    debug_cost(
        f"agent.transcription model={bucket['model']}",
        {"raw_usage": usage, "computed_tokens": token_map, "event_cost_usd": event_cost},
    )
    for key, value in token_map.items():
        bucket[key] = int(bucket.get(key, 0)) + int(value)
    bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]) + event_cost, 6)
    remember_usage_event(ledger, event_id)

    return ledger_with_combined(save_ledger(ledger))


def record_tts_usage(
    chars: int,
    event_id: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    ledger = load_ledger()
    if session_id and session_id != ledger.get("session_id"):
        debug_cost(
            "tts stale-session skipped",
            {"event_id": event_id, "event_session_id": session_id, "ledger_session_id": ledger.get("session_id")},
        )
        return ledger_with_combined(ledger)
    if usage_event_already_recorded(ledger, event_id):
        debug_cost("tts duplicate skipped", {"event_id": event_id})
        return ledger_with_combined(ledger)
    agent = ledger["agent"]
    agent["model"] = model or ELEVENLABS_TTS_MODEL

    event_cost, token_map = tts_cost_from_chars(agent["model"], chars)
    debug_cost(
        f"tts model={agent['model']}",
        {"chars": chars, "event_cost_usd": event_cost},
    )
    bucket = agent["response_usage"]
    for key, value in token_map.items():
        bucket[key] = int(bucket.get(key, 0)) + int(value)
    bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]) + event_cost, 6)
    agent["events"] = int(agent.get("events", 0)) + 1
    remember_usage_event(ledger, event_id)

    return ledger_with_combined(save_ledger(ledger))


def record_stt_usage(
    seconds: float,
    event_id: str | None = None,
    session_id: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    ledger = load_ledger()
    if session_id and session_id != ledger.get("session_id"):
        debug_cost(
            "stt stale-session skipped",
            {"event_id": event_id, "event_session_id": session_id, "ledger_session_id": ledger.get("session_id")},
        )
        return ledger_with_combined(ledger)
    if usage_event_already_recorded(ledger, event_id):
        debug_cost("stt duplicate skipped", {"event_id": event_id})
        return ledger_with_combined(ledger)
    agent = ledger["agent"]
    bucket = agent["transcription_usage"]
    bucket["model"] = model or SARVAM_STT_MODEL

    event_cost, token_map = stt_cost_from_seconds(bucket["model"], seconds)
    debug_cost(
        f"stt model={bucket['model']}",
        {"seconds": seconds, "event_cost_usd": event_cost},
    )
    for key, value in token_map.items():
        bucket[key] = int(bucket.get(key, 0)) + int(value)
    bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]) + event_cost, 6)
    remember_usage_event(ledger, event_id)

    return ledger_with_combined(save_ledger(ledger))


# Back-compat aliases for call sites not yet renamed.
record_sarvam_tts_usage = record_tts_usage
record_sarvam_stt_usage = record_stt_usage


def record_supervisor_usage(model: str, usage: Any) -> dict[str, Any]:
    ledger = load_ledger()
    supervisor = ledger["supervisor"]
    supervisor["model"] = model or supervisor["model"]

    event_cost, token_map = text_cost_from_usage(supervisor["model"], usage)
    debug_cost(
        f"supervisor model={supervisor['model']}",
        {"raw_usage": str(usage)[:500], "computed_tokens": token_map, "event_cost_usd": event_cost},
    )
    for key, value in token_map.items():
        supervisor[key] = int(supervisor.get(key, 0)) + int(value)
    supervisor["estimated_cost_usd"] = round(float(supervisor["estimated_cost_usd"]) + event_cost, 6)
    supervisor["events"] = int(supervisor.get("events", 0)) + 1

    return ledger_with_combined(save_ledger(ledger))


def record_language_coach_usage(model: str, usage: Any) -> dict[str, Any]:
    ledger = load_ledger()
    language_coach = ledger["language_coach"]
    language_coach["model"] = model or language_coach["model"]

    event_cost, token_map = text_cost_from_usage(language_coach["model"], usage)
    debug_cost(
        f"language_coach model={language_coach['model']}",
        {"raw_usage": str(usage)[:500], "computed_tokens": token_map, "event_cost_usd": event_cost},
    )
    for key, value in token_map.items():
        language_coach[key] = int(language_coach.get(key, 0)) + int(value)
    language_coach["estimated_cost_usd"] = round(
        float(language_coach["estimated_cost_usd"]) + event_cost,
        6,
    )
    language_coach["events"] = int(language_coach.get("events", 0)) + 1

    return ledger_with_combined(save_ledger(ledger))


def record_chat_agent_usage(model: str, usage: Any) -> dict[str, Any]:
    ledger = load_ledger()
    bucket = ledger.setdefault("chat_agent", base_chat_ledger())
    bucket["model"] = model or bucket["model"]

    event_cost, token_map = text_cost_from_usage(bucket["model"], usage)
    debug_cost(
        f"chat_agent model={bucket['model']}",
        {"raw_usage": str(usage)[:500], "computed_tokens": token_map, "event_cost_usd": event_cost},
    )
    for key, value in token_map.items():
        bucket[key] = int(bucket.get(key, 0)) + int(value)
    bucket["estimated_cost_usd"] = round(float(bucket["estimated_cost_usd"]) + event_cost, 6)
    bucket["events"] = int(bucket.get("events", 0)) + 1

    return ledger_with_combined(save_ledger(ledger))


def reset_runtime_state() -> dict[str, Any]:
    write_json(BOARD_FILE, default_board())
    write_json(LEDGER_FILE, default_ledger())
    for path in (CALL_LOG_FILE, SUPERVISOR_FLAGS_FILE, TOOL_LOG_FILE):
        path.write_text("", encoding="utf-8")
    return {
        "board": load_board(),
        "costs": ledger_with_combined(load_ledger()),
    }


def make_supervisor_issue(raw_issue: dict[str, Any], turn_number: int) -> dict[str, Any]:
    severity = str(raw_issue.get("severity", "medium")).lower()
    if severity not in {"low", "medium", "high"}:
        severity = "medium"
    category = str(raw_issue.get("category", "other")).lower()
    title = str(raw_issue.get("title", "Untitled finding")).strip() or "Untitled finding"
    evidence = str(raw_issue.get("evidence", "")).strip()
    suggested_fix = str(raw_issue.get("suggested_fix", "")).strip()

    return {
        "id": f"issue_{uuid.uuid4().hex[:10]}",
        "title": title,
        "category": category,
        "severity": severity,
        "evidence": evidence,
        "suggested_fix": suggested_fix,
        "turn_number": turn_number,
        "status": "new",
        "created_at": utc_now_iso(),
    }


def parse_supervisor_output(raw_text: str, turn_number: int) -> list[dict[str, Any]]:
    parsed = extract_json_payload(raw_text) or {"issues": []}
    issues = parsed.get("issues", [])
    if not isinstance(issues, list):
        return []
    return [make_supervisor_issue(issue, turn_number) for issue in issues if isinstance(issue, dict)]


def create_supervisor_review(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str | None]:
    return deterministic_supervisor_review(payload), None, None


def create_language_coach_review(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    transcript = str(payload.get("transcript", "") or "").strip()
    current_language_id = str(payload.get("current_language_id") or DEFAULT_LANGUAGE_ID)
    preferred_language_id = str(payload.get("preferred_language_id") or DEFAULT_LANGUAGE_ID)
    transcript_quality = transcript_quality_signal(transcript)

    if not transcript:
        return (
            fallback_language_advice("", current_language_id, preferred_language_id, "unclear"),
            None,
            None,
        )

    if is_likely_stt_hallucination(transcript):
        advice = fallback_language_advice(transcript, current_language_id, preferred_language_id, "suspect")
        advice["nudge"] = (
            "Last user turn looks like a transcription hallucination, not real speech. "
            "Do NOT respond to it. Stay quiet and wait for the customer to actually speak."
        )
        advice["rationale"] = "STT echoed prompt vocabulary on silence; dropping turn."
        return (advice, None, None)

    explicit_request_language_id = explicit_language_request_language_id(transcript)
    if explicit_request_language_id:
        explicit_request_language_id = supported_render_language_id(explicit_request_language_id)
        return (
            explicit_language_advice(
                explicit_request_language_id,
                current_language_id,
                transcript_quality if transcript_quality in {"good", "unclear"} else "good",
            ),
            None,
            None,
        )

    # Deterministic lock: if customer turn is plain English (Latin-only, no Hinglish
    # tokens), force agent to reply in English. Skips LLM call; cannot drift to Hinglish.
    if is_plain_english(transcript):
        return (
            inferred_language_advice(
                "english",
                current_language_id,
                transcript_quality if transcript_quality in {"good", "unclear"} else "good",
            ),
            None,
            None,
        )

    if transcript_quality == "suspect":
        return (
            fallback_language_advice(transcript, current_language_id, preferred_language_id, transcript_quality),
            None,
            None,
        )
    advice = fallback_language_advice(
        transcript,
        current_language_id,
        preferred_language_id,
        transcript_quality,
    )
    advice["suggested_language_id"] = supported_render_language_id(advice.get("suggested_language_id"))
    advice["transcription_language_id"] = advice["suggested_language_id"]
    return advice, None, None


LLM_TURN_TOOLS = {
    "log_promise_to_pay",
    "log_already_paid",
    "resend_invoice",
    "log_dispute",
    "update_contact",
    "transfer_to_human",
    "get_invoices",
    "get_customer",
}

LLM_COLLECTIONS_SYSTEM = """You are the DHL Express India payment follow-up agent for an outbound call.
Persona name and voice are provided per turn.

The user message will contain a CANONICAL GROUND TRUTH DOCUMENT followed by a LIVE GROUNDED CONTEXT block. Together they are the ONLY source of truth for this call. Treat them as immutable. Do not rely on training-data knowledge of "DHL invoices" or "typical Indian B2B amounts" — only on what is in those two blocks.

HARD RULES (never violate):
- Never invent an invoice number, amount, due date, overdue days, total outstanding, month, year, name, phone number, email, or history line. Only quote values that appear verbatim in the GROUND TRUTH document or LIVE GROUNDED CONTEXT. If you are tempted to round, summarise, or pick a confident-sounding number that is not literally in those blocks, do not — either quote the exact value or omit numbers from the reply.
- If the customer asks "how much do I owe / what's the total / what are the amounts", you must use ONLY the per-invoice amounts and the explicit "Total outstanding" line. Do not blend, average, or invent partial sums.
- The hard prohibitions section of the GROUND TRUTH document is binding — re-read it before producing any turn that mentions a number, date, or name.
- Only two payment methods are sanctioned: DHL MyBill self-serve portal, and Virtual Account Number bank transfer. Never mention UPI, cheque, card, or any other channel.
- For promise-to-pay, accept dates only within the next 2 business days. If the customer offers a date further out, ask politely for a tighter date.
- Do not ask the customer for account number, company name, or registered email — you already have those.
- If the customer asks about resolved issues / past disputes, summarize the history lines verbatim from GROUND TRUTH.
- If the customer is in distress or a safety concern, hand off to human immediately.
- Match the customer's language. If they speak in plain English, reply in English. Hinglish opening is OK; switch fully on explicit request or a clearly English customer turn.
- Keep replies short and natural — one short paragraph max. Do not dump every invoice unless the customer explicitly asks for the full list.
- Never tell the customer you are "calling about collections", a "collections case", or "recovery". Say you are calling about pending invoices, overdue invoices, or pending payment on the DHL account.
- Sound like a live Indian B2B payment follow-up caller, not a translator, legal notice, training script, or chatbot.
- For Hindi/Hinglish, use spoken Indian business language. Keep common business words like payment, invoice, due date, approval, account, portal, hold, clear, release, and date in English script when that sounds more natural.
- Avoid bookish or bureaucratic wording such as "कृपया अवगत कराइए", "संदर्भ में", "उक्त", "भुगतान लंबित है", "निराकरण", "व्यवस्था करें", or "कृपया पुष्टि करें". Prefer spoken phrasing like "बताइए", "date share कर दीजिए", "payment अभी तक hold क्यों है?", "issue resolve हो गया", and "payment clear कर दीजिए".
- Tone target: warm, direct, lightly firm, and a little informal. You are calling to secure a payment commitment, not to educate the customer or read a policy memo.
- Sound like a real phone caller, not a polished corporate announcer. Short everyday phrasing is better than formal phrasing.

OUTPUT (strict JSON, no markdown):
{
  "intent": one of ["greet_identity","state_purpose","explain_invoices","answer_history","payment_options","capture_promise","already_paid","invoice_copy","dispute","cash_flow","approval_pending","wrong_contact","escalate","close","other"],
  "reply": "the exact line the agent will speak, in the chosen language",
  "language": one of ["english","hinglish","hindi","bengali","marathi","tamil"],
  "tool_calls": [ { "name": "<tool>", "args": { ... } } ]   // pick from: log_promise_to_pay, log_already_paid, resend_invoice, log_dispute, update_contact, transfer_to_human. Empty array if no side-effect needed.
}
"""


@lru_cache(maxsize=1)
def load_ground_truth_doc() -> str:
    """Read backend/data/GROUND_TRUTH.md once. This file is the canonical
    source of truth for the LLM system prompt — every name, invoice number,
    amount, date, payment channel, and policy constant the agent may speak
    must come from here. Cached because the file is static for a given
    deployment."""
    try:
        return GROUND_TRUTH_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def build_grounded_context(account_number: str) -> str:
    customer = get_customer(account_number) or {}
    invoices = get_invoices(account_number)
    constants = get_collections_constants()
    methods = get_payment_methods()
    total = customer_outstanding(invoices)
    today = datetime.now(UTC).strftime("%Y-%m-%d")

    invoice_lines = []
    for inv in invoices:
        history = "; ".join(inv.get("history") or []) or "no prior issues logged"
        invoice_lines.append(
            f"- {inv.get('invoice_no')} ({inv.get('invoice_type')}): "
            f"{inv.get('currency','INR')} {inv.get('amount')}, "
            f"{inv.get('overdue_days')} days overdue, due {inv.get('due_date')}. "
            f"History: {history}"
        )

    method_lines = [f"- {m['label']}: {m['details']}" for m in methods]

    transfer = customer.get("human_transfer") or {}
    transfer_line = (
        f"{transfer.get('name')} ({transfer.get('designation')}, {transfer.get('phone')})"
        if transfer
        else "the collections desk"
    )

    return "\n".join([
        f"TODAY: {today}",
        "",
        "GROUND TRUTH — CUSTOMER:",
        f"- Account: {customer.get('account_number')}",
        f"- Company: {customer.get('company_name')}",
        f"- Primary contact: {customer.get('contact_name')}",
        f"- Alternate contact: {customer.get('alternate_contact_name')}",
        f"- Registered email: {customer.get('registered_email')}",
        f"- Phone: {customer.get('phone')}",
        f"- Payment terms: {customer.get('payment_terms')}",
        f"- Languages preferred: {', '.join(customer.get('language_preferences') or [])}",
        "",
        "GROUND TRUTH — INVOICES:",
        *invoice_lines,
        f"Total outstanding: INR {total} across {len(invoices)} invoices.",
        "",
        "SANCTIONED PAYMENT METHODS:",
        *method_lines,
        "",
        "POLICY CONSTANTS:",
        f"- Promise-to-pay window: {constants.get('promise_date_max_business_days')} business days from today.",
        f"- Proof-of-payment email: {constants.get('proof_of_payment_email')}.",
        f"- Soft monthly target: collect before day {constants.get('monthly_collection_target_day')}.",
        f"- Human escalation contact: {transfer_line}.",
        "",
        "COLLECTION NOTES:",
        *[f"- {n}" for n in customer.get("collection_notes") or []],
    ])


@lru_cache(maxsize=16)
def build_llm_grounding_snapshot(account_number: str) -> str:
    customer = get_customer(account_number) or {}
    invoices = get_invoices(account_number)
    constants = get_collections_constants()
    methods = get_payment_methods()
    snapshot = {
        "customer": {
            "account_number": customer.get("account_number"),
            "company_name": customer.get("company_name"),
            "primary_contact": customer.get("contact_name"),
            "alternate_contact": customer.get("alternate_contact_name"),
            "registered_email": customer.get("registered_email"),
            "phone": customer.get("phone"),
            "payment_terms": customer.get("payment_terms"),
            "language_preferences": customer.get("language_preferences") or [],
            "collection_notes": customer.get("collection_notes") or [],
        },
        "invoices": [
            {
                "invoice_no": inv.get("invoice_no"),
                "invoice_type": inv.get("invoice_type"),
                "amount_inr": inv.get("amount"),
                "due_date": inv.get("due_date"),
                "overdue_days": inv.get("overdue_days"),
                "history": inv.get("history") or [],
            }
            for inv in invoices
        ],
        "totals": {
            "invoice_count": len(invoices),
            "total_outstanding_inr": customer_outstanding(invoices),
        },
        "payment_methods": [
            {"label": method.get("label"), "details": method.get("details")}
            for method in methods
        ],
        "policy_constants": {
            "promise_to_pay_max_business_days": constants.get("promise_date_max_business_days"),
            "proof_of_payment_email": constants.get("proof_of_payment_email"),
            "monthly_collection_target_day": constants.get("monthly_collection_target_day"),
            "allowed_payment_methods": ["DHL MyBill", "Virtual Account Number bank transfer"],
        },
    }
    return json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))


def llm_collections_turn(
    messages: list[dict[str, Any]],
    account_number: str,
    voice: str | None,
    language_advice: dict[str, Any] | None,
) -> tuple[str, list[dict[str, Any]], list[Any], str | None]:
    if not OPENAI_CLIENT:
        text, tools_log, _ = generate_collections_reply(
            messages=messages,
            account_number=account_number,
            voice=voice,
            language_advice=language_advice,
        )
        return text, tools_log, [], None

    persona = persona_for_voice(voice)
    suggested = (language_advice or {}).get("suggested_language_id") or "hinglish"
    detected = (language_advice or {}).get("detected_language_id") or suggested
    nudge = (language_advice or {}).get("nudge") or ""

    transcript_lines = []
    for msg in messages:
        role = msg.get("role")
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if role == "customer":
            transcript_lines.append(f"CUSTOMER: {text}")
        elif role == "assistant":
            transcript_lines.append(f"AGENT: {text}")
        elif role == "system":
            transcript_lines.append(f"SYSTEM: {text}")

    grounded = build_grounded_context(account_number)
    compact_grounding = build_llm_grounding_snapshot(account_number)
    language_directive = {
        "english": "HARD LANGUAGE LOCK: reply 100% in English. Zero Hindi/Hinglish/Bengali words. No 'aap', 'main', 'hoon', 'kar', 'kya', 'haan', 'ji', 'namaste'. Use only English script and English vocabulary.",
        "hinglish": (
            "HARD LANGUAGE LOCK: reply in Hindi-dominant code-mix. Hindi words MUST be written in Devanagari script. "
            "Keep brand names and common business words like DHL, MyBill, payment, invoice, due date, approval, account, portal, hold, clear, release, and line by line in English script. "
            "Sound like a live Indian payment follow-up caller on the phone: short, spoken, slightly informal, and natural. "
            "Prefer phrasing like 'payment अभी तक hold क्यों है?', 'date बता दीजिए', 'मैं note कर लेता हूँ', 'देख लीजिए', and 'payment clear कर दीजिए'. "
            "Avoid bookish Hindi like 'कृपया अवगत कराइए', 'संदर्भ में', 'भुगतान लंबित है', or 'कृपया पुष्टि करें'. "
            "Avoid sounding too polished or stiff. Use everyday phone phrasing like 'ठीक है', 'कोई दिक्कत है क्या', 'एक rough date दे दीजिए', and 'मैं note कर लेता हूँ'. "
            "NEVER write romanized Hindi such as 'main', 'aap', 'batao', 'kyun', or 'hoon'."
        ),
        "hindi": (
            "HARD LANGUAGE LOCK: reply in spoken Hindi using Devanagari script. "
            "Keep proper nouns and natural business words like DHL, MyBill, payment, invoice, due date, approval, account, and portal in English script when needed. "
            "Use simple spoken office Hindi, slightly informal, not literary or bureaucratic Hindi. "
            "Prefer phrasing like 'payment अभी तक क्यों रुकी है?', 'date बता दीजिए', 'मैं note कर लेता हूँ', and 'ठीक है, समझ गया'. "
            "Avoid sounding too official or polished. Favor short phone-call wording over formal sentences. "
            "Do not use romanized Hindi like 'main', 'aap', 'batao', or 'kyun'."
        ),
        "bengali": "HARD LANGUAGE LOCK: reply entirely in Bengali. First words must already be Bengali.",
        "marathi": (
            "HARD LANGUAGE LOCK: reply entirely in Marathi using Devanagari script. "
            "Do not drift into Hindi or romanized Marathi. Keep proper nouns like DHL or MyBill in English script only if needed."
        ),
        "tamil": (
            "HARD LANGUAGE LOCK: reply entirely in Tamil using Tamil script. "
            "Do not answer in English first. Keep proper nouns like DHL or MyBill in English script only if needed."
        ),
    }.get(suggested, f"Reply in {suggested}.")

    recent_transcript_lines = transcript_lines[-8:]
    user_prompt = "\n".join([
        f"AGENT PERSONA: {persona['name']} ({persona['gender']}). Voice: {voice or DEFAULT_REALTIME_VOICE}.",
        f"Suggested reply language: {suggested}. Detected customer language: {detected}.",
        language_directive,
        f"Language coach note: {nudge}",
        "",
        "CANONICAL FACTS JSON (derived from the same ground-truth source used by the app):",
        compact_grounding,
        "",
        "LIVE GROUNDED CONTEXT FOR THIS CALL:",
        grounded,
        "",
        "TRANSCRIPT SO FAR:",
        *(recent_transcript_lines or ["(no turns yet — this is the very first agent line)"]),
        "",
        "STYLE TARGET:",
        "- Outbound DHL pending-invoice follow-up call.",
        "- Spoken, not written.",
        "- Warm, lightly firm, and slightly informal.",
        "- No policy-manual Hindi.",
        "- No lecture, no script-reading, no translator tone.",
        "- No overly polished or announcer-like phrasing.",
        "",
        "Produce the next agent turn now as JSON per the schema in the system message.",
        "Reminder: every numeric/name/date you state must appear verbatim in the facts above. Anything else is a fabrication and forbidden.",
    ])

    usage_events: list[Any] = []
    try:
        completion = OPENAI_CLIENT.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": LLM_COLLECTIONS_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=220,
            **_chat_kwargs(CHAT_MODEL, 0.4),
        )
    except Exception as exc:  # noqa: BLE001
        return "", [], [], f"LLM turn failed: {exc}"

    if completion.usage:
        usage_events.append(completion.usage)

    raw = (completion.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "", [], usage_events, f"LLM returned non-JSON: {raw[:200]}"

    reply = str(parsed.get("reply") or "").strip()
    raw_tool_calls = parsed.get("tool_calls") or []
    invoices = get_invoices(account_number)
    customer = get_customer(account_number) or {}
    constants = get_collections_constants()

    reply = scrub_forbidden_payment_methods(reply)
    reply = scrub_invented_invoice_numbers(reply, invoices)

    invented_amount = reply_has_invented_amount(reply, invoices)
    if invented_amount:
        try:
            valid_lines = "\n".join(
                f"- {inv.get('invoice_no')}: INR {int(inv.get('amount') or 0)}" for inv in invoices
            )
            grand_total = sum(int(inv.get("amount") or 0) for inv in invoices)
            retry = OPENAI_CLIENT.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": LLM_COLLECTIONS_SYSTEM},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply contained an invented amount {invented_amount}. "
                            f"The ONLY valid amounts are:\n{valid_lines}\nTotal outstanding: INR {grand_total}.\n"
                            "Rewrite the same reply using only these exact numbers (or omit numbers entirely). "
                            "Keep all other facts and intent identical. Return JSON in the same schema."
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=220,
                **_chat_kwargs(CHAT_MODEL, 0.1),
            )
            if retry.usage:
                usage_events.append(retry.usage)
            retry_raw = (retry.choices[0].message.content or "").strip()
            retry_parsed = json.loads(retry_raw)
            retry_reply = str(retry_parsed.get("reply") or "").strip()
            if retry_reply:
                retry_reply = scrub_forbidden_payment_methods(retry_reply)
                retry_reply = scrub_invented_invoice_numbers(retry_reply, invoices)
                still_invented = reply_has_invented_amount(retry_reply, invoices)
                if not still_invented:
                    reply = retry_reply
                else:
                    # Second pass also hallucinated; fall back to deterministic
                    # regex-tree reply rather than speak a fabricated number.
                    return "", [], usage_events, "LLM produced invented amounts twice"
        except Exception:  # noqa: BLE001
            return "", [], usage_events, "LLM produced invented amount and retry failed"

    if suggested == "english" and reply_violates_english_lock(reply):
        try:
            retry = OPENAI_CLIENT.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": LLM_COLLECTIONS_SYSTEM},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Your previous reply contained Hinglish/Hindi words. Rewrite the same reply 100% in English. Keep all facts and intent identical. Return JSON in the same schema."},
                ],
                response_format={"type": "json_object"},
                max_tokens=220,
                **_chat_kwargs(CHAT_MODEL, 0.2),
            )
            if retry.usage:
                usage_events.append(retry.usage)
            retry_raw = (retry.choices[0].message.content or "").strip()
            retry_parsed = json.loads(retry_raw)
            retry_reply = str(retry_parsed.get("reply") or "").strip()
            if retry_reply:
                retry_reply = scrub_forbidden_payment_methods(retry_reply)
                retry_reply = scrub_invented_invoice_numbers(retry_reply, invoices)
                reply = retry_reply
        except Exception:  # noqa: BLE001
            pass

    latest_customer_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "customer":
            latest_customer_text = (msg.get("text") or "").strip()
            break
    customer_with_ctx = dict(customer)
    customer_with_ctx["__latest_customer_text"] = latest_customer_text

    executed: list[dict[str, Any]] = []
    if isinstance(raw_tool_calls, list):
        for call in raw_tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            args = call.get("args") if isinstance(call.get("args"), dict) else {}
            if name not in LLM_TURN_TOOLS:
                continue
            args = validate_tool_args(name, args, account_number, invoices, customer_with_ctx, constants)
            if args is None:
                continue
            result = run_tool(name, args)
            executed.append(build_tool_call_entry(name, args, result))

    if not reply:
        return "", executed, usage_events, "LLM returned empty reply"

    return reply, executed, usage_events, None


_FORBIDDEN_PAYMENT_SUBSTITUTIONS = (
    (r"\bUPI\b", "DHL MyBill"),
    (r"\bcheques?\b", "DHL MyBill"),
    (r"\bcredit card\b", "DHL MyBill"),
    (r"\bdebit card\b", "DHL MyBill"),
    (r"\bGoogle Pay\b", "DHL MyBill"),
    (r"\bPhonePe\b", "DHL MyBill"),
    (r"\bPaytm\b", "DHL MyBill"),
    # Keep everyday language intact, but rewrite explicit cash/check payment instructions.
    (r"\b(?:payment|payments?)\s+(?:by|via|through|with|using|in)\s+(?:cash|check)\b", "payment via DHL MyBill"),
    (r"\bpay\s+(?:by|via|through|with|using|in)\s+(?:cash|check)\b", "pay via DHL MyBill"),
    (r"\b(?:cash|check)\s+payment\b", "DHL MyBill payment"),
    (r"\b(?:via|through|using)\s+(?:cash|check)\b", "via DHL MyBill"),
)


_HINGLISH_LOCK_TOKENS = re.compile(
    r"\b(aap|aapko|aapke|aapka|main|mein|hoon|hain|kar|karna|karke|karte|karti|karta|"
    r"kya|kyu|kyun|nahi|nahin|haan|ji|namaste|theek|thik|accha|acha|raha|rahi|rahe|"
    r"baat|paisa|paise|abhi|phir|kuch|sahi|baad|pehle|liye|wala|wali|saath|baare|"
    r"din|dino|kal|aaj|kabhi|jab|tab|matlab|samjha|samjhi|bilkul|chal|bata|batao|"
    r"sun|suno|dekh|dekho|hota|hoti|hone|honge|tha|thi|the|hua|hui|huye|kis|kisi|"
    r"sakte|sakti|sakta|sakein|sakoon|sakoonga|payenge|payega|payegi|deti|deta|"
    r"dete|leti|leta|lete|mera|meri|mere|tera|teri|tere|hamara|hamari|hamare|"
    r"shukriya|dhanyavaad|maaf|kripya|zaroor|haanji|hanji|theek hai|kal ke|"
    r"bhej|bhejna|bhejna hai|note kar|jo|jis|wo|woh|ye|yeh|is|isko|usko|inhe|"
    r"unhe|kabhi|jaldi|jaldi se|aur|ya|toh|to|hi|na)\b",
    re.IGNORECASE,
)
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_BENGALI_SCRIPT_RE = re.compile(r"[ঀ-৿]")


def reply_violates_english_lock(text: str) -> bool:
    if not text:
        return False
    if _DEVANAGARI_RE.search(text) or _BENGALI_SCRIPT_RE.search(text):
        return True
    return bool(ROMANIZED_INDIC_TOKEN_RE.search(text))


def scrub_forbidden_payment_methods(text: str) -> str:
    cleaned = text
    for pattern, replacement in _FORBIDDEN_PAYMENT_SUBSTITUTIONS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def scrub_invented_invoice_numbers(text: str, invoices: list[dict[str, Any]]) -> str:
    valid = {str(inv.get("invoice_no") or "").upper() for inv in invoices if inv.get("invoice_no")}
    if not valid:
        return text

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0).upper()
        return match.group(0) if token in valid else "[invoice on file]"

    return re.sub(r"\bDHL\d{4,}\b", _replace, text)


_CURRENCY_AMOUNT_RE = re.compile(
    r"(?:INR|Rs\.?|₹)\s*([0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)


def reply_has_invented_amount(text: str, invoices: list[dict[str, Any]]) -> str | None:
    """Return the offending amount string if the reply names a currency amount
    that is not in the ground-truth invoice list (per-invoice amount or total).
    Used to detect hallucinations like 'INR 12,784' when truth is 13,600 / 34,650 /
    9,670 / 57,920."""
    if not text:
        return None
    valid_amounts: set[int] = set()
    total = 0
    for inv in invoices:
        amount = inv.get("amount")
        if isinstance(amount, (int, float)):
            valid_amounts.add(int(amount))
            total += int(amount)
    if total:
        valid_amounts.add(total)
    if not valid_amounts:
        return None
    for match in _CURRENCY_AMOUNT_RE.finditer(text):
        raw = match.group(1).replace(",", "")
        try:
            value = int(float(raw))
        except ValueError:
            continue
        if value in valid_amounts:
            continue
        # Sub-amount tolerance: a partial-payment offer like "INR 5,000" is
        # a legit negotiating ask, but only if the model is clearly proposing
        # a partial. We treat any currency-prefixed number that doesn't match
        # ground truth and is >= 1000 as a hallucination, since the agent
        # should not be naming totals or per-invoice values that aren't real.
        if value >= 1000:
            return match.group(0)
    return None


def validate_tool_args(
    name: str,
    args: dict[str, Any],
    account_number: str,
    invoices: list[dict[str, Any]],
    customer: dict[str, Any],
    constants: dict[str, Any],
) -> dict[str, Any] | None:
    valid_invoices = {str(inv.get("invoice_no")) for inv in invoices}
    args = dict(args)
    args.setdefault("account_number", account_number)

    if name == "log_promise_to_pay":
        promise = str(args.get("promise_date") or "").strip()
        _, parsed = parse_customer_date(promise)
        if not promise_date_is_within_window(parsed, int(constants.get("promise_date_max_business_days") or 2)):
            return None
        invoice_no = str(args.get("invoice_no") or "")
        if invoice_no and invoice_no not in valid_invoices:
            args["invoice_no"] = invoices[0].get("invoice_no") if invoices else None
        return args

    if name in {"log_already_paid", "resend_invoice", "log_dispute"}:
        invoice_no = str(args.get("invoice_no") or "")
        if invoice_no and invoice_no not in valid_invoices:
            args["invoice_no"] = invoices[0].get("invoice_no") if invoices else None
        if name == "resend_invoice":
            args.setdefault("email", customer.get("registered_email"))
        if name == "log_dispute":
            reason_raw = str(args.get("reason") or "").strip()
            generic = {
                "", "dispute raised", "dispute", "customer disputes invoice",
                "customer raised a dispute", "billing dispute", "n/a", "none",
            }
            if reason_raw.lower() in generic:
                fallback = customer.get("__latest_customer_text") if isinstance(customer, dict) else None
                if fallback:
                    args["reason"] = str(fallback).strip()
        return args

    if name == "update_contact":
        return args

    if name == "transfer_to_human":
        args.setdefault("reason", "Escalated by agent during collections call.")
        return args

    if name in {"get_invoices", "get_customer"}:
        return args

    return None


def run_chat_agent_turn(
    messages: list[dict[str, Any]],
    voice: str | None,
    account_number: str,
    coaching_hints: list[str] | None = None,
    language_advice: dict[str, Any] | None = None,
) -> tuple[str, list[dict[str, Any]], list[Any], str | None]:
    del coaching_hints
    if POLICY_ENGINE_MODE == "llm" and should_use_fast_deterministic_turn(messages):
        text, tools_log, _ = generate_collections_reply(
            messages=messages,
            account_number=account_number,
            voice=voice,
            language_advice=language_advice,
        )
        if text:
            return text, tools_log, [], None
    if POLICY_ENGINE_MODE != "llm":
        text, tools_log, model = generate_collections_reply(
            messages=messages,
            account_number=account_number,
            voice=voice,
            language_advice=language_advice,
        )
        if text or POLICY_ENGINE_MODE == "deterministic":
            return text, tools_log, [], None

    text, tools_log, usage_events, error = llm_collections_turn(
        messages=messages,
        account_number=account_number,
        voice=voice,
        language_advice=language_advice,
    )
    if error or not text:
        fallback_text, fallback_tools, _ = generate_collections_reply(
            messages=messages,
            account_number=account_number,
            voice=voice,
            language_advice=language_advice,
        )
        if fallback_text:
            return fallback_text, fallback_tools, usage_events, None
        return "", tools_log, usage_events, error
    return text, tools_log, usage_events, None


def create_call_summary(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    return deterministic_call_summary(payload), None, None


def success_json(data: dict[str, Any], status: int = 200):
    return jsonify(data), status


def error_json(message: str, status: int = 400):
    return jsonify({"error": message}), status


def tool_get_customer(payload: dict[str, Any]) -> dict[str, Any]:
    account_number = payload.get("account_number", DEFAULT_ACCOUNT_ID)
    customer = get_customer(account_number)
    if not customer:
        return {"ok": False, "error": f"Unknown account {account_number}"}
    invoices = get_invoices(account_number)
    return {
        "ok": True,
        "customer": customer,
        "summary": {
            "total_outstanding": customer_outstanding(invoices),
            "invoice_count": len(invoices),
        },
    }


def tool_get_invoices(payload: dict[str, Any]) -> dict[str, Any]:
    account_number = payload.get("account_number", DEFAULT_ACCOUNT_ID)
    invoices = get_invoices(account_number)
    return {
        "ok": True,
        "account_number": account_number,
        "invoices": invoices,
        "total_outstanding": customer_outstanding(invoices),
    }


def tool_log_promise_to_pay(payload: dict[str, Any]) -> dict[str, Any]:
    promise_id = f"ptp_{uuid.uuid4().hex[:8]}"
    return {
        "ok": True,
        "ptp_id": promise_id,
        "account_number": payload.get("account_number", DEFAULT_ACCOUNT_ID),
        "promise_date": payload.get("promise_date"),
        "notes": payload.get("notes", ""),
    }


def tool_log_already_paid(payload: dict[str, Any]) -> dict[str, Any]:
    verification_task_id = f"verify_{uuid.uuid4().hex[:8]}"
    return {
        "ok": True,
        "verification_task_id": verification_task_id,
        "invoice_no": payload.get("invoice_no"),
        "reference_number": payload.get("reference_number", ""),
        "paid_date": payload.get("paid_date", ""),
        "message": "Payment claim recorded. Ask the customer to email proof of payment to yogesh.jhamb@dhl.com.",
    }


def tool_resend_invoice(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "invoice_no": payload.get("invoice_no"),
        "email": payload.get("email"),
        "channel": "registered_email",
        "message": "Mock resend triggered. The invoice will be sent to the registered email address.",
    }


def tool_log_dispute(payload: dict[str, Any]) -> dict[str, Any]:
    dispute_id = f"disp_{uuid.uuid4().hex[:8]}"
    return {
        "ok": True,
        "dispute_id": dispute_id,
        "invoice_no": payload.get("invoice_no"),
        "reason": payload.get("reason"),
        "undisputed_amount": payload.get("undisputed_amount"),
    }


def tool_update_contact(payload: dict[str, Any]) -> dict[str, Any]:
    customer = get_customer(payload.get("account_number", DEFAULT_ACCOUNT_ID))
    if not customer:
        return {"ok": False, "error": "Customer not found"}
    return {
        "ok": True,
        "account_number": customer["account_number"],
        "contact_name": payload.get("contact_name") or customer.get("contact_name"),
        "phone": payload.get("phone") or customer.get("phone"),
        "email": payload.get("email") or customer.get("registered_email"),
        "message": "Alternate contact captured for follow-up.",
    }


def tool_transfer_to_human(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": True,
        "agent": HUMAN_AGENT["name"],
        "phone": HUMAN_AGENT["phone"],
        "team": HUMAN_AGENT["team"],
        "reason": payload.get("reason"),
        "customer_summary": payload.get("customer_summary", ""),
    }


TOOL_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "get_customer": tool_get_customer,
    "get_invoices": tool_get_invoices,
    "log_promise_to_pay": tool_log_promise_to_pay,
    "log_already_paid": tool_log_already_paid,
    "resend_invoice": tool_resend_invoice,
    "log_dispute": tool_log_dispute,
    "update_contact": tool_update_contact,
    "transfer_to_human": tool_transfer_to_human,
}


ensure_state()

app = Flask(__name__, static_folder=None)
CORS(app)
sock = Sock(app)

if KEEP_ALIVE_AVAILABLE:
    try:
        init_keep_alive()
        print("Keep-alive service initialized", flush=True)
    except Exception as exc:
        print(f"Failed to initialize keep-alive: {exc}", flush=True)


@app.get("/health")
def health():
    return success_json(
        {
            "ok": True,
            "time": utc_now_iso(),
            "realtime_model": REALTIME_MODEL,
            "supervisor_model": SUPERVISOR_MODEL,
            "has_openai_key": bool(OPENAI_API_KEY),
        }
    )


@app.get("/api/bootstrap")
def bootstrap():
    customer = get_customer(DEFAULT_ACCOUNT_ID)
    invoices = get_invoices(DEFAULT_ACCOUNT_ID)
    if not customer:
        return error_json(f"Customer fixture {DEFAULT_ACCOUNT_ID} not found.", 500)
    default_voice = DEFAULT_REALTIME_VOICE

    payload = {
        "account_number": DEFAULT_ACCOUNT_ID,
        "customer": customer,
        "invoices": invoices,
        "total_outstanding": customer_outstanding(invoices),
        "human_agent": HUMAN_AGENT,
        "agent_prompt": compose_agent_instructions(DEFAULT_ACCOUNT_ID, default_voice),
        "agent_persona": persona_for_voice(default_voice),
        "realtime_tools": REALTIME_TOOLS,
        "board": load_board(),
        "costs": ledger_with_combined(load_ledger()),
        "call_history": load_call_history(),
        "config": {
            "tts_provider": "elevenlabs",
            "stt_provider": "sarvam",
            "tts_model": ELEVENLABS_TTS_MODEL,
            "stt_model": SARVAM_STT_MODEL,
            "realtime_model": ELEVENLABS_TTS_MODEL,  # legacy key kept for frontend cost panel
            "supported_realtime_models": [
                {"id": ELEVENLABS_TTS_MODEL, "label": f"ElevenLabs ({ELEVENLABS_TTS_MODEL})"},
            ],
            "realtime_voice": default_voice,
            "transcription_model": SARVAM_STT_MODEL,
            "supervisor_model": SUPERVISOR_MODEL,
            "language_coach_model": LANGUAGE_COACH_MODEL,
            "chat_model": CHAT_MODEL,
            "default_language_id": DEFAULT_LANGUAGE_ID,
            "supported_languages": supported_languages_payload(),
            "tts_voices": deepcopy(TTS_VOICES),
            "language_codes": dict(LANGUAGE_CODES),
            "pricing_reference": {
                "openai_currency": "USD",
                "elevenlabs": {
                    "currency": "USD",
                    "tts_usd_per_million_chars": {
                        ELEVENLABS_TTS_MODEL: ELEVENLABS_USD_PER_MILLION_CHARS,
                    },
                },
                "sarvam_stt": {
                    "currency": "INR",
                    "inr_per_usd": SARVAM_INR_PER_USD,
                    "stt_inr_per_hour": {
                        SARVAM_STT_MODEL: SARVAM_STT_INR_PER_HOUR,
                    },
                },
            },
            "tts_sample_rate": ELEVENLABS_TTS_SAMPLE_RATE_BROWSER,
            "stt_sample_rate": SARVAM_STT_SAMPLE_RATE,
            "stt_mode": SARVAM_STT_MODE,
            "telephony": {
                "provider": "exotel",
                "enabled": exotel_enabled(),
                "caller_id": sanitize_phone_number(EXOTEL_CALLER_ID, keep_plus=False) if EXOTEL_CALLER_ID else "",
                "stream_sample_rate": EXOTEL_STREAM_SAMPLE_RATE,
            },
        },
    }
    return success_json(payload)


@app.get("/api/customer/<account_number>")
def customer_route(account_number: str):
    customer = get_customer(account_number)
    if not customer:
        return error_json(f"Customer {account_number} not found.", 404)
    return success_json({"customer": customer})


@app.get("/api/invoices/<account_number>")
def invoices_route(account_number: str):
    return success_json({"account_number": account_number, "invoices": get_invoices(account_number)})


@app.post("/api/session")
def create_session():
    """Issue a voice session. The frontend then opens /api/tts/stream and
    /api/stt/stream WebSockets using this session_id. No client secret leaves the
    backend — the ElevenLabs (TTS) / Sarvam (STT) keys stay server-side.
    """
    if not ELEVENLABS_API_KEY:
        return error_json("ELEVENLABS_API_KEY is missing on the backend.", 500)
    if not SARVAM_API_KEY:
        return error_json("SARVAM_API_KEY is missing on the backend.", 500)

    body = request.get_json(silent=True) or {}
    requested_session_id = str(body.get("session_id") or "").strip()
    language_id = str(body.get("language_id") or DEFAULT_LANGUAGE_ID)
    default_voice = DEFAULT_REALTIME_VOICE
    voice = str(body.get("voice") or default_voice)
    if voice.lower() not in VOICE_PERSONAS:
        voice = default_voice
    resolved_tts_voice = elevenlabs_voice_id(voice)
    session_id = requested_session_id or uuid.uuid4().hex

    return success_json(
        {
            "session_id": session_id,
            "voice": voice,
            "resolved_tts_voice": resolved_tts_voice,
            "agent_persona": persona_for_voice(voice),
            "language_id": language_id,
            "language_code": language_code_for_id(language_id),
            "tts_language_code": language_code_for_id(language_id),
            "stt_language_code": sarvam_stt_language_code(language_id),
            "tts_ws_path": "/api/tts/stream",
            "stt_ws_path": "/api/stt/stream",
            "tts_sample_rate": ELEVENLABS_TTS_SAMPLE_RATE_BROWSER,
            "stt_sample_rate": SARVAM_STT_SAMPLE_RATE,
            "tts_model": ELEVENLABS_TTS_MODEL,
            "stt_model": SARVAM_STT_MODEL,
            "stt_mode": SARVAM_STT_MODE,
        }
    )


@app.get("/api/exotel/calls/active")
def exotel_active_call():
    prune_stale_phone_call_sessions()
    active_snapshot = None
    recent_snapshot = None
    with PHONE_CALL_SESSIONS_LOCK:
        for session in PHONE_CALL_SESSIONS.values():
            snapshot = session.snapshot()
            recent_snapshot = snapshot
            if snapshot["active"]:
                active_snapshot = snapshot
                break
    return success_json({"active_call": active_snapshot, "last_call": recent_snapshot})


@app.post("/api/exotel/calls/start")
def exotel_start_call():
    if not exotel_enabled():
        return error_json(
            "Exotel is not fully configured. Set EXOTEL_ACCOUNT_SID, EXOTEL_API_KEY, EXOTEL_API_TOKEN, EXOTEL_CALLER_ID, and RENDER_EXTERNAL_URL.",
            500,
        )
    if has_active_phone_call_session():
        return error_json("A phone demo call is already active. End that call before starting another.", 409)

    body = request.get_json(silent=True) or {}
    account_number = str(body.get("account_number") or DEFAULT_ACCOUNT_ID).strip() or DEFAULT_ACCOUNT_ID
    language_id = supported_render_language_id(str(body.get("language_id") or DEFAULT_LANGUAGE_ID))
    requested_voice = str(body.get("voice") or DEFAULT_REALTIME_VOICE).strip().lower()
    voice = requested_voice if requested_voice in VOICE_PERSONAS else DEFAULT_REALTIME_VOICE
    target_number = sanitize_phone_number(
        str(body.get("to_number") or body.get("target_number") or "").strip(),
        keep_plus=True,
    )
    caller_id = str(body.get("caller_id") or EXOTEL_CALLER_ID).strip()
    if not target_number:
        return error_json("to_number is required.")
    if not get_customer(account_number):
        return error_json(f"Customer fixture {account_number} not found.", 404)

    ledger = default_ledger(realtime_model=ELEVENLABS_TTS_MODEL, transcription_model=SARVAM_STT_MODEL)
    write_json(LEDGER_FILE, ledger)
    session_id = ledger["session_id"]
    session = PhoneCallSession(
        session_id=session_id,
        account_number=account_number,
        target_number=target_number,
        caller_id=caller_id,
        language_id=language_id,
        voice=voice,
    )
    with PHONE_CALL_SESSIONS_LOCK:
        PHONE_CALL_SESSIONS[session_id] = session

    try:
        stream_url = build_exotel_stream_url(session_id)
        status_callback_url = build_exotel_status_callback_url()
        outbound_payload = build_exotel_connect_payload(
            to_number=target_number,
            caller_id=caller_id,
            stream_url=stream_url,
            status_callback_url=status_callback_url,
        )
        multipart_payload = {key: (None, value) for key, value in outbound_payload.items()}
        response = requests.post(
            f"{EXOTEL_API_BASE_URL}/v1/Accounts/{EXOTEL_ACCOUNT_SID}/Calls/connect",
            headers={
                "Authorization": exotel_basic_auth_header(),
                "Accept": "application/xml, text/xml, */*",
            },
            files=multipart_payload,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001
        with PHONE_CALL_SESSIONS_LOCK:
            PHONE_CALL_SESSIONS.pop(session_id, None)
        return error_json(f"Exotel call start failed: {str(exc)[:300]}", 502)

    if not response.ok:
        with PHONE_CALL_SESSIONS_LOCK:
            PHONE_CALL_SESSIONS.pop(session_id, None)
        return error_json(
            f"Exotel call start failed {response.status_code}: {response.text[:400]}",
            502,
        )

    raw_response_text = response.text[:4000]
    try:
        exotel_payload = response.json()
    except ValueError:
        exotel_payload = {"raw_response": raw_response_text}
    call_sid = parse_exotel_call_sid(exotel_payload if isinstance(exotel_payload, dict) else None, raw_response_text)
    session.register_call_sid(call_sid)
    session.log_event("dial_requested", {"call_sid": call_sid, "exotel_response": exotel_payload})
    return success_json({"ok": True, "session": session.snapshot(), "exotel": exotel_payload}, 202)


@app.post("/api/exotel/calls/reset")
def exotel_reset_call():
    body = request.get_json(silent=True) or {}
    session_id = str(body.get("session_id") or "").strip()
    call_sid = str(body.get("call_sid") or "").strip()
    session = get_phone_call_session(session_id=session_id or None, call_sid=call_sid or None)
    if session is None:
        with PHONE_CALL_SESSIONS_LOCK:
            active_sessions = [candidate for candidate in PHONE_CALL_SESSIONS.values() if candidate.snapshot()["active"]]
        session = active_sessions[0] if active_sessions else None
    if session is None:
        return success_json({"ok": True, "cleared": False})
    session.finish("manually_reset")
    return success_json({"ok": True, "cleared": True, "session": session.snapshot()})


@app.post("/api/exotel/status")
def exotel_status():
    payload = request.get_json(silent=True) or request.form.to_dict(flat=True) or {}
    call_sid = str(payload.get("CallSid") or payload.get("Sid") or "").strip()
    session_id = str(payload.get("session_id") or "").strip()
    session = get_phone_call_session(session_id=session_id or None, call_sid=call_sid or None)
    if session:
        if call_sid:
            session.register_call_sid(call_sid)
        session.update_status(payload)
    return success_json({"ok": True})


def _elevenlabs_tts_url(voice_id: str, sample_rate: int) -> str:
    """ElevenLabs streaming endpoint emitting raw PCM at the requested rate."""
    params = urlencode({"output_format": f"pcm_{int(sample_rate)}"})
    return f"{ELEVENLABS_BASE_URL}/v1/text-to-speech/{voice_id}/stream?{params}"


def _tone_tag(tone: str | None) -> str:
    """Map a tone keyword to an eleven_v3 audio tag prefix (e.g. '[empathetic] ').
    Empty when tags are disabled, the model isn't v3, or tone is unset/'none'."""
    if not ELEVENLABS_AUDIO_TAGS or ELEVENLABS_TTS_MODEL not in _AUDIO_TAG_MODELS:
        return ""
    t = (tone or ELEVENLABS_TONE_DEFAULT or "").strip().lower()
    mapping = {
        "professional": ELEVENLABS_TONE_DEFAULT,
        "default": ELEVENLABS_TONE_DEFAULT,
        "empathetic": ELEVENLABS_TONE_EMPATHETIC,
        "firm": ELEVENLABS_TONE_FIRM,
    }
    tag = mapping.get(t, t)
    if not tag or tag.lower() == "none":
        return ""
    return f"[{tag}] "


def _elevenlabs_tts_body(
    text: str,
    language_id: str | None = None,
    previous_text: str | None = None,
    tone: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "text": _tone_tag(tone) + text,
        "model_id": ELEVENLABS_TTS_MODEL,
        "voice_settings": elevenlabs_voice_settings(),
        # "on" makes ElevenLabs spell out the numbers/dates we deliberately leave
        # as digits in the target language (so Hindi "3" -> "teen", dates read
        # naturally). IDs/phones/amounts are already pre-converted to English.
        "apply_text_normalization": ELEVENLABS_TEXT_NORMALIZATION,
    }
    # Pronunciation dictionary: consistent brand/payment/name pronunciation
    # (DHL -> "D H L", NEFT spelled, MyBill -> "My Bill"). Alias rules work on all
    # models incl. eleven_v3; verified on the streaming endpoint.
    if ELEVENLABS_PRON_DICT_ID and ELEVENLABS_PRON_DICT_VERSION_ID:
        body["pronunciation_dictionary_locators"] = [
            {
                "pronunciation_dictionary_id": ELEVENLABS_PRON_DICT_ID,
                "version_id": ELEVENLABS_PRON_DICT_VERSION_ID,
            }
        ]
    # Only send language_code for codes ElevenLabs accepts. For other Indic
    # languages the API 400s on the param even though it can synthesize the text;
    # omitting it lets the input script drive synthesis instead of failing.
    code = elevenlabs_language_code(language_id) if language_id else ""
    if code and code in ELEVENLABS_SUPPORTED_LANGUAGE_CODES:
        body["language_code"] = code
    # Prior agent utterance as context so prosody/intonation flows naturally across
    # turns instead of each reply being synthesized cold. Capped to keep the
    # request small. NOTE: eleven_v3 REJECTS previous_text/next_text (400
    # unsupported_model), so only send it on models that accept it.
    if previous_text and ELEVENLABS_TTS_MODEL not in _NO_CONTINUITY_MODELS:
        body["previous_text"] = previous_text[-600:]
    return body


def _elevenlabs_tts_stream(
    text: str,
    voice_id: str,
    *,
    sample_rate: int,
    language_id: str | None = None,
    previous_text: str | None = None,
    tone: str | None = None,
) -> Any:
    """Open a streaming ElevenLabs TTS response. Returns the requests.Response so
    the caller can relay resp.iter_content() chunks (raw int16 LE PCM) and close()
    it for barge-in cancellation."""
    if not ELEVENLABS_API_KEY:
        raise RuntimeError("ELEVENLABS_API_KEY missing")
    if not voice_id:
        raise RuntimeError("ElevenLabs voice id missing (set ELEVENLABS_DEFAULT_*).")
    resp = requests.post(
        _elevenlabs_tts_url(voice_id, sample_rate),
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json=_elevenlabs_tts_body(text, language_id, previous_text, tone),
        stream=True,
        timeout=30,
    )
    if not resp.ok:
        detail = ""
        try:
            detail = resp.text[:300]
        except Exception:
            pass
        resp.close()
        raise RuntimeError(f"ElevenLabs TTS failed {resp.status_code}: {detail}")
    return resp


def _elevenlabs_tts_rest(
    text: str,
    voice_id: str,
    *,
    sample_rate: int,
    language_id: str | None = None,
    previous_text: str | None = None,
    tone: str | None = None,
) -> bytes:
    """Synchronous fallback: collect the whole PCM stream into bytes."""
    resp = _elevenlabs_tts_stream(
        text, voice_id, sample_rate=sample_rate, language_id=language_id,
        previous_text=previous_text, tone=tone,
    )
    try:
        return resp.content
    finally:
        resp.close()


def _resample_pcm16(pcm: bytes, in_rate: int, out_rate: int) -> bytes:
    """Linear-interpolate mono int16 LE PCM from in_rate to out_rate. Used to lift
    8kHz Exotel telephony audio up to the 24kHz OpenAI realtime STT expects."""
    if in_rate == out_rate or not pcm:
        return pcm
    sample_count = len(pcm) // 2
    if sample_count <= 0:
        return pcm
    samples = struct.unpack("<" + ("h" * sample_count), pcm[: sample_count * 2])
    ratio = in_rate / out_rate
    out_len = max(int(sample_count / ratio), 1)
    out = bytearray(out_len * 2)
    for i in range(out_len):
        src = i * ratio
        i0 = int(src)
        i1 = min(i0 + 1, sample_count - 1)
        frac = src - i0
        value = int(samples[i0] * (1.0 - frac) + samples[i1] * frac)
        struct.pack_into("<h", out, i * 2, max(-32768, min(32767, value)))
    return bytes(out)


def _sarvam_stt_rest(pcm: bytes, sample_rate: int, language_code: str) -> tuple[str, str]:
    """REST fallback. Returns (transcript, detected_language_code). Uses Saarika
    auto-detect so the customer can switch languages mid-call."""
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY missing")
    wav_bytes = _pcm_to_wav(pcm, sample_rate)
    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {"language_code": "unknown", "model": SARVAM_STT_MODEL}
    resp = requests.post(
        f"{SARVAM_BASE_URL}/speech-to-text",
        headers={"api-subscription-key": SARVAM_API_KEY},
        files=files,
        data=data,
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Sarvam STT failed {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    transcript = str(payload.get("transcript") or "").strip()
    detected = str(payload.get("language_code") or language_code or "hi-IN").strip()
    return transcript, detected


@sock.route(EXOTEL_STREAM_PATH)
def exotel_media(ws):
    if not WEBSOCKET_CLIENT_AVAILABLE:
        ws.send(json.dumps({"event": "error", "message": "websocket-client not installed"}))
        return
    requested_session_id = str(request.args.get("session_id") or "").strip()
    session = get_phone_call_session(session_id=requested_session_id or None)
    if session is not None:
        session.attach_transport(ws)
        session.log_event("websocket_connected", {"session_id": requested_session_id})
    disconnected_reason = "stream_disconnected"
    try:
        while True:
            raw = ws.receive(timeout=120)
            if raw is None:
                break
            if isinstance(raw, (bytes, bytearray)):
                continue
            try:
                payload = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            event_type = str(payload.get("event") or "").strip().lower()
            if event_type == "connected":
                if session is not None:
                    session.log_event("connected", payload)
                continue
            if event_type == "start":
                start_payload = payload.get("start") if isinstance(payload.get("start"), dict) else {}
                stream_sid = str(
                    start_payload.get("stream_sid")
                    or start_payload.get("streamSid")
                    or payload.get("stream_sid")
                    or payload.get("streamSid")
                    or ""
                ).strip()
                call_sid = str(
                    start_payload.get("call_sid")
                    or start_payload.get("callSid")
                    or start_payload.get("call_id")
                    or payload.get("call_sid")
                    or ""
                ).strip()
                if session is None:
                    session = get_phone_call_session(call_sid=call_sid or None)
                if session is None and requested_session_id:
                    session = get_phone_call_session(session_id=requested_session_id)
                if session is None:
                    session = get_single_active_phone_call_session()
                if session is None:
                    ws.send(json.dumps({"event": "error", "message": "Unknown or expired phone session"}))
                    disconnected_reason = "unknown_session"
                    break
                session.attach_transport(ws)
                session.log_event(
                    "websocket_connected",
                    {"session_id": session.session_id, "requested_session_id": requested_session_id},
                )
                if call_sid:
                    session.register_call_sid(call_sid)
                session.start_stream(stream_sid)
                continue
            if event_type == "media":
                if session is None:
                    continue
                media = payload.get("media") if isinstance(payload.get("media"), dict) else {}
                b64 = str(media.get("payload") or "").strip()
                if not b64:
                    continue
                try:
                    session.forward_audio(base64.b64decode(b64))
                except Exception:
                    continue
                continue
            if event_type == "mark":
                if session is None:
                    continue
                mark_payload = payload.get("mark") if isinstance(payload.get("mark"), dict) else {}
                session.handle_mark(str(mark_payload.get("name") or "").strip())
                continue
            if event_type == "stop":
                disconnected_reason = "completed"
                if session is not None:
                    session.finish("completed")
                break
            if event_type == "dtmf":
                if session is not None:
                    session.log_event("dtmf", payload)
                continue
    finally:
        if session is not None:
            session.finish(disconnected_reason)


@sock.route("/api/tts/stream")
def tts_stream(ws):
    """Browser <-> backend WS for streaming TTS, proxying ElevenLabs Flash.

    Browser -> backend (JSON text frames):
      {"type": "hello", "session_id": "...", "voice": "...", "language_code": "..."}
      {"type": "speak", "text": "...", "language_code": "...optional override...",
       "utterance_id": "...optional..."}
      {"type": "cancel"}        # barge-in: drop any buffered audio

    Backend -> browser:
      {"type": "ready"}
      {"type": "audio_start", "utterance_id": "...", "sample_rate": 24000, "format": "pcm_s16le"}
      <binary frame: int16 little-endian PCM chunk>
      ... (multiple chunks) ...
      {"type": "audio_end", "utterance_id": "...", "chars": N}
      {"type": "error", "message": "..."}
    """
    if not ELEVENLABS_API_KEY:
        ws.send(json.dumps({"type": "error", "message": "ELEVENLABS_API_KEY missing on backend"}))
        return

    sample_rate = ELEVENLABS_TTS_SAMPLE_RATE_BROWSER
    state: dict[str, Any] = {
        "session_id": None,
        "voice": DEFAULT_REALTIME_VOICE,
        "language_code": "hi-IN",
        "current_resp": None,   # active requests.Response (ElevenLabs stream)
        "current_serial": 0,
        "current_utterance_id": None,
        "current_chars": 0,
        "last_spoken_text": None,  # prior utterance -> previous_text for prosody
        "stop": False,
    }
    upstream_lock = threading.Lock()

    def close_resp(resp: Any | None) -> None:
        if resp is None:
            return
        try:
            resp.close()
        except Exception:
            pass

    def relay_resp(resp: Any, utterance_id: str, chars: int, serial: int) -> None:
        sent_end = False
        try:
            for chunk in resp.iter_content(chunk_size=4096):
                if state["stop"] or not chunk:
                    if state["stop"]:
                        return
                    continue
                with upstream_lock:
                    if serial != state["current_serial"] or state["current_resp"] is not resp:
                        return
                try:
                    ws.send(chunk)
                except Exception:
                    return
            try:
                ws.send(json.dumps({"type": "audio_end", "utterance_id": utterance_id, "chars": chars}))
                sent_end = True
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            with upstream_lock:
                still_active = serial == state["current_serial"] and state["current_resp"] is resp
            if still_active:
                try:
                    ws.send(json.dumps({"type": "error", "message": f"ElevenLabs stream error: {str(exc)[:200]}"}))
                except Exception:
                    pass
        finally:
            close_resp(resp)
            with upstream_lock:
                still_active = serial == state["current_serial"] and state["current_resp"] is resp
                if still_active:
                    state["current_resp"] = None
                    state["current_utterance_id"] = None
                    state["current_chars"] = 0
            if not sent_end and still_active:
                try:
                    ws.send(json.dumps({"type": "audio_end", "utterance_id": utterance_id, "chars": chars}))
                except Exception:
                    pass

    ws.send(json.dumps({"type": "ready"}))

    try:
        while True:
            try:
                raw = ws.receive(timeout=120)
            except Exception:
                return
            if raw is None:
                return
            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            msg_type = msg.get("type")

            if msg_type == "hello":
                state["session_id"] = str(msg.get("session_id") or "")
                voice = str(msg.get("voice") or DEFAULT_REALTIME_VOICE)
                if voice.lower() in VOICE_PERSONAS:
                    state["voice"] = voice
                state["language_code"] = str(msg.get("language_code") or "hi-IN")
                continue

            if msg_type == "cancel":
                with upstream_lock:
                    resp = state["current_resp"]
                    state["current_serial"] += 1
                    state["current_resp"] = None
                    state["current_utterance_id"] = None
                    state["current_chars"] = 0
                close_resp(resp)
                ws.send(json.dumps({"type": "cancelled"}))
                continue

            if msg_type == "speak":
                text = str(msg.get("text") or "").strip()
                if not text:
                    continue
                language_code = str(msg.get("language_code") or state["language_code"])
                language_id = msg.get("language_id")
                speech_text = prepare_tts_text(text, language_code, language_id)
                if not speech_text:
                    continue
                utterance_id = str(msg.get("utterance_id") or uuid.uuid4().hex[:10])
                state["language_code"] = language_code
                with upstream_lock:
                    previous_resp = state["current_resp"]
                    state["current_serial"] += 1
                    serial = state["current_serial"]
                    state["current_resp"] = None
                    state["current_utterance_id"] = utterance_id
                    state["current_chars"] = len(speech_text)
                close_resp(previous_resp)
                previous_text = state.get("last_spoken_text")
                tone = str(msg.get("tone") or "").strip() or None
                try:
                    resp = _elevenlabs_tts_stream(
                        speech_text,
                        elevenlabs_voice_id(state["voice"]),
                        sample_rate=sample_rate,
                        language_id=language_id,
                        previous_text=previous_text,
                        tone=tone,
                    )
                except Exception as exc:  # noqa: BLE001
                    ws.send(json.dumps({"type": "error", "message": f"ElevenLabs TTS connect failed: {str(exc)[:200]}"}))
                    continue
                state["last_spoken_text"] = speech_text
                with upstream_lock:
                    if serial != state["current_serial"]:
                        close_resp(resp)
                        continue
                    state["current_resp"] = resp

                ws.send(json.dumps({
                    "type": "audio_start",
                    "utterance_id": utterance_id,
                    "sample_rate": sample_rate,
                    "format": "pcm_s16le",
                }))
                relay_thread = threading.Thread(
                    target=relay_resp,
                    args=(resp, utterance_id, len(speech_text), serial),
                    daemon=True,
                )
                relay_thread.start()
                try:
                    record_tts_usage(
                        chars=len(speech_text),
                        event_id=f"tts_{utterance_id}",
                        session_id=state.get("session_id") or None,
                    )
                except Exception:
                    pass
                continue
    finally:
        state["stop"] = True
        with upstream_lock:
            resp = state["current_resp"]
            state["current_resp"] = None
        close_resp(resp)


@sock.route("/api/stt/stream")
def stt_stream(ws):
    """Browser <-> backend WS for streaming STT, proxying Sarvam Saarika WS.

    Browser -> backend:
      {"type": "hello", "session_id": "...", "language_code": "hi-IN",
       "sample_rate": 16000}
      <binary frame: int16 PCM little-endian mono @ sample_rate>
      {"type": "flush"}     # ask Sarvam to emit a final ASAP
      {"type": "discard"}   # drop buffered audio (too short to be speech)
      {"type": "stop"}

    Backend -> browser:
      {"type": "ready"}
      {"type": "partial", "text": "..."}
      {"type": "final", "text": "...", "language_code": "hi-IN"}
      {"type": "error", "message": "..."}
    """
    if not SARVAM_API_KEY:
        ws.send(json.dumps({"type": "error", "message": "SARVAM_API_KEY missing on backend"}))
        return
    if not WEBSOCKET_CLIENT_AVAILABLE:
        ws.send(json.dumps({"type": "error", "message": "websocket-client not installed"}))
        return

    state: dict[str, Any] = {
        "session_id": None,
        "language_code": sarvam_stt_language_code(DEFAULT_LANGUAGE_ID),
        "sample_rate": SARVAM_STT_SAMPLE_RATE,
        "upstream": None,
        "audio_seconds_unbilled": 0.0,
        "stop": False,
    }
    upstream_lock = threading.Lock()
    pump_thread: threading.Thread | None = None

    def close_upstream(upstream: Any | None) -> None:
        if upstream is None:
            return
        try:
            upstream.close()
        except Exception:
            pass

    def reset_upstream(expected: Any | None = None, *, reset_billing: bool = False) -> None:
        with upstream_lock:
            upstream = state["upstream"]
            if expected is not None and upstream is not expected:
                upstream = None
            else:
                state["upstream"] = None
        if reset_billing:
            state["audio_seconds_unbilled"] = 0.0
        close_upstream(upstream)

    def open_upstream() -> Any:
        def _connect(language_code: str) -> Any:
            params = urlencode({
                "language-code": language_code or "unknown",
                "model": SARVAM_STT_MODEL,
                "mode": SARVAM_STT_MODE,
                "sample_rate": state["sample_rate"],
                "input_audio_codec": "pcm_s16le",
                "flush_signal": "true",
            })
            url = f"{SARVAM_STT_WS_URL}?{params}"
            return ws_client.create_connection(
                url,
                header=[f"Api-Subscription-Key: {SARVAM_API_KEY}"],
                timeout=15,
            )

        language_code = state["language_code"] or "unknown"
        try:
            return _connect(language_code)
        except Exception:
            if language_code != "unknown":
                raise
            return _connect(sarvam_language_code(DEFAULT_LANGUAGE_ID))

    def ensure_upstream() -> Any:
        nonlocal pump_thread
        with upstream_lock:
            upstream = state["upstream"]
        if upstream is None:
            upstream = open_upstream()
            with upstream_lock:
                current = state["upstream"]
                if current is None:
                    state["upstream"] = upstream
                else:
                    close_upstream(upstream)
                    upstream = current
        if pump_thread is None or not pump_thread.is_alive():
            pump_thread = threading.Thread(target=upstream_pump, daemon=True)
            pump_thread.start()
        return upstream

    def upstream_pump():
        while not state["stop"]:
            with upstream_lock:
                upstream = state["upstream"]
            if upstream is None:
                time.sleep(0.05)
                continue
            try:
                raw = upstream.recv()
            except Exception:
                reset_upstream(upstream)
                continue
            if not raw:
                reset_upstream(upstream)
                continue
            try:
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            except (TypeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            ptype = payload.get("type")
            data = payload.get("data") or {}
            # Saarika WS shape (validated against avr-asr-sarvam + Sarvam docs):
            #   {"type":"data","data":{"transcript":"...","language_code":"hi-IN"}}
            #   {"type":"error","data":{"message":"..."}}
            # Also tolerate alternative shapes some Saarika versions emit.
            text = (
                data.get("transcript")
                or payload.get("transcript")
                or payload.get("text")
                or ""
            ).strip()
            detected = (
                data.get("language_code")
                or payload.get("language_code")
                or payload.get("detected_language_code")
                or state["language_code"]
                or "hi-IN"
            )
            if ptype == "error":
                msg = data.get("message") or "Sarvam STT upstream error"
                try:
                    ws.send(json.dumps({"type": "error", "message": str(msg)[:300]}))
                except Exception:
                    return
                reset_upstream(upstream)
                continue
            if ptype in {"partial", "interim"} and text:
                try:
                    ws.send(json.dumps({"type": "partial", "text": text, "language_code": detected}))
                except Exception:
                    return
                continue
            # Treat data/transcript/final as a committed transcript.
            if ptype in {"data", "transcript", "final"} and text:
                if not is_stt_hallucination(text):
                    try:
                        ws.send(json.dumps({"type": "final", "text": text, "language_code": detected}))
                    except Exception:
                        return
                # Bill accumulated seconds when we get a final.
                seconds = state["audio_seconds_unbilled"]
                state["audio_seconds_unbilled"] = 0.0
                if seconds > 0:
                    try:
                        record_stt_usage(
                            seconds=seconds,
                            event_id=f"stt_{uuid.uuid4().hex[:10]}",
                            session_id=state.get("session_id") or None,
                        )
                    except Exception:
                        pass
    ws.send(json.dumps({"type": "ready"}))

    try:
        while True:
            try:
                raw = ws.receive(timeout=120)
            except Exception:
                return
            if raw is None:
                return

            if isinstance(raw, (bytes, bytearray)):
                chunk_seconds = len(raw) / (2 * max(state["sample_rate"], 1))
                # Sarvam expects base64-wrapped JSON audio frames.
                b64 = base64.b64encode(bytes(raw)).decode("ascii")
                msg = json.dumps({
                    "audio": {
                        "data": b64,
                        "sample_rate": str(state["sample_rate"]),
                        "encoding": "audio/wav",
                    }
                })
                sent = False
                for _ in range(2):
                    try:
                        upstream = ensure_upstream()
                    except Exception as exc:  # noqa: BLE001
                        ws.send(json.dumps({"type": "error", "message": f"Sarvam STT connect failed: {str(exc)[:200]}"}))
                        break
                    try:
                        upstream.send(msg)
                        state["audio_seconds_unbilled"] += chunk_seconds
                        sent = True
                        break
                    except Exception:
                        reset_upstream(upstream)
                continue

            try:
                msg = json.loads(raw)
            except (TypeError, json.JSONDecodeError):
                continue
            mtype = msg.get("type")
            if mtype == "hello":
                state["session_id"] = str(msg.get("session_id") or "")
                hinted = str(msg.get("language_code") or "").strip()
                if hinted:
                    state["language_code"] = hinted
                try:
                    state["sample_rate"] = int(msg.get("sample_rate") or SARVAM_STT_SAMPLE_RATE)
                except (TypeError, ValueError):
                    state["sample_rate"] = SARVAM_STT_SAMPLE_RATE
            elif mtype == "flush":
                if state["upstream"] is not None:
                    try:
                        state["upstream"].send(json.dumps({"type": "flush"}))
                    except Exception:
                        reset_upstream(state["upstream"])
            elif mtype == "discard":
                # Sarvam has no native discard command. Reset the upstream so a
                # short/garbled interruption cannot poison the next utterance.
                reset_upstream(reset_billing=True)
            elif mtype == "stop":
                return
    finally:
        state["stop"] = True
        reset_upstream(reset_billing=True)


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw int16 PCM in a minimal RIFF/WAV header."""
    num_samples = len(pcm) // 2
    byte_rate = sample_rate * 2
    block_align = 2
    data_size = num_samples * 2
    riff_size = 36 + data_size
    header = b"RIFF" + struct.pack("<I", riff_size) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16)
    header += b"data" + struct.pack("<I", data_size)
    return header + pcm


def pcm16_rms(pcm_bytes: bytes) -> float:
    if not pcm_bytes or len(pcm_bytes) < 2:
        return 0.0
    sample_count = len(pcm_bytes) // 2
    if sample_count <= 0:
        return 0.0
    samples = struct.unpack("<" + ("h" * sample_count), pcm_bytes[: sample_count * 2])
    energy = sum(sample * sample for sample in samples)
    return math.sqrt(energy / sample_count)


def _wav_to_pcm(wav_bytes: bytes, *, fallback_sample_rate: int) -> tuple[bytes, int]:
    import io
    import wave

    if not wav_bytes.startswith(b"RIFF"):
        # Sarvam returns raw PCM when output_audio_codec=linear16. Only try WAV
        # decoding when the response actually carries a RIFF header.
        return wav_bytes, fallback_sample_rate
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        pcm = handle.readframes(handle.getnframes())
    return pcm, sample_rate


@lru_cache(maxsize=4)
def load_phone_ambience_pcm(sample_rate: int) -> bytes:
    if not PHONE_AMBIENCE_ENABLED or not PHONE_AMBIENCE_FILE.exists():
        return b""
    import wave

    with wave.open(str(PHONE_AMBIENCE_FILE), "rb") as handle:
        channels = handle.getnchannels()
        source_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)

    if sample_width != 2 or channels not in {1, 2} or source_rate <= 0 or frame_count <= 0:
        return b""

    frame_size = channels * sample_width
    output_frames = max(int(frame_count * sample_rate / source_rate), 1)
    source_view = memoryview(raw)
    output = bytearray(output_frames * 2)

    def mono_sample(frame_index: int) -> int:
        safe_index = max(0, min(frame_count - 1, frame_index))
        offset = safe_index * frame_size
        if channels == 1:
            return struct.unpack_from("<h", source_view, offset)[0]
        left = struct.unpack_from("<h", source_view, offset)[0]
        right = struct.unpack_from("<h", source_view, offset + 2)[0]
        return int((left + right) / 2)

    if source_rate == sample_rate:
        for out_index in range(output_frames):
            struct.pack_into("<h", output, out_index * 2, mono_sample(out_index))
        return bytes(output)

    ratio = source_rate / sample_rate
    for out_index in range(output_frames):
        source_position = out_index * ratio
        left_index = int(source_position)
        right_index = min(left_index + 1, frame_count - 1)
        blend = source_position - left_index
        left_sample = mono_sample(left_index)
        right_sample = mono_sample(right_index)
        sample = int(round(left_sample + (right_sample - left_sample) * blend))
        struct.pack_into("<h", output, out_index * 2, sample)
    return bytes(output)


def apply_pcm16_gain(pcm_bytes: bytes, gain: float) -> bytes:
    if not pcm_bytes or gain <= 0:
        return b"\x00" * len(pcm_bytes)
    if abs(gain - 1.0) < 1e-6:
        return pcm_bytes
    sample_count = len(pcm_bytes) // 2
    source_view = memoryview(pcm_bytes)
    output = bytearray(len(pcm_bytes))
    for index in range(sample_count):
        sample = struct.unpack_from("<h", source_view, index * 2)[0]
        scaled = max(-32768, min(32767, int(round(sample * gain))))
        struct.pack_into("<h", output, index * 2, scaled)
    return bytes(output)


@lru_cache(maxsize=4)
def load_phone_ambience_profile(sample_rate: int) -> tuple[bytes, int]:
    pcm = load_phone_ambience_pcm(sample_rate)
    if not pcm:
        return b"", 0

    # Normalize against the OVERALL loudness of the clip (not the loudest window)
    # so the steady bed sits at PHONE_AMBIENCE_TARGET_RMS.
    reference_rms = pcm16_rms(pcm)
    if reference_rms <= 0:
        return pcm, 0

    # Start from the first window that reaches a representative level (>= 60% of
    # overall RMS), skipping any quiet intro so the call does not open on near
    # silence — but NOT the loudest moment, so it does not open on a peak.
    window_bytes = max(int(sample_rate * 2 * 0.5), 320)
    step_bytes = max(window_bytes // 2, 2)
    scan_limit = min(len(pcm), max(window_bytes * 16, window_bytes))
    threshold = reference_rms * 0.6
    start_offset = 0
    upper_bound = max(scan_limit - window_bytes, 0)
    for offset in range(0, upper_bound + 1, step_bytes):
        if pcm16_rms(pcm[offset : offset + window_bytes]) >= threshold:
            start_offset = offset
            break

    # Allow BOTH attenuation and boost. Previously gain<=1.0 returned the clip
    # un-touched, so a target below the clip's RMS could never make it quieter —
    # that left the bed far too loud (measured 6836 RMS in the field).
    gain = min(PHONE_AMBIENCE_MAX_NORMALIZE_GAIN, PHONE_AMBIENCE_TARGET_RMS / reference_rms)
    if abs(gain - 1.0) < 1e-3:
        return pcm, start_offset
    return apply_pcm16_gain(pcm, gain), start_offset


def mix_pcm16_le(foreground: bytes, background: bytes) -> bytes:
    if not foreground or not background:
        return foreground
    sample_count = min(len(foreground), len(background)) // 2
    fg_view = memoryview(foreground)
    bg_view = memoryview(background)
    output = bytearray(len(foreground))
    for index in range(sample_count):
        fg = struct.unpack_from("<h", fg_view, index * 2)[0]
        bg = struct.unpack_from("<h", bg_view, index * 2)[0]
        mixed = max(-32768, min(32767, fg + bg))
        struct.pack_into("<h", output, index * 2, mixed)
    if len(foreground) > sample_count * 2:
        output[sample_count * 2 :] = foreground[sample_count * 2 :]
    return bytes(output)


def parse_exotel_call_sid(payload: dict[str, Any] | None, raw_text: str | None = None) -> str:
    if isinstance(payload, dict):
        call_info = payload.get("Call") if isinstance(payload.get("Call"), dict) else payload
        if isinstance(call_info, dict):
            for key in ("Sid", "sid", "CallSid", "call_sid"):
                value = str(call_info.get(key) or "").strip()
                if value:
                    return value
    raw = str(raw_text or "").strip()
    if not raw:
        return ""
    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError:
        return ""
    sid_node = root.find(".//Sid")
    if sid_node is None or sid_node.text is None:
        return ""
    return sid_node.text.strip()


PHONE_CALL_SESSIONS: dict[str, "PhoneCallSession"] = {}
PHONE_CALL_SESSIONS_BY_CALL_SID: dict[str, str] = {}
PHONE_CALL_SESSIONS_LOCK = threading.RLock()


class PhoneCallSession:
    def __init__(
        self,
        *,
        session_id: str,
        account_number: str,
        target_number: str,
        caller_id: str,
        language_id: str,
        voice: str,
    ) -> None:
        self.session_id = session_id
        self.account_number = account_number
        self.target_number = sanitize_phone_number(target_number, keep_plus=True)
        self.caller_id = sanitize_phone_number(caller_id, keep_plus=False)
        self.language_id = supported_render_language_id(language_id)
        self.active_language_id = self.language_id
        self.voice = voice if voice.lower() in VOICE_PERSONAS else DEFAULT_REALTIME_VOICE
        self.customer = get_customer(account_number) or {}
        self.invoices = get_invoices(account_number)
        self.persona = persona_for_voice(self.voice)
        self.agent_prompt = compose_agent_instructions(account_number, self.voice)
        self.language_advice = default_language_advice(self.language_id)
        self.transcript: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.coaching_hints: list[str] = []
        self.disposition = "Call in progress"
        self.turn_number = 0
        self.summary: dict[str, Any] | None = None
        self.status = "queued"
        self.call_sid: str | None = None
        self.stream_sid: str | None = None
        self.created_at = utc_now()
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self._lock = threading.RLock()
        self._ws: Any | None = None
        self._stop = False
        self._turn_commit_timer: threading.Timer | None = None
        self._turn_commit_buffer: list[str] = []
        self._current_response_id: str | None = None
        self._current_mark_name: str | None = None
        self._current_response_text: str | None = None
        self._current_tts_serial = 0
        self._customer_revision = 0
        self._tts_upstream: Any | None = None
        self._playback_finish_timer: threading.Timer | None = None
        self._last_agent_speak_start_at: float | None = None
        self._pending_barge_in_at: float | None = None
        self._stt_upstream: Any | None = None
        self._stt_thread: threading.Thread | None = None
        self._stt_audio_seconds_unbilled = 0.0
        self._speech_seconds_since_flush = 0.0
        self._silence_seconds_since_speech = 0.0
        self._flush_sent_for_current_pause = False
        self._event_log: deque[dict[str, Any]] = deque(maxlen=80)
        self._finalized = False
        self._greeting_started = False
        self._ambience_cursor_bytes = 0
        self._ambience_thread: threading.Thread | None = None
        self._ambience_started_logged = False
        self._media_send_lock = threading.Lock()
        self._pending_language_candidate: str | None = None
        self._pending_language_candidate_at: float | None = None
        self._last_unspeakable_reprompt_at: float | None = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = self.status in {"queued", "dialing", "connected"}
            return {
                "session_id": self.session_id,
                "account_number": self.account_number,
                "target_number": self.target_number,
                "caller_id": self.caller_id,
                "call_sid": self.call_sid,
                "stream_sid": self.stream_sid,
                "status": self.status,
                "active": active,
                "language_id": self.language_id,
                "active_language_id": self.active_language_id,
                "voice": self.voice,
                "resolved_tts_voice": elevenlabs_voice_id(self.voice),
                "disposition": self.disposition,
                "turn_number": self.turn_number,
                "transcript_count": len(self.transcript),
                "tool_call_count": len(self.tool_calls),
                "summary": self.summary,
                "created_at": self.created_at.isoformat(),
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "ended_at": self.ended_at.isoformat() if self.ended_at else None,
                "events": list(self._event_log),
            }

    def register_call_sid(self, call_sid: str | None) -> None:
        normalized = str(call_sid or "").strip()
        if not normalized:
            return
        with self._lock:
            self.call_sid = normalized
            if self.status == "queued":
                self.status = "dialing"
        with PHONE_CALL_SESSIONS_LOCK:
            PHONE_CALL_SESSIONS_BY_CALL_SID[normalized] = self.session_id

    def attach_transport(self, ws: Any) -> None:
        with self._lock:
            self._ws = ws

    def log_event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        with self._lock:
            self._event_log.append(
                {
                    "kind": kind,
                    "at": utc_now_iso(),
                    "payload": payload or {},
                }
            )

    def _send_json(self, payload: dict[str, Any]) -> None:
        with self._lock:
            ws = self._ws
        if ws is None:
            return
        ws.send(json.dumps(payload))

    def _append_transcript(self, role: str, text: str, *, entry_id: str | None = None, status: str = "final") -> None:
        normalized = normalize_whitespace(text).strip()
        if not normalized:
            return
        with self._lock:
            self.transcript.append(
                {
                    "id": entry_id or f"{role}_{uuid.uuid4().hex[:10]}",
                    "role": role,
                    "text": normalized,
                    "timestamp": utc_now_iso(),
                    "status": status,
                }
            )

    def _is_customer_revision_current(self, revision: int) -> bool:
        with self._lock:
            return not self._stop and revision == self._customer_revision

    def _opening_barge_in_protection_state(self) -> tuple[bool, float | None]:
        with self._lock:
            active = bool(self._greeting_started and self.turn_number == 0 and self._current_response_id)
            speak_started_at = self._last_agent_speak_start_at
        return active, speak_started_at

    def _latest_assistant_text(self) -> str:
        with self._lock:
            for entry in reversed(self.transcript):
                if entry.get("role") == "assistant":
                    return str(entry.get("text") or "")
        return ""

    def _clear_pending_language_candidate_locked(self) -> None:
        self._pending_language_candidate = None
        self._pending_language_candidate_at = None

    def _confirm_tentative_language_candidate(self, candidate_language_id: str) -> bool:
        normalized = supported_render_language_id(candidate_language_id or self.active_language_id)
        now = time.time()
        with self._lock:
            if (
                self._pending_language_candidate == normalized
                and self._pending_language_candidate_at is not None
                and now - self._pending_language_candidate_at <= PHONE_LANGUAGE_SWITCH_CONFIRM_WINDOW_SECONDS
            ):
                self._clear_pending_language_candidate_locked()
                return True
            self._pending_language_candidate = normalized
            self._pending_language_candidate_at = now
        return False

    def _cancel_playback_finish_timer_locked(self) -> None:
        timer = self._playback_finish_timer
        self._playback_finish_timer = None
        if timer is not None:
            timer.cancel()

    def _close_tts_upstream(self, expected: Any | None = None) -> None:
        with self._lock:
            upstream = self._tts_upstream
            if expected is not None and upstream is not expected:
                upstream = None
            else:
                self._tts_upstream = None
        if upstream is None:
            return
        try:
            upstream.close()
        except Exception:
            pass

    def _next_ambience_segment(self, byte_count: int) -> bytes:
        loop_pcm, preferred_start = load_phone_ambience_profile(EXOTEL_STREAM_SAMPLE_RATE)
        if not loop_pcm or byte_count <= 0:
            return b""
        with self._lock:
            cursor = self._ambience_cursor_bytes
            total = len(loop_pcm)
            if total <= 0:
                return b""
            if cursor <= 0 and preferred_start > 0:
                cursor = min(preferred_start, max(total - 2, 0))
                self._ambience_cursor_bytes = cursor
            end = cursor + byte_count
            if end <= total:
                segment = loop_pcm[cursor:end]
            else:
                wrap = end - total
                segment = loop_pcm[cursor:] + loop_pcm[:wrap]
            self._ambience_cursor_bytes = (cursor + byte_count) % total
        if len(segment) < byte_count:
            segment = segment + (b"\x00" * (byte_count - len(segment)))
        return segment

    def _send_pcm_chunk(self, pcm_bytes: bytes, *, ambience_gain: float = 0.0) -> bool:
        if not pcm_bytes:
            return False
        with self._lock:
            if self._stop or not self.stream_sid:
                return False
            stream_sid = self.stream_sid
        payload_pcm = pcm_bytes
        if ambience_gain > 0:
            ambience = self._next_ambience_segment(len(pcm_bytes))
            if ambience:
                payload_pcm = mix_pcm16_le(pcm_bytes, apply_pcm16_gain(ambience, ambience_gain))
        with self._media_send_lock:
            self._send_json(
                {
                    "event": "media",
                    "stream_sid": stream_sid,
                    "media": {
                        "payload": base64.b64encode(payload_pcm).decode("ascii"),
                    },
                }
            )
        return True

    def _start_ambience_loop(self) -> None:
        if not PHONE_AMBIENCE_ENABLED or self._ambience_thread is not None and self._ambience_thread.is_alive():
            return
        # Surface a missing/unloadable ambience file once per call. Without this
        # the pump runs but every segment is empty, which presents as silent
        # dead air with no error in the log (the original phone-vs-browser bug).
        if not self._ambience_started_logged:
            profile_pcm, _ = load_phone_ambience_profile(EXOTEL_STREAM_SAMPLE_RATE)
            if not profile_pcm:
                self.log_event(
                    "ambience_error",
                    {
                        "message": "ambience file missing or unloadable; phone will have dead air",
                        "path": str(PHONE_AMBIENCE_FILE),
                        "exists": PHONE_AMBIENCE_FILE.exists(),
                    },
                )
        thread = threading.Thread(target=self._ambience_pump, daemon=True)
        self._ambience_thread = thread
        thread.start()

    def _emit_idle_ambience_once(self) -> None:
        if not PHONE_AMBIENCE_ENABLED:
            return
        with self._lock:
            if self._stop or not self.stream_sid:
                return
            if self._current_response_id or self._current_mark_name or self._tts_upstream:
                return
        ambience = self._next_ambience_segment(max(int(EXOTEL_STREAM_SAMPLE_RATE * 2 * 0.32), 320))
        if not ambience:
            return
        try:
            scaled = apply_pcm16_gain(ambience, PHONE_AMBIENCE_IDLE_GAIN)
            if not self._ambience_started_logged:
                self._ambience_started_logged = True
                self.log_event(
                    "ambience_first_audio",
                    {
                        "gain": PHONE_AMBIENCE_IDLE_GAIN,
                        "rms": int(round(pcm16_rms(scaled))),
                    },
                )
            if not self._send_pcm_chunk(scaled, ambience_gain=0.0):
                self.log_event("ambience_error", {"message": "idle ambience send returned false"})
        except Exception as exc:
            self.log_event("ambience_error", {"message": str(exc)[:160]})

    def _ambience_pump(self) -> None:
        chunk_size = max(int(EXOTEL_STREAM_SAMPLE_RATE * 2 * 0.1), 320)
        sleep_seconds = chunk_size / (2 * max(EXOTEL_STREAM_SAMPLE_RATE, 1))
        while True:
            with self._lock:
                if self._stop:
                    return
                playback_active = bool(self._current_response_id or self._current_mark_name or self._tts_upstream)
                stream_ready = bool(self.stream_sid)
            if not stream_ready or playback_active:
                time.sleep(0.05)
                continue
            ambience = self._next_ambience_segment(chunk_size)
            if ambience:
                try:
                    scaled = apply_pcm16_gain(ambience, PHONE_AMBIENCE_IDLE_GAIN)
                    if not self._ambience_started_logged:
                        self._ambience_started_logged = True
                        self.log_event(
                            "ambience_first_audio",
                            {
                                "gain": PHONE_AMBIENCE_IDLE_GAIN,
                                "rms": int(round(pcm16_rms(scaled))),
                            },
                        )
                    if not self._send_pcm_chunk(scaled, ambience_gain=0.0):
                        self.log_event("ambience_error", {"message": "ambience loop send returned false"})
                        time.sleep(0.1)
                        continue
                except Exception as exc:
                    self.log_event("ambience_error", {"message": str(exc)[:160]})
                    time.sleep(0.1)
                    continue
            time.sleep(sleep_seconds)

    def _complete_active_playback(self, serial: int, mark_name: str, source: str) -> bool:
        with self._lock:
            if serial != self._current_tts_serial or mark_name != self._current_mark_name:
                return False
            self._cancel_playback_finish_timer_locked()
            self._current_response_id = None
            self._current_mark_name = None
            self._current_response_text = None
            self._last_agent_speak_start_at = None
            self.turn_number += 1
        self.log_event("playback_completed", {"source": source})
        self._start_ambience_loop()
        self._emit_idle_ambience_once()
        threading.Thread(target=self._run_supervisor_review, daemon=True).start()
        return True

    def _handle_language_detection(
        self,
        detected_code: str,
        text: str = "",
        candidate_language_id: str | None = None,
    ) -> None:
        if text and not should_apply_language_switch_hint(text) and not candidate_language_id:
            return
        raw_mapped = candidate_language_id or language_id_for_sarvam_code(detected_code)
        if not raw_mapped:
            return
        mapped = supported_render_language_id(raw_mapped)
        with self._lock:
            if mapped == self.active_language_id:
                self._clear_pending_language_candidate_locked()
                return
            self.active_language_id = mapped
            self._clear_pending_language_candidate_locked()
            next_advice = default_language_advice(mapped)
            next_advice["detected_language_id"] = mapped
            next_advice["suggested_language_id"] = mapped
            next_advice["should_switch"] = True
            next_advice["confidence"] = "high"
            next_advice["nudge"] = f"Customer switched to {mapped}. Reply in {mapped}."
            self.language_advice = next_advice
        self._append_transcript("system", next_advice["nudge"])

    def _opening_text(self) -> str:
        supported = supported_languages_payload()
        opening_label = next(
            (item.get("agent_label") for item in supported if item.get("id") == self.language_id),
            "Hinglish",
        )
        return build_opening_text(self.customer, self.persona, str(opening_label))

    def start_greeting(self) -> None:
        with self._lock:
            if self._greeting_started or self._stop:
                return
            self._greeting_started = True
        opening_text = self._opening_text()
        if opening_text:
            self._speak_reply(opening_text)

    def _cancel_active_playback(self, reason: str) -> None:
        with self._lock:
            if not self._current_response_id and not self._current_mark_name:
                return
            self._cancel_playback_finish_timer_locked()
            self._current_tts_serial += 1
            self._current_response_id = None
            self._current_mark_name = None
            self._current_response_text = None
            self._last_agent_speak_start_at = None
            self._pending_barge_in_at = time.time()
        self._close_tts_upstream()
        self.log_event("playback_cleared", {"reason": reason})
        try:
            self._send_json({"event": "clear", "stream_sid": self.stream_sid})
        except Exception:
            pass

    def _reprompt_after_unspeakable_turn(self, drop_reason: str) -> None:
        """Speak a short nudge when we had to drop the customer's turn (e.g. they
        spoke a language we cannot render) so the call does not stall into silence.
        Rate-limited so repeated unspeakable turns do not loop."""
        now = time.time()
        with self._lock:
            if self._stop or not self.stream_sid:
                return
            # Do not step on an agent reply that is already (re)starting.
            if self._current_response_id or self._current_mark_name:
                return
            last = self._last_unspeakable_reprompt_at
            if last is not None and now - last < 6.0:
                return
            self._last_unspeakable_reprompt_at = now
        active = supported_render_language_id(self.active_language_id)
        prompts = {
            "english": "Sorry, I didn't catch that. Could you please continue in English or Hindi?",
            "hindi": "माफ कीजिए, मैं समझ नहीं पाया। क्या आप Hindi या English में बता सकते हैं?",
            "hinglish": "Sorry, main samajh nahi paaya. Kya aap Hindi ya English mein bata sakte hain?",
            "marathi": "क्षमा करा, मला समजलं नाही। तुम्ही Marathi किंवा English मध्ये सांगू शकता का?",
            "bengali": "ক্ষমা করবেন, আমি বুঝতে পারিনি। আপনি কি Bengali বা English-এ বলতে পারবেন?",
            "tamil": "மன்னிக்கவும், புரியவில்லை. Tamil அல்லது English-ல் சால்ல முடியுமா?",
        }
        message = prompts.get(active, prompts["english"])
        self._speak_reply(message)

    def _speak_reply(self, text: str, tone: str | None = None) -> None:
        normalized = normalize_whitespace(text).strip()
        if not normalized:
            return
        with self._lock:
            if self._stop:
                return
            if self._current_response_id:
                self._cancel_active_playback("superseded")
            self._current_tts_serial += 1
            serial = self._current_tts_serial
            response_id = f"utt_{uuid.uuid4().hex[:10]}"
            mark_name = f"mark_{response_id}"
            self._current_response_id = response_id
            self._current_mark_name = mark_name
            self._current_response_text = normalized
            self._last_agent_speak_start_at = time.time()
        self._append_transcript("assistant", normalized, entry_id=f"assistant_{response_id}")
        self.log_event("assistant_reply", {"utterance_id": response_id, "text": normalized})
        threading.Thread(
            target=self._render_and_send_tts,
            args=(serial, response_id, mark_name, normalized, tone),
            daemon=True,
        ).start()

    def _render_and_send_tts(self, serial: int, response_id: str, mark_name: str, text: str, tone: str | None = None) -> None:
        language_code = language_code_for_id(self.active_language_id)
        speech_text = prepare_tts_text(text, language_code, self.active_language_id)
        if not speech_text:
            return

        try:
            record_tts_usage(
                chars=len(speech_text),
                event_id=f"tts_{response_id}",
                session_id=self.session_id,
                model=ELEVENLABS_TTS_MODEL,
            )
        except Exception:
            pass

        pcm_bytes = b""
        sent_audio_bytes = 0
        send_started_at = time.time()
        logged_first_audio = False
        voice_id = elevenlabs_voice_id(self.voice)
        previous_text = getattr(self, "_last_spoken_tts_text", None)
        self._last_spoken_tts_text = speech_text
        try:
            upstream = _elevenlabs_tts_stream(
                speech_text,
                voice_id,
                sample_rate=ELEVENLABS_TTS_SAMPLE_RATE_PHONE,
                language_id=self.active_language_id,
                previous_text=previous_text,
                tone=tone,
            )
            with self._lock:
                if self._stop or serial != self._current_tts_serial:
                    try:
                        upstream.close()
                    except Exception:
                        pass
                    return
                self._tts_upstream = upstream

            # ElevenLabs streams raw int16 LE PCM at the phone sample rate; relay
            # each chunk straight onto the Exotel media stream.
            for chunk in upstream.iter_content(chunk_size=640):
                if not chunk:
                    continue
                with self._lock:
                    if self._stop or serial != self._current_tts_serial or self._tts_upstream is not upstream:
                        return
                    stream_sid = self.stream_sid
                if not stream_sid:
                    continue
                sent_audio_bytes += len(chunk)
                if not logged_first_audio:
                    logged_first_audio = True
                    self.log_event("tts_first_audio", {"source": "elevenlabs_stream"})
                self._send_pcm_chunk(chunk, ambience_gain=PHONE_AMBIENCE_TTS_GAIN)

            if sent_audio_bytes <= 0:
                raise RuntimeError("ElevenLabs TTS stream returned no audio")
        except Exception:
            try:
                pcm_bytes = _elevenlabs_tts_rest(
                    speech_text,
                    voice_id,
                    sample_rate=ELEVENLABS_TTS_SAMPLE_RATE_PHONE,
                    language_id=self.active_language_id,
                    previous_text=previous_text,
                    tone=tone,
                )
            except Exception as exc:  # noqa: BLE001
                self._append_transcript("system", f"TTS error: {str(exc)[:200]}")
                self.log_event("tts_error", {"message": str(exc)[:200]})
                with self._lock:
                    if serial == self._current_tts_serial:
                        self._cancel_playback_finish_timer_locked()
                        self._current_response_id = None
                        self._current_mark_name = None
                        self._current_response_text = None
                        self._last_agent_speak_start_at = None
                return
            chunk_size = max(int(EXOTEL_STREAM_SAMPLE_RATE * 2 * 0.05), 320)
            for index in range(0, len(pcm_bytes), chunk_size):
                with self._lock:
                    if self._stop or serial != self._current_tts_serial or not self.stream_sid:
                        return
                    stream_sid = self.stream_sid
                chunk = pcm_bytes[index : index + chunk_size]
                if not chunk:
                    continue
                sent_audio_bytes += len(chunk)
                if not logged_first_audio:
                    logged_first_audio = True
                    self.log_event("tts_first_audio", {"source": "elevenlabs_rest_fallback"})
                try:
                    self._send_pcm_chunk(chunk, ambience_gain=PHONE_AMBIENCE_TTS_GAIN)
                except Exception:
                    return
        finally:
            self._close_tts_upstream()

        with self._lock:
            if self._stop or serial != self._current_tts_serial or not self.stream_sid:
                return
            stream_sid = self.stream_sid
        try:
            self._send_json(
                {
                    "event": "mark",
                    "stream_sid": stream_sid,
                    "mark": {"name": mark_name},
                }
            )
        except Exception:
            pass
        playback_seconds = sent_audio_bytes / (2 * max(EXOTEL_STREAM_SAMPLE_RATE, 1))
        remaining_playback_seconds = max(playback_seconds - max(time.time() - send_started_at, 0.0), 0.0)
        fallback_timer = threading.Timer(
            remaining_playback_seconds + 0.2,
            self._complete_active_playback,
            args=(serial, mark_name, "timer_fallback"),
        )
        fallback_timer.daemon = True
        with self._lock:
            if self._stop or serial != self._current_tts_serial or mark_name != self._current_mark_name:
                return
            self._cancel_playback_finish_timer_locked()
            self._playback_finish_timer = fallback_timer
        fallback_timer.start()

    def _close_stt_upstream(self, expected: Any | None = None, *, reset_billing: bool = False) -> None:
        with self._lock:
            upstream = self._stt_upstream
            if expected is not None and upstream is not expected:
                upstream = None
            else:
                self._stt_upstream = None
            if reset_billing:
                self._stt_audio_seconds_unbilled = 0.0
                self._speech_seconds_since_flush = 0.0
                self._silence_seconds_since_speech = 0.0
                self._flush_sent_for_current_pause = False
        if upstream is None:
            return
        try:
            upstream.close()
        except Exception:
            pass

    def _request_stt_flush(self) -> None:
        with self._lock:
            upstream = self._stt_upstream
        if upstream is None:
            return
        try:
            upstream.send(json.dumps({"type": "flush"}))
            self.log_event("stt_flush", {})
        except Exception:
            self._close_stt_upstream(upstream)

    def _open_stt_upstream(self) -> Any:
        def _connect(language_code: str) -> Any:
            params = urlencode(
                {
                    "language-code": language_code or "unknown",
                    "model": SARVAM_STT_MODEL,
                    "mode": SARVAM_STT_MODE,
                    "sample_rate": EXOTEL_STREAM_SAMPLE_RATE,
                    "input_audio_codec": "pcm_s16le",
                    "flush_signal": "true",
                }
            )
            return ws_client.create_connection(
                f"{SARVAM_STT_WS_URL}?{params}",
                header=[f"Api-Subscription-Key: {SARVAM_API_KEY}"],
                timeout=15,
            )

        language_code = sarvam_stt_language_code(self.language_id)
        try:
            return _connect(language_code)
        except Exception:
            if language_code != "unknown":
                raise
            return _connect(sarvam_language_code(DEFAULT_LANGUAGE_ID))

    def _ensure_stt_upstream(self) -> Any:
        with self._lock:
            upstream = self._stt_upstream
        if upstream is None:
            upstream = self._open_stt_upstream()
            with self._lock:
                current = self._stt_upstream
                if current is None:
                    self._stt_upstream = upstream
                else:
                    try:
                        upstream.close()
                    except Exception:
                        pass
                    upstream = current
        if self._stt_thread is None or not self._stt_thread.is_alive():
            self._stt_thread = threading.Thread(target=self._stt_pump, daemon=True)
            self._stt_thread.start()
        return upstream

    def _stt_pump(self) -> None:
        while True:
            with self._lock:
                stop = self._stop
                upstream = self._stt_upstream
            if stop:
                return
            if upstream is None:
                time.sleep(0.05)
                continue
            try:
                raw = upstream.recv()
            except Exception:
                self._close_stt_upstream(upstream)
                continue
            if not raw:
                self._close_stt_upstream(upstream)
                continue
            try:
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            except (TypeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            ptype = payload.get("type")
            data = payload.get("data") or {}
            text = (data.get("transcript") or payload.get("transcript") or payload.get("text") or "").strip()
            detected = (
                data.get("language_code")
                or payload.get("language_code")
                or payload.get("detected_language_code")
                or sarvam_language_code(self.active_language_id)
            )
            # Sarvam may report a numeric confidence for the detected language /
            # transcript. Absent on many frames; we treat missing as "unknown"
            # and fall back to evidence-strength heuristics in the switch gate.
            detected_confidence = _coerce_confidence(
                data.get("language_confidence")
                if data.get("language_confidence") is not None
                else data.get("confidence")
                if data.get("confidence") is not None
                else payload.get("language_confidence")
                if payload.get("language_confidence") is not None
                else payload.get("confidence")
            )
            if ptype == "error":
                message = str(data.get("message") or "Sarvam STT upstream error")[:300]
                self._append_transcript("system", f"Sarvam STT error: {message}")
                self.log_event("stt_error", {"message": message})
                self._close_stt_upstream(upstream)
                continue
            if ptype in {"partial", "interim"} and text:
                self._handle_partial_transcript(text, str(detected))
                continue
            if ptype in {"data", "transcript", "final"} and text:
                self.log_event("stt_final_received", {"text": text[:120]})
                seconds = 0.0
                with self._lock:
                    seconds = self._stt_audio_seconds_unbilled
                    self._stt_audio_seconds_unbilled = 0.0
                    self._speech_seconds_since_flush = 0.0
                    self._silence_seconds_since_speech = 0.0
                    self._flush_sent_for_current_pause = False
                if seconds > 0:
                    try:
                        record_stt_usage(
                            seconds=seconds,
                            event_id=f"stt_{uuid.uuid4().hex[:10]}",
                            session_id=self.session_id,
                            model=SARVAM_STT_MODEL,
                        )
                    except Exception:
                        pass
                self._handle_final_transcript(text, str(detected), detected_confidence)

    def forward_audio(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes:
            return
        chunk_seconds = len(pcm_bytes) / (2 * max(EXOTEL_STREAM_SAMPLE_RATE, 1))
        rms = pcm16_rms(pcm_bytes)
        payload = json.dumps(
            {
                "audio": {
                    "data": base64.b64encode(pcm_bytes).decode("ascii"),
                    "sample_rate": str(EXOTEL_STREAM_SAMPLE_RATE),
                    "encoding": "audio/wav",
                }
            }
        )
        sent = False
        for _ in range(2):
            try:
                upstream = self._ensure_stt_upstream()
                upstream.send(payload)
                sent = True
                break
            except Exception:
                self._close_stt_upstream(upstream if "upstream" in locals() else None)
        if sent:
            with self._lock:
                self._stt_audio_seconds_unbilled += chunk_seconds
                if rms >= 350:
                    self._speech_seconds_since_flush += chunk_seconds
                    self._silence_seconds_since_speech = 0.0
                    self._flush_sent_for_current_pause = False
                elif self._speech_seconds_since_flush > 0:
                    self._silence_seconds_since_speech += chunk_seconds
                    should_flush = (
                        self._silence_seconds_since_speech >= PHONE_STT_SILENCE_FLUSH_SECONDS
                        and self._speech_seconds_since_flush >= PHONE_STT_MIN_SPEECH_SECONDS
                        and not self._flush_sent_for_current_pause
                    )
                    if should_flush:
                        self._flush_sent_for_current_pause = True
                        threading.Thread(target=self._request_stt_flush, daemon=True).start()

    def _handle_partial_transcript(self, text: str, detected_code: str) -> None:
        trimmed = normalize_whitespace(text).strip()
        if not trimmed or is_likely_stt_hallucination(trimmed):
            return
        opening_protection_active, _ = self._opening_barge_in_protection_state()
        with self._lock:
            active_response_id = self._current_response_id
            active_response_text = self._current_response_text or self._latest_assistant_text()
            speech_seconds = self._speech_seconds_since_flush
        if not active_response_id:
            return
        if looks_like_agent_echo(trimmed, active_response_text):
            return
        tokens = stt_word_tokens(trimmed)
        if not tokens:
            return
        if opening_protection_active:
            return
        if len(tokens) < 2 and speech_seconds < 0.25:
            return
        self._cancel_active_playback("barge_in_partial")

    def _handle_final_transcript(
        self, text: str, detected_code: str, detected_confidence: float | None = None
    ) -> None:
        trimmed = normalize_whitespace(text).strip()
        if not trimmed:
            return
        tokens = stt_word_tokens(trimmed)
        language_signal = phone_language_switch_signal(
            trimmed,
            detected_code,
            self.active_language_id,
            self.language_id,
        )
        now = time.time()
        explicit_language_request = bool(explicit_language_request_language_id(trimmed))
        opening_protection_active, opening_started_at = self._opening_barge_in_protection_state()
        allow_short_opening_reply = False
        with self._lock:
            recent_barge_in = self._pending_barge_in_at and (now - self._pending_barge_in_at < 2.0)
            last_speak_at = self._last_agent_speak_start_at
            active_response_id = self._current_response_id
            active_response_text = self._current_response_text or self._latest_assistant_text()
            self._pending_barge_in_at = None
        if active_response_id:
            if looks_like_agent_echo(trimmed, active_response_text):
                self.log_event("stt_dropped", {"reason": "agent_echo", "text": trimmed[:120]})
                return
            if opening_protection_active and not explicit_language_request:
                opening_elapsed = now - opening_started_at if opening_started_at is not None else None
                weak_opening_interrupt = is_weak_opening_interruption(trimmed)
                if weak_opening_interrupt:
                    reason = (
                        "greeting_grace"
                        if opening_elapsed is not None and opening_elapsed < PHONE_GREETING_BARGE_IN_GRACE_SECONDS
                        else "greeting_short"
                    )
                    self.log_event("stt_dropped", {"reason": reason, "text": trimmed[:120]})
                    return
                allow_short_opening_reply = True
            if not recent_barge_in and len(tokens) < 3 and not allow_short_opening_reply:
                self.log_event("stt_dropped", {"reason": "short_during_playback", "text": trimmed[:120]})
                return
            if not recent_barge_in:
                self._cancel_active_playback("barge_in_final")
        elif last_speak_at is not None and now - last_speak_at < 0.9 and len(tokens) <= 2:
            return
        if (
            last_speak_at is not None
            and now - last_speak_at < 2.0
            and looks_like_agent_echo(trimmed, active_response_text)
        ):
            self.log_event("stt_dropped", {"reason": "agent_echo_post_playback", "text": trimmed[:120]})
            return
        if is_stt_hallucination(trimmed):
            self._append_transcript("system", "Dropped transcript hallucination (STT echoed prompt on silence).")
            return
        signal_action = str(language_signal.get("action") or "keep")
        signal_candidate = supported_render_language_id(
            language_signal.get("candidate_language_id") or self.active_language_id
        )
        if signal_action == "drop":
            drop_reason = str(language_signal.get("reason") or "unexpected_language_fragment")
            self.log_event(
                "stt_dropped",
                {
                    "reason": drop_reason,
                    "text": trimmed[:120],
                },
            )
            # We dropped the customer turn AND (above) may have cancelled the
            # agent's in-flight reply on barge-in. Without a re-prompt the call
            # falls to permanent silence (ambience only). Nudge the customer to
            # continue in a language we can actually speak back.
            self._reprompt_after_unspeakable_turn(drop_reason)
            return

        # Stickiness gate: defer a language `switch` for one corroborating turn
        # when the evidence is weak, so a single misrecognized utterance cannot
        # hijack the call language (the Tamil-misfire bug). No latency cost — it
        # only delays an uncertain switch by one turn; replies/audio are unaffected.
        #
        # A switch is gated when EITHER:
        #   - Sarvam reports a confidence below the floor, OR
        #   - confidence is absent AND the evidence is thin: an untrusted reason,
        #     or a trusted reason carried by a very short utterance (<= 2 tokens),
        #     which is exactly how the Tamil misfire slipped through.
        signal_reason = str(language_signal.get("reason") or "")
        if signal_action == "switch" and PHONE_LANGUAGE_SWITCH_MIN_CONFIDENCE > 0 and (
            signal_candidate != supported_render_language_id(self.active_language_id)
        ):
            trusted_reason = signal_reason in PHONE_TRUSTED_SWITCH_REASONS
            if detected_confidence is not None:
                weak = detected_confidence < PHONE_LANGUAGE_SWITCH_MIN_CONFIDENCE
            else:
                weak = (not trusted_reason) or len(tokens) <= 2
            if weak:
                signal_action = "tentative"
                self.log_event(
                    "language_switch_gated",
                    {
                        "candidate": signal_candidate,
                        "reason": signal_reason,
                        "confidence": detected_confidence,
                        "tokens": len(tokens),
                    },
                )

        if signal_action == "tentative" and signal_candidate != supported_render_language_id(self.active_language_id):
            if not self._confirm_tentative_language_candidate(signal_candidate):
                self.log_event(
                    "stt_dropped",
                    {
                        "reason": str(language_signal.get("reason") or "tentative_language_switch"),
                        "text": trimmed[:120],
                    },
                )
                return
        else:
            with self._lock:
                self._clear_pending_language_candidate_locked()
        self._handle_language_detection(detected_code, trimmed, signal_candidate)
        self._append_transcript("customer", trimmed)
        with self._lock:
            self._customer_revision += 1
            self._turn_commit_buffer.append(trimmed)
            pending_text = " ".join(self._turn_commit_buffer).strip()
            timer = self._turn_commit_timer
        if timer is not None:
            timer.cancel()
        next_timer = threading.Timer(phone_turn_commit_delay_seconds(pending_text), self._commit_buffered_turn)
        next_timer.daemon = True
        with self._lock:
            self._turn_commit_timer = next_timer
        next_timer.start()

    def _commit_buffered_turn(self) -> None:
        with self._lock:
            merged = " ".join(self._turn_commit_buffer).strip()
            self._turn_commit_buffer = []
            self._turn_commit_timer = None
            revision = self._customer_revision
        if not merged or is_likely_stt_hallucination(merged):
            return
        threading.Thread(target=self._run_unified_customer_turn, args=(merged, revision), daemon=True).start()

    def _message_history(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {"role": entry["role"], "text": entry["text"]}
                for entry in self.transcript
                if entry.get("role") in {"assistant", "customer"}
            ]

    def _recent_transcript(self) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self.transcript[-6:])

    def _run_unified_customer_turn(self, transcript_text: str, revision: int) -> None:
        turn_started_at = time.time()
        self.log_event("turn_processing_started", {"text": transcript_text[:120], "revision": revision})
        if not self._is_customer_revision_current(revision):
            self.log_event("turn_processing_dropped", {"reason": "stale_before_start", "revision": revision})
            return
        advice, _, lc_error = create_language_coach_review(
            {
                "transcript": transcript_text,
                "current_language_id": self.active_language_id,
                "preferred_language_id": self.language_id,
                "recent_transcript": self._recent_transcript(),
            }
        )
        if lc_error:
            self._append_transcript("system", f"Language coach error: {lc_error}")
            return
        next_language_id = supported_render_language_id(advice.get("suggested_language_id") or self.active_language_id)
        if not self._is_customer_revision_current(revision):
            self.log_event("turn_processing_dropped", {"reason": "stale_after_language_coach", "revision": revision})
            return
        with self._lock:
            self.language_advice = advice
            self.active_language_id = next_language_id
        if advice.get("should_switch") or advice.get("transcript_quality") != "good":
            self._append_transcript("system", f"Language coach: {advice.get('nudge')}")

        text, tool_calls, usage_events, error = run_chat_agent_turn(
            self._message_history(),
            self.voice,
            self.account_number,
            coaching_hints=self.coaching_hints[:5],
            language_advice=advice,
        )
        for usage in usage_events:
            try:
                record_chat_agent_usage(CHAT_MODEL, usage)
            except Exception:
                pass
        if not self._is_customer_revision_current(revision):
            self.log_event("turn_processing_dropped", {"reason": "stale_after_chat", "revision": revision})
            return
        self.log_event(
            "turn_processing_finished",
            {
                "duration_ms": int(round((time.time() - turn_started_at) * 1000)),
                "has_reply": bool(text and text.strip()),
                "tool_call_count": len(tool_calls),
                "error": bool(error),
                "revision": revision,
            },
        )
        if error and not text and not tool_calls:
            self._append_transcript("system", f"Call policy error: {error}")
            return

        self._apply_tool_calls(tool_calls)
        if text and text.strip():
            self._speak_reply(text, tone=collections_reply_tone(text, tool_calls))

    def _apply_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        if not calls:
            return
        with self._lock:
            self.tool_calls = list(calls[::-1]) + self.tool_calls
        for call in calls:
            next_disposition = determine_disposition(call.get("name") or "")
            if next_disposition:
                with self._lock:
                    self.disposition = next_disposition
            if call.get("name") == "get_invoices" and isinstance(call.get("result", {}).get("invoices"), list):
                self.invoices = call["result"]["invoices"]
            if call.get("name") == "get_customer" and call.get("result", {}).get("customer"):
                self.customer = call["result"]["customer"]

    def _run_supervisor_review(self) -> None:
        issues, usage, error = create_supervisor_review(
            {
                "customer": self.customer,
                "invoices": self.invoices,
                "transcript": deepcopy(self.transcript),
                "tool_calls": deepcopy(self.tool_calls),
                "disposition": self.disposition,
                "turn_number": self.turn_number,
            }
        )
        if error:
            self._append_transcript("system", f"Supervisor error: {error}")
            return
        if usage:
            try:
                record_supervisor_usage(SUPERVISOR_MODEL, usage)
            except Exception:
                pass
        if issues:
            update_board(issues)
            self.coaching_hints = [str(issue.get("suggested_fix") or "").strip() for issue in issues if issue.get("suggested_fix")]
            if self.coaching_hints:
                self._append_transcript("system", f"Supervisor coach: {self.coaching_hints[0]}")

    def handle_mark(self, name: str) -> None:
        with self._lock:
            if not name or name != self._current_mark_name:
                return
            serial = self._current_tts_serial
        self._complete_active_playback(serial, name, "exotel_mark")

    def start_stream(self, stream_sid: str | None) -> None:
        with self._lock:
            self.stream_sid = str(stream_sid or "").strip() or self.stream_sid
            if self.started_at is None:
                self.started_at = utc_now()
            self.status = "connected"
        self.log_event("stream_started", {"stream_sid": self.stream_sid})
        self._start_ambience_loop()
        self.start_greeting()

    def update_status(self, payload: dict[str, Any]) -> None:
        self.log_event("status_callback", payload)
        call_status = str(
            payload.get("CallStatus")
            or payload.get("Status")
            or payload.get("status")
            or ""
        ).strip().lower()
        if call_status in {"completed", "failed", "busy", "no-answer", "canceled"}:
            self.finish(call_status)

    def finish(self, reason: str) -> None:
        with self._lock:
            if self._finalized:
                return
            self._finalized = True
            self._stop = True
            self.status = reason or "completed"
            self.ended_at = utc_now()
            timer = self._turn_commit_timer
            self._turn_commit_timer = None
            self._cancel_playback_finish_timer_locked()
        if timer is not None:
            timer.cancel()
        self._close_stt_upstream(reset_billing=True)
        self._cancel_active_playback("call_finished")
        self.log_event("call_finished", {"reason": reason})
        if self.transcript or self.tool_calls:
            # Summary generation is the risky (LLM) step. Isolate it so a summary
            # failure can NEVER prevent the call from being logged to the wrap-up
            # dashboard. Previously a throw here (or below) was swallowed and the
            # whole phone call silently vanished from history.
            try:
                summary, usage, error = create_call_summary(
                    {
                        "customer": self.customer,
                        "invoices": self.invoices,
                        "transcript": deepcopy(self.transcript),
                        "tool_calls": deepcopy(self.tool_calls),
                        "disposition": self.disposition,
                    }
                )
                if not error:
                    self.summary = summary
                if usage:
                    record_supervisor_usage(SUPERVISOR_MODEL, usage)
            except Exception as exc:  # noqa: BLE001
                self.summary = None
                self.log_event("call_summary_error", {"message": str(exc)[:200]})
            try:
                started_at = self.started_at or self.created_at
                duration_sec = max(0, int((self.ended_at - started_at).total_seconds()))
                try:
                    costs = ledger_with_combined(load_ledger())
                except Exception as exc:  # noqa: BLE001
                    self.log_event("call_log_cost_error", {"message": str(exc)[:200]})
                    costs = {"combined": {"estimated_cost_usd": 0.0, "total_tokens": 0}}
                log_payload = {
                    "account_number": self.account_number,
                    "disposition": self.disposition,
                    "transcript": deepcopy(self.transcript),
                    "tool_calls": deepcopy(self.tool_calls),
                    "duration_sec": duration_sec,
                    "cost_usd": costs["combined"]["estimated_cost_usd"],
                    "total_units": costs["combined"]["total_tokens"],
                    "costs": costs,
                    "summary": self.summary or {},
                    "notes": f"Exotel phone demo to {self.target_number}",
                }
                entry = {
                    "id": f"call_{uuid.uuid4().hex[:10]}",
                    "account_number": log_payload["account_number"],
                    "mode": "voice",
                    "disposition": log_payload["disposition"],
                    "transcript": log_payload["transcript"],
                    "tool_calls": log_payload["tool_calls"],
                    "duration_sec": log_payload["duration_sec"],
                    "cost_usd": log_payload["cost_usd"],
                    "total_units": log_payload["total_units"],
                    "mode_cost_usd": (
                        float(log_payload["costs"].get("agent", {}).get("estimated_cost_usd", 0.0))
                        if isinstance(log_payload.get("costs"), dict)
                        else 0.0
                    ),
                    "mode_tokens": (
                        int(log_payload["costs"].get("agent", {}).get("total_tokens", 0))
                        if isinstance(log_payload.get("costs"), dict)
                        else 0
                    ),
                    "costs": log_payload["costs"],
                    "summary": log_payload["summary"],
                    "notes": log_payload["notes"],
                    "timestamp": utc_now_iso(),
                }
                append_jsonl(CALL_LOG_FILE, entry)
                self.log_event("call_logged", {"id": entry["id"], "path": str(CALL_LOG_FILE)})
            except Exception as exc:  # noqa: BLE001
                # Never silently drop a completed call from the wrap-up history.
                self.log_event(
                    "call_log_error",
                    {"message": str(exc)[:200], "path": str(CALL_LOG_FILE)},
                )


def get_phone_call_session(*, session_id: str | None = None, call_sid: str | None = None) -> PhoneCallSession | None:
    with PHONE_CALL_SESSIONS_LOCK:
        if session_id:
            return PHONE_CALL_SESSIONS.get(str(session_id).strip())
        if call_sid:
            mapped = PHONE_CALL_SESSIONS_BY_CALL_SID.get(str(call_sid).strip())
            return PHONE_CALL_SESSIONS.get(mapped) if mapped else None
    return None


def prune_stale_phone_call_sessions(timeout_seconds: int = 90) -> None:
    with PHONE_CALL_SESSIONS_LOCK:
        sessions = list(PHONE_CALL_SESSIONS.values())
    now = utc_now()
    for session in sessions:
        snapshot = session.snapshot()
        if not snapshot["active"]:
            continue
        if snapshot["started_at"]:
            continue
        created_at = session.created_at
        if (now - created_at).total_seconds() >= timeout_seconds:
            session.finish("startup_timeout")


def get_single_active_phone_call_session() -> PhoneCallSession | None:
    prune_stale_phone_call_sessions()
    with PHONE_CALL_SESSIONS_LOCK:
        active_sessions = [session for session in PHONE_CALL_SESSIONS.values() if session.snapshot()["active"]]
    if len(active_sessions) == 1:
        return active_sessions[0]
    return None


def has_active_phone_call_session() -> bool:
    prune_stale_phone_call_sessions()
    with PHONE_CALL_SESSIONS_LOCK:
        for session in PHONE_CALL_SESSIONS.values():
            snapshot = session.snapshot()
            if snapshot["active"]:
                return True
    return False


@app.post("/api/tool/<tool_name>")
def tool_route(tool_name: str):
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return error_json(f"Unknown tool {tool_name}", 404)

    payload = request.get_json(silent=True) or {}
    result = handler(payload)
    log_tool_action(tool_name, payload, result)
    return success_json(result)


@app.post("/api/call/log")
def call_log():
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode") or "voice").strip().lower()
    if mode not in {"voice", "chat"}:
        mode = "voice"
    entry = {
        "id": f"call_{uuid.uuid4().hex[:10]}",
        "account_number": payload.get("account_number", DEFAULT_ACCOUNT_ID),
        "mode": mode,
        "disposition": payload.get("disposition"),
        "transcript": payload.get("transcript", []),
        "tool_calls": payload.get("tool_calls", []),
        "duration_sec": int(payload.get("duration_sec", 0) or 0),
        "cost_usd": float(payload.get("cost_usd", 0.0) or 0.0),
        "total_units": int(payload.get("total_units", 0) or 0),
        "mode_cost_usd": float(payload.get("mode_cost_usd", 0.0) or 0.0),
        "mode_tokens": int(payload.get("mode_tokens", 0) or 0),
        "costs": payload.get("costs") or {},
        "summary": payload.get("summary") or {},
        "notes": payload.get("notes", ""),
        "timestamp": utc_now_iso(),
    }
    append_jsonl(CALL_LOG_FILE, entry)
    return success_json({"ok": True, "entry_id": entry["id"]})


@app.get("/api/call/history")
def call_history():
    return success_json({"history": load_call_history()})


def collections_reply_tone(assistant_text: str, tool_calls: list[dict[str, Any]]) -> str:
    """Pick an eleven_v3 delivery tone for a collections reply from its content.
    Empathetic when we're acknowledging hardship / safety / escalation; firm when
    pressing on overdue/refusal/shipment-stop leverage; professional otherwise."""
    names = {str((tc or {}).get("name") or "") for tc in (tool_calls or [])}
    if names & {"transfer_to_human"}:
        return "empathetic"
    lowered = (assistant_text or "").casefold()
    empathetic_cues = (
        "cash flow", "samajh sakta", "समझ सकता", "समजू शकतो", "afsos", "अफ़सोस",
        "sorry", "माफ", "क्षमा", "understand", "no problem", "koi baat nahi",
        "काही हरकत", "call you back", "call karoon", "दोबारा",
    )
    firm_cues = (
        "overdue", "shipment", "stop", "credit-worthi", "priorit", "must", "policy",
        "agreed terms", "next 2 business days", "clear date", "hold",
    )
    if any(cue in lowered for cue in empathetic_cues):
        return "empathetic"
    if any(cue in lowered for cue in firm_cues):
        return "firm"
    return "professional"


@app.post("/api/chat/turn")
def chat_turn():
    payload = request.get_json(silent=True) or {}
    raw_messages = payload.get("messages") or []
    if not isinstance(raw_messages, list):
        return error_json("messages must be a list of {role, text} entries.")
    transcript_text = str(payload.get("transcript") or "").strip()
    if not transcript_text:
        for entry in reversed(raw_messages):
            if isinstance(entry, dict) and str(entry.get("role") or "") == "customer":
                transcript_text = str(entry.get("text") or "").strip()
                if transcript_text:
                    break
    messages = collapse_trailing_customer_messages(raw_messages, transcript_text)
    account_number = str(payload.get("account_number") or DEFAULT_ACCOUNT_ID)
    voice = str(payload.get("voice") or DEFAULT_REALTIME_VOICE)

    coaching_hints = payload.get("coaching_hints") or []
    if not isinstance(coaching_hints, list):
        coaching_hints = []
    language_advice = payload.get("language_advice") if isinstance(payload.get("language_advice"), dict) else None

    text, tool_calls, usage_events, error = run_chat_agent_turn(
        messages,
        voice,
        account_number,
        coaching_hints=coaching_hints,
        language_advice=language_advice,
    )
    if error and not text and not tool_calls:
        return error_json(error, 500)

    costs = ledger_with_combined(load_ledger())
    for usage in usage_events:
        costs = record_chat_agent_usage(CHAT_MODEL, usage)

    model_label = CHAT_MODEL if usage_events else DETERMINISTIC_CHAT_MODEL
    return success_json(
        {
            "assistant_text": text,
            "tool_calls": tool_calls,
            "costs": costs,
            "model": model_label,
            "tone": collections_reply_tone(text, tool_calls),
        }
    )


@app.post("/api/turn/customer")
def customer_turn_unified():
    # Unified per-turn endpoint: runs the deterministic language coach inline
    # and immediately produces the approved next-utterance via the policy
    # engine. Saves one HTTP roundtrip per voice turn vs. calling
    # /api/language/detect followed by /api/chat/turn from the browser.
    payload = request.get_json(silent=True) or {}

    transcript_text = str(payload.get("transcript", "") or "")
    current_language_id = str(payload.get("current_language_id") or DEFAULT_LANGUAGE_ID)
    preferred_language_id = str(payload.get("preferred_language_id") or DEFAULT_LANGUAGE_ID)
    recent_transcript = payload.get("recent_transcript") or []

    advice, _, lc_error = create_language_coach_review(
        {
            "transcript": transcript_text,
            "current_language_id": current_language_id,
            "preferred_language_id": preferred_language_id,
            "recent_transcript": recent_transcript,
        }
    )
    if lc_error:
        return error_json(lc_error, 500)

    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return error_json("messages must be a list of {role, text} entries.")
    account_number = str(payload.get("account_number") or DEFAULT_ACCOUNT_ID)
    voice = str(payload.get("voice") or DEFAULT_REALTIME_VOICE)
    coaching_hints = payload.get("coaching_hints") or []
    if not isinstance(coaching_hints, list):
        coaching_hints = []

    text, tool_calls, usage_events, error = run_chat_agent_turn(
        messages,
        voice,
        account_number,
        coaching_hints=coaching_hints,
        language_advice=advice,
    )
    if error and not text and not tool_calls:
        return error_json(error, 500)

    costs = ledger_with_combined(load_ledger())
    for usage in usage_events:
        costs = record_chat_agent_usage(CHAT_MODEL, usage)

    model_label = CHAT_MODEL if usage_events else DETERMINISTIC_CHAT_MODEL
    return success_json(
        {
            "advice": advice,
            "assistant_text": text,
            "tool_calls": tool_calls,
            "costs": costs,
            "model": model_label,
            "tone": collections_reply_tone(text, tool_calls),
        }
    )


@app.post("/api/call/summarize")
def call_summarize():
    payload = request.get_json(silent=True) or {}
    summary_payload = {
        "customer": payload.get("customer", {}),
        "invoices": payload.get("invoices", []),
        "transcript": payload.get("transcript", []),
        "tool_calls": payload.get("tool_calls", []),
        "disposition": payload.get("disposition"),
    }

    summary, usage, error = create_call_summary(summary_payload)
    if error:
        return error_json(error, 500)

    costs = ledger_with_combined(load_ledger())
    if usage:
        # Summary work is QA-style review, billed alongside the supervisor agent.
        costs = record_supervisor_usage(SUPERVISOR_MODEL, usage)

    return success_json({"summary": summary, "costs": costs})


@app.post("/api/supervisor/evaluate")
def supervisor_evaluate():
    payload = request.get_json(silent=True) or {}
    turn_number = int(payload.get("turn_number", 0) or 0)

    payload.setdefault(
        "agent_persona",
        persona_for_voice(payload.get("voice") or DEFAULT_REALTIME_VOICE),
    )

    # Hand the supervisor a window of its own recent findings so it can dedupe
    # rather than re-flag the same issue every turn.
    recent_window = 2
    recent_findings: list[dict[str, Any]] = []
    try:
        if SUPERVISOR_FLAGS_FILE.exists():
            lines = SUPERVISOR_FLAGS_FILE.read_text(encoding="utf-8").strip().splitlines()
            for raw in lines[-12:]:
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                turn = int(parsed.get("turn_number", 0) or 0)
                if turn_number - turn <= recent_window:
                    recent_findings.append(
                        {
                            "title": parsed.get("title"),
                            "category": parsed.get("category"),
                            "turn_number": turn,
                        }
                    )
    except OSError:
        recent_findings = []
    payload.setdefault("recent_findings", recent_findings)

    issues, usage, error = create_supervisor_review(payload)
    if error:
        return error_json(error, 500)

    # Safety net: drop anything the supervisor still re-raised after coaching.
    recent_keys = {
        f"{(f.get('title') or '').strip().lower()}::{(f.get('category') or '').strip().lower()}"
        for f in recent_findings
    }
    issues = [
        issue
        for issue in issues
        if f"{(issue.get('title') or '').strip().lower()}::{(issue.get('category') or '').strip().lower()}"
        not in recent_keys
    ]

    for issue in issues:
        append_jsonl(SUPERVISOR_FLAGS_FILE, issue)

    board = update_board(issues)
    costs = ledger_with_combined(load_ledger())
    if usage:
        costs = record_supervisor_usage(SUPERVISOR_MODEL, usage)

    return success_json(
        {
            "issues": issues,
            "board": board,
            "costs": costs,
            "turn_number": turn_number,
        }
    )


@app.post("/api/language/detect")
def language_detect():
    payload = request.get_json(silent=True) or {}
    advice, usage, error = create_language_coach_review(payload)
    if error:
        return error_json(error, 500)

    costs = ledger_with_combined(load_ledger())
    if usage:
        costs = record_language_coach_usage(LANGUAGE_COACH_MODEL, usage)

    return success_json(
        {
            "advice": advice,
            "costs": costs,
        }
    )


@app.get("/api/supervisor/issues")
def supervisor_issues():
    return success_json(load_board())


@app.patch("/api/supervisor/issues/<issue_id>")
def supervisor_issue_update(issue_id: str):
    payload = request.get_json(silent=True) or {}
    target_status = str(payload.get("status", "")).lower()
    if target_status not in {"new", "reviewing", "accepted", "dismissed"}:
        return error_json("Status must be one of new, reviewing, accepted, dismissed.")

    board = load_board()
    found_issue = None
    extracted_issue = None

    for column in board.get("columns", []):
        remaining = []
        for issue in column.get("issues", []):
            if issue.get("id") == issue_id:
                extracted_issue = issue
            else:
                remaining.append(issue)
        column["issues"] = remaining

    if extracted_issue:
        extracted_issue["status"] = target_status
        extracted_issue["updated_at"] = utc_now_iso()
        for column in board.get("columns", []):
            if column.get("id") == target_status:
                column["issues"].insert(0, extracted_issue)
                found_issue = extracted_issue
                break

    if not found_issue:
        return error_json(f"Issue {issue_id} not found.", 404)

    board["updated_at"] = utc_now_iso()
    write_json(BOARD_FILE, board)
    return success_json({"ok": True, "issue": found_issue, "board": board})


@app.get("/api/metrics/costs")
def metrics_costs():
    return success_json(ledger_with_combined(load_ledger()))


@app.post("/api/metrics/costs/event")
def metrics_cost_event():
    payload = request.get_json(silent=True) or {}
    source = str(payload.get("source", "")).lower()
    usage_type = str(payload.get("usage_type", "response")).lower()
    usage = payload.get("usage", {}) or {}
    model = str(payload.get("model") or REALTIME_MODEL)
    event_id = str(payload.get("event_id") or "").strip() or None
    session_id = str(payload.get("session_id") or "").strip() or None

    if source == "agent" and usage_type == "response":
        return success_json(record_agent_response_usage(model, usage, event_id=event_id, session_id=session_id))
    if source == "agent" and usage_type == "transcription":
        return success_json(record_agent_transcription_usage(usage, event_id=event_id, session_id=session_id))
    if source in {"voice", "sarvam"} and usage_type == "tts":
        chars = int(payload.get("chars", 0) or usage.get("chars", 0) or 0)
        return success_json(
            record_tts_usage(chars=chars, event_id=event_id, session_id=session_id, model=model)
        )
    if source in {"voice", "sarvam"} and usage_type == "stt":
        seconds = float(payload.get("seconds", 0.0) or usage.get("seconds", 0.0) or 0.0)
        return success_json(
            record_stt_usage(seconds=seconds, event_id=event_id, session_id=session_id, model=model)
        )
    if source == "supervisor":
        return success_json(record_supervisor_usage(model, usage))
    if source == "language_coach":
        return success_json(record_language_coach_usage(model, usage))
    return error_json("Unsupported cost event source or usage_type.")


@app.post("/api/metrics/costs/reset")
def metrics_cost_reset():
    payload = request.get_json(silent=True) or {}
    realtime_model = str(payload.get("model") or REALTIME_MODEL)
    transcription_model = str(payload.get("transcription_model") or REALTIME_TRANSCRIPTION_MODEL)
    ledger = default_ledger(realtime_model=realtime_model, transcription_model=transcription_model)
    write_json(LEDGER_FILE, ledger)
    return success_json(ledger_with_combined(ledger))


@app.post("/api/demo/reset")
def demo_reset():
    return success_json(reset_runtime_state())


def frontend_ready() -> bool:
    return FRONTEND_INDEX_FILE.exists()


@app.get("/")
def frontend_index():
    if frontend_ready():
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")
    return error_json("Frontend build not found. Run `npm run build` in frontend/ first.", 404)


@app.get("/<path:path>")
def frontend_assets(path: str):
    if path.startswith("api/"):
        return error_json("Not found.", 404)
    asset_path = FRONTEND_DIST_DIR / path
    if asset_path.exists() and asset_path.is_file():
        return send_from_directory(FRONTEND_DIST_DIR, path)
    if frontend_ready():
        return send_from_directory(FRONTEND_DIST_DIR, "index.html")
    return error_json("Frontend build not found. Run `npm run build` in frontend/ first.", 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
