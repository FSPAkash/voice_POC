from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from openai import OpenAI

try:
    from keep_alive import init_keep_alive
    KEEP_ALIVE_AVAILABLE = True
except ImportError as _keep_alive_import_error:
    print(f"Keep-alive service not available: {_keep_alive_import_error}", flush=True)
    KEEP_ALIVE_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROMPTS_DIR = BASE_DIR / "prompts"
FRONTEND_DIST_DIR = BASE_DIR.parent / "frontend" / "dist"
FRONTEND_INDEX_FILE = FRONTEND_DIST_DIR / "index.html"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
REALTIME_MODEL = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-mini")
SUPERVISOR_MODEL = os.environ.get("OPENAI_SUPERVISOR_MODEL", "gpt-4.1-mini")
LANGUAGE_COACH_MODEL = os.environ.get("OPENAI_LANGUAGE_COACH_MODEL", "gpt-4.1-mini")
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4.1")
REALTIME_TRANSCRIPTION_MODEL = os.environ.get(
    "OPENAI_REALTIME_TRANSCRIPTION_MODEL",
    "gpt-4o-mini-transcribe",
)
DEFAULT_REALTIME_VOICE = os.environ.get("OPENAI_REALTIME_VOICE", "cedar")

# Voice -> agent persona. The realtime API picks the voice; the prompt must use a
# matching name and pronouns so the customer never hears a male name on a female voice.
VOICE_PERSONAS: dict[str, dict[str, str]] = {
    "marin": {"name": "Priya", "gender": "female", "pronouns": "she/her"},
    "coral": {"name": "Priya", "gender": "female", "pronouns": "she/her"},
    "shimmer": {"name": "Aanya", "gender": "female", "pronouns": "she/her"},
    "sage": {"name": "Meera", "gender": "female", "pronouns": "she/her"},
    "cedar": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "ash": {"name": "Yogesh", "gender": "male", "pronouns": "he/him"},
    "ballad": {"name": "Rohan", "gender": "male", "pronouns": "he/him"},
    "echo": {"name": "Rohan", "gender": "male", "pronouns": "he/him"},
    "verse": {"name": "Arjun", "gender": "male", "pronouns": "he/him"},
    "alloy": {"name": "Aarav", "gender": "neutral", "pronouns": "they/them"},
}
DEFAULT_PERSONA = {"name": "Yogesh", "gender": "male", "pronouns": "he/him"}
SUPPORTED_REALTIME_MODELS = [
    {"id": "gpt-realtime-mini", "label": "GPT Realtime Mini"},
    {"id": "gpt-realtime", "label": "GPT Realtime"},
]


def persona_for_voice(voice: str | None) -> dict[str, str]:
    return VOICE_PERSONAS.get((voice or DEFAULT_REALTIME_VOICE).lower(), DEFAULT_PERSONA)
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

# Per-million-token USD prices. Sources cross-checked against OpenAI's published
# pricing (https://openai.com/api/pricing) for the GA models we use.
# Keep this table in sync; UI reads dollars from /api/metrics/costs only.
DEFAULT_PRICE_TABLE = {
    # gpt-realtime GA (audio + text). Audio rates are the dominant cost driver.
    "gpt-realtime": {
        "text_input_per_million": 4.0,
        "text_cached_input_per_million": 0.4,
        "text_output_per_million": 16.0,
        "audio_input_per_million": 32.0,
        "audio_cached_input_per_million": 0.4,
        "audio_output_per_million": 64.0,
    },
    # gpt-realtime-mini fallback for cheaper sessions if the operator switches.
    "gpt-realtime-mini": {
        "text_input_per_million": 0.6,
        "text_cached_input_per_million": 0.06,
        "text_output_per_million": 2.4,
        "audio_input_per_million": 10.0,
        "audio_cached_input_per_million": 0.3,
        "audio_output_per_million": 20.0,
    },
    # gpt-4o-realtime-preview (older preview voice model — pricier than GA).
    "gpt-4o-realtime-preview": {
        "text_input_per_million": 5.0,
        "text_cached_input_per_million": 2.5,
        "text_output_per_million": 20.0,
        "audio_input_per_million": 40.0,
        "audio_cached_input_per_million": 2.5,
        "audio_output_per_million": 80.0,
    },
    # ASR models bill audio input separately from prompt text and transcript output.
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
DEFAULT_LANGUAGE_ID = "hinglish"
LANGUAGE_REQUEST_ALIASES: dict[str, tuple[str, ...]] = {
    "english": ("english", "angrezi", "inglish"),
    "hinglish": ("hinglish",),
    "hindi": ("hindi", "hindee", "hindhi"),
    "bengali": ("bengali", "bangla", "bangali"),
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

REALTIME_TOOLS = [
    {
        "type": "function",
        "name": "get_customer",
        "description": "Fetch the DHL customer record for the active account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {
                    "type": "string",
                    "description": "DHL account number like DHL001.",
                }
            },
            "required": ["account_number"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_invoices",
        "description": "List the overdue invoices for a DHL customer account.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {
                    "type": "string",
                    "description": "DHL account number like DHL001.",
                }
            },
            "required": ["account_number"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "log_promise_to_pay",
        "description": "Record the promise-to-pay date that the customer commits to.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string"},
                "invoice_no": {"type": "string"},
                "promise_date": {"type": "string"},
                "notes": {"type": "string"},
            },
            "required": ["account_number", "promise_date"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "log_already_paid",
        "description": "Record a claim that an invoice has already been paid.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_no": {"type": "string"},
                "reference_number": {"type": "string"},
                "paid_date": {"type": "string"},
            },
            "required": ["invoice_no"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "resend_invoice",
        "description": "Trigger the mock invoice resend flow for MyBill or email.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_no": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["invoice_no", "email"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "log_dispute",
        "description": "Open a dispute ticket for an invoice and capture notes. The `reason` field MUST contain the customer's specific dispute language verbatim (e.g. 'charges are too high', 'wrong weight billed') — never a generic placeholder like 'dispute raised' or 'customer disputes invoice'. If the customer disputes multiple invoices, call log_dispute once per disputed invoice_no with the same reason text.",
        "parameters": {
            "type": "object",
            "properties": {
                "invoice_no": {"type": "string"},
                "reason": {"type": "string", "description": "Customer's dispute language quoted verbatim from the transcript. Must not be empty or generic."},
                "undisputed_amount": {"type": "number"},
            },
            "required": ["invoice_no", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "update_contact",
        "description": "Capture an alternate contact person, email, or phone number.",
        "parameters": {
            "type": "object",
            "properties": {
                "account_number": {"type": "string"},
                "contact_name": {"type": "string"},
                "phone": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["account_number"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "transfer_to_human",
        "description": "Hand the case to a human DHL collections executive.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "customer_summary": {"type": "string"},
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    },
]


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
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
            if any(re.search(pattern.format(alias=alias_pattern), text) for pattern in command_patterns):
                return language_id
            if any(re.search(pattern.format(alias=alias_pattern), text) for pattern in contextual_patterns):
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


def supported_realtime_models_payload() -> list[dict[str, str]]:
    return [deepcopy(item) for item in SUPPORTED_REALTIME_MODELS]


STT_PROMPT_VOCAB = (
    "DHL, DHL Express India, MyBill, Virtual Account Number, invoice, overdue, "
    "promise to pay, credit note, waybill, AP team, accounts payable, "
    "Hinglish, namaste, dhanyavaad, shukriya, theek hai, accha, paisa."
)


def build_transcription_config(language_id: str | None) -> dict[str, Any]:
    option = language_option(language_id)
    transcription = {
        "model": REALTIME_TRANSCRIPTION_MODEL,
        "prompt": STT_PROMPT_VOCAB,
    }
    if option.get("transcription_language"):
        transcription["language"] = option["transcription_language"]
    return transcription


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
    "aap", "accha", "acha", "haan", "haanji", "hanji", "ji", "hoon", "hai",
    "main", "mein", "mera", "meri", "kar", "karta", "karti", "karunga", "karungi",
    "raha", "rahi", "rahe", "bilkul", "namaste", "theek", "thik", "kya", "kyun",
    "nahi", "nahin", "matlab", "samjha", "samjhi", "dheere", "din", "paisa",
    "paise", "rupee", "rupaye", "thoda", "bahut", "abhi", "phir", "kuch",
    "sahi", "galat", "lekin", "magar", "ya", "aur", "wala", "wali",
}

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
    if any(w in HINGLISH_TOKENS for w in words):
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


RENDERABLE_LANGUAGE_IDS = {"english", "hinglish", "hindi", "bengali"}
DETERMINISTIC_CHAT_MODEL = "deterministic-call-engine"
DETERMINISTIC_SUPERVISOR_MODEL = "deterministic-supervisor"
DETERMINISTIC_LANGUAGE_COACH_MODEL = "deterministic-language-coach"
MAX_PROCESSED_USAGE_EVENT_IDS = 4096
REALTIME_RENDERER_INSTRUCTIONS = (
    "You are a voice renderer for a DHL collections application. "
    "Speak only the exact approved reply supplied by the application. "
    "Never invent facts, never call tools, and never continue the conversation on your own."
)

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


def last_entry(entries: list[dict[str, Any]], role: str) -> dict[str, Any] | None:
    for entry in reversed(entries):
        if entry.get("role") == role:
            return entry
    return None


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


def payment_options_text(language_id: str) -> str:
    if language_id == "hinglish":
        return (
            "Aapke liye do approved payment options hain: DHL MyBill self-serve portal, "
            "ya Virtual Account Number bank transfer."
        )
    if language_id == "hindi":
        return (
            "Aapke liye do approved payment options hain: DHL MyBill self-serve portal, "
            "ya Virtual Account Number bank transfer."
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
            f"Pehli invoice {invoice.get('invoice_no')} hai, {amount} ki, "
            f"jo {overdue_days} din se overdue hai aur due date {due_date} thi."
        )
    if language_id == "hindi":
        return (
            f"Pehli invoice {invoice.get('invoice_no')} hai, {amount} ki, "
            f"jo {overdue_days} din se overdue hai aur due date {due_date} thi."
        )
    if language_id == "bengali":
        return (
            f"Prothom invoice {invoice.get('invoice_no')}, {amount}, "
            f"eta {overdue_days} din overdue ebong due date chhilo {due_date}."
        )
    return (
        f"The first overdue invoice is {invoice.get('invoice_no')} for {amount}, "
        f"which is {overdue_days} days overdue and was due on {due_date}."
    )


def total_summary_text(customer: dict[str, Any], invoices: list[dict[str, Any]], language_id: str) -> str:
    total = format_currency(customer_outstanding(invoices), invoices[0].get("currency", "INR") if invoices else "INR")
    company = str(customer.get("company_name") or "your company")
    if language_id == "hinglish":
        return (
            f"Reason yeh hai ki {company} ke DHL account par total {total} ka outstanding hai "
            f"across {len(invoices)} overdue invoices."
        )
    if language_id == "hindi":
        return (
            f"Main isliye call kar raha hoon kyunki {company} ke DHL account par total {total} ka outstanding hai "
            f"aur {len(invoices)} invoices overdue hain."
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


def opening_purpose_text(customer: dict[str, Any], invoices: list[dict[str, Any]], language_id: str) -> str:
    company = str(customer.get("company_name") or "your company")
    target_invoice = invoices[0] if invoices else {}
    total_text = total_summary_text(customer, invoices, language_id)
    invoice_text = invoice_summary_line(target_invoice, language_id) if target_invoice else ""
    if language_id == "hinglish":
        return (
            f"Thank you for confirming. Mera naam Yogesh hai, main DHL Express India se aapke credit account ke regarding call kar raha hoon. "
            f"{total_text} {invoice_text} Kya aap bata sakte hain ki payment ab tak kyon nahin hui?"
        ).strip()
    if language_id == "bengali":
        return (
            f"Dhonnobad confirm korar jonno. Amar naam Yogesh, ami DHL Express India theke apnar credit account niye call korchi. "
            f"{total_text} {invoice_text} Payment ekhono keno hoyni, bolben?"
        ).strip()
    return (
        f"Thank you for confirming. My name is Yogesh and I am calling from DHL Express India regarding your credit account. "
        f"{total_text} {invoice_text} Could you please help me understand why the payment has not been made yet?"
    ).strip()


def resolved_history_text(invoices: list[dict[str, Any]], language_id: str) -> str:
    interesting = [invoice for invoice in invoices if invoice.get("history")]
    if not interesting:
        if language_id == "hinglish":
            return "In invoices par koi prior dispute logged nahin hai. Sirf payment abhi pending hai."
        if language_id == "bengali":
            return "Ei invoice-gulor upor kono prior dispute nei. Sudhu payment pending."
        return "There are no prior disputes on these invoices. Payment is simply pending."

    lines: list[str] = []
    for invoice in interesting[:2]:
        history = invoice.get("history") or []
        if language_id == "hinglish":
            lines.append(
                f"{invoice.get('invoice_no')} par pehle issue tha, lekin woh resolve ho chuka hai aur credit note issue ho chuka hai."
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
            if language_id == "hinglish":
                lines.append(f"Aapki taraf se credit note receipt bhi confirm ho chuki thi for {invoice.get('invoice_no')}.")
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
    if "today" in lowered:
        return ("today", now)
    if "tomorrow" in lowered:
        target = now + timedelta(days=1)
        return ("tomorrow", target)

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
        "is_affirmative": bool(re.search(r"\b(yes|yeah|yep|yup|haan|ha|ji|speaking|that.s me|this is he|this is she|correct)\b", lowered)),
        "why_calling": bool(re.search(r"\b(you called me|why are you calling|what is this regarding|what is this about|what do you want|what.s this call|kis baare mein)\b", lowered)),
        "payment_options": bool(re.search(r"\b(payment option|payment method|options|how can i pay|how do i pay|how to pay|where do i pay)\b", lowered)),
        "invoice_copy": bool(re.search(r"\b(invoice copy|send.*invoice|resend.*invoice|not received|didn.t receive|haven.t received|don.t have the invoice)\b", lowered)),
        "resolved_issues": bool(re.search(r"\b(resolved issue|resolved issues|conflict|dispute history|past dispute|credit note)\b", lowered)),
        "one_at_a_time": bool(re.search(r"\b(one (?:by one|at a time)|one invoice at a time|line at a time|slowly|slow down|too fast|one (?:request|thing) (?:at a time|line))\b", lowered)),
        "already_paid": bool(re.search(r"\b(already paid|i paid|payment done|payment made|we paid|paid it|paid that|paid this)\b", lowered)),
        "dispute": bool(re.search(r"\b(dispute|wrong charge|billing error|price mismatch|delayed shipment|incorrect amount)\b", lowered)),
        "wrong_contact": bool(re.search(r"\b(not the right person|wrong person|not the right contact|wrong number)\b", lowered)),
        "identity_confusion": bool(re.search(r"\b(who is anthony|i am mark|i am not anthony|this is mark)\b", lowered)),
        "cash_flow": bool(re.search(r"\b(cash flow|no funds|tight on cash|payment cycle|business problem|short on cash|liquidity)\b", lowered)),
        "approval_pending": bool(re.search(r"\b(approval|approver|po pending|purchase order|internal approval|waiting for approval)\b", lowered)),
        "discount": bool(re.search(r"\b(discount|waive|waiver|reduce|reduction)\b", lowered)),
        "asks_timeline": bool(re.search(r"\b(timeline|when do i need to pay|what is my timeline|by when|deadline)\b", lowered)),
        "refusal": bool(re.search(r"\b(don.t call me again|cannot pay|can.t pay|no commitment|refuse|won.t pay)\b", lowered)),
        "human_request": bool(re.search(r"\b(human|live agent|representative|collections executive|real person|talk to (?:a )?person)\b", lowered)),
        "safety": bool(re.search(r"\b(kill myself|suicide|not safe|enemy|tried to kill|hurt myself)\b", lowered)),
        "details": bool(re.search(r"\b(details|what are the details|tell me more|elaborate|explain)\b", lowered)),
        "count_invoices": bool(re.search(r"\b(how many invoice|how many bill|number of invoice|count of invoice|how many are (?:there|outstanding|overdue|pending))\b", lowered)),
        "which_invoice": bool(re.search(r"\b(which invoice|what invoice|invoice numbers?|list (?:the )?invoices?|all invoices)\b", lowered)),
        "amount_query": bool(re.search(r"\b(how much|what.s the amount|total amount|total outstanding|what do i owe|how much do i owe)\b", lowered)),
        "repeat_request": bool(re.search(r"\b(repeat|say again|come again|pardon|sorry,? what)\b", lowered)),
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


def generate_collections_reply(
    messages: list[dict[str, Any]],
    account_number: str,
    voice: str | None,
    language_advice: dict[str, Any] | None = None,
    prior_tool_calls: list[dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]], str]:
    del voice  # conversation policy is deterministic and voice-agnostic here
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
    constants = get_collections_constants()
    target_invoice = invoice_mentioned_in_text(customer_text, invoices) or (invoices[0] if invoices else {})

    if count_entries(entries, "assistant") == 0:
        contact = customer_display_name(customer) or "the accounts payable contact"
        if language_id == "hinglish":
            text = f"Good day, mera naam Yogesh hai aur main DHL Express India se bol raha hoon. Kya main {contact} se baat kar raha hoon?"
        elif language_id == "bengali":
            text = f"Good day, amar naam Yogesh, ami DHL Express India theke bolchi. Ami ki {contact}-er sathe kotha bolchi?"
        else:
            text = f"Good day, my name is Yogesh and I am calling from DHL Express India. Am I speaking with {contact}?"
        return (text, tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["safety"]:
        args = {
            "reason": "Customer expressed serious distress or safety concern during the collections call.",
            "customer_summary": customer_text,
        }
        result = run_tool("transfer_to_human", args)
        tool_calls.append(build_tool_call_entry("transfer_to_human", args, result))
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

    if count_entries(entries, "assistant") <= 1 and signals["is_affirmative"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        total = total_summary_text(customer, invoices, language_id)
        if language_id == "hinglish":
            ask = "Kya aap bata sakte hain ki payment ab tak kyon nahin hui?"
        else:
            ask = "Could you share why payment has not been made yet?"
        return (
            f"{total} {ask}",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["wrong_contact"] or signals["identity_confusion"]:
        if language_id == "hinglish":
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
                else "Isi wajah se main aaj call kar raha hoon, aur samajhna chahta hoon ki payment ab tak kyon nahin hui."
            ),
        ]
        return (" ".join(part for part in parts if part), tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["resolved_issues"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        text = resolved_history_text(invoices, language_id)
        ask = (
            "With those issues resolved, may I ask what is holding the payment back now?"
            if language_id == "english"
            else "Ab jab yeh issues resolve ho chuke hain, kya main pooch sakta hoon ki payment ab tak kyon hold hai?"
        )
        return (f"{text} {ask}", tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["one_at_a_time"]:
        tool_calls.extend(ensure_invoice_tool(prior_tool_calls, account_number))
        line = invoice_summary_line(target_invoice, language_id) if target_invoice else ""
        if language_id == "hinglish":
            return (
                f"Theek hai, ek-ek karke batata hoon. {line} Kya is invoice ke liye payment date confirm kar sakte hain?",
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
        if language_id == "hinglish":
            ask = "Kya aap bata sakte hain ki payment ab tak kyon pending hai?"
        else:
            ask = "Could you share why payment is still pending?"
        return (f"{total} {lines} {ask}".strip(), tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["repeat_request"]:
        if language_id == "hinglish":
            return (
                "Maaf kijiye. Main dheere se dohra deta hoon. " + total_summary_text(customer, invoices, language_id),
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            "Apologies, let me repeat. " + total_summary_text(customer, invoices, language_id),
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["asks_timeline"]:
        if language_id == "hinglish":
            return (
                "As per agreed terms, yeh invoices already overdue hain. "
                "Kya aap next 2 business days ke andar ek specific payment date confirm kar sakte hain?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            "As per the agreed terms, these invoices are already overdue. Could you confirm a specific payment date within the next 2 business days?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["discount"]:
        if language_id == "hinglish":
            return (
                "Discount approve karne ka authority mere paas nahin hai. "
                "Lekin agar aap payment date confirm kar dein, toh main usko note kar sakta hoon. "
                "Kya aap specific date share karenge?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            "I do not have the authority to offer a discount. However, I can note the payment if you confirm a specific date. Could you share that date?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    raw_date, parsed_date = parse_customer_date(customer_text)
    if raw_date:
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
            if language_id == "hinglish":
                return (
                    f"Thank you. Maine note kar liya hai ki payment {raw_date} tak release hogi. "
                    f"Please ensure payment us date tak ho jaye. {recap}",
                    tool_calls,
                    DETERMINISTIC_CHAT_MODEL,
                )
            return (
                f"Thank you. I have noted that payment will be released by {raw_date}. "
                f"Please ensure it is made by then. {recap}",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        if language_id == "hinglish":
            return (
                f"{raw_date} thoda zyada door lag raha hai. "
                "Kya aap next 2 business days ke andar ek specific date confirm kar sakte hain?",
                tool_calls,
                DETERMINISTIC_CHAT_MODEL,
            )
        return (
            f"{raw_date} is a bit too far out. Could you confirm a specific date within the next 2 business days?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["already_paid"]:
        args = {
            "invoice_no": target_invoice.get("invoice_no"),
            "reference_number": "",
            "paid_date": "",
        }
        result = run_tool("log_already_paid", args)
        tool_calls.append(build_tool_call_entry("log_already_paid", args, result))
        email = constants["proof_of_payment_email"]
        if language_id == "hinglish":
            return (
                f"Understood, thank you. Kya aap transaction reference number aur paid date share kar denge? "
                f"Please payment proof {email} par email kar dijiye, aur hum 24 hours ke andar verify karenge.",
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
        if language_id == "hinglish":
            return (
                f"Bilkul. Aap pehle DHL MyBill portal par registered email se login karke invoice dekh sakte hain. "
                f"Agar convenient ho, maine {customer.get('registered_email')} par invoice resend bhi trigger kar diya hai. "
                "Invoice milte hi kindly review karke payment arrange kar dijiye.",
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

    if signals["dispute"]:
        args = {
            "invoice_no": target_invoice.get("invoice_no"),
            "reason": customer_text,
            "undisputed_amount": None,
        }
        result = run_tool("log_dispute", args)
        tool_calls.append(build_tool_call_entry("log_dispute", args, result))
        if language_id == "hinglish":
            return (
                "I understand your concern. Maine isko dispute ke roop mein log kar diya hai aur concerned team ko route kar diya jayega. "
                "Agar koi undisputed amount hai, kya aap usko clear kar sakte hain meanwhile?"
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        return (
            "I understand your concern. I have logged this as a dispute and it will be routed to the concerned team. "
            "If there is any undisputed amount, could you clear that in the meantime?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["payment_options"]:
        extra = (
            "If you need the specific Virtual Account Number, I can have the collections desk share it after the call."
            if language_id == "english"
            else "Agar aapko specific Virtual Account Number chahiye, toh collections desk call ke baad share kar sakti hai."
        )
        return (f"{payment_options_text(language_id)} {extra}", tool_calls, DETERMINISTIC_CHAT_MODEL)

    if signals["cash_flow"]:
        if language_id == "hinglish":
            return (
                "Samajh sakta hoon ki cash flow tight ho sakta hai. "
                "Kya aap partial payment abhi kar sakte hain, ya full payment ke liye ek specific date confirm kar sakte hain?"
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        return (
            "I understand cash flow can be tight. Could you make a partial payment now, or confirm a specific date for the full payment?",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if signals["approval_pending"]:
        if language_id == "hinglish":
            return (
                "Understood. Kya aap approver ka naam aur expected approval date confirm kar sakte hain? "
                "Invoice already overdue hai, isliye request hai ki isko priority di jaye."
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
        if language_id == "hinglish":
            return (
                "Main aapki position note kar raha hoon. Payment abhi bhi overdue hai, "
                "isliye main is case ko human collections executive ko follow-up ke liye transfer kar raha hoon."
            , tool_calls, DETERMINISTIC_CHAT_MODEL)
        return (
            "I respect your position. The payment remains overdue, so I am transferring this case to a human collections executive for follow-up.",
            tool_calls,
            DETERMINISTIC_CHAT_MODEL,
        )

    if language_id == "hinglish":
        return (
            "Thank you for sharing that. Kya aap payment ke liye ek specific date confirm kar sakte hain, ideally next 2 business days ke andar?"
        , tool_calls, DETERMINISTIC_CHAT_MODEL)
    return (
        "Thank you for sharing that. Could you confirm a specific payment date, ideally within the next 2 business days?",
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
        "price_table_version": "openai-pricing-2026-05-22",
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

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    input_details = usage.get("input_tokens_details", {}) or usage.get("input_token_details", {}) or {}
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


def save_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    agent_total = (
        float(ledger["agent"]["response_usage"]["estimated_cost_usd"])
        + float(ledger["agent"]["transcription_usage"]["estimated_cost_usd"])
    )
    ledger["agent"]["estimated_cost_usd"] = round(agent_total, 6)
    ledger["agent"]["total_tokens"] = (
        sum(
            int(value)
            for key, value in ledger["agent"]["response_usage"].items()
            if key.endswith("_tokens")
        )
        + int(ledger["agent"]["transcription_usage"]["audio_input_tokens"])
        + int(ledger["agent"]["transcription_usage"].get("text_input_tokens", 0))
        + int(ledger["agent"]["transcription_usage"]["text_output_tokens"])
    )
    ledger["supervisor"]["total_tokens"] = (
        int(ledger["supervisor"]["text_input_tokens"])
        + int(ledger["supervisor"]["text_cached_input_tokens"])
        + int(ledger["supervisor"]["text_output_tokens"])
    )
    ledger["language_coach"]["total_tokens"] = (
        int(ledger["language_coach"]["text_input_tokens"])
        + int(ledger["language_coach"]["text_cached_input_tokens"])
        + int(ledger["language_coach"]["text_output_tokens"])
    )
    chat_bucket = ledger.setdefault("chat_agent", base_chat_ledger())
    chat_bucket["total_tokens"] = (
        int(chat_bucket.get("text_input_tokens", 0))
        + int(chat_bucket.get("text_cached_input_tokens", 0))
        + int(chat_bucket.get("text_output_tokens", 0))
    )
    ledger["updated_at"] = utc_now_iso()
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
        "price_table_version": ledger.get("price_table_version", "openai-pricing-2026-05-22"),
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

LLM_COLLECTIONS_SYSTEM = """You are the DHL Express India collections agent for an outbound call.
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

OUTPUT (strict JSON, no markdown):
{
  "intent": one of ["greet_identity","state_purpose","explain_invoices","answer_history","payment_options","capture_promise","already_paid","invoice_copy","dispute","cash_flow","approval_pending","wrong_contact","escalate","close","other"],
  "reply": "the exact line the agent will speak, in the chosen language",
  "language": one of ["english","hinglish","hindi","bengali"],
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
    language_directive = {
        "english": "HARD LANGUAGE LOCK: reply 100% in English. Zero Hindi/Hinglish/Bengali words. No 'aap', 'main', 'hoon', 'kar', 'kya', 'haan', 'ji', 'namaste'. Use only English script and English vocabulary.",
        "hinglish": "Reply in Hinglish (romanised Latin script only — never Devanagari).",
        "hindi": "Reply in Hindi.",
        "bengali": "HARD LANGUAGE LOCK: reply entirely in Bengali. First words must already be Bengali.",
    }.get(suggested, f"Reply in {suggested}.")

    ground_truth_doc = load_ground_truth_doc()
    user_prompt = "\n".join([
        f"AGENT PERSONA: {persona['name']} ({persona['gender']}). Voice: {voice or DEFAULT_REALTIME_VOICE}.",
        f"Suggested reply language: {suggested}. Detected customer language: {detected}.",
        language_directive,
        f"Language coach note: {nudge}",
        "",
        "=========================================================",
        "CANONICAL GROUND TRUTH DOCUMENT (the ONLY source of truth):",
        "Every name, invoice number, amount, date, overdue-day count,",
        "history line, payment method, and policy constant you speak",
        "MUST come from the document below or the live grounded context",
        "that follows it. Do not invent, round, blend, or approximate.",
        "If a fact is not in here, omit it from your reply.",
        "=========================================================",
        ground_truth_doc or "(GROUND_TRUTH.md not found — refuse to quote any specific number, name, or date.)",
        "=========================================================",
        "",
        "LIVE GROUNDED CONTEXT FOR THIS CALL (mirrors the doc above):",
        grounded,
        "",
        "TRANSCRIPT SO FAR:",
        *(transcript_lines or ["(no turns yet — this is the very first agent line)"]),
        "",
        "Produce the next agent turn now as JSON per the schema in the system message.",
        "Reminder: every numeric/name/date you state must appear verbatim in the GROUND TRUTH document above. Anything else is a fabrication and forbidden.",
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
            temperature=0.4,
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
                temperature=0.1,
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
                temperature=0.2,
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


_FORBIDDEN_PAYMENT_TERMS = (
    r"\bUPI\b",
    r"\bcheque\b",
    r"\bchecks?\b",
    r"\bcredit card\b",
    r"\bdebit card\b",
    r"\bGoogle Pay\b",
    r"\bPhonePe\b",
    r"\bPaytm\b",
    r"\bcash\b",
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
    return bool(_HINGLISH_LOCK_TOKENS.search(text))


def scrub_forbidden_payment_methods(text: str) -> str:
    cleaned = text
    for pattern in _FORBIDDEN_PAYMENT_TERMS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            cleaned = re.sub(pattern, "DHL MyBill", cleaned, flags=re.IGNORECASE)
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
            "supervisor_model": DETERMINISTIC_SUPERVISOR_MODEL,
            "has_openai_key": bool(OPENAI_API_KEY),
        }
    )


@app.get("/api/bootstrap")
def bootstrap():
    customer = get_customer(DEFAULT_ACCOUNT_ID)
    invoices = get_invoices(DEFAULT_ACCOUNT_ID)
    if not customer:
        return error_json(f"Customer fixture {DEFAULT_ACCOUNT_ID} not found.", 500)

    payload = {
        "account_number": DEFAULT_ACCOUNT_ID,
        "customer": customer,
        "invoices": invoices,
        "total_outstanding": customer_outstanding(invoices),
        "human_agent": HUMAN_AGENT,
        "agent_prompt": compose_agent_instructions(DEFAULT_ACCOUNT_ID, DEFAULT_REALTIME_VOICE),
        "agent_persona": persona_for_voice(DEFAULT_REALTIME_VOICE),
        "realtime_tools": REALTIME_TOOLS,
        "board": load_board(),
        "costs": ledger_with_combined(load_ledger()),
        "config": {
            "realtime_model": REALTIME_MODEL,
            "supported_realtime_models": supported_realtime_models_payload(),
            "realtime_voice": DEFAULT_REALTIME_VOICE,
            "transcription_model": REALTIME_TRANSCRIPTION_MODEL,
            "supervisor_model": DETERMINISTIC_SUPERVISOR_MODEL,
            "language_coach_model": DETERMINISTIC_LANGUAGE_COACH_MODEL,
            "chat_model": DETERMINISTIC_CHAT_MODEL,
            "default_language_id": DEFAULT_LANGUAGE_ID,
            "supported_languages": supported_languages_payload(),
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
    if not OPENAI_API_KEY:
        return error_json("OPENAI_API_KEY is missing on the backend.", 500)

    body = request.get_json(silent=True) or {}
    voice = str(body.get("voice") or DEFAULT_REALTIME_VOICE)
    instructions = REALTIME_RENDERER_INSTRUCTIONS
    model = str(body.get("model") or REALTIME_MODEL)
    language_id = str(body.get("language_id") or DEFAULT_LANGUAGE_ID)

    session_payload = {
        "session": {
            "type": "realtime",
            "model": model,
            "instructions": instructions,
            "audio": {
                "input": {
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.6,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 700,
                        "create_response": False,
                        "interrupt_response": True,
                    },
                    "noise_reduction": {"type": "near_field"},
                    "transcription": build_transcription_config(language_id),
                },
                "output": {"voice": voice},
            },
        }
    }

    response = requests.post(
        f"{OPENAI_BASE_URL}/realtime/client_secrets",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json=session_payload,
        timeout=30,
    )

    if not response.ok:
        return error_json(
            f"Realtime session creation failed: {response.status_code} {response.text}",
            502,
        )

    data = response.json()
    normalized = {
        "client_secret": {
            "value": data.get("value") or data.get("client_secret", {}).get("value"),
            "expires_at": data.get("expires_at") or data.get("client_secret", {}).get("expires_at"),
        },
        "session": data.get("session", {}),
        "model": model,
        "voice": voice,
        "transcription_model": REALTIME_TRANSCRIPTION_MODEL,
        "language_id": language_id,
    }
    return success_json(normalized)


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
    entry = {
        "id": f"call_{uuid.uuid4().hex[:10]}",
        "account_number": payload.get("account_number", DEFAULT_ACCOUNT_ID),
        "disposition": payload.get("disposition"),
        "transcript": payload.get("transcript", []),
        "tool_calls": payload.get("tool_calls", []),
        "summary": payload.get("summary") or {},
        "notes": payload.get("notes", ""),
        "timestamp": utc_now_iso(),
    }
    append_jsonl(CALL_LOG_FILE, entry)
    return success_json({"ok": True, "entry_id": entry["id"]})


@app.post("/api/chat/turn")
def chat_turn():
    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages") or []
    if not isinstance(messages, list):
        return error_json("messages must be a list of {role, text} entries.")
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
