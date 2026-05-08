"""Generate backend/data/GROUND_TRUTH.md from backend/data/sap_mock.json.

The GROUND_TRUTH.md document is prepended to every LLM policy-engine
turn as the canonical source of truth. Whenever sap_mock.json changes
(or, post-PoC, when SAP-backed data is refreshed), this script must be
re-run so the doc stays in sync.

Run directly:
    python backend/scripts/generate_ground_truth.py

Imported from `app.py` at startup via `regenerate_ground_truth_doc()`.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
SAP_FILE = DATA_DIR / "sap_mock.json"
GROUND_TRUTH_FILE = DATA_DIR / "GROUND_TRUTH.md"


def _format_currency(amount: int, currency: str = "INR") -> str:
    return f"{currency} {int(amount):,}"


def _date_renderings(iso_date: str) -> list[str]:
    """Return common natural-language renderings of a YYYY-MM-DD date so the
    LLM can recognise any of them as the same canonical value."""
    try:
        parsed = date.fromisoformat(iso_date)
    except ValueError:
        return [iso_date]
    long_form = parsed.strftime("%-d %B %Y") if sys.platform != "win32" else parsed.strftime("%#d %B %Y")
    short_form = parsed.strftime("%-d-%b-%y") if sys.platform != "win32" else parsed.strftime("%#d-%b-%y")
    return [iso_date, long_form, short_form]


def _render_invoice_table(inv: dict[str, Any]) -> str:
    rows = [
        ("Invoice number", f"`{inv.get('invoice_no')}`"),
        ("Invoice type", f"`{inv.get('invoice_type')}`"),
        ("Amount", f"`{_format_currency(int(inv.get('amount') or 0), inv.get('currency') or 'INR')}`"),
        ("Currency", f"`{inv.get('currency') or 'INR'}`"),
        ("Invoice date", f"`{inv.get('invoice_date')}`"),
        ("Due date", f"`{inv.get('due_date')}`"),
        ("Overdue days", f"`{inv.get('overdue_days')}`"),
    ]
    history = inv.get("history") or []
    history_text = " ".join(history) if history else "No prior history logged."
    rows.append(("History", history_text))
    body = "\n".join(f"| {label} | {value} |" for label, value in rows)
    return "| Field | Value |\n|---|---|\n" + body


def generate_doc(sap: dict[str, Any]) -> str:
    customers = sap.get("customers") or {}
    invoices_by_account = sap.get("invoices") or {}
    payment_methods = sap.get("payment_methods") or []
    constants = {
        "promise_date_max_business_days": sap.get("promise_date_max_business_days"),
        "monthly_collection_target_day": sap.get("monthly_collection_target_day"),
        "proof_of_payment_email": sap.get("proof_of_payment_email"),
        "dispositions": sap.get("dispositions") or [],
    }

    default_account = sap.get("default_account_id") or next(iter(customers.keys()), "")
    if default_account not in customers:
        raise SystemExit(
            f"sap_mock.json default_account_id '{default_account}' not present in customers"
        )

    customer = customers[default_account]
    invoices = invoices_by_account.get(default_account) or []
    transfer = customer.get("human_transfer") or {}

    total = sum(int(inv.get("amount") or 0) for inv in invoices)
    currency = (invoices[0].get("currency") if invoices else None) or "INR"

    valid_amounts = sorted({int(inv.get("amount") or 0) for inv in invoices} | {total})
    valid_overdue_days = sorted({int(inv.get("overdue_days") or 0) for inv in invoices})

    date_renderings: list[str] = []
    for inv in invoices:
        for field in ("invoice_date", "due_date"):
            iso = str(inv.get(field) or "")
            if iso:
                renderings = _date_renderings(iso)
                date_renderings.append(
                    f"- `{iso}` / `{renderings[1]}` / `{renderings[2]}` ({field.replace('_', ' ')} {inv.get('invoice_no')})"
                )

    forbidden_channels = (
        "UPI, cheques, debit/credit cards, generic NEFT to other accounts, "
        "Google Pay, PhonePe, Paytm, cash, or any other channel not in the list above."
    )

    payment_rows = "\n".join(
        f"| `{m.get('id')}` | {m.get('label')} | {m.get('details')} |" for m in payment_methods
    )

    invoice_sections = []
    for idx, inv in enumerate(invoices, start=1):
        invoice_sections.append(f"### 2.{idx} Invoice {inv.get('invoice_no')}\n\n{_render_invoice_table(inv)}\n")

    parts = [
        "# GROUND TRUTH — DHL Express India Collections POC",
        "",
        "> This document is the ONLY source of truth for the live DHL collections agent.",
        "> Every customer name, contact, invoice number, amount, due date, overdue-day count,",
        "> history line, payment method, escalation contact, and policy constant the agent speaks",
        "> **must** come from this file (mirrored in `backend/data/sap_mock.json`).",
        ">",
        "> If a fact is not in this document, the agent MUST NOT state it. No rounding, blending,",
        "> averaging, summarising, or \"confident-sounding\" approximations are allowed. When uncertain,",
        "> the agent must omit the number rather than invent one.",
        ">",
        "> **Auto-generated** from `backend/data/sap_mock.json` by",
        "> `backend/scripts/generate_ground_truth.py`. Do not edit by hand — re-run the script after",
        "> changing the SAP fixture (or after a SAP-backed data refresh, post-PoC).",
        "",
        "---",
        "",
        "## 1. Customer of record",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Account number | `{customer.get('account_number')}` |",
        f"| Company name | `{customer.get('company_name')}` |",
        f"| Primary contact | `{customer.get('contact_name')}` |",
        f"| Alternate contact | `{customer.get('alternate_contact_name')}` |",
        f"| Direct phone | `{customer.get('phone')}` |",
        f"| Registered email | `{customer.get('registered_email')}` |",
        f"| Billing city | `{customer.get('billing_city')}` |",
        f"| Payment terms | `{customer.get('payment_terms')}` |",
        "| Language preferences | " + ", ".join(f"`{lang}`" for lang in customer.get("language_preferences") or []) + " |",
        "",
        "Behaviour rules tied to this customer:",
        f"- If `{customer.get('contact_name')}` is unavailable, ask for `{customer.get('alternate_contact_name')}` on the same line — do not improvise other names.",
        "- Never ask the customer for the account number or company name. They are pre-loaded.",
        "",
        "---",
        "",
        "## 2. Outstanding invoices",
        "",
        f"There are **exactly {len(invoices)}** invoices on this account. Total outstanding is **{_format_currency(total, currency)}**.",
        "",
        *invoice_sections,
        "### 2.x Totals",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Number of invoices | `{len(invoices)}` |",
        f"| Total outstanding | `{_format_currency(total, currency)}` |",
        "",
        "---",
        "",
        "## 3. Allowed numeric values the agent may quote",
        "",
        "The agent may speak **only** these currency values:",
        "",
        *[f"- `{value}` ({_format_currency(value, currency)})" for value in valid_amounts],
        "",
        f"The agent may speak only these overdue-day values: {', '.join(f'`{v}`' for v in valid_overdue_days)}.",
        "",
        "The agent may speak only these dates (and equivalent natural-language forms):",
        "",
        *date_renderings,
        "",
        "Any other amount, date, or count is a fabrication and is forbidden.",
        "",
        "---",
        "",
        "## 4. Sanctioned payment methods",
        "",
        "| ID | Label | Details |",
        "|---|---|---|",
        payment_rows,
        "",
        f"Forbidden channels (never mention): {forbidden_channels}",
        "",
        "---",
        "",
        "## 5. Policy constants",
        "",
        "| Constant | Value | Notes |",
        "|---|---|---|",
        f"| Promise-to-pay window | `{constants['promise_date_max_business_days']} business days` | Any date the customer offers further out must be politely refused; ask for a tighter date inside the window. |",
        f"| Soft monthly collection target | `{constants['monthly_collection_target_day']}th of the month` | Try your best to secure payment before this day. |",
        f"| Proof-of-payment email | `{constants['proof_of_payment_email']}` | Where customers email transaction reference + date when claiming \"already paid\". |",
        "",
        "Allowed call dispositions (the only values that may be logged):",
        "",
        *[f"- `{d}`" for d in constants["dispositions"]],
        "",
        "---",
        "",
        "## 6. Human escalation contact",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Name | `{transfer.get('name')}` |",
        f"| Designation | `{transfer.get('designation')}` |",
        f"| Phone | `{transfer.get('phone')}` |",
        "",
        "Trigger criteria for escalation: customer raises an instalment, dispute, or refund query and **insists** on a human; or any safety/distress signal.",
        "",
        "---",
        "",
        "## 7. Conversation flow rules (from KNOWLEDGE_BASE.md §9–11)",
        "",
        "### 7.1 Opening",
        "1. Greet (good morning / afternoon / evening based on local time).",
        f"2. Confirm identity by **name** (\"Am I speaking with {customer.get('contact_name')}?\"). Never ask for account number or company name.",
        f"3. If the contact is unavailable → ask for `{customer.get('alternate_contact_name')}`; if neither is available → ask to be connected to whoever handles accounts payable for the company.",
        "",
        "### 7.2 State purpose",
        "After identity is confirmed:",
        "- State that this is regarding the customer's overdue DHL credit account.",
        f"- State the total outstanding (`{_format_currency(total, currency)}` across `{len(invoices)}` invoices) — do not invent a different total.",
        "- Ask why payment has not been made.",
        "",
        "### 7.3 Branch A — customer agrees to pay",
        "1. Ask for an exact payment date.",
        f"2. If the date is vague or more than {constants['promise_date_max_business_days']} business days out → push back politely and ask for a tighter date.",
        "3. Once captured → confirm commitment and log a `promise-to-pay` disposition.",
        "",
        "### 7.4 Branch B — customer claims already paid",
        "1. Acknowledge politely.",
        f"2. Ask the customer to email proof of payment (transaction reference + date) to `{constants['proof_of_payment_email']}`.",
        "3. State that DHL will verify and revert within 24 hours.",
        "",
        "### 7.5 Branch C — customer cannot pay / objects",
        "Ask the reason and route by reason:",
        "- **Cash flow** → ask for partial payment now or a confirmed date for the full amount; if no date, raise the polite \"account on stop\" lever.",
        "- **Internal approval / PO pending** → ask for the approver name or expected approval date.",
        "- **Invoice not received** → guide to MyBill self-serve portal; if access fails, offer to resend invoice copy to the registered email; verify whether the customer's email matches master records.",
        "- **Dispute on charges** → ask for the dispute reason; log dispute; request that the customer clear any undisputed amount in the meantime.",
        "- **Temporary business issue / payment cycle** → restate that the invoice is overdue and ask for a payment date or payment-cycle date.",
        "",
        "### 7.6 Branch D — wrong contact",
        "Ask to be connected to the accounts payable / finance person; capture any new contact details offered.",
        "",
        "### 7.7 Escalation / closing",
        "- If after probing there is still no commitment, inform the customer that the case will be followed up by collections, confirm preferred contact number/email, and close politely.",
        "- Closing line on commitment: \"Thank you for your cooperation. I have noted your commitment for [Date]. We will follow up if needed.\"",
        "- Closing line without commitment: \"Thank you for your time. We will update our records and do the needful action as per process. Have a good day.\"",
        "",
        "---",
        "",
        "## 8. Tone and abuse rules (from KNOWLEDGE_BASE.md §10 / §6)",
        "",
        "- Tone is **polite but firm** at all times.",
        "- Never argue with the customer.",
        "- Always push for a specific date when capturing a commitment.",
        "- Show empathy.",
        "- Never use abusive, coercive, or threatening language.",
        "- Stay in control of the conversation; listen actively.",
        "",
        "---",
        "",
        "## 9. Hard prohibitions (anti-hallucination)",
        "",
        "The agent MUST NOT, under any circumstance:",
        "",
        "1. State an invoice number that is not one of " + ", ".join(f"`{inv.get('invoice_no')}`" for inv in invoices) + ".",
        "2. State a currency amount other than " + ", ".join(f"`{_format_currency(v, currency)}`" for v in valid_amounts) + " (the per-invoice values and the total).",
        "3. State an overdue-days count other than " + ", ".join(f"`{v}`" for v in valid_overdue_days) + ".",
        f"4. State an invoice date or due date other than the {len(date_renderings)} dates listed in §3.",
        "5. Invent a month or year not present in §3.",
        "6. Mention any payment channel other than the labels listed in §4.",
        "7. Quote a customer name, contact name, phone number, email, or escalation contact other than those listed in §1 and §6.",
        "8. Promise actions outside the toolset (e.g. \"I'll waive the invoice\", \"I'll apply a discount\") — none of those are sanctioned.",
        "9. Round, average, summarise, or otherwise approximate any of the above. Quote exact ground-truth values or omit numbers entirely.",
        "",
        "If the agent is ever uncertain about a number, name, or date, the correct behaviour is to omit it from the reply (or call `get_invoices` / `get_customer` again) — never to guess.",
        "",
    ]

    return "\n".join(parts)


def regenerate_ground_truth_doc() -> Path:
    sap = json.loads(SAP_FILE.read_text(encoding="utf-8"))
    doc = generate_doc(sap)
    GROUND_TRUTH_FILE.write_text(doc, encoding="utf-8")
    return GROUND_TRUTH_FILE


if __name__ == "__main__":
    out = regenerate_ground_truth_doc()
    print(f"Wrote {out}")
