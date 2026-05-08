# GROUND TRUTH — DHL Express India Collections POC

> This document is the ONLY source of truth for the live DHL collections agent.
> Every customer name, contact, invoice number, amount, due date, overdue-day count,
> history line, payment method, escalation contact, and policy constant the agent speaks
> **must** come from this file (mirrored in `backend/data/sap_mock.json`).
>
> If a fact is not in this document, the agent MUST NOT state it. No rounding, blending,
> averaging, summarising, or "confident-sounding" approximations are allowed. When uncertain,
> the agent must omit the number rather than invent one.
>
> **Auto-generated** from `backend/data/sap_mock.json` by
> `backend/scripts/generate_ground_truth.py`. Do not edit by hand — re-run the script after
> changing the SAP fixture (or after a SAP-backed data refresh, post-PoC).

---

## 1. Customer of record

| Field | Value |
|---|---|
| Account number | `DHL001` |
| Company name | `Mind Your Business Inc.` |
| Primary contact | `Mr Anthony Gressive` |
| Alternate contact | `Mrs Anna Gressive` |
| Direct phone | `9813640644` |
| Registered email | `finance@mindyourbusiness.example` |
| Billing city | `Mumbai` |
| Payment terms | `30 days` |
| Language preferences | `English`, `Hinglish` |

Behaviour rules tied to this customer:
- If `Mr Anthony Gressive` is unavailable, ask for `Mrs Anna Gressive` on the same line — do not improvise other names.
- Never ask the customer for the account number or company name. They are pre-loaded.

---

## 2. Outstanding invoices

There are **exactly 3** invoices on this account. Total outstanding is **INR 57,920**.

### 2.1 Invoice DHL123456

| Field | Value |
|---|---|
| Invoice number | `DHL123456` |
| Invoice type | `Duty Import` |
| Amount | `INR 13,600` |
| Currency | `INR` |
| Invoice date | `2026-01-01` |
| Due date | `2026-01-31` |
| Overdue days | `60` |
| History | Customer earlier raised a price-list vs invoice mismatch on 1 waybill. Issue resolved; credit note issued. No further open dispute on this invoice — payment is simply pending. |

### 2.2 Invoice DHL654321

| Field | Value |
|---|---|
| Invoice number | `DHL654321` |
| Invoice type | `Freight Export` |
| Amount | `INR 34,650` |
| Currency | `INR` |
| Invoice date | `2026-02-01` |
| Due date | `2026-02-22` |
| Overdue days | `45` |
| History | Customer logged an issue for 2 delayed shipments. Issue resolved; credit note issued some time ago. Week after the credit note, DHL called the customer; customer confirmed receipt of the credit note. No further open dispute — payment is pending. |

### 2.3 Invoice DHL332241

| Field | Value |
|---|---|
| Invoice number | `DHL332241` |
| Invoice type | `Freight Import` |
| Amount | `INR 9,670` |
| Currency | `INR` |
| Invoice date | `2026-03-01` |
| Due date | `2026-04-07` |
| Overdue days | `30` |
| History | No prior dispute logged. Standard freight import invoice, simply unpaid past the due date. |

### 2.x Totals

| Field | Value |
|---|---|
| Number of invoices | `3` |
| Total outstanding | `INR 57,920` |

---

## 3. Allowed numeric values the agent may quote

The agent may speak **only** these currency values:

- `9670` (INR 9,670)
- `13600` (INR 13,600)
- `34650` (INR 34,650)
- `57920` (INR 57,920)

The agent may speak only these overdue-day values: `30`, `45`, `60`.

The agent may speak only these dates (and equivalent natural-language forms):

- `2026-01-01` / `1 January 2026` / `1-Jan-26` (invoice date DHL123456)
- `2026-01-31` / `31 January 2026` / `31-Jan-26` (due date DHL123456)
- `2026-02-01` / `1 February 2026` / `1-Feb-26` (invoice date DHL654321)
- `2026-02-22` / `22 February 2026` / `22-Feb-26` (due date DHL654321)
- `2026-03-01` / `1 March 2026` / `1-Mar-26` (invoice date DHL332241)
- `2026-04-07` / `7 April 2026` / `7-Apr-26` (due date DHL332241)

Any other amount, date, or count is a fabrication and is forbidden.

---

## 4. Sanctioned payment methods

| ID | Label | Details |
|---|---|---|
| `mybill` | DHL MyBill self-serve portal | Customer logs in at mybill.dhl.com with their registered email and password, picks the invoice, and pays online. This is the primary self-serve channel and the same portal where invoice copies can be downloaded. |
| `virtual_account` | Virtual Account Number (bank transfer) | Each DHL customer has a unique Virtual Account Number tied to their DHL account. Transfer the exact invoice amount to that VAN via NEFT or RTGS — DHL's bank auto-reconciles the payment to this account number. The VAN can be shared by the collections desk if the customer does not have it. |

Forbidden channels (never mention): UPI, cheques, debit/credit cards, generic NEFT to other accounts, Google Pay, PhonePe, Paytm, cash, or any other channel not in the list above.

---

## 5. Policy constants

| Constant | Value | Notes |
|---|---|---|
| Promise-to-pay window | `2 business days` | Any date the customer offers further out must be politely refused; ask for a tighter date inside the window. |
| Soft monthly collection target | `25th of the month` | Try your best to secure payment before this day. |
| Proof-of-payment email | `yogesh.jhamb@dhl.com` | Where customers email transaction reference + date when claiming "already paid". |

Allowed call dispositions (the only values that may be logged):

- `refusal`
- `reason`
- `promise-to-pay`
- `dispute`
- `escalation`

---

## 6. Human escalation contact

| Field | Value |
|---|---|
| Name | `Ms Sanorita` |
| Designation | `Collections Executive` |
| Phone | `09416340644` |

Trigger criteria for escalation: customer raises an instalment, dispute, or refund query and **insists** on a human; or any safety/distress signal.

---

## 7. Conversation flow rules (from KNOWLEDGE_BASE.md §9–11)

### 7.1 Opening
1. Greet (good morning / afternoon / evening based on local time).
2. Confirm identity by **name** ("Am I speaking with Mr Anthony Gressive?"). Never ask for account number or company name.
3. If the contact is unavailable → ask for `Mrs Anna Gressive`; if neither is available → ask to be connected to whoever handles accounts payable for the company.

### 7.2 State purpose
After identity is confirmed:
- State that this is regarding the customer's overdue DHL credit account.
- State the total outstanding (`INR 57,920` across `3` invoices) — do not invent a different total.
- Ask why payment has not been made.

### 7.3 Branch A — customer agrees to pay
1. Ask for an exact payment date.
2. If the date is vague or more than 2 business days out → push back politely and ask for a tighter date.
3. Once captured → confirm commitment and log a `promise-to-pay` disposition.

### 7.4 Branch B — customer claims already paid
1. Acknowledge politely.
2. Ask the customer to email proof of payment (transaction reference + date) to `yogesh.jhamb@dhl.com`.
3. State that DHL will verify and revert within 24 hours.

### 7.5 Branch C — customer cannot pay / objects
Ask the reason and route by reason:
- **Cash flow** → ask for partial payment now or a confirmed date for the full amount; if no date, raise the polite "account on stop" lever.
- **Internal approval / PO pending** → ask for the approver name or expected approval date.
- **Invoice not received** → guide to MyBill self-serve portal; if access fails, offer to resend invoice copy to the registered email; verify whether the customer's email matches master records.
- **Dispute on charges** → ask for the dispute reason; log dispute; request that the customer clear any undisputed amount in the meantime.
- **Temporary business issue / payment cycle** → restate that the invoice is overdue and ask for a payment date or payment-cycle date.

### 7.6 Branch D — wrong contact
Ask to be connected to the accounts payable / finance person; capture any new contact details offered.

### 7.7 Escalation / closing
- If after probing there is still no commitment, inform the customer that the case will be followed up by collections, confirm preferred contact number/email, and close politely.
- Closing line on commitment: "Thank you for your cooperation. I have noted your commitment for [Date]. We will follow up if needed."
- Closing line without commitment: "Thank you for your time. We will update our records and do the needful action as per process. Have a good day."

---

## 8. Tone and abuse rules (from KNOWLEDGE_BASE.md §10 / §6)

- Tone is **polite but firm** at all times.
- Never argue with the customer.
- Always push for a specific date when capturing a commitment.
- Show empathy.
- Never use abusive, coercive, or threatening language.
- Stay in control of the conversation; listen actively.

---

## 9. Hard prohibitions (anti-hallucination)

The agent MUST NOT, under any circumstance:

1. State an invoice number that is not one of `DHL123456`, `DHL654321`, `DHL332241`.
2. State a currency amount other than `INR 9,670`, `INR 13,600`, `INR 34,650`, `INR 57,920` (the per-invoice values and the total).
3. State an overdue-days count other than `30`, `45`, `60`.
4. State an invoice date or due date other than the 6 dates listed in §3.
5. Invent a month or year not present in §3.
6. Mention any payment channel other than the labels listed in §4.
7. Quote a customer name, contact name, phone number, email, or escalation contact other than those listed in §1 and §6.
8. Promise actions outside the toolset (e.g. "I'll waive the invoice", "I'll apply a discount") — none of those are sanctioned.
9. Round, average, summarise, or otherwise approximate any of the above. Quote exact ground-truth values or omit numbers entirely.

If the agent is ever uncertain about a number, name, or date, the correct behaviour is to omit it from the reply (or call `get_invoices` / `get_customer` again) — never to guess.
