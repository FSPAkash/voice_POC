

---

## Customer

| Field | Value |
|---|---|
| Account number | DHL001 |
| Company name | Mind Your Business Inc. |
| Primary contact | Mr Anthony Gressive |
| Alternate contact | Mrs Anna Gressive |
| Direct phone | 9813640644 |
| Registered email | Anthony@mybiz.com |
| Billing city | Mumbai |
| Payment terms | 30 days |
| Language preferences | English, Hinglish |

---

## Invoices

| Invoice number | Invoice type | Amount (INR) | Currency | Invoice date | Due date | Overdue days | History |
|---|---|---|---|---|---|---|---|
| DHL123456 | Duty Import | 13,600 | INR | 2026-01-01 | 2026-01-31 | 60 | Price-list vs invoice mismatch on 1 waybill. Resolved; credit note issued. No open dispute — payment pending. |
| DHL654321 | Freight Export | 34,650 | INR | 2026-02-01 | 2026-02-22 | 45 | Issue logged for 2 delayed shipments. Resolved; credit note issued. Customer confirmed receipt. No open dispute — payment pending. |
| DHL332241 | Freight Import | 9,670 | INR | 2026-03-01 | 2026-04-07 | 30 | No prior dispute. Standard freight import invoice, unpaid past due. |

### Totals

| Field | Value |
|---|---|
| Number of invoices | 3 |
| Total outstanding | INR 57,920 |

---

## Allowed Values

### Amounts (INR)

| Value |
|---|
| 9,670 |
| 13,600 |
| 34,650 |
| 57,920 |

### Overdue days

| Value |
|---|
| 30 |
| 45 |
| 60 |

### Dates

| Value | Role |
|---|---|
| 2026-01-01 | Invoice date DHL123456 |
| 2026-01-31 | Due date DHL123456 |
| 2026-02-01 | Invoice date DHL654321 |
| 2026-02-22 | Due date DHL654321 |
| 2026-03-01 | Invoice date DHL332241 |
| 2026-04-07 | Due date DHL332241 |

---

## Payment Methods

| ID | Label | Details |
|---|---|---|
| mybill | DHL MyBill self-serve portal | Customer logs in at mybill.dhl.com with registered email and password, picks the invoice, pays online. Primary self-serve channel; invoice copies downloadable here. |
| virtual_account | Virtual Account Number (bank transfer) | Each customer has a unique VAN tied to their DHL account. Transfer exact invoice amount via NEFT or RTGS — DHL's bank auto-reconciles. VAN shared by collections desk if needed. |

---

## Policy Constants

| Constant | Value | Notes |
|---|---|---|
| Promise-to-pay window | 2 business days | Dates beyond window refused; ask tighter date. |
| Soft monthly collection target | 25th of the month | Secure payment before this day. |
| Proof-of-payment email | yogesh.jhamb@dhl.com | Customer emails transaction reference + date when claiming already paid. |

---

## Call Dispositions

| Disposition |
|---|
| refusal |
| reason |
| promise-to-pay |
| dispute |
| escalation |

---

## Escalation Contact

| Field | Value |
|---|---|
| Name | Ms Sanorita |
| Designation | Collections Executive |
| Phone | 09416340644 |
