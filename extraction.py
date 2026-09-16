"""
Extraction logic for the Lease Contract OCR MVP.

Supports two interchangeable providers, both of which receive the raw PDF
and read its pages natively (so scanned / image-only contracts work on
either path - no local OCR step needed):
  - Google Gemini  (genuinely free tier via Google AI Studio - no credit card)
  - OpenAI         (paid, requires API credit)

Responsible for:
  - Pulling raw text out of an uploaded PDF (for the on-screen source preview)
  - Calling the chosen provider with a strict JSON schema to get structured
    Contract -> Item -> Payment/Option data back, using Envoria's controlled
    payment-type / payment-cycle / pre-post-payment / option-type values
  - Applying the "payment start/end falls back to lease start/end" rule

NOTE ON SCOPE: this file defines the extraction contract (prompt + schema +
controlled vocabularies). The Streamlit review UI in app.py has not been
updated to match this schema yet (it still expects the older, flatter field
names and has no Options section) - that's a separate follow-up.
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Controlled vocabularies (Envoria business rules)
#
# The backend/database side should store the CODE (stable, enum-like); the
# LABEL is what a human-facing UI would display. The extraction schema below
# constrains the model to the CODE values only - it must never invent a new
# payment type, cycle, pre/post value, or option type.
# ---------------------------------------------------------------------------

PAYMENT_TYPE_CODES = [
    "LEASE_FEE",
    "LEASE_FEE_INDEX",
    "LEASE_FEE_VARIABLE",
    "NON_LEASE_FEE",
    "INITIAL_DIRECT_COSTS",
    "INCENTIVE_PAYMENT",
    "TAX",
]
PAYMENT_TYPE_LABELS = {
    "LEASE_FEE": "Lease Fee",
    "LEASE_FEE_INDEX": "Lease Fee (Index)",
    "LEASE_FEE_VARIABLE": "Lease Fee (Variable)",
    "NON_LEASE_FEE": "Non-Lease Fee",
    "INITIAL_DIRECT_COSTS": "Initial Direct Costs",
    "INCENTIVE_PAYMENT": "Incentive Payment",
    "TAX": "Tax",
}

PAYMENT_CYCLE_CODES = ["MONTHLY", "QUARTERLY", "ANNUALLY", "BI_ANNUALLY", "ONE_TIME"]
PAYMENT_CYCLE_LABELS = {
    "MONTHLY": "Monthly",
    "QUARTERLY": "Quarterly",
    "ANNUALLY": "Annually",
    "BI_ANNUALLY": "Bi-Annually",
    "ONE_TIME": "One-time",
}

PRE_POST_PAYMENT_CODES = ["PREPAYMENT", "POSTPAYMENT"]
PRE_POST_PAYMENT_LABELS = {
    "PREPAYMENT": "Prepayment",
    "POSTPAYMENT": "Postpayment",
}

OPTION_TYPE_CODES = ["PURCHASE_OPTION", "EXTENSION_OPTION", "TERMINATION_OPTION"]
OPTION_TYPE_LABELS = {
    "PURCHASE_OPTION": "Purchase Option",
    "EXTENSION_OPTION": "Extension Option",
    "TERMINATION_OPTION": "Termination Option",
}

PROVIDERS = ["gemini", "openai"]
DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
}
MODEL_OPTIONS = {
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    "openai": ["gpt-4o-mini", "gpt-4o"],
}

# ---------------------------------------------------------------------------
# JSON schema (plain JSON Schema - shared by both providers)
#   OpenAI: wrapped as {"name", "schema", "strict"} for Structured Outputs
#   Gemini: passed as-is to `response_json_schema`
#
# Contract -> Item[1..n] -> { Payment[0..n], Option[0..n] }
#
# Payments and Options belong to a specific Item, never directly to the
# Contract - this hierarchy must be preserved by the model.
# ---------------------------------------------------------------------------

_NULLABLE_ENUM = lambda codes: {"type": ["string", "null"], "enum": codes + [None]}  # noqa: E731

PAYMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "payment_id": {
            "type": ["string", "null"],
            "description": (
                "Identifier for this payment SERIES (not one cash transaction). "
                "A recurring monthly payment is ONE series, not 12 records. Use "
                "an ID only if the document states one; otherwise null."
            ),
        },
        "payment_type": {
            **_NULLABLE_ENUM(PAYMENT_TYPE_CODES),
            "description": (
                "Classify by MEANING, not keyword matching. LEASE_FEE = the "
                "standard fixed contractual rent (negative values allowed, e.g. "
                "a temporary rent reduction). LEASE_FEE_INDEX = payment tied to "
                "an index/rate (CPI, inflation) - an ordinary fixed rent "
                "increase is NOT this, only an explicit index/rate mechanism "
                "is. LEASE_FEE_VARIABLE = depends on turnover, sales, usage, or "
                "performance. NON_LEASE_FEE = service charges, maintenance, "
                "utilities, other non-lease components. INITIAL_DIRECT_COSTS = "
                "incremental costs of obtaining the lease (commission, broker "
                "fees) - only classify here with clear evidence. "
                "INCENTIVE_PAYMENT = amounts/costs the lessor grants or assumes "
                "for the lessee. TAX = a separately identified tax component "
                "(e.g. VAT) - do not tag an entire payment as TAX just because "
                "VAT is mentioned; only the tax component itself. Use null if "
                "the type cannot be determined - never guess."
            ),
        },
        "payment_cycle": {
            **_NULLABLE_ENUM(PAYMENT_CYCLE_CODES),
            "description": (
                "Frequency of this payment series. Map 'every month'/'monthly' "
                "-> MONTHLY, 'every 3 months'/'quarterly' -> QUARTERLY, "
                "'once a year'/'annually'/'yearly' -> ANNUALLY, 'every six "
                "months'/'semi-annually'/'twice a year' -> BI_ANNUALLY, "
                "'one-off'/'once'/'single payment' -> ONE_TIME. Only these five "
                "values are allowed. If the frequency cannot be determined, use "
                "null instead of inventing one."
            ),
        },
        "payment_start": {
            "type": ["string", "null"],
            "description": "ISO format YYYY-MM-DD. Start of this payment series.",
        },
        "payment_end": {
            "type": ["string", "null"],
            "description": "ISO format YYYY-MM-DD. End of this payment series.",
        },
        "pre_post_payment": {
            **_NULLABLE_ENUM(PRE_POST_PAYMENT_CODES),
            "description": (
                "Whether payment is due at the start or end of each period. "
                "'payable in advance' / 'at the beginning of each month' / "
                "'first day of each period' -> PREPAYMENT. 'payable in "
                "arrears' / 'at the end of each month' / 'end of the period' "
                "-> POSTPAYMENT. Use null if the document does not say."
            ),
        },
        "payment_value": {
            "type": ["number", "null"],
            "description": (
                "The recurring (or one-time) amount for this series, in the "
                "item's currency. Negative values are valid for LEASE_FEE "
                "concessions/reductions."
            ),
        },
        "confidence": {
            "type": ["number", "null"],
            "description": (
                "Your confidence in this payment series as a whole, 0.0-1.0. "
                "Null if you cannot meaningfully assess it."
            ),
        },
        "review_required": {
            "type": "boolean",
            "description": (
                "True if any field in this payment series is uncertain, "
                "ambiguous, or could not be confidently determined (e.g. "
                "unclear table, payment not clearly assignable to this item, "
                "illegible amount). False only when the series is clear."
            ),
        },
    },
    "required": [
        "payment_id",
        "payment_type",
        "payment_cycle",
        "payment_start",
        "payment_end",
        "pre_post_payment",
        "payment_value",
        "confidence",
        "review_required",
    ],
}

OPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "option_id": {"type": ["string", "null"]},
        "option_type": {
            **_NULLABLE_ENUM(OPTION_TYPE_CODES),
            "description": (
                "Classify by MEANING even when the contract doesn't use these "
                "exact words. PURCHASE_OPTION: right/option to buy the asset "
                "(purchase right, right to acquire ownership). "
                "EXTENSION_OPTION: right to extend/renew the lease (renewal "
                "right, right to continue). TERMINATION_OPTION: right to end "
                "the lease early (break clause, cancellation right, early "
                "termination). Use null if a clause seems option-like but the "
                "type genuinely cannot be determined."
            ),
        },
        "valid_from": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD."},
        "valid_to": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD."},
        "expected_exercise_date": {
            "type": ["string", "null"],
            "description": "ISO YYYY-MM-DD, only if the document states or implies one.",
        },
        "characteristic_value": {
            "type": ["string", "null"],
            "description": (
                "The option's key term as stated, e.g. a purchase price, an "
                "extension length ('3 years'), or a notice period - kept as "
                "text since its shape varies by option type."
            ),
        },
        "reasonably_certain": {
            "type": ["boolean", "null"],
            "description": (
                "DO NOT set this merely because an option exists. Leave null "
                "unless the document itself explicitly states the lessee's "
                "intent or certainty regarding exercise. The existence of an "
                "option is NOT evidence that exercise is reasonably certain - "
                "this is an accounting assessment, not an extraction fact, and "
                "is normally left for the user to determine."
            ),
        },
        "confidence": {"type": ["number", "null"]},
        "review_required": {
            "type": "boolean",
            "description": "True if the option's type, dates, or terms are unclear.",
        },
    },
    "required": [
        "option_id",
        "option_type",
        "valid_from",
        "valid_to",
        "expected_exercise_date",
        "characteristic_value",
        "reasonably_certain",
        "confidence",
        "review_required",
    ],
}

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "contract": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "contract_number": {"type": ["string", "null"]},
                "contract_name": {"type": ["string", "null"]},
                "entity_company_code": {
                    "type": ["string", "null"],
                    "description": "The organization / lessee entity party to the contract.",
                },
            },
            "required": ["contract_number", "contract_name", "entity_company_code"],
        },
        "items": {
            "type": "array",
            "description": (
                "One entry per leased asset/item. A contract with three "
                "vehicles has three items - never merge their payments or "
                "options into one shared list."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "item_number": {"type": ["string", "null"]},
                    "asset_name": {"type": ["string", "null"]},
                    "asset_class": {"type": ["string", "null"]},
                    "currency": {
                        "type": ["string", "null"],
                        "description": "ISO 4217 code, e.g. EUR, USD. Infer from symbols if needed.",
                    },
                    "lease_start": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD."},
                    "lease_end": {"type": ["string", "null"], "description": "ISO YYYY-MM-DD."},
                    "interest_rate": {
                        "type": ["number", "null"],
                        "description": "Percentage, e.g. 4.5 for 4.5%. Null if not mentioned.",
                    },
                    "payments": {
                        "type": "array",
                        "description": (
                            "This item's payment series (see PAYMENT_TYPE_CODES / "
                            "PAYMENT_CYCLE_CODES). Keep distinct series separate "
                            "even if they belong to the same item - e.g. a base "
                            "Lease Fee, an indexed Lease Fee starting later, and "
                            "a Non-Lease Fee service charge are THREE series, "
                            "never merged."
                        ),
                        "items": PAYMENT_SCHEMA,
                    },
                    "options": {
                        "type": "array",
                        "description": "This item's Purchase/Extension/Termination options, if any.",
                        "items": OPTION_SCHEMA,
                    },
                },
                "required": [
                    "item_number",
                    "asset_name",
                    "asset_class",
                    "currency",
                    "lease_start",
                    "lease_end",
                    "interest_rate",
                    "payments",
                    "options",
                ],
            },
        },
        "evidence": {
            "type": "array",
            "description": (
                "One entry per field you filled in, saying where in the "
                "document that value came from. Used to highlight the source "
                "passage in the original PDF for the reviewer."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "Dotted path of the field this supports, e.g. "
                            "'contract.entity_company_code', "
                            "'items[0].lease_start', "
                            "'items[0].payments[1].payment_value', "
                            "'items[0].options[0].option_type'."
                        ),
                    },
                    "page": {
                        "type": "integer",
                        "description": "1-based page number the value was read from.",
                    },
                    "quote": {
                        "type": "string",
                        "description": (
                            "Short VERBATIM snippet copied exactly from the "
                            "document (a few words, max ~10) containing the "
                            "value. Must match the document's own wording "
                            "character-for-character so it can be located on "
                            "the page - do not paraphrase, translate or "
                            "reformat it."
                        ),
                    },
                },
                "required": ["field", "page", "quote"],
            },
        },
    },
    "required": ["contract", "items", "evidence"],
}

SYSTEM_PROMPT = """\
You are a precise data-extraction assistant for lease / leasing contracts, \
feeding a lease accounting (IFRS 16) system. Read the contract document \
supplied by the user and return ONLY the structured data requested by the \
JSON schema.

Core hierarchy - this must be preserved exactly:
  Contract
    -> Item (one per leased asset; a contract with several assets has
       several items - never flatten their data into one shared list)
        -> Payments (payment series belonging to that item)
        -> Options (Purchase / Extension / Termination options belonging
           to that item)
A payment or option belongs to the specific item it relates to. If a
contract covers three vehicles, determine - whenever the document allows
it - which payments and options belong to which vehicle. Do not merge
everything into one generic list.

General rules:
- Extract only information that is actually present in the document. Use
  null for anything not stated - never invent or guess a value.
- Normalize all dates to ISO format YYYY-MM-DD.
- interest_rate is a plain percentage number (e.g. 4.5 for "4.5%"). Null if
  no rate is mentioned.
- currency must be an ISO 4217 code (e.g. EUR, USD, GBP). Infer it from
  currency symbols (e.g. "€" -> EUR, "$" -> USD) if the code itself is
  not written out.
- item_number / contract_number / payment_id / option_id: use an ID only if
  the document states one explicitly; otherwise null.
- Keep all extracted text values (names, ids) in their original language and
  spelling - do not translate them.
- When something is genuinely unclear (illegible number, ambiguous table,
  a payment that cannot confidently be assigned to one item, handwritten
  changes, low-quality scan), do not force a classification: leave the
  affected field(s) null and set that payment's/option's review_required
  to true. Accuracy and traceability outweigh forcing an answer.

Payment series (each belongs to one item):
- A payment_id identifies a SERIES, not an individual cash transaction. A
  contract stating "EUR 5,000 monthly, Jan-Dec 2026" is ONE series with
  payment_cycle=MONTHLY and payment_value=5000 - do NOT expand it into 12
  separate records.
- payment_type must be exactly one of the controlled values, chosen by
  MEANING, not by keyword-spotting:
    LEASE_FEE            - the standard fixed contractual rent/lease
                            payment (base rent, monthly rent, fixed rent).
                            Included in the Right-of-Use asset and Lease
                            Liability. Negative values are allowed (e.g. a
                            temporary rent reduction/concession).
    LEASE_FEE_INDEX       - payment that changes based on an index or rate
                            (CPI-linked, inflation-linked rent). Only use
                            this when the contract provides an explicit
                            index/rate mechanism - an ordinary fixed rent
                            step-up is still LEASE_FEE, not this.
    LEASE_FEE_VARIABLE    - payment based on turnover, sales, usage, or
                            performance (not fixed, not index-based).
    NON_LEASE_FEE         - separately identifiable non-lease services:
                            service charges, maintenance, utilities,
                            operating costs.
    INITIAL_DIRECT_COSTS  - incremental costs of obtaining the lease
                            (commission, broker fees, directly attributable
                            legal fees). Only use this with clear evidence
                            the cost is directly attributable to obtaining
                            the lease.
    INCENTIVE_PAYMENT     - payments or cost assumptions granted BY THE
                            LESSOR TO THE LESSEE (lease incentives,
                            relocation reimbursement, lessor-funded
                            rent-free period). Distinguish this from an
                            ordinary negative LEASE_FEE where possible.
    TAX                   - a separately identified tax component (e.g.
                            VAT). Do not tag an entire payment as TAX just
                            because VAT is mentioned somewhere; only the
                            identifiable tax component itself.
  If the payment does not clearly fit one of these, use null and set
  review_required=true rather than guessing.
- payment_cycle is exactly one of: MONTHLY, QUARTERLY, ANNUALLY,
  BI_ANNUALLY, ONE_TIME. Map "every month"/"monthly" -> MONTHLY, "every 3
  months"/"quarterly" -> QUARTERLY, "once a year"/"yearly" -> ANNUALLY,
  "every six months"/"semi-annually"/"twice a year" -> BI_ANNUALLY,
  "one-off"/"once"/"single payment" -> ONE_TIME. Never invent another
  cycle value - if unclear, use null.
- pre_post_payment is PREPAYMENT ("payable in advance", "beginning of each
  month/period") or POSTPAYMENT ("payable in arrears", "end of each
  month/period"). Use null if the document does not say.
- If a payment's own start/end date is not explicitly stated, leave
  payment_start/payment_end as null (the application defaults them to the
  item's lease_start/lease_end - do not copy the lease dates in yourself).

Options (each belongs to one item; types: PURCHASE_OPTION,
EXTENSION_OPTION, TERMINATION_OPTION):
- Recognize option language even when the contract does not use these
  exact terms:
    PURCHASE_OPTION    - right/option to purchase the asset, right to buy,
                          right to acquire ownership at/around lease end.
    EXTENSION_OPTION    - right to extend or renew the lease, renewal
                          right, right to continue the lease.
    TERMINATION_OPTION - right to terminate early, break clause,
                          cancellation right, early termination right.
  Classify by the MEANING of the clause, not by exact wording.
- characteristic_value holds the option's key stated term as text (a
  purchase price, an extension length such as "3 years", a notice period),
  whichever applies to that option type.
- reasonably_certain is an ACCOUNTING ASSESSMENT, not an extraction fact.
  The mere existence of an option does NOT make exercise reasonably
  certain. Leave this null unless the document itself explicitly states
  the lessee's certainty or intent to exercise. Example: "the lessee has
  the option to extend the lease for another three years" gives you
  option_type=EXTENSION_OPTION and characteristic_value="3 years", but
  reasonably_certain stays null - it is not automatically true.

Evidence (important for the reviewer's side-by-side view):
- For EVERY field you fill in with a non-null value, add one entry to the
  `evidence` array recording where in the document you read it.
- `field` is the dotted path, e.g. "contract.entity_company_code",
  "items[0].lease_end", "items[0].payments[1].payment_value",
  "items[0].options[0].option_type".
- `page` is the 1-based page number.
- `quote` must be a SHORT, VERBATIM snippet copied character-for-character
  from the document - a few words (max ~10) that contain the value. It is
  used to find and highlight the passage on the rendered page, so it must
  match the document's own wording exactly. Do NOT paraphrase, translate,
  reformat dates/numbers, or quote text that isn't literally on the page.
  For example, if the document says "am 01.04.2017(Datum) in Chemnitz" and
  you set lease_start to "2017-04-01", the quote should be "01.04.2017",
  not "2017-04-01".
- Do not add evidence entries for fields you left null.
"""


# ---------------------------------------------------------------------------
# PDF text extraction (used for the on-screen "source document" preview only)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_obj) -> str:
    """Extract raw text from a PDF file-like object."""
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text).strip()


# ---------------------------------------------------------------------------
# Provider: Google Gemini  (free tier - https://aistudio.google.com/apikey)
# ---------------------------------------------------------------------------

def call_gemini_extraction(
    pdf_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_MODELS["gemini"],
) -> dict[str, Any]:
    """Send the PDF directly to Gemini (native document understanding) and
    return the parsed structured JSON. No local text extraction needed -
    Gemini reads the PDF itself, which also works for scanned/image pages.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            SYSTEM_PROMPT
            + "\n\nExtract the structured lease data from this contract document.",
        ],
        config={
            "response_mime_type": "application/json",
            "response_json_schema": EXTRACTION_SCHEMA,
            "temperature": 0,
        },
    )
    return json.loads(response.text)


# ---------------------------------------------------------------------------
# Provider: OpenAI (paid - requires API credit)
# ---------------------------------------------------------------------------

def call_openai_extraction(
    pdf_bytes: bytes,
    api_key: str,
    model: str = DEFAULT_MODELS["openai"],
    filename: str = "contract.pdf",
) -> dict[str, Any]:
    """Send the PDF directly to OpenAI and return the parsed structured JSON.

    Uses Chat Completions' `file` content part with a base64 data URI. OpenAI
    processes both the extracted text *and* a rendered image of each page, so
    this works on scanned / image-only PDFs too - no local OCR needed.
    """
    import base64

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file": {
                            "filename": filename,
                            "file_data": f"data:application/pdf;base64,{b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract the structured lease data from this "
                            "contract document."
                        ),
                    },
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "lease_contract_extraction",
                "strict": True,
                "schema": EXTRACTION_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    return json.loads(content)


# ---------------------------------------------------------------------------
# Post-processing: payment date fallback rule
# ---------------------------------------------------------------------------

def apply_payment_date_fallback(data: dict[str, Any]) -> dict[str, Any]:
    """If a payment has no start/end date, default it to the item's lease dates."""
    for item in data.get("items", []):
        lease_start = item.get("lease_start")
        lease_end = item.get("lease_end")
        for payment in item.get("payments", []):
            if not payment.get("payment_start"):
                payment["payment_start"] = lease_start
            if not payment.get("payment_end"):
                payment["payment_end"] = lease_end
    return data


def extract_contract(
    file_obj,
    provider: str,
    api_key: str,
    model: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Full pipeline: PDF -> provider extraction -> fallback rule applied.

    `provider` is "gemini" or "openai". Returns (structured_data, source_text).
    Both providers get the raw PDF bytes and read the pages themselves, so
    scanned/image-only documents work either way. The source_text is only
    used for the UI preview panel and may legitimately be empty for scans.
    """
    import io

    # Rewind first: Streamlit's UploadedFile object persists across reruns, so
    # a second Scan click on the same file would otherwise read 0 bytes.
    try:
        file_obj.seek(0)
    except (AttributeError, OSError):
        pass

    pdf_bytes = file_obj.read()
    if not pdf_bytes:
        raise ValueError(
            "The uploaded file appears to be empty. Try re-uploading the PDF."
        )

    # Text is pulled out only for the on-screen source preview. Both providers
    # receive the raw PDF, so a scanned/image-only document (empty text here)
    # is still fine.
    text = extract_text_from_pdf(io.BytesIO(pdf_bytes))
    filename = getattr(file_obj, "name", "contract.pdf")

    if provider == "gemini":
        data = call_gemini_extraction(
            pdf_bytes, api_key=api_key, model=model or DEFAULT_MODELS["gemini"]
        )
    elif provider == "openai":
        data = call_openai_extraction(
            pdf_bytes,
            api_key=api_key,
            model=model or DEFAULT_MODELS["openai"],
            filename=filename,
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    data = apply_payment_date_fallback(data)
    return data, text


# ---------------------------------------------------------------------------
# Validation (lightweight, MVP-level)
# ---------------------------------------------------------------------------

def validate(data: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation error strings (empty = valid)."""
    errors: list[str] = []
    contract = data.get("contract", {})

    if not contract.get("contract_name"):
        errors.append("Contract name is missing.")
    if not contract.get("entity_company_code"):
        errors.append("Organization / entity is missing.")

    items = data.get("items", [])
    if not items:
        errors.append("At least one item is required.")

    for idx, item in enumerate(items, start=1):
        label = item.get("asset_name") or f"Item {idx}"
        if not item.get("asset_name"):
            errors.append(f"Item {idx}: asset name is missing.")
        if not item.get("lease_start"):
            errors.append(f"{label}: lease start date is missing.")
        if not item.get("lease_end"):
            errors.append(f"{label}: lease end date is missing.")
        if not item.get("currency"):
            errors.append(f"{label}: currency is missing.")

        payments = item.get("payments", [])
        if not payments:
            errors.append(f"{label}: at least one payment is required.")
        for pidx, payment in enumerate(payments, start=1):
            if payment.get("payment_type") not in PAYMENT_TYPE_CODES:
                errors.append(f"{label}, payment {pidx}: invalid or missing payment type.")
            if payment.get("payment_cycle") not in PAYMENT_CYCLE_CODES:
                errors.append(f"{label}, payment {pidx}: invalid or missing payment cycle.")
            if payment.get("pre_post_payment") not in PRE_POST_PAYMENT_CODES:
                errors.append(f"{label}, payment {pidx}: prepayment/postpayment not set.")
            if payment.get("payment_value") in (None, ""):
                errors.append(f"{label}, payment {pidx}: payment value is missing.")
            if payment.get("review_required"):
                errors.append(f"{label}, payment {pidx}: flagged for manual review.")

        for oidx, option in enumerate(item.get("options", []), start=1):
            if option.get("option_type") not in OPTION_TYPE_CODES:
                errors.append(f"{label}, option {oidx}: invalid or missing option type.")
            if option.get("review_required"):
                errors.append(f"{label}, option {oidx}: flagged for manual review.")

    return errors


# ---------------------------------------------------------------------------
# Example data - lets you try the review/edit/confirm UI with zero API calls
# ---------------------------------------------------------------------------

def example_contract() -> dict[str, Any]:
    """An example built from the bundled `sample_contract.pdf` (a German
    Simson moped leasing contract), including real `evidence` entries whose
    quotes appear verbatim in that PDF.

    This lets you demo the full side-by-side review + source-highlighting UI
    without an API key or any provider credit.
    """
    data: dict[str, Any] = {
        "contract": {
            "contract_number": None,
            "contract_name": "Leasing-Vertrag",
            "entity_company_code": "Simson Leasing Chemnitz",
        },
        "items": [
            {
                "item_number": "76-S51(Grün)",
                "asset_name": "Simson S51Enduro 12 V Vape 4 Gang",
                "asset_class": "Kleinkraftrad",
                "currency": "EUR",
                "lease_start": "2017-04-01",
                "lease_end": "2020-03-31",
                "interest_rate": None,
                "payments": [
                    {
                        "payment_id": None,
                        "payment_type": "LEASE_FEE",
                        "payment_cycle": "MONTHLY",
                        "payment_start": None,   # -> filled from lease dates
                        "payment_end": None,
                        "pre_post_payment": "POSTPAYMENT",
                        "payment_value": 78.0,
                        "confidence": 0.95,
                        "review_required": False,
                    },
                    {
                        "payment_id": None,
                        "payment_type": "INITIAL_DIRECT_COSTS",
                        "payment_cycle": "ONE_TIME",
                        "payment_start": "2017-04-01",
                        "payment_end": "2017-04-01",
                        "pre_post_payment": "PREPAYMENT",
                        "payment_value": 350.0,
                        "confidence": 0.8,
                        "review_required": False,
                    },
                ],
                "options": [],
            }
        ],
        "evidence": [
            {"field": "contract.contract_name", "page": 1, "quote": "Leasing-Vertrag"},
            {"field": "contract.entity_company_code", "page": 1, "quote": "Simson Leasing Chemnitz"},
            {"field": "items[0].asset_name", "page": 1, "quote": "Simson S51Enduro"},
            {"field": "items[0].item_number", "page": 1, "quote": "76-S51"},
            {"field": "items[0].asset_class", "page": 1, "quote": "Kleinkraftrad"},
            {"field": "items[0].lease_start", "page": 1, "quote": "01.04.2017"},
            {"field": "items[0].lease_end", "page": 1, "quote": "36 Monate"},
            {"field": "items[0].payments[0].payment_value", "page": 2, "quote": "78€"},
            {"field": "items[0].payments[1].payment_value", "page": 2, "quote": "350€"},
        ],
    }
    return apply_payment_date_fallback(data)
