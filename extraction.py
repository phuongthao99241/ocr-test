"""
Extraction logic for the Lease Contract OCR MVP.

Responsible for:
  - Pulling raw text out of an uploaded PDF
  - Calling the OpenAI API with a strict JSON schema to get structured
    contract / item / payment data back
  - Applying the "payment start/end falls back to lease start/end" rule
"""

from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Allowed values / constants
# ---------------------------------------------------------------------------

PAYMENT_CYCLES = ["Monthly", "Quarterly", "Annually", "Bi-Annually", "One-time"]

DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# JSON schema used for OpenAI Structured Outputs
# (Contract -> Items[1..n] -> Payments[1..n])
# ---------------------------------------------------------------------------

EXTRACTION_JSON_SCHEMA: dict[str, Any] = {
    "name": "lease_contract_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "contract": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "contract_id": {"type": ["string", "null"]},
                    "contract_name": {"type": ["string", "null"]},
                    "organization": {"type": ["string", "null"]},
                },
                "required": ["contract_id", "contract_name", "organization"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "item_id": {"type": ["string", "null"]},
                        "item_name": {"type": ["string", "null"]},
                        "asset_class": {"type": ["string", "null"]},
                        "lease_start_date": {
                            "type": ["string", "null"],
                            "description": "ISO format YYYY-MM-DD",
                        },
                        "lease_end_date": {
                            "type": ["string", "null"],
                            "description": "ISO format YYYY-MM-DD",
                        },
                        "interest_rate": {
                            "type": ["number", "null"],
                            "description": "Percentage, e.g. 4.5 for 4.5%. Null if not mentioned.",
                        },
                        "currency": {
                            "type": ["string", "null"],
                            "description": "ISO 4217 currency code, e.g. EUR, USD.",
                        },
                        "payments": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "payment_id": {"type": ["string", "null"]},
                                    "payment_cycle": {
                                        "type": "string",
                                        "enum": PAYMENT_CYCLES,
                                    },
                                    "payment_start": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "ISO format YYYY-MM-DD. Null if not "
                                            "explicitly stated in the document."
                                        ),
                                    },
                                    "payment_end": {
                                        "type": ["string", "null"],
                                        "description": (
                                            "ISO format YYYY-MM-DD. Null if not "
                                            "explicitly stated in the document."
                                        ),
                                    },
                                    "payment_value": {"type": ["number", "null"]},
                                },
                                "required": [
                                    "payment_id",
                                    "payment_cycle",
                                    "payment_start",
                                    "payment_end",
                                    "payment_value",
                                ],
                            },
                        },
                    },
                    "required": [
                        "item_id",
                        "item_name",
                        "asset_class",
                        "lease_start_date",
                        "lease_end_date",
                        "interest_rate",
                        "currency",
                        "payments",
                    ],
                },
            },
        },
        "required": ["contract", "items"],
    },
}

SYSTEM_PROMPT = """\
You are a precise data-extraction assistant for lease / leasing contracts.
Read the contract text supplied by the user and return ONLY the structured
data requested by the JSON schema.

Rules:
- Extract only information that is actually present in the document. Use
  null for anything not stated - never invent or guess a value.
- Normalize all dates to ISO format YYYY-MM-DD.
- interest_rate is a plain percentage number (e.g. 4.5 for "4.5%"). Use null
  if no interest rate / rate is mentioned anywhere in the contract.
- currency must be an ISO 4217 code (e.g. EUR, USD, GBP). Infer it from
  currency symbols (e.g. "€" -> EUR, "$" -> USD) if the code itself is not
  written out.
- A contract can contain one or more leased items (assets). Each item can
  have one or more payments - for example a recurring lease rate AND a
  one-time deposit/security payment both belong to the same item.
- payment_cycle must be exactly one of: Monthly, Quarterly, Annually,
  Bi-Annually, One-time. A one-off deposit or down payment is "One-time".
- If a payment's own start or end date is not explicitly stated, leave
  payment_start / payment_end as null (the application will default them to
  the item's lease_start_date / lease_end_date - do not copy the lease dates
  in yourself).
- contract_id / item_id / payment_id: use an ID if the document states one
  explicitly (e.g. a contract or vehicle number); otherwise use null.
- Keep all extracted text values (names, ids) in their original language and
  spelling - do not translate them.
"""


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_obj) -> str:
    """Extract raw text from an uploaded PDF file-like object."""
    from pypdf import PdfReader

    reader = PdfReader(file_obj)
    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n".join(pages_text).strip()


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

def call_openai_extraction(
    document_text: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Send the contract text to OpenAI and return the parsed structured JSON."""
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Extract the structured lease data from the following "
                    "contract document:\n\n" + document_text
                ),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": EXTRACTION_JSON_SCHEMA,
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
        lease_start = item.get("lease_start_date")
        lease_end = item.get("lease_end_date")
        for payment in item.get("payments", []):
            if not payment.get("payment_start"):
                payment["payment_start"] = lease_start
            if not payment.get("payment_end"):
                payment["payment_end"] = lease_end
    return data


def extract_contract(
    file_obj,
    api_key: str,
    model: str = DEFAULT_MODEL,
) -> tuple[dict[str, Any], str]:
    """Full pipeline: PDF -> text -> OpenAI extraction -> fallback rule applied.

    Returns (structured_data, source_text).
    """
    text = extract_text_from_pdf(file_obj)
    if not text:
        raise ValueError(
            "No extractable text found in this PDF. If it's a scanned "
            "image-only PDF, OCR pre-processing would be needed (out of "
            "scope for this MVP)."
        )
    data = call_openai_extraction(text, api_key=api_key, model=model)
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
    if not contract.get("organization"):
        errors.append("Organization is missing.")

    items = data.get("items", [])
    if not items:
        errors.append("At least one item is required.")

    for idx, item in enumerate(items, start=1):
        label = item.get("item_name") or f"Item {idx}"
        if not item.get("item_name"):
            errors.append(f"Item {idx}: item name is missing.")
        if not item.get("lease_start_date"):
            errors.append(f"{label}: lease start date is missing.")
        if not item.get("lease_end_date"):
            errors.append(f"{label}: lease end date is missing.")
        if not item.get("currency"):
            errors.append(f"{label}: currency is missing.")

        payments = item.get("payments", [])
        if not payments:
            errors.append(f"{label}: at least one payment is required.")
        for pidx, payment in enumerate(payments, start=1):
            if payment.get("payment_cycle") not in PAYMENT_CYCLES:
                errors.append(f"{label}, payment {pidx}: invalid payment cycle.")
            if payment.get("payment_value") in (None, ""):
                errors.append(f"{label}, payment {pidx}: payment value is missing.")

    return errors
