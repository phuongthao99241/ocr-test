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
    contract / item / payment data back
  - Applying the "payment start/end falls back to lease start/end" rule
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Allowed values / constants
# ---------------------------------------------------------------------------

PAYMENT_CYCLES = ["Monthly", "Quarterly", "Annually", "Bi-Annually", "One-time"]

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
# Contract -> Items[1..n] -> Payments[1..n]
# ---------------------------------------------------------------------------

EXTRACTION_SCHEMA: dict[str, Any] = {
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
}

SYSTEM_PROMPT = """\
You are a precise data-extraction assistant for lease / leasing contracts.
Read the contract document supplied by the user and return ONLY the
structured data requested by the JSON schema.

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


# ---------------------------------------------------------------------------
# Example data - lets you try the review/edit/confirm UI with zero API calls
# ---------------------------------------------------------------------------

def example_contract() -> dict[str, Any]:
    """A hand-built example matching the schema exactly.

    Demonstrates: 1 contract -> 2 items; item 1 has two payments (a recurring
    monthly rate + a one-time deposit), item 2 has a single quarterly payment
    whose start/end were left blank on purpose, so you can see the
    lease-date fallback rule fill them in.
    """
    data: dict[str, Any] = {
        "contract": {
            "contract_id": "LC-2024-0031",
            "contract_name": "Fleet & Equipment Master Lease - Q1 2024",
            "organization": "Nordwind Fuhrpark GmbH",
        },
        "items": [
            {
                "item_id": "VEH-1042",
                "item_name": "Mercedes-Benz Sprinter 316 CDI",
                "asset_class": "Commercial Vehicle",
                "lease_start_date": "2024-02-01",
                "lease_end_date": "2027-01-31",
                "interest_rate": 3.9,
                "currency": "EUR",
                "payments": [
                    {
                        "payment_id": "P-1042-01",
                        "payment_cycle": "Monthly",
                        "payment_start": "2024-02-01",
                        "payment_end": "2027-01-31",
                        "payment_value": 640.00,
                    },
                    {
                        "payment_id": "P-1042-02",
                        "payment_cycle": "One-time",
                        "payment_start": "2024-02-01",
                        "payment_end": "2024-02-01",
                        "payment_value": 1500.00,
                    },
                ],
            },
            {
                "item_id": "EQ-0087",
                "item_name": "Konica Minolta bizhub C4051i (multifunction printer)",
                "asset_class": "Office Equipment",
                "lease_start_date": "2024-03-15",
                "lease_end_date": "2026-03-14",
                "interest_rate": None,
                "currency": "EUR",
                "payments": [
                    {
                        "payment_id": "P-0087-01",
                        "payment_cycle": "Quarterly",
                        # left blank on purpose - apply_payment_date_fallback()
                        # below fills these in from the item's lease dates.
                        "payment_start": None,
                        "payment_end": None,
                        "payment_value": 285.00,
                    }
                ],
            },
        ],
    }
    return apply_payment_date_fallback(data)
