"""
Lease Contract OCR/AI Extraction - MVP (Phase 1)

Upload one lease contract PDF -> extract structured data via the OpenAI
API -> review & correct it side-by-side with the source -> confirm and
export as JSON.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st

from extraction import (
    DEFAULT_MODEL,
    PAYMENT_CYCLES,
    extract_contract,
    validate,
)

st.set_page_config(
    page_title="Lease Contract Extraction - MVP",
    page_icon="\U0001F4C4",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "data" not in st.session_state:
    st.session_state.data = None          # extracted / edited structured data
if "source_text" not in st.session_state:
    st.session_state.source_text = None   # raw text pulled from the PDF
if "confirmed" not in st.session_state:
    st.session_state.confirmed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = None


def reset_all() -> None:
    st.session_state.data = None
    st.session_state.source_text = None
    st.session_state.confirmed = False
    st.session_state.file_name = None


def blank_payment() -> dict:
    return {
        "payment_id": f"P-{uuid.uuid4().hex[:6]}",
        "payment_cycle": "Monthly",
        "payment_start": None,
        "payment_end": None,
        "payment_value": None,
    }


def blank_item() -> dict:
    return {
        "item_id": f"I-{uuid.uuid4().hex[:6]}",
        "item_name": "",
        "asset_class": "",
        "lease_start_date": None,
        "lease_end_date": None,
        "interest_rate": None,
        "currency": "EUR",
        "payments": [blank_payment()],
    }


# ---------------------------------------------------------------------------
# Sidebar - API configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    api_key = st.text_input(
        "OpenAI API key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Reads OPENAI_API_KEY from the environment if set. "
        "Never uploaded anywhere except directly to OpenAI's API.",
    )
    model = st.selectbox(
        "Model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
        help="Must be a model that supports Structured Outputs "
        "(json_schema response format).",
    )

    st.divider()
    st.caption(
        "This is the **Phase 1 (MVP)** slice of the OCR contract-scanning "
        "roadmap: one contract at a time, manual review, no automatic "
        "contract creation, no batch upload."
    )
    if st.session_state.data is not None:
        st.divider()
        if st.button("\U0001F504 Start over / upload another contract"):
            reset_all()
            st.rerun()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("Lease Contract Extraction")
st.caption("Upload \u2192 Scan \u2192 Review \u2192 Confirm - one contract at a time.")

# ---------------------------------------------------------------------------
# Step 1: Upload + Scan
# ---------------------------------------------------------------------------

if st.session_state.data is None:
    uploaded = st.file_uploader("Upload one lease contract (PDF)", type=["pdf"])

    col_a, col_b = st.columns([1, 4])
    with col_a:
        scan_clicked = st.button("\U0001F50D Scan contract", type="primary", disabled=uploaded is None)

    if scan_clicked:
        if not api_key:
            st.error("Please enter your OpenAI API key in the sidebar first.")
        elif uploaded is None:
            st.error("Please upload a PDF first.")
        else:
            with st.spinner("Reading the PDF and calling the OpenAI API..."):
                try:
                    data, text = extract_contract(uploaded, api_key=api_key, model=model)
                    st.session_state.data = data
                    st.session_state.source_text = text
                    st.session_state.file_name = uploaded.name
                    st.session_state.confirmed = False
                    st.rerun()
                except Exception as exc:  # noqa: BLE001 - surface any failure to the user
                    st.error(f"Extraction failed: {exc}")

    st.info(
        "No contract loaded yet. Upload a PDF and click **Scan contract** to "
        "extract contract, item and payment data with OpenAI."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Step 2: Review & edit (side-by-side with source)
# ---------------------------------------------------------------------------

data = st.session_state.data

left, right = st.columns([2, 3])

with left:
    st.subheader("Source document")
    st.caption(st.session_state.file_name or "")
    st.text_area(
        "Extracted text (read-only)",
        value=st.session_state.source_text or "",
        height=650,
        disabled=True,
    )

with right:
    st.subheader("Extracted data - review & correct")

    # --- Contract ---
    st.markdown("##### Contract")
    c1, c2, c3 = st.columns(3)
    data["contract"]["contract_id"] = c1.text_input(
        "Contract ID", value=data["contract"].get("contract_id") or ""
    )
    data["contract"]["contract_name"] = c2.text_input(
        "Contract name", value=data["contract"].get("contract_name") or ""
    )
    data["contract"]["organization"] = c3.text_input(
        "Organization", value=data["contract"].get("organization") or ""
    )

    st.markdown("##### Items")

    items = data.get("items", [])
    items_to_remove = []

    for i, item in enumerate(items):
        header = item.get("item_name") or f"Item {i + 1}"
        with st.expander(f"\U0001F4E6 {header}", expanded=True):
            ic1, ic2, ic3 = st.columns(3)
            item["item_id"] = ic1.text_input(
                "Item ID", value=item.get("item_id") or "", key=f"item_id_{i}"
            )
            item["item_name"] = ic2.text_input(
                "Item name", value=item.get("item_name") or "", key=f"item_name_{i}"
            )
            item["asset_class"] = ic3.text_input(
                "Asset class", value=item.get("asset_class") or "", key=f"asset_class_{i}"
            )

            ic4, ic5, ic6, ic7 = st.columns(4)
            item["lease_start_date"] = ic4.text_input(
                "Lease start (YYYY-MM-DD)",
                value=item.get("lease_start_date") or "",
                key=f"lease_start_{i}",
            )
            item["lease_end_date"] = ic5.text_input(
                "Lease end (YYYY-MM-DD)",
                value=item.get("lease_end_date") or "",
                key=f"lease_end_{i}",
            )
            item["interest_rate"] = ic6.number_input(
                "Interest rate (%)",
                value=float(item.get("interest_rate") or 0.0),
                step=0.1,
                format="%.2f",
                key=f"interest_rate_{i}",
            )
            item["currency"] = ic7.text_input(
                "Currency", value=item.get("currency") or "", key=f"currency_{i}"
            )

            st.markdown("**Payments** (recurring rate, one-time deposit, etc.)")

            payments = item.get("payments", [])
            payments_df = pd.DataFrame(
                payments,
                columns=[
                    "payment_id",
                    "payment_cycle",
                    "payment_start",
                    "payment_end",
                    "payment_value",
                ],
            )

            edited_df = st.data_editor(
                payments_df,
                num_rows="dynamic",
                use_container_width=True,
                key=f"payments_editor_{i}",
                column_config={
                    "payment_id": st.column_config.TextColumn("Payment ID"),
                    "payment_cycle": st.column_config.SelectboxColumn(
                        "Cycle", options=PAYMENT_CYCLES, required=True
                    ),
                    "payment_start": st.column_config.TextColumn(
                        "Start (YYYY-MM-DD)",
                        help="Defaults to the item's lease start date if left blank.",
                    ),
                    "payment_end": st.column_config.TextColumn(
                        "End (YYYY-MM-DD)",
                        help="Defaults to the item's lease end date if left blank.",
                    ),
                    "payment_value": st.column_config.NumberColumn(
                        "Value", format="%.2f"
                    ),
                },
            )

            new_payments = edited_df.to_dict("records")
            for p in new_payments:
                if not p.get("payment_id"):
                    p["payment_id"] = f"P-{uuid.uuid4().hex[:6]}"
                if not p.get("payment_start"):
                    p["payment_start"] = item.get("lease_start_date")
                if not p.get("payment_end"):
                    p["payment_end"] = item.get("lease_end_date")
            item["payments"] = new_payments

            if st.button(f"\U0001F5D1\uFE0F Remove this item", key=f"remove_item_{i}"):
                items_to_remove.append(i)

    for idx in sorted(items_to_remove, reverse=True):
        items.pop(idx)
    if items_to_remove:
        st.rerun()

    if st.button("\u2795 Add item"):
        items.append(blank_item())
        st.rerun()

    data["items"] = items
    st.session_state.data = data

    st.divider()

    # --- Confirm ---
    confirm_col, status_col = st.columns([1, 3])
    with confirm_col:
        confirm_clicked = st.button("\u2705 Confirm extraction", type="primary")

    if confirm_clicked:
        errors = validate(data)
        if errors:
            st.session_state.confirmed = False
            with status_col:
                st.error("Please resolve the following before confirming:")
                for e in errors:
                    st.markdown(f"- {e}")
        else:
            st.session_state.confirmed = True

    if st.session_state.confirmed:
        st.success(
            "Extraction confirmed. In later phases this is where the data "
            "would flow into contract creation via the existing application "
            "API. For this MVP, download it below."
        )

        export_payload = {
            "exported_at": datetime.utcnow().isoformat() + "Z",
            "source_file": st.session_state.file_name,
            **data,
        }
        json_bytes = json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8")

        st.download_button(
            "\u2B07\uFE0F Download JSON",
            data=json_bytes,
            file_name=f"{(data['contract'].get('contract_id') or 'contract')}_extracted.json",
            mime="application/json",
        )

        # Flat payments view - illustrates JSON (internal) -> CSV (external) principle
        rows = []
        for item in data["items"]:
            for payment in item["payments"]:
                rows.append(
                    {
                        "contract_id": data["contract"].get("contract_id"),
                        "contract_name": data["contract"].get("contract_name"),
                        "organization": data["contract"].get("organization"),
                        "item_id": item.get("item_id"),
                        "item_name": item.get("item_name"),
                        "asset_class": item.get("asset_class"),
                        "lease_start_date": item.get("lease_start_date"),
                        "lease_end_date": item.get("lease_end_date"),
                        "interest_rate": item.get("interest_rate"),
                        "currency": item.get("currency"),
                        "payment_id": payment.get("payment_id"),
                        "payment_cycle": payment.get("payment_cycle"),
                        "payment_start": payment.get("payment_start"),
                        "payment_end": payment.get("payment_end"),
                        "payment_value": payment.get("payment_value"),
                    }
                )
        csv_bytes = pd.DataFrame(rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "\u2B07\uFE0F Download CSV (flattened payments)",
            data=csv_bytes,
            file_name=f"{(data['contract'].get('contract_id') or 'contract')}_payments.csv",
            mime="text/csv",
        )
