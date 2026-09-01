# Lease Contract Extraction - MVP

A minimal Streamlit app that implements **Phase 1 (MVP: Single Contract Scan)**
of the OCR/AI contract-scanning roadmap: upload one lease contract PDF,
extract structured data with the OpenAI API, review and correct it
side-by-side with the source text, and confirm/export it. No batch upload,
confidence scoring, or automatic contract creation yet - that's later phases.

## Data model

```
Contract (1)
 └── Item (1..n)
      ├── item_id, item_name, asset_class
      ├── lease_start_date, lease_end_date
      ├── interest_rate, currency
      └── Payment (1..n)
           ├── payment_id
           ├── payment_cycle: Monthly | Quarterly | Annually | Bi-Annually | One-time
           ├── payment_start   (defaults to the item's lease_start_date if not stated)
           ├── payment_end     (defaults to the item's lease_end_date if not stated)
           └── payment_value
```

One contract can have several items (leased assets); one item can have
several payments (e.g. a recurring lease rate **and** a one-time deposit).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI API key either as an environment variable:

```bash
cp .env.example .env
# edit .env and fill in OPENAI_API_KEY
export OPENAI_API_KEY=sk-...     # or use `python-dotenv` / your shell profile
```

...or just paste it into the app's sidebar at runtime (it's only sent to
OpenAI's API, never stored).

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

A sample contract, `sample_contract.pdf`, is included so you can try the
flow immediately.

## How it works

1. **Upload** a PDF and click **Scan contract**.
2. `extraction.py` pulls the raw text out of the PDF with `pypdf`, then
   calls the OpenAI Chat Completions API with a strict JSON Schema
   (Structured Outputs) describing the contract/item/payment shape, so the
   model's response is always valid, parseable JSON.
3. If a payment's start/end date isn't explicitly stated in the contract,
   the app fills it in from the item's lease start/end date (per the
   requirement: "if not mentioned, take the lease start/end date").
4. The **source text** and the **editable extracted data** are shown
   side-by-side. Every field is editable; payments are an add/remove-rows
   table (`st.data_editor`) per item, so you can correct anything the model
   got wrong before confirming.
5. **Confirm extraction** runs lightweight validation (required fields
   present, valid payment cycle, etc.) and, once it passes, lets you
   download the result as **JSON** (the internal format) and as a
   **flattened CSV** (one row per payment - a stand-in for the "CSV export
   for validation" step of a later phase).

## Notes / limitations (by design, for an MVP)

- One contract at a time - no batch upload yet.
- No confidence scores or source-page highlighting yet (Phase 2).
- Nothing is written back into any accounting/contract system - the
  "Confirm" step only produces a JSON/CSV export (Phase 5 would connect
  this to real contract creation via an existing application API).
- Scanned/image-only PDFs aren't OCR'd - `pypdf` only reads embedded text.
  If you need that, a proper OCR step (e.g. `pytesseract`, or sending page
  images to a vision-capable model) would sit in front of this pipeline.
- Uses `gpt-4o-mini` by default (`gpt-4o` also selectable in the sidebar) -
  any model that supports Structured Outputs (`response_format:
  json_schema`) will work.
