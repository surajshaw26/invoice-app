import streamlit as st
import os
import io
import re
import warnings
import json
import base64
import logging
import time
import pandas as pd
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import OpenAI

# Optional rendering stack for the document overlay (#1).
# If either library is missing the app still runs; the inspector just falls
# back to showing field coordinates in the table instead of the page image.
DPI = 144  # render resolution for the document overlay
try:
    import fitz  # PyMuPDF -- renders PDF pages to images
    from PIL import Image, ImageDraw
    _RENDER_OK = True
except Exception:
    _RENDER_OK = False

# --- Persistent audit logging (replaces scattered print() calls) ---
# Writes a timestamped record to a log FILE and the console. The handler guard
# matters because Streamlit re-runs this whole script on every interaction --
# without it we would attach duplicate handlers and get duplicate log lines.
AUDIT_LOG_FILE = "extraction_audit.log"
logger = logging.getLogger("extraction_portal")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _log_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    _file_handler = logging.FileHandler(AUDIT_LOG_FILE, encoding="utf-8")
    _file_handler.setFormatter(_log_fmt)
    logger.addHandler(_file_handler)
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(_log_fmt)
    logger.addHandler(_console_handler)
    logger.propagate = False

# Load configuration keys
load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")

# App Layout Configuration
st.set_page_config(layout="wide", page_title="Intelligent Document Extraction & Validation Portal")

# Custom CSS for Premium Dashboard Look
st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🗂 Intelligent Document Extraction & Validation Portal")
st.markdown("Automated invoice parsing via **Azure Document Intelligence** with **LLM fallback verification**.")
st.markdown("---")

# =====================================================================
# COLUMN SCHEMA CONFIG  --  edit these lists to change your columns
# =====================================================================
# Master "Invoices" sheet: fixed, ordered columns.
# Left  = Azure field name  (do NOT change -- must match Azure exactly)
# Right = the column label you want to see in Excel.
TARGET_SCHEMA = [
    # --- Parties ---
    ("VendorName",        "Vendor"),
    ("VendorEmail",       "Vendor Email"),
    ("CustomerName",      "Customer"),
    ("CustomerId",        "Customer ID"),
    # --- Invoice identity & dates ---
    ("InvoiceId",         "Invoice Number"),
    ("PurchaseOrder",     "PO Number"),
    ("InvoiceDate",       "Invoice Date"),
    ("DueDate",           "Due Date"),
    ("PaymentTerm",       "Payment Terms"),
    # --- Money ---
    ("SubTotal",          "Subtotal"),
    ("TotalTax",          "Tax"),
    ("TaxDetails",        "Tax Details"),
    ("InvoiceTotal",      "Invoice Total"),
    ("AmountDue",         "Amount Due"),
    ("PaymentDetails",    "Payment Details"),
    # --- Addresses (mandatory) ---
    ("VendorAddress",     "Vendor Address"),
    ("BillingAddress",    "Billing Address"),
    ("CustomerAddress",   "Customer Address"),
    ("ShippingAddress",   "Shipping Address"),
    ("RemittanceAddress", "Remittance Address"),
]

# Second "Line Items" sheet: columns pulled from each line of the invoice.
# Left = Azure line-item sub-field, Right = your column label.
LINE_ITEM_FIELDS = [
    ("Description", "Description"),
    ("Quantity",    "Quantity"),
    ("Unit",        "Unit"),
    ("UnitPrice",   "Unit Price"),
    ("Amount",      "Amount"),
    ("ProductCode", "Product Code"),
    ("Tax",         "Tax"),
]

# Fixed metadata/audit columns pinned at the front of the master sheet.
AUDIT_COLUMNS = ["File Name", "Processing Status", "Time Elapsed", "Processing Cost"]
# Derived classification column (PO vs Non-PO), shown right after the audit block.
INVOICE_TYPE_COLUMN = "Invoice Type"

# Hard limit on total columns in the master sheet (schema + parked one-offs).
MAX_TOTAL_COLUMNS = 100

# =====================================================================
# PRICING & FALLBACK SETTINGS  (#6 / #7)  --  one place for the knobs
# =====================================================================
# Costs (update here if Azure / OpenAI change their rates):
AZURE_COST_PER_PAGE          = 0.01              # prebuilt-invoice ≈ $10 / 1,000 pages
OPENAI_INPUT_COST_PER_TOKEN  = 0.15 / 1_000_000  # gpt-4o-mini input  ($0.15 / 1M tokens)
OPENAI_OUTPUT_COST_PER_TOKEN = 0.60 / 1_000_000  # gpt-4o-mini output ($0.60 / 1M tokens)

# LLM fallback behaviour:
CONFIDENCE_THRESHOLD     = 0.80           # Azure scores below this get LLM help (and show as "Low")
LLM_FALLBACK_MODEL       = "gpt-4o-mini"  # multimodal: can read the page IMAGE, not just OCR text
MAX_FALLBACK_PAGES       = 3              # most page-images to send in the single fallback call
FALLBACK_TEXT_CHAR_LIMIT = 8000           # cap on raw OCR text sent as backup context

# Network resilience (#7): retry transient Azure / OpenAI errors with backoff.
RETRY_ATTEMPTS   = 3                       # total tries
RETRY_BASE_DELAY = 1.0                     # seconds; doubles each retry (1s, 2s, 4s)

# Batch limits:
MAX_FILES = 50                             # hard cap on files processed per batch run

# =====================================================================
# FIX #8 CONFIG  --  how each column is STORED in the Excel download
# =====================================================================
# Excel (via openpyxl) decides a cell is a number/date/text from the value
# AND its number format. We set both explicitly per column so the download is
# clean and consistent instead of "everything is text".
#   'text'   -> forced text  (preserves leading zeros, e.g. "003104"; stops
#               long codes turning into 1.23E+15 scientific notation)
#   'money'  -> a real number with a thousands/decimal format (summable in Excel)
#   'date'   -> a real date
#   'number' -> a plain real number (e.g. quantities)
# Any column NOT listed below is left exactly as extracted (plain text).
#
# Change just these two strings to restyle every money / date cell at once:
MONEY_NUMBER_FORMAT = "#,##0.00"    # e.g. "$#,##0.00" for a $ sign, "€#,##0.00", "£#,##0.00"
DATE_NUMBER_FORMAT  = "yyyy-mm-dd"  # e.g. "mm/dd/yyyy" or "dd/mm/yyyy"

# Master "Invoices" sheet (keys = the column LABELS from TARGET_SCHEMA).
INVOICE_COLUMN_TYPES = {
    "Customer ID":   "text",
    "Invoice Number":"text",
    "PO Number":     "text",
    "Subtotal":      "money",
    "Tax":           "money",
    "Invoice Total": "money",
    "Amount Due":    "money",
    "Invoice Date":  "date",
    "Due Date":      "date",
}

# "Line Items" sheet (keys = the column labels on that tab).
# NOTE: per-line "Tax" is intentionally left as text -- it can be an amount
# ("$2.40") OR a rate ("8%"), and forcing it numeric would mangle the rate.
LINE_ITEM_COLUMN_TYPES = {
    "Source File":    "text",
    "Invoice Number": "text",
    "Product Code":   "text",
    "Quantity":       "number",
    "Unit Price":     "money",
    "Amount":         "money",
}


# --- FIX #8 helpers: parse extracted text into real numbers / dates ---
# Every parser is best-effort and SAFE: if a value can't be parsed it is left
# exactly as Azure read it (as text), so a surprise format never deletes data.
def _parse_money(s):
    """'$1,234.50' / '242.70' / '(50.00)' -> float. Unparseable -> None.
    Assumes ',' is a thousands separator and '.' is the decimal point."""
    t = (s or "").strip()
    if t == "":
        return None
    neg = False
    if t.startswith("(") and t.endswith(")"):      # accounting-style negative
        neg, t = True, t[1:-1]
    cleaned = re.sub(r"[^0-9.-]", "", t.replace(",", ""))  # drop $, €, letters, spaces
    if cleaned in ("", "-", ".", "-."):
        return None
    try:
        val = float(cleaned)
    except ValueError:
        return None
    return -val if neg else val


def _parse_date(s):
    """'12 May 2026' / '03/20/2026' / '2026-05-12' -> datetime.date.
    Day/month order defaults to US month-first for ambiguous numeric dates.
    Anything outside 1990-2100 (or unparseable) -> None (kept as text)."""
    t = (s or "").strip()
    if t == "":
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # silence pandas format-inference notices
        ts = pd.to_datetime(t, errors="coerce", dayfirst=False)
    if ts is None or pd.isna(ts):
        return None
    if not (1990 <= ts.year <= 2100):              # guard against junk like "3104" -> year 3104
        return None
    return ts.date()


def _parse_plain_number(s):
    """'10' -> 10, '2.5' -> 2.5, '1,000' -> 1000. Unparseable -> None."""
    t = (s or "").strip().replace(",", "")
    if t == "":
        return None
    try:
        f = float(t)
    except ValueError:
        return None
    return int(f) if f == int(f) else f


def _fmt_money_str(s):
    """Display-only: consistent '1,234.50' for the on-screen grid (keeps text if unparseable)."""
    s = "" if s is None else str(s)
    if s.strip() == "":
        return ""
    num = _parse_money(s)
    return f"{num:,.2f}" if num is not None else s


def _fmt_date_str(s):
    """Display-only: consistent 'YYYY-MM-DD' for the on-screen grid (keeps text if unparseable)."""
    s = "" if s is None else str(s)
    d = _parse_date(s)
    return d.strftime("%Y-%m-%d") if d is not None else s


def _grid_display(df, col_types):
    """Return a copy of df with money/date columns reformatted to consistent
    strings so the on-screen grid matches the cleaned Excel download."""
    out = df.copy()
    for col, kind in col_types.items():
        if col not in out.columns:
            continue
        if kind == "money":
            out[col] = out[col].map(_fmt_money_str)
        elif kind == "date":
            out[col] = out[col].map(_fmt_date_str)
    return out


def _xl_write_typed(ws, df, col_types, money_fmt=MONEY_NUMBER_FORMAT, date_fmt=DATE_NUMBER_FORMAT):
    """Overwrite the already-written worksheet cells with correctly TYPED values
    and number formats. Reads the intended value from `df` (always the original
    extracted string), so it is immune to whatever the default writer did."""
    for c_idx, col in enumerate(df.columns, start=1):
        kind = col_types.get(col)
        if not kind:
            continue                       # leave column exactly as written (plain text)
        for r_off, raw in enumerate(df[col].tolist(), start=2):  # row 1 = header
            cell = ws.cell(row=r_off, column=c_idx)
            s = "" if raw is None else str(raw)
            if s.strip() == "":
                cell.value = None          # truly blank cell
                continue
            if kind == "text":
                cell.value = s
                cell.number_format = "@"   # '@' = force Excel to treat as text
            elif kind == "money":
                num = _parse_money(s)
                if num is None:
                    cell.value = s         # keep original text, don't fake a number
                else:
                    cell.value = num
                    cell.number_format = money_fmt
            elif kind == "date":
                d = _parse_date(s)
                if d is None:
                    cell.value = s
                else:
                    cell.value = d
                    cell.number_format = date_fmt
            elif kind == "number":
                num = _parse_plain_number(s)
                cell.value = s if num is None else num
    return ws


@st.cache_resource
def get_clients():
    try:
        azure_client = DocumentIntelligenceClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))
        openai_client = OpenAI(api_key=OPENAI_KEY)
        return azure_client, openai_client
    except Exception as e:
        return None, None


# --- FIX #3 (+ array/object handling): clean value from an Azure DocumentField ---
def clean_field_value(field):
    """Return clean, human-readable text for an Azure DocumentField.
    This SDK has no generic `.value`; `.content` is the literal text Azure
    read off the page and is clean for most field types. Array/object fields
    (e.g. TaxDetails, PaymentDetails) are summarised into readable text instead
    of dumping the raw object."""
    if field is None:
        return ""
    # 1) Literal text Azure read off the page
    if getattr(field, "content", None):
        return field.content
    ftype = getattr(field, "type", None)
    # 2) Array fields: summarise each element
    if ftype == "array" and getattr(field, "value_array", None):
        parts = [clean_field_value(el) for el in field.value_array]
        return " | ".join(p for p in parts if p)
    # 3) Object fields: join "key: value" pairs
    if ftype == "object" and getattr(field, "value_object", None):
        pairs = []
        for k, sub in field.value_object.items():
            sv = clean_field_value(sub)
            if sv:
                pairs.append(f"{k}: {sv}")
        return "; ".join(pairs)
    # 4) Scalar typed values (fallback when there is no literal content)
    for attr in ("value_string", "value_currency", "value_date", "value_number",
                 "value_integer", "value_phone_number", "value_address",
                 "value_time", "value_boolean"):
        v = getattr(field, attr, None)
        if v is not None:
            return str(v)
    return ""


def extract_line_items(items_field):
    """Turn Azure's 'Items' array field into a list of flat row dicts, one per
    line item.

    FREIGHT SAFEGUARD: only rows that carry a Quantity or a Unit Price are kept.
    Freight invoices (FedEx etc.) have no products, so Azure fills 'Items' with
    charge lines (Fuel Surcharge, Discount, ...) that have neither -- those are
    dropped so they never pollute the product line-item tab. NOTE: a side effect
    is that flat lines without a quantity/unit price (e.g. a single service fee)
    are also excluded. Loosen the condition below if you want to keep those."""
    rows = []
    if items_field is None:
        return rows
    array = getattr(items_field, "value_array", None) or []
    for item in array:
        obj = getattr(item, "value_object", None) or {}
        row = {}
        for az, friendly in LINE_ITEM_FIELDS:
            row[friendly] = clean_field_value(obj.get(az))
        if str(row.get("Quantity", "")).strip() or str(row.get("Unit Price", "")).strip():
            rows.append(row)
    # Number the kept rows sequentially
    out = []
    for n, r in enumerate(rows, start=1):
        out.append({"Line #": n, **r})
    return out


# ---------------------------------------------------------------------
# #1 -- document overlay helpers (render a page, draw field bounding boxes)
# ---------------------------------------------------------------------
def _box_color(meta):
    """Colour a field's box by how it was resolved (matches the inspector badges)."""
    if meta.get("source") == "LLM Fallback":
        return "#2b6cb0"   # blue   -- LLM fallback (no confidence score)
    c = meta.get("confidence")
    if c is None:
        return "#6c757d"   # grey   -- Azure returned no score (unknown, not "low")
    if c >= CONFIDENCE_THRESHOLD:
        return "#28a745"   # green  -- Azure match
    return "#f0ad4e"       # orange -- low confidence


@st.cache_data(show_spinner=False)
def _render_page_image(file_bytes, file_name, page_number, dpi=DPI):
    """Render one page of the uploaded document to a PIL image (cached)."""
    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if ext in ("png", "jpg", "jpeg"):
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72.0, dpi / 72.0))
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


def _annotate_page(base_img, page_number, numbered_fields, pages_meta, dpi=DPI, focus_idx=None):
    """Draw colour-coded, numbered boxes for every field located on this page.
    If focus_idx is given, that field's box is emphasised (thicker + highlight colour)."""
    img = base_img.copy()
    draw = ImageDraw.Draw(img)
    unit = pages_meta.get(page_number, {}).get("unit", "inch")
    scale = dpi if unit == "inch" else 1.0   # inch->px uses dpi; image px is 1:1
    for idx, _fname, meta in numbered_fields:
        focused = (focus_idx is not None and idx == focus_idx)
        color = "#e6007a" if focused else _box_color(meta)   # magenta highlight when focused
        width = 6 if focused else 3
        for reg in meta.get("regions", []):
            if reg.get("page") != page_number:
                continue
            poly = reg.get("polygon", [])
            pts = [(poly[i] * scale, poly[i + 1] * scale) for i in range(0, len(poly) - 1, 2)]
            if len(pts) >= 2:
                draw.polygon(pts, outline=color, width=width)
                x0, y0 = pts[0]
                draw.text((x0 + 2, max(0, y0 - 12)), str(idx), fill=color)
    return img


def _unique_key(name, used):
    """Return a key not already in `used`, appending ' (2)', ' (3)' ... on
    collision, then record it. Lets two files share a name (or two invoices
    share a label) without one silently overwriting the other in the results."""
    key = name
    counter = 2
    while key in used:
        key = f"{name} ({counter})"
        counter += 1
    used.add(key)
    return key


def _call_with_retry(fn, attempts=RETRY_ATTEMPTS, base_delay=RETRY_BASE_DELAY, label="call"):
    """Run fn(); on failure, retry with exponential backoff (base, 2x, 4x ...).
    Re-raises the last error if every attempt fails. Used for both the Azure and
    the OpenAI calls so a transient network blip doesn't fail the work (#7)."""
    last_err = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i < attempts - 1:
                wait = base_delay * (2 ** i)
                logger.warning("%s attempt %d/%d failed (%s); retrying in %.0fs", label, i + 1, attempts, e, wait)
                time.sleep(wait)
    raise last_err


def _encode_page_png(file_bytes, file_name, page_number, max_side=2000):
    """Render one page and return it as a base64 PNG data-URL for the vision model,
    or None if it can't be rendered. Down-scales very large pages to cap token cost."""
    if not _RENDER_OK:
        return None
    try:
        img = _render_page_image(file_bytes, file_name, page_number, DPI)
        w, h = img.size
        longest = max(w, h)
        if longest > max_side:
            scale = max_side / float(longest)
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception as e:
        logger.warning("Could not render page %d for the LLM: %s", page_number, e)
        return None


def _llm_fallback_batch(client, field_names, document_fields, raw_text, file_bytes, file_name):
    """#6: ONE structured, (optionally) multimodal call that re-reads ALL the
    low-confidence fields at once -- instead of one text-only call per field.

    Sends the relevant page IMAGE(S) when they can be rendered (so the model
    actually looks at the document rather than re-reading Azure's possibly-garbled
    OCR), with the raw OCR text as backup context. Returns (values_by_field, cost).
    Raises on a hard failure so the caller can keep the Azure data intact (#7)."""
    # Which pages to show the model: the pages the low-confidence fields sit on.
    pages = sorted({reg["page"]
                    for f in field_names
                    for reg in document_fields.get(f, {}).get("regions", [])})
    if not pages:
        pages = [1]
    pages = pages[:MAX_FALLBACK_PAGES]

    # List each field with Azure's uncertain guess as a (possibly-wrong) hint.
    hint_lines = []
    for f in field_names:
        guess = document_fields.get(f, {}).get("value", "")
        hint_lines.append(f"- {f} (OCR's low-confidence guess: {guess!r})")
    field_block = "\n".join(hint_lines)

    system_msg = (
        "You are a precise invoice data extractor. You are given an invoice (as page "
        "image(s) and/or its raw OCR text) plus a list of fields an OCR system read with "
        "LOW confidence. Re-read the document and determine the correct value for each "
        "requested field. Respond with a SINGLE JSON object mapping each requested field "
        "name to its value as a string. If a field is genuinely not present, use null for "
        "that field. Use the exact field names given. Return ONLY the JSON object."
    )

    user_content = [{
        "type": "text",
        "text": ("Fields to extract (the parenthesised guess may be wrong):\n"
                 f"{field_block}\n\n"
                 "Raw OCR text of the document (may contain errors):\n"
                 f"{(raw_text or '')[:FALLBACK_TEXT_CHAR_LIMIT]}"),
    }]

    # Attach page images when we can render them (the real accuracy win).
    img_count = 0
    for pg in pages:
        data_url = _encode_page_png(file_bytes, file_name, pg)
        if data_url:
            user_content.append({"type": "image_url",
                                 "image_url": {"url": data_url, "detail": "high"}})
            img_count += 1

    logger.info("LLM fallback: 1 call for %d field(s) with %d page image(s)", len(field_names), img_count)
    response = _call_with_retry(
        lambda: client.chat.completions.create(
            model=LLM_FALLBACK_MODEL,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_content}],
            temperature=0.0,
            response_format={"type": "json_object"},
        ),
        label="OpenAI fallback",
    )

    # Cost from this single call (image tokens are already counted in prompt_tokens).
    p_tokens = response.usage.prompt_tokens
    c_tokens = response.usage.completion_tokens
    cost = p_tokens * OPENAI_INPUT_COST_PER_TOKEN + c_tokens * OPENAI_OUTPUT_COST_PER_TOKEN

    # Parse the JSON (tolerate stray code fences); a parse failure raises -> Azure kept.
    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(content)
    except Exception:
        cleaned = content.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("LLM did not return a JSON object")

    # Map the model's keys back to the EXACT requested field names (tolerant of casing).
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    norm_map = {norm(k): v for k, v in parsed.items()}
    values = {f: (None if norm_map.get(norm(f)) is None else str(norm_map.get(norm(f))))
              for f in field_names}
    return values, cost


azure_client, openai_client = get_clients()

s_state = st.session_state
if "batch_results" not in s_state:
    s_state.batch_results = {}
if "selected_file" not in s_state:
    s_state.selected_file = None

# --- SIDEBAR: INGESTION CONTROL PANEL ---
with st.sidebar:
    st.header("📤 Ingestion Layer")
    st.markdown("Upload documents to queue processing batches.")

    uploaded_files = st.file_uploader(
        f"Select Invoice Files (Max {MAX_FILES})",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.success(f"📦 {len(uploaded_files)} document(s) in queue ready.")
        extract_btn = st.button("🚀 Run Batch Extraction", type="primary", use_container_width=True)

        if extract_btn:
            if not azure_client or not openai_client:
                st.error("AI engines initialization failed. Check your keys inside `.env`.")
            else:
                # Fresh run: clear previous results so re-running repopulates cleanly
                # (and so leftovers / duplicate-named files from a prior run don't linger).
                s_state.batch_results = {}
                s_state.selected_file = None

                # Enforce the file cap: never process more than MAX_FILES in one batch.
                files_to_process = uploaded_files[:MAX_FILES]
                if len(uploaded_files) > MAX_FILES:
                    skipped = len(uploaded_files) - MAX_FILES
                    st.warning(f"Batch limit is {MAX_FILES} files. Processing the first {MAX_FILES}; "
                               f"{skipped} file(s) were skipped — run those in a separate batch.")
                    logger.warning("File cap hit: %d uploaded, processing %d, skipping %d",
                                   len(uploaded_files), MAX_FILES, skipped)

                progress_bar = st.progress(0)
                status_text = st.empty()
                used_keys = set()          # guarantees a unique key per result (duplicate filenames)
                total = len(files_to_process)

                for index, file in enumerate(files_to_process):
                    status_text.text(f"Processing: {file.name}")
                    file_bytes = file.read()
                    file_start_time = time.time()
                    logger.info("START processing: %s", file.name)

                    # Outer guard: any unexpected error marks just THIS file failed and
                    # lets the batch continue (one bad file never kills the whole run).
                    try:
                        # --- AZURE EXTRACTION (with retry/backoff #7) ---
                        # One Azure call per file; if it ultimately fails, the file failed.
                        try:
                            azure_result = _call_with_retry(
                                lambda: azure_client.begin_analyze_document("prebuilt-invoice", body=file_bytes).result(),
                                label="Azure",
                            )
                        except Exception as azure_err:
                            raise RuntimeError(f"Azure extraction failed after retries: {azure_err}") from azure_err

                        # 💸 Azure page cost (centralised rate)
                        page_count = len(azure_result.pages) if azure_result.pages else 1
                        azure_cost = page_count * AZURE_COST_PER_PAGE
                        raw_extracted_text = azure_result.content or ""
                        logger.info("Azure OK: %s | pages=%d | azure_cost=$%.4f", file.name, page_count, azure_cost)

                        # NEW (#1): page geometry, needed to map polygon coords to pixels.
                        # Shared by every invoice in this file; built before the fallback
                        # so it can pick which page images to send.
                        pages_meta = {}
                        for p in (azure_result.pages or []):
                            pages_meta[p.page_number] = {
                                "width": getattr(p, "width", None),
                                "height": getattr(p, "height", None),
                                "unit": getattr(p, "unit", "inch"),
                            }

                        # A single PDF can hold MULTIPLE invoices -> Azure returns several
                        # "documents". Process EACH as its own result instead of silently
                        # reading only the first one.
                        invoice_parts = []
                        for doc in (azure_result.documents or []):
                            doc_fields = doc.fields or {}

                            # ---- PHASE 1: take Azure's reading for every field ----
                            document_fields = {}
                            low_conf = []
                            for field_name, field_data in doc_fields.items():
                                val = clean_field_value(field_data)            # FIX #3: clean text
                                conf = getattr(field_data, "confidence", None)  # FIX #4: real score / None
                                regions = []
                                for br in (getattr(field_data, "bounding_regions", None) or []):
                                    poly = [float(x) for x in (getattr(br, "polygon", None) or [])]
                                    regions.append({"page": br.page_number, "polygon": poly})
                                document_fields[field_name] = {
                                    "value": val, "confidence": conf,
                                    "source": "Azure AI Engine", "regions": regions,
                                }
                                if conf is not None and conf < CONFIDENCE_THRESHOLD and field_name != "Items":
                                    low_conf.append(field_name)

                            line_items = extract_line_items(doc_fields.get("Items"))

                            # ---- PHASE 2 (#6): ONE multimodal call per invoice, isolated (#7) ----
                            # If the LLM step fails we KEEP every Azure value and skip the
                            # enhancement -- the file is still a success.
                            inv_openai_cost = 0.0
                            if low_conf and openai_client is not None:
                                try:
                                    llm_values, call_cost = _llm_fallback_batch(
                                        openai_client, low_conf, document_fields,
                                        raw_extracted_text, file_bytes, file.name,
                                    )
                                    inv_openai_cost += call_cost
                                    applied = 0
                                    for fname in low_conf:
                                        new_val = llm_values.get(fname)
                                        if new_val is not None and str(new_val).strip() != "":
                                            document_fields[fname].update({
                                                "value": str(new_val),
                                                "confidence": None,          # LLM has no real score (#4)
                                                "source": "LLM Fallback",
                                            })
                                            applied += 1
                                    logger.info("LLM fallback applied to %d/%d field(s) [%s]",
                                                applied, len(low_conf), file.name)
                                except Exception as llm_err:
                                    logger.warning("LLM fallback skipped for %s; Azure values kept. (%s)",
                                                   file.name, llm_err)

                            invoice_parts.append({
                                "fields": document_fields,
                                "line_items": line_items,
                                "openai_cost": inv_openai_cost,
                            })

                        # If Azure returned NO documents, still record one (empty) result so
                        # the file shows up in the batch rather than vanishing silently.
                        if not invoice_parts:
                            invoice_parts.append({"fields": {}, "line_items": [], "openai_cost": 0.0})

                        # One Azure call covered the whole file, so split its cost + elapsed
                        # time evenly across the invoices found; OpenAI cost is per-invoice.
                        elapsed_time = time.time() - file_start_time
                        n_docs = len(invoice_parts)
                        per_time = elapsed_time / n_docs
                        per_azure = azure_cost / n_docs

                        for di, part in enumerate(invoice_parts):
                            label = file.name if n_docs <= 1 else f"{file.name} — invoice {di + 1} of {n_docs}"
                            key = _unique_key(label, used_keys)   # never overwrite a duplicate name
                            s_state.batch_results[key] = {
                                "fields": part["fields"],
                                "line_items": part["line_items"],
                                "raw_text": raw_extracted_text,
                                "time_taken": per_time,
                                "total_cost": per_azure + part["openai_cost"],
                                "page_count": page_count,
                                "file_bytes": file_bytes,    # shared whole-file bytes for rendering
                                "pages_meta": pages_meta,     # shared coordinate mapping
                                "source_file": file.name,     # real filename (rendering + linking)
                            }
                    except Exception as ex:
                        logger.error("Error processing %s: %s", file.name, ex)
                        key = _unique_key(file.name, used_keys)
                        s_state.batch_results[key] = {
                            "error": str(ex), "fields": {}, "line_items": [],
                            "file_bytes": file_bytes, "pages_meta": {},
                            "source_file": file.name,
                        }

                    progress_bar.progress((index + 1) / total)

                status_text.empty()
                st.toast("Batch processing completed successfully!", icon="🎉")

    # 📜 Audit log -- persists on disk across runs; download it here.
    st.markdown("---")
    st.markdown("**📜 Audit log**")
    if os.path.exists(AUDIT_LOG_FILE):
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as _logf:
            _log_text = _logf.read()
        st.download_button("Download audit log", data=_log_text,
                           file_name=AUDIT_LOG_FILE, mime="text/plain",
                           use_container_width=True)
    else:
        st.caption("No log yet — run a batch to create one.")

# --- MAIN WORKSPACE PANEL ---
if s_state.batch_results:
    target_pairs = TARGET_SCHEMA
    target_set = {az for az, _ in target_pairs}
    target_cols = [friendly for _, friendly in target_pairs]
    front_cols = AUDIT_COLUMNS + [INVOICE_TYPE_COLUMN]

    # ---- Build the master "Invoices" rows (audit + type + schema + parked one-offs) ----
    invoice_rows = []
    overflow_counts = {}
    for f_name, data in s_state.batch_results.items():
        failed = "error" in data
        row = {
            "File Name": f_name,
            "Processing Status": "Failed" if failed else "Success",
            "Time Elapsed": "-" if failed else f"{data.get('time_taken', 0):.2f}s",
            "Processing Cost": "-" if failed else f"${data.get('total_cost', 0):.4f}",
        }
        fields = data.get("fields", {})
        # NEW: PO / Non-PO classification (derived from whether a PO number was found)
        po_meta = fields.get("PurchaseOrder")
        row[INVOICE_TYPE_COLUMN] = "" if failed else ("PO" if (po_meta and str(po_meta["value"]).strip()) else "Non-PO")
        # Fixed target-schema columns (always present, blank if missing)
        for az, friendly in target_pairs:
            meta = fields.get(az)
            row[friendly] = "" if not meta else str(meta["value"])
        # Parked one-off columns: any other field Azure returned (never "Items")
        for az, meta in fields.items():
            if az in target_set or az == "Items":
                continue
            value = str(meta["value"])
            row[az] = value
            if value.strip():
                overflow_counts[az] = overflow_counts.get(az, 0) + 1
        invoice_rows.append(row)

    # Order parked columns by how often they were populated (most common first)
    all_overflow_keys = set()
    for r in invoice_rows:
        for k in r:
            if k not in front_cols and k not in target_cols:
                all_overflow_keys.add(k)
    overflow_ordered = sorted(all_overflow_keys, key=lambda k: (-overflow_counts.get(k, 0), k))

    final_cols = front_cols + target_cols + overflow_ordered
    capped = len(final_cols) > MAX_TOTAL_COLUMNS
    if capped:
        final_cols = final_cols[:MAX_TOTAL_COLUMNS]

    df_export = pd.DataFrame(invoice_rows).reindex(columns=final_cols).fillna("")

    # ---- Build the "Line Items" rows (one row per line, linked to its invoice) ----
    item_rows = []
    for f_name, data in s_state.batch_results.items():
        if "error" in data:
            continue
        meta = data.get("fields", {}).get("InvoiceId")
        inv_no = "" if not meta else str(meta["value"])
        for li in data.get("line_items", []):
            r = {"Source File": f_name, "Invoice Number": inv_no}
            r.update(li)
            item_rows.append(r)

    item_cols = ["Source File", "Invoice Number", "Line #"] + [friendly for _, friendly in LINE_ITEM_FIELDS]
    if item_rows:
        df_items = pd.DataFrame(item_rows).reindex(columns=item_cols).fillna("")
    else:
        df_items = pd.DataFrame(columns=item_cols)

    # ---- Generate the two-tab Excel buffer ----
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df_export.to_excel(writer, index=False, sheet_name='Invoices')
        df_items.to_excel(writer, index=False, sheet_name='Line Items')
        # FIX #8: enforce real column types/formats (identifiers->text so leading
        # zeros survive; money->summable numbers; dates->real dates).
        _xl_write_typed(writer.sheets['Invoices'], df_export, INVOICE_COLUMN_TYPES)
        _xl_write_typed(writer.sheets['Line Items'], df_items, LINE_ITEM_COLUMN_TYPES)
    buffer.seek(0)

    # UI Header Action Layer
    col_title, col_dl = st.columns([3, 1])
    with col_title:
        st.subheader("📊 Global Consolidated Batch Matrix")
    with col_dl:
        st.download_button(
            label="📥 Export Master Excel Sheet",
            data=buffer,
            file_name="extracted_invoice_matrix.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if capped:
        st.warning(f"Hit the {MAX_TOTAL_COLUMNS}-column limit; the least-common parked fields were dropped from the master sheet.")
    parked_count = len(final_cols) - len(front_cols) - len(target_cols)
    st.caption(f"{len(target_cols)} target columns + {parked_count} parked one-off column(s). Line items are on the 'Line Items' tab of the Excel download.")

    # Render Main Spreadsheet Grid
    st.dataframe(_grid_display(df_export, INVOICE_COLUMN_TYPES), use_container_width=True, hide_index=True)

    # Line items preview (also written to the second Excel tab)
    with st.expander(f"📦 Line Items — {len(df_items)} row(s) across the batch"):
        st.dataframe(_grid_display(df_items, LINE_ITEM_COLUMN_TYPES), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Split Review Segment Panel
    st.subheader("🔍 Granular Audit & Extraction Review Workspace")
    col_selector, col_viewer = st.columns([1, 2])

    with col_selector:
        st.write("**Select File to Inspect:**")
        for filename in s_state.batch_results.keys():
            is_active = "primary" if s_state.selected_file == filename else "secondary"
            if st.button(f"📄 {filename}", key=f"sel_{filename}", use_container_width=True, type=is_active):
                s_state.selected_file = filename
                st.rerun()

    with col_viewer:
        if s_state.selected_file and s_state.selected_file in s_state.batch_results:
            active_doc = s_state.batch_results[s_state.selected_file]
            st.markdown(f"#### Active Target Workspace: `{s_state.selected_file}`")

            # 📈 Premium UI Metric Row Blocks
            if "error" not in active_doc:
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric(label="📄 Document Length", value=f"{active_doc.get('page_count', 1)} Pages")
                with m_col2:
                    st.metric(label="⏱️ Processing Speed", value=f"{active_doc.get('time_taken', 0):.2f} sec")
                with m_col3:
                    st.metric(label="💰 Estimated API Cost", value=f"${active_doc.get('total_cost', 0):.4f}")
                st.markdown("<br>", unsafe_allow_html=True)

            if "error" in active_doc:
                st.error(f"Processing Failure: {active_doc['error']}")
            elif not active_doc["fields"]:
                st.warning("No data fields parsed from document structure.")
            else:
                # number the fields (skip Items) so boxes and table rows share an index
                numbered = []
                for i, (fname, fmeta) in enumerate(
                    [(k, v) for k, v in active_doc["fields"].items() if k != "Items"], start=1
                ):
                    numbered.append((i, fname, fmeta))

                # ---------- #1: visual bounding-box overlay ----------
                file_bytes = active_doc.get("file_bytes")
                pages_meta = active_doc.get("pages_meta", {})
                can_render = (file_bytes is not None) and _RENDER_OK
                pages_with_boxes = sorted({
                    reg["page"]
                    for _, _, fmeta in numbered
                    for reg in fmeta.get("regions", [])
                })

                if can_render and pages_with_boxes:
                    st.markdown("**📍 Field locations on the document** — boxes are colour-coded: "
                                "🟢 Azure match · 🟠 low confidence · ⚪ no score · 🔵 LLM fallback. "
                                "Numbers match the table below.")

                    # Map each numbered field to the page its (first) box sits on.
                    field_page = {}
                    for idx, fname, fmeta in numbered:
                        regs = fmeta.get("regions", [])
                        if regs and regs[0].get("page") is not None:
                            field_page[idx] = (regs[0]["page"], fname, fmeta)

                    # "Focus a field" jumps to its page and highlights its box -- the
                    # native, dependency-free stand-in for clicking a box (Streamlit
                    # can't capture clicks on an image without a custom component).
                    focus_options = ["Show all fields"] + [
                        f"{idx} · {fname}" for idx, (pg, fname, _m) in sorted(field_page.items())
                    ]
                    focus_choice = st.selectbox("🔍 Focus a field", focus_options,
                                                key=f"focus_{s_state.selected_file}")
                    focus_idx = None
                    if focus_choice != "Show all fields":
                        focus_idx = int(focus_choice.split(" · ", 1)[0])

                    # Decide which single page to show (no more stacked images).
                    if focus_idx is not None:
                        page_to_show = field_page[focus_idx][0]
                        st.caption(f"Showing page {page_to_show} — where field {focus_choice} is located.")
                    elif len(pages_with_boxes) > 1:
                        page_to_show = st.radio("Page", pages_with_boxes, horizontal=True,
                                                format_func=lambda p: f"Page {p}",
                                                key=f"page_{s_state.selected_file}")
                    else:
                        page_to_show = pages_with_boxes[0]

                    try:
                        base = _render_page_image(file_bytes, active_doc.get("source_file", s_state.selected_file), page_to_show, DPI)
                        annotated = _annotate_page(base, page_to_show, numbered, pages_meta, DPI, focus_idx=focus_idx)
                        st.image(annotated, caption=f"Page {page_to_show}", use_container_width=True)
                    except Exception as render_err:
                        st.caption(f"(Could not render page {page_to_show}: {render_err})")

                    # When a field is focused, surface its details right under the image.
                    if focus_idx is not None:
                        _, ff_name, ff_meta = field_page[focus_idx]
                        ff_conf = ff_meta.get("confidence")
                        if ff_meta.get("source") == "LLM Fallback":
                            ff_conf_txt = "LLM fallback (no score)"
                        elif ff_conf is None:
                            ff_conf_txt = "N/A"
                        elif ff_conf >= CONFIDENCE_THRESHOLD:
                            ff_conf_txt = f"{ff_conf*100:.1f}% (Azure)"
                        else:
                            ff_conf_txt = f"{ff_conf*100:.1f}% (low)"
                        ff_val = "" if ff_meta.get("value") is None else str(ff_meta["value"])
                        st.info(f"**{ff_name}** → {ff_val}  ·  confidence: {ff_conf_txt}")
                elif not _RENDER_OK:
                    st.info("Install PyMuPDF and Pillow (`pip install PyMuPDF Pillow`) to see the visual "
                            "page overlay. Field locations are listed in the table below regardless.")

                # ---------- field table (native st.dataframe -- safe + sortable) ----------
                # Built from DATA, not HTML, so a value containing <, &, or quotes
                # (e.g. "Smith & Sons <Ltd>") can never break the layout. This
                # replaces the old hand-assembled, unescaped HTML table.
                table_rows = []
                for idx, f_name, f_meta in numbered:
                    conf = f_meta.get("confidence")   # float, or None when there is no real score

                    # Confidence shown honestly (mirrors the overlay box colours from #4):
                    if f_meta["source"] == "LLM Fallback":
                        conf_text = "🔄 LLM — no score"      # LLM gives no calibrated number
                    elif conf is None:
                        conf_text = "⚪ N/A"                  # Azure attached no score
                    elif conf >= CONFIDENCE_THRESHOLD:
                        conf_text = f"🟢 {conf*100:.1f}% Match"
                    else:
                        conf_text = f"⚠️ {conf*100:.1f}% Low"

                    regions = f_meta.get("regions", [])
                    if regions and len(regions[0].get("polygon", [])) >= 2:
                        pg = regions[0]["page"]
                        poly = regions[0]["polygon"]
                        unit = pages_meta.get(pg, {}).get("unit", "inch")
                        ua = "in" if unit == "inch" else "px"
                        loc = f"p{pg} · ({poly[0]:.2f}, {poly[1]:.2f}) {ua}"
                    else:
                        loc = "—"

                    table_rows.append({
                        "#": idx,
                        "Field": f_name,
                        "Extracted Value": "" if f_meta.get("value") is None else str(f_meta["value"]),
                        "Confidence": conf_text,
                        "Source": f_meta["source"],
                        "Location": loc,
                    })

                if table_rows:
                    df_fields = pd.DataFrame(table_rows)
                    st.dataframe(
                        df_fields,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "#": st.column_config.NumberColumn(
                                "#", help="Matches the numbered box on the page above.", width="small"
                            ),
                            "Field": st.column_config.TextColumn("Field", width="medium"),
                            "Extracted Value": st.column_config.TextColumn("Extracted Value", width="large"),
                            "Confidence": st.column_config.TextColumn(
                                "Confidence",
                                help="🟢 ≥80% Azure · ⚠️ <80% Azure (routed to LLM) · ⚪ no score · 🔄 LLM fallback",
                                width="small",
                            ),
                            "Source": st.column_config.TextColumn("Source", width="small"),
                            "Location": st.column_config.TextColumn(
                                "Location",
                                help="Page and top-left coordinate where Azure found the field.",
                                width="small",
                            ),
                        },
                    )
                else:
                    st.caption("No field-level data to display (this document only produced line items).")
        else:
            st.info("💡 Select an individual invoice tracking token from the left selector panel to view granular data mappings and audit probability matrices.")
else:
    st.info("👈 Upload your targeted multi-page invoice bundles in the left panel ingestion layer and trigger execution to begin compilation rows.")
