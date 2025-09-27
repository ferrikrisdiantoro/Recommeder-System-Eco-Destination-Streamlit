from typing import List, Tuple
import os

# Optional: LlamaParse untuk PDF
try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None

# Fallback PDF reader
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

# CSV support (pandas)
try:
    import pandas as pd
except Exception:
    pd = None


def _read_pdf_basic(path: str) -> str:
    if not PdfReader:
        return ""
    try:
        reader = PdfReader(path)
        txts = []
        for p in reader.pages:
            txts.append(p.extract_text() or "")
        return "\n".join(txts)
    except Exception:
        return ""


def _compose_csv_text(path: str, max_rows: int | None = None) -> str:
    """
    Baca CSV → jadikan ringkasan baris-per-baris agar mudah diretrieval.
    Prioritaskan kolom umum; jika kolom tidak ada, fallback ke semua kolom.
    """
    if pd is None:
        # fallback: baca mentah
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    if df.empty:
        return ""

    # batas baris bila diperlukan
    if isinstance(max_rows, int) and max_rows > 0:
        df = df.head(max_rows)

    # Kolom yang sering ada di dataset ecotourism kita
    preferred_cols = [
        "id", "place_id", "place_name", "place_description",
        "category", "city", "address",
        "price", "price_str", "price_num",
        "rating", "rating_avg",
        "image", "map_url",
    ]

    # Tentukan kolom yang akan dipakai
    cols_present = [c for c in preferred_cols if c in df.columns]
    if not cols_present:
        # fallback: jika tidak ada kolom preferensi, pakai sampel kolom (maks 8)
        cols_present = df.columns.tolist()[:8]

    lines = []
    for _, row in df.iterrows():
        parts = []
        for c in cols_present:
            val = row.get(c, "")
            # stringify aman
            try:
                sval = str(val)
            except Exception:
                sval = ""
            if sval and sval.strip() and sval.lower() != "nan":
                parts.append(f"{c}: {sval}")
        if parts:
            lines.append(" | ".join(parts))

    # Tambahkan header kecil
    header = f"[CSV] {os.path.basename(path)} — {len(df)} baris, kolom dipakai: {', '.join(cols_present)}"
    body = "\n".join(lines)
    return header + "\n" + body if body else header


def parse_files(paths: List[str], llama_api_key: str = "") -> List[Tuple[str, str]]:
    """
    Returns list of (source, text)
    - PDF: LlamaParse jika ada API key; fallback PyPDF2.
    - CSV: pandas → ringkasan baris per baris (lebih ramah retrieval).
    - TXT/MD/LAINNYA: baca sebagai teks biasa.
    """
    docs: List[Tuple[str, str]] = []
    use_llama = bool(llama_api_key and LlamaParse is not None)
    parser = None
    if use_llama:
        parser = LlamaParse(api_key=llama_api_key, result_type="text")

    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        base = os.path.basename(p)
        text = ""

        # PDF
        if ext == ".pdf":
            if use_llama:
                try:
                    result = parser.load_data(p)
                    text = "\n".join([d.text for d in result if getattr(d, "text", "")])
                except Exception:
                    text = ""
            if not text:
                text = _read_pdf_basic(p)

        # CSV
        elif ext == ".csv":
            text = _compose_csv_text(p, max_rows=None)

        # TXT/MD/LAINNYA
        else:
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception:
                text = ""

        if text.strip():
            docs.append((base, text))

    return docs
