from typing import List, Tuple, Dict, Any
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


def _to_str(x) -> str:
    try:
        s = str(x)
        return "" if s.lower() == "nan" else s
    except Exception:
        return ""


def _csv_rows_as_docs(path: str) -> List[Dict[str, Any]]:
    """Setiap baris CSV menjadi satu dokumen ATOMIK (tanpa chunking) agar metadata per-baris ikut ke index."""
    if pd is None:
        # fallback: baca mentah jika pandas tidak ada
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            base = os.path.basename(path)
            return [{
                "source": base,
                "text": raw,
                "page": 0,
                "meta": {},
                "is_atomic": False,   # akan di-chunk biasa
            }]
        except Exception:
            return []

    try:
        df = pd.read_csv(path)
    except Exception:
        # fallback: teks mentah
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            base = os.path.basename(path)
            return [{
                "source": base,
                "text": raw,
                "page": 0,
                "meta": {},
                "is_atomic": False,
            }]
        except Exception:
            return []

    if df.empty:
        return []

    base = os.path.basename(path)

    # kolom umum (kalau ada)
    preferred_cols = [
        "id", "place_id", "place_name", "place_description",
        "category", "city", "address",
        "price", "price_str", "price_num",
        "rating", "rating_avg",
        "image", "map_url",
    ]
    used_cols = [c for c in preferred_cols if c in df.columns]
    if not used_cols:
        # pakai subset kolom agar ringkas
        used_cols = df.columns.tolist()[:10]

    docs: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        meta: Dict[str, Any] = {}
        parts = []
        for c in used_cols:
            val = _to_str(row.get(c, ""))
            if val:
                parts.append(f"{c}: {val}")
                # masukkan juga sebagai metadata terstruktur
                # NB: metadata harus Bool|Int|Float|Str untuk Chroma Rust → convert ke str/int/float
                if c in ("price_num", "rating", "rating_avg"):
                    try:
                        meta[c] = float(val)
                    except Exception:
                        meta[c] = _to_str(val)
                else:
                    meta[c] = _to_str(val)

        # teks ringkas baris
        text = " | ".join(parts) if parts else ""

        # page = nomor baris (1-based) agar sitasi berguna
        docs.append({
            "source": base,
            "text": text,
            "page": int(i) + 1,
            "meta": meta,
            "is_atomic": True,   # jangan di-chunk lagi
        })

    return docs


def parse_files(paths: List[str], llama_api_key: str = "") -> List[Dict[str, Any]]:
    """
    Returns list of dict:
      {
        "source": <basename>,
        "text": <str>,
        "page": <int>,
        "meta": <dict>,
        "is_atomic": <bool>   # True => dipakai apa adanya, False => akan di-chunk
      }
    """
    out: List[Dict[str, Any]] = []
    use_llama = bool(llama_api_key and LlamaParse is not None)
    parser = None
    if use_llama:
        parser = LlamaParse(api_key=llama_api_key, result_type="text")

    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        base = os.path.basename(p)

        # CSV → per-baris sebagai dokumen atomik
        if ext == ".csv":
            out.extend(_csv_rows_as_docs(p))
            continue

        # PDF
        if ext == ".pdf":
            text = ""
            if use_llama:
                try:
                    result = parser.load_data(p)
                    text = "\n".join([d.text for d in result if getattr(d, "text", "")])
                except Exception:
                    text = ""
            if not text:
                text = _read_pdf_basic(p)
            if text.strip():
                out.append({
                    "source": base,
                    "text": text,
                    "page": 0,
                    "meta": {},
                    "is_atomic": False,  # akan di-chunk
                })
            continue

        # TXT/MD/LAINNYA
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
        except Exception:
            raw = ""
        if raw.strip():
            out.append({
                "source": base,
                "text": raw,
                "page": 0,
                "meta": {},
                "is_atomic": False,
            })

    return out
