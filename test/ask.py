# ask.py — Retrieval + Gemini Answer (default)
# Usage:
#   python ask.py "pertanyaan"
#   python ask.py "pertanyaan" --k 5 --retrieval-only   # kalau mau lihat Top-K saja
#
# Env opsional:
#   GOOGLE_API_KEY=...           # kalau belum ada, script minta input
#   CHROMA_DB_PATH=./chroma_db
#   CHROMA_COLLECTION=demo_book
#   EMBED_MODEL=text-embedding-004
#   CHAT_MODEL=gemini-1.5-pro
#   TOP_K=5

import os, sys, re, textwrap, argparse

# Redam log bising (mungkin tidak 100% hilang di semua OS)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import chromadb
from chromadb.config import Settings as ChromaSettings
import google.generativeai as genai

def clean_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r", " ").replace("\n", " ")
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)   # gabung "do- ngeng" -> "dongeng"
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pretty_snippet(s: str, width: int = 240) -> str:
    s = clean_text(s)
    return textwrap.shorten(s, width=width, placeholder="…")

def build_prompt(query: str, blocks: list[str]) -> str:
    context = "\n\n".join(blocks)
    return f"""
Anda adalah asisten RAG berbahasa Indonesia. Jawab pertanyaan secara ringkas dan jelas
berdasarkan KONTEKS berikut, dan sertakan sitasi [n] di akhir kalimat relevan.

# PERTANYAAN
{query}

# KONTEKS (blok bernomor)
{context}

# ATURAN
- 2–4 kalimat saja, to the point.
- Jika informasi tidak cukup, katakan tidak tahu.
- Wajib sertakan sitasi [n] yang sesuai.
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", help="Pertanyaan pengguna")
    ap.add_argument("--k", type=int, default=int(os.environ.get("TOP_K", "5")), help="Top-K")
    ap.add_argument("--retrieval-only", action="store_true", help="Hanya tampilkan Top-K tanpa merangkum Gemini")
    ap.add_argument("--chunk-limit", type=int, default=900, help="Batas karakter per chunk ke LLM (default 900)")
    args = ap.parse_args()

    query = args.query
    TOP_K = max(1, args.k)

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = input("Masukkan GOOGLE_API_KEY: ").strip()
    if not key:
        raise SystemExit("Tidak ada API key.")

    genai.configure(api_key=key)
    emb_model = os.environ.get("EMBED_MODEL", "text-embedding-004")
    chat_model = os.environ.get("CHAT_MODEL", "gemini-1.5-pro")

    chroma_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "demo_book")

    client = chromadb.PersistentClient(path=chroma_path, settings=ChromaSettings(allow_reset=True))
    col = client.get_or_create_collection(collection_name)

    # 1) Embed query
    q = genai.embed_content(model=emb_model, content=query)
    qvec = q.get("embedding") or q.get("embeddings") or []
    if not qvec:
        raise SystemExit("Embedding query gagal (vector kosong).")

    # 2) Retrieve Top-K
    res = col.query(query_embeddings=[qvec], n_results=TOP_K)
    ids   = res.get("ids", [[]])[0]
    docs  = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    if not ids:
        print("Tidak ada hasil.")
        return

    # 3) Tampilkan Top-K yang rapi
    print("Top-K hasil (rapi):")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        score = dists[i] if i < len(dists) else None
        score_txt = f"{float(score):.3f}" if isinstance(score, (int, float)) else "n/a"
        src = meta.get("source") or "-"
        page = meta.get("page")
        page_txt = f"p.{page}" if isinstance(page, int) and page >= 0 else "-"
        snippet = pretty_snippet(docs[i], width=240)
        print(f"{i+1}. [{src}] {ids[i]} • score={score_txt} • {page_txt}")
        print(f"   {snippet}")

    if args.retrieval_only:
        return

    # 4) Rangkai jawaban dengan Gemini dari Top-K
    blocks = []
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        src = meta.get("source") or "source"
        page = meta.get("page")
        page_txt = f"p.{page}" if isinstance(page, int) and page >= 0 else "-"
        chunk_text = clean_text(docs[i])[: max(100, args.chunk_limit)]  # potong biar hemat konteks
        blocks.append(f"[{i+1}] ({src} {page_txt})\n{chunk_text}")

    prompt = build_prompt(query, blocks)

    try:
        model = genai.GenerativeModel(chat_model)
        resp = model.generate_content(prompt)
        answer = (resp.text or "").strip()
    except Exception as e:
        answer = f"(LLM error) {e}"

    print("\n=== Jawaban (dirangkai Gemini) ===")
    print(answer)

    print("\nSumber:")
    for i in range(len(ids)):
        meta = metas[i] if i < len(metas) else {}
        src = meta.get("source") or "source"
        page = meta.get("page")
        page_txt = f"p.{page}" if isinstance(page, int) and page >= 0 else "-"
        print(f"[{i+1}] {src} {page_txt}")

if __name__ == "__main__":
    main()
