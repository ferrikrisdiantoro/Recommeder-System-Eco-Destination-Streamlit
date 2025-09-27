# store_chroma.py (fixed)
# Usage:
#   python store_chroma.py
# Inputs:
#   chunks.jsonl + embeddings.jsonl
# Effect:
#   Menulis ke koleksi Chroma (vector DB) dengan metadata tersanitasi.

import os, sys, json, math
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import Any, Dict

CHUNKS = "chunks.jsonl"
EMBEDS = "embeddings.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def as_str(x: Any) -> str:
    return "" if x is None else str(x)

def as_int(x: Any, default: int = -1) -> int:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return default
        return int(float(x))
    except Exception:
        return default

def sanitize_meta(c: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be Bool/Int/Float/Str/SparseVector (no None)."""
    meta = {
        "source": as_str(c.get("source")),
        "page": as_int(c.get("page"), -1),
    }
    # Jika kamu punya field lain (tags, section, dll), tambahkan dan sanitasi di sini:
    if "tags" in c:
        meta["tags"] = as_str(c.get("tags"))
    return meta

def main():
    if not os.path.exists(CHUNKS) or not os.path.exists(EMBEDS):
        print("butuh chunks.jsonl dan embeddings.jsonl. Jalankan langkah sebelumnya.", file=sys.stderr)
        sys.exit(2)

    chroma_path = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
    collection_name = os.environ.get("CHROMA_COLLECTION", "demo_book")
    batch_size = int(os.environ.get("CHROMA_BATCH", "512"))

    client = chromadb.PersistentClient(
        path=chroma_path,
        settings=ChromaSettings(allow_reset=True)
    )
    col = client.get_or_create_collection(collection_name)

    # Muat data
    chunk_map = {c["id"]: c for c in read_jsonl(CHUNKS)}
    embeds = list(read_jsonl(EMBEDS))

    # Pastikan dimensi embedding konsisten; tentukan dari item valid pertama
    emb_dim = None
    valid = []
    skipped_no_chunk = 0
    skipped_no_emb = 0
    skipped_dim_mismatch = 0

    for e in embeds:
        cid = e.get("id")
        vec = e.get("embedding")
        c = chunk_map.get(cid)

        if not c:
            skipped_no_chunk += 1
            continue
        if not isinstance(vec, list) or not vec:
            skipped_no_emb += 1
            continue

        if emb_dim is None:
            emb_dim = len(vec)
        if len(vec) != emb_dim:
            skipped_dim_mismatch += 1
            continue

        valid.append((cid, c, vec))

    if not valid:
        print("Tidak ada pasangan chunk+embedding yang valid.", file=sys.stderr)
        print(f"skipped_no_chunk={skipped_no_chunk}, skipped_no_emb={skipped_no_emb}, skipped_dim_mismatch={skipped_dim_mismatch}")
        sys.exit(3)

    # Tulis ke Chroma per batch
    total = len(valid)
    n_batches = math.ceil(total / batch_size)
    for b in range(n_batches):
        part = valid[b*batch_size : (b+1)*batch_size]
        ids   = [cid for cid, _, _ in part]
        docs  = [str(c.get("text", "")) for _, c, _ in part]
        metas = [sanitize_meta(c) for _, c, _ in part]
        embs  = [vec for _, _, vec in part]

        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    print(f"[ok] stored to Chroma collection '{collection_name}' (count={total}). Path: {chroma_path}")
    if skipped_no_chunk or skipped_no_emb or skipped_dim_mismatch:
        print(f"[info] skipped_no_chunk={skipped_no_chunk}, skipped_no_emb={skipped_no_emb}, skipped_dim_mismatch={skipped_dim_mismatch}")

if __name__ == "__main__":
    main()
