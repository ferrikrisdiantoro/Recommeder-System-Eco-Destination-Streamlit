import os, sys, json
import google.generativeai as genai

IN_JSONL = "chunks.jsonl"
OUT_JSONL = "embeddings.jsonl"

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        key = input("Masukkan GOOGLE_API_KEY (kosong untuk batal): ").strip()
    if not key:
        raise SystemExit("Tidak ada API key. Set env GOOGLE_API_KEY atau masukkan manual.")

    genai.configure(api_key=key)
    model = os.environ.get("EMBED_MODEL", "text-embedding-004")

    if not os.path.exists(IN_JSONL):
        print("chunks.jsonl tidak ditemukan. Jalankan chunking.py terlebih dahulu.")
        sys.exit(2)

    total, dim = 0, None
    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for rec in read_jsonl(IN_JSONL):
            text = rec.get("text"," ") or " "
            try:
                r = genai.embed_content(model=model, content=text)
                vec = r.get("embedding") or r.get("embeddings") or []
            except Exception as e:
                print("[warn] embedding error:", e)
                vec = []

            if dim is None and isinstance(vec, list):
                dim = len(vec)
            w.write(json.dumps({"id": rec["id"], "embedding": vec}) + "\n")
            total += 1

    print(f"[ok] saved: {OUT_JSONL} (records: {total}, dim: {dim})")

if __name__ == "__main__":
    main()
