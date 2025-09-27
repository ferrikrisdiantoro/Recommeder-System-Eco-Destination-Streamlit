# chunking.py
# Usage:
#   python chunking.py
# Input :
#   parsed.txt
# Output:
#   chunks.jsonl  (tiap baris: {"id","source","page","text"})

import os, json
from langchain.text_splitter import RecursiveCharacterTextSplitter

SRC_TXT = "parsed.txt"
OUT_JSONL = "chunks.jsonl"

def main():
    if not os.path.exists(SRC_TXT):
        raise SystemExit("parsed.txt tidak ditemukan. Jalankan parsing.py terlebih dahulu.")

    with open(SRC_TXT, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "200"))
    SOURCE = os.environ.get("SOURCE_NAME", "book.pdf")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)

    with open(OUT_JSONL, "w", encoding="utf-8") as w:
        for i, ch in enumerate(chunks, 1):
            rec = {
                "id": f"{SOURCE}::chunk{i:04d}",
                "source": SOURCE,
                "page": None,
                "text": ch
            }
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[ok] saved: {OUT_JSONL} (total chunks: {len(chunks)})")

if __name__ == "__main__":
    main()
