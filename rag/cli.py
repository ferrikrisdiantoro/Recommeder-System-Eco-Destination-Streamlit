import argparse, os, sys, json, shutil
from typing import List
from .config import RAGSettings
from .index import RagIndex
from .chain import ask

def _load_index() -> tuple[RAGSettings, RagIndex]:
    settings = RAGSettings.from_env()
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY belum diset di env/secrets.")
    idx = RagIndex(settings)
    return settings, idx

def cmd_health(args):
    settings, idx = _load_index()
    try:
        cnt = idx.collection.count()  # chroma >=0.5.x
    except Exception:
        cnt = None
    print(json.dumps({
        "chroma_db_path": settings.chroma_db_path,
        "collection": "rag_docs",
        "vectors": cnt,
        "embedding_model": settings.embedding_model,
        "chat_model": settings.chat_model
    }, indent=2, ensure_ascii=False))

def _gather_paths(files: List[str], dir_: str | None) -> List[str]:
    paths = []
    if dir_:
        for root, _, fnames in os.walk(dir_):
            for n in fnames:
                if n.startswith("."): 
                    continue
                paths.append(os.path.join(root, n))
    for f in files or []:
        paths.append(f)
    # de-dup & keep existing
    uniq = []
    seen = set()
    for p in paths:
        rp = os.path.realpath(p)
        if rp in seen or not os.path.exists(rp):
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq

def cmd_ingest(args):
    _, idx = _load_index()
    paths = _gather_paths(args.files, args.dir)
    if not paths:
        print("Tidak ada file untuk diindeks.", file=sys.stderr)
        sys.exit(1)
    n = idx.ingest_paths(paths, tags=args.tags or "")
    print(json.dumps({"ingested_chunks": n, "files": paths}, indent=2, ensure_ascii=False))

def cmd_query(args):
    settings, idx = _load_index()
    q = args.question.strip()
    if not q:
        print("Pertanyaan kosong.", file=sys.stderr)
        sys.exit(1)
    ans, cites = ask(index=idx, settings=settings, query=q, k=args.k,
                     model=args.model or None, temperature=args.temperature)
    if args.json:
        print(json.dumps({"answer": ans, "citations": cites}, indent=2, ensure_ascii=False))
    else:
        print("\n=== JAWABAN ===\n")
        print(ans)
        if cites:
            print("\n--- Sitasi ---")
            for c in cites:
                src = c.get("source") or "source"
                pg = c.get("page")
                print(f"- {src}{', p.'+str(pg) if pg else ''}")

def cmd_reset(args):
    settings, _ = _load_index()
    path = settings.chroma_db_path
    if not os.path.isdir(path):
        print(f"Tidak ada folder Chroma di: {path}")
        return
    if not args.yes:
        resp = input(f"Yakin hapus index di '{path}'? [y/N] ").strip().lower()
        if resp != "y":
            print("Batal.")
            return
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)
    print(f"Index direset: {path}")

def main():
    p = argparse.ArgumentParser(prog="python -m rag.cli", description="RAG CLI (Gemini + LlamaParse + Chroma)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("health", help="Cek status index & model")
    s.set_defaults(func=cmd_health)

    s = sub.add_parser("ingest", help="Indeks file/direktori ke Chroma")
    s.add_argument("files", nargs="*", help="File untuk diindeks (PDF/TXT/...)")
    s.add_argument("--dir", help="Direktori (recursive) untuk diindeks")
    s.add_argument("--tags", default="", help="Tag metadata opsional")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("query", help="Ajukan pertanyaan ke RAG")
    s.add_argument("question", help="Pertanyaan")
    s.add_argument("-k", type=int, default=6, help="Top-K retriever")
    s.add_argument("--model", default=None, help="Override model chat Gemini")
    s.add_argument("--temperature", type=float, default=0.3, help="Suhu generasi")
    s.add_argument("--json", action="store_true", help="Output JSON")
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("reset", help="Reset/hapus index (Chroma)")
    s.add_argument("-y", "--yes", action="store_true", help="Tanpa konfirmasi")
    s.set_defaults(func=cmd_reset)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
