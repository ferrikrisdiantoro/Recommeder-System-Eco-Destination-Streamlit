from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from .embed import GeminiEmbedder
from .parser import parse_files
from .chunk import chunk_text
from .config import RAGSettings

def _sanitize_meta(m: Dict[str, Any]) -> Dict[str, Any]:
    """
    Chroma (Rust bindings) hanya terima Bool | Int | Float | Str | SparseVector.
    - Buang key dengan None
    - 'page' -> int
    - 'source'/'tags' -> str
    """
    clean: Dict[str, Any] = {}
    for k, v in (m or {}).items():
        if v is None:
            continue
        if k == "page":
            try:
                clean[k] = int(v)
            except Exception:
                clean[k] = 0
        elif isinstance(v, (bool, int, float, str)):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean

def _normalize_where(where: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Chroma >=0.5 tidak menerima where={} (kosong), dan prefer operator ($and/$or).
    - Jika None/{} -> return None (jangan kirim 'where' ke query)
    - Jika sudah pakai operator ($and/$or/...) -> biarkan
    - Jika dict sederhana {k: v} -> ubah ke {"$and":[{k: {"$eq": v}}, ...]}
    """
    if not where:
        return None
    if any(str(k).startswith("$") for k in where.keys()):
        return where
    clauses = []
    for k, v in where.items():
        clauses.append({k: {"$eq": v}})
    return {"$and": clauses} if clauses else None

class RagIndex:
    def __init__(self, settings: RAGSettings):
        self.settings = settings
        os.makedirs(self.settings.chroma_db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.settings.chroma_db_path,
            settings=ChromaSettings(allow_reset=True)
        )
        self.collection = self.client.get_or_create_collection("rag_docs")
        self.embedder = GeminiEmbedder(api_key=self.settings.google_api_key, model=self.settings.embedding_model)

    # -------- Ingestion --------
    def ingest_paths(self, paths: List[str], tags: Optional[str] = "") -> int:
        docs = parse_files(paths, llama_api_key=self.settings.llama_cloud_api_key)
        all_chunks = []
        for src, txt in docs:
            chs = chunk_text(source=src, text=txt, chunk_size=1200, chunk_overlap=200)
            for c in chs:
                c["tags"] = tags or ""   # str, bukan None
            all_chunks.extend(chs)

        if not all_chunks:
            return 0

        ids = [c["id"] for c in all_chunks]
        texts = [c["text"] for c in all_chunks]

        # Bersihkan metadata dari nilai None
        metas = [_sanitize_meta({
            "source": c.get("source", ""),
            "page": c.get("page", 0),
            "tags": c.get("tags", "")
        }) for c in all_chunks]

        # Embeddings
        embs = self.embedder.embed(texts)

        # Tambahkan ke Chroma
        self.collection.add(ids=ids, documents=texts, embeddings=embs, metadatas=metas)
        return len(all_chunks)

    # -------- Query --------
    def retrieve(self, query: str, k: int = 6, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        qemb = self.embedder.embed_one(query)
        qkwargs: Dict[str, Any] = {"query_embeddings": [qemb], "n_results": k}

        norm_where = _normalize_where(where)
        if norm_where is not None:
            qkwargs["where"] = norm_where  # hanya kirim kalau valid

        res = self.collection.query(**qkwargs)

        out: List[Dict[str, Any]] = []
        if not res or not res.get("ids"):
            return out

        ids = res["ids"][0]
        docs = res.get("documents", [[]])[0] if res.get("documents") else []
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        dists = res.get("distances", [[]])[0] if res.get("distances") else []

        for i in range(len(ids)):
            md = metas[i] if i < len(metas) else {}
            out.append({
                "id": ids[i],
                "text": docs[i] if i < len(docs) else "",
                "metadata": md,
                "score": float(dists[i]) if i < len(dists) else None,
                "source": (md or {}).get("source"),
                "page": (md or {}).get("page"),
            })
        return out
