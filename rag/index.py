from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from .embed import GeminiEmbedder
from .parser import parse_files
from .chunk import chunk_text
from .config import RAGSettings

def _sanitize_meta(m: Dict[str, Any]) -> Dict[str, Any]:
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

    # -------- Utilities --------
    def count(self) -> int:
        try:
            return int(self.collection.count())
        except Exception:
            got = self.collection.get(limit=1)
            return len(got.get("ids", []))

    def has_source(self, path_or_basename: str) -> bool:
        base = os.path.basename(path_or_basename)
        try:
            res = self.collection.get(where={"source": {"$eq": base}}, limit=1)
            ids = res.get("ids", [])
            if isinstance(ids, list) and ids:
                first = ids[0]
                if isinstance(first, list):
                    return len(first) > 0
                return True
            return False
        except Exception:
            return False

    # -------- Ingestion --------
    def ingest_paths(self, paths: List[str], tags: Optional[str] = "") -> int:
        docs = parse_files(paths, llama_api_key=self.settings.llama_cloud_api_key)

        all_ids: List[str] = []
        all_texts: List[str] = []
        all_metas: List[Dict[str, Any]] = []

        for doc in docs:
            source = doc.get("source", "")
            base_page = int(doc.get("page", 0))
            base_meta = doc.get("meta", {}) or {}
            is_atomic = bool(doc.get("is_atomic", False))

            if is_atomic:
                # satu baris CSV = satu chunk
                cid = f"{source}::row{base_page}"
                all_ids.append(cid)
                all_texts.append(doc.get("text", ""))
                md = {"source": source, "page": base_page, "tags": tags or ""}
                md.update(base_meta)
                all_metas.append(_sanitize_meta(md))
            else:
                # dokumen panjang → chunking
                chs = chunk_text(source=source, text=doc.get("text", ""), chunk_size=1200, chunk_overlap=200)
                for c in chs:
                    c["tags"] = tags or ""
                    # gunakan page asal (0) untuk pdf/txt, boleh juga pakai nomor chunk
                    page_val = c.get("page", base_page)
                    cid = c["id"]
                    all_ids.append(cid)
                    all_texts.append(c["text"])
                    md = {"source": c["source"], "page": page_val, "tags": c["tags"]}
                    md.update(base_meta)  # meta umum dari dokumen asal
                    all_metas.append(_sanitize_meta(md))

        if not all_ids:
            return 0

        embs = self.embedder.embed(all_texts)
        self.collection.add(ids=all_ids, documents=all_texts, embeddings=embs, metadatas=all_metas)
        return len(all_ids)

    # -------- Query --------
    def retrieve(self, query: str, k: int = 6, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        qemb = self.embedder.embed_one(query)
        qkwargs: Dict[str, Any] = {"query_embeddings": [qemb], "n_results": k}

        norm_where = _normalize_where(where)
        if norm_where is not None:
            qkwargs["where"] = norm_where

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
