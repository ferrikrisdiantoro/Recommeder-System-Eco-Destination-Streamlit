from typing import Tuple, List, Dict, Any, Optional
from .index import RagIndex
from .config import RAGSettings
from .llm import chat_gemini

SYSTEM_PROMPT = """Kamu adalah asisten RAG yang akurat untuk audiens Indonesia.
- Jawab hanya berdasarkan konteks; jika tidak cukup, katakan tidak tahu dan sarankan unggah/tambah dokumen.
- Cantumkan sitasi ringkas seperti (filename, p.X) saat menyebut fakta.
- Jika metadata `map_url` tersedia dan relevan, tampilkan tautannya.
- **Selalu format daftar sebagai Markdown bullet**: satu item per baris, awali dengan tanda minus `- `. Jangan menaruh bullet dalam satu paragraf panjang.
- Bahasa Indonesia yang ringkas, jelas, dan rapi.
"""

def _history_to_text(history: Optional[List[Dict[str, str]]], max_turns: int = 6) -> str:
    if not history:
        return ""
    pairs = history[-max_turns:]
    lines = []
    for m in pairs:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        else:
            lines.append(f"Assistant: {content}")
    return "\n".join(lines)

def _dedupe_citations(hits: List[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("source") or "source"
        if src in seen:
            continue
        seen.add(src)
        out.append({"source": src, "page": h.get("page", None)})
        if len(out) >= limit:
            break
    return out

def ask(index: RagIndex, settings: RAGSettings, query: str, k: int = 6,
        model: Optional[str] = None, temperature: float = 0.3,
        history: Optional[List[Dict[str, str]]] = None) -> Tuple[str, List[Dict[str, Any]]]:

    hits = index.retrieve(query, k=k)

    ctx_blocks = []
    structured_facts = []
    for h in hits:
        md = h.get("metadata") or {}
        ctx_blocks.append({
            "text": h.get("text", ""),
            "source": h.get("source"),
            "page": h.get("page"),
            "meta": md
        })
        place = md.get("place_name") or md.get("place_id") or None
        map_url = md.get("map_url")
        if place and map_url:
            structured_facts.append(f"- MAP: {place} -> {map_url}")

    history_text = _history_to_text(history, max_turns=6)

    answer = chat_gemini(
        system_prompt=SYSTEM_PROMPT,
        user_query=query,
        context_blocks=ctx_blocks,
        history_text=history_text,
        structured_facts="\n".join(structured_facts) if structured_facts else "",
        model=model or settings.chat_model,
        temperature=temperature
    )

    citations = _dedupe_citations(hits, limit=3)
    return answer, citations
