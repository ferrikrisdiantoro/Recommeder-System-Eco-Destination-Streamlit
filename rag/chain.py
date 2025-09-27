from typing import Tuple, List, Dict, Any
from .index import RagIndex
from .config import RAGSettings
from .llm import chat_gemini

SYSTEM_PROMPT = """Kamu adalah asisten RAG yang akurat untuk audiens Indonesia.
Selalu sertakan sitasi seperti (filename, p.X). Hindari halusinasi; jika konteks tidak memadai, katakan tidak tahu dan sarankan unggah dokumen.
Jawab ringkas, to the point. Gunakan bullet bila menjelaskan langkah/prosedur.
"""

def ask(index: RagIndex, settings: RAGSettings, query: str, k: int = 6,
        model: str = None, temperature: float = 0.3) -> Tuple[str, List[Dict[str, Any]]]:
    hits = index.retrieve(query, k=k)
    ctx = [{"text": h["text"], "source": h.get("source"), "page": h.get("page")} for h in hits]
    answer = chat_gemini(
        system_prompt=SYSTEM_PROMPT,
        user_query=query,
        context_blocks=ctx,
        model=model or settings.chat_model,
        temperature=temperature
    )
    citations = [{"source": h.get("source"), "page": h.get("page")} for h in hits]
    return answer, citations
