from typing import List, Dict, Any, Optional
import os
import google.generativeai as genai

def chat_gemini(
    system_prompt: str,
    user_query: str,
    context_blocks: List[Dict[str, Any]],
    history_text: str = "",
    structured_facts: str = "",
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> str:
    """
    Bangun prompt dengan:
    - Riwayat percakapan ringkas (untuk follow-up)
    - Konteks retrieval (teks + metadata disarikan)
    - Structured facts (mis. MAP link) agar model eksplisit mengekspose jika diminta.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)

    ctx_text = []
    for i, c in enumerate(context_blocks, 1):
        src = c.get("source", "unknown")
        page = c.get("page", "-")
        text = c.get("text", "") or ""
        # ringkas blok dengan label sumber
        ctx_text.append(f"[CTX {i}] ({src} p.{page})\n{text}")

        # tambahkan highlight metadata penting agar mudah 'terbaca' model
        meta = c.get("meta") or {}
        prom_meta = []
        for key in ("place_name", "city", "category", "price_str", "price_num", "rating", "rating_avg", "map_url"):
            if key in meta and meta[key]:
                prom_meta.append(f"{key}={meta[key]}")
        if prom_meta:
            ctx_text.append(f"[META {i}] {', '.join(prom_meta)}")

    ctx_blob = "\n\n".join(ctx_text)

    prompt = f"""{system_prompt}

# Conversation so far (last turns)
{history_text or '(none)'}

# User Query
{user_query}

# Retrieved Context
{ctx_blob}

# Structured Facts (if any)
{structured_facts or '(none)'}

# Instructions
- Gunakan *Conversation so far* untuk memahami rujukan (mis. "itu", "tempat tsb").
- Jawab **hanya** dari konteks. Jika tidak ada jawaban dalam konteks, katakan tidak tahu.
- Jika pertanyaan meminta link peta dan metadata `map_url` tersedia, tampilkan tautannya.
- Tampilkan sitasi ringkas di dalam jawaban saat menyebut fakta (format: (filename, p.X)).
- Bahasa Indonesia, ringkas dan jelas.
"""

    try:
        model_obj = genai.GenerativeModel(model_name=model)
        resp = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        text = getattr(resp, "text", "") or ""
        if not text and getattr(resp, "candidates", None):
            parts = []
            for cand in resp.candidates:
                for part in getattr(cand, "content", {}).get("parts", []):
                    parts.append(str(part))
            text = "\n".join(parts)
    except Exception as e:
        text = f"(LLM error) {e}"

    return text
