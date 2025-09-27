from typing import List, Dict, Any
import os
import google.generativeai as genai

def chat_gemini(
    system_prompt: str,
    user_query: str,
    context_blocks: List[Dict[str, Any]],
    model: str = "gemini-2.5-flash",
    temperature: float = 0.3,
) -> str:
    """
    Generate jawaban memakai Gemini tanpa mereset konfigurasi global.
    - Tidak memanggil genai.configure() kosong.
    - Jika ada GOOGLE_API_KEY di env, set sekali (aman, tidak menimpa dengan None).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)  # hanya set jika tersedia

    citations_text = ""
    for i, c in enumerate(context_blocks, 1):
        src = c.get("source", "unknown")
        page = c.get("page", "-")
        citations_text += f"\n[CTX {i}] ({src} p.{page})\n{c.get('text','')}\n"

    prompt = f"""{system_prompt}

# User Query
{user_query}

# Context (from retrieval)
{citations_text}

# Instructions
- Jawab **hanya** berdasarkan konteks di atas; jika tidak cukup, katakan tidak tahu dan sarankan unggah dokumen/pertanyaan lebih spesifik.
- Cantumkan sitasi ringkas seperti (Source, p.X) bila menyebut fakta.
- Bahasa Indonesia yang jelas dan ringkas.
"""
    try:
        model_obj = genai.GenerativeModel(model_name=model)
        resp = model_obj.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        text = getattr(resp, "text", "") or ""
        # fallback kalau .text kosong
        if not text and getattr(resp, "candidates", None):
            parts = []
            for cand in resp.candidates:
                for part in getattr(cand, "content", {}).get("parts", []):
                    parts.append(str(part))
            text = "\n".join(parts)
    except Exception as e:
        text = f"(LLM error) {e}"
    return text
