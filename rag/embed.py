from typing import List
import google.generativeai as genai

class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-004"):
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required.")
        genai.configure(api_key=api_key)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            tt = t if (t and t.strip()) else " "
            try:
                r = genai.embed_content(model=self.model, content=tt)
                vec = r.get("embedding") or r.get("embeddings") or []
            except Exception:
                # fallback zero vector
                vec = [0.0] * 768
            out.append(vec)
        return out

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
