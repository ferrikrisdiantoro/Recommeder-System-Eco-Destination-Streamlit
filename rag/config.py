from dataclasses import dataclass
import os

@dataclass
class RAGSettings:
    google_api_key: str = "AIzaSyA7J02YavJnOQVehsUUJT47svizCdVqLp4"
    chroma_db_path: str = "./chroma_db"
    llama_cloud_api_key: str = ""
    embedding_model: str = "text-embedding-004"
    chat_model: str = "gemini-2.5-flash"

    @classmethod
    def from_env(cls):
        return cls(
            google_api_key=os.environ.get("GOOGLE_API_KEY", "AIzaSyA7J02YavJnOQVehsUUJT47svizCdVqLp4"),
            chroma_db_path=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
            llama_cloud_api_key=os.environ.get("LLAMA_CLOUD_API_KEY", ""),
            embedding_model=os.environ.get("EMBED_MODEL", "text-embedding-004"),
            chat_model=os.environ.get("CHAT_MODEL", "gemini-2.5-flash"),
        )
