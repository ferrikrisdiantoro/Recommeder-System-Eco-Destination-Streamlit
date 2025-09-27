from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(source: str, text: str, chunk_size: int = 1200, chunk_overlap: int = 200) -> List[Dict]:
    """
    Chunking menggunakan LangChain (RecursiveCharacterTextSplitter).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_text(text)
    out = []
    for i, ch in enumerate(chunks, 1):
        out.append({
            "id": f"{source}::chunk{i}",
            "source": source,
            "page": 0,            # <— jangan None
            "text": ch
        })
    return out
