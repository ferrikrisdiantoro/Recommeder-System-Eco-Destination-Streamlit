import os, sys

# Optional: LlamaParse
try:
    from llama_parse import LlamaParse
except Exception:
    LlamaParse = None

# Fallback: PyPDF2
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

def parse_with_llamaparse(pdf_path: str, api_key: str) -> str:
    parser = LlamaParse(api_key=api_key, result_type="text")
    result = parser.load_data(pdf_path)
    return "\n".join([d.text for d in result if getattr(d, "text", "")])

def parse_with_pypdf2(pdf_path: str) -> str:
    if not PdfReader:
        return ""
    reader = PdfReader(pdf_path)
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return "\n".join(texts)

def main():
    pdf = sys.argv[1] if len(sys.argv) > 1 else "perahu-kertas.pdf"
    if not os.path.exists(pdf):
        print(f"[error] File not found: {pdf}")
        sys.exit(2)

    llama_key = os.environ.get("LLAMA_CLOUD_API_KEY")
    text = ""

    if llama_key and LlamaParse is not None:
        try:
            print("[info] Parsing via LlamaParse ...")
            text = parse_with_llamaparse(pdf, llama_key)
        except Exception as e:
            print("[warn] LlamaParse failed, fallback PyPDF2:", e)

    if not text:
        print("[info] Parsing via PyPDF2 fallback ...")
        text = parse_with_pypdf2(pdf)

    if not text.strip():
        print("[error] Parsing produced empty text.")
        sys.exit(3)

    with open("parsed.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("[ok] saved: parsed.txt")
    print("chars:", len(text))

if __name__ == "__main__":
    main()
