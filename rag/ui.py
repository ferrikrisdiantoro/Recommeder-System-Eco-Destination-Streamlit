# rag/ui.py — FULL REPLACEMENT
from typing import List, Dict
import streamlit as st
from .config import RAGSettings
from .index import RagIndex
from .chain import ask

# Sedikit CSS untuk chips sitasi
CHAT_CSS = """
<style>
.badge { font-size:12px; padding:2px 8px; border:1px solid #33415566; border-radius:9999px; display:inline-block; }
.cites { margin-top:6px; display:flex; gap:6px; flex-wrap: wrap; }
.cite { font-size:12px; padding:2px 8px; border:1px solid #33415566; border-radius:999px; }
.toolbar { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
</style>
"""

def _history_for_chain() -> List[Dict[str, str]]:
    # Ambil seluruh riwayat (user+assistant) untuk konteks follow-up
    return st.session_state.get("rag_msgs", [])

def render_chatbot_panel(settings: RAGSettings, index: RagIndex):
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    st.subheader("🤖 Chatbot AI (RAG)")

    if "rag_msgs" not in st.session_state:
        st.session_state["rag_msgs"] = [
            {"role": "assistant", "content": "Halo! Silakan ajukan pertanyaan. Aku akan jawab berdasarkan dokumen yang sudah diindeks."}
        ]
    if "rag_busy" not in st.session_state:
        st.session_state["rag_busy"] = False

    # Toolbar: clear + info index
    with st.container():
        c1, c2 = st.columns([1, 6])
        with c1:
            if st.button("🧹 Clear"):
                st.session_state["rag_msgs"] = [
                    {"role": "assistant", "content": "Riwayat dibersihkan. Tanyakan sesuatu terkait dokumen yang sudah diindeks ya!"}
                ]
                st.rerun()
        with c2:
            try:
                st.markdown(f'<span class="badge">Index: {index.count()} vektor</span>', unsafe_allow_html=True)
            except Exception:
                st.markdown(f'<span class="badge">Index: n/a</span>', unsafe_allow_html=True)

    # ===== Messages (pakai chat_message agar alignment & markdown bagus)
    for m in st.session_state["rag_msgs"]:
        role = m.get("role", "assistant")
        content = m.get("content", "")
        cites = m.get("citations", [])
        with st.chat_message("user" if role == "user" else "assistant",
                             avatar="🧑" if role == "user" else "🤖"):
            # Render markdown murni → bullet/list rapi
            st.markdown(content)
            # Chips sitasi (sudah dide-dup di chain)
            if cites and role != "user":
                chips = " ".join(
                    [f'<span class="cite">{(c.get("source") or "src")}{", p."+str(c.get("page")) if c.get("page") else ""}</span>'
                     for c in cites]
                )
                st.markdown(f'<div class="cites">{chips}</div>', unsafe_allow_html=True)

    # ===== Input (sticky di bawah)
    prompt = st.chat_input("Tulis pertanyaan…")
    if prompt:
        st.session_state["rag_msgs"].append({"role": "user", "content": prompt})
        st.session_state["rag_busy"] = True
        try:
            ans, cites = ask(
                index=index,
                settings=settings,
                query=prompt,
                k=6,
                history=_history_for_chain()  # kirim riwayat → follow-up paham
            )
            st.session_state["rag_msgs"].append(
                {"role": "assistant", "content": ans, "citations": cites}
            )
        except Exception as e:
            st.session_state["rag_msgs"].append(
                {"role": "assistant", "content": f"(error) {e}"}
            )
        finally:
            st.session_state["rag_busy"] = False
            st.rerun()

# Backward compatibility untuk app.py lama
def render_chatbot(settings: RAGSettings, index: RagIndex):
    return render_chatbot_panel(settings, index)
