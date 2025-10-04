# rag/ui.py — Chatbot panel di MAIN PAGE (tanpa popup/JS)

import os
import streamlit as st
from typing import List, Dict, Any
from .config import RAGSettings
from .index import RagIndex
from .chain import ask


def _ensure_state():
    if "rag_msgs" not in st.session_state:
        st.session_state["rag_msgs"] = [
            {"role": "assistant", "content": "Halo! Unggah dokumen (opsional), lalu ajukan pertanyaan. Aku akan jawab berbasis konteks dan sertakan sitasi."}
        ]
    if "rag_busy" not in st.session_state:
        st.session_state["rag_busy"] = False
    if "rag_temp" not in st.session_state:
        st.session_state["rag_temp"] = 0.3


def _ingest_block(index: RagIndex):
    with st.expander("📥 Indeks Dokumen", expanded=False):
        files = st.file_uploader(
            "Unggah PDF/TXT/CSV/DOCX (multi-file didukung)",
            type=None,
            accept_multiple_files=True,
            help="PDF: LlamaParse (opsional) atau PyPDF2; CSV: diringkas per baris; TXT/DOCX: baca mentah."
        )
        tag = st.text_input("Tag (opsional, untuk filter metadata)", value="", key="rag_tag_panel")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Indeks", width='stretch', disabled=st.session_state["rag_busy"], key="rag_btn_ingest_panel"):
                if not files:
                    st.info("Pilih file terlebih dahulu.")
                else:
                    st.session_state["rag_busy"] = True
                    try:
                        tmpdir = os.path.join(".rag_tmp")
                        os.makedirs(tmpdir, exist_ok=True)
                        paths = []
                        for f in files:
                            p = os.path.join(tmpdir, f.name)
                            with open(p, "wb") as w:
                                w.write(f.read())
                            paths.append(p)
                        n = index.ingest_paths(paths, tags=tag)
                        st.success(f"Berhasil mengindeks {n} chunk.")
                    except Exception as e:
                        st.error(f"Gagal ingest: {e}")
                    finally:
                        st.session_state["rag_busy"] = False
        with col2:
            if st.button("Bersihkan Riwayat Chat", width='stretch', key="rag_btn_clear_panel"):
                st.session_state["rag_msgs"] = []


def _render_history_markdown():
    # Alternatif tampilan markdown (stabil untuk semua versi)
    for m in st.session_state["rag_msgs"]:
        role = m.get("role", "assistant")
        txt = m.get("content", "")
        cites = m.get("citations", [])
        if role == "user":
            st.markdown(f"**Anda:** {txt}")
        else:
            st.markdown(f"**AI:** {txt}")
            if cites:
                chips = []
                for c in cites:
                    src = c.get("source") or "source"
                    pg = c.get("page")
                    chips.append(f"`{src}{', p.'+str(pg) if pg else ''}`")
                st.caption("Sumber: " + "  ".join(chips))


def render_chatbot_panel(settings: RAGSettings, index: RagIndex):
    """
    Tampilkan chatbot RAG di MAIN PAGE (bukan sidebar). 
    Dipanggil dari app.py saat tombol di sidebar di-klik.
    """
    _ensure_state()

    st.header("🤖 AI Chatbot (RAG)")
    _ingest_block(index)

    st.divider()
    # Tampilkan riwayat
    _render_history_markdown()

    st.divider()
    # Pengaturan ringan
    with st.expander("⚙️ Pengaturan", expanded=False):
        st.session_state["rag_temp"] = st.slider(
            "Suhu (temperature)",
            0.0, 1.0, float(st.session_state.get("rag_temp", 0.3)), 0.1,
            help="Lebih tinggi = jawaban lebih kreatif (biasanya 0.2–0.4 untuk RAG)."
        )

    # Input chat di main page
    prompt = st.text_input("Tulis pertanyaan…", key="rag_input_main")
    if st.button("Kirim", width='content', disabled=st.session_state["rag_busy"], key="rag_btn_send_main"):
        qq = (prompt or "").strip()
        if not qq:
            st.info("Pertanyaan masih kosong.")
        else:
            st.session_state["rag_msgs"].append({"role": "user", "content": qq})
            st.session_state["rag_busy"] = True
            try:
                ans, cites = ask(
                    index=index,
                    settings=settings,
                    query=qq,
                    k=6,
                    temperature=float(st.session_state["rag_temp"]),
                )
                st.session_state["rag_msgs"].append({"role": "assistant", "content": ans, "citations": cites})
            except Exception as e:
                st.session_state["rag_msgs"].append({"role": "assistant", "content": f"(error) {e}"})
            finally:
                st.session_state["rag_busy"] = False
                st.rerun()
