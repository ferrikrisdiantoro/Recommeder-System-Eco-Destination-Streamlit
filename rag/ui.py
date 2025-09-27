# rag/ui.py  — FULL REPLACEMENT

import os
import streamlit as st
from typing import List, Union
from .config import RAGSettings
from .index import RagIndex
from .chain import ask

CSS = """
<style>
#rag-fab {
  position: fixed; right: 18px; bottom: 18px; z-index: 998;
}
#rag-fab a {
  display: inline-flex; align-items: center; justify-content: center;
  height: 56px; width: 56px; border-radius: 9999px; font-size: 24px;
  text-decoration: none;
  border: 1px solid #33415522; background: #1f2937; color: #fff;
  box-shadow: 0 10px 20px rgba(0,0,0,.35);
}
.rag-overlay {
  position: fixed; right: 18px; bottom: 86px; width: min(520px, 96vw);
  height: min(70vh, 720px); background: #0b1020; color: #e5e7eb;
  border: 1px solid #33415544; border-radius: 16px; z-index: 999;
  box-shadow: 0 20px 50px rgba(0,0,0,.50); padding: 12px; overflow: hidden;
}
.rag-header { display:flex; justify-content:space-between; align-items:center; padding: 4px 6px 10px 6px; }
.rag-title { display:flex; gap:8px; align-items:center; font-weight:700; }
.rag-body { position:absolute; inset:54px 8px 56px 8px; overflow:auto; }
.rag-input { position:absolute; left:8px; right:8px; bottom:8px; display:flex; gap:8px; }
.rag-input input[type="text"] {
  flex:1; border-radius: 12px; border:1px solid #33415566; padding: 10px 12px;
  background: #0f172a; color: #e5e7eb;
}
.rag-input button {
  border-radius: 12px; border:1px solid #33415566; padding: 10px 14px; background: #1f2937; color:#fff; cursor:pointer;
}
.rag-msg { margin: 8px 0; padding: 10px 12px; border-radius: 12px; max-width: 90%; }
.rag-user { background:#111827; margin-left:auto; }
.rag-bot { background:#0f172a; border:1px solid #33415544; }
.rag-cites { margin-top:6px; display:flex; gap:6px; flex-wrap: wrap; }
.rag-cite { font-size:12px; padding:2px 8px; border:1px solid #33415566; border-radius:999px; }
.rag-upload { margin: 2px 0 8px 0; }
.rag-close {
  display:inline-flex; align-items:center; justify-content:center;
  height:32px; padding:0 12px; border-radius:10px;
  background:#7f1d1d; color:#fff; border:1px solid #ef444433; text-decoration:none;
}
.rag-actions { display:flex; gap:8px; align-items:center; }
</style>
"""

def _get_query_param(name: str) -> Union[str, None]:
    """Aman di berbagai versi Streamlit."""
    try:
        qp = st.query_params.get(name)
    except Exception:
        try:
            qp = st.experimental_get_query_params().get(name)
            if isinstance(qp, list):
                qp = qp[0] if qp else None
        except Exception:
            qp = None
    if isinstance(qp, list):
        return qp[0] if qp else None
    return qp

def _set_query_param(key: str, value: str):
    """Set query param dengan fallback untuk versi lama."""
    try:
        st.query_params[key] = value
    except Exception:
        st.experimental_set_query_params(**{**st.experimental_get_query_params(), key: value})

def _build_href_with_param(key: str, value: str) -> str:
    """Bangun href yang mempertahankan param lain + set key=value."""
    try:
        current = dict(st.query_params)
    except Exception:
        try:
            current = {k: (v[0] if isinstance(v, list) else v)
                       for k, v in st.experimental_get_query_params().items()}
        except Exception:
            current = {}
    current[key] = value
    items = []
    for k, v in current.items():
        if v is None:
            continue
        items.append(f"{k}={v}")
    qs = "&".join(items)
    return f"?{qs}" if qs else "?"

def render_chatbot(settings: RAGSettings, index: RagIndex):
    st.markdown(CSS, unsafe_allow_html=True)

    # State default
    if "rag_open" not in st.session_state:
        st.session_state["rag_open"] = False
    if "rag_msgs" not in st.session_state:
        st.session_state["rag_msgs"] = [
            {"role": "assistant", "content": "Halo! Unggah dokumen lalu ajukan pertanyaan. Aku akan jawab dengan sitasi."}
        ]
    if "rag_busy" not in st.session_state:
        st.session_state["rag_busy"] = False

    # Sinkronkan dengan query param (tanpa JS)
    qp = _get_query_param("_rag")
    if qp:
        if str(qp).lower() in ("1", "true", "open", "yes"):
            st.session_state["rag_open"] = True
        elif str(qp).lower() in ("0", "false", "close", "no"):
            st.session_state["rag_open"] = False

    # FAB: pakai <a> dengan target="_self" agar tidak buka tab baru
    open_href = _build_href_with_param("_rag", "1")
    st.markdown(
        f'''
        <div id="rag-fab">
          <a href="{open_href}" target="_self" title="Buka Chatbot">💬</a>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Tombol fallback (opsional)
    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("💬 Buka Chatbot", key="open_rag_btn"):
            st.session_state["rag_open"] = True
            _set_query_param("_rag", "1")
            st.rerun()
    with c2:
        if st.button("✖ Tutup Chatbot", key="close_rag_btn"):
            st.session_state["rag_open"] = False
            _set_query_param("_rag", "0")
            st.rerun()

    if not st.session_state["rag_open"]:
        return

    # Overlay
    close_href = _build_href_with_param("_rag", "0")
    st.markdown('<div class="rag-overlay">', unsafe_allow_html=True)

    # Header
    st.markdown(
        f'''
        <div class="rag-header">
          <div class="rag-title">🌻 <span>Sunrise AI — RAG Chat</span></div>
          <div class="rag-actions">
            <a class="rag-close" href="{close_href}" target="_self">Tutup</a>
          </div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    # Upload
    with st.container():
        st.markdown('<div class="rag-upload">', unsafe_allow_html=True)
        files = st.file_uploader(
            "Unggah dokumen (PDF/TXT/DOCX—DOCX diperlakukan sebagai teks bila memungkinkan)",
            accept_multiple_files=True
        )
        tag = st.text_input("Tag (opsional, untuk filter metadata)", value="", key="rag_tag")
        if st.button("📥 Indeks Dokumen", key="rag_ingest_btn", disabled=st.session_state["rag_busy"]):
            if files:
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
            else:
                st.info("Pilih file untuk diunggah.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Body messages
    st.markdown('<div class="rag-body">', unsafe_allow_html=True)
    for m in st.session_state["rag_msgs"]:
        role = m.get("role")
        msgc = m.get("content", "")
        cites = m.get("citations", [])
        klass = "rag-bot" if role != "user" else "rag-user"
        st.markdown(f'<div class="rag-msg {klass}">{msgc}</div>', unsafe_allow_html=True)
        if cites and role != "user":
            chips = " ".join([
                f'<span class="rag-cite">{(c.get("source") or "source")}{", p."+str(c.get("page")) if c.get("page") else ""}</span>'
                for c in cites
            ])
            st.markdown(f'<div class="rag-cites">{chips}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Input area
    with st.container():
        st.markdown('<div class="rag-input">', unsafe_allow_html=True)
        q = st.text_input("Tulis pertanyaan...", key="rag_input_q")
        col_send = st.columns([5,1])[1]
        with col_send:
            if st.button("Kirim", key="rag_send_btn", disabled=st.session_state["rag_busy"]):
                qq = (q or "").strip()
                if qq:
                    st.session_state["rag_msgs"].append({"role": "user", "content": qq})
                    st.session_state["rag_busy"] = True
                    try:
                        ans, cites = ask(index=index, settings=settings, query=qq, k=6)
                        st.session_state["rag_msgs"].append(
                            {"role": "assistant", "content": ans, "citations": cites}
                        )
                    except Exception as e:
                        st.session_state["rag_msgs"].append({"role": "assistant", "content": f"(error) {e}"})
                    finally:
                        st.session_state["rag_busy"] = False
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
