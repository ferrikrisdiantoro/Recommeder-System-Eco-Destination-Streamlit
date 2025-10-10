import os
import math
import streamlit as st
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from db import Base, engine, SessionLocal
from models import User, Place, Rating, Comment, Bookmark
from utils import seed_places_if_empty, hash_password, check_password, display_price
from recommender import RecommenderService

# ========= [RAG ADDON] =========
from rag.config import RAGSettings
from rag.index import RagIndex
from rag.ui import render_chatbot_panel   # panel chatbot di main page (dipanggil saat toggle)
# ===========================================

st.set_page_config(page_title="EcoTourism Recsys (Streamlit)", page_icon="🌿", layout="wide")

# Bootstrap DB
Base.metadata.create_all(bind=engine)
with SessionLocal() as sess:
    seed_places_if_empty(sess)

# Load recommender artefak
BASE_DIR = os.getcwd()
cbf_dir = os.path.join(BASE_DIR, "models", "cbf")
cf_dir  = os.path.join(BASE_DIR, "models", "cf")
data_dir= os.path.join(BASE_DIR, "data")
try:
    RECS = RecommenderService(
        cbf_dir=os.environ.get("CBF_DIR", cbf_dir),
        cf_dir=os.environ.get("CF_DIR", cf_dir),
        fallback_data_dir=data_dir,
    )
except Exception as e:
    st.sidebar.error(f"Gagal load artefak: {e}")
    RECS = None

# ======== [RAG ADDON] init + PRE-INGEST CSV (cached) ========
DEFAULT_BOOTSTRAP_CSV = "/mnt/d/Projek/Freelancer/cl9_fw - Mentoring build System Recommender (DONE)/streamlit_recsys/models/cbf/places_clean.csv"

# DummyIndex agar app tidak crash jika GOOGLE_API_KEY kosong / RagIndex gagal terbuat
class _DummyIndex:
    def __init__(self, settings, err_msg="RAG is not available"):
        self.settings = settings
        self._err = err_msg or "RAG is not available"

    def count(self) -> int:
        return 0

    def has_source(self, name: str) -> bool:
        return False

    def ingest_paths(self, *args, **kwargs):
        raise RuntimeError(self._err)

    def retrieve(self, *args, **kwargs):
        raise RuntimeError(self._err)

@st.cache_resource(show_spinner=True)
def _init_rag_with_bootstrap():
    # 1) Settings
    try:
        settings = RAGSettings.from_env()
    except Exception:
        settings = RAGSettings(
            google_api_key=os.environ.get("GOOGLE_API_KEY", ""),
            chroma_db_path=os.environ.get("CHROMA_DB_PATH", "./chroma_db"),
            llama_cloud_api_key=os.environ.get("LLAMA_CLOUD_API_KEY", ""),
            embedding_model=os.environ.get("EMBED_MODEL", "text-embedding-004"),
            chat_model=os.environ.get("CHAT_MODEL", "gemini-1.5-pro"),
        )

    # 2) Index (tahan crash → fallback ke DummyIndex)
    idx = None
    create_err = ""
    try:
        idx = RagIndex(settings)
    except Exception as e:
        create_err = str(e)
        idx = _DummyIndex(settings, err_msg=str(e))

    # 3) Pre-ingest file CSV “di belakang”
    bootstrap_csv = os.environ.get("RAG_BOOTSTRAP_CSV", DEFAULT_BOOTSTRAP_CSV)
    bootstrap_csv = bootstrap_csv.strip('"').strip("'") if bootstrap_csv else ""
    ingest_report = {"path": bootstrap_csv, "skipped": False, "ingested": 0, "reason": ""}

    if bootstrap_csv and os.path.exists(bootstrap_csv):
        base = os.path.basename(bootstrap_csv)
        try:
            if hasattr(idx, "has_source") and idx.has_source(base):
                ingest_report["skipped"] = True
                ingest_report["reason"] = "sudah terindeks"
            else:
                try:
                    n = idx.ingest_paths([bootstrap_csv], tags="bootstrap:places_clean")
                    ingest_report["ingested"] = int(n)
                except Exception as e:
                    ingest_report["reason"] = f"gagal ingest: {e}"
        except Exception as e:
            ingest_report["reason"] = f"index not ready: {e or create_err}"
    else:
        ingest_report["reason"] = "file tidak ditemukan"

    return settings, idx, ingest_report, create_err

RAG_SETTINGS, RAG_INDEX, BOOTSTRAP_INFO, RAG_CREATE_ERR = _init_rag_with_bootstrap()
# ===========================================

# ====== Session utils ======
def get_sess_user():
    return st.session_state.get("user")

def login_user(u: User):
    st.session_state["user"] = {"id": u.id, "name": u.name, "email": u.email}

def logout_user():
    st.session_state.pop("user", None)

# Flag tampilan chatbot di main page
if "show_chatbot" not in st.session_state:
    st.session_state["show_chatbot"] = True  # langsung tampil chatbot saat start

# Sidebar (Auth + toggle tombol)
with st.sidebar:
    st.markdown("## 🌿 EcoTourism Recsys")
    u = get_sess_user()
    if not u:
        st.info("Belum login")
        with st.expander("Login"):
            email = st.text_input("Email", key="login_email")
            pw = st.text_input("Password", type="password", key="login_pw")
            if st.button("Masuk", width="stretch"):
                with SessionLocal() as sess:
                    row = sess.scalar(select(User).where(User.email == (email or "").strip().lower()))
                    if not row:
                        st.error("Email tidak ditemukan")
                    else:
                        if not check_password(pw or "", row.password_hash):
                            st.error("Password salah")
                        else:
                            login_user(row)
                            st.rerun()
        with st.expander("Register"):
            name_r = st.text_input("Nama", key="reg_name")
            email_r= st.text_input("Email", key="reg_email")
            pw_r   = st.text_input("Password", type="password", key="reg_pw")
            if st.button("Daftar", width="stretch"):
                if not name_r or not email_r or not pw_r:
                    st.error("Lengkapi form")
                else:
                    with SessionLocal() as sess:
                        exist = sess.scalar(select(User).where(User.email == email_r.strip().lower()))
                        if exist:
                            st.error("Email sudah terdaftar")
                        else:
                            from models import User as U
                            hashed = hash_password(pw_r)
                            u = U(name=name_r.strip(), email=email_r.strip().lower(), password_hash=hashed)
                            sess.add(u); sess.commit()
                            st.success("Register sukses. Silakan login.")
    else:
        st.success(f"Hi, {u['name']}")
        if st.button("Logout", width="stretch"):
            logout_user()
            st.rerun()

    st.divider()
    # Informasi bootstrap
    with st.expander("ℹ️ Status Index RAG", expanded=False):
        st.write(f"Chroma DB Path: `{RAG_SETTINGS.chroma_db_path}`")
        try:
            total_vec = RAG_INDEX.count() if hasattr(RAG_INDEX, 'count') else 0
        except Exception:
            total_vec = 0
        st.write(f"Total vektor: `{total_vec}`")
        st.write("Bootstrap CSV:")
        st.json(BOOTSTRAP_INFO)
        if RAG_CREATE_ERR:
            st.warning(f"RAG init note: {RAG_CREATE_ERR}")
        if not os.environ.get("GOOGLE_API_KEY", ""):
            st.info("Tip: set env `GOOGLE_API_KEY` untuk mengaktifkan embedding & chat Gemini.")

    # Toggle Chatbot / Rekomendasi
    if not st.session_state["show_chatbot"]:
        if st.button("🤖 Buka Chatbot AI", width="stretch"):
            st.session_state["show_chatbot"] = True
            st.rerun()
    else:
        if st.button("⬅️ Kembali ke Rekomendasi", width="stretch"):
            st.session_state["show_chatbot"] = False
            st.rerun()

st.title("🏞️ EcoTourism Recsys — Streamlit")

# ===== Main Area: Chatbot ATAU Tabs =====
if st.session_state["show_chatbot"]:
    render_chatbot_panel(settings=RAG_SETTINGS, index=RAG_INDEX)
else:
    tab_anon, tab_home, tab_places = st.tabs(["Populer", "Home (AI)", "Cari Tempat"])

    # -------------------- Tab Populer --------------------
    with tab_anon:
        st.subheader("Rekomendasi Populer")
        if RECS is None:
            st.warning("Artefak belum tersedia.")
        else:
            df = RECS.top_rated(k=12)
            for i in range(0, len(df), 3):
                cols = st.columns(3)
                for j, col in enumerate(cols):
                    if i + j < len(df):
                        r = df.iloc[i+j]
                        with col:
                            img = r.get("image","")
                            if img:
                                st.image(img, width="stretch")
                            st.markdown(f"**{r.get('place_name','')}**")
                            st.caption(f"{r.get('city','-')} • {r.get('category','-')}")
                            st.write(f"Harga: **{display_price(r.get('price',''), 0)}**")
                            st.write(f"Rating: **{float(r.get('rating',0.0)):.1f}**")

    # ------------- Tab Home (AI) — PAGINATION + Interaktif -------------
    with tab_home:
        st.subheader("Rekomendasi AI (Hybrid)")
        u = get_sess_user()
        if not u:
            st.info("Login dahulu untuk melihat rekomendasi personal.")
        elif RECS is None:
            st.warning("Artefak belum tersedia.")
        else:
            # Ambil preferensi user
            with SessionLocal() as sess:
                rows = sess.execute(select(Rating).where(Rating.user_id == u["id"])).scalars().all()
                user_ratings = {int(r.place_id): float(r.rating) for r in rows}

            if not user_ratings:
                st.warning("Belum ada preferensi. Buka tab 'Cari Tempat' lalu beri rating, atau gunakan menu Onboarding di bawah.")
            else:
                total_items = len(RECS.item_ids) if getattr(RECS, "item_ids", None) is not None else 0
                if total_items <= 1:
                    total_items = 50  # fallback

                alpha = float(os.environ.get("HYBRID_ALPHA", 0.6))
                try:
                    recdf = RECS.recommend_hybrid_for_user(user_ratings, k=total_items, alpha=alpha)
                except Exception:
                    recdf = RECS.recommend_hybrid_for_user(user_ratings, k=max(1, total_items - 1), alpha=alpha)

                if recdf is None or len(recdf) == 0:
                    st.info("Tidak ada rekomendasi yang bisa dihitung.")
                else:
                    # State pagination (jangan set untuk key widget)
                    if "home_ai_page" not in st.session_state:
                        st.session_state["home_ai_page"] = 1

                    top_bar = st.columns([2, 2, 3, 3])
                    with top_bar[0]:
                        options = [6, 9, 12, 24, 48]
                        current_ps = st.session_state.get("home_ai_page_size", 12)
                        if current_ps not in options:
                            current_ps = 12
                        page_size = st.selectbox(
                            "Jumlah per halaman",
                            options=options,
                            index=options.index(current_ps),
                            key="home_ai_page_size",
                        )

                    total = len(recdf)
                    total_pages = max(1, math.ceil(total / page_size))

                    # clamp page jika page_size berubah
                    st.session_state["home_ai_page"] = min(st.session_state["home_ai_page"], total_pages)
                    page = st.session_state["home_ai_page"]

                    with top_bar[1]:
                        st.write("")
                        prev_disabled = page <= 1
                        if st.button("◀ Sebelumnya", disabled=prev_disabled, key="home_ai_prev", width="content"):
                            if page > 1:
                                st.session_state["home_ai_page"] = page - 1
                                st.rerun()
                    with top_bar[2]:
                        st.write("")
                        next_disabled = page >= total_pages
                        if st.button("Berikutnya ▶", disabled=next_disabled, key="home_ai_next", width="content"):
                            if page < total_pages:
                                st.session_state["home_ai_page"] = page + 1
                                st.rerun()
                    with top_bar[3]:
                        st.write("")
                        st.markdown(
                            f"**Halaman {page}/{total_pages}** — Menampilkan "
                            f"**{(page-1)*page_size + 1}–{min(page*page_size, total)}** dari **{total}** item (alpha={alpha})."
                        )
                        st.caption("Anda dapat memberi rating & komentar langsung di sini.")

                    # Slice current page
                    start = (page - 1) * page_size
                    end = min(start + page_size, total)
                    page_df = recdf.iloc[start:end]

                    # Render grid (3 kolom)
                    for i in range(0, len(page_df), 3):
                        cols = st.columns(3)
                        for j, col in enumerate(cols):
                            if i + j >= len(page_df):
                                continue
                            r = page_df.iloc[i + j]
                            pid = int(r.get("id", r.get("place_id", -1)))
                            if pid < 0:
                                continue

                            # Muat row Place lengkap
                            with SessionLocal() as sess:
                                p = sess.get(Place, pid)

                            with col:
                                with st.container(border=True):
                                    if p and p.image:
                                        st.image(p.image, width="stretch")

                                    name = (p.place_name if p else r.get("place_name", "")) or ""
                                    city = (p.city if p else r.get("city", "")) or "-"
                                    cat  = (p.category if p else r.get("category", "")) or "-"
                                    price_text = display_price(p.price_str if p else r.get("price", ""),
                                                               p.price_num if p else 0.0)
                                    rating_val = float((p.rating_avg if p else r.get("rating", 0.0)) or 0.0)

                                    st.markdown(f"### {name}")
                                    st.caption(f"{city} • {cat}")
                                    st.write(f"Harga: **{price_text}**")
                                    st.write(f"Rating rata-rata: **{rating_val:.1f}**")

                                    if p and p.map_url:
                                        st.link_button("🌍 Map", p.map_url, width="content")

                                    if p and (p.place_description or "").strip():
                                        with st.expander("Detail"):
                                            st.write(p.place_description)

                                    user_logged = get_sess_user()
                                    if user_logged:
                                        # Bookmark
                                        if st.button("🔖 Bookmark", key=f"home_bm_{pid}", width="content"):
                                            with SessionLocal() as sess:
                                                if not sess.query(Bookmark).filter_by(user_id=user_logged["id"], place_id=pid).first():
                                                    sess.add(Bookmark(user_id=user_logged["id"], place_id=pid))
                                                    sess.commit()
                                            st.success("Ditambahkan ke bookmark")

                                        # Rating
                                        with SessionLocal() as sess:
                                            my_r = 0.0
                                            rr = sess.query(Rating).filter_by(user_id=user_logged["id"], place_id=pid).first()
                                            if rr:
                                                my_r = float(rr.rating)

                                        new_rating = st.slider(
                                            "Beri rating",
                                            min_value=1, max_value=5,
                                            value=int(my_r) if my_r else 5,
                                            key=f"home_rate_{pid}"
                                        )
                                        if st.button("Simpan Rating", key=f"home_save_{pid}", width="content"):
                                            with SessionLocal() as sess:
                                                rrow = sess.query(Rating).filter_by(user_id=user_logged["id"], place_id=pid).first()
                                                if rrow:
                                                    rrow.rating = float(new_rating)
                                                else:
                                                    sess.add(Rating(user_id=user_logged["id"], place_id=pid, rating=float(new_rating)))
                                                sess.commit()
                                                # Update place's avg
                                                avg, cnt = sess.query(func.avg(Rating.rating), func.count(Rating.id)).filter(Rating.place_id == pid).first()
                                                if p:
                                                    p.rating_avg = float(avg or 0.0)
                                                    sess.commit()
                                            st.success("Rating tersimpan.")
                                            st.rerun()

                                        # Comments
                                        st.markdown("**Komentar**")
                                        comment_text = st.text_input(
                                            "Tulis komentar…",
                                            key=f"home_c_{pid}",
                                            label_visibility="collapsed",
                                            placeholder="Tulis komentar…"
                                        )
                                        if st.button("Kirim Komentar", key=f"home_send_{pid}", width="content"):
                                            ct = (comment_text or "").strip()
                                            if ct:
                                                with SessionLocal() as sess:
                                                    sess.add(Comment(user_id=user_logged["id"], place_id=pid, text=ct))
                                                    sess.commit()
                                                st.success("Komentar terkirim")
                                                st.rerun()

                                        # Show recent comments
                                        with SessionLocal() as sess:
                                            cr = (
                                                sess.query(Comment, User)
                                                .join(User, Comment.user_id == User.id)
                                                .filter(Comment.place_id == pid)
                                                .order_by(Comment.created_at.desc())
                                                .all()
                                            )
                                        if cr:
                                            for c, user in cr:
                                                st.write(f"- *{user.name}* • {c.created_at}: {c.text}")
                                        else:
                                            st.caption("_Belum ada komentar_")
                                    else:
                                        st.info("Login untuk memberi rating, komentar, dan bookmark.")

        st.caption("Anda juga bisa membuka tab 'Cari Tempat' untuk memberi rating pada item lain dan memicu rekomendasi ulang.")

    # -------------------- Tab Cari Tempat --------------------
    with tab_places:
        q = st.text_input("Cari nama tempat")
        city = st.text_input("Kota")
        cat = st.text_input("Kategori")
        limit = st.slider("Limit", 6, 60, 18, 6)

        with SessionLocal() as sess:
            query = sess.query(Place)
            if q:
                query = query.filter(Place.place_name.ilike(f"%{q.strip()}%"))
            if city:
                query = query.filter(Place.city.ilike(f"%{city.strip()}%"))
            if cat:
                query = query.filter(Place.category.ilike(f"%{cat.strip()}%"))
            rows = query.limit(limit).all()

        st.write(f"Menampilkan {len(rows)} tempat.")
        for p in rows:
            with st.container(border=True):
                cols = st.columns([1,2])
                with cols[0]:
                    if p.image:
                        st.image(p.image, width="stretch")
                with cols[1]:
                    st.markdown(f"### {p.place_name}")
                    st.caption(f"{p.city or '-'} • {p.category or '-'}")
                    st.write(f"Harga: **{display_price(p.price_str, p.price_num)}**")
                    with SessionLocal() as sess:
                        avg, cnt = sess.query(func.avg(Rating.rating), func.count(Rating.id)).filter(Rating.place_id == p.id).first()
                        avg = float(avg or 0.0); cnt = int(cnt or 0)
                    st.write(f"Rating rata-rata: **{avg:.1f}** ({cnt} rating)")
                    if p.place_description:
                        st.write(p.place_description)

                    if p.map_url:
                        st.link_button("Map", p.map_url, width="content")

                    u = get_sess_user()
                    if u:
                        # Bookmark
                        if st.button(f"🔖 Bookmark {p.id}", key=f"bm_{p.id}", width="content"):
                            with SessionLocal() as sess:
                                if not sess.query(Bookmark).filter_by(user_id=u["id"], place_id=p.id).first():
                                    sess.add(Bookmark(user_id=u["id"], place_id=p.id)); sess.commit()
                            st.success("Ditambahkan ke bookmark")

                        # Rating
                        my_r = 0.0
                        with SessionLocal() as sess:
                            r = sess.query(Rating).filter_by(user_id=u["id"], place_id=p.id).first()
                            if r: my_r = float(r.rating)
                        new_rating = st.slider("Beri rating", 1, 5, int(my_r) if my_r else 5, key=f"rate_{p.id}")
                        if st.button(f"Simpan Rating {p.id}", key=f"save_{p.id}", width="content"):
                            with SessionLocal() as sess:
                                r = sess.query(Rating).filter_by(user_id=u["id"], place_id=p.id).first()
                                if r: r.rating = float(new_rating)
                                else: sess.add(Rating(user_id=u["id"], place_id=p.id, rating=float(new_rating)))
                                sess.commit()
                                avg, cnt = sess.query(func.avg(Rating.rating), func.count(Rating.id)).filter(Rating.place_id == p.id).first()
                                p.rating_avg = float(avg or 0.0)
                                sess.commit()
                            st.success("Rating tersimpan. Rekomendasi akan berubah setelah Anda memberi beberapa rating.")
                            st.rerun()

                        # Comments
                        st.markdown("**Komentar**")
                        comment_text = st.text_input(f"Tulis komentar... {p.id}", key=f"c_{p.id}")
                        if st.button(f"Kirim Komentar {p.id}", key=f"send_{p.id}", width="content"):
                            ct = (comment_text or "").strip()
                            if ct:
                                with SessionLocal() as sess:
                                    sess.add(Comment(user_id=u["id"], place_id=p.id, text=ct))
                                    sess.commit()
                                st.success("Komentar terkirim")
                                st.rerun()
                        with SessionLocal() as sess:
                            cr = sess.query(Comment, User).join(User, Comment.user_id == User.id)\
                                  .filter(Comment.place_id == p.id)\
                                  .order_by(Comment.created_at.desc()).all()
                        for c, user in cr:
                            st.write(f"- *{user.name}* • {c.created_at}: {c.text}")
                    else:
                        st.info("Login untuk memberi rating, komentar, dan bookmark.")

# -------------------- Onboarding Cepat --------------------
st.divider()
st.markdown("#### 🔧 Onboarding Cepat")
st.caption("Pilih beberapa tempat favorit untuk memberi rating 5 otomatis.")
if st.button("Mulai Onboarding (ambil sampel 18 tempat)", width="content"):
    st.session_state["onboarding"] = True
if st.session_state.get("onboarding"):
    if RECS is None:
        st.warning("Artefak belum tersedia.")
    else:
        df = RECS.sample_places(n=18)
        sel = st.multiselect("Pilih tempat favorit", options=df["id"].tolist(),
                             format_func=lambda pid: df[df["id"]==pid]["place_name"].iloc[0])
        u = get_sess_user()
        if not u:
            st.info("Login dulu untuk menyimpan preferensi.")
        else:
            if st.button("Simpan preferensi", width="content"):
                with SessionLocal() as sess:
                    for pid in sel:
                        r = sess.query(Rating).filter_by(user_id=u["id"], place_id=int(pid)).first()
                        if r: r.rating = 5.0
                        else: sess.add(Rating(user_id=u["id"], place_id=int(pid), rating=5.0))
                    sess.commit()
                st.success("Preferensi tersimpan. Buka tab Home (AI) untuk melihat rekomendasi.")
