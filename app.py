import os
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

    # 2) Index
    idx = RagIndex(settings)

    # 3) Pre-ingest file CSV “di belakang” (sekali saja berkat cache_resource)
    bootstrap_csv = os.environ.get("RAG_BOOTSTRAP_CSV", DEFAULT_BOOTSTRAP_CSV)
    bootstrap_csv = bootstrap_csv.strip('"').strip("'") if bootstrap_csv else ""
    ingest_report = {"path": bootstrap_csv, "skipped": False, "ingested": 0, "reason": ""}

    if bootstrap_csv and os.path.exists(bootstrap_csv):
        base = os.path.basename(bootstrap_csv)
        if idx.has_source(base):
            ingest_report["skipped"] = True
            ingest_report["reason"] = "sudah terindeks"
        else:
            try:
                n = idx.ingest_paths([bootstrap_csv], tags="bootstrap:places_clean")
                ingest_report["ingested"] = int(n)
            except Exception as e:
                ingest_report["reason"] = f"gagal ingest: {e}"
    else:
        ingest_report["reason"] = "file tidak ditemukan"

    return settings, idx, ingest_report

RAG_SETTINGS, RAG_INDEX, BOOTSTRAP_INFO = _init_rag_with_bootstrap()
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
            if st.button("Masuk", use_container_width=True):
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
            if st.button("Daftar", use_container_width=True):
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
        if st.button("Logout", use_container_width=True):
            logout_user()
            st.rerun()

    st.divider()
    # Informasi bootstrap (sekali tampil; berguna saat debugging)
    with st.expander("ℹ️ Status Index RAG", expanded=False):
        st.write(f"Chroma DB Path: `{RAG_SETTINGS.chroma_db_path}`")
        st.write(f"Total vektor: `{RAG_INDEX.count()}`")
        st.write("Bootstrap CSV:")
        st.json(BOOTSTRAP_INFO)

    # Toggle Chatbot / Rekomendasi
    if not st.session_state["show_chatbot"]:
        if st.button("🤖 Buka Chatbot AI", use_container_width=True):
            st.session_state["show_chatbot"] = True
            st.rerun()
    else:
        if st.button("⬅️ Kembali ke Rekomendasi", use_container_width=True):
            st.session_state["show_chatbot"] = False
            st.rerun()

st.title("🏞️ EcoTourism Recsys — Streamlit")

# ===== Main Area: Chatbot ATAU Tabs =====
if st.session_state["show_chatbot"]:
    render_chatbot_panel(settings=RAG_SETTINGS, index=RAG_INDEX)
else:
    # === Tabs Rekomendasi (seperti sebelumnya) ===
    from models import User as U
    tab_anon, tab_home, tab_places = st.tabs(["Populer", "Home (AI)", "Cari Tempat"])

    # Tab Populer
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
                            st.image(r.get("image",""), use_container_width=True)
                            st.markdown(f"**{r.get('place_name','')}**")
                            st.caption(f"{r.get('city','-')} • {r.get('category','-')}")
                            st.write(f"Harga: **{display_price(r.get('price',''), 0)}**")
                            st.write(f"Rating: **{float(r.get('rating',0.0)):.1f}**")

    # Tab Home (AI)
    with tab_home:
        st.subheader("Rekomendasi AI (Hybrid)")
        u = get_sess_user()
        if not u:
            st.info("Login dahulu untuk melihat rekomendasi personal.")
        elif RECS is None:
            st.warning("Artefak belum tersedia.")
        else:
            from models import Rating
            with SessionLocal() as sess:
                rows = sess.execute(select(Rating).where(Rating.user_id == u["id"])).scalars().all()
                user_ratings = {int(r.place_id): float(r.rating) for r in rows}
            if not user_ratings:
                st.warning("Belum ada preferensi. Buka tab 'Cari Tempat' lalu beri rating, atau gunakan menu Onboarding di bawah.")
            else:
                alpha = float(os.environ.get("HYBRID_ALPHA", 0.6))
                recdf = RECS.recommend_hybrid_for_user(user_ratings, k=12, alpha=alpha)
                for i in range(0, len(recdf), 3):
                    cols = st.columns(3)
                    for j, col in enumerate(cols):
                        if i + j < len(recdf):
                            r = recdf.iloc[i+j]
                            with col:
                                st.image(r.get("image",""), use_container_width=True)
                                st.markdown(f"**{r.get('place_name','')}**")
                                st.caption(f"{r.get('city','-')} • {r.get('category','-')}")
                                st.write(f"Harga: **{display_price(r.get('price',''), 0)}**")
                                st.write(f"Rating: **{float(r.get('rating',0.0)):.1f}**")
        st.caption("Klik tab 'Cari Tempat' untuk memberi rating dan memicu rekomendasi ulang.")

    # Tab Cari Tempat
    with tab_places:
        from models import Place, Rating, Comment, Bookmark, User as U
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
                        st.image(p.image, use_container_width=True)
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
                        st.link_button("Map", p.map_url)

                    u = get_sess_user()
                    if u:
                        if st.button(f"🔖 Bookmark {p.id}", key=f"bm_{p.id}"):
                            with SessionLocal() as sess:
                                if not sess.query(Bookmark).filter_by(user_id=u["id"], place_id=p.id).first():
                                    sess.add(Bookmark(user_id=u["id"], place_id=p.id)); sess.commit()
                            st.success("Ditambahkan ke bookmark")

                        my_r = 0.0
                        with SessionLocal() as sess:
                            r = sess.query(Rating).filter_by(user_id=u["id"], place_id=p.id).first()
                            if r: my_r = float(r.rating)
                        new_rating = st.slider("Beri rating", 1, 5, int(my_r) if my_r else 5, key=f"rate_{p.id}")
                        if st.button(f"Simpan Rating {p.id}", key=f"save_{p.id}"):
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

                        st.markdown("**Komentar**")
                        comment_text = st.text_input(f"Tulis komentar... {p.id}", key=f"c_{p.id}")
                        if st.button(f"Kirim Komentar {p.id}", key=f"send_{p.id}"):
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

    st.divider()
    st.markdown("#### 🔧 Onboarding Cepat")
    st.caption("Pilih beberapa tempat favorit untuk memberi rating 5 otomatis.")
    if st.button("Mulai Onboarding (ambil sampel 18 tempat)"):
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
                if st.button("Simpan preferensi"):
                    with SessionLocal() as sess:
                        for pid in sel:
                            from models import Rating
                            r = sess.query(Rating).filter_by(user_id=u["id"], place_id=int(pid)).first()
                            if r: r.rating = 5.0
                            else: sess.add(Rating(user_id=u["id"], place_id=int(pid), rating=5.0))
                        sess.commit()
                    st.success("Preferensi tersimpan. Buka tab Home (AI) untuk melihat rekomendasi.")
