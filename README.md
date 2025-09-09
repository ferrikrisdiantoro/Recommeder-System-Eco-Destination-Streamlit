# 🌿 EcoTourism Recommender System (Streamlit)

Aplikasi **Hybrid Recommender System** untuk ekowisata Indonesia.  
Dibangun dengan **Streamlit + SQLAlchemy + PostgreSQL/SQLite** serta memanfaatkan artefak model **Content-Based Filtering (CBF)** dan **Collaborative Filtering (CF)**.

---

## ✨ Fitur Utama

- **Autentikasi User**  
  - Register & Login (bcrypt, simpan session di `st.session_state`)  
  - Logout  

- **Manajemen Tempat Wisata**
  - Tampil daftar tempat, filter by nama/kota/kategori  
  - Detail tempat dengan deskripsi, alamat, galeri, peta  

- **Interaksi User**
  - Rating (1–5, otomatis update rata-rata)  
  - Komentar publik (mirip Google Review)  
  - Bookmark tempat favorit  

- **Rekomendasi**
  - **Populer**: rekomendasi berdasarkan rating tertinggi (anonim)  
  - **Hybrid**: rekomendasi personal gabungan CF + CBF  
  - **Onboarding cepat**: pilih beberapa tempat → auto rating 5  

- **Database**
  - Default: SQLite `eco.db`  
  - Bisa ganti ke PostgreSQL dengan `DATABASE_URL`  
  - Auto seeding dari CSV (`models/place_clean.csv`, `models/cbf/places_clean.csv`, atau `data/eco_place.csv`)  

---

## 📂 Struktur Project

```
streamlit_recsys/
│
├── app.py              # Entry point Streamlit
├── db.py               # Config DB & engine
├── models.py           # Skema tabel (User, Place, Rating, Comment, Bookmark)
├── recommender.py      # Service rekomendasi (CBF + CF + Hybrid)
├── utils.py            # Helper auth, seeding, price formatting
│
├── models/             # Folder artefak CBF/CF
│   ├── cbf/
│   │   ├── cbf_item_matrix.npz
│   │   ├── cbf_artifacts.joblib
│   │   └── places_clean.csv   (opsional)
│   └── cf/
│       ├── cf_item_sim.npy
│       └── cf_artifacts.joblib
│
├── data/
│   └── eco_place.csv   # fallback data
│
├── requirements.txt    # Dependensi Python
└── README.md           # Dokumentasi
```

---

## ⚙️ Instalasi & Menjalankan

### 1. Clone / Extract
```bash
cd streamlit_recsys
```

### 2. Buat virtualenv & install requirements
```bash
python -m venv .venv
source .venv/bin/activate     # Linux/Mac
.venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 3. (Opsional) Konfigurasi Database PostgreSQL
```bash
# Linux/Mac
export DATABASE_URL="postgresql+psycopg2://eco_user:12345@127.0.0.1:5432/eco"

# Windows PowerShell
$env:DATABASE_URL = "postgresql+psycopg2://eco_user:12345@127.0.0.1:5432/eco"
```

> Default tanpa konfigurasi → pakai `sqlite:///eco.db`

### 4. Jalankan Aplikasi
```bash
streamlit run app.py --server.port 8501
```

Akses di browser:  
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧪 Testing Cepat

1. Register user baru  
2. Login → buka tab **Cari Tempat** → beri rating / komentar  
3. Cek tab **Home (AI)** → rekomendasi akan muncul & dinamis  
4. Logout → tab **Populer** tetap bisa dipakai  

---

## 🚀 Catatan Teknis

- **Password** disimpan dengan `bcrypt.hashpw()` (bytes → kolom `LargeBinary` / `BYTEA`)  
- **Artefak model** harus tersedia di `models/cbf` dan `models/cf`  
- **Perubahan kode penting**: semua `st.image(..., use_column_width=True)` sudah diganti ke `use_container_width=True` (menghilangkan warning deprecation)  
- Untuk **migrasi DB** jangka panjang, disarankan pakai **Alembic**  

---

## 👨‍💻 Developer Notes

- Jika login error `memoryview` → pastikan sudah pakai versi `utils.py` terbaru yang robust terhadap `bytes/memoryview/str`.  
- Jika artefak CBF/CF tidak ditemukan, aplikasi tetap jalan dengan fallback `eco_place.csv`.  
- Untuk deploy ke server (Heroku/Render/Cloud Run), cukup set env `DATABASE_URL` + `HYBRID_ALPHA` (opsional).  

---

## 📜 Lisensi
Demo project untuk mentoring & pembelajaran. Bebas dikembangkan lebih lanjut sesuai kebutuhan client.
