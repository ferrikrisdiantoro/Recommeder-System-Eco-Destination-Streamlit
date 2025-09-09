import os, numpy as np, pandas as pd, joblib
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import warnings

try:
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:
    class InconsistentVersionWarning(Warning):
        pass


class RecommenderService:
    """
    Loader artefak CBF/CF + fungsi rekomendasi.
    Lebih robust:
    - Menerima kolom id, place_id, atau Unnamed: 0. Jika tidak ada, gunakan place_id_order dari artefak.
    - Casting id & rating aman (filter baris di DataFrame, bukan dropna di Series).
    - Fallback kolom price/image, dan default kolom wajib (place_name, city, category, price, image).
    """

    def __init__(self, cbf_dir: str, cf_dir: str, fallback_data_dir: str | None = None):
        self.cbf_dir = Path(cbf_dir)
        self.cf_dir = Path(cf_dir)
        self.fallback_data_dir = Path(fallback_data_dir) if fallback_data_dir else None

        self.places_df: pd.DataFrame | None = None
        self.place_id_order: list[int] = []
        self.X = None

        self.item_sim = None
        self.item_ids: list[int] = []
        self.item_to_col: dict[int, int] = {}

        self._load_all()

    # ---------- Public ----------
    def top_rated(self, k=20):
        df = self.places_df.copy()
        # Beberapa dataset pakai rating_avg, beberapa rating → pilih yang ada
        if "rating" not in df.columns and "rating_avg" in df.columns:
            df["rating"] = pd.to_numeric(df["rating_avg"], errors="coerce").fillna(0.0)
        return df.sort_values("rating", ascending=False).head(k)

    def sample_places(self, n=20, seed=42):
        df = self.places_df.sample(n=min(n, len(self.places_df)), random_state=seed)
        keep = ["id", "place_name", "city", "category", "price", "rating", "image"]
        for c in keep:
            if c not in df.columns:
                df[c] = "" if c not in ["rating"] else 0.0
        return df[keep]

    def recommend_hybrid_for_user(self, user_ratings: dict, k=20, alpha=0.6):
        # --- CF score ---
        s_cf = np.zeros(len(self.item_ids), dtype=float)
        if self.item_sim is not None and len(self.item_ids) > 0:
            v = np.zeros(len(self.item_ids), dtype=float)
            for pid, r in (user_ratings or {}).items():
                j = self.item_to_col.get(int(pid))
                if j is not None:
                    v[j] = float(r)
            s_cf = self.item_sim.dot(v)

        # --- CBF score ---
        s_cbf = np.zeros(len(self.place_id_order), dtype=float)
        if self.X is not None and len(self.place_id_order) == self.X.shape[0]:
            pid_to_row = {pid: i for i, pid in enumerate(self.place_id_order)}
            for pid, r in (user_ratings or {}).items():
                i = pid_to_row.get(int(pid))
                if i is not None:
                    sims = cosine_similarity(self.X[i], self.X).ravel()
                    s_cbf += sims * float(r)

        # --- Mask seen ---
        seen_cols = set()
        for pid in (user_ratings or {}):
            j = self.item_to_col.get(int(pid))
            if j is not None:
                seen_cols.add(j)

        # Align CBF → CF order
        pid_to_row = {pid: i for i, pid in enumerate(self.place_id_order)}
        s_cbf_aligned = np.zeros_like(s_cf)
        for idx, pid in enumerate(self.item_ids):
            i = pid_to_row.get(pid)
            if i is not None:
                s_cbf_aligned[idx] = s_cbf[i]

        # Normalize & blend
        def norm01(x):
            mn, mx = np.nanmin(x), np.nanmax(x)
            if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-9:
                return np.zeros_like(x)
            return (x - mn) / (mx - mn + 1e-9)

        s = alpha * norm01(s_cf) + (1 - alpha) * norm01(s_cbf_aligned)
        for j in seen_cols:
            s[j] = -np.inf

        # Top-K
        k = min(k, len(s) - 1) if len(s) > 1 else 1
        top_idx = np.argpartition(-s, kth=k - 1)[:k]
        top_idx = top_idx[np.argsort(-s[top_idx])]
        top_pids = [self.item_ids[j] for j in top_idx]

        # Kembalikan metadata siap UI
        cols = ["place_name", "city", "category", "price", "rating", "image"]
        for c in cols:
            if c not in self.places_df.columns:
                self.places_df[c] = "" if c != "rating" else 0.0
        meta = (
            self.places_df.set_index("id")
            .reindex(top_pids)[cols]
            .reset_index()
            .rename(columns={"index": "place_id"})
        )
        meta["hybrid_score"] = np.array(s[top_idx]).round(4)
        return meta

    # ---------- Loaders ----------
    def _load_all(self):
        self._load_cbf()
        self._load_cf()
        self._sanity_align_ids()

    def _load_cbf(self):
        mat_p = self.cbf_dir / "cbf_item_matrix.npz"
        art_p = self.cbf_dir / "cbf_artifacts.joblib"
        if not (mat_p.exists() and art_p.exists()):
            raise FileNotFoundError(f"Artefak CBF tidak ditemukan di {self.cbf_dir}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            obj = joblib.load(art_p)

        # Urutan ID item sesuai matrix CBF
        self.place_id_order = list(obj.get("place_id_order", []))
        self.X = load_npz(mat_p)

        # ----- Metadata places -----
        places_csv = self.cbf_dir / "places_clean.csv"
        if places_csv.exists():
            df = pd.read_csv(places_csv)
        else:
            if not self.fallback_data_dir:
                raise FileNotFoundError("places_clean.csv tidak ada dan fallback_data_dir tidak diset")
            eco_p = self.fallback_data_dir / "eco_place.csv"
            if not eco_p.exists():
                raise FileNotFoundError(f"{eco_p} tidak ditemukan")
            raw = pd.read_csv(eco_p)
            raw = raw.rename(columns={
                "place_id": "id",
                "place_img": "image",
                "description_location": "address",
                "gallery_photo_img1": "gallery1",
                "gallery_photo_img2": "gallery2",
                "gallery_photo_img3": "gallery3",
                "place_map": "map_url",
            })
            keep = ["id", "place_name", "place_description", "category", "city",
                    "address", "price", "rating", "image", "gallery1", "gallery2", "gallery3", "map_url"]
            for c in keep:
                if c not in raw.columns:
                    raw[c] = "" if c != "rating" else 0.0
            df = raw[keep].copy()

        # ---- Pastikan kolom ID ada & valid ----
        cols_have = set(df.columns)
        if "id" not in cols_have:
            if "place_id" in cols_have:
                df = df.rename(columns={"place_id": "id"})
            elif "Unnamed: 0" in cols_have:
                df = df.rename(columns={"Unnamed: 0": "id"})
            elif self.place_id_order and len(self.place_id_order) == len(df):
                # gunakan urutan ID dari artefak
                df = df.copy()
                df["id"] = list(self.place_id_order)
            else:
                raise KeyError(
                    "Kolom id tidak ada pada places, dan tidak bisa diinfer dari place_id_order. "
                    "Pastikan ada kolom 'id' / 'place_id' / 'Unnamed: 0' atau artefak memuat place_id_order yang cocok."
                )

        # Casting aman: filter baris NA di DataFrame
        df["id"] = pd.to_numeric(df["id"], errors="coerce")
        df = df.dropna(subset=["id"]).copy()
        df["id"] = df["id"].astype(int)

        # rating
        if "rating" not in df.columns and "rating_avg" in df.columns:
            df["rating"] = pd.to_numeric(df["rating_avg"], errors="coerce").fillna(0.0)
        else:
            df["rating"] = pd.to_numeric(df.get("rating", 0.0), errors="coerce").fillna(0.0)

        # Kolom standar untuk UI
        if "price" not in df.columns and "price_str" in df.columns:
            df = df.rename(columns={"price_str": "price"})
        if "image" not in df.columns and "place_img" in df.columns:
            df = df.rename(columns={"place_img": "image"})
        for c in ["place_name", "category", "city", "price", "image"]:
            if c not in df.columns:
                df[c] = ""

        self.places_df = df

        # Sinkronisasi panjang X vs metadata (jika perlu)
        if self.X.shape[0] != len(self.place_id_order):
            # kalau artefak tidak simetris, minimal pastikan mapping tetap konsisten
            self.place_id_order = self.place_id_order[: self.X.shape[0]]

    def _load_cf(self):
        sim_p = self.cf_dir / "cf_item_sim.npy"
        art_p = self.cf_dir / "cf_artifacts.joblib"
        if not (sim_p.exists() and art_p.exists()):
            raise FileNotFoundError(f"Artefak CF tidak ditemukan di {self.cf_dir}")

        self.item_sim = np.load(sim_p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            obj = joblib.load(art_p)
        self.item_ids = list(obj.get("item_ids", []))
        self.item_to_col = dict(obj.get("item_to_col", {}))

    def _sanity_align_ids(self):
        if self.places_df is None:
            return
        valid_ids = set(self.places_df["id"].tolist())
        if not self.item_ids:
            return
        keep_mask = np.array([pid in valid_ids for pid in self.item_ids], dtype=bool)
        if keep_mask.size and (not keep_mask.all()):
            idx = np.where(keep_mask)[0]
            self.item_sim = self.item_sim[np.ix_(idx, idx)]
            self.item_ids = [self.item_ids[i] for i in idx]
            self.item_to_col = {pid: j for j, pid in enumerate(self.item_ids)}
