"""Benchmark KNF-01: membuktikan klaim "berjalan pada perangkat standar tanpa GPU".

Laporan mengklaim sistem ringan dan layak dipakai pada waktu konsultasi yang singkat, namun
klaim itu tidak pernah diukur. Skrip ini mengukurnya: waktu setiap tahap satu siklus
rekomendasi, jejak memori, ukuran artefak, dan spesifikasi perangkat yang dipakai.

Tahap yang diukur (masing-masing diulang N_REPEAT kali, dilaporkan median dan p95):
  1. Muat artefak     : bundel produksi (keluarga mengikuti aplikasi), pengklasifikasi kondisi, indeks ChromaDB, model embedding
  2. Rekayasa fitur   : jendela logbook -> vektor fitur
  3. Prediksi         : regresi RF + pengklasifikasi kondisi + interval antar-pohon
  4. Retrieval        : embedding kueri + pencarian MMR pada korpus terindeks
  5. Generasi (LLM)   : panggilan Gemini (opsional; butuh GOOGLE_API_KEY dan kuota)

Pemuatan artefak hanya terjadi sekali saat aplikasi dinyalakan (di-cache Streamlit), sehingga
yang menentukan pengalaman dokter adalah tahap 2-5, bukan tahap 1.

Keluaran: results/benchmark/deployability.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import json
import os
import pickle
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

N_REPEAT = 30
HORIZON = 6
OUT = ROOT / "results/benchmark"


def timed(fn, n: int = N_REPEAT) -> dict:
    """Jalankan fn() n kali; kembalikan statistik waktu dalam milidetik."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(times)
    return {
        "median_ms": round(float(np.median(a)), 1),
        "p95_ms": round(float(np.percentile(a, 95)), 1),
        "min_ms": round(float(a.min()), 1),
        "n_ulangan": n,
    }


def device_info() -> dict:
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "prosesor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "gpu_dipakai": False,
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
        info["jumlah_core_logis"] = psutil.cpu_count(logical=True)
    except ImportError:
        info["ram_total_gb"] = None
        info["jumlah_core_logis"] = os.cpu_count()
    try:
        import torch
        info["gpu_tersedia_di_mesin"] = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        info["gpu_tersedia_di_mesin"] = None
    return info


def rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1024**2, 1)
    except ImportError:
        return None


def artifact_sizes() -> dict:
    def size_mb(p: Path) -> float | None:
        if p.is_file():
            return round(p.stat().st_size / 1024**2, 1)
        if p.is_dir():
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            return round(total / 1024**2, 1)
        return None

    # Nama artefak dan jumlah potongan DIBACA dari keadaan sebenarnya, tidak lagi
    # ditulis tetap. Versi sebelumnya menuliskan "rf_inference_bundle_h6.pkl" dan
    # "chroma_db (2.585 chunk)" secara tetap, sehingga keluarannya tetap melaporkan
    # artefak dan korpus lama meskipun produksi sudah berpindah.
    nama_bundel, _ = _bundel_produksi()
    n_chunk = _jumlah_chunk()
    return {
        Path(nama_bundel).name: size_mb(ROOT / nama_bundel),
        f"chroma_db ({n_chunk} chunk)": size_mb(ROOT / "models/chroma_db"),
    }


def _bundel_produksi():
    """Pilih bundel dengan urutan keluarga yang SAMA dengan aplikasi.

    app/streamlit_app.py:110 memakai KELUARGA_BUNDLE = [GBM, RF] dan mengambil
    keluarga pertama yang berkasnya ada. Benchmark ini WAJIB mengikuti urutan itu;
    bila tidak, ia mengukur model yang berbeda dari yang dijalankan pengguna —
    padahal angkanya dipakai untuk mengklaim batasan desain terpenuhi.
    """
    for keluarga in ("gbm", "rf"):
        p = ROOT / f"models/{keluarga}_inference_bundle_h{HORIZON}.pkl"
        if p.exists():
            return f"models/{keluarga}_inference_bundle_h{HORIZON}.pkl", keluarga
    raise FileNotFoundError("tidak ada bundel inference untuk horizon ini")


def _jumlah_chunk() -> int:
    import sqlite3
    db = ROOT / "models/chroma_db/chroma.sqlite3"
    if not db.exists():
        return 0
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return int(con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0])
    finally:
        con.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mc = cfg["model"]

    from src.data.preprocessor import DataPreprocessor
    from src.rag.retriever import MMRRetriever

    mem_awal = rss_mb()
    result = {
        "catatan": "Seluruh pengukuran pada CPU; GPU tidak dipakai (KNF-01).",
        "perangkat": device_info(),
        "ukuran_artefak_mb": artifact_sizes(),
        "tahap": {},
    }

    # ── 1. Muat artefak (sekali saat aplikasi dinyalakan) ──────────────────
    jalur_bundel, keluarga = _bundel_produksi()
    result["keluarga_model_diukur"] = keluarga
    t0 = time.perf_counter()
    with open(ROOT / jalur_bundel, "rb") as f:
        bundle = pickle.load(f)
    t_rf = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    jalur_clf = ROOT / f"models/{keluarga}_condition_classifier_h{HORIZON}.pkl"
    if not jalur_clf.exists():  # keluarga ini belum punya pengklasifikasi
        jalur_clf = ROOT / f"models/rf_condition_classifier_h{HORIZON}.pkl"
    with open(jalur_clf, "rb") as f:
        clf = pickle.load(f)
    t_clf = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    retriever = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                             embed_provider="sentence-transformers")
    t_ret_load = (time.perf_counter() - t0) * 1000

    result["tahap"]["1_muat_artefak_sekali"] = {
        "model_regresi_ms": round(t_rf, 1),
        "pengklasifikasi_kondisi_ms": round(t_clf, 1),
        "retriever_dan_embedding_ms": round(t_ret_load, 1),
        "total_ms": round(t_rf + t_clf + t_ret_load, 1),
        "keterangan": "Hanya sekali saat aplikasi start (di-cache); tidak dirasakan per konsultasi.",
    }

    # ── Siapkan satu jendela logbook nyata ────────────────────────────────
    # Dataset produksi mengikuti config.data.unified_dataset. Berkas lama
    # ohio_t1dm_merged.csv TIDAK dipakai lagi: ia tidak memuat kolom
    # glucose_source yang kini diwajibkan kontrak data, sehingga skrip ini
    # gagal pada tahap rekayasa fitur bila masih membacanya.
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm.csv", parse_dates=["timestamp"])
    df = df[df.glucose_source == "CGM"].copy()
    df = df[df.patient_id == sorted(df.patient_id.unique())[-1]].tail(400).copy()
    pre = DataPreprocessor(cfg)

    # ── 2. Rekayasa fitur ─────────────────────────────────────────────────
    def do_features():
        d = pre.handle_missing_values(df.copy())
        return pre.engineer_features(d, **mc["feature_engineering"])

    result["tahap"]["2_rekayasa_fitur"] = timed(do_features, n=10)
    feat = do_features()
    # Daftar fitur dan panjang jendela diambil dari BUNDEL, bukan dari config.
    # Bundel produksi h6 dilatih dengan tujuh fitur, sedangkan
    # config.model.engineered_features mendaftar sembilan karena memuat dua
    # fitur yang khusus dibutuhkan skenario SMBG (glucose_rate dan
    # time_since_prev_glucose). Inferensi produksi memakai bundle["features"],
    # sehingga benchmark harus memakainya juga agar yang terukur adalah jalur
    # yang benar-benar dijalankan.
    window = feat.tail(int(bundle.get("sequence_length", mc["sequence_length"])))
    X = window[bundle["features"]].to_numpy(float)
    Xs = bundle["scaler"].transform(X).reshape(1, -1)

    # Pengklasifikasi kondisi adalah ARTEFAK TERPISAH dengan skema fiturnya
    # sendiri (sembilan fitur) dan scaler-nya sendiri, sedangkan bundel regresi
    # memakai tujuh. Backend memperlakukan keduanya terpisah; benchmark ini
    # mengikutinya. Memakai Xs milik regresi untuk pengklasifikasi membuat
    # skrip gagal, dan sebelum ini itulah yang terjadi.
    X_clf = window[clf["features"]].to_numpy(float)
    Xs_clf = clf["scaler"].transform(X_clf).reshape(1, -1)

    # ── 3. Prediksi (regresi + kondisi + interval antar-pohon) ────────────
    # Sumber sigma mengikuti bundel, bukan diandaikan. RF memakai sebaran antar-pohon
    # (estimators_), sedangkan HistGradientBoostingRegressor TIDAK punya estimators_
    # sehingga memakai sebaran kuantil yang disimpan bundel sebagai std_models.
    # Versi sebelumnya memanggil estimators_ tanpa syarat, sehingga benchmark ini
    # HANYA dapat berjalan pada Random Forest dan diam-diam mengukur model yang
    # bukan produksi.
    std_models = bundle.get("std_models")
    lebar_gauss = 3.92  # (q0,975 - q0,025) pada normal baku; sama dengan gbm_model.py

    def do_predict():
        pred = bundle["model"].predict(Xs)[0] + float(window["glucose"].iloc[-1])
        cond = clf["model"].predict(Xs_clf)[0]
        if std_models:
            sigma = (std_models["high"].predict(Xs)[0]
                     - std_models["low"].predict(Xs)[0]) / lebar_gauss
        else:
            sigma = np.std([t.predict(Xs)[0] for t in bundle["model"].estimators_])
        return pred, cond, sigma

    result["tahap"]["3_prediksi"] = timed(do_predict)
    pred, cond, sigma = do_predict()
    result["tahap"]["3_prediksi"]["keterangan"] = (
        "Mencakup regresi, pengklasifikasi kondisi, dan interval ketidakpastian "
        f"({bundle.get('std_method', 'sebaran antar-pohon')})."
    )

    # 4. Retrieval -- mode mengikuti rag.retrieval_mode pada config.yaml
    query = ("Kadar glukosa darah diprediksi 165 mg/dL (30 menit ke depan). "
             "Hiperglikemia, gula darah tinggi di atas 180 mg/dL. Penyebab, gejala, dan penanganan.")

    def do_retrieve():
        return retriever.retrieve(query, top_k=cfg["rag"]["top_k_retrieval"])

    result["tahap"]["4_retrieval"] = timed(do_retrieve, n=20)
    result["tahap"]["4_retrieval"]["keterangan"] = f"Penelusuran mode {retriever.retrieval_mode} pada {_jumlah_chunk()} chunk."

    # ── 5. Generasi LLM (opsional; butuh API key + kuota) ─────────────────
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    llm = {"dijalankan": False, "alasan": "GOOGLE_API_KEY tidak tersedia"}
    if os.getenv("GOOGLE_API_KEY"):
        try:
            from src.rag.pipeline import RAGPipeline
            pipe = RAGPipeline()
            state = {"current_glucose": 150.0, "insulin_on_board": 1.0, "carbs_on_board": 20.0,
                     "activity_level": 0, "stress_level": 5, "predicted_condition": "hyperglycemia"}
            t0 = time.perf_counter()
            pipe.answer(patient_state=state, prediction=float(pred))
            llm = {"dijalankan": True, "sekali_panggil_ms": round((time.perf_counter() - t0) * 1000, 1),
                   "keterangan": "Satu panggilan Gemini; kuota free-tier 20 request/hari."}
        except Exception as exc:  # noqa: BLE001
            llm = {"dijalankan": False, "alasan": str(exc)[:150]}
    result["tahap"]["5_generasi_llm"] = llm

    # ── Total satu siklus tanpa LLM (bagian yang sepenuhnya lokal) ────────
    lokal = (result["tahap"]["2_rekayasa_fitur"]["median_ms"]
             + result["tahap"]["3_prediksi"]["median_ms"]
             + result["tahap"]["4_retrieval"]["median_ms"])
    result["total_siklus_lokal_ms"] = round(lokal, 1)
    result["memori_rss_mb"] = {"sebelum_muat": mem_awal, "setelah_muat_semua": rss_mb()}

    (OUT / "deployability.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== Perangkat ===")
    for k, v in result["perangkat"].items():
        print(f"  {k}: {v}")
    print("\n=== Waktu per tahap (median) ===")
    print(f"  1. Muat artefak (sekali) : {result['tahap']['1_muat_artefak_sekali']['total_ms']:.0f} ms")
    print(f"  2. Rekayasa fitur        : {result['tahap']['2_rekayasa_fitur']['median_ms']:.1f} ms")
    print(f"  3. Prediksi              : {result['tahap']['3_prediksi']['median_ms']:.1f} ms")
    print(f"  4. Retrieval             : {result['tahap']['4_retrieval']['median_ms']:.1f} ms")
    if llm["dijalankan"]:
        print(f"  5. Generasi LLM          : {llm['sekali_panggil_ms']:.0f} ms")
    print(f"\n  Total siklus lokal (tanpa LLM): {lokal:.1f} ms")
    print(f"  Memori RSS setelah semua dimuat: {result['memori_rss_mb']['setelah_muat_semua']} MB")
    print(f"\nDisimpan ke {OUT / 'deployability.json'}")


if __name__ == "__main__":
    main()
