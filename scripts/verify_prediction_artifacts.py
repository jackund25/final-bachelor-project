#!/usr/bin/env python3
"""Verifikasi artefak prediksi: apakah bundle tersimpan MEREPRODUKSI angka laporan.

Skrip ini merealisasikan "Verifikasi Fungsional" (Bab V). Berbeda dari sekadar mengecek
keberadaan berkas, skrip ini memuat bundle inferensi yang benar-benar dipakai backend,
menjalankannya ulang pada pembagian uji yang sama, lalu membandingkan metriknya
terhadap metrik yang tercatat di sebelah bundle DAN terhadap angka yang tertulis pada
naskah. Perbedaan sekecil apa pun ditandai GAGAL.

Jalankan pada environment proyek (conda: diabetes-ta) agar versi pustaka sesuai:
bundle di-pickle dengan scikit-learn 1.3.0 dan tidak dapat dibuka versi yang lebih baru.

    python scripts/verify_prediction_artifacts.py

DIPERBARUI 24 Agustus 2026 mengikuti penyatuan parser OhioT1DM. Yang berubah:

* sumber data menjadi satu berkas gabungan ``data/raw/ohio_t1dm.csv`` yang disaring
  menurut ``glucose_source``;
* pembagian uji memakai ``dataset_split`` resmi, bukan menyisihkan dua pasien terakhir;
* jendela dibentuk dari WAKTU NYATA lewat ``src.models.persiapan_data``, modul yang
  sama yang dipakai pelatihan — sehingga verifikasi ini tidak dapat lolos karena
  kebetulan memakai perlakuan yang berbeda;
* penamaan bundle mengikuti modalitas dan horizon dalam menit.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.persiapan_data import (  # noqa: E402
    bentuk_jendela_bagian,
    muat_dataset_modalitas,
)
from src.utils.metrics import calculate_all_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

# ANGKA YANG WAJIB DAPAT DIREPRODUKSI DARI ARTEFAK PRODUKSI.
#
# Nilai di bawah adalah metrik hold-out bundel produksi pada pembagian resmi OhioT1DM
# (temporal dalam-pasien), sebagaimana tersimpan di models/*_metrics.json. Baris
# finger-stick juga yang dikutip Bab VI (Tabel VI.5); dua baris CGM tidak dikutip
# naskah karena Tabel VI.3 membandingkan tiga model pada protokol bersama
# (results/eval_prediksi/summary_all_horizons.csv), bukan bundel produksi. Bila
# artefak dilatih ulang, angka di sini dan berkas metrik harus diperbarui bersama.
DILAPORKAN = {
    ("CGM", 30.0): {"RMSE": 18.67, "MAE": 13.19, "Clarke_A+B": 95.87},
    ("CGM", 60.0): {"RMSE": 31.18, "MAE": 22.96, "Clarke_A+B": 87.49},
    # Finger-stick +4 jam sengaja ikut diverifikasi meski mutunya rendah. Angka yang
    # buruk tetap harus dapat direproduksi; menyembunyikannya dari verifikasi justru
    # membuatnya tak terpantau ketika dokter mengujinya pada Skenario 2.
    ("FINGER_STICK", 240.0): {"RMSE": 68.24, "MAE": 55.62, "Clarke_A+B": 52.52},
}

TOL = 0.05  # mg/dL / poin persen


def _stem(sumber: str, horizon_min: float) -> str:
    """Penamaan artefak mengikuti yang ditulis src/models/gbm_model.py."""
    if sumber == "FINGER_STICK":
        return "gbm_finger_stick_h4h_seq8"
    return f"gbm_{sumber.lower()}_h{int(horizon_min)}m"


def evaluasi(cfg: dict, sumber: str, horizon_min: float) -> dict:
    """Muat bundle, jalankan pada pembagian uji resmi, kembalikan metrik."""
    modal = muat_dataset_modalitas(cfg, sumber=sumber, horizon_min=horizon_min)

    uji = modal.df[modal.df["dataset_split"] == "test"]
    X, y, anchor = bentuk_jendela_bagian(modal, uji, horizon_min)

    with open(MODELS_DIR / f"{_stem(sumber, horizon_min)}_inference_bundle.pkl", "rb") as f:
        bundle = pickle.load(f)

    n, seq, n_feat = X.shape
    X_scaled = bundle["scaler"].transform(X.reshape(-1, n_feat)).reshape(n, seq * n_feat)

    pred = bundle["model"].predict(X_scaled)
    if bundle.get("predict_delta"):
        pred = pred + anchor

    metrik = calculate_all_metrics(y, pred)
    metrik["Clarke_A+B"] = metrik["Clarke_A"] + metrik["Clarke_B"]
    metrik["_n_uji"] = float(len(y))
    return metrik


def main() -> int:
    cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))

    keluarga = cfg["model"].get("name", "GradientBoosting")
    if keluarga != "GradientBoosting":
        logger.error(
            f"config.model.name = '{keluarga}'. Skrip ini memverifikasi bundle GBM, "
            "yaitu keluarga yang dipakai backend produksi."
        )
        return 1

    berkas_data = (
        PROJECT_ROOT / "data" / "raw"
        / cfg["data"].get("unified_dataset", "ohio_t1dm.csv")
    )
    if not berkas_data.exists():
        logger.error(
            f"Dataset gabungan tidak ditemukan: {berkas_data}. "
            "Jalankan `python -m src.data.ohio_parser` lebih dulu."
        )
        return 1

    ok = True

    for (sumber, horizon_min), dilaporkan in DILAPORKAN.items():
        label = f"{sumber} +{horizon_min:g} menit"
        stem = _stem(sumber, horizon_min)

        bundle_path = MODELS_DIR / f"{stem}_inference_bundle.pkl"
        metrics_path = MODELS_DIR / f"{stem}_metrics.json"

        if not bundle_path.exists():
            logger.error(f"[{label}] bundle hilang: {bundle_path.name}")
            ok = False
            continue

        try:
            aktual = evaluasi(cfg, sumber, horizon_min)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[{label}] gagal dievaluasi: {type(exc).__name__}: {exc}")
            ok = False
            continue

        logger.info(
            f"[{label}] reproduksi atas {int(aktual['_n_uji'])} jendela: "
            f"RMSE {aktual['RMSE']:.2f} | MAE {aktual['MAE']:.2f} | "
            f"Clarke A+B {aktual['Clarke_A+B']:.2f}%"
        )

        # 1) Cocok dengan metrik yang tersimpan saat pelatihan?
        if metrics_path.exists():
            tersimpan = json.loads(metrics_path.read_text(encoding="utf-8"))
            tersimpan.setdefault(
                "Clarke_A+B", tersimpan.get("Clarke_A", 0) + tersimpan.get("Clarke_B", 0)
            )
            for kunci in ("RMSE", "MAE", "Clarke_A+B"):
                if abs(aktual[kunci] - tersimpan[kunci]) > TOL:
                    logger.error(
                        f"[{label}] {kunci}: reproduksi {aktual[kunci]:.2f} != "
                        f"tersimpan {tersimpan[kunci]:.2f}"
                    )
                    ok = False
        else:
            logger.error(f"[{label}] metrik tersimpan hilang: {metrics_path.name}")
            ok = False

        # 2) Cocok dengan angka yang DILAPORKAN di naskah?
        for kunci, diharapkan in dilaporkan.items():
            if abs(aktual[kunci] - diharapkan) > TOL:
                logger.error(
                    f"[{label}] {kunci}: reproduksi {aktual[kunci]:.2f} != "
                    f"naskah {diharapkan:.2f}"
                )
                ok = False

    logger.info("=" * 70)
    if ok:
        logger.info("HASIL: bundle mereproduksi seluruh angka yang dilaporkan. VERIFIED.")
        logger.warning(
            "CATATAN: angka di atas berasal dari pembagian RESMI OhioT1DM, yang bersifat "
            "temporal dalam-pasien. Bab VI dan README masih memuat angka pipeline lama "
            "berbasis pembagian lintas-pasien. Keduanya tidak sebanding."
        )
        return 0

    logger.error("HASIL: bundle TIDAK mereproduksi angka laporan. FAILED.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
