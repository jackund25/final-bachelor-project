"""Pengklasifikasi kondisi masa depan: memperbaiki leher botol PC-RAG sekaligus deteksi hipoglikemia.

Latar. Evaluasi pada kasus nyata (eval_retrieval_realcases.py) menunjukkan bahwa manfaat
Prediction-Conditioned RAG dibatasi bukan oleh mekanismenya --- dengan glukosa masa depan yang
benar, retrieval nyaris sempurna (MRR 0,99) --- melainkan oleh prediktornya: pada kasus divergen,
regresi Random Forest hanya benar menebak KONDISI masa depan 15,8% kali. Akarnya sama dengan
rendahnya sensitivitas hipoglikemia (14%): regresi yang meminimalkan galat kuadrat menyusut ke
tengah (under-dispersed), sehingga jarang berani melewati ambang 70/180 mg/dL.

Gagasan. PC-RAG sebenarnya tidak membutuhkan NILAI glukosa, melainkan KONDISI-nya
(hipoglikemia / normal / hiperglikemia) untuk membentuk kueri. Karena itu, alih-alih
menurunkan kondisi dari regresi lalu mengambangkannya, kondisi diprediksi LANGSUNG oleh
pengklasifikasi tiga kelas yang sadar-biaya (class_weight balanced), sehingga kelas minoritas
yang justru paling penting secara klinis (hipoglikemia) tidak lagi tenggelam.

Model ini melengkapi --- bukan menggantikan --- model regresi: regresi tetap dipakai untuk
menampilkan nilai prediksi dan intervalnya kepada dokter, sedangkan pengklasifikasi dipakai
untuk (a) peringatan dini hipoglikemia dan (b) pengondisian kueri PC-RAG.

KELUARGA MODEL mengikuti config.model.name, dapat ditimpa lewat --model.
Sejak 13 Agustus 2026 produksi memakai Gradient Boosting. Mekanisme intinya TIDAK
berubah: ``class_weight="balanced"`` tetap dipakai, dan HistGradientBoostingClassifier
mendukungnya sejak scikit-learn 1.2. Yang berubah hanya keluarga pohonnya.

Alasan ikut berganti: pengklasifikasi RF berukuran 195,69 MB, yaitu 98,5% dari seluruh
jejak penyimpanan produksi setelah regresor berpindah ke GBM (1,45 MB per horizon).
Membiarkannya sebagai RF akan membatalkan sebagian besar klaim efisiensi yang menjadi
alasan penggantian model pada Rumusan Masalah 1.

BASELINE PEMBANDING ikut mengikuti keluarga yang sama. Baris "regresi lalu ambang"
harus berasal dari regresor yang BENAR-BENAR dipakai produksi; membandingkan
pengklasifikasi GBM terhadap regresi RF akan mencampur dua perubahan sekaligus.

Keluaran:
  models/{gbm,rf}_condition_classifier_h6.pkl
  results/eval_prediksi/condition_classifier.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import CONDITION_CLASSES, classify_glucose_3class  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402

HORIZON = 6          # +30 menit
SEED = 42
# Ambang dan daftar kelas dari SATU sumber kebenaran (src/constants.py).
# Tetap 3 kelas: memecah hipoglikemia di ambang 54 tidak mengubah tindakan klinis,
# sementara data latih hanya punya 998 sampel hipoglikemia berat (0,72%).
CLASSES = CONDITION_CLASSES
classify_glucose = classify_glucose_3class


def build(cfg: dict):
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(mc["engineered_features"])

    patients = sorted(df["patient_id"].unique())
    train_df, test_df = pre.split_by_patient(df, patients[-2:])

    # Segmentasi jeda yang sama dengan model regresi (Tugas 5). Kalau hanya salah
    # satu yang disaring, keduanya dilatih atas himpunan jendela berbeda dan
    # perbandingan "regresi lalu ambang" vs "pengklasifikasi" tidak lagi setara.
    max_gap_steps = mc.get("max_gap_steps")
    cadence_min = cfg.get("data", {}).get("sampling_interval_min", 5)

    def seqs(d):
        X, y, anc = pre.create_sequences(
            d, mc["sequence_length"], HORIZON, return_anchor=True,
            max_gap_steps=max_gap_steps, source_interval_min=cadence_min,
        )
        n, s, f = X.shape
        return X.reshape(n, s * f), y, anc

    Xtr, ytr, atr = seqs(train_df)
    Xte, yte, ate = seqs(test_df)
    return (Xtr, ytr, atr), (Xte, yte, ate), mc


def metrics_for(y_true_lbl: np.ndarray, y_pred_lbl: np.ndarray) -> dict:
    cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=CLASSES)
    out = {"akurasi_keseluruhan_%": round(100 * float((y_true_lbl == y_pred_lbl).mean()), 1)}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        sens = 100 * tp / (tp + fn) if tp + fn else 0.0
        ppv = 100 * tp / (tp + fp) if tp + fp else 0.0
        out[c] = {
            "n": int(cm[i, :].sum()),
            "sensitivitas_%": round(float(sens), 1),
            "PPV_%": round(float(ppv), 1),
        }
    return out


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    ap = argparse.ArgumentParser(description="Latih pengklasifikasi kondisi masa depan")
    ap.add_argument("--model", choices=["gbm", "rf"],
                    default=("rf" if cfg["model"].get("name") == "RandomForest" else "gbm"),
                    help="keluarga model; bawaan mengikuti config.model.name")
    args = ap.parse_args()
    prefix = args.model

    (Xtr, ytr, atr), (Xte, yte, ate), mc = build(cfg)

    lbl_tr = np.array([classify_glucose(v) for v in ytr])
    lbl_te = np.array([classify_glucose(v) for v in yte])
    cur_te = np.array([classify_glucose(v) for v in ate])
    divergent = cur_te != lbl_te

    print(f"Latih: {len(ytr)} | Uji: {len(yte)} | kasus divergen di data uji: {int(divergent.sum())}")
    print("Distribusi kelas (uji):", {c: int((lbl_te == c).sum()) for c in CLASSES})

    # --- Baseline: kondisi diturunkan dari REGRESI (pendekatan laporan saat ini)
    # Regresor pembanding WAJIB sekeluarga dengan pengklasifikasinya, kalau tidak
    # selisih yang terukur mencampur "pengklasifikasi lebih baik daripada regresi"
    # dengan "GBM lebih baik daripada RF".
    bundle_path = ROOT / f"models/{prefix}_inference_bundle_h{HORIZON}.pkl"
    if not bundle_path.exists():
        raise SystemExit(
            f"Bundle regresi tidak ditemukan: {bundle_path.name}\n"
            f"Latih dulu: PYTHONPATH=. python -m src.models."
            f"{'gbm_model' if prefix == 'gbm' else 'rf_model'} --horizon {HORIZON}"
        )
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)
    n_feat = len(mc["engineered_features"])
    Xte_s = bundle["scaler"].transform(Xte.reshape(-1, n_feat)).reshape(len(Xte), -1)
    reg_pred = bundle["model"].predict(Xte_s) + (ate if bundle["predict_delta"] else 0)
    lbl_reg = np.array([classify_glucose(v) for v in reg_pred])

    # --- Usulan: pengklasifikasi kondisi sadar-biaya
    scaler = bundle["scaler"]
    Xtr_s = scaler.transform(Xtr.reshape(-1, n_feat)).reshape(len(Xtr), -1)
    # class_weight="balanced" adalah INTI percobaan ini pada kedua keluarga: tanpanya
    # kelas hipoglikemia (3,3% sampel) tenggelam dan sensitivitasnya kembali runtuh.
    if prefix == "gbm":
        clf = HistGradientBoostingClassifier(
            class_weight="balanced",
            random_state=mc.get("gradient_boosting", {}).get("random_state", SEED),
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=mc["random_forest"]["n_estimators"],
            max_depth=mc["random_forest"]["max_depth"],
            min_samples_split=mc["random_forest"]["min_samples_split"],
            class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
    clf.fit(Xtr_s, lbl_tr)
    lbl_clf = clf.predict(Xte_s)

    res = {
        "horizon_menit": HORIZON * 5,
        # Provenans ditulis ke berkas hasil, bukan hanya tersirat dari nama berkas.
        "keluarga_model": type(clf).__name__,
        "regresor_pembanding": type(bundle["model"]).__name__,
        "class_weight": "balanced",
        "catatan": (
            "Kondisi masa depan diprediksi langsung oleh pengklasifikasi tiga kelas "
            "(class_weight=balanced), dibandingkan terhadap kondisi yang diturunkan dari regresi "
            "pada keluarga model yang SAMA."
        ),
        "n_uji": int(len(yte)),
        "n_divergen": int(divergent.sum()),
        "regresi_lalu_ambang": metrics_for(lbl_te, lbl_reg),
        "pengklasifikasi_kondisi": metrics_for(lbl_te, lbl_clf),
        "pada_kasus_divergen": {
            "akurasi_kondisi_regresi_%": round(100 * float((lbl_reg[divergent] == lbl_te[divergent]).mean()), 1),
            "akurasi_kondisi_pengklasifikasi_%": round(100 * float((lbl_clf[divergent] == lbl_te[divergent]).mean()), 1),
        },
    }

    print("\n--- Kondisi dari REGRESI (pendekatan saat ini) ---")
    for c in CLASSES:
        d = res["regresi_lalu_ambang"][c]
        print(f"  {c:15s} n={d['n']:6d}  sensitivitas {d['sensitivitas_%']:5.1f}%  PPV {d['PPV_%']:5.1f}%")
    print(f"  akurasi keseluruhan: {res['regresi_lalu_ambang']['akurasi_keseluruhan_%']}%")

    print("\n--- Kondisi dari PENGKLASIFIKASI (usulan) ---")
    for c in CLASSES:
        d = res["pengklasifikasi_kondisi"][c]
        print(f"  {c:15s} n={d['n']:6d}  sensitivitas {d['sensitivitas_%']:5.1f}%  PPV {d['PPV_%']:5.1f}%")
    print(f"  akurasi keseluruhan: {res['pengklasifikasi_kondisi']['akurasi_keseluruhan_%']}%")

    print("\n--- Pada kasus divergen (yang menuntut antisipasi) ---")
    print(f"  regresi        : {res['pada_kasus_divergen']['akurasi_kondisi_regresi_%']}%")
    print(f"  pengklasifikasi: {res['pada_kasus_divergen']['akurasi_kondisi_pengklasifikasi_%']}%")

    clf_path = ROOT / f"models/{prefix}_condition_classifier_h{HORIZON}.pkl"
    with open(clf_path, "wb") as f:
        pickle.dump({"model": clf, "scaler": scaler, "classes": CLASSES,
                     "features": mc["engineered_features"],
                     "sequence_length": mc["sequence_length"],
                     "prediction_horizon": HORIZON,
                     "model_family": type(clf).__name__}, f)
    print(f"\nPengklasifikasi -> {clf_path.name} ({clf_path.stat().st_size / 1e6:.2f} MB)")

    dest = ROOT / "results/eval_prediksi/condition_classifier.json"
    dest.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
