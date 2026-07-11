"""Evaluasi keamanan klinis: metrik khusus HIPOGLIKEMIA + ketidakpastian prediksi RF.

Melengkapi eval agregat (RMSE/MAE/Clarke) dengan dua aspek penting untuk alat medis:
 1. Deteksi hipoglikemia (safety-critical): sensitivitas/spesifisitas/PPV pada ambang <70 mg/dL,
    plus RMSE khusus kejadian hipo & zona Clarke D (gagal deteksi) pada hipo.
 2. Ketidakpastian: sebaran prediksi antar-pohon RF → interval 95% + cakupan (coverage).

Mereproduksi split & preprocessing IDENTIK dengan pelatihan (rf_model.py), memakai model &
scaler dari bundle produksi. Output: results/eval_prediksi/hypo_safety_uncertainty.json
"""
import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor

HYPO = 70.0
OUT = Path("results/eval_prediksi/hypo_safety_uncertainty.json")


def main() -> None:
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    b = pickle.load(open("models/rf_inference_bundle.pkl", "rb"))
    model, scaler, feats = b["model"], b["scaler"], b["features"]
    seq_len = int(b["sequence_length"]); horizon = int(b["prediction_horizon"])
    use_eng, predict_delta = bool(b.get("use_engineered")), bool(b.get("predict_delta"))
    fe = dict(b.get("feature_engineering", {}))

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv").sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)
    prep.feature_columns = list(feats)

    # split per-pasien IDENTIK dgn training (2 pasien terakhir = test)
    pids = sorted(df["patient_id"].unique().tolist())
    _, test_df = prep.split_by_patient(df, pids[-2:])
    X_test, y_test, anc = prep.create_sequences(test_df, seq_len, horizon, return_anchor=True)

    # skala pakai scaler produksi (per-fitur), lalu flatten seperti RF
    n, s, nf = X_test.shape
    Xs = scaler.transform(X_test.reshape(-1, nf)).reshape(n, s, nf).reshape(n, s * nf)

    # prediksi + rekonstruksi delta
    delta = model.predict(Xs)
    y_pred = delta + anc if predict_delta else delta

    # ketidakpastian: sebaran antar-pohon (delta = skala absolut karena anchor konstan/sampel)
    tree_delta = np.stack([t.predict(Xs) for t in model.estimators_])  # (n_trees, n)
    std = tree_delta.std(axis=0)
    lo, hi = y_pred - 1.96 * std, y_pred + 1.96 * std
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)) * 100)

    # deteksi hipoglikemia (biner <70)
    true_hypo, pred_hypo = y_test < HYPO, y_pred < HYPO
    tp = int(np.sum(true_hypo & pred_hypo)); fn = int(np.sum(true_hypo & ~pred_hypo))
    fp = int(np.sum(~true_hypo & pred_hypo)); tn = int(np.sum(~true_hypo & ~pred_hypo))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # recall hipo (paling penting)
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else float("nan")

    # RMSE khusus kejadian hipo + Clarke D pada hipo (true<70 tapi pred 70-180 = gagal deteksi)
    m = true_hypo
    hypo_rmse = float(np.sqrt(np.mean((y_test[m] - y_pred[m]) ** 2))) if m.any() else float("nan")
    hypo_missed_to_normal = int(np.sum(m & (y_pred >= 70) & (y_pred <= 180)))

    res = {
        "horizon_min": horizon * 5, "n_test": int(n), "n_hypo_events": int(true_hypo.sum()),
        "hypo_detection": {"sensitivity_%": round(sens * 100, 1), "specificity_%": round(spec * 100, 1),
                           "PPV_%": round(ppv * 100, 1), "F1_%": round(f1 * 100, 1),
                           "TP": tp, "FN": fn, "FP": fp, "TN": tn},
        "hypo_rmse_mgdl": round(hypo_rmse, 2),
        "hypo_missed_to_normal_range": hypo_missed_to_normal,
        "uncertainty": {"mean_std_mgdl": round(float(std.mean()), 2),
                        "median_std_mgdl": round(float(np.median(std)), 2),
                        "interval95_coverage_%": round(coverage, 1),
                        "mean_interval_width_mgdl": round(float(np.mean(hi - lo)), 1)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)

    print(f"=== Keamanan Hipoglikemia & Ketidakpastian (+{horizon*5} mnt, n_test={n}) ===")
    print(f"Kejadian hipo di test: {int(true_hypo.sum())}")
    print(f"Deteksi hipo  : sensitivitas {res['hypo_detection']['sensitivity_%']}% | "
          f"spesifisitas {res['hypo_detection']['specificity_%']}% | PPV {res['hypo_detection']['PPV_%']}%")
    print(f"RMSE saat hipo: {res['hypo_rmse_mgdl']} mg/dL | hipo terlewat ke rentang normal: {hypo_missed_to_normal}")
    print(f"Ketidakpastian: std rata2 {res['uncertainty']['mean_std_mgdl']} mg/dL | "
          f"cakupan interval 95% = {coverage:.1f}% (ideal ~95%)")
    print(f"\nOutput -> {OUT}")


if __name__ == "__main__":
    main()
