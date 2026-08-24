"""Evaluasi keamanan klinis: metrik khusus HIPOGLIKEMIA + ketidakpastian prediksi.

Melengkapi evaluasi agregat (RMSE/MAE/Clarke) dengan dua aspek yang menentukan bagi
alat bantu medis:

 1. Deteksi hipoglikemia (safety-critical): sensitivitas, spesifisitas, dan PPV pada
    ambang <70 mg/dL, ditambah RMSE khusus kejadian hipo dan jumlah hipo yang terlewat
    ke rentang normal — yaitu kegagalan yang paling berbahaya secara klinis.
 2. Ketidakpastian: interval 95% dan cakupan empirisnya.

DIPERBARUI 24 Agustus 2026. Dua hal berubah dan keduanya penting:

* Model yang dievaluasi kini GBM, bukan RF. Sebelumnya skrip ini memuat
  ``rf_inference_bundle.pkl`` — artefak yang sudah tidak ada DAN bukan model produksi.
  Selama itu berlangsung, "ketidakpastian" yang dilaporkan berasal dari model yang
  tidak pernah melayani satu pun permintaan.

* Sumber intervalnya berubah mengikuti model. RF memperoleh sebaran dari selisih
  antar-pohon (``model.estimators_``); GBM tidak punya padanannya, dan sebagai gantinya
  melatih DUA model kuantil pada 2,5% dan 97,5%. Interval 95% karena itu diambil
  LANGSUNG dari kedua kuantil tersebut, bukan dari ±1,96·sigma. Menyamakan keduanya
  dengan mengasumsikan sebaran normal akan salah pada ekor — dan ekor itulah yang
  justru menjadi pokok perhatian hipoglikemia.

Pembagian uji mengikuti pembagian RESMI OhioT1DM, sama dengan pelatihan.

    python scripts/eval_hypo_uncertainty.py --source CGM --horizon-min 30
"""
import torch  # noqa: F401
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import yaml

from src.constants import GLUCOSE_HIGH, GLUCOSE_LOW
from src.models.persiapan_data import bentuk_jendela_bagian, muat_dataset_modalitas

HYPO = 70.0
OUT_DIR = Path("results/eval_prediksi")


def _stem(sumber: str, horizon_min: float) -> str:
    if sumber == "FINGER_STICK":
        return "gbm_finger_stick_h4h_seq8"
    return f"gbm_{sumber.lower()}_h{int(horizon_min)}m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keamanan hipoglikemia dan ketidakpastian prediksi produksi"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default="CGM", choices=["CGM", "FINGER_STICK"])
    parser.add_argument("--horizon-min", type=float, default=30.0)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    stem = _stem(args.source, args.horizon_min)
    bundle_path = Path("models") / f"{stem}_inference_bundle.pkl"
    if not bundle_path.exists():
        raise SystemExit(
            f"Bundle produksi tidak ditemukan: {bundle_path}. "
            "Latih GBM lebih dulu lewat `python -m src.models.gbm_model`."
        )

    b = pickle.load(open(bundle_path, "rb"))
    model, scaler = b["model"], b["scaler"]
    predict_delta = bool(b.get("predict_delta"))
    horizon_min = float(b.get("prediction_horizon_min", args.horizon_min))

    modal = muat_dataset_modalitas(cfg, sumber=args.source, horizon_min=horizon_min)
    uji = modal.df[modal.df["dataset_split"] == "test"]
    X_test, y_test, anc = bentuk_jendela_bagian(modal, uji, horizon_min)

    n, s, nf = X_test.shape
    Xs = scaler.transform(X_test.reshape(-1, nf)).reshape(n, s * nf)

    pred_fit = model.predict(Xs)
    y_pred = pred_fit + anc if predict_delta else pred_fit

    # --- Interval 95% dari dua model kuantil ---------------------------------
    std_models = b.get("std_models") or {}
    q_low, q_high = std_models.get("low"), std_models.get("high")

    if q_low is None or q_high is None:
        raise SystemExit(
            "Bundle tidak memuat model kuantil, sehingga interval prediksi tidak dapat "
            "dihitung. Latih ulang GBM dengan with_uncertainty aktif."
        )

    lo_fit, hi_fit = q_low.predict(Xs), q_high.predict(Xs)
    lo = lo_fit + anc if predict_delta else lo_fit
    hi = hi_fit + anc if predict_delta else hi_fit

    # Kuantil tidak dijamin terurut; tukar bila terbalik agar lebar interval tidak negatif.
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)) * 100)
    lebar = hi - lo

    # --- Deteksi hipoglikemia -------------------------------------------------
    true_hypo, pred_hypo = y_test < HYPO, y_pred < HYPO
    tp = int(np.sum(true_hypo & pred_hypo))
    fn = int(np.sum(true_hypo & ~pred_hypo))
    fp = int(np.sum(~true_hypo & pred_hypo))
    tn = int(np.sum(~true_hypo & ~pred_hypo))

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else float("nan")

    m = true_hypo
    hypo_rmse = (
        float(np.sqrt(np.mean((y_test[m] - y_pred[m]) ** 2))) if m.any() else float("nan")
    )
    hypo_missed_to_normal = int(
        np.sum(m & (y_pred >= GLUCOSE_LOW) & (y_pred <= GLUCOSE_HIGH))
    )

    res = {
        "model_family": b.get("model_family", "unknown"),
        "glucose_source": args.source,
        "horizon_min": horizon_min,
        "split": "official_dataset_split (temporal within-patient)",
        "n_test": int(n),
        "n_hypo_events": int(true_hypo.sum()),
        "hypo_detection": {
            "sensitivity_%": round(sens * 100, 1),
            "specificity_%": round(spec * 100, 1),
            "PPV_%": round(ppv * 100, 1),
            "F1_%": round(f1 * 100, 1),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        },
        "hypo_rmse_mgdl": round(hypo_rmse, 2),
        "hypo_missed_to_normal_range": hypo_missed_to_normal,
        "uncertainty": {
            "method": b.get("std_method", "quantile_spread"),
            "quantiles": b.get("std_quantiles"),
            "interval95_coverage_%": round(coverage, 1),
            "mean_interval_width_mgdl": round(float(np.mean(lebar)), 1),
            "median_interval_width_mgdl": round(float(np.median(lebar)), 1),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"hypo_safety_uncertainty_{stem}.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)

    print(f"=== Keamanan Hipoglikemia & Ketidakpastian ({args.source} +{horizon_min:g} mnt) ===")
    print(f"Model              : {res['model_family']}")
    print(f"Jendela uji        : {n}  |  kejadian hipo: {int(true_hypo.sum())}")
    print(
        f"Deteksi hipo       : sensitivitas {res['hypo_detection']['sensitivity_%']}% | "
        f"spesifisitas {res['hypo_detection']['specificity_%']}% | "
        f"PPV {res['hypo_detection']['PPV_%']}%"
    )
    print(
        f"RMSE saat hipo     : {res['hypo_rmse_mgdl']} mg/dL | "
        f"hipo terlewat ke rentang normal: {hypo_missed_to_normal}"
    )
    print(
        f"Interval 95%       : cakupan {coverage:.1f}% (ideal ~95%) | "
        f"lebar rata-rata {res['uncertainty']['mean_interval_width_mgdl']} mg/dL"
    )
    print(f"\nOutput -> {out}")


if __name__ == "__main__":
    main()
