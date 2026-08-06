"""Kalibrasi interval prediksi RF via SPLIT CONFORMAL PREDICTION.

Interval baseline (pred +/- 1.96*std antar-pohon) hanya ~84% cakupan (underconfident).
Conformal memberi jaminan cakupan distribution-free: kuantil residual kalibrasi menentukan
lebar interval sehingga cakupan test mendekati target (mis. 95%).

Dua varian:
  - absolut       : interval = pred +/- q            (lebar seragam)
  - ternormalisasi: interval = pred +/- q*std        (adaptif: lebar mengikuti ketidakpastian)

Split per-pasien (train/kalibrasi/test terpisah) → meniru deployment untuk pasien baru.

Horizon dipilih lewat --horizon (dalam LANGKAH, bukan menit). Aplikasi menampilkan dua
horizon, dan faktor konformalnya TIDAK boleh dipakai lintas horizon: prediksi 60 menit
jauh lebih tidak pasti daripada 30 menit, sehingga memakai satu faktor untuk keduanya
membuat salah satu interval keliru lebar atau keliru sempit.

Output: results/eval_prediksi/conformal_h{N}.json

Catatan penting (diperbaiki pada A3): sebelumnya skrip ini membangun jendela TANPA
``max_gap_steps``, sementara pelatihan produksi (src/models/rf_model.py) memakainya sejak
Tugas 5. Modelnya karena itu dilatih atas jendela yang boleh melintasi jeda sensor
sedangkan model produksi tidak, sehingga faktor yang dihasilkan mengkalibrasi model yang
berbeda dari yang benar-benar dipakai aplikasi.
"""
import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import time
from pathlib import Path
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor

OUT_DIR = Path("results/eval_prediksi")
EPS = 1e-6


def rf_std(model, Xf):
    return np.stack([t.predict(Xf) for t in model.estimators_]).std(axis=0)


def conformal_q(scores, alpha):
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)  # koreksi finite-sample
    return float(np.quantile(scores, level, method="higher"))


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    m = cfg["model"]

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=m.get("default_horizon", 6),
                    help="horizon dalam LANGKAH (6 = 30 menit, 12 = 60 menit)")
    args = ap.parse_args()

    seq_len = m.get("sequence_length", 12); horizon = args.horizon
    use_eng = m.get("use_engineered", False); predict_delta = m.get("predict_delta", False)
    fe = m.get("feature_engineering", {})
    feats = m["engineered_features"] if use_eng else m["features"]
    rf = m.get("random_forest", {}); seed = cfg.get("data", {}).get("seed", 42)

    # Segmentasi jeda sensor WAJIB sama dengan pelatihan produksi (Tugas 5), kalau tidak
    # faktor konformalnya mengkalibrasi model yang berbeda dari yang dipakai aplikasi.
    max_gap_steps = m.get("max_gap_steps")
    cadence_min = float(cfg.get("data", {}).get("sampling_interval_min", 5))

    out_path = OUT_DIR / f"conformal_h{horizon}.json"
    t0 = time.time()

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv").sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)
    prep.feature_columns = list(feats)

    pids = sorted(df["patient_id"].unique().tolist())
    test_p, cal_p, train_p = pids[-2:], pids[-4:-2], pids[:-4]
    print(f"train={len(train_p)} kalibrasi={cal_p} test={test_p}")

    def seqs(sub):
        return prep.create_sequences(
            df[df["patient_id"].isin(sub)], seq_len, horizon, return_anchor=True,
            max_gap_steps=max_gap_steps, source_interval_min=cadence_min,
        )

    Xtr, ytr, atr = seqs(train_p); Xca, yca, aca = seqs(cal_p); Xte, yte, ate = seqs(test_p)
    p2 = DataPreprocessor(cfg)
    Xtr_s, _ = p2.normalize_data(Xtr, None)
    Xca_s = p2.scaler.transform(Xca.reshape(-1, Xca.shape[2])).reshape(Xca.shape)
    Xte_s = p2.scaler.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)
    Xtr_f = Xtr_s.reshape(len(ytr), -1); Xca_f = Xca_s.reshape(len(yca), -1); Xte_f = Xte_s.reshape(len(yte), -1)

    model = RandomForestRegressor(n_estimators=rf.get("n_estimators", 200), max_depth=rf.get("max_depth", 20),
                                  min_samples_split=rf.get("min_samples_split", 5), random_state=seed, n_jobs=-1)
    model.fit(Xtr_f, (ytr - atr) if predict_delta else ytr)

    def predict(Xf, anc):
        out = model.predict(Xf)
        return (out + anc) if predict_delta else out

    yca_p, yte_p = predict(Xca_f, aca), predict(Xte_f, ate)
    std_ca, std_te = rf_std(model, Xca_f), rf_std(model, Xte_f)
    res_ca = np.abs(yca - yca_p)  # residual kalibrasi

    def coverage(lo, hi):
        return float(np.mean((yte >= lo) & (yte <= hi)) * 100)

    def width(lo, hi):
        return float(np.mean(hi - lo))

    out = {
        "horizon_steps": int(horizon),
        "horizon_min": int(horizon * cadence_min),
        "max_gap_steps": max_gap_steps,
        "n_cal": int(len(yca)), "n_test": int(len(yte)),
        "levels": {},
    }
    for alpha, tgt in [(0.10, 90), (0.05, 95)]:
        # baseline: +/- z*std
        z = 1.645 if tgt == 90 else 1.96
        lo_b, hi_b = yte_p - z * std_te, yte_p + z * std_te
        # conformal absolut
        q_abs = conformal_q(res_ca, alpha)
        lo_a, hi_a = yte_p - q_abs, yte_p + q_abs
        # conformal ternormalisasi
        q_norm = conformal_q(res_ca / (std_ca + EPS), alpha)
        lo_n, hi_n = yte_p - q_norm * std_te, yte_p + q_norm * std_te
        out["levels"][tgt] = {
            "baseline_z_std": {"coverage%": round(coverage(lo_b, hi_b), 1), "mean_width": round(width(lo_b, hi_b), 1)},
            "conformal_absolute": {"coverage%": round(coverage(lo_a, hi_a), 1), "mean_width": round(width(lo_a, hi_a), 1), "q": round(q_abs, 2)},
            "conformal_normalized": {"coverage%": round(coverage(lo_n, hi_n), 1), "mean_width": round(width(lo_n, hi_n), 1), "q": round(q_norm, 2)},
        }

    out["durasi_detik"] = round(time.time() - t0, 1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)

    print(f"\n=== KALIBRASI INTERVAL (+{out['horizon_min']} mnt, n_cal={len(yca)}, "
          f"n_test={len(yte)}, max_gap_steps={max_gap_steps}) ===")
    for tgt, d in out["levels"].items():
        print(f"\nTarget cakupan {tgt}%:")
        for name, s in d.items():
            print(f"  {name:22s}: cakupan {s['coverage%']:>5}%  | lebar rata2 {s['mean_width']:>5} mg/dL")
    print(f"\nDurasi {out['durasi_detik']} dtk. Output -> {out_path}")


if __name__ == "__main__":
    main()
