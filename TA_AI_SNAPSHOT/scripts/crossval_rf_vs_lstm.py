"""T1.1 — Uji signifikansi RF vs LSTM pada basis pembanding yang BENAR-BENAR sama.

Melatih RF DAN LSTM pada fold yang SAMA (test sample identik) sehingga error-nya
berpasangan. Uji: (a) fold-level paired (Wilcoxon + paired-t pada RMSE), (b) pooled
per-sample (Wilcoxon pada |error|, daya tinggi).

MENGAPA SKRIP INI DIPERBAIKI (T1.1)
-----------------------------------
Versi sebelumnya memanggil ``create_sequences()`` **tanpa** ``max_gap_steps``, sedangkan
pelatihan produksi memakainya sejak Tugas 5. Akibatnya:

- Hasil di `crossval_rf_vs_lstm.json` (9 Juli) dihitung atas jendela yang boleh melintasi
  jeda sensor, untuk KEDUA model.
- RF produksi kini RMSE 21,12 setelah segmentasi, sedangkan angka lama menyebut 22,98.
  Menyandingkan 21,12 (RF tersegmentasi) dengan 22,54 (LSTM TAK tersegmentasi) akan
  menguntungkan RF secara tidak sah.

Skrip ini menerapkan `max_gap_steps` dari config untuk kedua model, sehingga perbandingannya
sah. Selain itu ditambahkan hal-hal yang dituntut T1.1 tetapi belum ada:

- kedua horizon (--horizon, dalam LANGKAH)
- MAE dan MAPE, bukan hanya RMSE
- rincian zona Clarke A sampai E, bukan hanya A+B
- waktu latih, waktu inferensi, dan ukuran model (KNF-01 dan Subbab III.4.2)

Keluaran: results/eval_prediksi/crossval_rf_vs_lstm_h{N}.json
"""
import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
from src.models.lstm_model import LSTMGlucoseModel
from src.utils.metrics import calculate_all_metrics

K = 6
OUT_DIR = Path("results/eval_prediksi")
ZONA = ["Clarke_A", "Clarke_B", "Clarke_C", "Clarke_D", "Clarke_E"]


def ukuran_rf_mb(model) -> float:
    """Ukuran model RF setelah diserialisasi (MB) — sebanding dengan bundle produksi."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(model, f)
        p = f.name
    n = os.path.getsize(p)
    os.unlink(p)
    return round(n / 1e6, 2)


def ukuran_lstm_mb(lm) -> float:
    """Ukuran model LSTM: jumlah parameter x 4 byte (float32)."""
    try:
        return round(lm.model.count_params() * 4 / 1e6, 3)
    except Exception:  # noqa: BLE001
        return float("nan")


def ringkas_metrik(y, yp) -> dict:
    m = calculate_all_metrics(y, yp)
    out = {k: round(float(m[k]), 3) for k in ("RMSE", "MAE") if k in m}
    for k in ("MAPE", "Clarke_A+B", *ZONA):
        if k in m:
            out[k] = round(float(m[k]), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=6,
                    help="horizon dalam LANGKAH (6 = 30 menit, 12 = 60 menit)")
    args = ap.parse_args()
    HORIZON = args.horizon

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg.setdefault("model", {}).setdefault("lstm", {})["verbose"] = 0
    m = cfg["model"]
    seq_len = m.get("sequence_length", 12)
    use_eng = m.get("use_engineered", False)
    predict_delta = m.get("predict_delta", False)
    fe = m.get("feature_engineering", {})
    feats = m["engineered_features"] if use_eng else m["features"]
    rf = m.get("random_forest", {})
    seed = cfg.get("data", {}).get("seed", 42)

    # INTI PERBAIKAN T1.1: segmentasi jeda sensor, sama seperti pelatihan produksi.
    max_gap_steps = m.get("max_gap_steps")
    cadence_min = float(cfg.get("data", {}).get("sampling_interval_min", 5))
    if max_gap_steps is None:
        raise SystemExit("config.model.max_gap_steps tidak ada. Perbandingan tidak akan "
                         "sepadan dengan pelatihan produksi; perbaiki config lebih dulu.")

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv") \
        .sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)

    pids = sorted(df["patient_id"].unique().tolist())
    folds = [pids[i::K] for i in range(K)]

    print(f"Horizon +{HORIZON * cadence_min:.0f} mnt | max_gap_steps={max_gap_steps} "
          f"| {K} fold lintas-pasien")

    fold_rows, rf_err_all, lstm_err_all = [], [], []
    t_rf, t_ls, inf_rf, inf_ls, sz_rf, sz_ls = [], [], [], [], [], []

    for fi, test_pids in enumerate(folds):
        prep.feature_columns = list(feats)
        tr, te = prep.split_by_patient(df, test_pids)
        seq_kw = {"max_gap_steps": max_gap_steps, "source_interval_min": cadence_min}
        Xtr, ytr, atr = prep.create_sequences(tr, seq_len, HORIZON, return_anchor=True, **seq_kw)
        Xte, yte, ate = prep.create_sequences(te, seq_len, HORIZON, return_anchor=True, **seq_kw)
        p2 = DataPreprocessor(cfg)
        Xtr_s, Xte_s = p2.normalize_data(Xtr, Xte)
        ytr_fit = (ytr - atr) if predict_delta else ytr

        # ── RF ──
        rfm = RandomForestRegressor(n_estimators=rf.get("n_estimators", 200),
                                    max_depth=rf.get("max_depth", 20),
                                    min_samples_split=rf.get("min_samples_split", 5),
                                    random_state=seed, n_jobs=-1)
        t0 = time.time()
        rfm.fit(Xtr_s.reshape(len(ytr), -1), ytr_fit)
        t_rf.append(time.time() - t0)
        t0 = time.time()
        yp_rf = rfm.predict(Xte_s.reshape(len(yte), -1))
        inf_rf.append((time.time() - t0) / max(len(yte), 1) * 1000)   # ms per sampel
        sz_rf.append(ukuran_rf_mb(rfm))
        yp_rf = yp_rf + ate if predict_delta else yp_rf

        # ── LSTM ── val split 10% dari train untuk early stopping
        nval = max(1, int(0.1 * len(ytr)))
        lm = LSTMGlucoseModel(cfg)
        t0 = time.time()
        lm.train(Xtr_s[:-nval], ytr_fit[:-nval], Xtr_s[-nval:], ytr_fit[-nval:])
        t_ls.append(time.time() - t0)
        t0 = time.time()
        yp_ls = lm.predict(Xte_s)
        inf_ls.append((time.time() - t0) / max(len(yte), 1) * 1000)
        sz_ls.append(ukuran_lstm_mb(lm))
        yp_ls = yp_ls + ate if predict_delta else yp_ls

        rf_met, ls_met = ringkas_metrik(yte, yp_rf), ringkas_metrik(yte, yp_ls)
        rf_err_all.append(np.abs(yte - yp_rf))
        lstm_err_all.append(np.abs(yte - yp_ls))
        fold_rows.append({"fold": fi, "test": test_pids, "n": int(len(yte)),
                          "RF": rf_met, "LSTM": ls_met})
        print(f"  fold {fi} {test_pids}: n={len(yte):>6} | "
              f"RF RMSE {rf_met['RMSE']:.2f} | LSTM RMSE {ls_met['RMSE']:.2f}", flush=True)

    def rerata(kunci, model):
        v = [f[model][kunci] for f in fold_rows if kunci in f[model]]
        return [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] if v else None

    rf_rmse = np.array([f["RF"]["RMSE"] for f in fold_rows])
    ls_rmse = np.array([f["LSTM"]["RMSE"] for f in fold_rows])
    w_p = stats.wilcoxon(rf_rmse, ls_rmse).pvalue if len(set(rf_rmse - ls_rmse)) > 1 else float("nan")
    t_p = stats.ttest_rel(rf_rmse, ls_rmse).pvalue
    rf_e, ls_e = np.concatenate(rf_err_all), np.concatenate(lstm_err_all)
    ws_p = stats.wilcoxon(rf_e, ls_e).pvalue

    res = {
        "catatan": ("Kedua model dilatih atas jendela TERSEGMENTASI (max_gap_steps), sama "
                    "seperti pelatihan produksi. Versi 9 Juli TIDAK tersegmentasi dan angkanya "
                    "tidak sebanding."),
        "horizon_steps": HORIZON, "horizon_min": int(HORIZON * cadence_min),
        "max_gap_steps": max_gap_steps, "k_fold": K,
        "folds": fold_rows,
        "rerata_lintas_fold": {
            "RF": {k: rerata(k, "RF") for k in ("RMSE", "MAE", "MAPE", "Clarke_A+B", *ZONA)},
            "LSTM": {k: rerata(k, "LSTM") for k in ("RMSE", "MAE", "MAPE", "Clarke_A+B", *ZONA)},
        },
        "biaya": {
            "RF": {"waktu_latih_dtk": round(float(np.mean(t_rf)), 1),
                   "inferensi_ms_per_sampel": round(float(np.mean(inf_rf)), 4),
                   "ukuran_model_MB": round(float(np.mean(sz_rf)), 2)},
            "LSTM": {"waktu_latih_dtk": round(float(np.mean(t_ls)), 1),
                     "inferensi_ms_per_sampel": round(float(np.mean(inf_ls)), 4),
                     "ukuran_model_MB": round(float(np.mean(sz_ls)), 3)},
        },
        "uji": {
            "fold_level_paired": {"wilcoxon_p": round(float(w_p), 4),
                                  "ttest_rel_p": round(float(t_p), 4), "n_fold": K},
            "sample_level_wilcoxon": {"p": float(ws_p), "n": int(len(rf_e)),
                                      "median_abs_err_RF": round(float(np.median(rf_e)), 2),
                                      "median_abs_err_LSTM": round(float(np.median(ls_e)), 2)},
        },
    }
    OUT = OUT_DIR / f"crossval_rf_vs_lstm_h{HORIZON}.json"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)

    r = res["rerata_lintas_fold"]
    b = res["biaya"]
    print(f"\n=== RF vs LSTM (+{res['horizon_min']} mnt, {K} fold, TERSEGMENTASI) ===")
    print(f"{'metrik':<14}{'RF':>18}{'LSTM':>18}")
    for k in ("RMSE", "MAE", "MAPE", "Clarke_A+B", *ZONA):
        a, c = r["RF"].get(k), r["LSTM"].get(k)
        if a and c:
            print(f"{k:<14}{a[0]:>10.2f} ±{a[1]:<6.2f}{c[0]:>10.2f} ±{c[1]:<6.2f}")
    print(f"\n{'biaya':<24}{'RF':>14}{'LSTM':>14}")
    print(f"{'waktu latih (dtk)':<24}{b['RF']['waktu_latih_dtk']:>14.1f}{b['LSTM']['waktu_latih_dtk']:>14.1f}")
    print(f"{'inferensi (ms/sampel)':<24}{b['RF']['inferensi_ms_per_sampel']:>14.4f}{b['LSTM']['inferensi_ms_per_sampel']:>14.4f}")
    print(f"{'ukuran model (MB)':<24}{b['RF']['ukuran_model_MB']:>14.2f}{b['LSTM']['ukuran_model_MB']:>14.3f}")
    print(f"\nFold-level paired: Wilcoxon p={w_p:.4f} | paired-t p={t_p:.4f}")
    print(f"Sample-level Wilcoxon p={ws_p:.2e} (n={len(rf_e)})")
    print(f"Kesimpulan fold-level: {'SIGNIFIKAN' if w_p < 0.05 else 'TIDAK signifikan'} (alpha=0,05)")
    print(f"Output -> {OUT}")


if __name__ == "__main__":
    main()
