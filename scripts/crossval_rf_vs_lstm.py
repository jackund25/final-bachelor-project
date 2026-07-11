"""Uji signifikansi RF vs LSTM — 6-fold CV lintas-pasien (horizon +30 mnt).

Melatih RF DAN LSTM pada fold yang SAMA (test sample identik) → error berpasangan.
Uji: (a) fold-level paired (n=6, Wilcoxon + paired-t pada RMSE), (b) pooled per-sample
(Wilcoxon pada |error|, high power). Menjawab: apakah keunggulan LSTM signifikan, atau
RF setara (mendukung pilihan RF yang ringan/interpretable)?

Output: results/eval_prediksi/crossval_rf_vs_lstm.json
"""
import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
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
HORIZON = 6  # +30 menit (primer)
OUT = Path("results/eval_prediksi/crossval_rf_vs_lstm.json")


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cfg.setdefault("model", {}).setdefault("lstm", {})["verbose"] = 0  # diam saat CV
    m = cfg["model"]
    seq_len = m.get("sequence_length", 12)
    use_eng = m.get("use_engineered", False); predict_delta = m.get("predict_delta", False)
    fe = m.get("feature_engineering", {})
    feats = m["engineered_features"] if use_eng else m["features"]
    rf = m.get("random_forest", {}); seed = cfg.get("data", {}).get("seed", 42)

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv").sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)

    pids = sorted(df["patient_id"].unique().tolist())
    folds = [pids[i::K] for i in range(K)]

    fold_rows, rf_err_all, lstm_err_all = [], [], []
    for fi, test_pids in enumerate(folds):
        prep.feature_columns = list(feats)
        tr, te = prep.split_by_patient(df, test_pids)
        Xtr, ytr, atr = prep.create_sequences(tr, seq_len, HORIZON, return_anchor=True)
        Xte, yte, ate = prep.create_sequences(te, seq_len, HORIZON, return_anchor=True)
        p2 = DataPreprocessor(cfg)
        Xtr_s, Xte_s = p2.normalize_data(Xtr, Xte)
        ytr_fit = (ytr - atr) if predict_delta else ytr

        # RF (flatten)
        rfm = RandomForestRegressor(n_estimators=rf.get("n_estimators", 200),
                                    max_depth=rf.get("max_depth", 20),
                                    min_samples_split=rf.get("min_samples_split", 5),
                                    random_state=seed, n_jobs=-1)
        rfm.fit(Xtr_s.reshape(len(ytr), -1), ytr_fit)
        yp_rf = rfm.predict(Xte_s.reshape(len(yte), -1))
        yp_rf = yp_rf + ate if predict_delta else yp_rf

        # LSTM (3D) — val split 10% dari train untuk early stopping
        nval = max(1, int(0.1 * len(ytr)))
        lm = LSTMGlucoseModel(cfg)
        lm.train(Xtr_s[:-nval], ytr_fit[:-nval], Xtr_s[-nval:], ytr_fit[-nval:])
        yp_ls = lm.predict(Xte_s)
        yp_ls = yp_ls + ate if predict_delta else yp_ls

        rf_met = calculate_all_metrics(yte, yp_rf)
        ls_met = calculate_all_metrics(yte, yp_ls)
        rf_err_all.append(np.abs(yte - yp_rf)); lstm_err_all.append(np.abs(yte - yp_ls))
        fold_rows.append({"fold": fi, "test": test_pids, "n": int(len(yte)),
                          "RF_RMSE": round(rf_met["RMSE"], 2), "LSTM_RMSE": round(ls_met["RMSE"], 2),
                          "RF_A+B": round(rf_met["Clarke_A+B"], 2), "LSTM_A+B": round(ls_met["Clarke_A+B"], 2)})
        print(f"fold {fi} {test_pids}: RF RMSE {rf_met['RMSE']:.2f} | LSTM RMSE {ls_met['RMSE']:.2f}")

    rf_rmse = np.array([r["RF_RMSE"] for r in fold_rows])
    ls_rmse = np.array([r["LSTM_RMSE"] for r in fold_rows])
    # (a) fold-level paired (n=6)
    w_stat, w_p = stats.wilcoxon(rf_rmse, ls_rmse) if len(set(rf_rmse - ls_rmse)) > 1 else (float("nan"), float("nan"))
    t_stat, t_p = stats.ttest_rel(rf_rmse, ls_rmse)
    # (b) pooled per-sample |error|
    rf_e = np.concatenate(rf_err_all); ls_e = np.concatenate(lstm_err_all)
    ws_stat, ws_p = stats.wilcoxon(rf_e, ls_e)

    res = {"horizon_min": HORIZON * 5, "folds": fold_rows,
           "RF_RMSE_mean_std": [round(float(rf_rmse.mean()), 2), round(float(rf_rmse.std()), 2)],
           "LSTM_RMSE_mean_std": [round(float(ls_rmse.mean()), 2), round(float(ls_rmse.std()), 2)],
           "fold_level_paired": {"wilcoxon_p": round(float(w_p), 4), "ttest_rel_p": round(float(t_p), 4)},
           "sample_level_wilcoxon": {"p": float(ws_p), "n": int(len(rf_e)),
                                     "median_abs_err_RF": round(float(np.median(rf_e)), 2),
                                     "median_abs_err_LSTM": round(float(np.median(ls_e)), 2)}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)

    print(f"\n=== RF vs LSTM (+{HORIZON*5} mnt, 6 fold) ===")
    print(f"RF   RMSE: {rf_rmse.mean():.2f} ± {rf_rmse.std():.2f}")
    print(f"LSTM RMSE: {ls_rmse.mean():.2f} ± {ls_rmse.std():.2f}")
    print(f"Fold-level paired: Wilcoxon p={w_p:.4f} | paired-t p={t_p:.4f}")
    print(f"Sample-level Wilcoxon p={ws_p:.2e} (n={len(rf_e)}) | "
          f"median |err| RF={np.median(rf_e):.2f} LSTM={np.median(ls_e):.2f}")
    sig = "SIGNIFIKAN" if w_p < 0.05 else "TIDAK signifikan"
    print(f"\nKesimpulan fold-level: perbedaan RMSE RF vs LSTM {sig} (alpha=0.05).")
    print(f"Output -> {OUT}")


if __name__ == "__main__":
    main()
