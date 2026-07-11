"""Cross-validation LINTAS-PASIEN untuk RF — menguji robustness akurasi.

Evaluasi utama TA memakai satu split (2 pasien hold-out). Skrip ini melakukan k-fold
lintas-pasien: tiap fold menahan sekelompok pasien sebagai test, melatih pada sisanya,
lalu melaporkan distribusi metrik (mean +/- std) untuk kedua horizon. Menjawab:
"apakah 94% Clarke A+B robust, atau kebetulan pada 2 pasien tertentu?"

Preprocessing IDENTIK dgn training (engineered features, delta target, scaler fit-on-train
per fold). Output: results/eval_prediksi/crossval_rf.json + ringkasan per fold.
"""
import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
from pathlib import Path
import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
from src.utils.metrics import calculate_all_metrics

K = 6  # jumlah fold (12 pasien -> 2 pasien/test per fold)
OUT = Path("results/eval_prediksi/crossval_rf.json")


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    m = cfg["model"]
    seq_len = m.get("sequence_length", 12)
    horizons = m.get("prediction_horizons", [6, 12])
    use_eng = m.get("use_engineered", False)
    predict_delta = m.get("predict_delta", False)
    fe = m.get("feature_engineering", {})
    feats = m["engineered_features"] if use_eng else m["features"]
    rf = m.get("random_forest", {})
    seed = cfg.get("data", {}).get("seed", 42)

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv").sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)
    prep.feature_columns = list(feats)

    pids = sorted(df["patient_id"].unique().tolist())
    folds = [pids[i::K] for i in range(K)]  # bagi pasien ke K grup (round-robin)
    print(f"Pasien: {len(pids)} | fold: {K} | pasien/test per fold: {[len(f) for f in folds]}\n")

    all_res = {}
    for h in horizons:
        rows = []
        for fi, test_pids in enumerate(folds):
            prep.feature_columns = list(feats)
            train_df, test_df = prep.split_by_patient(df, test_pids)
            Xtr, ytr, atr = prep.create_sequences(train_df, seq_len, h, return_anchor=True)
            Xte, yte, ate = prep.create_sequences(test_df, seq_len, h, return_anchor=True)
            if len(yte) == 0 or len(ytr) == 0:
                continue
            prep2 = DataPreprocessor(cfg)  # scaler bersih per fold
            Xtr_s, Xte_s = prep2.normalize_data(Xtr, Xte)
            Xtr_f = Xtr_s.reshape(len(ytr), -1); Xte_f = Xte_s.reshape(len(yte), -1)
            ytr_fit = (ytr - atr) if predict_delta else ytr
            model = RandomForestRegressor(n_estimators=rf.get("n_estimators", 200),
                                          max_depth=rf.get("max_depth", 20),
                                          min_samples_split=rf.get("min_samples_split", 5),
                                          random_state=seed, n_jobs=-1)
            model.fit(Xtr_f, ytr_fit)
            yp = model.predict(Xte_f)
            yp = yp + ate if predict_delta else yp
            met = calculate_all_metrics(yte, yp)
            rows.append({"fold": fi, "test_patients": test_pids, "n": int(len(yte)),
                         "RMSE": met["RMSE"], "MAE": met["MAE"], "Clarke_A+B": met["Clarke_A+B"]})

        def stat(key):
            v = np.array([r[key] for r in rows])
            return {"mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2),
                    "min": round(float(v.min()), 2), "max": round(float(v.max()), 2)}

        all_res[f"h{h}_+{h*5}min"] = {"folds": rows,
                                      "RMSE": stat("RMSE"), "MAE": stat("MAE"),
                                      "Clarke_A+B": stat("Clarke_A+B")}
        print(f"=== Horizon +{h*5} menit ({len(rows)} fold) ===")
        print(f"  {'fold':>4} {'n':>7} {'RMSE':>7} {'MAE':>7} {'A+B%':>7}  test")
        for r in rows:
            print(f"  {r['fold']:>4} {r['n']:>7} {r['RMSE']:>7.2f} {r['MAE']:>7.2f} {r['Clarke_A+B']:>7.2f}  {r['test_patients']}")
        s = all_res[f"h{h}_+{h*5}min"]
        print(f"  MEAN±STD : RMSE {s['RMSE']['mean']}±{s['RMSE']['std']} | "
              f"MAE {s['MAE']['mean']}±{s['MAE']['std']} | "
              f"Clarke A+B {s['Clarke_A+B']['mean']}±{s['Clarke_A+B']['std']}%\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_res, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"Output -> {OUT}")


if __name__ == "__main__":
    main()
