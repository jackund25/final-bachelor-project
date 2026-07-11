"""Perbaikan deteksi hipoglikemia — dua pendekatan berbasis bukti.

Baseline: RF produksi hanya mendeteksi ~14% hipo (ambang <70). Skrip ini menguji:
  A. Penyesuaian AMBANG peringatan (post-hoc, tanpa latih ulang): sweep 70..95 mg/dL,
     tampilkan trade-off sensitivitas/spesifisitas/PPV.
  B. Latih ulang RF dengan SAMPLE WEIGHTING (bobot lebih pada wilayah hipo/near-hipo)
     agar model memprioritaskan ekor bawah — lalu bandingkan sensitivitas & RMSE.

Reproduksi split & preprocessing IDENTIK dgn training. Output: results/eval_prediksi/hypo_improve.json
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

HYPO = 70.0
OUT = Path("results/eval_prediksi/hypo_improve.json")


def hypo_metrics(y_true, flag_pred):
    """flag_pred = boolean 'diprediksi hipo'."""
    th = y_true < HYPO
    tp = int(np.sum(th & flag_pred)); fn = int(np.sum(th & ~flag_pred))
    fp = int(np.sum(~th & flag_pred)); tn = int(np.sum(~th & ~flag_pred))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return {"sens": round(sens * 100, 1), "spec": round(spec * 100, 1),
            "ppv": round(ppv * 100, 1), "TP": tp, "FN": fn, "FP": fp}


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    mcfg = cfg["model"]
    seq_len = mcfg.get("sequence_length", 12)
    horizon = mcfg.get("default_horizon", 6)
    use_eng = mcfg.get("use_engineered", False)
    predict_delta = mcfg.get("predict_delta", False)
    fe = mcfg.get("feature_engineering", {})
    feats = mcfg["engineered_features"] if use_eng else mcfg["features"]
    rf = mcfg.get("random_forest", {})
    seed = cfg.get("data", {}).get("seed", 42)

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv").sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **fe)
    prep.feature_columns = list(feats)

    pids = sorted(df["patient_id"].unique().tolist())
    train_df, test_df = prep.split_by_patient(df, pids[-2:])
    Xtr, ytr, atr = prep.create_sequences(train_df, seq_len, horizon, return_anchor=True)
    Xte, yte, ate = prep.create_sequences(test_df, seq_len, horizon, return_anchor=True)
    Xtr_s, Xte_s = prep.normalize_data(Xtr, Xte)
    Xtr_f = Xtr_s.reshape(len(ytr), -1); Xte_f = Xte_s.reshape(len(yte), -1)
    ytr_fit = (ytr - atr) if predict_delta else ytr

    def make_rf():
        return RandomForestRegressor(n_estimators=rf.get("n_estimators", 200),
                                     max_depth=rf.get("max_depth", 20),
                                     min_samples_split=rf.get("min_samples_split", 5),
                                     random_state=seed, n_jobs=-1)

    def predict(model):
        out = model.predict(Xte_f)
        return out + ate if predict_delta else out

    # ---- Baseline (unweighted, reproduksi produksi) ----
    base = make_rf(); base.fit(Xtr_f, ytr_fit)
    yb = predict(base)
    rmse_b = float(np.sqrt(np.mean((yte - yb) ** 2)))

    # ---- A. Sweep ambang pada baseline ----
    sweep = {}
    for thr in [70, 75, 80, 85, 90, 95]:
        sweep[thr] = hypo_metrics(yte, yb < thr)

    # ---- B. Latih ulang dengan sample weighting ----
    results_weight = {}
    for W in [3.0, 6.0, 10.0]:
        w = np.where(ytr < 80.0, W, 1.0)   # bobot wilayah hipo/near-hipo (true future <80)
        m = make_rf(); m.fit(Xtr_f, ytr_fit, sample_weight=w)
        yw = predict(m)
        rmse_w = float(np.sqrt(np.mean((yte - yw) ** 2)))
        results_weight[f"w{int(W)}_thr70"] = {**hypo_metrics(yte, yw < HYPO), "rmse": round(rmse_w, 2)}
        results_weight[f"w{int(W)}_thr80"] = {**hypo_metrics(yte, yw < 80.0), "rmse": round(rmse_w, 2)}

    res = {"n_test": int(len(yte)), "n_hypo": int(np.sum(yte < HYPO)),
           "baseline": {"rmse": round(rmse_b, 2), "hypo_thr70": hypo_metrics(yte, yb < HYPO)},
           "A_threshold_sweep_baseline": {str(k): v for k, v in sweep.items()},
           "B_sample_weighting": results_weight}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)

    print(f"n_test={len(yte)} | kejadian hipo={int(np.sum(yte < HYPO))} | RMSE baseline={rmse_b:.2f}\n")
    print("A. SWEEP AMBANG (baseline, tanpa latih ulang):")
    print(f"  {'ambang':>7} {'sens%':>6} {'spec%':>6} {'PPV%':>6}")
    for thr, m in sweep.items():
        print(f"  {thr:>7} {m['sens']:>6} {m['spec']:>6} {m['ppv']:>6}")
    print("\nB. SAMPLE WEIGHTING (latih ulang; bobot true<80):")
    print(f"  {'setting':>12} {'sens%':>6} {'spec%':>6} {'PPV%':>6} {'RMSE':>6}")
    for k, m in results_weight.items():
        print(f"  {k:>12} {m['sens']:>6} {m['spec']:>6} {m['ppv']:>6} {m['rmse']:>6}")
    print(f"\nOutput -> {OUT}")


if __name__ == "__main__":
    main()
