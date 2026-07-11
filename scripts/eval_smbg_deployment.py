"""Evaluasi skenario deployment SMBG: prediksi glukosa dari logbook/finger_stick nyata.

Motivasi: seluruh angka prediksi pada Bab VI dihitung pada kanal CGM (cadence 5 menit),
padahal klaim penerapan sistem adalah pada pasien tanpa CGM yang hanya memiliki data
logbook/SMBG. Skrip ini menutup celah tersebut dengan mengevaluasi model pada timeline
finger_stick NYATA OhioT1DM (data/raw/ohio_t1dm_smbg.csv), bukan hasil downsampling CGM.

Desain yang penting untuk validitas:

1. Timeline finger_stick tidak beraturan (median jarak ~125 menit). Karena itu horizon
   TIDAK dapat dinyatakan dalam "langkah"; horizon ditetapkan dalam MENIT (+30/+60) dan
   nilai kebenaran (ground truth) diambil dari kanal CGM pada t+h, yang merupakan kadar
   glukosa pasien yang sebenarnya pada saat itu.

2. IOB/COB memakai peluruhan eksponensial yang mengasumsikan interval seragam, sehingga
   TIDAK boleh dihitung langsung di atas timeline finger_stick yang tak beraturan.
   Keduanya dihitung pada grid 5-menit (tempat seluruh event insulin/karbohidrat tercatat
   beserta waktunya) lalu disampel pada waktu pembacaan finger_stick. Ini juga realistis:
   logbook mencatat waktu penyuntikan insulin dan waktu makan.

3. Masukan model HANYA berasal dari data yang dimiliki pasien tanpa CGM: pembacaan
   finger_stick, event insulin/karbohidrat, aktivitas, dan jam. Kanal CGM dipakai
   semata-mata sebagai ground truth target, tidak pernah sebagai fitur.

Keluaran: results/eval_prediksi/smbg_deployment.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402

FEATURES = ["glucose", "glucose_delta", "iob", "cob", "activity", "hour_sin", "hour_cos"]
SEQ_LEN = 6          # look-back 6 pembacaan SMBG (Tabel V.1)
HORIZONS_MIN = [30, 60]
TOLERANCE_MIN = 5    # toleransi pencocokan ground truth CGM
SEED = 42


def build_smbg_frame(cgm: pd.DataFrame, smbg: pd.DataFrame) -> pd.DataFrame:
    """Gabungkan anchor finger_stick dengan IOB/COB dari grid 5-menit + target CGM."""
    # feature_columns = kolom MENTAH; fitur engineered dibuat oleh engineer_features()
    pre = DataPreprocessor({"model": {"features": ["glucose", "carbs", "insulin", "activity"]}})
    cgm = pre.handle_missing_values(cgm)
    cgm = pre.engineer_features(cgm)  # iob/cob valid: grid 5-menit seragam

    rows = []
    for pid, s in smbg.groupby("patient_id", sort=False):
        c = cgm[cgm.patient_id == pid].sort_values("timestamp").reset_index(drop=True)
        if c.empty:
            continue
        s = s.sort_values("timestamp").reset_index(drop=True)

        # IOB/COB/activity pada saat pembacaan finger_stick (dari grid 5-menit)
        ctx = pd.merge_asof(
            s[["timestamp"]], c[["timestamp", "iob", "cob", "activity"]],
            on="timestamp", direction="nearest",
            tolerance=pd.Timedelta(minutes=TOLERANCE_MIN),
        )
        f = s[["timestamp", "patient_id", "glucose"]].copy()
        f[["iob", "cob", "activity"]] = ctx[["iob", "cob", "activity"]].values

        # tren dari pembacaan finger_stick sebelumnya (bukan dari CGM)
        f["glucose_delta"] = f["glucose"].diff().fillna(0.0)
        f["gap_min"] = f["timestamp"].diff().dt.total_seconds().div(60)
        hour = f["timestamp"].dt.hour + f["timestamp"].dt.minute / 60.0
        f["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
        f["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

        # ground truth: glukosa sebenarnya (CGM) pada t + h menit
        for h in HORIZONS_MIN:
            tgt = pd.merge_asof(
                pd.DataFrame({"timestamp": f["timestamp"] + pd.Timedelta(minutes=h)}),
                c[["timestamp", "glucose"]].rename(columns={"glucose": f"y{h}"}),
                on="timestamp", direction="nearest",
                tolerance=pd.Timedelta(minutes=TOLERANCE_MIN),
            )
            f[f"y{h}"] = tgt[f"y{h}"].values
        rows.append(f)

    return pd.concat(rows, ignore_index=True)


def make_windows(df: pd.DataFrame, horizon: int):
    """Jendela look-back SEQ_LEN pembacaan SMBG → target Δglukosa ke t+horizon."""
    X, y, anchor, pid_out = [], [], [], []
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        feat = g[FEATURES].to_numpy(float)
        tgt = g[f"y{horizon}"].to_numpy(float)
        glu = g["glucose"].to_numpy(float)
        for i in range(SEQ_LEN - 1, len(g)):
            if not np.isfinite(tgt[i]):
                continue
            win = feat[i - SEQ_LEN + 1: i + 1]
            if not np.isfinite(win).all():
                continue
            X.append(win.ravel())            # RF: jendela di-flatten
            y.append(tgt[i] - glu[i])        # target Δ (konsisten Persamaan V.4)
            anchor.append(glu[i])
            pid_out.append(pid)
    return np.array(X), np.array(y), np.array(anchor), np.array(pid_out)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    rf_cfg = cfg["model"]["random_forest"]

    cgm = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    smbg = pd.read_csv(ROOT / "data/raw/ohio_t1dm_smbg.csv", parse_dates=["timestamp"])

    df = build_smbg_frame(cgm, smbg)
    patients = sorted(df.patient_id.unique())
    test_patients = patients[-2:]  # split per pasien, sama seperti hold-out CGM
    print(f"Pasien uji: {test_patients}")

    gap = df["gap_min"].dropna()
    out = {
        "catatan": "Masukan hanya finger_stick + event logbook; CGM dipakai hanya sebagai ground truth.",
        "n_smbg_readings": int(len(df)),
        "test_patients": test_patients,
        "smbg_gap_min": {
            "median": round(float(gap.median()), 1),
            "mean": round(float(gap.mean()), 1),
        },
        "horizons": {},
    }

    for h in HORIZONS_MIN:
        X, y, anchor, pid = make_windows(df, h)
        tr = ~np.isin(pid, test_patients)
        te = ~tr
        model = RandomForestRegressor(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            min_samples_split=rf_cfg["min_samples_split"],
            random_state=SEED,
            n_jobs=-1,
        )
        model.fit(X[tr], y[tr])

        pred = anchor[te] + model.predict(X[te])   # rekonstruksi nilai absolut
        true = anchor[te] + y[te]
        m = calculate_all_metrics(true, pred)

        # baseline persistence: glukosa tidak berubah dari pembacaan terakhir
        m_persist = calculate_all_metrics(true, anchor[te])

        out["horizons"][f"+{h}"] = {
            "n_train": int(tr.sum()),
            "n_test": int(te.sum()),
            "RMSE": round(m["RMSE"], 2),
            "MAE": round(m["MAE"], 2),
            "MAPE": round(m["MAPE"], 2),
            "Clarke_A": round(m["Clarke_A"], 2),
            "Clarke_A+B": round(m["Clarke_A"] + m["Clarke_B"], 2),
            "persistence_RMSE": round(m_persist["RMSE"], 2),
            "persistence_Clarke_A+B": round(
                m_persist["Clarke_A"] + m_persist["Clarke_B"], 2
            ),
        }
        print(f"+{h} mnt | n_test={te.sum():4d} | RMSE {m['RMSE']:.2f} "
              f"| A+B {m['Clarke_A'] + m['Clarke_B']:.2f}% "
              f"| persistence RMSE {m_persist['RMSE']:.2f}")

    dest = ROOT / "results/eval_prediksi/smbg_deployment.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
