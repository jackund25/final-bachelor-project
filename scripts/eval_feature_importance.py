"""Bukti klaim "prediksi multimodal": kontribusi insulin & karbohidrat pada prediksi.

Laporan mengklaim bahwa dengan representasi berbasis fisiologi (IOB/COB), kontribusi
gabungan insulin dan karbohidrat naik dari ~3,4% (fitur mentah per-catatan) menjadi ~22,3%
(fitur engineered) pada horizon +30 menit. Skrip ini menghitung kedua angka tersebut secara
langsung dari feature_importances_ Random Forest, sehingga klaim itu dapat direproduksi.

Dua model dilatih pada split per-pasien yang sama:
  A. Fitur MENTAH      : [glucose, carbs, insulin, activity]        -> target glukosa absolut
  B. Fitur ENGINEERED  : [glucose, glucose_delta, iob, cob,
                          activity, hour_sin, hour_cos]             -> target delta glukosa

Karena Random Forest memakai jendela look-back yang di-flatten, importance dijumlahkan
lintas seluruh langkah waktu untuk tiap fitur.

Keluaran: results/eval_prediksi/feature_importance.json
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

HORIZON = 6  # +30 menit
SEED = 42

RAW_FEATURES = ["glucose", "carbs", "insulin", "activity"]
ENGINEERED_FEATURES = [
    "glucose", "glucose_delta", "iob", "cob", "activity", "hour_sin", "hour_cos",
]
# Fitur yang merepresentasikan insulin & karbohidrat pada tiap skema
RAW_MULTIMODAL = ["carbs", "insulin"]
ENG_MULTIMODAL = ["iob", "cob"]


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def permutation_importance_grouped(model, X, y_true, anchor, features, predict_delta,
                                   n_repeats=5, seed=SEED) -> dict:
    """Permutation importance per FITUR (bukan per kolom), pada data HOLD-OUT.

    Mengapa ini ditambahkan di samping ``feature_importances_``:

    1. **MDI bias ke fitur bernilai-unik banyak.** Gini importance memberi keuntungan
       sistematis kepada fitur kontinu yang titik pisahnya banyak. ``glucose`` dan
       ``glucose_delta`` jauh lebih beragam daripada ``iob``/``cob``, yang merupakan
       akumulasi meluruh dan sering bernilai nol. Bias itu bekerja **tepat melawan** klaim
       yang sedang diuji, sehingga MDI sendirian tidak cukup untuk membuktikannya.
    2. **MDI dihitung pada data LATIH.** Ia mengukur apa yang dipakai model untuk memecah
       simpul, bukan apa yang benar-benar membantu pada pasien yang belum pernah dilihat.

    Permutasi dilakukan BERKELOMPOK: kedua belas langkah waktu satu fitur diacak bersama.
    Mengacak satu langkah saja tidak menghapus informasi fitur itu — sebelas langkah
    lainnya masih memuatnya, dan importance-nya akan tampak nyaris nol untuk semua fitur.

    ``anchor`` tidak ikut diacak. Ia bukan masukan model melainkan offset rekonstruksi,
    sehingga membiarkannya utuh justru yang mengisolasi peran fitur di dalam model.
    """
    n, seq, n_feat = X.shape
    rng = np.random.default_rng(seed)

    def prediksi(Xa):
        p = model.predict(Xa.reshape(len(Xa), seq * n_feat))
        return p + anchor if predict_delta else p

    dasar = _rmse(y_true, prediksi(X))
    hasil = {}
    for j, nama in enumerate(features):
        kenaikan = []
        for _ in range(n_repeats):
            Xp = X.copy()
            urut = rng.permutation(n)
            Xp[:, :, j] = X[urut, :, j]     # seluruh langkah waktu fitur ini diacak bersama
            kenaikan.append(_rmse(y_true, prediksi(Xp)) - dasar)
        hasil[nama] = {"kenaikan_rmse": round(float(np.mean(kenaikan)), 4),
                       "sd": round(float(np.std(kenaikan)), 4)}
    total = sum(max(v["kenaikan_rmse"], 0.0) for v in hasil.values()) or 1.0
    for v in hasil.values():
        v["persen"] = round(100.0 * max(v["kenaikan_rmse"], 0.0) / total, 2)
    return {"rmse_dasar": round(dasar, 4), "n_uji": int(n), "n_repeats": n_repeats,
            "per_fitur": hasil}


def run(features: list[str], predict_delta: bool, cfg: dict, engineered: bool) -> dict:
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    if engineered:
        df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(features)

    patients = sorted(df["patient_id"].unique())
    train_df, test_df = pre.split_by_patient(df, patients[-2:])

    # Segmentasi jeda sensor WAJIB sama dengan pelatihan produksi (Tugas 5). Tanpa ini,
    # kontribusi fitur dihitung atas model yang dilatih pada himpunan jendela yang BERBEDA
    # dari model yang benar-benar dipakai sistem. Cacat yang sama pernah ditemukan pada
    # crossval_rf_vs_lstm.py (T1.1) dan conformal_calibration.py (A3): setiap skrip yang
    # memanggil create_sequences() sendiri melewatkan segmentasi.
    max_gap_steps = mc.get("max_gap_steps")
    if max_gap_steps is None:
        raise SystemExit("config.model.max_gap_steps tidak ada; kontribusi fitur tidak "
                         "akan sepadan dengan model produksi.")
    cadence_min = float(cfg.get("data", {}).get("sampling_interval_min", 5))
    X, y, anchor = pre.create_sequences(
        train_df, mc["sequence_length"], HORIZON, return_anchor=True,
        max_gap_steps=max_gap_steps, source_interval_min=cadence_min,
    )
    n, seq, n_feat = X.shape
    target = (y - anchor) if predict_delta else y

    model = RandomForestRegressor(
        n_estimators=mc["random_forest"]["n_estimators"],
        max_depth=mc["random_forest"]["max_depth"],
        min_samples_split=mc["random_forest"]["min_samples_split"],
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X.reshape(n, seq * n_feat), target)

    # importance dijumlahkan lintas langkah waktu, lalu dinormalisasi ke persen
    imp = model.feature_importances_.reshape(seq, n_feat).sum(axis=0)
    imp = 100.0 * imp / imp.sum()
    mdi = {f: round(float(v), 2) for f, v in zip(features, imp)}

    # Ukuran kedua pada pasien HOLD-OUT, memakai jendela tersegmentasi yang sama.
    Xte, yte, ate = pre.create_sequences(
        test_df, mc["sequence_length"], HORIZON, return_anchor=True,
        max_gap_steps=max_gap_steps, source_interval_min=cadence_min,
    )
    perm = permutation_importance_grouped(model, Xte, yte, ate, features, predict_delta)
    perm["pasien_uji"] = list(patients[-2:])
    return {"mdi": mdi, "permutasi": perm}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    raw = run(RAW_FEATURES, predict_delta=False, cfg=cfg, engineered=False)
    eng = run(ENGINEERED_FEATURES, predict_delta=True, cfg=cfg, engineered=True)

    def jumlah(blok, kunci_fitur, ukuran):
        if ukuran == "mdi":
            return round(sum(blok["mdi"][f] for f in kunci_fitur), 2)
        return round(sum(blok["permutasi"]["per_fitur"][f]["persen"] for f in kunci_fitur), 2)

    ringkas = {}
    for ukuran in ("mdi", "permutasi"):
        r = jumlah(raw, RAW_MULTIMODAL, ukuran)
        e = jumlah(eng, ENG_MULTIMODAL, ukuran)
        ringkas[ukuran] = {"mentah_%": r, "engineered_%": e,
                           "kelipatan": round(e / r, 2) if r else None}

    out = {
        "horizon_min": HORIZON * 5,
        "catatan": (
            "Importance dijumlahkan lintas langkah look-back, dinormalisasi ke persen. "
            "Kontribusi multimodal = insulin + karbohidrat. Jendela TERSEGMENTASI "
            "(max_gap_steps), sama dengan pelatihan produksi."),
        "dua_ukuran_mengapa": (
            "MDI (feature_importances_) dihitung pada data LATIH dan bias ke fitur "
            "bernilai-unik banyak; bias itu menguntungkan glucose/glucose_delta atas "
            "iob/cob, yaitu bekerja melawan klaim yang diuji. Permutation importance "
            "dihitung pada pasien HOLD-OUT dan mengukur kenaikan RMSE ketika fitur "
            "dirusak, sehingga tidak terkena kedua masalah itu. Keduanya dilaporkan; "
            "bila keduanya sepakat, klaimnya jauh lebih kuat daripada MDI sendirian."),
        "fitur_mentah": raw,
        "fitur_engineered": eng,
        "kontribusi_insulin_karbohidrat": ringkas,
    }

    def cetak(nama, blok, kunci):
        print(f"\n{nama}")
        print(f"  {'fitur':<16}{'MDI %':>9}{'perm %':>9}{'naik RMSE':>11}")
        for f in blok["mdi"]:
            p = blok["permutasi"]["per_fitur"][f]
            tanda = " *" if f in kunci else ""
            print(f"  {f:<16}{blok['mdi'][f]:>9.2f}{p['persen']:>9.2f}"
                  f"{p['kenaikan_rmse']:>11.3f}{tanda}")
        print(f"  RMSE dasar hold-out: {blok['permutasi']['rmse_dasar']:.3f} "
              f"(n={blok['permutasi']['n_uji']:,}, pasien {blok['permutasi']['pasien_uji']})")

    cetak("FITUR MENTAH", raw, RAW_MULTIMODAL)
    cetak("FITUR ENGINEERED", eng, ENG_MULTIMODAL)
    print(f"\n{'ukuran':<14}{'mentah':>10}{'engineered':>13}{'kelipatan':>12}")
    for u, v in ringkas.items():
        k = f"{v['kelipatan']}x" if v["kelipatan"] else "-"
        print(f"{u:<14}{v['mentah_%']:>9.2f}%{v['engineered_%']:>12.2f}%{k:>12}")

    dest = ROOT / "results/eval_prediksi/feature_importance.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDisimpan ke {dest}")


if __name__ == "__main__":
    main()
