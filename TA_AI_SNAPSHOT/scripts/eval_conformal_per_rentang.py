"""T1.3 — cakupan empiris interval konformal DIPECAH per rentang glukosa.

Subbab II.4.4 menyatakan jaminan cakupan konformal bersifat MARGINAL: berlaku rata-rata
dan tidak dijamin seragam pada setiap subkelompok, lalu menyebut bahwa rentang paling
kritis secara klinis justru paling jarang muncul pada data. Pernyataan itu harus
dibuktikan, bukan sekadar dikutip.

HIPOTESIS yang diuji: cakupan pada rentang hipoglikemia LEBIH RENDAH daripada cakupan
agregat, karena sampelnya paling sedikit. Bila terbukti, ia menguatkan laporan karena
membuktikan peringatan teoretis Subbab II.4.4 berlaku pada data ini.

Rentang memakai zona yang SAMA dengan src/constants.py, bukan zona baru:
    < 54          hipoglikemia berat
    54 - 70       hipoglikemia
    70 - 180      normal / rentang target
    180 - 250     hiperglikemia
    > 250         hiperglikemia berat

Dikerjakan untuk KEDUA horizon. Memakai faktor konformal yang sudah dikalibrasi A3
(results/eval_prediksi/conformal_h{N}.json), bukan menghitung ulang faktornya.

Keluaran: results/eval_prediksi/conformal_per_rentang_h{N}.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.conformal import conformal_factor  # noqa: E402
from src.constants import (  # noqa: E402
    GLUCOSE_CRITICAL_HIGH, GLUCOSE_CRITICAL_LOW, GLUCOSE_HIGH, GLUCOSE_LOW,
)
from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402

OUT_DIR = ROOT / "results/eval_prediksi"
EPS = 1e-6
# Ambang hipoglikemia gabungan (<70), dari src/constants.py — tidak ditulis ulang.
AMBANG_HIPO = float(GLUCOSE_LOW)

# Batas rentang diambil LANGSUNG dari src/constants.py, tidak ditulis ulang.
RENTANG = [
    ("hipoglikemia_berat", -np.inf, GLUCOSE_CRITICAL_LOW),
    ("hipoglikemia", GLUCOSE_CRITICAL_LOW, GLUCOSE_LOW),
    ("normal", GLUCOSE_LOW, GLUCOSE_HIGH),
    ("hiperglikemia", GLUCOSE_HIGH, GLUCOSE_CRITICAL_HIGH),
    ("hiperglikemia_berat", GLUCOSE_CRITICAL_HIGH, np.inf),
]


def rf_std(model, Xf):
    return np.stack([t.predict(Xf) for t in model.estimators_]).std(axis=0)


def _json_aman(o):
    """Ubah tipe numpy menjadi tipe Python untuk json.dumps."""
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"tidak dapat diserialkan: {type(o).__name__}")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Selang kepercayaan Wilson 95% untuk sebuah proporsi, dalam persen.

    Wilson, bukan normal-approksimasi (Wald). Rentang ekstrem di sini hanya berisi
    138-150 sampel dengan cakupan mendekati 0,95; pada p yang dekat batas dan n kecil,
    Wald menghasilkan selang yang terlalu sempit dan bahkan bisa melewati 100%.

    Tanpa selang ini, selisih cakupan antar-rentang tidak dapat dinilai — persis
    kekeliruan yang harus ditarik pada T1.1, ketika selisih 0,70 pp dinarasikan padahal
    simpangan bakunya sendiri 0,60 pp.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    pusat = (p + z * z / (2 * n)) / d
    lebar = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    # float() eksplisit: np.sqrt mengembalikan np.float64, dan perbandingan seperti
    # `hi < LEVEL` atasnya menghasilkan np.bool_ yang TIDAK dapat diserialkan json.
    # Sebuah jalan 22 menit pernah hilang tepat karena itu, di baris json.dumps terakhir.
    return (round(100 * max(0.0, pusat - lebar), 2), round(100 * min(1.0, pusat + lebar), 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--level", type=int, default=95, choices=[90, 95])
    args = ap.parse_args()
    H, LEVEL = args.horizon, args.level

    q = conformal_factor(H, level=LEVEL)
    if q is None:
        raise SystemExit(f"Faktor konformal horizon {H} level {LEVEL} belum ada. "
                         f"Jalankan scripts/conformal_calibration.py --horizon {H}.")

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    m = cfg["model"]
    seq_len = m.get("sequence_length", 12)
    use_eng = m.get("use_engineered", False)
    predict_delta = m.get("predict_delta", False)
    feats = m["engineered_features"] if use_eng else m["features"]
    rf = m.get("random_forest", {})
    seed = cfg.get("data", {}).get("seed", 42)
    max_gap = m.get("max_gap_steps")
    cadence = float(cfg.get("data", {}).get("sampling_interval_min", 5))

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv") \
        .sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **m.get("feature_engineering", {}))
    prep.feature_columns = list(feats)

    # Pembagian pasien IDENTIK dengan conformal_calibration.py supaya faktor q yang
    # dipakai memang dikalibrasi pada pasien yang berbeda dari pasien uji di sini.
    pids = sorted(df["patient_id"].unique().tolist())
    test_p, cal_p, train_p = pids[-2:], pids[-4:-2], pids[:-4]
    kw = {"max_gap_steps": max_gap, "source_interval_min": cadence}

    def seqs(sub):
        return prep.create_sequences(df[df["patient_id"].isin(sub)], seq_len, H,
                                     return_anchor=True, **kw)

    print(f"Horizon +{int(H * cadence)} mnt | level {LEVEL}% | q = {q}")
    print(f"latih={len(train_p)} | kalibrasi={cal_p} | uji={test_p}")

    t0 = time.time()
    Xtr, ytr, atr = seqs(train_p)
    Xte, yte, ate = seqs(test_p)
    p2 = DataPreprocessor(cfg)
    Xtr_s, _ = p2.normalize_data(Xtr, None)
    Xte_s = p2.scaler.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)
    Ftr, Fte = Xtr_s.reshape(len(ytr), -1), Xte_s.reshape(len(yte), -1)

    model = RandomForestRegressor(n_estimators=rf.get("n_estimators", 200),
                                  max_depth=rf.get("max_depth", 20),
                                  min_samples_split=rf.get("min_samples_split", 5),
                                  random_state=seed, n_jobs=-1)
    model.fit(Ftr, (ytr - atr) if predict_delta else ytr)
    yp = model.predict(Fte)
    yp = yp + ate if predict_delta else yp
    std = rf_std(model, Fte)
    lo, hi = yp - q * std, yp + q * std
    tercakup = (yte >= lo) & (yte <= hi)
    lebar = hi - lo
    print(f"pelatihan + prediksi selesai dalam {time.time() - t0:.0f} dtk\n")

    agregat = {
        "n": int(len(yte)),
        "cakupan_persen": round(100 * float(tercakup.mean()), 2),
        "ci95": wilson(int(tercakup.sum()), int(len(yte))),
        "lebar_rata2_mgdl": round(float(lebar.mean()), 1),
    }

    per_rentang = {}
    print(f"{'rentang':<22}{'n':>8}{'porsi':>8}{'cakupan':>10}{'CI95 Wilson':>18}"
          f"{'lebar':>9}{'selisih':>10}")
    for nama, lo_b, hi_b in RENTANG:
        # Rentang ditentukan oleh glukosa SEBENARNYA (yte), bukan prediksi — inilah yang
        # membuat pemecahan ini menguji sifat marginal jaminan konformal.
        mask = (yte >= lo_b) & (yte < hi_b)
        n = int(mask.sum())
        if n == 0:
            per_rentang[nama] = {"n": 0, "cakupan_persen": None, "lebar_rata2_mgdl": None,
                                 "catatan": "tidak ada sampel pada rentang ini"}
            print(f"{nama:<22}{0:>8}{'-':>8}{'-':>10}{'-':>9}{'-':>10}")
            continue
        cak = 100 * float(tercakup[mask].mean())
        lo, hi = wilson(int(tercakup[mask].sum()), n)
        per_rentang[nama] = {
            "n": n,
            "porsi_persen": round(100 * n / len(yte), 2),
            "cakupan_persen": round(cak, 2),
            "ci95": [lo, hi],
            "lebar_rata2_mgdl": round(float(lebar[mask].mean()), 1),
            "selisih_thd_agregat": round(cak - agregat["cakupan_persen"], 2),
            # Nominal terlewati hanya bila SELURUH selang berada di bawahnya. Cakupan
            # 92,97% dengan batas atas 96,1% tidak membuktikan apa pun.
            "di_bawah_nominal_meyakinkan": hi < LEVEL,
            "di_bawah_agregat_meyakinkan": hi < agregat["cakupan_persen"],
            "selang_memuat_nominal": lo <= LEVEL <= hi,
        }
        r = per_rentang[nama]
        tanda = " *" if r["di_bawah_nominal_meyakinkan"] else ""
        print(f"{nama:<22}{n:>8,}{r['porsi_persen']:>7.2f}%{cak:>9.2f}%"
              f"{f'[{lo:.1f}–{hi:.1f}]':>18}{r['lebar_rata2_mgdl']:>9.1f}"
              f"{r['selisih_thd_agregat']:>+10.2f}{tanda}")

    mask_hipo = yte < GLUCOSE_LOW
    n_hipo = int(mask_hipo.sum())
    k_hipo = int(tercakup[mask_hipo].sum()) if n_hipo else 0
    cak_hipo = (100 * k_hipo / n_hipo) if n_hipo else None
    ci_hipo = wilson(k_hipo, n_hipo) if n_hipo else (None, None)
    # Hipotesis dinyatakan terbukti hanya bila SELURUH selang hipoglikemia berada di
    # bawah cakupan agregat. Sekadar titik estimasi yang lebih rendah tidak cukup.
    hipotesis_terbukti = (cak_hipo is not None
                          and ci_hipo[1] < agregat["cakupan_persen"])
    hipotesis_arah_saja = (cak_hipo is not None
                           and cak_hipo < agregat["cakupan_persen"])

    out = {
        "catatan": ("Cakupan empiris interval konformal dipecah per rentang glukosa "
                    "SEBENARNYA. Menguji pernyataan Subbab II.4.4 bahwa jaminan konformal "
                    "bersifat marginal dan tidak seragam antar-subkelompok."),
        "horizon_steps": H, "horizon_min": int(H * cadence),
        "level_nominal_persen": LEVEL, "faktor_q": q,
        "sumber_faktor": f"results/eval_prediksi/conformal_h{H}.json (A3)",
        "batas_rentang": "src/constants.py (54 / 70 / 180 / 250)",
        "pasien_uji": test_p,
        "agregat": agregat,
        "per_rentang": per_rentang,
        "hipotesis": {
            "pernyataan": ("Cakupan pada rentang hipoglikemia lebih rendah daripada "
                           "cakupan agregat karena sampelnya paling sedikit."),
            "ditetapkan": "di muka, sebelum satu pun angka dilihat",
            "n_hipoglikemia_gabungan": n_hipo,
            "cakupan_hipoglikemia_gabungan": round(cak_hipo, 2) if cak_hipo else None,
            "ci95_hipoglikemia": list(ci_hipo),
            "cakupan_agregat": agregat["cakupan_persen"],
            "ci95_agregat": agregat["ci95"],
            "TERBUKTI": bool(hipotesis_terbukti),
            "arah_saja_tanpa_selang": bool(hipotesis_arah_saja),
            "kriteria": ("TERBUKTI hanya bila batas ATAS selang Wilson hipoglikemia "
                         "berada di bawah cakupan agregat. Titik estimasi yang lebih "
                         "rendah saja tidak cukup: pada n<800 selangnya beberapa poin "
                         "persen lebarnya."),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"conformal_per_rentang_h{H}.json"
    # Penjaga: tipe numpy apa pun yang lolos diubah, bukan menggagalkan penulisan di
    # ujung skrip berdurasi 11 menit. Penulisan hasil tidak boleh menjadi titik gagal.
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=_json_aman),
                 encoding="utf-8")

    print(f"\n* = seluruh selang di bawah nominal {LEVEL}%")
    print(f"\nAGREGAT: cakupan {agregat['cakupan_persen']}% "
          f"[{agregat['ci95'][0]}–{agregat['ci95'][1]}] (nominal {LEVEL}%), "
          f"lebar {agregat['lebar_rata2_mgdl']} mg/dL, n={agregat['n']:,}")
    if cak_hipo is not None:
        print(f"HIPOGLIKEMIA gabungan (<{AMBANG_HIPO:.0f}): cakupan {cak_hipo:.2f}% "
              f"[{ci_hipo[0]}–{ci_hipo[1]}], n={n_hipo:,} "
              f"({100 * n_hipo / len(yte):.2f}% data)")
        if hipotesis_terbukti:
            print(f"HIPOTESIS TERBUKTI: batas atas selang hipoglikemia ({ci_hipo[1]}%) "
                  f"di bawah cakupan agregat ({agregat['cakupan_persen']}%)")
        elif hipotesis_arah_saja:
            print(f"HIPOTESIS TIDAK TERBUKTI: arahnya benar tetapi selang hipoglikemia "
                  f"[{ci_hipo[0]}–{ci_hipo[1]}] masih memuat cakupan agregat "
                  f"({agregat['cakupan_persen']}%)")
        else:
            print(f"HIPOTESIS SALAH ARAH: cakupan hipoglikemia {cak_hipo:.2f}% justru "
                  f"TIDAK lebih rendah daripada agregat {agregat['cakupan_persen']}%")
    print(f"\nDisimpan ke {p}")


if __name__ == "__main__":
    main()
