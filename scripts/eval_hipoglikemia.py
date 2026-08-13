"""T2.1 — galat khusus HIPOGLIKEMIA untuk RF dan LSTM, dengan sebaran antar-fold.

MENGAPA PERLU TERPISAH DARI CROSSVAL BIASA
------------------------------------------
RMSE agregat menyembunyikan kegagalan yang paling penting secara klinis. Hipoglikemia
menempati sekitar 3% sampel; model yang tidak pernah memprediksi <70 sama sekali tetap
mendapat RMSE yang bagus. Angka agregat pada `crossval_rf_vs_lstm_h*.json` karena itu
TIDAK dapat menjawab "seberapa baik sistem ini pada kejadian yang justru ingin dicegah".

APA YANG SUDAH ADA DAN MENGAPA TIDAK CUKUP
------------------------------------------
`results/eval_prediksi/hypo_safety_uncertainty.json` sudah memuat sensitivitas 14,0% untuk
RF. Tiga alasan angka itu tidak memadai untuk Bab VI:

1. **Hanya RF.** Pemilihan model RF-vs-LSTM belum final, dan justru pada hipoglikemia-lah
   perbedaan keduanya paling mungkin berarti.
2. **Hanya satu horizon.**
3. **n_test = 27.193**, sedangkan set uji tersegmentasi h6 berisi 26.445 jendela. Angka itu
   dihitung SEBELUM segmentasi jeda sensor (Tugas 5), sehingga tidak sepadan dengan model
   produksi. Cacat yang sama dengan yang ditemukan pada T1.1 dan A3.

Skrip ini memakai struktur 6 fold lintas-pasien yang SAMA dengan `crossval_rf_vs_lstm.py`,
sehingga sebaran antar-fold ikut terukur. Itu bukan hiasan: pada T1.1 sebuah klaim selisih
0,70 pp harus ditarik karena simpangan bakunya sendiri 0,60 pp. Tanpa SD, selisih apa pun
antara RF dan LSTM di sini tidak dapat dinilai.

ARAH GALAT DIPISAH DARI BESARNYA
--------------------------------
Pada hipoglikemia, galat +30 mg/dL dan -30 mg/dL tidak sama bahayanya. Yang pertama
menyembunyikan kejadian rendah; yang kedua memicu peringatan yang tidak perlu. MAE
memperlakukan keduanya identik, jadi `bias` (rerata galat bertanda) dilaporkan terpisah.

Keluaran: results/eval_prediksi/hipoglikemia_h{N}.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import GLUCOSE_CRITICAL_LOW, GLUCOSE_LOW  # noqa: E402
from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.models.lstm_model import LSTMGlucoseModel  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402

K = 6
OUT_DIR = ROOT / "results/eval_prediksi"
AMBANG = float(GLUCOSE_LOW)          # 70 mg/dL
AMBANG_BERAT = float(GLUCOSE_CRITICAL_LOW)  # 54 mg/dL


def metrik_hipo(yte: np.ndarray, yp: np.ndarray) -> dict:
    """Metrik yang seluruhnya dihitung dari sudut pandang kejadian hipoglikemia."""
    benar_hipo = yte < AMBANG
    duga_hipo = yp < AMBANG
    n_hipo = int(benar_hipo.sum())

    tp = int((benar_hipo & duga_hipo).sum())
    fn = int((benar_hipo & ~duga_hipo).sum())
    fp = int((~benar_hipo & duga_hipo).sum())
    tn = int((~benar_hipo & ~duga_hipo).sum())

    def bagi(a, b):
        return round(100.0 * a / b, 3) if b else None

    sens = bagi(tp, tp + fn)
    ppv = bagi(tp, tp + fp)
    f1 = (round(2 * sens * ppv / (sens + ppv), 3)
          if sens and ppv and (sens + ppv) > 0 else 0.0)

    out = {
        "n_total": int(len(yte)),
        "n_hipo_benar": n_hipo,
        "porsi_hipo_persen": bagi(n_hipo, len(yte)),
        "n_hipo_berat_benar": int((yte < AMBANG_BERAT).sum()),
        "deteksi": {
            "sensitivitas_persen": sens, "spesifisitas_persen": bagi(tn, tn + fp),
            "ppv_persen": ppv, "f1_persen": f1,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        },
    }

    if n_hipo:
        e = yp[benar_hipo] - yte[benar_hipo]
        out["galat_pada_hipo"] = {
            "rmse": round(float(np.sqrt(np.mean(e ** 2))), 3),
            "mae": round(float(np.mean(np.abs(e))), 3),
            # Bertanda: POSITIF berarti model menduga LEBIH TINGGI daripada kenyataan,
            # yaitu arah yang menyembunyikan kejadian rendah.
            "bias": round(float(np.mean(e)), 3),
            "rerata_glukosa_benar": round(float(np.mean(yte[benar_hipo])), 2),
            "rerata_glukosa_diduga": round(float(np.mean(yp[benar_hipo])), 2),
        }
        # Terlewat ke rentang yang justru menenangkan: benar-benar hipo, tetapi diduga
        # berada di rentang target. Ini kelas kesalahan paling berbahaya.
        out["terlewat_ke_rentang_target"] = int((benar_hipo & (yp >= AMBANG)).sum())
        berat = yte < AMBANG_BERAT
        out["hipo_berat_terlewat"] = int((berat & (yp >= AMBANG)).sum())
        try:
            mh = calculate_all_metrics(yte[benar_hipo], yp[benar_hipo])
            out["clarke_pada_hipo"] = {k: round(float(mh[k]), 3)
                                       for k in ("Clarke_A", "Clarke_B", "Clarke_C",
                                                 "Clarke_D", "Clarke_E") if k in mh}
        except Exception:  # noqa: BLE001
            out["clarke_pada_hipo"] = None
    else:
        out["galat_pada_hipo"] = None
        out["terlewat_ke_rentang_target"] = 0
        out["hipo_berat_terlewat"] = 0
        out["clarke_pada_hipo"] = None

    # Bias keseluruhan, sebagai pembanding: kalau bias pada hipo jauh lebih positif
    # daripada bias keseluruhan, itu tanda penyusutan ke tengah, bukan bias global.
    out["bias_keseluruhan"] = round(float(np.mean(yp - yte)), 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()
    HORIZON = args.horizon

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("model", {}).setdefault("lstm", {})["verbose"] = 0
    m = cfg["model"]
    seq_len = m.get("sequence_length", 12)
    use_eng = m.get("use_engineered", False)
    predict_delta = m.get("predict_delta", False)
    feats = m["engineered_features"] if use_eng else m["features"]
    rf = m.get("random_forest", {})
    seed = cfg.get("data", {}).get("seed", 42)
    max_gap_steps = m.get("max_gap_steps")
    cadence_min = float(cfg.get("data", {}).get("sampling_interval_min", 5))
    if max_gap_steps is None:
        raise SystemExit("config.model.max_gap_steps tidak ada — hasil tidak akan sepadan "
                         "dengan model produksi.")

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv") \
        .sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if use_eng:
        df = prep.engineer_features(df, **m.get("feature_engineering", {}))

    pids = sorted(df["patient_id"].unique().tolist())
    folds = [pids[i::K] for i in range(K)]  # struktur IDENTIK dengan crossval_rf_vs_lstm.py

    print(f"T2.1 — galat khusus hipoglikemia | horizon +{HORIZON * cadence_min:.0f} mnt")
    print(f"ambang {AMBANG:.0f} mg/dL (berat {AMBANG_BERAT:.0f}) | "
          f"max_gap_steps={max_gap_steps} | {K} fold lintas-pasien\n")

    baris = []
    for fi, test_pids in enumerate(folds):
        t0 = time.time()
        prep.feature_columns = list(feats)
        tr, te = prep.split_by_patient(df, test_pids)
        kw = {"max_gap_steps": max_gap_steps, "source_interval_min": cadence_min}
        Xtr, ytr, atr = prep.create_sequences(tr, seq_len, HORIZON, return_anchor=True, **kw)
        Xte, yte, ate = prep.create_sequences(te, seq_len, HORIZON, return_anchor=True, **kw)
        p2 = DataPreprocessor(cfg)
        Xtr_s, Xte_s = p2.normalize_data(Xtr, Xte)
        ytr_fit = (ytr - atr) if predict_delta else ytr

        rfm = RandomForestRegressor(n_estimators=rf.get("n_estimators", 200),
                                    max_depth=rf.get("max_depth", 20),
                                    min_samples_split=rf.get("min_samples_split", 5),
                                    random_state=seed, n_jobs=-1)
        rfm.fit(Xtr_s.reshape(len(ytr), -1), ytr_fit)
        yp_rf = rfm.predict(Xte_s.reshape(len(yte), -1))
        yp_rf = yp_rf + ate if predict_delta else yp_rf

        nval = max(1, int(0.1 * len(ytr)))
        lm = LSTMGlucoseModel(cfg)
        lm.train(Xtr_s[:-nval], ytr_fit[:-nval], Xtr_s[-nval:], ytr_fit[-nval:])
        yp_ls = lm.predict(Xte_s)
        yp_ls = yp_ls + ate if predict_delta else yp_ls

        r_rf, r_ls = metrik_hipo(yte, yp_rf), metrik_hipo(yte, yp_ls)
        baris.append({"fold": fi, "test": test_pids, "RF": r_rf, "LSTM": r_ls})
        print(f"  fold {fi} {test_pids}: n_hipo={r_rf['n_hipo_benar']:>5} "
              f"({r_rf['porsi_hipo_persen']:.2f}%) | "
              f"sens RF {r_rf['deteksi']['sensitivitas_persen']:>5.1f}% "
              f"LSTM {r_ls['deteksi']['sensitivitas_persen']:>5.1f}% | "
              f"bias-hipo RF {r_rf['galat_pada_hipo']['bias']:>+6.1f} "
              f"LSTM {r_ls['galat_pada_hipo']['bias']:>+6.1f} | "
              f"{time.time() - t0:.0f} dtk", flush=True)

    def kumpul(model, *jalur):
        nilai = []
        for b in baris:
            v = b[model]
            for k in jalur:
                v = (v or {}).get(k) if isinstance(v, dict) else None
            if v is not None:
                nilai.append(float(v))
        if not nilai:
            return None
        return [round(float(np.mean(nilai)), 3), round(float(np.std(nilai)), 3),
                len(nilai)]

    KUNCI = [
        ("sensitivitas_persen", ("deteksi", "sensitivitas_persen")),
        ("spesifisitas_persen", ("deteksi", "spesifisitas_persen")),
        ("ppv_persen", ("deteksi", "ppv_persen")),
        ("f1_persen", ("deteksi", "f1_persen")),
        ("rmse_pada_hipo", ("galat_pada_hipo", "rmse")),
        ("mae_pada_hipo", ("galat_pada_hipo", "mae")),
        ("bias_pada_hipo", ("galat_pada_hipo", "bias")),
        ("bias_keseluruhan", ("bias_keseluruhan",)),
    ]
    rerata = {mdl: {nama: kumpul(mdl, *jalur) for nama, jalur in KUNCI}
              for mdl in ("RF", "LSTM")}

    # Uji berpasangan tingkat fold. Tingkat fold, bukan tingkat sampel: yang ingin
    # diketahui adalah apakah keunggulannya bertahan lintas pasien, bukan apakah ada
    # perbedaan yang terdeteksi oleh n besar.
    uji = {}
    for nama in ("sensitivitas_persen", "mae_pada_hipo", "bias_pada_hipo"):
        jalur = dict(KUNCI)[nama]
        a = np.array([float(_gali(b["RF"], jalur)) for b in baris])
        c = np.array([float(_gali(b["LSTM"], jalur)) for b in baris])
        try:
            p = float(stats.wilcoxon(a, c).pvalue) if len(set(a - c)) > 1 else float("nan")
        except ValueError:
            p = float("nan")
        selisih = float(np.mean(a - c))
        sd_gab = float(np.std(np.concatenate([a, c])))
        uji[nama] = {
            "rerata_RF": round(float(np.mean(a)), 3),
            "rerata_LSTM": round(float(np.mean(c)), 3),
            "selisih_RF_minus_LSTM": round(selisih, 3),
            "sd_antar_fold_gabungan": round(sd_gab, 3),
            "wilcoxon_p": None if np.isnan(p) else round(p, 4),
            "signifikan_alpha_0.05": bool(p == p and p < 0.05),
            # Penjaga terhadap kekeliruan T1.1: selisih yang lebih kecil daripada
            # simpangan baku antar-fold TIDAK boleh dinarasikan sebagai temuan.
            "selisih_melampaui_sd_antar_fold": bool(abs(selisih) > sd_gab),
        }

    tot_hipo = sum(b["RF"]["n_hipo_benar"] for b in baris)
    tot_n = sum(b["RF"]["n_total"] for b in baris)
    out = {
        "percobaan": "T2.1 — galat khusus hipoglikemia, RF vs LSTM",
        "catatan": ("Jendela TERSEGMENTASI (max_gap_steps), sama seperti pelatihan "
                    "produksi. Menggantikan hypo_safety_uncertainty.json yang dihitung "
                    "sebelum Tugas 5, hanya untuk RF, dan hanya satu horizon."),
        "horizon_steps": HORIZON, "horizon_min": int(HORIZON * cadence_min),
        "max_gap_steps": max_gap_steps, "k_fold": K,
        "ambang_hipoglikemia": AMBANG, "ambang_hipoglikemia_berat": AMBANG_BERAT,
        "basis_kejadian": {
            "n_jendela": tot_n, "n_hipo": tot_hipo,
            "porsi_hipo_persen": round(100 * tot_hipo / tot_n, 3),
            "n_hipo_berat": sum(b["RF"]["n_hipo_berat_benar"] for b in baris),
        },
        "terlewat_ke_rentang_target": {
            mdl: sum(b[mdl]["terlewat_ke_rentang_target"] for b in baris)
            for mdl in ("RF", "LSTM")},
        "hipo_berat_terlewat": {
            mdl: sum(b[mdl]["hipo_berat_terlewat"] for b in baris)
            for mdl in ("RF", "LSTM")},
        "rerata_lintas_fold": rerata,
        "uji_berpasangan_tingkat_fold": uji,
        "aturan_pelaporan": ("Selisih RF-LSTM yang lebih kecil daripada simpangan baku "
                             "antar-fold tidak boleh dinarasikan sebagai temuan; medan "
                             "selisih_melampaui_sd_antar_fold menandainya."),
        "folds": baris,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"hipoglikemia_h{HORIZON}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Hipoglikemia +{out['horizon_min']} mnt ({K} fold) ===")
    print(f"basis kejadian: {tot_hipo:,} dari {tot_n:,} jendela "
          f"({out['basis_kejadian']['porsi_hipo_persen']:.2f}%), "
          f"berat {out['basis_kejadian']['n_hipo_berat']:,}")
    print(f"\n{'metrik':<24}{'RF':>18}{'LSTM':>18}")
    for nama, _ in KUNCI:
        a, c = rerata["RF"][nama], rerata["LSTM"][nama]
        if a and c:
            print(f"{nama:<24}{a[0]:>10.2f} ±{a[1]:<6.2f}{c[0]:>10.2f} ±{c[1]:<6.2f}")
    print(f"\nterlewat ke rentang target : RF {out['terlewat_ke_rentang_target']['RF']:,} | "
          f"LSTM {out['terlewat_ke_rentang_target']['LSTM']:,}")
    print(f"hipo BERAT terlewat        : RF {out['hipo_berat_terlewat']['RF']:,} | "
          f"LSTM {out['hipo_berat_terlewat']['LSTM']:,}")
    # KEPUTUSAN #8 (docs/KEPUTUSAN_DIAMBIL.md): definisi SD yang dipakai WAJIB disebut.
    # Skrip ini memakai SD GABUNGAN, np.std atas nilai kedua model disatukan. Itu BUKAN
    # definisi yang ditetapkan #8 (SD selisih berpasangan), dan keduanya dapat memberi
    # verdik berlawanan — pada h12 sensitivitas, gabungan 5,862 meloloskan selisih 6,203
    # sedangkan SD selisih 6,877 menggugurkannya.
    #
    # Angka di sini SENGAJA TIDAK diubah: aturan sesi melarang mengubah angka lama, dan
    # menghitung ulang dengan definisi baru berarti menulis ulang hasil T2.1. Yang
    # diperbaiki hanya LABELNYA, supaya pembaca tahu definisi mana yang menghasilkan
    # verdik ini. Verdik menurut definisi terpilih ada di K14 dan KEPUTUSAN_DIAMBIL.
    print("\nUji berpasangan tingkat fold "
          "(SD = GABUNGAN kedua model; definisi #8 adalah SD SELISIH — lihat K14):")
    for nama, u in uji.items():
        layak = "layak menurut SD gabungan" if u["selisih_melampaui_sd_antar_fold"] else \
                "DI BAWAH SD gabungan — jangan dinarasikan"
        print(f"  {nama:<22} selisih {u['selisih_RF_minus_LSTM']:>+7.2f} "
              f"(SD gabungan {u['sd_antar_fold_gabungan']:.2f}) p={u['wilcoxon_p']} "
              f"-> {layak}")
    print("  CATATAN: verdik final memakai SD selisih berpasangan (#8). Pada h12 "
          "sensitivitas,\n  kedua definisi BERBEDA dan klaimnya berstatus RAPUH.")
    print(f"\nDisimpan ke {p}")


def _gali(d: dict, jalur):
    v = d
    for k in jalur:
        v = v.get(k) if isinstance(v, dict) else None
    return v


if __name__ == "__main__":
    main()
