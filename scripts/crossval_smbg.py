"""Validasi silang lintas-pasien untuk skenario FINGER_STICK.

Motivasi
--------
Seluruh angka FINGER_STICK sejauh ini — RMSE 68,24 dan Clarke A+B 52,52% —
berasal dari SATU pembagian data dengan 139 sampel uji. Pada ukuran sebesar itu,
satu titik estimasi tidak memberi tahu seberapa jauh angkanya dapat bergeser bila
pasien ujinya berbeda. Tanpa rentang, selisih antar-konfigurasi pada
``results/eval_prediksi/smbg_sweep.json`` tidak dapat dinilai bermakna atau tidak.

Skrip ini menjalankan validasi silang enam *fold* lintas-pasien, mengikuti pola
``scripts/crossval_rf_vs_lstm.py`` (``pids[i::K]``), sehingga tiap pasien menjadi
data uji tepat sekali dan tidak pernah dilihat model saat dilatih.

Pada tiap *fold*, model dan baseline *persistence* dinilai pada himpunan uji yang
SAMA, sehingga galatnya berpasangan dan dapat diuji dengan Wilcoxon berpasangan
pada tingkat *fold*.

Catatan pembagian data
----------------------
Pembagian di sini **per pasien**, berbeda dari ``dataset_split`` resmi yang dipakai
model produksi. Keduanya menjawab pertanyaan berbeda: split resmi menguji
generalisasi ke waktu yang belum dilihat pada pasien yang sama, sedangkan validasi
silang ini menguji generalisasi ke pasien yang sama sekali baru. Angka keduanya
tidak boleh dipertukarkan.

Subproses per konfigurasi
-------------------------
Enam *fold* dalam satu proses menumpuk pemakaian memori sampai gagal alokasi pada
mesin yang hanya menyisakan sekitar 1,7 GB bebas; membebaskan array secara
eksplisit saja tidak cukup. Tiap konfigurasi karena itu dijalankan pada subproses
tersendiri, yang mengembalikan memorinya sepenuhnya setelah selesai.

Keluaran: results/eval_prediksi/crossval_smbg.json
"""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402

SUMBER = "FINGER_STICK"
K = 6
SEED = 42
DEST = ROOT / "results/eval_prediksi/crossval_smbg.json"

# Konfigurasi yang diuji. Yang pertama adalah konfigurasi produksi; yang kedua
# adalah pesaing terdekatnya menurut sapuan (sampel terbanyak pada horizon 4 jam).
KONFIGURASI = [
    {"nama": "produksi_seq8_4j", "seq": 8, "nominal": 240.0, "bawah": 180.0, "atas": 300.0},
    {"nama": "seq5_4j", "seq": 5, "nominal": 240.0, "bawah": 180.0, "atas": 300.0},
]


def muat(cfg: dict):
    """Muat dan siapkan bingkai FINGER_STICK.

    Penyaringan kanal mendahului rekayasa fitur — urutan yang sama dengan
    ``src/models/gbm_model.py:391``. Membaliknya membuat iob/cob/glucose_delta
    dihitung dari baris tetangga yang berbeda.
    """
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm.csv", parse_dates=["timestamp"])
    df = df[df["glucose_source"].astype(str).str.upper() == SUMBER].copy()
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(
        df, max_interpolate_steps=mc.get("max_interpolate_steps")
    )
    df = pre.engineer_features(df, **mc.get("feature_engineering", {}))
    pre.feature_columns = list(mc["engineered_features"])
    return pre, df


def jendela(pre, d, k, profil):
    return pre.create_time_horizon_sequences(
        d,
        sequence_length=k["seq"],
        horizon_min=k["nominal"],
        target_tolerance_min=float(profil.get("target_tolerance_min", 0.0)),
        max_history_gap_min=profil.get("max_history_gap_min"),
        return_anchor=True,
        min_target_horizon_min=k["bawah"],
        max_target_horizon_min=k["atas"],
        min_history_interval_min=float(profil.get("min_history_interval_min", 0.0)),
    )


def ringkas(m: dict) -> dict:
    return {
        "RMSE": round(m["RMSE"], 2),
        "MAE": round(m["MAE"], 2),
        "Clarke_A": round(m["Clarke_A"], 2),
        "Clarke_A+B": round(m["Clarke_A+B"], 2),
        "Clarke_E": round(m["Clarke_E"], 2),
    }


def rerata_sd(nilai) -> dict:
    a = np.asarray(nilai, dtype=float)
    return {
        "rerata": round(float(a.mean()), 2),
        "sd": round(float(a.std(ddof=1)), 2),
        "min": round(float(a.min()), 2),
        "maks": round(float(a.max()), 2),
    }


def satu_konfigurasi(k: dict) -> dict:
    """Jalankan K fold untuk satu konfigurasi. Dipanggil di dalam subproses."""
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pre, df = muat(cfg)
    profil = cfg["model"]["source_profiles"][SUMBER]
    gseed = cfg["model"].get("gradient_boosting", {}).get("random_state", SEED)

    pasien = sorted(df["patient_id"].unique())
    lipatan = [pasien[i::K] for i in range(K)]
    baris = []

    for i, uji_pasien in enumerate(lipatan):
        latih_df = df[~df["patient_id"].isin(uji_pasien)]
        uji_df = df[df["patient_id"].isin(uji_pasien)]

        Xtr, ytr, atr = jendela(pre, latih_df, k, profil)
        Xte, yte, ate = jendela(pre, uji_df, k, profil)

        if len(yte) < 10 or len(ytr) < 50:
            print(f"  fold {i} {uji_pasien}: n uji {len(yte)} — dilewati")
            baris.append({"fold": i, "pasien_uji": list(uji_pasien),
                          "n_uji": int(len(yte)), "catatan": "terlalu sedikit"})
            continue

        # Penskala di-fit hanya pada data latih fold ini, lalu diterapkan ke data
        # uji — mencegah statistik pasien uji bocor ke pelatihan.
        Xtr_s, Xte_s = pre.normalize_data(Xtr, Xte)
        Xtr_s = Xtr_s.reshape(len(Xtr), -1)
        Xte_s = Xte_s.reshape(len(Xte), -1)

        model = HistGradientBoostingRegressor(random_state=gseed)
        model.fit(Xtr_s, ytr - atr)               # target delta
        duga = model.predict(Xte_s) + ate         # rekonstruksi ke nilai absolut

        m_model = calculate_all_metrics(yte, duga)
        m_pers = calculate_all_metrics(yte, ate)  # persistence

        baris.append({
            "fold": i,
            "pasien_uji": list(uji_pasien),
            "n_latih": int(len(ytr)),
            "n_uji": int(len(yte)),
            "model": ringkas(m_model),
            "persistence": ringkas(m_pers),
        })
        print(f"  fold {i} {uji_pasien}: n uji {len(yte):4d} | "
              f"model RMSE {m_model['RMSE']:6.2f} A+B {m_model['Clarke_A+B']:5.2f}% | "
              f"pers RMSE {m_pers['RMSE']:6.2f} A+B "
              f"{m_pers['Clarke_A'] + m_pers['Clarke_B']:5.2f}%")

        del Xtr, Xte, Xtr_s, Xte_s, model, duga
        gc.collect()

    sah = [b for b in baris if "model" in b]
    blok = {"folds": baris, "n_fold_sah": len(sah)}
    if len(sah) < 2:
        return blok

    r_model = [b["model"]["RMSE"] for b in sah]
    r_pers = [b["persistence"]["RMSE"] for b in sah]
    ab_model = [b["model"]["Clarke_A+B"] for b in sah]
    ab_pers = [b["persistence"]["Clarke_A+B"] for b in sah]

    blok["rerata_lintas_fold"] = {
        "model": {"RMSE": rerata_sd(r_model), "Clarke_A+B": rerata_sd(ab_model)},
        "persistence": {"RMSE": rerata_sd(r_pers), "Clarke_A+B": rerata_sd(ab_pers)},
    }

    # Uji berpasangan pada tingkat fold. Dengan K=6, nilai-p terkecil yang dapat
    # dicapai Wilcoxon dua sisi adalah 0,03125 — tepat di bawah 0,05. Karena itu
    # nilai-p di sini dibaca sebagai penunjuk arah yang konsisten, bukan bukti kuat.
    selisih = np.array(r_pers) - np.array(r_model)
    try:
        _, w_p = stats.wilcoxon(r_pers, r_model)
    except ValueError:
        w_p = float("nan")
    _, t_p = stats.ttest_rel(r_pers, r_model)

    blok["uji_berpasangan_RMSE"] = {
        "selisih_pers_minus_model": rerata_sd(selisih),
        "fold_model_menang": int((selisih > 0).sum()),
        "n_fold": len(sah),
        "wilcoxon_p": round(float(w_p), 4),
        "ttest_rel_p": round(float(t_p), 4),
        "catatan_p": "K=6 → p minimum Wilcoxon dua sisi = 0,03125",
    }

    rm = blok["rerata_lintas_fold"]["model"]
    rp = blok["rerata_lintas_fold"]["persistence"]
    print(f"  -- rerata {len(sah)} fold --")
    print(f"     model       RMSE {rm['RMSE']['rerata']:6.2f} ± {rm['RMSE']['sd']:5.2f} "
          f"(rentang {rm['RMSE']['min']:.2f}–{rm['RMSE']['maks']:.2f}) | "
          f"A+B {rm['Clarke_A+B']['rerata']:.2f} ± {rm['Clarke_A+B']['sd']:.2f}")
    print(f"     persistence RMSE {rp['RMSE']['rerata']:6.2f} ± {rp['RMSE']['sd']:5.2f} | "
          f"A+B {rp['Clarke_A+B']['rerata']:.2f} ± {rp['Clarke_A+B']['sd']:.2f}")
    print(f"     model menang pada {blok['uji_berpasangan_RMSE']['fold_model_menang']}"
          f"/{len(sah)} fold | Wilcoxon p={w_p:.4f}")
    return blok


def main() -> None:
    out = {
        "sumber": SUMBER,
        "k_fold": K,
        "pembagian": "per pasien (pids[i::K]) — BUKAN dataset_split resmi",
        "catatan": (
            "Model dan persistence dinilai pada himpunan uji yang sama tiap fold, "
            "sehingga galatnya berpasangan. Split per pasien menguji generalisasi "
            "ke pasien baru; angkanya tidak sebanding dengan hasil split resmi."
        ),
        "konfigurasi": {},
    }

    for k in KONFIGURASI:
        print(f"\n=== {k['nama']} (riwayat {k['seq']}, target "
              f"{k['bawah']:g}-{k['atas']:g} menit) ===")
        proses = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", k["nama"]],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        blok = None
        for garis in proses.stdout.splitlines():
            if garis.startswith("<<<JSON>>>"):
                blok = json.loads(garis[len("<<<JSON>>>"):])
            else:
                print(garis)
        if blok is None:
            ekor = (proses.stderr or "").strip().splitlines()
            print(f"  gagal: {ekor[-1] if ekor else 'tanpa keluaran'}")
            continue
        out["konfigurasi"][k["nama"]] = blok

    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDisimpan ke {DEST.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validasi silang FINGER_STICK.")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        pilih = next(x for x in KONFIGURASI if x["nama"] == args.worker)
        print("<<<JSON>>>" + json.dumps(satu_konfigurasi(pilih)))
    else:
        main()
