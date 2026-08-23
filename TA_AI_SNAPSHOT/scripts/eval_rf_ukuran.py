"""T1.1b — perkecil model Random Forest tanpa mengorbankan akurasi.

LATAR: berkas model RF 322-390 MB berhadapan dengan LSTM 0,124 MB (T1.1). Untuk narasi
keterterapan pada lingkungan bersumber daya terbatas, ini titik lemah yang lebih serius
daripada selisih akurasi.

ANALISIS PENDAHULUAN (dijalankan lebih dulu, hasilnya menentukan arah):
  n_estimators   200
  max_depth      20  -> TERCAPAI oleh 200 dari 200 pohon (100%)
  simpul/pohon   median 23.171
  total simpul   4.631.504
  ukuran berkas  333,5 MB  (~72 byte per simpul)

Karena kedalaman tercapai penuh, `max_depth` BENAR-BENAR MENGIKAT dan menurunkannya akan
mengecilkan model secara superlinear. `n_estimators` mengecilkan secara linear. Keduanya
karena itu diuji, termasuk kombinasinya.

KRITERIA PEMILIHAN — DITETAPKAN SEBELUM MELIHAT HASIL:
  konfigurasi TERKECIL yang RMSE-nya TIDAK BERBEDA SIGNIFIKAN dari konfigurasi sekarang.
  BUKAN konfigurasi dengan RMSE terendah, karena tujuan percobaan ini mengecilkan model.
  Uji beda: Wilcoxon berpasangan atas |error| per sampel terhadap konfigurasi acuan,
  alpha = 0,05. Tidak signifikan (p >= 0,05) berarti "tidak berbeda".

PROTOKOL BAGIAN C: set penyetelan dan set pelaporan dipisah per pasien, seluruh
konfigurasi dicatat ke results/tuning_log.json, kriteria ditetapkan di muka.

Keluaran: results/eval_prediksi/rf_ukuran_h{N}.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import json
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402
from tuning_protocol import KRITERIA, catat  # noqa: E402

OUT_DIR = ROOT / "results/eval_prediksi"
ALPHA = 0.05
# Sengaja BUKAN di results/: ini keadaan setengah jadi, bukan hasil. Menaruhnya di
# results/ mengundang kekeliruan yang persis pernah terjadi dengan crossfold.json —
# berkas setengah jadi yang duduk di antara berkas hasil dan terlihat sah.
CKPT = ROOT / ".cache/rf_ukuran_ckpt.npz"

# Ditetapkan di muka. Acuan (200, 20) ikut diuji sebagai pembanding.
KONFIGURASI = [
    (200, 20),   # acuan / produksi sekarang
    (100, 20), (50, 20),
    (200, 15), (200, 12), (200, 10),
    (100, 15), (100, 12), (50, 15), (50, 12), (50, 10),
]
ACUAN = (200, 20)


def ukuran_mb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(model, f)
        p = f.name
    n = os.path.getsize(p)
    os.unlink(p)
    return round(n / 1e6, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()
    H = args.horizon

    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    m = cfg["model"]
    seq_len = m.get("sequence_length", 12)
    use_eng = m.get("use_engineered", False)
    predict_delta = m.get("predict_delta", False)
    feats = m["engineered_features"] if use_eng else m["features"]
    rf_cfg = m.get("random_forest", {})
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

    # PROTOKOL BAGIAN C: pisah pasien menjadi set penyetelan dan set pelaporan.
    # Dibagi SEBELUM satu pun hasil dilihat; pasien uji tidak pernah ikut melatih.
    pids = sorted(df["patient_id"].unique().tolist())
    train_p = pids[:-4]
    setel_p, lapor_p = pids[-4:-2], pids[-2:]
    print(f"latih={len(train_p)} pasien | set penyetelan={setel_p} | set pelaporan={lapor_p}")
    print(f"Kriteria ditetapkan di muka: konfigurasi TERKECIL yang RMSE-nya tidak berbeda "
          f"signifikan dari {ACUAN} (Wilcoxon berpasangan, alpha={ALPHA})")
    print(f"Horizon +{int(H * cadence)} mnt | max_gap_steps={max_gap} | "
          f"{len(KONFIGURASI)} konfigurasi\n")

    kw = {"max_gap_steps": max_gap, "source_interval_min": cadence}

    def seqs(sub):
        return prep.create_sequences(df[df["patient_id"].isin(sub)], seq_len, H,
                                     return_anchor=True, **kw)

    Xtr, ytr, atr = seqs(train_p)
    Xse, yse, ase = seqs(setel_p)
    Xla, yla, ala = seqs(lapor_p)
    p2 = DataPreprocessor(cfg)
    Xtr_s, _ = p2.normalize_data(Xtr, None)
    Xse_s = p2.scaler.transform(Xse.reshape(-1, Xse.shape[2])).reshape(Xse.shape)
    Xla_s = p2.scaler.transform(Xla.reshape(-1, Xla.shape[2])).reshape(Xla.shape)
    Ftr = Xtr_s.reshape(len(ytr), -1)
    Fse = Xse_s.reshape(len(yse), -1)
    Fla = Xla_s.reshape(len(yla), -1)
    ytr_fit = (ytr - atr) if predict_delta else ytr
    print(f"jendela: latih {len(ytr):,} | penyetelan {len(yse):,} | pelaporan {len(yla):,}\n")

    # CHECKPOINT. Sapuan ini butuh ~40 menit dan versi pertamanya tidak menulis apa pun
    # sampai selesai — ketika prosesnya terbunuh pada konfigurasi ke-11, sepuluh hasil
    # yang sudah dihitung hilang seluruhnya. Pola yang sama sudah dipakai
    # eval_retrieval_crossfold.py dan seharusnya dipakai sejak awal di sini.
    #
    # Yang disimpan termasuk |galat| per sampel, karena tanpa itu uji Wilcoxon terhadap
    # acuan tidak dapat dijalankan ulang dan checkpoint-nya tidak ada gunanya.
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    hasil, err_setel = {}, {}
    if CKPT.exists():
        try:
            simpan = np.load(CKPT, allow_pickle=True)
            meta = json.loads(str(simpan["meta"]))
            # Checkpoint hanya sah bila dibuat untuk horizon dan pembagian pasien yang
            # sama. Kalau tidak, hasil lama akan diam-diam bercampur dengan sapuan baru.
            if (meta.get("horizon") == H and meta.get("setel") == setel_p
                    and meta.get("n_setel") == int(len(yse))):
                for kunci, v in json.loads(str(simpan["hasil"])).items():
                    k = tuple(int(x) for x in kunci.split("_"))
                    hasil[k] = v
                    err_setel[k] = simpan[f"err_{kunci}"]
                print(f"Checkpoint dimuat: {len(hasil)} konfigurasi sudah dihitung "
                      f"({CKPT.name}); sisanya dilanjutkan.\n")
            else:
                print(f"Checkpoint ADA tetapi tidak sepadan (horizon/pembagian berbeda) "
                      f"— diabaikan, sapuan diulang dari awal.\n")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"Checkpoint tidak terbaca ({exc}) — sapuan diulang dari awal.\n")

    def tulis_ckpt():
        arr = {f"err_{n}_{d}": err_setel[(n, d)] for n, d in err_setel}
        np.savez_compressed(
            CKPT,
            meta=json.dumps({"horizon": H, "setel": setel_p, "n_setel": int(len(yse))}),
            hasil=json.dumps({f"{n}_{d}": v for (n, d), v in hasil.items()}),
            **arr)

    for n_est, depth in KONFIGURASI:
        if (n_est, depth) in hasil:
            v = hasil[(n_est, depth)]
            print(f"  n={n_est:>3} d={depth:>2} | RMSE {v['RMSE']:6.2f} | "
                  f"{v['ukuran_MB']:>7.1f} MB | dari checkpoint", flush=True)
            continue
        t0 = time.time()
        mod = RandomForestRegressor(n_estimators=n_est, max_depth=depth,
                                    min_samples_split=rf_cfg.get("min_samples_split", 5),
                                    random_state=seed, n_jobs=-1)
        mod.fit(Ftr, ytr_fit)
        t_latih = time.time() - t0
        t0 = time.time()
        yp = mod.predict(Fse)
        t_inf = (time.time() - t0) / max(len(yse), 1) * 1000
        yp = yp + ase if predict_delta else yp
        met = calculate_all_metrics(yse, yp)
        mb = ukuran_mb(mod)
        d_tercapai = int(np.median([t.get_depth() for t in mod.estimators_]))
        simpul = int(sum(t.tree_.node_count for t in mod.estimators_))
        err_setel[(n_est, depth)] = np.abs(yse - yp)

        hasil[(n_est, depth)] = {
            "n_estimators": n_est, "max_depth": depth,
            "RMSE": round(float(met["RMSE"]), 3), "MAE": round(float(met["MAE"]), 3),
            "Clarke_A+B": round(float(met["Clarke_A+B"]), 2),
            "ukuran_MB": mb, "total_simpul": simpul,
            "kedalaman_tercapai_median": d_tercapai,
            "waktu_latih_dtk": round(t_latih, 1),
            "inferensi_ms_per_sampel": round(t_inf, 4),
        }
        # Kriteria percobaan ini adalah kesetaraan RMSE, BUKAN MRR. Tanpa timpaan ini
        # entri tersimpan sebagai kriteria "mrr" — log penyetelan justru artefak yang
        # harus membuktikan protokol dipatuhi, jadi salah stempel di situ merusak
        # gunanya.
        catat("rf_ukuran", f"n{n_est}_d{depth}", hasil[(n_est, depth)],
              catatan=(f"horizon={H}, set penyetelan n={len(yse)}. Kriteria: konfigurasi "
                       f"TERKECIL yang RMSE-nya tidak berbeda signifikan dari {ACUAN}."),
              kriteria="rmse_setara_dengan_acuan")
        tulis_ckpt()
        print(f"  n={n_est:>3} d={depth:>2} | RMSE {met['RMSE']:6.2f} | {mb:>7.1f} MB | "
              f"{simpul:>9,} simpul | d_capai {d_tercapai:>2} | {t_latih:>5.0f} dtk",
              flush=True)

    # Uji beda terhadap acuan pada SET PENYETELAN.
    e_acuan = err_setel[ACUAN]
    for k, v in hasil.items():
        if k == ACUAN:
            v["p_vs_acuan"] = None
            v["tidak_berbeda"] = True
            continue
        p = float(stats.wilcoxon(e_acuan, err_setel[k]).pvalue)
        v["p_vs_acuan"] = round(p, 4)
        v["tidak_berbeda"] = p >= ALPHA

    # KRITERIA: TERKECIL di antara yang tidak berbeda signifikan.
    kandidat = [k for k, v in hasil.items() if v["tidak_berbeda"]]
    terpilih = min(kandidat, key=lambda k: hasil[k]["ukuran_MB"]) if kandidat else None

    print(f"\n{'konfigurasi':<16}{'RMSE':>8}{'MB':>9}{'p vs acuan':>12}{'tidak beda':>12}")
    for k in KONFIGURASI:
        v = hasil[k]
        p = "acuan" if v["p_vs_acuan"] is None else f"{v['p_vs_acuan']:.4f}"
        print(f"n={k[0]:<4} d={k[1]:<6}{v['RMSE']:>8.2f}{v['ukuran_MB']:>9.1f}{p:>12}"
              f"{'YA' if v['tidak_berbeda'] else 'tidak':>12}")

    out = {
        "protokol": ("Bagian C. Set penyetelan dan pelaporan dipisah per pasien sebelum "
                     "hasil dilihat. Kriteria ditetapkan di muka: konfigurasi TERKECIL yang "
                     "RMSE-nya tidak berbeda signifikan dari acuan, BUKAN RMSE terendah."),
        "kriteria_utama": KRITERIA, "alpha": ALPHA,
        "horizon_steps": H, "horizon_min": int(H * cadence), "max_gap_steps": max_gap,
        "acuan": {"n_estimators": ACUAN[0], "max_depth": ACUAN[1]},
        "n_konfigurasi_diuji": len(KONFIGURASI),
        "pasien_latih": train_p, "pasien_penyetelan": setel_p, "pasien_pelaporan": lapor_p,
        "n_jendela": {"latih": int(len(ytr)), "penyetelan": int(len(yse)),
                      "pelaporan": int(len(yla))},
        "hasil_set_penyetelan": {f"n{k[0]}_d{k[1]}": v for k, v in hasil.items()},
    }

    if terpilih is None:
        out["terpilih"] = None
        out["kesimpulan"] = ("TIDAK ADA konfigurasi yang memenuhi kriteria. Seluruh "
                             "konfigurasi lebih kecil berbeda signifikan dari acuan. "
                             "Pertahankan konfigurasi sekarang; ukuran model besar "
                             "dinyatakan sebagai keterbatasan Bab VII.")
        print(f"\n{out['kesimpulan']}")
    else:
        # Ukur ulang acuan dan terpilih pada SET PELAPORAN.
        print(f"\nTerpilih pada set penyetelan: n={terpilih[0]} d={terpilih[1]} "
              f"({hasil[terpilih]['ukuran_MB']} MB)")
        print("Mengukur pada set pelaporan dengan konfigurasi terkunci ...")
        lapor = {}
        err_lapor = {}
        for k in {ACUAN, terpilih}:
            mod = RandomForestRegressor(n_estimators=k[0], max_depth=k[1],
                                        min_samples_split=rf_cfg.get("min_samples_split", 5),
                                        random_state=seed, n_jobs=-1)
            mod.fit(Ftr, ytr_fit)
            yp = mod.predict(Fla)
            yp = yp + ala if predict_delta else yp
            met = calculate_all_metrics(yla, yp)
            err_lapor[k] = np.abs(yla - yp)
            lapor[f"n{k[0]}_d{k[1]}"] = {
                "RMSE": round(float(met["RMSE"]), 3), "MAE": round(float(met["MAE"]), 3),
                "Clarke_A+B": round(float(met["Clarke_A+B"]), 2),
                "ukuran_MB": ukuran_mb(mod),
            }
        p_lapor = (None if terpilih == ACUAN
                   else round(float(stats.wilcoxon(err_lapor[ACUAN],
                                                   err_lapor[terpilih]).pvalue), 4))
        out["terpilih"] = {"n_estimators": terpilih[0], "max_depth": terpilih[1]}
        out["set_pelaporan"] = lapor
        out["p_set_pelaporan"] = p_lapor
        a, b = lapor[f"n{ACUAN[0]}_d{ACUAN[1]}"], lapor[f"n{terpilih[0]}_d{terpilih[1]}"]
        print(f"\n=== SET PELAPORAN (n={len(yla):,}, konfigurasi terkunci) ===")
        print(f"  acuan    n={ACUAN[0]} d={ACUAN[1]}: RMSE {a['RMSE']:.2f}  {a['ukuran_MB']:.1f} MB")
        print(f"  terpilih n={terpilih[0]} d={terpilih[1]}: RMSE {b['RMSE']:.2f}  {b['ukuran_MB']:.1f} MB")
        if p_lapor is not None:
            print(f"  Wilcoxon p = {p_lapor}  -> "
                  f"{'TIDAK berbeda' if p_lapor >= ALPHA else 'BERBEDA signifikan'}")
            hemat = 100 * (1 - b["ukuran_MB"] / a["ukuran_MB"])
            print(f"  penghematan ukuran: {hemat:.1f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"rf_ukuran_h{H}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    # Checkpoint dihapus setelah hasil akhir tertulis. Membiarkannya berarti sapuan
    # berikutnya dengan config.yaml yang sudah berubah akan memuat hasil lama tanpa
    # ada yang gagal.
    CKPT.unlink(missing_ok=True)
    print(f"\n{len(KONFIGURASI)} konfigurasi diuji. Disimpan ke {p}")


if __name__ == "__main__":
    main()
