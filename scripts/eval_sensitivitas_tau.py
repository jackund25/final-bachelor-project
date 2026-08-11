"""T1.4 / J3 — sensitivitas prediksi terhadap konstanta waktu peluruhan IOB dan COB.

Subbab II.3.5 menyebut nilai `insulin_tau_min = 240` (kerja insulin ~4 jam) dan
`carbs_tau_min = 180` (penyerapan karbohidrat ~3 jam) diambil dari literatur, dan
menjanjikan sensitivitasnya "diuji pada Bab VI". Sampai audit, tidak ada satu pun skrip
yang memvariasikan kedua nilai itu (`sensitivity_analysis.py` hanya menyapu `top_k` dan
`chunk_size`). Skrip ini menutup janji tersebut.

=== INI PENGUKURAN SENSITIVITAS, BUKAN PENYETELAN ===

Tujuannya menjawab "seberapa besar pengaruh pilihan tau terhadap galat prediksi", BUKAN
mencari tau terbaik. Nilai produksi TIDAK akan diubah berdasarkan hasil ini. Walau begitu
protokol Bagian C tetap dipatuhi sepenuhnya, karena kalau hasilnya nanti dipakai sebagai
alasan mengubah sesuatu, protokolnya harus sudah berlaku sejak sebelum angkanya terlihat:

  * Set penyetelan dan set pelaporan DIPISAH menurut pasien, tidak pernah bertukar.
  * SELURUH konfigurasi dicatat, bukan hanya yang terbaik.
  * Kriteria ditetapkan DI MUKA: RMSE pada set penyetelan, dibandingkan dengan konfigurasi
    produksi memakai Wilcoxon berpasangan atas |galat| pada alpha=0,05.
  * Set pelaporan hanya disentuh SEKALI di akhir, untuk konfigurasi produksi dan satu
    konfigurasi pembanding yang sudah ditetapkan oleh kriteria di atas.

=== RANCANGAN SAPUAN: SATU-PER-SATU, BUKAN GRID PENUH ===

Grid penuh 5x5 = 25 konfigurasi x ~9 menit = lebih dari 3,5 jam, dan sebagian besar
selnya tidak menjawab pertanyaan yang diajukan Subbab II.3.5. Yang ditanyakan adalah
kepekaan terhadap MASING-MASING konstanta, sehingga sapuan satu-per-satu (OAT) di sekitar
titik produksi sudah memadai: 5 nilai insulin_tau (carbs_tau ditahan di produksi) + 5
nilai carbs_tau (insulin_tau ditahan di produksi), berbagi satu titik produksi = 9
konfigurasi.

Batasan rancangan OAT dicatat apa adanya: ia TIDAK dapat menangkap interaksi antara kedua
konstanta. Bila hasilnya menunjukkan salah satu konstanta berpengaruh besar, interaksinya
layak diperiksa; bila keduanya datar, interaksi hampir pasti juga datar.

Keluaran: results/eval_prediksi/sensitivitas_tau_h{H}.json
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
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataPreprocessor  # noqa: E402

OUT_DIR = ROOT / "results/eval_prediksi"
CACHE_ROOT = ROOT / ".cache"
ALPHA = 0.05

# ── DEFINISI SKALA DERAU, DITETAPKAN DI MUKA (docs/PRAPENDAFTARAN_T1.4.md) ───────────
# T1.4 memakai pembagian pasien TUNGGAL, bukan validasi silang 6 fold seperti T1.1,
# T2.1, dan T4.1. Karena itu "SD antar-fold" — yang T4.1 temukan punya tiga definisi
# berbeda — TIDAK TERDEFINISI di sini, dalam ketiga variannya sekalipun.
#
# Yang dipakai sebagai gantinya: SD bootstrap RMSE pada set penyetelan. Ia mengukur
# ketidakpastian penaksiran RMSE itu sendiri pada himpunan yang sama, yaitu derau yang
# relevan ketika yang dibandingkan adalah RMSE dari himpunan identik.
#
# Kriteria "peka" memakai 2x, bukan 1x: rentang adalah statistik ekstrem (maks - min)
# yang melebar seiring banyaknya konfigurasi bahkan bila seluruhnya menaksir besaran
# yang sama.
N_BOOTSTRAP = 1000
BENIH_BOOTSTRAP = 42
FAKTOR_AMBANG_PEKA = 2.0

# Batas atas berprinsip dari J7 (permutation importance, hold-out n=26.445): memusnahkan
# iob sepenuhnya menaikkan RMSE 0,4369 mg/dL dan cob 0,6743. Menggeser tau TIDAK
# memusnahkan fiturnya, sehingga pengaruhnya wajib lebih kecil. Dipakai sebagai
# pemeriksaan kesehatan, bukan sebagai kriteria.
BIAYA_PERMUTASI_IOB = 0.4369
BIAYA_PERMUTASI_COB = 0.6743


def jalur_pendek(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sd_bootstrap_rmse(yte, yp, n_resample=N_BOOTSTRAP, benih=BENIH_BOOTSTRAP) -> dict:
    """SD bootstrap RMSE. Skala derau yang dideklarasikan di prapendaftaran."""
    rng = np.random.default_rng(benih)
    err2 = (np.asarray(yp) - np.asarray(yte)) ** 2
    n = len(err2)
    nilai = np.empty(n_resample)
    for i in range(n_resample):
        nilai[i] = np.sqrt(np.mean(err2[rng.integers(0, n, n)]))
    return {"rmse_titik": round(float(np.sqrt(np.mean(err2))), 4),
            "sd_bootstrap": round(float(nilai.std(ddof=1)), 4),
            "ci95": [round(float(np.percentile(nilai, 2.5)), 4),
                     round(float(np.percentile(nilai, 97.5)), 4)],
            "n_resample": int(n_resample), "benih": int(benih), "n": int(n)}


def sidik_jari(H, max_gap, train_p, setel_p, lapor_p, seed, rf_cfg, feats) -> dict:
    """Identitas run. Checkpoint dengan sidik jari berbeda ditolak, bukan dipakai."""
    return {"horizon_steps": int(H), "max_gap_steps": int(max_gap),
            "set_latih": list(train_p), "set_penyetelan": list(setel_p),
            "set_pelaporan": list(lapor_p), "seed": int(seed),
            "random_forest": dict(rf_cfg), "fitur": list(feats),
            "sapuan_insulin": list(SAPU_INSULIN), "sapuan_carbs": list(SAPU_CARBS)}


def _kunci(it, ct, himpunan) -> str:
    return f"{himpunan}_i{int(it)}_c{int(ct)}"


def muat_checkpoint(cache_dir: Path, sidik: dict) -> dict:
    """Kembalikan {kunci: (yte, yp)} yang sepadan. Yang tidak sepadan ditolak."""
    meta_p = cache_dir / "meta.json"
    if not meta_p.exists():
        return {}
    try:
        lama = json.loads(meta_p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  checkpoint META rusak, diabaikan: {exc}")
        return {}
    if lama != sidik:
        beda = [k for k in sidik if lama.get(k) != sidik[k]]
        print(f"  checkpoint DITOLAK — sidik jari berbeda pada: {beda}")
        print("  (checkpoint tidak sepadan lebih berbahaya daripada tidak ada checkpoint)")
        return {}
    ada = {}
    for p in sorted(cache_dir.glob("*.npz")):
        try:
            z = np.load(p)
            ada[p.stem] = (z["yte"], z["yp"], int(z["n_latih"]))
        except (OSError, KeyError) as exc:
            print(f"  {p.name} pada checkpoint rusak, akan dihitung ulang: {exc}")
    return ada


def simpan_checkpoint(cache_dir: Path, kunci: str, yte, yp, n_latih: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"{kunci}.npz", yte=yte, yp=yp, n_latih=n_latih)

# Titik produksi. Dibaca ulang dari config.yaml saat berjalan; nilai di sini hanya
# menentukan susunan sapuan OAT.
INSULIN_TAU_PROD = 240.0
CARBS_TAU_PROD = 180.0

# Rentang dipilih mengelilingi nilai literatur dengan lebar yang masuk akal secara
# fisiologis: kerja insulin dilaporkan 2-6 jam, penyerapan karbohidrat 1-5 jam
# tergantung komposisi makanan. Ditetapkan SEBELUM satu pun angka dilihat.
SAPU_INSULIN = [120.0, 180.0, 240.0, 300.0, 360.0]
SAPU_CARBS = [60.0, 120.0, 180.0, 240.0, 300.0]


def daftar_konfigurasi(insulin_prod: float, carbs_prod: float):
    """Sapuan OAT; titik produksi muncul tepat sekali dan ditandai."""
    konf = []
    for it in SAPU_INSULIN:
        konf.append((it, carbs_prod, "insulin"))
    for ct in SAPU_CARBS:
        if (insulin_prod, ct) not in [(a, b) for a, b, _ in konf]:
            konf.append((insulin_prod, ct, "carbs"))
    return konf


def latih_dan_nilai(df_mentah, prep_cfg, mc, feats, H, seq_len, max_gap, cadence,
                    insulin_tau, carbs_tau, train_p, nilai_p, seed):
    """Rekayasa fitur dengan tau tertentu, latih RF produksi, kembalikan galat per sampel.

    Rekayasa fitur diulang dari data MENTAH setiap konfigurasi. Menghitung ulang iob/cob
    di atas kolom iob/cob yang sudah ada akan menumpuk peluruhan dua kali.
    """
    pre = DataPreprocessor(prep_cfg)
    df = pre.handle_missing_values(df_mentah.copy())
    df = pre.engineer_features(df, insulin_tau_min=insulin_tau, carbs_tau_min=carbs_tau,
                               trend_steps=mc["feature_engineering"]["trend_steps"],
                               source_interval_min=cadence)
    pre.feature_columns = list(feats)
    kw = {"max_gap_steps": max_gap, "source_interval_min": cadence}

    Xtr, ytr, atr = pre.create_sequences(df[df["patient_id"].isin(train_p)], seq_len, H,
                                         return_anchor=True, **kw)
    Xte, yte, ate = pre.create_sequences(df[df["patient_id"].isin(nilai_p)], seq_len, H,
                                         return_anchor=True, **kw)
    p2 = DataPreprocessor(prep_cfg)
    Xtr_s, _ = p2.normalize_data(Xtr, None)
    Xte_s = p2.scaler.transform(Xte.reshape(-1, Xte.shape[2])).reshape(Xte.shape)

    rf = mc["random_forest"]
    model = RandomForestRegressor(n_estimators=rf["n_estimators"], max_depth=rf["max_depth"],
                                  min_samples_split=rf["min_samples_split"],
                                  random_state=seed, n_jobs=-1)
    model.fit(Xtr_s.reshape(len(ytr), -1), (ytr - atr) if mc["predict_delta"] else ytr)
    yp = model.predict(Xte_s.reshape(len(yte), -1))
    if mc["predict_delta"]:
        yp = yp + ate
    return yte, yp, len(ytr)


def ringkas(yte, yp):
    err = yp - yte
    return {
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "n": int(len(yte)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()
    H = args.horizon

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mc = cfg["model"]
    fe = mc["feature_engineering"]
    insulin_prod = float(fe["insulin_tau_min"])
    carbs_prod = float(fe["carbs_tau_min"])
    seq_len = mc["sequence_length"]
    feats = mc["engineered_features"]
    max_gap = mc.get("max_gap_steps")
    if max_gap is None:
        raise SystemExit("config.model.max_gap_steps tidak ada — hasil tidak akan sepadan "
                         "dengan model produksi.")
    cadence = float(cfg["data"]["sampling_interval_min"])
    seed = int(cfg["data"].get("seed", 42))

    df_mentah = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    pids = sorted(df_mentah["patient_id"].unique().tolist())
    # Pembagian IDENTIK dengan T1.1b dan conformal_calibration.py, supaya set pelaporan
    # tetap satu-satunya himpunan yang belum pernah dilihat di percobaan mana pun.
    train_p, setel_p, lapor_p = pids[:-4], pids[-4:-2], pids[-2:]

    konf = daftar_konfigurasi(insulin_prod, carbs_prod)
    print(f"J3 / T1.4 — sensitivitas tau IOB & COB | horizon +{int(H * cadence)} mnt")
    print(f"produksi: insulin_tau={insulin_prod:.0f} carbs_tau={carbs_prod:.0f}")
    print(f"latih={len(train_p)} | penyetelan={setel_p} | pelaporan={lapor_p}")
    print(f"Kriteria DI MUKA: RMSE pada set PENYETELAN; beda thd produksi diuji Wilcoxon "
          f"berpasangan atas |galat|, alpha={ALPHA}")
    print(f"Rancangan OAT, {len(konf)} konfigurasi (grid penuh 25 sengaja tidak dipakai)\n")

    sidik = sidik_jari(H, max_gap, train_p, setel_p, lapor_p, seed, mc["random_forest"],
                       feats)
    cache_dir = CACHE_ROOT / f"sensitivitas_tau_h{H}"
    print(f"checkpoint : {jalur_pendek(cache_dir)}")
    tersimpan = muat_checkpoint(cache_dir, sidik)
    if tersimpan:
        print(f"  {len(tersimpan)} konfigurasi dipulihkan: {sorted(tersimpan)}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text(json.dumps(sidik, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print(f"Skala derau DITETAPKAN DI MUKA: SD bootstrap RMSE ({N_BOOTSTRAP} resample, "
          f"benih {BENIH_BOOTSTRAP}); ambang peka {FAKTOR_AMBANG_PEKA:g}x")
    print("CATATAN: T1.4 memakai pembagian pasien TUNGGAL — 'SD antar-fold' tidak "
          "terdefinisi di sini\n")

    hasil, err_prod = [], None
    print(f"{'insulin_tau':>12}{'carbs_tau':>11}{'sumbu':>9}{'RMSE':>9}{'MAE':>8}"
          f"{'p-Wilcoxon':>12}{'dtk':>7}")
    for i, (it, ct, sumbu) in enumerate(konf, 1):
        kunci = _kunci(it, ct, "setel")
        t0 = time.time()
        if kunci in tersimpan:
            yte, yp, n_tr = tersimpan[kunci]
            dur = 0.0
        else:
            yte, yp, n_tr = latih_dan_nilai(df_mentah, cfg, mc, feats, H, seq_len, max_gap,
                                            cadence, it, ct, train_p, setel_p, seed)
            dur = time.time() - t0
            simpan_checkpoint(cache_dir, kunci, yte, yp, n_tr)
        r = ringkas(yte, yp)
        adalah_prod = (it == insulin_prod and ct == carbs_prod)
        abs_err = np.abs(yp - yte)
        if adalah_prod:
            err_prod = abs_err
            boot_prod = sd_bootstrap_rmse(yte, yp)
        r.update({"insulin_tau_min": it, "carbs_tau_min": ct, "sumbu_sapuan": sumbu,
                  "adalah_produksi": adalah_prod, "durasi_detik": round(dur, 1),
                  "n_jendela_latih": int(n_tr), "abs_err": abs_err})
        hasil.append(r)
        print(f"{it:>12.0f}{ct:>11.0f}{sumbu:>9}{r['rmse']:>9.3f}{r['mae']:>8.3f}"
              f"{'(acuan)' if adalah_prod else '-':>12}{dur:>7.0f}"
              f"   [{i}/{len(konf)}]")

    if err_prod is None:
        raise SystemExit("Titik produksi tidak ada dalam sapuan — hasil tak dapat dibandingkan.")

    # Uji signifikansi baru dijalankan SETELAH semua konfigurasi berjalan, terhadap acuan
    # produksi yang sudah ditetapkan di muka.
    print("\nUji beda terhadap konfigurasi produksi (Wilcoxon berpasangan atas |galat|):")
    for r in hasil:
        if r["adalah_produksi"]:
            r["p_wilcoxon"], r["berbeda_signifikan"] = None, False
        else:
            try:
                stat, p = wilcoxon(r["abs_err"], err_prod)
                r["p_wilcoxon"] = round(float(p), 6)
                r["berbeda_signifikan"] = bool(p < ALPHA)
            except ValueError:
                r["p_wilcoxon"], r["berbeda_signifikan"] = None, False
        tanda = "BEDA" if r["berbeda_signifikan"] else "setara"
        p_str = f"{r['p_wilcoxon']:.4g}" if r["p_wilcoxon"] is not None else "acuan"
        print(f"  insulin={r['insulin_tau_min']:.0f} carbs={r['carbs_tau_min']:.0f}"
              f" | RMSE {r['rmse']:.3f} | p={p_str} -> {tanda}")
        del r["abs_err"]

    rmse_prod = next(r["rmse"] for r in hasil if r["adalah_produksi"])
    rmse_semua = [r["rmse"] for r in hasil]
    rentang = max(rmse_semua) - min(rmse_semua)
    n_beda = sum(1 for r in hasil if r["berbeda_signifikan"])

    per_sumbu = {}
    for sumbu, nilai_prod in (("insulin", insulin_prod), ("carbs", carbs_prod)):
        anggota = [r for r in hasil
                   if r["sumbu_sapuan"] == sumbu or r["adalah_produksi"]]
        rs = [r["rmse"] for r in anggota]
        per_sumbu[sumbu] = {
            "n_nilai_diuji": len(anggota),
            "rmse_min": min(rs), "rmse_maks": max(rs),
            "rentang_rmse": round(max(rs) - min(rs), 4),
            "rentang_rmse_persen_thd_produksi": round(100 * (max(rs) - min(rs)) / rmse_prod, 2),
            "n_berbeda_signifikan": sum(1 for r in anggota if r["berbeda_signifikan"]),
        }

    # Pembanding untuk set pelaporan ditentukan oleh kriteria yang sudah ditetapkan:
    # konfigurasi dengan RMSE penyetelan TERENDAH. Hanya SATU yang dijalankan di set
    # pelaporan, di samping produksi.
    terbaik = min((r for r in hasil if not r["adalah_produksi"]), key=lambda r: r["rmse"])
    print(f"\nSet PELAPORAN disentuh sekali: produksi vs "
          f"(insulin={terbaik['insulin_tau_min']:.0f}, carbs={terbaik['carbs_tau_min']:.0f})")
    pelaporan = {}
    for nama, it, ct in (("produksi", insulin_prod, carbs_prod),
                         ("pembanding", terbaik["insulin_tau_min"], terbaik["carbs_tau_min"])):
        kunci = _kunci(it, ct, "lapor")
        if kunci in tersimpan:
            yte, yp, _ = tersimpan[kunci]
        else:
            yte, yp, n_tr_l = latih_dan_nilai(df_mentah, cfg, mc, feats, H, seq_len,
                                              max_gap, cadence, it, ct, train_p, lapor_p,
                                              seed)
            simpan_checkpoint(cache_dir, kunci, yte, yp, n_tr_l)
        pelaporan[nama] = ringkas(yte, yp)
        pelaporan[nama].update({"insulin_tau_min": it, "carbs_tau_min": ct,
                                "_abs_err": np.abs(yp - yte)})
        print(f"  {nama:<12} insulin={it:>4.0f} carbs={ct:>4.0f} | "
              f"RMSE {pelaporan[nama]['rmse']:.3f} | MAE {pelaporan[nama]['mae']:.3f}")
    try:
        _, p_lapor = wilcoxon(pelaporan["pembanding"].pop("_abs_err"),
                              pelaporan["produksi"].pop("_abs_err"))
        p_lapor = round(float(p_lapor), 6)
    except ValueError:
        p_lapor = None
    pelaporan["p_wilcoxon"] = p_lapor
    pelaporan["berbeda_signifikan"] = bool(p_lapor is not None and p_lapor < ALPHA)
    print(f"  Wilcoxon set pelaporan: p={p_lapor} -> "
          f"{'BEDA' if pelaporan['berbeda_signifikan'] else 'setara'}")

    # ── VERDIK KEPEKAAN, menurut skala derau yang dideklarasikan di muka ────────────
    ambang = FAKTOR_AMBANG_PEKA * boot_prod["sd_bootstrap"]
    verdik = {
        "definisi_skala_derau": (
            f"SD bootstrap RMSE pada set penyetelan, {N_BOOTSTRAP} resample, benih "
            f"{BENIH_BOOTSTRAP}. DITETAPKAN DI MUKA pada docs/PRAPENDAFTARAN_T1.4.md."),
        "mengapa_bukan_sd_antar_fold": (
            "T1.4 memakai pembagian pasien TUNGGAL, bukan validasi silang 6 fold. "
            "'SD antar-fold' — yang T4.1 temukan punya tiga definisi berbeda dan dapat "
            "memberi verdik berlawanan — TIDAK TERDEFINISI di sini. Dinyatakan agar T1.4 "
            "tidak dikutip seolah punya variabilitas tingkat fold."),
        "bootstrap_konfigurasi_produksi": boot_prod,
        "ambang_peka": round(float(ambang), 4),
        "faktor_ambang": FAKTOR_AMBANG_PEKA,
        "rentang_rmse_sapuan": round(rentang, 4),
        "PEKA": bool(rentang > ambang),
        "sd_deskriptif_9_konfigurasi": round(float(np.std(rmse_semua, ddof=1)), 4),
        "pemeriksaan_kesehatan_thd_J7": {
            "biaya_permutasi_iob": BIAYA_PERMUTASI_IOB,
            "biaya_permutasi_cob": BIAYA_PERMUTASI_COB,
            "rentang_sapuan_di_bawah_biaya_iob": bool(rentang < BIAYA_PERMUTASI_IOB),
            "arti_bila_TIDAK": (
                "Menggeser tau tidak memusnahkan fitur, sehingga pengaruhnya wajib lebih "
                "kecil daripada memusnahkannya. Bila rentang sapuan MELAMPAUI biaya "
                "permutasi iob, yang dicurigai lebih dulu adalah apakah engineer_features "
                "benar-benar dihitung ulang dari data MENTAH tiap konfigurasi — menumpuk "
                "peluruhan dua kali menghasilkan fitur RUSAK, bukan fitur ber-tau lain."),
        },
    }
    print(f"\nVERDIK KEPEKAAN (skala derau ditetapkan di muka)")
    print(f"  RMSE produksi              : {boot_prod['rmse_titik']:.4f} "
          f"(CI95 {boot_prod['ci95'][0]:.4f}–{boot_prod['ci95'][1]:.4f})")
    print(f"  SD bootstrap               : {boot_prod['sd_bootstrap']:.4f} mg/dL")
    print(f"  ambang peka ({FAKTOR_AMBANG_PEKA:g}x SD)        : {ambang:.4f} mg/dL")
    print(f"  rentang RMSE sapuan        : {rentang:.4f} mg/dL")
    print(f"  -> {'PEKA' if verdik['PEKA'] else 'TIDAK PEKA'}: rentang sapuan "
          f"{'melampaui' if verdik['PEKA'] else 'tidak melampaui'} ambang")
    print(f"  pemeriksaan thd J7         : rentang {rentang:.4f} lawan biaya permutasi "
          f"iob {BIAYA_PERMUTASI_IOB} -> "
          f"{'wajar' if verdik['pemeriksaan_kesehatan_thd_J7']['rentang_sapuan_di_bawah_biaya_iob'] else 'JANGGAL, periksa rekayasa fitur'}")

    out = {
        "percobaan": "T1.4 / J3 — sensitivitas insulin_tau_min & carbs_tau_min",
        "verdik_kepekaan": verdik,
        "checkpoint": {"direktori": jalur_pendek(cache_dir),
                       "sidik_jari": sidik,
                       "n_dipulihkan": len(tersimpan)},
        "sifat": ("PENGUKURAN SENSITIVITAS, bukan penyetelan. Nilai produksi tidak diubah "
                  "berdasarkan hasil ini."),
        "horizon_steps": H, "horizon_min": int(H * cadence),
        "protokol_bagian_c": {
            "set_penyetelan": setel_p, "set_pelaporan": lapor_p, "set_latih": train_p,
            "kriteria_ditetapkan_di_muka": ("RMSE pada set penyetelan; beda terhadap "
                                            "produksi diuji Wilcoxon berpasangan atas "
                                            f"|galat|, alpha={ALPHA}"),
            "jumlah_konfigurasi_diuji": len(konf),
            "seluruh_konfigurasi_dicatat": True,
            "set_pelaporan_disentuh_berapa_kali": 1,
        },
        "rancangan": {
            "jenis": "satu-per-satu (OAT) mengelilingi titik produksi",
            "sapuan_insulin_tau_min": SAPU_INSULIN,
            "sapuan_carbs_tau_min": SAPU_CARBS,
            "grid_penuh_tidak_dipakai": ("25 sel x ~9 menit > 3,5 jam, dan sebagian besar "
                                         "selnya tidak menjawab pertanyaan Subbab II.3.5"),
            "keterbatasan": ("OAT tidak dapat menangkap interaksi antara kedua konstanta. "
                             "Bila kedua sumbu datar, interaksinya hampir pasti juga datar; "
                             "bila salah satu berpengaruh besar, interaksi layak diperiksa."),
        },
        "produksi": {"insulin_tau_min": insulin_prod, "carbs_tau_min": carbs_prod,
                     "rmse_penyetelan": rmse_prod},
        "hasil_set_penyetelan": hasil,
        "per_sumbu": per_sumbu,
        "rentang_rmse_seluruh_konfigurasi": round(rentang, 4),
        "rentang_rmse_persen_thd_produksi": round(100 * rentang / rmse_prod, 2),
        "n_konfigurasi_berbeda_signifikan": n_beda,
        "hasil_set_pelaporan": pelaporan,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"sensitivitas_tau_h{H}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nRentang RMSE seluruh {len(konf)} konfigurasi: {rentang:.3f} mg/dL "
          f"({100 * rentang / rmse_prod:.2f}% dari RMSE produksi)")
    print(f"Konfigurasi yang berbeda signifikan dari produksi: {n_beda}/{len(konf) - 1}")
    for sumbu, v in per_sumbu.items():
        print(f"  sumbu {sumbu:<8}: rentang RMSE {v['rentang_rmse']:.3f} "
              f"({v['rentang_rmse_persen_thd_produksi']:.2f}%), "
              f"{v['n_berbeda_signifikan']} berbeda signifikan")
    print(f"\nDisimpan ke {p}")


if __name__ == "__main__":
    main()
