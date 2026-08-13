"""T4.1 — Gradient Boosting sebagai pembanding KETIGA, di samping RF dan LSTM.

MENGAPA PERLU PEMBANDING KETIGA
-------------------------------
Perbandingan dua model selalu dapat dibantah dengan "bagaimana kalau keluarga model lain
jauh lebih baik?". RF dan LSTM mewakili dua kutub (ansambel pohon bagging vs jaringan
rekuren), tetapi *gradient boosting* adalah keluarga yang paling sering menang pada data
tabular dan justru tidak terwakili. Tanpa itu, pembelaan atas pemilihan RF bertumpu pada
himpunan pembanding yang timpang.

TANPA PENYETELAN EKSTENSIF — DAN ITU DISENGAJA
----------------------------------------------
Hiperparameter dibiarkan pada bawaan scikit-learn, kecuali benih acak. Alasannya bukan
kemalasan: RF produksi juga tidak pernah disetel secara ekstensif (200/20/5 dipilih di
awal dan bertahan), sehingga menyetel GBM tetapi tidak menyetel RF akan membuat
perbandingannya berat sebelah ke arah yang berlawanan. Konsekuensinya dicatat terus
terang: **hasil ini adalah batas BAWAH kemampuan gradient boosting**, dan bila GBM menang
pada keadaan tak-tersetel itu, keunggulannya justru lebih meyakinkan; bila GBM kalah,
kesimpulannya hanya berlaku untuk keadaan tak-tersetel.

`HistGradientBoostingRegressor`, bukan `GradientBoostingRegressor`. Yang kedua membangun
pohon secara persis dan pada 104 ribu jendela x 84 fitur akan makan berjam-jam per fold;
yang pertama memakai histogram dan sepadan dengan anggaran komputasi RF. Ini pilihan
implementasi, bukan pilihan keluarga model.

PEMBANDINGAN YANG SAH
---------------------
Fold ditentukan dengan aturan yang IDENTIK dengan `crossval_rf_vs_lstm.py`
(``pids[i::6]``), sehingga angka LSTM yang sudah tersimpan dapat dibandingkan pada
tingkat fold tanpa melatih ulang LSTM. RF DILATIH ULANG di sini — bukan diambil dari
berkas — karena uji berpasangan tingkat sampel menuntut galat per sampel, dan sekaligus
menjadi pemeriksaan reproduksibilitas: RMSE per fold yang dihasilkan harus sama dengan
yang tersimpan. Bila tidak sama, ada sesuatu yang berubah dan seluruh perbandingan
ditandai tidak sepadan.

SENSITIVITAS HIPOGLIKEMIA IKUT DIUKUR
------------------------------------
RMSE dan Clarke tidak menyentuh pertanyaan yang paling menentukan bagi pemilihan model.
T2.1 menunjukkan LSTM unggul atas RF pada **sensitivitas hipoglikemia** di kedua horizon
(21,7% lawan 32,1%; 4,4% lawan 10,6%), dan itulah yang menantang keputusan mempertahankan
RF. Membandingkan GBM hanya pada RMSE meninggalkan aspek paling kritis secara klinis tanpa
pembanding ketiga.

Definisinya **dipakai ulang** dari `eval_hipoglikemia.py` lewat `metrik_hipo()`, bukan
ditulis ulang, sehingga ambang 70 dan 54, penanganan pembagi nol, dan konvensi tanda bias
identik dengan T2.1. Angka LSTM diambil dari `hipoglikemia_h{N}.json` yang aturan foldnya
(`pids[i::6]`) sudah identik — LSTM tidak dilatih ulang di sini.

CHECKPOINT PER FOLD
-------------------
Pelajaran T1.1b: sepuluh konfigurasi dan ~40 menit hilang karena keluaran hanya ditulis di
akhir. Tiap fold menyimpan `yte`, `yp_rf`, dan `yp_gb` ke `.cache/` — bukan `results/` —
sehingga galat per sampel tersedia dan uji Wilcoxon dapat dilanjutkan tanpa melatih ulang.
Metadata horizon, `max_gap_steps`, benih, dan pembagian fold ikut disimpan; checkpoint yang
tidak sepadan **ditolak dengan pesan jelas, bukan dipakai diam-diam**.

Keluaran: results/eval_prediksi/gradient_boosting_h{N}.json
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.loader import DiabetesDataLoader  # noqa: E402
from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.utils.metrics import calculate_all_metrics  # noqa: E402
# Definisi metrik hipoglikemia dari SATU sumber kebenaran (T2.1). Menulis ulang ambang
# 70/54 di sini akan membuat angka GBM tidak sepadan dengan angka RF dan LSTM.
from eval_hipoglikemia import metrik_hipo  # noqa: E402

K = 6
OUT_DIR = ROOT / "results/eval_prediksi"
CACHE_ROOT = ROOT / ".cache"
ZONA = ["Clarke_A", "Clarke_B", "Clarke_C", "Clarke_D", "Clarke_E"]
TOLERANSI_REPRODUKSI = 0.05   # mg/dL RMSE


def jalur_pendek(p: Path) -> str:
    """Path relatif terhadap ROOT bila memungkinkan; kalau tidak, apa adanya.

    ``relative_to`` melempar ValueError untuk path di luar repo, dan itu sempat
    menggagalkan seluruh jalan hanya karena sebuah baris cetak.
    """
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sidik_jari(H, max_gap, folds, seed, rf_cfg, feats) -> dict:
    """Identitas run. Checkpoint dengan sidik jari berbeda ditolak, bukan dipakai."""
    return {"horizon_steps": int(H), "max_gap_steps": int(max_gap), "k_fold": int(K),
            "folds": [list(f) for f in folds], "seed": int(seed),
            "random_forest": dict(rf_cfg), "fitur": list(feats),
            "gbm": "HistGradientBoostingRegressor(bawaan, random_state=seed)"}


def muat_checkpoint(cache_dir: Path, sidik: dict) -> dict:
    """Kembalikan {fold_index: data} yang sepadan. Yang tidak sepadan ditolak."""
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
        print(f"  (checkpoint tidak sepadan lebih berbahaya daripada tidak ada checkpoint)")
        return {}
    ada = {}
    for p in sorted(cache_dir.glob("fold*.npz")):
        i = int(p.stem.replace("fold", ""))
        try:
            z = np.load(p)
            ringkas_p = cache_dir / f"fold{i}.json"
            ada[i] = {"yte": z["yte"], "yp_rf": z["yp_rf"], "yp_gb": z["yp_gb"],
                      "ringkas": json.loads(ringkas_p.read_text(encoding="utf-8"))}
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            print(f"  fold {i} pada checkpoint rusak, akan dihitung ulang: {exc}")
    return ada


def simpan_checkpoint(cache_dir: Path, i: int, yte, yp_rf, yp_gb, ringkas: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / f"fold{i}.npz", yte=yte, yp_rf=yp_rf, yp_gb=yp_gb)
    (cache_dir / f"fold{i}.json").write_text(
        json.dumps(ringkas, indent=2, ensure_ascii=False), encoding="utf-8")


def ukuran_mb(model) -> float:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
        pickle.dump(model, f)
        p = f.name
    n = os.path.getsize(p)
    os.unlink(p)
    return round(n / 1e6, 3)


def ringkas(y, yp) -> dict:
    m = calculate_all_metrics(y, yp)
    return {k: round(float(m[k]), 3)
            for k in ("RMSE", "MAE", "MAPE", "Clarke_A+B", *ZONA) if k in m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()
    H = args.horizon

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    m = cfg["model"]
    seq_len = m["sequence_length"]
    feats = m["engineered_features"] if m.get("use_engineered") else m["features"]
    predict_delta = m.get("predict_delta", False)
    rf_cfg = m["random_forest"]
    seed = cfg["data"].get("seed", 42)
    max_gap = m.get("max_gap_steps")
    cadence = float(cfg["data"]["sampling_interval_min"])
    if max_gap is None:
        raise SystemExit("config.model.max_gap_steps tidak ada.")

    lama_path = OUT_DIR / f"crossval_rf_vs_lstm_h{H}.json"
    lama = json.loads(lama_path.read_text(encoding="utf-8")) if lama_path.exists() else None
    if lama is None:
        print(f"CATATAN: {lama_path.name} tidak ada — LSTM tidak dapat disertakan, "
              f"dan pemeriksaan reproduksibilitas RF dilewati.")

    # Metrik hipoglikemia LSTM diambil dari T2.1, bukan dilatih ulang. Aturan foldnya
    # sudah identik (pids[i::6]), sehingga per-fold-nya dapat disandingkan langsung.
    hipo_path = OUT_DIR / f"hipoglikemia_h{H}.json"
    hipo_lama = json.loads(hipo_path.read_text(encoding="utf-8")) \
        if hipo_path.exists() else None
    if hipo_lama is None:
        print(f"CATATAN: {hipo_path.name} tidak ada — metrik hipoglikemia LSTM tidak "
              f"dapat disandingkan; GBM dan RF tetap dihitung.")

    loader = DiabetesDataLoader(cfg["data"]["output_dir"])
    df = loader.load_csv("ohio_t1dm_merged.csv") \
        .sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    prep = DataPreprocessor(cfg)
    df = prep.handle_missing_values(df)
    if m.get("use_engineered"):
        df = prep.engineer_features(df, **m["feature_engineering"])

    pids = sorted(df["patient_id"].unique().tolist())
    folds = [pids[i::K] for i in range(K)]   # aturan IDENTIK dengan crossval_rf_vs_lstm.py

    print(f"T4.1 — Gradient Boosting sebagai pembanding ketiga")
    print(f"horizon +{int(H * cadence)} mnt | {K} fold lintas-pasien | "
          f"max_gap_steps={max_gap}")
    print("GBM memakai hiperparameter BAWAAN (tanpa penyetelan) — disengaja; "
          "hasilnya batas BAWAH kemampuan gradient boosting\n")

    sidik = sidik_jari(H, max_gap, folds, seed, rf_cfg, feats)
    cache_dir = CACHE_ROOT / f"gradient_boosting_h{H}"
    print(f"checkpoint : {jalur_pendek(cache_dir)}")
    tersimpan = muat_checkpoint(cache_dir, sidik)
    if tersimpan:
        print(f"  {len(tersimpan)} fold dipulihkan dari checkpoint: "
              f"{sorted(tersimpan)}  (tidak dilatih ulang)")
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "meta.json").write_text(json.dumps(sidik, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    print()

    baris, err_rf, err_gb = [], [], []
    t_rf, t_gb, inf_rf, inf_gb, sz_rf, sz_gb = [], [], [], [], [], []
    beda_reproduksi = []

    for fi, test_p in enumerate(folds):
        if fi in tersimpan:
            c = tersimpan[fi]
            yte, yp_rf, yp_gb = c["yte"], c["yp_rf"], c["yp_gb"]
            biaya = c["ringkas"]["biaya"]
        else:
            prep.feature_columns = list(feats)
            tr, te = prep.split_by_patient(df, test_p)
            kw = {"max_gap_steps": max_gap, "source_interval_min": cadence}
            Xtr, ytr, atr = prep.create_sequences(tr, seq_len, H, return_anchor=True, **kw)
            Xte, yte, ate = prep.create_sequences(te, seq_len, H, return_anchor=True, **kw)
            p2 = DataPreprocessor(cfg)
            Xtr_s, Xte_s = p2.normalize_data(Xtr, Xte)
            Ftr, Fte = Xtr_s.reshape(len(ytr), -1), Xte_s.reshape(len(yte), -1)
            ytr_fit = (ytr - atr) if predict_delta else ytr

            rfm = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
                                        max_depth=rf_cfg["max_depth"],
                                        min_samples_split=rf_cfg["min_samples_split"],
                                        random_state=seed, n_jobs=-1)
            t0 = time.time(); rfm.fit(Ftr, ytr_fit); dur_rf = time.time() - t0
            t0 = time.time(); yp_rf = rfm.predict(Fte)
            i_rf = (time.time() - t0) / max(len(yte), 1) * 1000
            s_rf = ukuran_mb(rfm)
            yp_rf = yp_rf + ate if predict_delta else yp_rf

            gbm = HistGradientBoostingRegressor(random_state=seed)
            t0 = time.time(); gbm.fit(Ftr, ytr_fit); dur_gb = time.time() - t0
            t0 = time.time(); yp_gb = gbm.predict(Fte)
            i_gb = (time.time() - t0) / max(len(yte), 1) * 1000
            s_gb = ukuran_mb(gbm)
            yp_gb = yp_gb + ate if predict_delta else yp_gb

            biaya = {"latih_rf_dtk": dur_rf, "latih_gb_dtk": dur_gb,
                     "inf_rf_ms": i_rf, "inf_gb_ms": i_gb,
                     "ukuran_rf_MB": s_rf, "ukuran_gb_MB": s_gb}

        t_rf.append(biaya["latih_rf_dtk"]); t_gb.append(biaya["latih_gb_dtk"])
        inf_rf.append(biaya["inf_rf_ms"]); inf_gb.append(biaya["inf_gb_ms"])
        sz_rf.append(biaya["ukuran_rf_MB"]); sz_gb.append(biaya["ukuran_gb_MB"])

        r_rf, r_gb = ringkas(yte, yp_rf), ringkas(yte, yp_gb)
        # Definisi identik T2.1, diimpor bukan ditulis ulang.
        h_rf, h_gb = metrik_hipo(yte, yp_rf), metrik_hipo(yte, yp_gb)
        err_rf.append(np.abs(yte - yp_rf))
        err_gb.append(np.abs(yte - yp_gb))

        r_ls = None
        if lama:
            cocok = [f for f in lama["folds"] if f["fold"] == fi]
            if cocok:
                r_ls = cocok[0]["LSTM"]
                beda = abs(cocok[0]["RF"]["RMSE"] - r_rf["RMSE"])
                beda_reproduksi.append(round(beda, 4))
        h_ls = None
        if hipo_lama:
            cocok_h = [f for f in hipo_lama["folds"] if f["fold"] == fi]
            if cocok_h:
                h_ls = cocok_h[0].get("LSTM")

        ringkas_fold = {"fold": fi, "test": test_p, "n": int(len(yte)),
                        "RF": r_rf, "GBM": r_gb, "LSTM": r_ls,
                        "hipo": {"RF": h_rf, "GBM": h_gb, "LSTM": h_ls},
                        "biaya": biaya}
        baris.append(ringkas_fold)
        if fi not in tersimpan:
            simpan_checkpoint(cache_dir, fi, yte, yp_rf, yp_gb, ringkas_fold)

        ls_txt = f" | LSTM {r_ls['RMSE']:.2f}" if r_ls else ""
        tanda = " [checkpoint]" if fi in tersimpan else ""
        print(f"  fold {fi} {test_p}: n={len(yte):>6} | RF {r_rf['RMSE']:.2f} "
              f"| GBM {r_gb['RMSE']:.2f}{ls_txt} "
              f"| sens hipo RF {h_rf['deteksi']['sensitivitas_persen']}% "
              f"GBM {h_gb['deteksi']['sensitivitas_persen']}%"
              f" | latih RF {t_rf[-1]:.0f}s GBM {t_gb[-1]:.0f}s{tanda}", flush=True)

    def rerata(mdl, k):
        v = [b[mdl][k] for b in baris if b[mdl] and k in b[mdl]]
        return [round(float(np.mean(v)), 3), round(float(np.std(v)), 3)] if v else None

    model_ada = ["RF", "GBM"] + (["LSTM"] if lama else [])
    rerata_semua = {mdl: {k: rerata(mdl, k)
                          for k in ("RMSE", "MAE", "MAPE", "Clarke_A+B", *ZONA)}
                    for mdl in model_ada}

    # ── Hipoglikemia: agregasi lintas fold, susunan sama seperti T2.1 ────────────────
    def _gali_hipo(h, k):
        if not h:
            return None
        return {"sensitivitas_persen": h["deteksi"]["sensitivitas_persen"],
                "spesifisitas_persen": h["deteksi"]["spesifisitas_persen"],
                "ppv_persen": h["deteksi"]["ppv_persen"],
                "f1_persen": h["deteksi"]["f1_persen"],
                "rmse_pada_hipo": (h.get("galat_pada_hipo") or {}).get("rmse"),
                "mae_pada_hipo": (h.get("galat_pada_hipo") or {}).get("mae"),
                "bias_pada_hipo": (h.get("galat_pada_hipo") or {}).get("bias"),
                "bias_keseluruhan": h.get("bias_keseluruhan")}.get(k)

    KUNCI_HIPO = ("sensitivitas_persen", "spesifisitas_persen", "ppv_persen", "f1_persen",
                  "rmse_pada_hipo", "mae_pada_hipo", "bias_pada_hipo", "bias_keseluruhan")
    model_hipo = [m for m in ("RF", "GBM", "LSTM")
                  if any(b["hipo"].get(m) for b in baris)]
    hipo_rerata = {}
    for mdl in model_hipo:
        hipo_rerata[mdl] = {}
        for k in KUNCI_HIPO:
            v = [_gali_hipo(b["hipo"][mdl], k) for b in baris if b["hipo"].get(mdl)]
            v = [x for x in v if x is not None]
            hipo_rerata[mdl][k] = ([round(float(np.mean(v)), 3),
                                    round(float(np.std(v)), 3), len(v)] if v else None)
    hipo_total = {
        "n_hipo_benar": int(sum(b["hipo"]["RF"]["n_hipo_benar"] for b in baris)),
        "n_total": int(sum(b["hipo"]["RF"]["n_total"] for b in baris)),
        "terlewat_ke_rentang_target": {
            m: int(sum(b["hipo"][m]["terlewat_ke_rentang_target"] for b in baris))
            for m in model_hipo if all(b["hipo"].get(m) for b in baris)},
        "hipo_berat_terlewat": {
            m: int(sum(b["hipo"][m]["hipo_berat_terlewat"] for b in baris))
            for m in model_hipo if all(b["hipo"].get(m) for b in baris)},
    }
    hipo_total["porsi_hipo_persen"] = round(
        100 * hipo_total["n_hipo_benar"] / max(hipo_total["n_total"], 1), 3)

    # Uji berpasangan tingkat fold, dengan penjaga "selisih harus melampaui SD antar-fold"
    # — aturan pelaporan yang sama dengan T2.1.
    uji_hipo = {}
    for a, b_ in (("RF", "GBM"), ("GBM", "LSTM"), ("RF", "LSTM")):
        if a not in model_hipo or b_ not in model_hipo:
            continue
        for k in ("sensitivitas_persen", "mae_pada_hipo", "bias_pada_hipo"):
            va = [_gali_hipo(x["hipo"][a], k) for x in baris if x["hipo"].get(a)]
            vb = [_gali_hipo(x["hipo"][b_], k) for x in baris if x["hipo"].get(b_)]
            if len(va) != len(vb) or not va or any(x is None for x in va + vb):
                continue
            va_a = np.array(va, dtype=float)
            vb_a = np.array(vb, dtype=float)
            d = va_a - vb_a
            try:
                p = float(stats.wilcoxon(va, vb).pvalue)
            except ValueError:
                p = float("nan")
            # TIGA definisi "SD antar-fold", dan ketiganya dapat memberi verdik BERBEDA.
            # Ditemukan saat T4.1: aturan pelaporan HANDOFF ("selisih lebih kecil daripada
            # SD antar-fold tidak boleh dinarasikan") tidak menyebut SD yang mana.
            #
            #  * gabungan  — yang dipakai T2.1: np.std atas 12 nilai kedua model disatukan.
            #                Melebar bila kedua model memang berbeda, yaitu bila efek yang
            #                sedang diuji besar. Statistik yang aneh untuk tujuan ini.
            #  * maks      — maks SD per model. Memuat variasi antar-pasien yang justru
            #                saling hapus pada rancangan berpasangan, sehingga terlalu ketat.
            #  * selisih   — SD dari selisih berpasangan. Untuk rancangan berpasangan inilah
            #                yang mengukur KEKONSISTENAN efek antar-fold, dan karena itu
            #                yang paling dapat dipertahankan.
            sd = {
                "gabungan_definisi_T2.1": round(float(np.std(np.concatenate([va_a, vb_a]))), 3),
                "maks_per_model": round(max(float(va_a.std()), float(vb_a.std())), 3),
                "selisih_berpasangan": round(float(d.std()), 3),
            }
            lewat = {nama: bool(abs(d.mean()) > v) for nama, v in sd.items()}
            uji_hipo[f"{a}_vs_{b_}__{k}"] = {
                "selisih_rerata": round(float(d.mean()), 3),
                "sd_antar_fold": sd,
                "melampaui_sd": lewat,
                "sd_sepakat": bool(len(set(lewat.values())) == 1),
                "wilcoxon_p": None if p != p else round(p, 6),
                # Verdik memakai definisi TERKETAT yang berlaku: bila ketiga definisi tidak
                # sepakat, klaimnya rapuh dan tidak dinarasikan tanpa menyebut definisinya.
                "layak_dinarasikan_konservatif": bool(
                    p == p and p < 0.05 and all(lewat.values())),
                "layak_menurut_definisi_T2.1": bool(
                    p == p and p < 0.05 and lewat["gabungan_definisi_T2.1"]),
            }

    rf_f = np.array([b["RF"]["RMSE"] for b in baris])
    gb_f = np.array([b["GBM"]["RMSE"] for b in baris])
    e_rf, e_gb = np.concatenate(err_rf), np.concatenate(err_gb)
    d_f = rf_f - gb_f
    # KEPUTUSAN #8 (11 Agustus 2026, docs/KEPUTUSAN_DIAMBIL.md): dua pertanyaan DIPISAH.
    #   konsistensi  -> SD selisih berpasangan std(A-B) + Wilcoxon tingkat fold
    #   kebermaknaan -> ambang orde besaran ISO 15197:2013, +-15 mg/dL untuk glukosa
    #                   < 100 mg/dL (KB-02_PERKENI-2021 hal. 25)
    # Ketiga definisi SD tetap dilaporkan: menetapkan satu tidak menghapus kewajiban
    # menunjukkan ketiganya.
    AMBANG_KLINIS_MGDL = 15.0
    sd_tiga = {
        "gabungan_definisi_T2.1": round(float(np.std(np.concatenate([rf_f, gb_f]))), 3),
        "maks_per_model": round(float(max(rf_f.std(), gb_f.std())), 3),
        "selisih_berpasangan": round(float(d_f.std()), 3),
    }
    lewat_tiga = {k: bool(abs(d_f.mean()) > v) for k, v in sd_tiga.items()}
    persen_ambang = round(100 * abs(float(d_f.mean())) / AMBANG_KLINIS_MGDL, 2)
    uji = {
        "RF_vs_GBM": {
            "fold_level": {
                "rerata_RMSE_RF": round(float(rf_f.mean()), 3),
                "rerata_RMSE_GBM": round(float(gb_f.mean()), 3),
                "selisih": round(float((rf_f - gb_f).mean()), 3),
                "sd_antar_fold_RF": round(float(rf_f.std()), 3),
                "sd_antar_fold_GBM": round(float(gb_f.std()), 3),
                "sd_tiga_definisi": sd_tiga,
                "melampaui_sd": lewat_tiga,
                "sd_sepakat": bool(len(set(lewat_tiga.values())) == 1),
                "KONSISTEN_definisi_terpilih": lewat_tiga["selisih_berpasangan"],
                "ambang_klinis_mg_dL": AMBANG_KLINIS_MGDL,
                "persen_dari_ambang_klinis": persen_ambang,
                "BERMAKNA_secara_klinis": bool(abs(float(d_f.mean())) >= AMBANG_KLINIS_MGDL),
                "RUMUSAN": (
                    "KONSISTEN TETAPI TIDAK BERMAKNA"
                    if lewat_tiga["selisih_berpasangan"]
                    and abs(float(d_f.mean())) < AMBANG_KLINIS_MGDL
                    else ("tidak konsisten" if not lewat_tiga["selisih_berpasangan"]
                          else "konsisten dan bermakna")),
                "keputusan_8": (
                    "docs/KEPUTUSAN_DIAMBIL.md. Konsistensi diukur SD selisih berpasangan; "
                    "kebermaknaan diukur ambang ORDE BESARAN dari ISO 15197:2013 (+-15 mg/dL "
                    "untuk glukosa < 100 mg/dL, KB-02_PERKENI-2021 hal. 25). Ambang itu "
                    "ANALOGI BERDASAR LITERATUR, bukan penerapan ISO pada perbandingan "
                    "model, dan diadopsi SETELAH hasil terlihat."),
                "wilcoxon_p": round(float(stats.wilcoxon(rf_f, gb_f).pvalue), 4),
                "ttest_rel_p": round(float(stats.ttest_rel(rf_f, gb_f).pvalue), 4),
                "selisih_melampaui_sd": bool(abs((rf_f - gb_f).mean()) >
                                             max(rf_f.std(), gb_f.std())),
            },
            "sample_level": {
                "wilcoxon_p": float(stats.wilcoxon(e_rf, e_gb).pvalue),
                "n": int(len(e_rf)),
                "median_abs_err_RF": round(float(np.median(e_rf)), 2),
                "median_abs_err_GBM": round(float(np.median(e_gb)), 2),
            },
        },
    }
    if lama:
        ls_f = np.array([b["LSTM"]["RMSE"] for b in baris if b["LSTM"]])
        if len(ls_f) == len(gb_f):
            uji["GBM_vs_LSTM"] = {
                "fold_level": {
                    "rerata_RMSE_GBM": round(float(gb_f.mean()), 3),
                    "rerata_RMSE_LSTM": round(float(ls_f.mean()), 3),
                    "selisih": round(float((gb_f - ls_f).mean()), 3),
                    "wilcoxon_p": round(float(stats.wilcoxon(gb_f, ls_f).pvalue), 4),
                    "catatan": ("Hanya tingkat fold. LSTM tidak dilatih ulang di sini, "
                                "sehingga galat per sampelnya tidak tersedia dan uji "
                                "tingkat sampel TIDAK dapat dilakukan."),
                },
            }

    reproduksi = None
    if beda_reproduksi:
        maks = max(beda_reproduksi)
        reproduksi = {
            "selisih_RMSE_per_fold": beda_reproduksi,
            "selisih_maks": maks,
            "toleransi": TOLERANSI_REPRODUKSI,
            "RF_tereproduksi": bool(maks <= TOLERANSI_REPRODUKSI),
        }

    out = {
        "percobaan": "T4.1 — Gradient Boosting sebagai pembanding ketiga",
        "horizon_steps": H, "horizon_min": int(H * cadence),
        "k_fold": K, "max_gap_steps": max_gap,
        "gbm": {
            "kelas": "sklearn.ensemble.HistGradientBoostingRegressor",
            "hiperparameter": "BAWAAN scikit-learn, kecuali random_state",
            "alasan_tanpa_penyetelan": (
                "RF produksi juga tidak pernah disetel ekstensif. Menyetel GBM tetapi "
                "tidak menyetel RF akan membuat perbandingan berat sebelah."),
            "akibat": ("Hasil ini batas BAWAH kemampuan gradient boosting. Bila GBM "
                       "menang, keunggulannya lebih meyakinkan; bila kalah, kesimpulannya "
                       "hanya berlaku untuk keadaan tak-tersetel."),
            "alasan_hist": ("GradientBoostingRegressor membangun pohon persis dan tidak "
                            "sepadan anggaran komputasinya pada 100+ ribu jendela. "
                            "Pilihan implementasi, bukan pilihan keluarga model."),
        },
        "pemeriksaan_reproduksibilitas_RF": reproduksi,
        "folds": baris,
        "rerata_lintas_fold": rerata_semua,
        "hipoglikemia": {
            "catatan": ("Definisi diimpor dari eval_hipoglikemia.metrik_hipo() — ambang "
                        "70 dan 54 mg/dL, konvensi tanda bias, dan penanganan pembagi nol "
                        "identik dengan T2.1. LSTM TIDAK dilatih ulang di sini; angkanya "
                        "diambil dari hipoglikemia_h{H}.json yang aturan foldnya sama."),
            "basis_kejadian": hipo_total,
            "rerata_lintas_fold": hipo_rerata,
            "uji_berpasangan_tingkat_fold": uji_hipo,
            "aturan_pelaporan": (
                "KEPUTUSAN #8, 11 Agustus 2026 (docs/KEPUTUSAN_DIAMBIL.md). Dua pertanyaan "
                "DIPISAH. Konsistensi: SD selisih berpasangan std(A-B) + Wilcoxon tingkat "
                "fold. Kebermaknaan: ambang ORDE BESARAN dari ISO 15197:2013, +-15 mg/dL "
                "untuk glukosa < 100 mg/dL. Ketiga definisi SD tetap dilaporkan. Ambang "
                "klinis TIDAK berlaku bagi besaran bersatuan poin persen (mis. "
                "sensitivitas); bagi besaran itu hanya konsistensi yang dapat dijawab."),
        },
        "checkpoint": {
            "direktori": jalur_pendek(cache_dir),
            "sidik_jari": sidik,
            "n_fold_dipulihkan": len(tersimpan),
            "isi": ("yte, yp_rf, yp_gb per fold — galat per sampel dapat diturunkan "
                    "sehingga Wilcoxon dapat dilanjutkan tanpa melatih ulang."),
        },
        "biaya": {
            "RF": {"waktu_latih_dtk": round(float(np.mean(t_rf)), 1),
                   "inferensi_ms_per_sampel": round(float(np.mean(inf_rf)), 4),
                   "ukuran_model_MB": round(float(np.mean(sz_rf)), 2)},
            "GBM": {"waktu_latih_dtk": round(float(np.mean(t_gb)), 1),
                    "inferensi_ms_per_sampel": round(float(np.mean(inf_gb)), 4),
                    "ukuran_model_MB": round(float(np.mean(sz_gb)), 3)},
        },
        "uji": uji,
        "aturan_pelaporan": ("Selisih yang lebih kecil daripada simpangan baku antar-fold "
                             "tidak boleh dinarasikan sebagai keunggulan."),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f"gradient_boosting_h{H}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== Tiga model, +{out['horizon_min']} mnt, {K} fold ===")
    print(f"{'metrik':<14}" + "".join(f"{mdl:>18}" for mdl in model_ada))
    for k in ("RMSE", "MAE", "MAPE", "Clarke_A+B", *ZONA):
        sel = [rerata_semua[mdl].get(k) for mdl in model_ada]
        if all(sel):
            print(f"{k:<14}" + "".join(f"{v[0]:>10.2f} ±{v[1]:<6.2f}" for v in sel))
    b = out["biaya"]
    print(f"\n{'biaya':<24}{'RF':>14}{'GBM':>14}")
    print(f"{'waktu latih (dtk)':<24}{b['RF']['waktu_latih_dtk']:>14.1f}"
          f"{b['GBM']['waktu_latih_dtk']:>14.1f}")
    print(f"{'inferensi (ms/sampel)':<24}{b['RF']['inferensi_ms_per_sampel']:>14.4f}"
          f"{b['GBM']['inferensi_ms_per_sampel']:>14.4f}")
    print(f"{'ukuran model (MB)':<24}{b['RF']['ukuran_model_MB']:>14.2f}"
          f"{b['GBM']['ukuran_model_MB']:>14.3f}")

    if reproduksi:
        status = "SESUAI" if reproduksi["RF_tereproduksi"] else "TIDAK SESUAI"
        print(f"\nReproduksibilitas RF thd crossval tersimpan: {status} "
              f"(selisih RMSE maks {reproduksi['selisih_maks']} mg/dL, "
              f"toleransi {TOLERANSI_REPRODUKSI})")
        if not reproduksi["RF_tereproduksi"]:
            print("  PERINGATAN: RF tidak tereproduksi — angka LSTM dari berkas lama "
                  "TIDAK sepadan dan tidak boleh disandingkan.")

    fl = uji["RF_vs_GBM"]["fold_level"]
    s3 = fl["sd_tiga_definisi"]
    print(f"\nRF vs GBM tingkat fold: selisih RMSE {fl['selisih']:+.3f} mg/dL "
          f"| Wilcoxon p={fl['wilcoxon_p']}")
    print(f"  SD antar-fold  : gabungan {s3['gabungan_definisi_T2.1']:.3f} | "
          f"maks {s3['maks_per_model']:.3f} | SELISIH {s3['selisih_berpasangan']:.3f}"
          f"   (sepakat: {'ya' if fl['sd_sepakat'] else 'TIDAK'})")
    print(f"  KONSISTEN?     : {'ya' if fl['KONSISTEN_definisi_terpilih'] else 'tidak'}"
          f"   (definisi terpilih #8: SD selisih berpasangan)")
    print(f"  BERMAKNA?      : {'ya' if fl['BERMAKNA_secara_klinis'] else 'tidak'}"
          f"   ({fl['persen_dari_ambang_klinis']}% dari ambang "
          f"{fl['ambang_klinis_mg_dL']:.0f} mg/dL, ISO 15197 sebagai ORDE BESARAN)")
    print(f"  -> {fl['RUMUSAN']}")
    print(f"RF vs GBM tingkat sampel: Wilcoxon p={uji['RF_vs_GBM']['sample_level']['wilcoxon_p']:.2e} "
          f"(n={uji['RF_vs_GBM']['sample_level']['n']:,})")

    # ── Hipoglikemia — aspek yang menantang keputusan mempertahankan RF ─────────────
    print(f"\n=== Kejadian hipoglikemia (<70 mg/dL): "
          f"{hipo_total['n_hipo_benar']:,} dari {hipo_total['n_total']:,} jendela "
          f"({hipo_total['porsi_hipo_persen']}%) ===")
    print(f"{'metrik':<22}" + "".join(f"{m:>18}" for m in model_hipo))
    for k in KUNCI_HIPO:
        sel = [hipo_rerata[m].get(k) for m in model_hipo]
        if all(sel):
            print(f"{k:<22}" + "".join(f"{v[0]:>10.2f} ±{v[1]:<6.2f}" for v in sel))
    for nama, kunci in (("terlewat ke rentang target", "terlewat_ke_rentang_target"),
                        ("hipo BERAT terlewat", "hipo_berat_terlewat")):
        d = hipo_total[kunci]
        if d:
            print(f"{nama:<22}" + "".join(f"{d.get(m, 0):>18,}" for m in model_hipo))

    if uji_hipo:
        print("\nUji berpasangan tingkat fold pada hipoglikemia")
        print("  SD antar-fold dilaporkan menurut TIGA definisi; kolom 'sepakat' menandai "
              "apakah ketiganya\n  memberi verdik sama. Yang tidak sepakat berarti klaimnya "
              "RAPUH terhadap pilihan definisi.")
        print(f"  {'perbandingan':<42}{'selisih':>9}{'SD gab':>9}{'SD maks':>9}"
              f"{'SD selisih':>12}{'p':>10}  verdik")
        for k, v in uji_hipo.items():
            s = v["sd_antar_fold"]
            if v["layak_dinarasikan_konservatif"]:
                verdik = "LAYAK (ketiga definisi sepakat)"
            elif v["layak_menurut_definisi_T2.1"]:
                verdik = "RAPUH — layak hanya menurut definisi T2.1"
            elif v["wilcoxon_p"] is not None and v["wilcoxon_p"] < 0.05:
                verdik = "DI BAWAH SD — jangan dinarasikan"
            else:
                verdik = "p tidak signifikan"
            print(f"  {k:<42}{v['selisih_rerata']:>+9.3f}"
                  f"{s['gabungan_definisi_T2.1']:>9.3f}{s['maks_per_model']:>9.3f}"
                  f"{s['selisih_berpasangan']:>12.3f}{str(v['wilcoxon_p']):>10}  {verdik}")
        rapuh = [k for k, v in uji_hipo.items() if not v["sd_sepakat"]]
        if rapuh:
            print(f"\n  PERINGATAN: {len(rapuh)} perbandingan verdiknya BERGANTUNG pada "
                  f"definisi SD yang dipilih:")
            for k in rapuh:
                print(f"    - {k}")
            print("  Aturan pelaporan HANDOFF tidak menyebut SD yang mana. Sampai "
                  "definisinya ditetapkan\n  pembimbing, klaim ini tidak dinarasikan tanpa "
                  "menyebut definisi yang dipakai.")

    print(f"\nDisimpan ke {p}")
    print(f"Checkpoint tetap di {jalur_pendek(cache_dir)} — hapus setelah hasil "
          f"akhir diperiksa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
