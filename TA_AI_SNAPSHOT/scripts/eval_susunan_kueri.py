"""T3.1 — susunan kalimat kueri prediction-conditioned.

MENGAPA PERCOBAAN INI NAIK NILAINYA
-----------------------------------
Tiga bukti terpisah menunjuk ke arah yang sama: **retrieval adalah leher botol, generasi
sudah bekerja baik.**

1. Pola biner `context_precision` pada RAGAS (T1.2).
2. Hit@1 23,3% berhadapan dengan Hit@5 91,7% — dokumen yang benar hampir selalu TERAMBIL,
   tetapi jarang berada di peringkat teratas. Ini masalah PERINGKAT, bukan masalah
   cakupan.
3. Kasus E01 mengambil konteks yang tidak relevan.

Susunan kalimat kueri adalah tuas yang langsung mengenai peringkat, dan biayanya nyaris
nol dibandingkan mengganti model embedding atau menambah reranker.

PROTOKOL BAGIAN C DIPATUHI PENUH
--------------------------------
Ini penyetelan parameter, bukan sekadar pengukuran, sehingga protokolnya mengikat:

* Set PENYETELAN dan set PELAPORAN dipisah menurut pasien dan tidak pernah bertukar.
* Kriteria ditetapkan DI MUKA: **MRR** pada set penyetelan. Bukan Hit@1, bukan nDCG.
  Ditetapkan sebelum satu pun angka dilihat.
* SELURUH varian dicatat ke `results/tuning_log.json`, bukan hanya yang terbaik.
* Set evaluasi TIDAK diubah setelah hasil terlihat.
* Set pelaporan disentuh SEKALI, di akhir, untuk varian produksi dan satu varian pemenang.

Varian ditetapkan di muka pula, dan masing-masing menguji satu dugaan yang dapat
dinyatakan sebelum diukur — bukan tujuh tebakan acak. Lihat VARIAN di bawah.

KETERBATASAN YANG MENENTUKAN SEBERAPA JAUH HASIL INI BOLEH DIPERCAYA
--------------------------------------------------------------------
Relevansi ditetapkan `classify_chunk()`, sebuah pelabel KATA KUNCI, bukan penilaian
manusia (keterbatasan K1). Seluruh angka MRR di sini mewarisi kekeliruan pelabel itu.
Bila susunan kueri tertentu menang karena kebetulan menarik chunk yang kaya kata kunci
pelabel, kemenangannya artifisial. T2.2 (verifikasi manual 30-50 pasangan) ada justru
untuk memeriksa hal ini, dan hasil T3.1 tidak boleh dinyatakan mantap sebelum T2.2 selesai.

Keluaran: results/eval_rag/susunan_kueri.json + entri pada results/tuning_log.json
"""
from __future__ import annotations

import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.rag.ablation_query import CONDITION_PHRASE  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from ablation_rag_fullkb import classify_chunk, classify_glucose, ndcg_at_k  # noqa: E402
# Kriteria dan pencatat diimpor dari modul protokol, BUKAN ditulis ulang di sini.
# Menulis ulang "mrr" sebagai konstanta lokal akan membuat dua sumber kebenaran, dan
# yang lokal dapat diubah setelah melihat hasil tanpa terlihat di mana pun.
from tuning_protocol import KRITERIA, catat  # noqa: E402

TOP_K = 5
HORIZON = 6
SEED = 42
N_KASUS = 90          # per himpunan, berimbang antar-kondisi
OUT = ROOT / "results/eval_rag/susunan_kueri.json"
LOG_PENYETELAN = ROOT / "results/tuning_log.json"

# Frasa kondisi ringkas, dipakai varian yang menguji panjang.
FRASA_RINGKAS = {
    "hipoglikemia": "Hipoglikemia, gula darah rendah.",
    "hiperglikemia": "Hiperglikemia, gula darah tinggi.",
    "normal": "Gula darah dalam rentang target.",
}
# Kata kunci telanjang, tanpa kalimat sama sekali.
KATA_KUNCI = {
    "hipoglikemia": "hipoglikemia gula darah rendah penanganan aturan 15-15 glukagon",
    "hiperglikemia": "hiperglikemia gula darah tinggi koreksi insulin ketoasidosis",
    "normal": "target glikemik time in range pemantauan glukosa rutin",
}


# ── VARIAN, DITETAPKAN DI MUKA ───────────────────────────────────────────────
# Tiap varian menguji satu dugaan yang dapat dinyatakan SEBELUM diukur.
def v_produksi(g, c):
    """Acuan. Angka lalu frasa kondisi dua klausa."""
    return f"Kadar glukosa darah {g:.0f} mg/dL. {CONDITION_PHRASE[c]}"


def v_hanya_angka(g, c):
    """Dugaan: frasa kondisi menanggung hampir seluruh daya temu; tanpa itu MRR anjlok.
    Korpus adalah pedoman klinis, bukan katalog angka."""
    return f"Kadar glukosa darah {g:.0f} mg/dL."


def v_hanya_kondisi(g, c):
    """Dugaan kebalikannya: angka nyaris tidak menyumbang apa pun, bahkan mungkin
    MENGGANGGU karena '58 mg/dL' tidak beririsan dengan kosakata pedoman."""
    return CONDITION_PHRASE[c]


def v_kondisi_dulu(g, c):
    """Dugaan: posisi berpengaruh. Isi identik dengan produksi, urutan dibalik, sehingga
    selisihnya murni efek posisi pada pooling embedding."""
    return f"{CONDITION_PHRASE[c]} Kadar glukosa darah {g:.0f} mg/dL."


def v_pertanyaan(g, c):
    """Dugaan: pedoman ditulis sebagai jawaban atas pertanyaan tatalaksana, sehingga
    kueri berbentuk pertanyaan lebih dekat secara semantik daripada kueri deskriptif."""
    kondisi = {"hipoglikemia": "hipoglikemia", "hiperglikemia": "hiperglikemia",
               "normal": "glukosa dalam rentang target"}[c]
    return (f"Bagaimana tatalaksana pasien diabetes dengan {kondisi} "
            f"pada kadar glukosa {g:.0f} mg/dL?")


def v_ringkas(g, c):
    """Dugaan: frasa produksi dua klausa mengencerkan vektor. Klausa kedua ('Penyebab,
    gejala, dan penanganan') generik dan muncul di hampir semua pedoman."""
    return f"Kadar glukosa darah {g:.0f} mg/dL. {FRASA_RINGKAS[c]}"


def v_kata_kunci(g, c):
    """Dugaan: tanpa struktur kalimat sama sekali. Kalau ini menang, yang bekerja adalah
    pencocokan leksikal, bukan kedekatan semantik — dan itu temuan tentang model
    embedding, bukan tentang susunan kalimat."""
    return KATA_KUNCI[c]


VARIAN = [
    ("produksi", v_produksi), ("hanya_angka", v_hanya_angka),
    ("hanya_kondisi", v_hanya_kondisi), ("kondisi_dulu", v_kondisi_dulu),
    ("pertanyaan", v_pertanyaan), ("ringkas", v_ringkas),
    ("kata_kunci", v_kata_kunci),
]


def bangun_kasus(cfg, pids_latih, pids_target, rng) -> pd.DataFrame:
    """Latih RF pada pasien latih, hasilkan prediksi untuk pasien target.

    Kasus dipilih BERIMBANG antar-kondisi sebenarnya. Tanpa itu, ~90% kasus akan berlabel
    'normal' dan MRR keseluruhan hanya akan mengukur satu kelas.
    """
    mc = cfg["model"]
    rf_cfg = mc["random_forest"]
    feats = mc["engineered_features"]
    cadence = float(cfg["data"]["sampling_interval_min"])

    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(feats)
    kw = {"max_gap_steps": mc["max_gap_steps"], "source_interval_min": cadence}

    Xtr, ytr, atr = pre.create_sequences(df[df.patient_id.isin(pids_latih)],
                                         mc["sequence_length"], HORIZON,
                                         return_anchor=True, **kw)
    Xte, yte, ate = pre.create_sequences(df[df.patient_id.isin(pids_target)],
                                         mc["sequence_length"], HORIZON,
                                         return_anchor=True, **kw)
    scaler = StandardScaler().fit(Xtr.reshape(-1, len(feats)))
    f = lambda X: scaler.transform(X.reshape(-1, len(feats))).reshape(len(X), -1)  # noqa: E731

    reg = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
                                max_depth=rf_cfg["max_depth"],
                                min_samples_split=rf_cfg["min_samples_split"],
                                random_state=SEED, n_jobs=-1)
    reg.fit(f(Xtr), ytr - atr)
    pred = reg.predict(f(Xte)) + ate

    kasus = pd.DataFrame({"predicted": pred, "actual_future": yte, "current": ate})
    kasus["cond_pred"] = [classify_glucose(v) for v in pred]
    kasus["cond_actual"] = [classify_glucose(v) for v in yte]

    per = max(1, N_KASUS // 3)
    bagian = []
    for _, g in kasus.groupby("cond_actual"):
        bagian.append(g.sample(n=min(len(g), per), random_state=SEED))
    return pd.concat(bagian).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def nilai_varian(r, kasus: pd.DataFrame, fn) -> dict:
    """Skor satu varian atas seluruh kasus. Kueri dibangun dari glukosa TERPREDIKSI dan
    kelas hasil ambang atas prediksi itu — persis jalur pc_rag produksi."""
    hit1, mrr, ndcg = [], [], []
    for _, c in kasus.iterrows():
        q = fn(float(c["predicted"]), str(c["cond_pred"]))
        docs = r.retrieve(q, top_k=TOP_K)
        topik = [classify_chunk(d["text"]) for d in docs]
        exp = str(c["cond_actual"])
        rank = (topik.index(exp) + 1) if exp in topik else 0
        hit1.append(int(bool(topik) and topik[0] == exp))
        mrr.append((1.0 / rank) if rank else 0.0)
        ndcg.append(ndcg_at_k([1 if t == exp else 0 for t in topik], TOP_K))
    return {"n": int(len(kasus)),
            "hit@1": round(float(np.mean(hit1)), 4),
            "mrr": round(float(np.mean(mrr)), 4),
            "ndcg@5": round(float(np.mean(ndcg)), 4),
            "_mrr_per_kasus": [float(x) for x in mrr]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persist", default="models/chroma_db")
    ap.add_argument("--collection", default="diabetes_kb")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    df_pid = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", usecols=["patient_id"])
    pids = sorted(df_pid["patient_id"].unique().tolist())
    latih, setel, lapor = pids[:-4], pids[-4:-2], pids[-2:]

    print("T3.1 — susunan kalimat kueri prediction-conditioned")
    print(f"latih={len(latih)} | PENYETELAN={setel} | PELAPORAN={lapor}")
    print(f"Kriteria DITETAPKAN DI MUKA: {KRITERIA.upper()} pada set penyetelan")
    print(f"{len(VARIAN)} varian, seluruhnya dicatat ke {LOG_PENYETELAN.name}")
    print("Relevansi dari classify_chunk() — pelabel KATA KUNCI (keterbatasan K1); "
          "hasil belum mantap sebelum T2.2\n")

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    kasus_setel = bangun_kasus(cfg, latih, setel, rng)
    print(f"set penyetelan : {len(kasus_setel)} kasus "
          f"({dict(kasus_setel['cond_actual'].value_counts())})")
    kasus_lapor = bangun_kasus(cfg, latih, lapor, rng)
    print(f"set pelaporan  : {len(kasus_lapor)} kasus "
          f"({dict(kasus_lapor['cond_actual'].value_counts())})  "
          f"[{time.time() - t0:.0f} dtk]\n")

    r = MMRRetriever(persist_dir=args.persist, collection_name=args.collection,
                     embed_provider="sentence-transformers")

    print(f"{'varian':<16}{'MRR':>9}{'Hit@1':>9}{'nDCG@5':>9}   contoh kueri")
    hasil_setel = {}
    for nama, fn in VARIAN:
        s = nilai_varian(r, kasus_setel, fn)
        hasil_setel[nama] = s
        contoh = fn(58.0, "hipoglikemia")
        print(f"{nama:<16}{s['mrr']:>9.4f}{s['hit@1']:>9.4f}{s['ndcg@5']:>9.4f}   "
              f"{contoh[:58]}{'...' if len(contoh) > 58 else ''}")

    mrr_prod = hasil_setel["produksi"]["_mrr_per_kasus"]
    for nama, s in hasil_setel.items():
        if nama == "produksi":
            s["p_wilcoxon_vs_produksi"], s["berbeda_signifikan"] = None, False
            continue
        try:
            p = float(stats.wilcoxon(s["_mrr_per_kasus"], mrr_prod).pvalue)
        except ValueError:
            p = float("nan")
        s["p_wilcoxon_vs_produksi"] = None if p != p else round(p, 6)
        s["berbeda_signifikan"] = bool(p == p and p < 0.05)

    pemenang = max(hasil_setel.items(), key=lambda kv: kv[1][KRITERIA])[0]
    print(f"\nPemenang menurut kriteria {KRITERIA.upper()} pada set PENYETELAN: "
          f"{pemenang} ({hasil_setel[pemenang][KRITERIA]:.4f})")
    if pemenang == "produksi":
        print("Susunan produksi tidak terkalahkan — tidak ada usulan perubahan.")
    else:
        p = hasil_setel[pemenang]["p_wilcoxon_vs_produksi"]
        print(f"  selisih thd produksi: "
              f"{hasil_setel[pemenang][KRITERIA] - hasil_setel['produksi'][KRITERIA]:+.4f} "
              f"| Wilcoxon p={p} -> "
              f"{'signifikan' if hasil_setel[pemenang]['berbeda_signifikan'] else 'TIDAK signifikan'}")

    # Set pelaporan disentuh SEKALI: produksi + pemenang.
    print(f"\nSet PELAPORAN disentuh sekali: produksi vs {pemenang}")
    fn_map = dict(VARIAN)
    hasil_lapor = {}
    for nama in dict.fromkeys(["produksi", pemenang]):
        s = nilai_varian(r, kasus_lapor, fn_map[nama])
        hasil_lapor[nama] = s
        print(f"  {nama:<16} MRR {s['mrr']:.4f} | Hit@1 {s['hit@1']:.4f} "
              f"| nDCG@5 {s['ndcg@5']:.4f}")
    if pemenang != "produksi":
        try:
            p_l = float(stats.wilcoxon(hasil_lapor[pemenang]["_mrr_per_kasus"],
                                       hasil_lapor["produksi"]["_mrr_per_kasus"]).pvalue)
        except ValueError:
            p_l = float("nan")
        hasil_lapor["p_wilcoxon"] = None if p_l != p_l else round(p_l, 6)
        hasil_lapor["bertahan_di_set_pelaporan"] = bool(
            p_l == p_l and p_l < 0.05
            and hasil_lapor[pemenang][KRITERIA] > hasil_lapor["produksi"][KRITERIA])
        print(f"  Wilcoxon p={hasil_lapor['p_wilcoxon']} -> keunggulan "
              f"{'BERTAHAN' if hasil_lapor['bertahan_di_set_pelaporan'] else 'TIDAK bertahan'} "
              f"di set pelaporan")

    for d in (hasil_setel, hasil_lapor):
        for v in d.values():
            if isinstance(v, dict):
                v.pop("_mrr_per_kasus", None)

    out = {
        "percobaan": "T3.1 — susunan kalimat kueri prediction-conditioned",
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "protokol_bagian_c": {
            "set_latih": latih, "set_penyetelan": setel, "set_pelaporan": lapor,
            "kriteria_ditetapkan_di_muka": KRITERIA,
            "jumlah_varian_diuji": len(VARIAN),
            "seluruh_varian_dicatat": True,
            "set_pelaporan_disentuh_berapa_kali": 1,
            "set_evaluasi_diubah_setelah_melihat_hasil": False,
        },
        "keterbatasan_menentukan": (
            "Relevansi ditetapkan classify_chunk(), pelabel KATA KUNCI, bukan penilaian "
            "manusia (K1). Bila satu susunan menang karena menarik chunk yang kaya kata "
            "kunci pelabel, kemenangannya artifisial. T3.1 tidak boleh dinyatakan mantap "
            "sebelum T2.2 (verifikasi manual) selesai."),
        "top_k": TOP_K, "horizon_steps": HORIZON,
        "n_kasus_per_himpunan": N_KASUS,
        "contoh_kueri": {nama: fn(58.0, "hipoglikemia") for nama, fn in VARIAN},
        "hasil_set_penyetelan": hasil_setel,
        "pemenang_set_penyetelan": pemenang,
        "hasil_set_pelaporan": hasil_lapor,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    # Log penyetelan kumulatif lewat modul protokol — SETIAP varian dicatat, termasuk
    # yang kalah (Bagian C butir 3).
    for nama, s in hasil_setel.items():
        catat(parameter="susunan_kueri", nilai=nama,
              metrik_penyetelan={**{k: s[k] for k in ("n", "hit@1", "mrr", "ndcg@5")},
                                 "p_wilcoxon_vs_produksi": s.get("p_wilcoxon_vs_produksi")},
              catatan=(f"T3.1 — set penyetelan {setel} n={s['n']}, "
                       f"{len(VARIAN)} varian diuji, pemenang={pemenang}. "
                       f"Relevansi dari pelabel kata kunci (K1)."),
              log_path=LOG_PENYETELAN)

    print(f"\nDisimpan ke {OUT}")
    print(f"Log penyetelan diperbarui: {LOG_PENYETELAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
