"""T3.2 — horizon mana yang lebih baik menjadi sumber pengondisian retrieval?

PERTANYAAN YANG SENGAJA TIDAK DIJAWAB
-------------------------------------
Rumusan yang tampak paling wajar adalah: "untuk tiap horizon, seberapa baik retrieval yang
dikondisikan pada prediksi horizon itu menemukan dokumen yang relevan bagi kondisi
sebenarnya pada horizon itu?" Rumusan itu **ditolak**, karena membandingkan +30 dan +60
dengan sasaran yang berbeda mencampur dua hal:

  * seberapa sulit MEMPREDIKSI horizon tersebut (RMSE +60 menit hampir dua kali RMSE
    +30 menit — 34,8 lawan 21,1 pada crossval), dan
  * seberapa baik angka itu MENGARAHKAN retrieval.

Selisih apa pun yang muncul tidak dapat dikaitkan ke salah satunya. Melaporkan angka
seperti itu lebih buruk daripada tidak melaporkan apa pun.

PERTANYAAN YANG DIJAWAB
-----------------------
Sasaran DIPAKUKAN pada satu hal: kondisi sebenarnya pada **+60 menit**. Lalu ditanyakan
sumber pengondisian mana yang paling baik menemukannya. Ini juga pertanyaan klinis yang
sebenarnya — rekomendasi antisipatif ditujukan pada keadaan yang akan datang, dan yang
ingin diketahui adalah angka mana yang paling baik mengarahkan sistem ke sana.

Empat lengan, seluruhnya dinilai terhadap sasaran yang SAMA dan pada kasus yang SAMA:

  ==================  ===========================================================
  ``current``         glukosa saat ini (dasar pembanding; bukan PC-RAG sama sekali)
  ``pred_h6``         prediksi +30 menit  <- konfigurasi produksi
  ``pred_h12``        prediksi +60 menit
  ``oracle_h12``      glukosa +60 menit yang SEBENARNYA (langit-langit, tidak dapat dicapai)
  ==================  ===========================================================

Karena panjang jendela sama untuk kedua horizon, keempat lengan dapat dihitung dari SATU
himpunan jendela yang identik: model h6 dan h12 diterapkan pada matriks X yang sama.
Dengan begitu pemasangannya sempurna dan uji berpasangan sah.

PROTOKOL BAGIAN C
-----------------
Ini pilihan rancangan yang dapat diadopsi, sehingga protokolnya mengikat: set penyetelan
dan pelaporan dipisah per pasien, kriteria **MRR** ditetapkan di muka, seluruh lengan
dicatat, set pelaporan disentuh sekali.

Keterbatasan K1 berlaku sama seperti T3.1: relevansi berasal dari pelabel kata kunci.

Keluaran: results/eval_rag/horizon_retrieval.json
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
from itertools import combinations
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
from src.rag.ablation_query import build_ablation_query  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from ablation_rag_fullkb import classify_chunk, classify_glucose, ndcg_at_k  # noqa: E402
# Kriteria diimpor, bukan ditulis ulang — satu sumber kebenaran (lihat T3.1).
from tuning_protocol import KRITERIA, catat  # noqa: E402

TOP_K = 5
H_PENDEK, H_PANJANG = 6, 12
SEED = 42
N_KASUS = 90
LENGAN = ["current", "pred_h6", "pred_h12", "oracle_h12"]
OUT = ROOT / "results/eval_rag/horizon_retrieval.json"
LOG_PENYETELAN = ROOT / "results/tuning_log.json"


def siapkan(cfg):
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(mc["engineered_features"])
    return df, pre


def bangun_kasus(cfg, df, pre, pids_latih, pids_target):
    """Latih regresor h6 DAN h12, lalu terapkan keduanya pada jendela yang SAMA.

    Jendela dibentuk untuk horizon PANJANG. Itu himpunan yang lebih kecil (butuh lebih
    banyak data masa depan), dan setiap jendelanya juga sah bagi horizon pendek karena
    panjang look-back-nya identik. Dengan satu himpunan jendela, keempat lengan
    berpasangan sempurna.
    """
    mc = cfg["model"]
    rf_cfg = mc["random_forest"]
    feats = mc["engineered_features"]
    cadence = float(cfg["data"]["sampling_interval_min"])
    kw = {"max_gap_steps": mc["max_gap_steps"], "source_interval_min": cadence}
    seq = mc["sequence_length"]

    d_tr = df[df.patient_id.isin(pids_latih)]
    d_te = df[df.patient_id.isin(pids_target)]

    # Latihan memakai jendela horizonnya masing-masing (itu memang cara model dilatih).
    Xtr6, ytr6, atr6 = pre.create_sequences(d_tr, seq, H_PENDEK, return_anchor=True, **kw)
    Xtr12, ytr12, atr12 = pre.create_sequences(d_tr, seq, H_PANJANG, return_anchor=True, **kw)
    # Penilaian memakai SATU himpunan jendela: horizon panjang.
    Xte, yte12, ate = pre.create_sequences(d_te, seq, H_PANJANG, return_anchor=True, **kw)

    scaler = StandardScaler().fit(Xtr12.reshape(-1, len(feats)))
    f = lambda X: scaler.transform(X.reshape(-1, len(feats))).reshape(len(X), -1)  # noqa: E731

    def latih(X, y, a):
        m = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
                                  max_depth=rf_cfg["max_depth"],
                                  min_samples_split=rf_cfg["min_samples_split"],
                                  random_state=SEED, n_jobs=-1)
        m.fit(f(X), y - a)
        return m

    Xte_s = f(Xte)
    pred6 = latih(Xtr6, ytr6, atr6).predict(Xte_s) + ate
    pred12 = latih(Xtr12, ytr12, atr12).predict(Xte_s) + ate

    kasus = pd.DataFrame({"current": ate, "pred_h6": pred6, "pred_h12": pred12,
                          "oracle_h12": yte12})
    # SASARAN DIPAKUKAN: kondisi sebenarnya pada +60 menit, sama untuk keempat lengan.
    kasus["sasaran"] = [classify_glucose(v) for v in yte12]

    per = max(1, N_KASUS // 3)
    bagian = [g.sample(n=min(len(g), per), random_state=SEED)
              for _, g in kasus.groupby("sasaran")]
    return pd.concat(bagian).sample(frac=1.0, random_state=SEED).reset_index(drop=True)


def nilai(r, kasus, lengan) -> dict:
    hit1, mrr, ndcg = [], [], []
    for _, c in kasus.iterrows():
        g = float(c[lengan])
        q = build_ablation_query(g)   # susunan produksi, identik untuk keempat lengan
        docs = r.retrieve(q, top_k=TOP_K)
        topik = [classify_chunk(d["text"]) for d in docs]
        exp = str(c["sasaran"])
        rank = (topik.index(exp) + 1) if exp in topik else 0
        hit1.append(int(bool(topik) and topik[0] == exp))
        mrr.append((1.0 / rank) if rank else 0.0)
        ndcg.append(ndcg_at_k([1 if t == exp else 0 for t in topik], TOP_K))
    return {"n": int(len(kasus)),
            "hit@1": round(float(np.mean(hit1)), 4),
            "mrr": round(float(np.mean(mrr)), 4),
            "ndcg@5": round(float(np.mean(ndcg)), 4),
            "_mrr": [float(x) for x in mrr]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persist", default="models/chroma_db")
    ap.add_argument("--collection", default="diabetes_kb")
    args = ap.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    df, pre = siapkan(cfg)
    pids = sorted(df["patient_id"].unique().tolist())
    latih, setel, lapor = pids[:-4], pids[-4:-2], pids[-2:]

    print("T3.2 — horizon mana yang lebih baik menjadi sumber pengondisian retrieval")
    print(f"SASARAN DIPAKUKAN: kondisi sebenarnya pada +{H_PANJANG * 5} menit, "
          f"sama untuk keempat lengan")
    print(f"latih={len(latih)} | PENYETELAN={setel} | PELAPORAN={lapor}")
    print(f"Kriteria DITETAPKAN DI MUKA: {KRITERIA.upper()} pada set penyetelan\n")

    t0 = time.time()
    k_setel = bangun_kasus(cfg, df, pre, latih, setel)
    k_lapor = bangun_kasus(cfg, df, pre, latih, lapor)
    print(f"penyetelan {len(k_setel)} kasus {dict(k_setel['sasaran'].value_counts())} | "
          f"pelaporan {len(k_lapor)} kasus {dict(k_lapor['sasaran'].value_counts())} "
          f"[{time.time() - t0:.0f} dtk]\n")

    r = MMRRetriever(persist_dir=args.persist, collection_name=args.collection,
                     embed_provider="sentence-transformers")

    print(f"{'lengan':<14}{'MRR':>9}{'Hit@1':>9}{'nDCG@5':>9}   glukosa rerata")
    setel_res = {}
    for a in LENGAN:
        s = nilai(r, k_setel, a)
        setel_res[a] = s
        print(f"{a:<14}{s['mrr']:>9.4f}{s['hit@1']:>9.4f}{s['ndcg@5']:>9.4f}"
              f"   {k_setel[a].mean():>8.1f} mg/dL")

    # Seluruh pasangan diuji, bukan hanya pasangan yang menarik setelah angka terlihat.
    uji = {}
    for a, b in combinations(LENGAN, 2):
        try:
            p = float(stats.wilcoxon(setel_res[a]["_mrr"], setel_res[b]["_mrr"]).pvalue)
        except ValueError:
            p = float("nan")
        uji[f"{a}_vs_{b}"] = {
            "selisih_mrr": round(setel_res[a]["mrr"] - setel_res[b]["mrr"], 4),
            "wilcoxon_p": None if p != p else round(p, 6),
            "signifikan": bool(p == p and p < 0.05),
        }
    print("\nUji berpasangan pada set penyetelan (seluruh pasangan):")
    for k, v in uji.items():
        print(f"  {k:<28} selisih MRR {v['selisih_mrr']:>+7.4f} | p={v['wilcoxon_p']} "
              f"-> {'signifikan' if v['signifikan'] else 'tidak signifikan'}")

    dapat_dicapai = [a for a in LENGAN if a != "oracle_h12"]
    pemenang = max(dapat_dicapai, key=lambda a: setel_res[a][KRITERIA])
    print(f"\nPemenang di antara lengan yang DAPAT DICAPAI (oracle dikecualikan): "
          f"{pemenang} ({setel_res[pemenang][KRITERIA]:.4f})")
    print(f"Langit-langit oracle_h12: {setel_res['oracle_h12'][KRITERIA]:.4f} "
          f"— selisih {setel_res['oracle_h12'][KRITERIA] - setel_res[pemenang][KRITERIA]:+.4f} "
          f"adalah bagian yang hilang karena galat prediksi, bukan karena retrieval")

    print(f"\nSet PELAPORAN disentuh sekali (keempat lengan):")
    lapor_res = {}
    for a in LENGAN:
        s = nilai(r, k_lapor, a)
        lapor_res[a] = s
        print(f"  {a:<14} MRR {s['mrr']:.4f} | Hit@1 {s['hit@1']:.4f} "
              f"| nDCG@5 {s['ndcg@5']:.4f}")
    try:
        p_l = float(stats.wilcoxon(lapor_res[pemenang]["_mrr"],
                                   lapor_res["pred_h6"]["_mrr"]).pvalue) \
            if pemenang != "pred_h6" else float("nan")
    except ValueError:
        p_l = float("nan")

    for d in (setel_res, lapor_res):
        for v in d.values():
            v.pop("_mrr", None)

    out = {
        "percobaan": "T3.2 — horizon sebagai sumber pengondisian retrieval",
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "rumusan_yang_ditolak": (
            "Membandingkan tiap horizon terhadap sasaran horizonnya sendiri akan mencampur "
            "kesulitan MEMPREDIKSI horizon itu (RMSE +60 mnt 34,8 lawan +30 mnt 21,1) "
            "dengan mutu pengarahan retrieval. Selisihnya tidak dapat dikaitkan ke "
            "salah satunya."),
        "rumusan_yang_dipakai": (
            f"Sasaran dipakukan pada kondisi sebenarnya +{H_PANJANG * 5} menit; keempat "
            f"lengan dinilai terhadap sasaran yang sama, pada kasus yang sama, dengan "
            f"susunan kueri yang sama."),
        "protokol_bagian_c": {
            "set_latih": latih, "set_penyetelan": setel, "set_pelaporan": lapor,
            "kriteria_ditetapkan_di_muka": KRITERIA,
            "jumlah_lengan": len(LENGAN), "seluruh_lengan_dicatat": True,
            "set_pelaporan_disentuh_berapa_kali": 1,
            "seluruh_pasangan_diuji": True,
        },
        "keterbatasan_K1": ("Relevansi dari classify_chunk(), pelabel kata kunci, bukan "
                            "penilaian manusia. Berlaku sama seperti T3.1."),
        "top_k": TOP_K, "n_kasus_per_himpunan": N_KASUS,
        "hasil_set_penyetelan": setel_res,
        "uji_berpasangan_set_penyetelan": uji,
        "pemenang_dapat_dicapai": pemenang,
        "langit_langit_oracle": setel_res["oracle_h12"][KRITERIA],
        "hasil_set_pelaporan": lapor_res,
        "p_wilcoxon_pelaporan_pemenang_vs_produksi": None if p_l != p_l else round(p_l, 6),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    for a in LENGAN:
        catat(parameter="horizon_pengondisian", nilai=a,
              metrik_penyetelan={k: setel_res[a][k]
                                 for k in ("n", "hit@1", "mrr", "ndcg@5")},
              catatan=(f"T3.2 — sasaran dipakukan +{H_PANJANG * 5} mnt, set penyetelan "
                       f"{setel}, pemenang dapat dicapai={pemenang}. "
                       f"oracle_h12 = langit-langit, tidak dapat diadopsi."),
              log_path=LOG_PENYETELAN)

    print(f"\nDisimpan ke {OUT}")
    print(f"Log penyetelan diperbarui: {LOG_PENYETELAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
