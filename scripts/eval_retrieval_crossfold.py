"""Validasi silang lintas-pasien untuk lapisan RETRIEVAL (bukan hanya prediksi).

Prediksi glukosa sudah divalidasi 6-fold lintas-pasien (crossval_rf_vs_lstm.py), tetapi
evaluasi retrieval pada kasus nyata (eval_retrieval_realcases.py) hanya memakai dua pasien
hold-out. Akibatnya belum diketahui apakah manfaat PC-RAG konsisten antar-pasien atau
kebetulan muncul pada dua pasien tersebut.

Skrip ini mengulang evaluasi retrieval untuk keenam fold: pada tiap fold, model regresi dan
pengklasifikasi kondisi dilatih ulang dari nol pada 10 pasien, lalu retrieval dievaluasi pada
2 pasien uji yang tidak pernah dilihat. Hasilnya dilaporkan sebagai rerata +/- simpangan baku
lintas fold, sehingga variasi antar-pasien terlihat.

Keluaran: results/retrieval_realcases/crossfold.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import json
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from ablation_rag_fullkb import CONDITION_PHRASE, classify_chunk, classify_glucose, ndcg_at_k  # noqa: E402,F401
from src.rag.ablation_query import build_ablation_query  # noqa: E402

TOP_K = 5
HORIZON = 6
N_PER_SET = 60      # kasus per himpunan per fold (ditekan agar runtime wajar)
MIN_GAP = 6
SEED = 42
MODES = ["standard", "pc_rag", "pc_rag_classifier", "oracle"]
# Tag korpus untuk penamaan keluaran. Korpus berpindah dari additional_docs/
# (14 PDF PERKENI/ADA) ke books/ (KB-01..KB-12) pada Tugas 1, sehingga angka lama
# tidak berlaku lagi. Sufiks ini membuat hasil baru berdampingan dengan hasil lama
# tanpa menimpanya, supaya keduanya bisa dibandingkan di laporan.
CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12")
OUT = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/crossfold.json"
# Keluaran parsial per fold, sengaja bernama BEDA dari OUT supaya hasil setengah jadi
# tidak pernah terbaca sebagai hasil lengkap.
PARSIAL = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/crossfold_PARSIAL.json"


def build_frame(cfg: dict):
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(mc["engineered_features"])
    return df, pre


def seqs(pre, d, mc, cadence_min: float = 5.0):
    """Bangun jendela; ikut menyaring jeda sensor bila config menetapkannya (Tugas 5).

    Tanpa ini, model di dalam crossfold dilatih atas jendela yang boleh melintasi jeda
    sensor sementara model produksi tidak — sehingga crossfold mengukur mutu retrieval
    yang dikondisikan pada prediktor yang berbeda dari yang benar-benar dipakai sistem.
    """
    X, y, anc = pre.create_sequences(
        d, mc["sequence_length"], HORIZON, return_anchor=True,
        max_gap_steps=mc.get("max_gap_steps"), source_interval_min=cadence_min,
    )
    n, s, f = X.shape
    return X.reshape(n, s * f), y, anc


def build_query(g: float, is_pred: bool, cond: str | None = None) -> str:
    """Semua mode ber-STRUKTUR IDENTIK; hanya angka (dan `cond`) yang berbeda.

    ``is_pred`` dipertahankan agar pemanggil tetap terbaca eksplisit, tetapi sudah TIDAK
    lagi memengaruhi teks kueri: sufiks "(prediksi 30 menit ke depan)" dulu hanya melekat
    pada mode berbasis prediksi, sehingga perbandingannya terhadap mode current tidak
    mengisolasi sumber pengondisian. Lihat src/rag/ablation_query.py.
    """
    del is_pred  # sengaja diabaikan — lihat docstring
    return build_ablation_query(g, cond=cond)


def score(r, query: str, expected: str) -> dict:
    docs = r.retrieve(query, top_k=TOP_K)
    topics = [classify_chunk(d["text"]) for d in docs]
    rank = (topics.index(expected) + 1) if expected in topics else 0
    rels = [1 if t == expected else 0 for t in topics]
    return {"hit@1": int(bool(topics) and topics[0] == expected),
            "mrr": (1.0 / rank) if rank else 0.0,
            "ndcg": ndcg_at_k(rels, TOP_K)}


# ── T13: lengan penelusuran leksikal dan hibrida ─────────────────────────────
#
# WHY  Model embedding produksi berbahasa INGGRIS sedangkan korpusnya INDONESIA.
#      T9 menutup jalan mengganti model (multibahasa merusak pembedaan kondisi),
#      sehingga jalan yang tersisa adalah pencocokan LEKSIKAL, yang tidak
#      bergantung pada ruang semantik sama sekali.
# HOW  Digabung dengan Reciprocal Rank Fusion karena RRF bekerja pada PERINGKAT,
#      bukan skor mentah: skor kosinus dan skor BM25 berskala berbeda dan tidak
#      dapat dijumlahkan. k_rrf=60 dipertahankan pada nilai bakunya dan TIDAK
#      disetel, agar tidak menambah parameter bebas.
# Lihat docs/PRAPENDAFTARAN_T13_HIBRIDA.md.
K_RRF = 60
LENGAN = ["vektor", "bm25", "hibrida"]


def tokenisasi(teks: str) -> list:
    """Angka ambang ("70", "180") justru sinyal yang diharapkan ditangkap BM25,
    sehingga angka TIDAK dibuang. Identik dengan scripts/eval_hybrid_bm25.py."""
    import re
    return re.findall(r"[a-z0-9]+", teks.lower())


def bangun_bm25(persist_dir: str, collection: str):
    """Indeks BM25 atas korpus yang SAMA dengan indeks vektor.

    Dokumen diambil dari ChromaDB, bukan dibaca ulang dari PDF, supaya kedua
    lengan benar-benar menelusuri korpus yang identik — bila tidak, selisihnya
    dapat berasal dari perbedaan korpus dan bukan dari cara menelusurinya.
    """
    import chromadb
    from rank_bm25 import BM25Okapi

    col = chromadb.PersistentClient(path=persist_dir).get_collection(collection)
    teks = col.get(include=["documents"])["documents"]
    return BM25Okapi([tokenisasi(t) for t in teks]), teks


def _rrf(daftar_peringkat: list, k_rrf: int = K_RRF) -> list:
    skor = {}
    for daftar in daftar_peringkat:
        for pos, idx in enumerate(daftar, start=1):
            skor[idx] = skor.get(idx, 0.0) + 1.0 / (k_rrf + pos)
    return [i for i, _ in sorted(skor.items(), key=lambda kv: -kv[1])]


def _metrik_dari_topik(topics: list, expected: str) -> dict:
    rank = (topics.index(expected) + 1) if expected in topics else 0
    rels = [1 if t == expected else 0 for t in topics]
    return {"hit@1": int(bool(topics) and topics[0] == expected),
            "mrr": (1.0 / rank) if rank else 0.0,
            "ndcg": ndcg_at_k(rels, TOP_K)}


def score_lengan(r, bm25, teks_korpus, query: str, expected: str, n_kandidat: int = 50) -> dict:
    """Skor ketiga lengan atas SATU kueri, memakai kolam kandidat yang sebanding."""
    import numpy as _np

    hasil = {}

    # Lengan vektor: jalur produksi, tidak diubah sedikit pun agar D1 dapat diperiksa.
    hasil["vektor"] = score(r, query, expected)

    # Lengan BM25.
    skor_bm = bm25.get_scores(tokenisasi(query))
    urut_bm = list(_np.argsort(-skor_bm)[:n_kandidat])
    hasil["bm25"] = _metrik_dari_topik(
        [classify_chunk(teks_korpus[i]) for i in urut_bm[:TOP_K]], expected)

    # Lengan hibrida: peringkat vektor dipetakan ke indeks korpus lewat teksnya.
    docs_v = r.retrieve(query, top_k=n_kandidat)
    peta = {t: i for i, t in enumerate(teks_korpus)}
    urut_v = [peta[d["text"]] for d in docs_v if d["text"] in peta]
    gabung = _rrf([urut_v, urut_bm])[:TOP_K]
    hasil["hibrida"] = _metrik_dari_topik(
        [classify_chunk(teks_korpus[i]) for i in gabung], expected)
    return hasil


def pick(cases: pd.DataFrame, divergent: bool, rng) -> pd.DataFrame:
    sub = cases[cases["divergent"]] if divergent else cases
    if divergent:
        picked, last = [], -10**9
        for _, row in sub.sort_values("idx").iterrows():
            if row["idx"] - last >= MIN_GAP:
                picked.append(row)
                last = row["idx"]
        sub = pd.DataFrame(picked)
        parts = []
        per = max(1, N_PER_SET // max(1, sub["cond_actual"].nunique()))
        for _, g in sub.groupby("cond_actual"):
            parts.append(g.sample(n=min(len(g), per), random_state=SEED))
        sub = pd.concat(parts)
    if len(sub) > N_PER_SET:
        sub = sub.sample(n=N_PER_SET, random_state=SEED)
    return sub.reset_index(drop=True)


def main() -> None:
    # Gagal cepat: pastikan lokasi keluaran dapat ditulis SEBELUM memulai komputasi
    # enam fold yang memakan lebih dari satu jam.
    OUT.parent.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mc = cfg["model"]
    rf_cfg = mc["random_forest"]
    # Keluarga model mengikuti config.model.name. Mode `standard` dan `oracle` TIDAK
    # menyentuh model sama sekali (lihat docs/PRAPENDAFTARAN_T6_GBM_RETRIEVAL.md), jadi
    # keduanya berfungsi sebagai kontrol: keduanya wajib tereproduksi persis.
    keluarga = "rf" if mc.get("name") == "RandomForest" else "gbm"
    gseed = mc.get("gradient_boosting", {}).get("random_state", SEED)
    print(f"Keluarga prediktor: {keluarga} (config.model.name = {mc.get('name')})")
    df, pre = build_frame(cfg)

    patients = sorted(df["patient_id"].unique())
    folds = [patients[i:i + 2] for i in range(0, len(patients), 2)]
    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    bm25, teks_korpus = bangun_bm25("models/chroma_db", "diabetes_kb")
    print(f"Indeks BM25 dibangun atas {len(teks_korpus)} potongan yang SAMA "
          f"dengan indeks vektor (T13).")

    per_fold = []
    for fi, test_p in enumerate(folds):
        train_df = df[~df.patient_id.isin(test_p)]
        test_df = df[df.patient_id.isin(test_p)]

        Xtr, ytr, atr = seqs(pre, train_df, mc)
        Xte, yte, ate = seqs(pre, test_df, mc)

        scaler = StandardScaler().fit(Xtr.reshape(-1, len(mc["engineered_features"])))

        def sc(X):
            n = len(X)
            return scaler.transform(X.reshape(-1, len(mc["engineered_features"]))).reshape(n, -1)

        Xtr_s, Xte_s = sc(Xtr), sc(Xte)

        # class_weight="balanced" tetap dipakai pada kedua keluarga: tanpanya kelas
        # hipoglikemia (3,3% sampel) tenggelam dan pengondisian kueri kehilangan
        # justru kasus yang paling perlu diantisipasi.
        if keluarga == "gbm":
            reg = HistGradientBoostingRegressor(random_state=gseed)
            clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=gseed)
        else:
            reg = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"],
                                        max_depth=rf_cfg["max_depth"],
                                        min_samples_split=rf_cfg["min_samples_split"],
                                        random_state=SEED, n_jobs=-1)
            clf = RandomForestClassifier(n_estimators=rf_cfg["n_estimators"],
                                         max_depth=rf_cfg["max_depth"],
                                         min_samples_split=rf_cfg["min_samples_split"],
                                         class_weight="balanced", random_state=SEED, n_jobs=-1)

        reg.fit(Xtr_s, ytr - atr)
        pred = reg.predict(Xte_s) + ate

        lbl_tr = np.array([classify_glucose(v) for v in ytr])
        clf.fit(Xtr_s, lbl_tr)
        cond_clf = clf.predict(Xte_s)

        cases = pd.DataFrame({
            "idx": np.arange(len(yte)), "current": ate, "predicted": pred,
            "actual_future": yte, "cond_clf": cond_clf,
        })
        cases["cond_current"] = [classify_glucose(v) for v in ate]
        cases["cond_actual"] = [classify_glucose(v) for v in yte]
        cases["divergent"] = cases["cond_current"] != cases["cond_actual"]

        rng = np.random.default_rng(SEED + fi)
        fold_res = {"fold": fi, "pasien_uji": test_p,
                    "n_divergen_tersedia": int(cases["divergent"].sum())}

        for label, divergent in (("divergen", True), ("natural", False)):
            sub = pick(cases, divergent, rng)
            agg = {m: {"hit@1": [], "mrr": []} for m in MODES}
            # T13: lengan tambahan diakumulasi terpisah agar struktur lama utuh.
            agg_l = {l: {m: {"hit@1": [], "mrr": []} for m in MODES} for l in LENGAN}
            for _, c in sub.iterrows():
                exp = c["cond_actual"]
                for m in MODES:
                    if m == "standard":
                        q = build_query(float(c["current"]), False)
                    elif m == "pc_rag":
                        q = build_query(float(c["predicted"]), True)
                    elif m == "pc_rag_classifier":
                        q = build_query(float(c["predicted"]), True, cond=str(c["cond_clf"]))
                    else:
                        q = build_query(float(c["actual_future"]), True)
                    sl = score_lengan(r, bm25, teks_korpus, q, exp)
                    s = sl["vektor"]  # lengan produksi; nilainya identik score()
                    agg[m]["hit@1"].append(s["hit@1"])
                    agg[m]["mrr"].append(s["mrr"])
                    for l in LENGAN:
                        agg_l[l][m]["hit@1"].append(sl[l]["hit@1"])
                        agg_l[l][m]["mrr"].append(sl[l]["mrr"])
            fold_res[label] = {"n": int(len(sub)),
                               **{m: {"hit@1": round(float(np.mean(agg[m]["hit@1"])), 3),
                                      "mrr": round(float(np.mean(agg[m]["mrr"])), 3)} for m in MODES}}
            fold_res[f"{label}_lengan"] = {
                l: {m: {"hit@1": round(float(np.mean(agg_l[l][m]["hit@1"])), 3),
                        "mrr": round(float(np.mean(agg_l[l][m]["mrr"])), 3)} for m in MODES}
                for l in LENGAN}

        per_fold.append(fold_res)
        d, n = fold_res["divergen"], fold_res["natural"]
        print(f"fold {fi} {test_p} | divergen MRR: std {d['standard']['mrr']:.3f} "
              f"pc {d['pc_rag']['mrr']:.3f} clf {d['pc_rag_classifier']['mrr']:.3f} "
              f"oracle {d['oracle']['mrr']:.3f} || natural MRR: std {n['standard']['mrr']:.3f} "
              f"pc {n['pc_rag']['mrr']:.3f} clf {n['pc_rag_classifier']['mrr']:.3f}")

        # Simpan hasil PARSIAL setelah tiap fold. Skrip ini butuh 1,5-2 jam dan melatih 12
        # RandomForest; dua kali ia mati di tengah (fold 5 dari 6) dan seluruh pekerjaannya
        # hilang karena keluaran hanya ditulis di akhir. Berkas parsial ini sengaja BERBEDA
        # nama dari keluaran final, supaya hasil setengah jadi tidak pernah terbaca sebagai
        # hasil lengkap.
        PARSIAL.parent.mkdir(parents=True, exist_ok=True)
        PARSIAL.write_text(json.dumps(
            {"PERINGATAN": f"HASIL PARSIAL — baru {len(per_fold)} dari {len(folds)} fold. "
                           f"JANGAN dipakai sebagai hasil akhir.",
             "n_fold_selesai": len(per_fold), "n_fold_target": len(folds),
             "per_fold": per_fold}, indent=2), encoding="utf-8")

    ringkas = {}
    for label in ("divergen", "natural"):
        ringkas[label] = {}
        for m in MODES:
            vals = np.array([f[label][m]["mrr"] for f in per_fold])
            h1 = np.array([f[label][m]["hit@1"] for f in per_fold])
            ringkas[label][m] = {
                "mrr_rerata": round(float(vals.mean()), 3),
                "mrr_sd": round(float(vals.std(ddof=1)), 3),
                "hit@1_rerata": round(float(h1.mean()), 3),
                "hit@1_sd": round(float(h1.std(ddof=1)), 3),
            }

    # KONFIGURASI EFEKTIF ditulis ke berkas hasil. Sebelumnya hanya `top_k` yang
    # tercatat, sehingga satu-satunya bukti bahwa suatu berkas berasal dari
    # lambda_mult 0,0 adalah NAMA DIREKTORINYA — dan itu sudah pernah menyesatkan:
    # tabel HANDOFF sempat memuat angka lambda 0,5 sebagai hasil produksi.
    from src.config import load_rag_config
    rag_cfg = load_rag_config()
    out = {
        "catatan": ("Enam fold lintas-pasien. Pada tiap fold, model regresi dan pengklasifikasi "
                    "dilatih ulang dari nol pada 10 pasien; retrieval dievaluasi pada 2 pasien "
                    "yang tak pernah dilihat. Ground truth = kondisi glukosa yang benar-benar terjadi."),
        "konfigurasi_efektif": {
            "keluarga_prediktor": type(reg).__name__,
            "keluarga_pengklasifikasi": type(clf).__name__,
            "class_weight": "balanced",
            "lambda_mult": rag_cfg.lambda_mult,
            "fetch_k": rag_cfg.fetch_k,
            "top_k": rag_cfg.top_k,
            "chunk_size": rag_cfg.chunk_size,
            "chunk_overlap": rag_cfg.chunk_overlap,
            "embedding_model": rag_cfg.embedding_model,
            "collection_name": rag_cfg.collection_name,
            "max_gap_steps": mc.get("max_gap_steps"),
            "horizon_steps": HORIZON,
            "seed": SEED,
            "corpus_tag": CORPUS_TAG,
        },
        "kontrol_tak_bergantung_prediktor": (
            "Mode `standard` dan `oracle`, serta pemilihan kasus divergen, TIDAK menyentuh "
            "model sama sekali. Ketiganya wajib tereproduksi persis terhadap jalan sebelumnya; "
            "bila tidak, yang berubah adalah jalur pengukurannya. Lihat "
            "docs/PRAPENDAFTARAN_T6_GBM_RETRIEVAL.md dugaan D1."
        ),
        "n_fold": len(folds), "top_k": TOP_K, "n_kasus_per_himpunan_per_fold": N_PER_SET,
        "ringkasan_lintas_fold": ringkas,
        "per_fold": per_fold,
    }
    # Direktori keluaran dibuat SEBELUM menulis. Tanpa ini, seluruh komputasi 6 fold
    # (yang memakan lebih dari satu jam) hilang di baris terakhir hanya karena
    # direktorinya belum ada.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== Rerata +/- SD lintas 6 fold (MRR) ===")
    for label in ("divergen", "natural"):
        print(f"  [{label}]")
        for m in MODES:
            s = ringkas[label][m]
            print(f"    {m:20s} {s['mrr_rerata']:.3f} +/- {s['mrr_sd']:.3f}")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
