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
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from ablation_rag_fullkb import CONDITION_PHRASE, classify_chunk, classify_glucose, ndcg_at_k  # noqa: E402

TOP_K = 5
HORIZON = 6
N_PER_SET = 60      # kasus per himpunan per fold (ditekan agar runtime wajar)
MIN_GAP = 6
SEED = 42
MODES = ["standard", "pc_rag", "pc_rag_classifier", "oracle"]
OUT = ROOT / "results/retrieval_realcases/crossfold.json"


def build_frame(cfg: dict):
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(mc["engineered_features"])
    return df, pre


def seqs(pre, d, mc):
    X, y, anc = pre.create_sequences(d, mc["sequence_length"], HORIZON, return_anchor=True)
    n, s, f = X.shape
    return X.reshape(n, s * f), y, anc


def build_query(g: float, is_pred: bool, cond: str | None = None) -> str:
    c = cond or classify_glucose(g)
    horizon = " (prediksi 30 menit ke depan)" if is_pred else ""
    return f"Kadar glukosa darah {g:.0f} mg/dL{horizon}. {CONDITION_PHRASE[c]}"


def score(r, query: str, expected: str) -> dict:
    docs = r.retrieve(query, top_k=TOP_K)
    topics = [classify_chunk(d["text"]) for d in docs]
    rank = (topics.index(expected) + 1) if expected in topics else 0
    rels = [1 if t == expected else 0 for t in topics]
    return {"hit@1": int(bool(topics) and topics[0] == expected),
            "mrr": (1.0 / rank) if rank else 0.0,
            "ndcg": ndcg_at_k(rels, TOP_K)}


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
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    mc = cfg["model"]
    rf_cfg = mc["random_forest"]
    df, pre = build_frame(cfg)

    patients = sorted(df["patient_id"].unique())
    folds = [patients[i:i + 2] for i in range(0, len(patients), 2)]
    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")

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

        reg = RandomForestRegressor(n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth"],
                                    min_samples_split=rf_cfg["min_samples_split"],
                                    random_state=SEED, n_jobs=-1)
        reg.fit(Xtr_s, ytr - atr)
        pred = reg.predict(Xte_s) + ate

        lbl_tr = np.array([classify_glucose(v) for v in ytr])
        clf = RandomForestClassifier(n_estimators=rf_cfg["n_estimators"], max_depth=rf_cfg["max_depth"],
                                     min_samples_split=rf_cfg["min_samples_split"],
                                     class_weight="balanced", random_state=SEED, n_jobs=-1)
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
                    s = score(r, q, exp)
                    agg[m]["hit@1"].append(s["hit@1"])
                    agg[m]["mrr"].append(s["mrr"])
            fold_res[label] = {"n": int(len(sub)),
                               **{m: {"hit@1": round(float(np.mean(agg[m]["hit@1"])), 3),
                                      "mrr": round(float(np.mean(agg[m]["mrr"])), 3)} for m in MODES}}

        per_fold.append(fold_res)
        d, n = fold_res["divergen"], fold_res["natural"]
        print(f"fold {fi} {test_p} | divergen MRR: std {d['standard']['mrr']:.3f} "
              f"pc {d['pc_rag']['mrr']:.3f} clf {d['pc_rag_classifier']['mrr']:.3f} "
              f"oracle {d['oracle']['mrr']:.3f} || natural MRR: std {n['standard']['mrr']:.3f} "
              f"pc {n['pc_rag']['mrr']:.3f} clf {n['pc_rag_classifier']['mrr']:.3f}")

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

    out = {
        "catatan": ("Enam fold lintas-pasien. Pada tiap fold, model regresi dan pengklasifikasi "
                    "dilatih ulang dari nol pada 10 pasien; retrieval dievaluasi pada 2 pasien "
                    "yang tak pernah dilihat. Ground truth = kondisi glukosa yang benar-benar terjadi."),
        "n_fold": len(folds), "top_k": TOP_K, "n_kasus_per_himpunan_per_fold": N_PER_SET,
        "ringkasan_lintas_fold": ringkas,
        "per_fold": per_fold,
    }
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
