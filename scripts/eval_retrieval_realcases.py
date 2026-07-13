"""Evaluasi retrieval PC-RAG pada KASUS NYATA + uji rantai kausal prediksi -> retrieval.

Motivasi. Evaluasi retrieval awal (ablation_rag_fullkb.py) memakai enam kasus yang disusun
manual. Dua kelemahan mendasar mengikutinya: (a) jumlah kasus terlalu kecil untuk uji
signifikansi, dan (b) kondisi yang "diharapkan" ditetapkan dari nilai yang DIPREDIKSI,
sehingga PC-RAG dinilai terhadap sasaran yang ia tentukan sendiri — metrik bisa dituduh
menang karena konstruksi, bukan karena metodenya bekerja.

Skrip ini memperbaiki keduanya.

1. KASUS NYATA. Kasus diambil dari jendela sungguhan milik dua pasien hold-out yang tidak
   pernah dilihat model saat pelatihan, bukan dari skenario karangan.

2. GROUND TRUTH DARI MASA DEPAN YANG SEBENARNYA. Kondisi relevan sebuah kasus ditetapkan
   dari kadar glukosa yang BENAR-BENAR TERJADI pada t+h (kanal CGM), bukan dari nilai
   prediksi. Konsekuensinya penting: bila model salah memprediksi, PC-RAG akan membentuk
   kueri untuk kondisi yang keliru dan DIHUKUM oleh metrik. Metrik ini karenanya menguji
   metode, bukan sekadar mencocokkan kata kunci antara kueri dan dokumen.

3. RANTAI KAUSAL. Kualitas prediksi divariasikan secara terkontrol untuk menguji apakah
   manfaat retrieval benar-benar mengalir dari kualitas prediksi:
     - standard    : kueri dari kondisi TERKINI (setara baseline persistence)
     - pc_rag      : kueri dari prediksi Random Forest
     - noise_sigma : kueri dari prediksi RF + galat gaussian (sigma = 10/20/40 mg/dL)
     - oracle      : kueri dari glukosa masa depan yang SEBENARNYA (batas atas teoretis)
   Bila metrik hanya "menghadiahi kata kunci", keempatnya akan setara. Bila manfaat memang
   mengalir dari prediksi, skor akan menurun monoton seiring memburuknya prediksi.

Fokus analisis adalah kasus DIVERGEN: kondisi terkini berbeda dari kondisi yang akan datang
(mis. sekarang normal, 30 menit lagi hipoglikemia). Di sanalah antisipasi bernilai; pada
kasus non-divergen, kueri dari kondisi terkini sudah menunjuk kondisi yang benar.

Keluaran: results/retrieval_realcases/{summary.json, per_case.csv}
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.preprocessor import DataPreprocessor  # noqa: E402
from src.rag.retriever import MMRRetriever  # noqa: E402
from ablation_rag_fullkb import (  # noqa: E402  — dipakai ulang agar identik dengan evaluasi awal
    CONDITION_PHRASE, classify_chunk, classify_glucose, ndcg_at_k,
)

TOP_K = 5
HORIZON = 6              # +30 menit
MAX_CASES = 120          # kasus divergen yang dievaluasi
MIN_GAP_STEPS = 6        # jarak minimum antar-kasus (30 menit) agar tidak nyaris duplikat
NOISE_SIGMAS = [10.0, 20.0, 40.0]
CONFORMAL_K = 3.3        # faktor interval konformal ternormalisasi (identik dengan aplikasi)
SEED = 42
OUT = ROOT / "results/retrieval_realcases"


def build_query(glucose: float, is_prediction: bool) -> str:
    """Identik dengan pembentuk kueri pada evaluasi awal (agar sebanding)."""
    cond = classify_glucose(glucose)
    horizon = " (prediksi 30 menit ke depan)" if is_prediction else ""
    return f"Kadar glukosa darah {glucose:.0f} mg/dL{horizon}. {CONDITION_PHRASE[cond]}"


def build_interval_query(pred: float, lo: float, hi: float) -> str:
    """PC-RAG sadar-ketidakpastian: kueri dikondisikan pada INTERVAL prediksi, bukan titik.

    Prediksi titik Random Forest cenderung menyusut ke tengah (under-dispersed), sehingga
    jarang melewati ambang 70/180 meski risikonya nyata. Dengan mengondisikan kueri pada
    interval konformal, kondisi berisiko yang tercakup interval tetap diambilkan dokumennya
    walau prediksi titiknya masih "normal".
    """
    conds = {classify_glucose(pred)}
    if lo < 70:
        conds.add("hipoglikemia")
    if hi > 180:
        conds.add("hiperglikemia")
    # dahulukan kondisi berisiko agar frasa risiko berada di awal kueri
    order = ["hipoglikemia", "hiperglikemia", "normal"]
    phrases = " ".join(CONDITION_PHRASE[c] for c in order if c in conds)
    return (f"Kadar glukosa darah diprediksi {pred:.0f} mg/dL "
            f"(interval 95%: {lo:.0f}-{hi:.0f} mg/dL, 30 menit ke depan). {phrases}")


def collect_cases(cfg: dict) -> pd.DataFrame:
    """Ambil jendela nyata dari pasien hold-out: current, prediksi RF, dan masa depan aktual."""
    mc = cfg["model"]
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])

    pre = DataPreprocessor(cfg)
    df = pre.handle_missing_values(df)
    df = pre.engineer_features(df, **mc["feature_engineering"])
    pre.feature_columns = list(mc["engineered_features"])

    patients = sorted(df["patient_id"].unique())
    _, test_df = pre.split_by_patient(df, patients[-2:])

    X, y, anchor = pre.create_sequences(
        test_df, mc["sequence_length"], HORIZON, return_anchor=True
    )
    with open(ROOT / f"models/rf_inference_bundle_h{HORIZON}.pkl", "rb") as f:
        bundle = pickle.load(f)

    n, seq, n_feat = X.shape
    Xs = bundle["scaler"].transform(X.reshape(-1, n_feat)).reshape(n, seq * n_feat)
    model = bundle["model"]
    pred = model.predict(Xs)

    # simpangan baku antar-pohon -> interval konformal ternormalisasi (K=3,3; lihat Bab VI)
    per_tree = np.stack([t.predict(Xs) for t in model.estimators_])
    sigma = per_tree.std(axis=0)

    if bundle["predict_delta"]:
        pred = pred + anchor

    # pengklasifikasi kondisi (train_condition_classifier.py) — bila tersedia
    clf_path = ROOT / f"models/rf_condition_classifier_h{HORIZON}.pkl"
    cond_clf = None
    if clf_path.exists():
        with open(clf_path, "rb") as f:
            cond_clf = pickle.load(f)["model"].predict(Xs)

    cases = pd.DataFrame({
        "idx": np.arange(n),
        "current": anchor,
        "predicted": pred,
        "sigma": sigma,
        "lo95": pred - CONFORMAL_K * sigma,
        "hi95": pred + CONFORMAL_K * sigma,
        "actual_future": y,
    })
    if cond_clf is not None:
        cases["cond_classifier"] = cond_clf
    cases["cond_current"] = [classify_glucose(g) for g in cases["current"]]
    cases["cond_predicted"] = [classify_glucose(g) for g in cases["predicted"]]
    cases["cond_actual"] = [classify_glucose(g) for g in cases["actual_future"]]
    cases["divergent"] = cases["cond_current"] != cases["cond_actual"]
    return cases


def sample_divergent(cases: pd.DataFrame) -> pd.DataFrame:
    """Kasus divergen, dijarangkan agar tidak nyaris duplikat, seimbang hipo/hiper."""
    rng = np.random.default_rng(SEED)
    div = cases[cases["divergent"]].copy()

    picked, last = [], -10**9
    for _, row in div.sort_values("idx").iterrows():
        if row["idx"] - last >= MIN_GAP_STEPS:
            picked.append(row)
            last = row["idx"]
    div = pd.DataFrame(picked)

    # seimbangkan antar-kondisi masa depan
    per_cond = max(1, MAX_CASES // max(1, div["cond_actual"].nunique()))
    parts = []
    for cond, g in div.groupby("cond_actual"):
        take = min(len(g), per_cond)
        parts.append(g.sample(n=take, random_state=SEED) if len(g) > take else g)
    out = pd.concat(parts).sort_values("idx").reset_index(drop=True)
    if len(out) > MAX_CASES:
        out = out.sample(n=MAX_CASES, random_state=SEED).sort_values("idx").reset_index(drop=True)
    return out


def score(retriever: MMRRetriever, query: str, expected: str) -> dict:
    docs = retriever.retrieve(query, top_k=TOP_K)
    topics = [classify_chunk(d["text"]) for d in docs]
    rank = (topics.index(expected) + 1) if expected in topics else 0
    rels = [1 if t == expected else 0 for t in topics]
    return {
        "hit@1": int(bool(topics) and topics[0] == expected),
        f"hit@{TOP_K}": int(expected in topics),
        "mrr": (1.0 / rank) if rank else 0.0,
        f"ndcg@{TOP_K}": ndcg_at_k(rels, TOP_K),
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sample_natural(cases: pd.DataFrame, n: int = 120) -> pd.DataFrame:
    """Sampel acak dari SELURUH jendela (distribusi natural: mayoritas non-divergen).

    Ini konteks penerapan sesungguhnya — sistem tidak tahu sebuah kasus divergen atau tidak.
    """
    rng = np.random.default_rng(SEED + 1)
    idx = rng.choice(len(cases), size=min(n, len(cases)), replace=False)
    return cases.iloc[np.sort(idx)].reset_index(drop=True)


def evaluate(cases: pd.DataFrame, r: MMRRetriever, label: str) -> tuple[pd.DataFrame, dict, dict]:
    """Jalankan seluruh mode retrieval pada satu himpunan kasus."""
    rng = np.random.default_rng(SEED)

    modes = ["random", "standard", "pc_rag", "pc_rag_interval"]
    if "cond_classifier" in cases.columns:
        modes += ["pc_rag_classifier", "pc_rag_combined"]
    modes += [f"noise_{int(s)}" for s in NOISE_SIGMAS] + ["oracle"]
    rows = []
    for _, c in cases.iterrows():
        expected = c["cond_actual"]          # <-- ground truth = masa depan SEBENARNYA
        for mode in modes:
            covered = None
            if mode == "random":
                g = float(rng.choice([55.0, 120.0, 220.0]))  # kondisi acak (hipo/normal/hiper)
                q, is_pred = build_query(g, True), True
            elif mode == "standard":
                g, is_pred = float(c["current"]), False
                q = build_query(g, is_pred)
            elif mode == "pc_rag":
                g, is_pred = float(c["predicted"]), True
                q = build_query(g, is_pred)
            elif mode == "pc_rag_interval":
                g, is_pred = float(c["predicted"]), True
                q = build_interval_query(g, float(c["lo95"]), float(c["hi95"]))
                # kondisi "tercakup" bila interval memuat kondisi sebenarnya
                cov = {classify_glucose(g)}
                if c["lo95"] < 70:
                    cov.add("hipoglikemia")
                if c["hi95"] > 180:
                    cov.add("hiperglikemia")
                covered = int(expected in cov)
            elif mode == "pc_rag_classifier":
                # kueri dikondisikan pada KONDISI hasil pengklasifikasi, bukan pada nilai regresi
                cond = str(c["cond_classifier"])
                g, is_pred = float(c["predicted"]), True
                q = (f"Kadar glukosa darah diprediksi {g:.0f} mg/dL "
                     f"(30 menit ke depan). {CONDITION_PHRASE[cond]}")
                covered = int(cond == expected)
            elif mode == "pc_rag_combined":
                # Konfigurasi yang BENAR-BENAR dijalankan aplikasi: kondisi dari pengklasifikasi,
                # ditambah kondisi berisiko yang masih tercakup interval konformal.
                cond = str(c["cond_classifier"])
                g, is_pred = float(c["predicted"]), True
                cov = {cond}
                if c["lo95"] < 70:
                    cov.add("hipoglikemia")
                if c["hi95"] > 180:
                    cov.add("hiperglikemia")
                order = ["hipoglikemia", "hiperglikemia", "normal"]
                phrases = " ".join(CONDITION_PHRASE[x] for x in order if x in cov)
                q = (f"Kadar glukosa darah diprediksi {g:.0f} mg/dL "
                     f"(interval 95%: {c['lo95']:.0f}-{c['hi95']:.0f} mg/dL, 30 menit ke depan). "
                     f"{phrases}")
                covered = int(expected in cov)
            elif mode == "oracle":
                g, is_pred = float(c["actual_future"]), True
                q = build_query(g, is_pred)
            else:
                sigma = float(mode.split("_")[1])
                g, is_pred = float(c["predicted"] + rng.normal(0, sigma)), True
                q = build_query(g, is_pred)

            m = score(r, q, expected)
            hit_cond = covered if covered is not None else int(classify_glucose(g) == expected)
            rows.append({
                "idx": int(c["idx"]), "mode": mode,
                "current": round(float(c["current"]), 1),
                "query_glucose": round(g, 1),
                "actual_future": round(float(c["actual_future"]), 1),
                "expected": expected,
                "condition_correct": hit_cond,
                **m,
            })

    df = pd.DataFrame(rows)
    df["himpunan"] = label
    df.to_csv(OUT / f"per_case_{label}.csv", index=False)

    metrics = ["hit@1", f"hit@{TOP_K}", "mrr", f"ndcg@{TOP_K}"]
    summary = {}
    for mode in modes:
        d = df[df["mode"] == mode]
        entry = {"condition_correct(%)": round(100 * d["condition_correct"].mean(), 1)}
        for met in metrics:
            v = d[met].to_numpy(float)
            lo, hi = bootstrap_ci(v)
            entry[met] = round(float(v.mean()), 3)
            entry[f"{met}_ci95"] = [round(lo, 3), round(hi, 3)]
        summary[mode] = entry

    # uji signifikansi berpasangan terhadap RAG standar (per kasus)
    piv = df.pivot(index="idx", columns="mode", values="mrr")
    sig = {}
    for mode in modes:
        if mode == "standard":
            continue
        diff = piv[mode] - piv["standard"]
        if np.allclose(diff, 0):
            sig[mode] = {"wilcoxon_p": 1.0, "selisih_rerata_mrr": 0.0}
            continue
        _, p = wilcoxon(piv[mode], piv["standard"])
        sig[mode] = {"wilcoxon_p": float(p), "selisih_rerata_mrr": round(float(diff.mean()), 3)}

    print(f"\n### Himpunan: {label} (n={len(cases)})")
    print(f"{'mode':<18} {'kondisi benar':>13} {'Hit@1':>7} {'MRR':>7}  {'MRR CI95':>16}  {'p vs standar':>12}")
    for mode in modes:
        s = summary[mode]
        ci = s["mrr_ci95"]
        pv = "" if mode == "standard" else f"{sig[mode]['wilcoxon_p']:.1e}"
        print(f"{mode:<18} {s['condition_correct(%)']:>12.1f}% {s['hit@1']:>7.3f} "
              f"{s['mrr']:>7.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  {pv:>12}")
    return df, summary, sig


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    cases_all = collect_cases(cfg)
    div = sample_divergent(cases_all)
    nat = sample_natural(cases_all)

    print(f"Total jendela hold-out  : {len(cases_all)}")
    print(f"Kasus divergen tersedia : {int(cases_all['divergent'].sum())} "
          f"({100*cases_all['divergent'].mean():.1f}% dari seluruh jendela)")
    print(f"Himpunan divergen       : {len(div)} ({dict(div['cond_actual'].value_counts())})")
    print(f"Himpunan natural        : {len(nat)} ({dict(nat['cond_actual'].value_counts())}), "
          f"divergen di dalamnya: {int(nat['divergent'].sum())}")

    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")

    _, s_div, sig_div = evaluate(div, r, "divergen")
    _, s_nat, sig_nat = evaluate(nat, r, "natural")

    out = {
        "catatan": (
            "Kasus nyata dari 2 pasien hold-out. Ground truth = kondisi glukosa yang BENAR-BENAR "
            "terjadi pada t+30 menit (bukan nilai prediksi), sehingga kesalahan prediksi dihukum. "
            "Himpunan 'divergen' = kondisi terkini berbeda dari kondisi masa depan (tempat "
            "antisipasi bernilai, diseimbangkan antar-kelas). Himpunan 'natural' = sampel acak "
            "dari seluruh jendela (konteks penerapan sesungguhnya)."
        ),
        "n_windows_holdout": int(len(cases_all)),
        "n_divergent_available": int(cases_all["divergent"].sum()),
        "proporsi_divergen(%)": round(100 * float(cases_all["divergent"].mean()), 1),
        "top_k": TOP_K,
        "divergen": {
            "n": int(len(div)),
            "distribusi": {k: int(v) for k, v in div["cond_actual"].value_counts().items()},
            "modes": s_div,
            "signifikansi_vs_standard": sig_div,
        },
        "natural": {
            "n": int(len(nat)),
            "distribusi": {k: int(v) for k, v in nat["cond_actual"].value_counts().items()},
            "n_divergen_di_dalamnya": int(nat["divergent"].sum()),
            "modes": s_nat,
            "signifikansi_vs_standard": sig_nat,
        },
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nDisimpan ke {OUT}/")


if __name__ == "__main__":
    main()
