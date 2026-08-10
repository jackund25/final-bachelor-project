"""T5 — Baseline ablation: standard RAG vs Prediction-Conditioned RAG.

Bukti novelty: pada kasus DIVERGEN (glukosa saat ini normal, tapi PREDIKSI hipo/hiper),
mode prediction-conditioned mengambil panduan klinis ANTISIPATIF yang dilewatkan mode
standard (yang hanya melihat kondisi saat ini).

Mode:
- standard               : query dibangun dari glukosa SAAT INI (predicted = current).
- prediction_conditioned : query dibangun dari glukosa PREDIKSI (PredictionConditionedQueryBuilder).

Retrieval memakai embedding yang sama dengan sistem (sentence-transformers all-MiniLM-L6-v2)
atas knowledge base klinis (manual_kb.json). Ground truth = dokumen yang topiknya sesuai
kondisi PREDIKSI.

Output: results/baseline_ablation/
  - ablation_per_case.csv      (hasil tiap kasus x mode)
  - ablation_summary.csv       (agregat metrik per mode)
  - qualitative_examples.md    (contoh kualitatif untuk Bab VI)
"""
from __future__ import annotations

# PENTING (Windows + conda): torch HARUS di-import sebelum numpy/pandas, jika tidak
# inisialisasi c10.dll gagal (WinError 1114) karena konflik DLL runtime. Lihat docs/journey/.
import torch  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.patient_state import PatientState
from src.rag.conditioned_query import PredictionConditionedQueryBuilder, QueryStrategy

TOP_K = 3
KB_PATH = Path("data/knowledge_base/manual_kb.json")
OUT_DIR = Path("results/baseline_ablation")


from src.constants import classify_glucose_3class

# Ambang dari SATU sumber kebenaran (src/constants.py).
classify = classify_glucose_3class

# Horizon prediksi dalam MENIT, diturunkan dari config (model.default_horizon x
# data.sampling_interval_min). Sebelumnya sebagian skrip menuliskan "60 menit" dan
# sebagian "30 menit" secara hardcoded, atas korpus dan kasus yang sama — sehingga
# angkanya tidak sebanding satu sama lain maupun dengan produksi.
from src.config import cfg_get

HORIZON_MIN = int(cfg_get("model.default_horizon", 6) * cfg_get("data.sampling_interval_min", 5))


# Pemetaan kondisi → topik dokumen ground-truth di KB
COND_TO_TOPIC = {"hipoglikemia": "Hipoglikemia", "hiperglikemia": "Hiperglikemia", "normal": "Target Kontrol Glikemik"}

# ──────────────────────────────────────────────────────────────
# Test set "kasus divergen": current NORMAL (70-180), prediksi HIPO/HIPER.
# expected_topic = dokumen ground-truth (sesuai kondisi PREDIKSI).
# ──────────────────────────────────────────────────────────────
TEST_CASES = [
    {"id": "D1", "scenario": "Insulin menumpuk pasca-bolus", "current": 112.0, "predicted": 58.0,
     "expected_topic": "Hipoglikemia", "feature_row": {"insulin": 4.5, "carbs": 0.0, "activity": 0, "stress": 4}},
    {"id": "D2", "scenario": "Olahraga tanpa penyesuaian insulin", "current": 98.0, "predicted": 64.0,
     "expected_topic": "Hipoglikemia", "feature_row": {"insulin": 2.0, "carbs": 0.0, "activity": 60, "stress": 3}},
    {"id": "D3", "scenario": "Tren turun cepat sebelum tidur", "current": 128.0, "predicted": 66.0,
     "expected_topic": "Hipoglikemia", "feature_row": {"insulin": 3.0, "carbs": 0.0, "activity": 20, "stress": 5}},
    {"id": "D4", "scenario": "Makan tinggi karbohidrat", "current": 150.0, "predicted": 214.0,
     "expected_topic": "Hiperglikemia", "feature_row": {"insulin": 0.0, "carbs": 70.0, "activity": 0, "stress": 5}},
    {"id": "D5", "scenario": "Karbohidrat + stres tinggi", "current": 162.0, "predicted": 205.0,
     "expected_topic": "Hiperglikemia", "feature_row": {"insulin": 0.0, "carbs": 45.0, "activity": 0, "stress": 8}},
    {"id": "D6", "scenario": "Dosis insulin kurang", "current": 140.0, "predicted": 238.0,
     "expected_topic": "Hiperglikemia", "feature_row": {"insulin": 0.0, "carbs": 55.0, "activity": 0, "stress": 6}},
]


def load_kb():
    docs = json.loads(KB_PATH.read_text(encoding="utf-8"))
    texts = [d["text"] for d in docs]
    topics = [d["topic"] for d in docs]
    return docs, texts, topics


# Frasa fokus per kondisi — dari SATU sumber kebenaran (src/rag/ablation_query.py),
# sama persis dengan yang dipakai ablation_rag_fullkb, crossfold, dan realcases.
from src.rag.ablation_query import CONDITION_PHRASE, build_ablation_query  # noqa: E402,F401


def targeted_condition(case, mode: str) -> str:
    """Kondisi yang menjadi fokus query: dari current (standard) atau predicted (conditioned)."""
    g = case["current"] if mode == "standard" else case["predicted"]
    return classify(g)


def build_query(case, mode: str) -> str:
    """Template BENAR-BENAR identik; berbeda HANYA pada angka (current vs predicted).

    - standard               : query fokus pada kondisi glukosa SAAT INI.
    - prediction_conditioned : query fokus pada kondisi glukosa PREDIKSI.

    Sufiks "(prediksi N menit ke depan)" dihapus dari lengan conditioned: ia membuat
    kedua kueri berbeda pada dua hal sekaligus, sehingga selisih metrik tidak bisa
    dikaitkan murni pada sumber pengondisian.
    """
    g = case["current"] if mode == "standard" else case["predicted"]
    return build_ablation_query(g, cond=classify(g))


def main():
    from sentence_transformers import SentenceTransformer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    docs, texts, topics = load_kb()
    print(f"KB dimuat: {len(docs)} dokumen | topik: {topics}")

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    doc_emb = model.encode(texts, normalize_embeddings=True)

    rows = []
    qual = []
    for case in TEST_CASES:
        needed_cond = classify(case["predicted"])  # kondisi klinis yang BENAR-BENAR dibutuhkan
        for mode in ("standard", "prediction_conditioned"):
            query = build_query(case, mode)
            tgt_cond = targeted_condition(case, mode)
            targets_needed = int(tgt_cond == needed_cond)
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            sims = doc_emb @ q_emb  # cosine (embeddings ternormalisasi)
            order = np.argsort(-sims)[:TOP_K]
            retrieved_topics = [topics[i] for i in order]
            hit = case["expected_topic"] in retrieved_topics
            hit1 = retrieved_topics[0] == case["expected_topic"]
            rank = (retrieved_topics.index(case["expected_topic"]) + 1) if hit else 0
            mrr = (1.0 / rank) if rank else 0.0
            precision = sum(1 for t in retrieved_topics if t == case["expected_topic"]) / TOP_K

            rows.append({
                "case": case["id"], "scenario": case["scenario"],
                "current": case["current"], "predicted": case["predicted"],
                "expected_topic": case["expected_topic"], "mode": mode,
                "targeted_condition": tgt_cond, "targets_needed": targets_needed,
                "top1_topic": retrieved_topics[0],
                "retrieved_topics": " | ".join(retrieved_topics),
                "hit@1": int(hit1), "hit@{}".format(TOP_K): int(hit),
                "rank": rank, "mrr": round(mrr, 3),
                "precision@{}".format(TOP_K): round(precision, 3),
            })

            if case["id"] in ("D2", "D4"):
                qual.append((case, mode, query, retrieved_topics, sims[order]))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ablation_per_case.csv", index=False)

    # Agregat per mode
    hitcol = f"hit@{TOP_K}"; precol = f"precision@{TOP_K}"
    summary = df.groupby("mode").agg(
        n_cases=("case", "count"),
        targets_needed=("targets_needed", "mean"),
        hit_at_1=("hit@1", "mean"),
        hit_at_k=(hitcol, "mean"),
        mrr=("mrr", "mean"),
        precision=(precol, "mean"),
    ).reset_index()
    for c in ["targets_needed", "hit_at_1", "hit_at_k", "precision"]:
        summary[c] = (summary[c] * 100).round(1)
    summary["mrr"] = summary["mrr"].round(3)
    summary = summary.rename(columns={
        "targets_needed": "targets_needed(%)", "hit_at_1": "hit@1(%)",
        "hit_at_k": f"hit@{TOP_K}(%)", "precision": f"precision@{TOP_K}(%)"})
    summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)

    # Contoh kualitatif untuk Bab VI
    lines = ["# Contoh Kualitatif — Ablation Standard vs Prediction-Conditioned RAG\n"]
    for case, mode, query, rtopics, rsims in qual:
        lines.append(f"\n## Kasus {case['id']} ({case['scenario']}) — mode: {mode}")
        lines.append(f"- Current: {case['current']} mg/dL | Predicted: {case['predicted']} mg/dL "
                     f"| Ground truth: **{case['expected_topic']}**")
        lines.append(f"- Query: _{query}_")
        lines.append(f"- Top-{TOP_K} dokumen ter-retrieve: " +
                     ", ".join(f"{t} ({s:.2f})" for t, s in zip(rtopics, rsims)))
        hit = "✓ menemukan" if case["expected_topic"] in rtopics else "✗ TIDAK menemukan"
        lines.append(f"- Hasil: {hit} dokumen antisipatif.")
    (OUT_DIR / "qualitative_examples.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== HASIL PER KASUS ===")
    print(df[["case", "mode", "expected_topic", "targeted_condition", "targets_needed", "top1_topic", "hit@1"]].to_string(index=False))
    print("\n=== RINGKASAN PER MODE ===")
    print(summary.to_string(index=False))
    print(f"\nOutput -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
