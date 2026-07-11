"""Evaluasi novelty di level GENERASI (bukan hanya retrieval) — konsisten & bebas judge.

Generasi memakai PIPELINE APP ASLI (Gemini flash-lite) → output nyata sistem.
Penilaian LOKAL & deterministik (tanpa LLM-judge → tanpa tembok kuota):
  1. Kemiripan embedding jawaban ↔ REFERENSI ANTISIPATIF benar (kondisi terprediksi).
  2. Cakupan aksi klinis (checklist per kondisi).

Mode:
  - standard               : answer(state, prediction=current)   → query dari kondisi saat-ini
  - prediction_conditioned : answer(state, prediction=predicted) → query dari prediksi
Hipotesis: conditioned lebih mirip referensi antisipatif & mencakup lebih banyak aksi benar.

Kredensial via env (set GOOGLE_API_KEY ke key flash-lite yang segar). Output:
results/eval_prediksi/generation_novelty.json
"""
import torch  # noqa: F401
import os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
from pathlib import Path
import numpy as np
from src.rag import RAGPipeline

OUT = Path("results/eval_prediksi/generation_novelty.json")

CASES = [
    {"id": "D1", "cur": 112.0, "pred": 58.0, "cond": "hipo", "iob": 4.5, "cob": 0, "act": 0, "str": 4},
    {"id": "D2", "cur": 98.0, "pred": 64.0, "cond": "hipo", "iob": 2.0, "cob": 0, "act": 60, "str": 3},
    {"id": "D3", "cur": 128.0, "pred": 66.0, "cond": "hipo", "iob": 3.0, "cob": 0, "act": 20, "str": 5},
    {"id": "D4", "cur": 150.0, "pred": 214.0, "cond": "hiper", "iob": 0, "cob": 70, "act": 0, "str": 5},
    {"id": "D5", "cur": 162.0, "pred": 205.0, "cond": "hiper", "iob": 0, "cob": 45, "act": 0, "str": 8},
    {"id": "D6", "cur": 140.0, "pred": 238.0, "cond": "hiper", "iob": 0, "cob": 55, "act": 0, "str": 6},
]
REF = {
    "hipo": ("Antisipasi hipoglikemia: berikan 15 gram karbohidrat cepat seperti tablet glukosa atau jus, "
             "tunggu 15 menit lalu cek ulang glukosa, ulangi bila masih di bawah 70 mg/dL. Pantau ketat, "
             "siapkan sumber glukosa; glukagon bila berat."),
    "hiper": ("Antisipasi hiperglikemia: perbanyak minum air putih, lakukan aktivitas fisik ringan, hindari "
              "tambahan karbohidrat, periksa keton bila di atas 250 mg/dL, konsultasi dokter bila menetap."),
}
ACTIONS = {
    "hipo": ["karbohidrat", "15", "cek", "pantau", "glukosa"],
    "hiper": ["air", "aktivitas", "hindari", "keton", "dokter"],
}


def coverage(text, cond):
    t = text.lower()
    items = ACTIONS[cond]
    hit = sum(1 for kw in items if kw in t)
    return hit / len(items)


def main():
    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    ref_vec = {k: emb.encode(v, normalize_embeddings=True) for k, v in REF.items()}

    p = RAGPipeline(kb_dir="data/knowledge_base", llm_provider="gemini",
                    embed_provider="sentence-transformers")
    p.build()
    print(f"Model generasi: {os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')} | key set: {bool(os.getenv('GOOGLE_API_KEY'))}")

    rows = []
    for case in CASES:
        state = {"current_glucose": case["cur"], "insulin_on_board": case["iob"],
                 "carbs_on_board": case["cob"], "activity_level": case["act"], "stress_level": case["str"]}
        for mode in ("standard", "prediction_conditioned"):
            pval = case["cur"] if mode == "standard" else case["pred"]
            try:
                res = p.answer(patient_state=state, prediction=pval)
                expl = res.get("explanation", "") or ""
            except Exception as e:  # noqa: BLE001
                expl = ""
                print(f"  {case['id']} {mode}: GAGAL {str(e)[:70]}")
            sim = float(np.dot(emb.encode(expl, normalize_embeddings=True), ref_vec[case["cond"]])) if expl else float("nan")
            cov = coverage(expl, case["cond"]) if expl else float("nan")
            rows.append({"case": case["id"], "cond": case["cond"], "mode": mode,
                         "sim_ref": round(sim, 3), "action_cov": round(cov, 3), "len": len(expl)})
            print(f"  {case['id']} {mode:22s}: sim={sim:.3f} cov={cov:.2f} ({len(expl)} char)")
            time.sleep(7)  # rate limit 10/menit free-tier

    # agregat per mode
    def agg(mode, key):
        v = [r[key] for r in rows if r["mode"] == mode and not np.isnan(r[key])]
        return round(float(np.mean(v)), 3) if v else float("nan")

    summary = {m: {"sim_ref_mean": agg(m, "sim_ref"), "action_cov_mean": agg(m, "action_cov")}
               for m in ("standard", "prediction_conditioned")}
    res = {"n_cases": len(CASES), "per_sample": rows, "summary": summary}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2)

    print("\n=== RINGKASAN (novelty di level generasi) ===")
    print(f"  {'mode':24s} {'sim-ref':>8} {'cakupan aksi':>13}")
    for m, s in summary.items():
        print(f"  {m:24s} {s['sim_ref_mean']:>8} {s['action_cov_mean']:>13}")
    c, st = summary["prediction_conditioned"], summary["standard"]
    print(f"\nConditioned {'UNGGUL' if c['sim_ref_mean'] > st['sim_ref_mean'] else 'TIDAK unggul'} pada kemiripan; "
          f"{'UNGGUL' if c['action_cov_mean'] > st['action_cov_mean'] else 'TIDAK unggul'} pada cakupan aksi.")
    print(f"Output -> {OUT}")


if __name__ == "__main__":
    main()
