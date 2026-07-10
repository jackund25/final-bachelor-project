#!/usr/bin/env python3
"""T6 — Evaluasi RAGAS: standard vs Prediction-Conditioned RAG (judge: Gemini).

Untuk tiap kasus (patient state), kedua mode menjawab pertanyaan klinis yang SAMA namun
dengan konteks yang di-retrieve berbeda (standard: dikondisikan glukosa saat ini;
prediction_conditioned: dikondisikan glukosa prediksi). RAGAS menilai:
  - faithfulness       : jawaban berdasar konteks (anti-halusinasi)
  - answer_relevancy   : relevansi jawaban thd pertanyaan
  - context_recall     : konteks memuat informasi rujukan (butuh reference)
  - context_precision  : konteks yang diambil relevan (butuh reference)

Judge = Gemini (LLM-as-judge; lihat caveat di bawah). Embedding = all-MiniLM (lokal).
Output: results/ragas/  (per_sample.csv, summary.csv, caveat.md)

CAVEAT LLM-as-judge: skor RAGAS dihasilkan oleh LLM (Gemini) yang menilai output LLM lain,
sehingga berpotensi bias self-referential & bervariasi antar-run. Angka bersifat indikatif,
bukan absolut. context_recall/precision (berbasis reference) lebih objektif drpd faithfulness.
"""
from __future__ import annotations

# torch sebelum numpy (Windows/conda c10.dll); model embedding sudah ter-cache → offline.
import torch  # noqa: F401
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

OUT_DIR = Path("results/ragas")
TOP_K = 4


def classify(g: float) -> str:
    if g < 70:
        return "hipoglikemia"
    if g > 180:
        return "hiperglikemia"
    return "normal"


# Kasus divergen (subset T5) + pertanyaan klinis + reference (panduan antisipatif benar)
REF_HYPO = ("Antisipasi hipoglikemia: berikan 15 gram karbohidrat cepat (3-4 tablet glukosa "
            "atau 120 ml jus buah), tunggu 15 menit, cek ulang; ulangi bila masih <70 mg/dL "
            "(aturan 15-15). Pantau ketat dan siapkan sumber glukosa.")
REF_HYPER = ("Antisipasi hiperglikemia: perbanyak minum air putih, lakukan aktivitas fisik "
             "ringan, hindari tambahan karbohidrat, periksa keton bila >250 mg/dL, dan "
             "konsultasi dokter bila menetap >2 jam atau >300 mg/dL.")
QUESTION = ("Berikan penilaian risiko dan tindakan pencegahan yang tepat untuk pasien diabetes "
            "ini dalam 60 menit ke depan.")

ALL_CASES = [
    {"id": "D1", "current": 112.0, "predicted": 58.0, "reference": REF_HYPO},
    {"id": "D2", "current": 98.0, "predicted": 64.0, "reference": REF_HYPO},
    {"id": "D3", "current": 128.0, "predicted": 66.0, "reference": REF_HYPO},
    {"id": "D4", "current": 150.0, "predicted": 214.0, "reference": REF_HYPER},
    {"id": "D5", "current": 162.0, "predicted": 205.0, "reference": REF_HYPER},
]
# Free-tier = 20 req/HARI/model. Batasi ke subset agar judge tetap <=20/hari (satu judge konsisten).
# Default 3 kasus divergen (1 hipo + 2 hiper) x 2 mode x [faithfulness(2)+context_recall(1)] = 18 panggilan.
_wanted = [c.strip() for c in os.getenv("RAGAS_CASES", "D2,D4,D5").split(",") if c.strip()]
CASES = [c for c in ALL_CASES if c["id"] in _wanted] or ALL_CASES


def retrieval_query(case, mode: str) -> str:
    g = case["current"] if mode == "standard" else case["predicted"]
    cond = classify(g)
    horizon = "" if mode == "standard" else " (prediksi 60 menit ke depan)"
    return (f"Kadar glukosa darah {g:.0f} mg/dL{horizon}, kondisi {cond}. "
            f"Penilaian risiko, penanganan, dan pencegahan untuk kondisi {cond}.")


def generate_answer(llm, question: str, contexts: list[str]) -> str:
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Anda asisten klinis diabetes. Jawab pertanyaan HANYA berdasarkan KONTEKS berikut, "
        "ringkas dan actionable. Jangan mengarang di luar konteks.\n\n"
        f"KONTEKS:\n{ctx}\n\nPERTANYAAN: {question}\n\nJAWABAN:"
    )
    return llm.invoke(prompt).content


def main() -> int:
    load_dotenv(".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Event loop RAGAS + Windows: nest_asyncio agar loop bisa dipakai ulang.
    import nest_asyncio
    nest_asyncio.apply()

    # Pakai endpoint OpenAI-compatible Gemini (klien httpx, BUKAN gRPC) → hindari
    # "RuntimeError: Event loop is closed" yang bikin skor NaN di klien gRPC langchain-google-genai.
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from src.rag.retriever import MMRRetriever

    from ragas import EvaluationDataset, SingleTurnSample, evaluate, RunConfig
    from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    api_key = os.getenv("GOOGLE_API_KEY")
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"))
    gen_model = os.getenv("RAGAS_GEN_MODEL", "gemini-flash-latest")

    # Endpoint & kredensial dapat dioverride. Default = endpoint OpenAI-compatible Gemini (httpx,
    # bukan gRPC → aman di event loop RAGAS). Untuk judge LOKAL tanpa kuota, set:
    #   RAGAS_JUDGE_BASE_URL=http://localhost:11434/v1  RAGAS_JUDGE_API_KEY=ollama
    #   RAGAS_JUDGE_MODEL=llama3.1:8b  (Ollama). Sama untuk generator (RAGAS_GEN_*).
    gemini_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    judge_base = os.getenv("RAGAS_JUDGE_BASE_URL", gemini_base)
    judge_key = os.getenv("RAGAS_JUDGE_API_KEY", api_key)
    gen_base = os.getenv("RAGAS_GEN_BASE_URL", gemini_base)
    gen_key = os.getenv("RAGAS_GEN_API_KEY", api_key)

    # Rate limiter hanya perlu untuk remote free-tier (10/menit). Lokal → set RAGAS_RPS tinggi.
    from langchain_core.rate_limiters import InMemoryRateLimiter
    rps = float(os.getenv("RAGAS_RPS", "0.14"))
    rate_limiter = InMemoryRateLimiter(requests_per_second=rps, check_every_n_seconds=0.25,
                                       max_bucket_size=2)

    gen_llm = ChatOpenAI(model=gen_model, api_key=gen_key, base_url=gen_base, temperature=0.2,
                         max_tokens=1024, max_retries=5, rate_limiter=rate_limiter)
    judge_llm = ChatOpenAI(model=judge_model, api_key=judge_key, base_url=judge_base, temperature=0.0,
                           max_tokens=4096, max_retries=5, rate_limiter=rate_limiter)
    print(f"Generator: {gen_model} @ {gen_base}")
    print(f"Judge: {judge_model} @ {judge_base} | kasus: {[c['id'] for c in CASES]}")
    hf_emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"},
                                   encode_kwargs={"normalize_embeddings": True})

    retriever = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                             embed_provider="sentence-transformers")
    if not retriever.is_ready:
        print("MMRRetriever tidak siap — pastikan T7 (re-ingest) sudah dijalankan.")
        return 1

    print("Membangun sampel (retrieve + generate) untuk kedua mode...")
    rows_meta = []
    samples_by_mode = {"standard": [], "prediction_conditioned": []}
    for case in CASES:
        for mode in ("standard", "prediction_conditioned"):
            q = retrieval_query(case, mode)
            docs = retriever.retrieve(q, top_k=TOP_K)
            contexts = [d["text"] for d in docs]
            answer = generate_answer(gen_llm, QUESTION, contexts)
            samples_by_mode[mode].append(SingleTurnSample(
                user_input=QUESTION, retrieved_contexts=contexts,
                response=answer, reference=case["reference"],
            ))
            rows_meta.append({"case": case["id"], "mode": mode,
                              "predicted": case["predicted"], "cond": classify(case["predicted"]),
                              "n_ctx": len(contexts)})
            print(f"  {case['id']} [{mode}] konteks={len(contexts)} jawaban={len(answer)} char")

    judge = LangchainLLMWrapper(judge_llm)
    emb = LangchainEmbeddingsWrapper(hf_emb)
    # Free-tier budget + kompatibilitas Gemini:
    #   context_recall (1 panggilan/sampel, berbasis reference — paling objektif & terbukti ter-parse)
    #   answer_relevancy (1 panggilan + embedding lokal).
    #   faithfulness DIBUANG: output Gemini tak ter-parse oleh RAGAS (NaN) + boros panggilan (per-statement).
    #   context_precision DIBUANG: ~4 panggilan/sampel (boros kuota).
    metrics = [context_recall, answer_relevancy]
    run_cfg = RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=60)

    all_scores = []
    for mode, samples in samples_by_mode.items():
        print(f"\nMenilai RAGAS untuk mode: {mode} ({len(samples)} sampel)...")
        ds = EvaluationDataset(samples=samples)
        result = evaluate(dataset=ds, metrics=metrics, llm=judge, embeddings=emb, run_config=run_cfg)
        dfm = result.to_pandas()
        dfm.insert(0, "mode", mode)
        dfm.insert(1, "case", [c["id"] for c in CASES])
        all_scores.append(dfm)

    per_sample = pd.concat(all_scores, ignore_index=True)
    per_sample.to_csv(OUT_DIR / "per_sample.csv", index=False)

    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"] if c in per_sample.columns]
    summary = per_sample.groupby("mode")[metric_cols].mean().round(3).reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)

    (OUT_DIR / "caveat.md").write_text(
        "# Caveat LLM-as-judge (RAGAS)\n\n"
        "Skor RAGAS dihasilkan Gemini (LLM) yang menilai output LLM → berpotensi bias "
        "self-referential dan bervariasi antar-run. context_recall/precision (berbasis reference) "
        "lebih objektif daripada faithfulness/answer_relevancy. Angka bersifat indikatif.\n\n"
        f"Judge: {judge_model} | Generator: {gen_model} | n_kasus: {len(CASES)} | "
        f"kasus: {[c['id'] for c in CASES]} | top_k: {TOP_K}\n", encoding="utf-8")

    print("\n=== RINGKASAN RAGAS (rata-rata per mode) ===")
    print(summary.to_string(index=False))
    print(f"\nOutput -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
