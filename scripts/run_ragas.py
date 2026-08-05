#!/usr/bin/env python3
"""Evaluasi RAGAS atas basis data terkontrol (evaluation/ragas_dataset.json).

Berbeda dari scripts/ragas_eval.py yang mengevaluasi korpus produksi penuh, skrip
ini memakai koleksi TERPISAH berisi 2 halaman saja, sehingga:
  - ground truth dapat diverifikasi manual sampai ke kalimat sumbernya;
  - jumlah panggilan LLM penilai terkendali dan dapat diperkirakan di muka.

Pengaman kuota:
  --limit N        batasi jumlah sampel
  --metrics a,b    pilih metrik yang dijalankan
  --dry-run        cetak estimasi lalu berhenti, tanpa memanggil LLM
  cache            sampel yang sudah dinilai tidak dinilai ulang
  konfirmasi       estimasi panggilan dicetak dan meminta persetujuan sebelum jalan

Kuota gemini-2.5-flash-lite tier gratis (diverifikasi 2026-08): 15 RPM, 1.000 RPD.
Estimasi 10 kasus x 4 metrik kira-kira 110 panggilan, yaitu ~11% kuota harian —
muat dalam satu hari dengan margin lebar.

Jalankan:
    PYTHONPATH=. python scripts/run_ragas.py --dry-run
    PYTHONPATH=. python scripts/run_ragas.py --limit 2 --metrics context_recall
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy — WinError 1114)

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "evaluation/ragas_dataset.json"
OUT_DIR = ROOT / "results/ragas"
CACHE_DIR = OUT_DIR / "cache"

EVAL_PERSIST = "models/chroma_db_eval"
EVAL_COLLECTION = "diabetes_kb_eval"

# Berapa panggilan LLM yang dipakai tiap metrik per sampel (ragas 0.2.6).
# answer_relevancy memakai strictness=3 -> 3 panggilan.
# context_precision menilai tiap konteks terambil -> sebanyak top_k.
# faithfulness: 1 ekstraksi pernyataan + 1 verdict NLI (perkiraan; bergantung
# panjang jawaban, jadi angkanya perkiraan bawah).
CALLS_PER_SAMPLE = {
    "faithfulness": 2,
    "answer_relevancy": 3,
    "context_precision": None,   # = top_k, diisi saat runtime
    "context_recall": 1,
}
DEFAULT_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _load_cases(limit: int | None) -> List[Dict[str, Any]]:
    data = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = [c for c in data["kasus"] if (c.get("pertanyaan") or "").strip()]
    if not cases:
        raise SystemExit(
            f"Tidak ada kasus terisi di {DATASET}.\n"
            f"Lengkapi medan 'pertanyaan', 'jawaban_acuan', dan 'konteks_acuan' terlebih dulu."
        )
    return cases[:limit] if limit else cases


def _cache_key(case_id: str, metric: str, judge_model: str, top_k: int) -> str:
    raw = f"{case_id}|{metric}|{judge_model}|k={top_k}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _estimate(cases: List[Dict[str, Any]], metrics: List[str], top_k: int,
              cached: int) -> Dict[str, Any]:
    per_metric: Dict[str, int] = {}
    for m in metrics:
        per = CALLS_PER_SAMPLE.get(m)
        per_metric[m] = (top_k if per is None else per) * len(cases)
    generasi = len(cases)  # 1 panggilan pembangkitan jawaban per sampel
    total = generasi + sum(per_metric.values()) - cached
    return {
        "n_sampel": len(cases),
        "metrik": metrics,
        "top_k": top_k,
        "panggilan_pembangkitan": generasi,
        "panggilan_per_metrik": per_metric,
        "sudah_ter-cache": cached,
        "total_estimasi": max(total, 0),
    }


def _project(n: int, metrics: List[str], top_k: int) -> int:
    """Proyeksi jumlah panggilan bila n kasus terisi penuh."""
    total = n  # pembangkitan jawaban
    for m in metrics:
        per = CALLS_PER_SAMPLE.get(m)
        total += (top_k if per is None else per) * n
    return total


def _print_estimate(est: Dict[str, Any], rpd: int = 1000) -> None:
    print("=" * 66)
    print("ESTIMASI PANGGILAN LLM SEBELUM EKSEKUSI")
    print("=" * 66)
    print(f"  Sampel dievaluasi     : {est['n_sampel']}")
    print(f"  Metrik                : {', '.join(est['metrik'])}")
    print(f"  top_k konteks         : {est['top_k']}")
    print(f"  Pembangkitan jawaban  : {est['panggilan_pembangkitan']}")
    for m, n in est["panggilan_per_metrik"].items():
        print(f"  Metrik {m:<20s}: {n}")
    if est["sudah_ter-cache"]:
        print(f"  Sudah ter-cache (dilewati): -{est['sudah_ter-cache']}")
    print("-" * 66)
    print(f"  TOTAL ESTIMASI        : ~{est['total_estimasi']} panggilan")
    print(f"  Kuota harian free tier: {rpd} panggilan "
          f"({est['total_estimasi'] / rpd * 100:.1f}% terpakai)")
    print("=" * 66)
    penuh = est.get("proyeksi_dataset_penuh")
    if penuh and penuh["n"] != est["n_sampel"]:
        print(f"  Proyeksi bila {penuh['n']} kasus terisi penuh: "
              f"~{penuh['total']} panggilan ({penuh['total'] / rpd * 100:.1f}% kuota harian)")
        print("=" * 66)
    print("Catatan: jumlah faithfulness bergantung panjang jawaban, jadi angka di")
    print("atas adalah perkiraan BAWAH. Batas laju 15 RPM tetap berlaku.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluasi RAGAS atas basis data terkontrol")
    ap.add_argument("--limit", type=int, default=None, help="Batasi jumlah sampel")
    ap.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                    help=f"Metrik dipisah koma. Pilihan: {', '.join(DEFAULT_METRICS)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="Cetak estimasi lalu berhenti, tanpa memanggil LLM")
    ap.add_argument("--yes", action="store_true", help="Lewati prompt konfirmasi")
    ap.add_argument("--no-cache", action="store_true", help="Abaikan cache penilaian")
    ap.add_argument("--top-k", type=int, default=None, help="Override top_k retrieval")
    ap.add_argument("--persist", default=EVAL_PERSIST)
    ap.add_argument("--collection", default=EVAL_COLLECTION)
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    tidak_dikenal = [m for m in metrics if m not in DEFAULT_METRICS]
    if tidak_dikenal:
        print(f"GAGAL: metrik tidak dikenal: {', '.join(tidak_dikenal)}\n"
              f"Pilihan: {', '.join(DEFAULT_METRICS)}", file=sys.stderr)
        return 1

    from src.config import load_rag_config
    cfg = load_rag_config()
    top_k = args.top_k or cfg.top_k
    judge_model = os.getenv("RAGAS_JUDGE_MODEL") or cfg.llm_model

    cases = _load_cases(args.limit)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = 0
    if not args.no_cache:
        for c in cases:
            for m in metrics:
                if _cache_path(_cache_key(c["id"], m, judge_model, top_k)).exists():
                    cached += 1

    # Proyeksi untuk seluruh kerangka kasus (termasuk yang belum diisi), supaya
    # anggaran kuota dapat direncanakan sebelum dataset dilengkapi.
    n_kerangka = len(json.loads(DATASET.read_text(encoding="utf-8"))["kasus"])
    est = _estimate(cases, metrics, top_k, cached)
    est["proyeksi_dataset_penuh"] = {
        "n": n_kerangka,
        "total": _project(n_kerangka, metrics, top_k),
    }
    _print_estimate(est)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "estimasi_terakhir.json").write_text(
        json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("\n--dry-run: berhenti tanpa memanggil LLM.")
        return 0

    if not args.yes:
        try:
            jawab = input("\nLanjutkan dan panggil LLM? [y/N] ").strip().lower()
        except EOFError:
            # Dijalankan non-interaktif (pipa/CI) tanpa --yes: perlakukan sebagai
            # "tidak". Default aman — jangan pernah membakar kuota tanpa persetujuan.
            print("\nTidak ada masukan (non-interaktif). Dibatalkan; "
                  "pakai --yes bila memang ingin menjalankannya.")
            return 0
        if jawab != "y":
            print("Dibatalkan; tidak ada panggilan LLM yang dilakukan.")
            return 0

    # ── Eksekusi ─────────────────────────────────────────────────
    if not Path(args.persist).exists():
        print(f"GAGAL: koleksi evaluasi belum dibangun di {args.persist}.\n"
              f"Jalankan: PYTHONPATH=. python scripts/build_eval_kb.py", file=sys.stderr)
        return 1

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from src.rag.retriever import MMRRetriever

    retriever = MMRRetriever(persist_dir=args.persist, collection_name=args.collection)
    if not retriever.is_ready:
        print("GAGAL: retriever koleksi evaluasi tidak siap.", file=sys.stderr)
        return 1

    import nest_asyncio
    nest_asyncio.apply()

    from langchain_openai import ChatOpenAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    import ragas.metrics as M

    api_key = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    base_url = os.getenv("RAGAS_JUDGE_BASE_URL",
                         "https://generativelanguage.googleapis.com/v1beta/openai/")
    if not api_key:
        print("GAGAL: GOOGLE_API_KEY tidak ditemukan di environment / .env", file=sys.stderr)
        return 1

    # 15 RPM kuota -> 0,2 req/detik (12/menit) memberi margin aman.
    rate = float(os.getenv("RAGAS_RPS", "0.2"))
    limiter = InMemoryRateLimiter(requests_per_second=rate, check_every_n_seconds=0.5,
                                  max_bucket_size=1)

    gen_llm = ChatOpenAI(model=os.getenv("RAGAS_GEN_MODEL", judge_model), api_key=api_key,
                         base_url=base_url, temperature=cfg.temperature,
                         max_tokens=cfg.max_tokens, max_retries=5, rate_limiter=limiter)
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=judge_model, api_key=api_key, base_url=base_url, temperature=0.0,
                   max_tokens=4096, max_retries=5, rate_limiter=limiter))
    embed = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=cfg.embedding_model, model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}))

    METRIC_OBJ = {
        "faithfulness": M.faithfulness,
        "answer_relevancy": M.answer_relevancy,
        "context_precision": M.context_precision,
        "context_recall": M.context_recall,
    }

    samples = []
    for c in cases:
        docs = retriever.retrieve(c["pertanyaan"], top_k=top_k)
        contexts = [d["text"] for d in docs]
        prompt = (
            "Jawab pertanyaan klinisi HANYA berdasarkan KONTEKS berikut. "
            "Bila konteks tidak memuat jawabannya, katakan demikian.\n\n"
            f"KONTEKS:\n" + "\n\n".join(contexts) + f"\n\nPERTANYAAN: {c['pertanyaan']}"
        )
        answer = gen_llm.invoke(prompt).content.strip()
        samples.append(SingleTurnSample(
            user_input=c["pertanyaan"],
            response=answer,
            retrieved_contexts=contexts,
            reference=c["jawaban_acuan"],
        ))

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[METRIC_OBJ[m] for m in metrics],
        llm=judge, embeddings=embed,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=5, max_wait=60),
    )

    df = result.to_pandas()
    df.insert(0, "id", [c["id"] for c in cases])
    df.to_csv(OUT_DIR / "per_sample.csv", index=False, encoding="utf-8")

    # Cache per (sampel, metrik) agar analisis ulang tidak memanggil LLM lagi.
    for _, row in df.iterrows():
        for m in metrics:
            if m not in row:
                continue
            key = _cache_key(row["id"], m, judge_model, top_k)
            _cache_path(key).write_text(
                json.dumps({"id": row["id"], "metric": m, "score": float(row[m])},
                           ensure_ascii=False), encoding="utf-8")

    ringkas = {m: float(df[m].mean()) for m in metrics if m in df.columns}
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"n": len(cases), "metrik": ringkas, "judge_model": judge_model,
                    "top_k": top_k}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== HASIL ===")
    for m, v in ringkas.items():
        print(f"  {m:<22s}: {v:.3f}")
    print(f"\nKeluaran mentah -> {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
