"""
Ablation test untuk mengukur kontribusi BGE Cross-Encoder Reranker.

Perbandingan:
    A. Hybrid tanpa reranker
       BM25 + Vector/MMR -> RRF -> Top-K

    B. Hybrid + BGE reranker
       BM25 + Vector/MMR -> RRF -> Candidate Pool -> BGE -> Top-K

Metrik:
    - Hit@1
    - Hit@3
    - Hit@5
    - MRR

Eksperimen juga mencatat rank target sebelum dan sesudah reranking.

Jalankan dari ROOT project:
    python test_reranker_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

# ROOT PROJECT

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.rag.retriever import MMRRetriever


# CONFIG

PERSIST_DIR = "models/chroma_db_eval"
COLLECTION_NAME = "diabetes_kb_eval"

EMBED_PROVIDER = "sentence-transformers"
RETRIEVAL_MODE = "hibrida"

TOP_K = 5

# Candidate pool untuk BGE.
# Mengikuti konfigurasi eksperimen E02 sebelumnya.
CANDIDATE_K = 50

# File dataset evaluasi.
DATASET_PATH = ROOT / "evaluation" / "ragas_dataset.json"

# Target phrase E02.
# Dipakai hanya sebagai fallback/diagnostic.
E02_TARGET_PHRASE = "turunkan dosis 4 unit"


# DISPLAY


def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def short_text(text: str, length: int = 220) -> str:
    text = " ".join(str(text).split())
    if len(text) > length:
        return text[:length] + "..."
    return text


# TARGET MATCHING


def target_matches(doc: dict, target: dict) -> bool:
    """
    Menentukan apakah dokumen merupakan target/reference yang diharapkan.

    Prioritas:
    1. Jika target memiliki field 'target_text', cocokkan berdasarkan teks.
    2. Jika target memiliki 'target_source', cocokkan source.
    3. Jika target memiliki 'reference', coba cocokkan teks/reference.
    4. Untuk E02, gunakan phrase yang sudah diketahui.
    """

    text = str(doc.get("text", ""))
    source = str(doc.get("source", ""))

    target_text = str(target.get("target_text", "") or "")
    target_source = str(target.get("target_source", "") or "")
    reference = str(target.get("reference", "") or "")

    if target_text:
        # Exact substring.
        if target_text in text:
            return True

        # Normalisasi whitespace.
        norm_doc = " ".join(text.split()).lower()
        norm_target = " ".join(target_text.split()).lower()

        if norm_target and norm_target in norm_doc:
            return True

    if target_source and target_source == source:
        return True

    if reference:
        norm_doc = " ".join(text.split()).lower()
        norm_ref = " ".join(reference.split()).lower()

        if norm_ref and norm_ref in norm_doc:
            return True

    # E02 fallback.
    if E02_TARGET_PHRASE.lower() in text.lower():
        return True

    return False


# RETRIEVAL


def retrieve_hybrid_raw(
    retriever: MMRRetriever,
    query: str,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:
    """
    Menjalankan jalur hybrid sampai RRF, TANPA BGE.

    Ini sengaja tidak memakai retriever.retrieve() karena method tersebut
    dapat langsung melakukan reranking ketika reranker aktif.

    Pipeline:
        BM25
          +
        Vector/MMR
          ↓
        RRF
          ↓
        candidate pool
    """

    bm25_rank = retriever._peringkat_bm25(
        query,
        retriever.rrf_pool,
    )

    docs_v = retriever._vector_store.max_marginal_relevance_search(
        query,
        k=retriever.rrf_pool,
        fetch_k=max(
            retriever.rrf_pool,
            retriever.fetch_k,
        ),
        lambda_mult=retriever.lambda_mult,
        filter=None,
    )

    vector_rank = [
        retriever._bm25_peta[d.page_content]
        for d in docs_v
        if d.page_content in retriever._bm25_peta
    ]

    rrf_rank = retriever._gabung_rrf(
        [vector_rank, bm25_rank]
    )

    candidate_indices = rrf_rank[:candidate_k]

    candidates = retriever._hasil_dari_indeks(
        candidate_indices
    )

    return candidates


# RANK HELPERS


def find_target_rank(
    docs: list[dict],
    target: dict,
) -> int:
    """
    Return 1-based rank.

    Return 0 jika target tidak ditemukan.
    """

    for i, doc in enumerate(docs, start=1):
        if target_matches(doc, target):
            return i

    return 0


def reciprocal_rank(rank: int) -> float:
    if rank <= 0:
        return 0.0

    return 1.0 / rank


def hit_at_k(rank: int, k: int) -> int:
    return int(rank > 0 and rank <= k)


# DATASET


def load_cases() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan:\n{DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    # Format yang digunakan project saat ini:
    #
    # {
    #   "kasus": [...]
    # }
    if isinstance(data, dict) and "kasus" in data:
        cases = data["kasus"]

    elif isinstance(data, list):
        cases = data

    else:
        raise ValueError(
            "Format ragas_dataset.json tidak dikenali. "
            "Diharapkan object dengan key 'kasus' atau list."
        )

    return cases


# TARGET EXTRACTION


def get_expected_target(case: dict) -> dict:
    """
    Ambil informasi target/reference dari satu kasus.

    Dataset dapat memiliki struktur berbeda-beda. Fungsi ini dibuat
    defensif agar eksperimen tidak langsung gagal hanya karena nama
    field target berbeda.
    """

    target = {
        "target_text": "",
        "target_source": "",
        "reference": "",
    }

    # Kandidat field teks target.
    for key in (
        "target_text",
        "expected_text",
        "reference_text",
        "target_chunk",
        "relevant_text",
    ):
        value = case.get(key)
        if value:
            target["target_text"] = str(value)
            break

    # Kandidat source.
    for key in (
        "target_source",
        "reference_source",
        "source",
    ):
        value = case.get(key)
        if value:
            target["target_source"] = str(value)
            break

    # Kandidat reference.
    for key in (
        "reference",
        "ground_truth",
        "jawaban_referensi",
        "expected_answer",
    ):
        value = case.get(key)
        if value:
            target["reference"] = str(value)
            break

    return target


# QUERY EXTRACTION


def get_query(case: dict) -> str:
    """
    Ambil query dari dataset.
    """

    for key in (
        "pertanyaan",
        "query",
        "question",
        "user_question",
    ):
        value = case.get(key)

        if value:
            return str(value)

    raise ValueError(
        f"Kasus {case.get('id', '?')} tidak memiliki field pertanyaan/query."
    )


# SINGLE CASE


def evaluate_case(
    retriever: MMRRetriever,
    case: dict,
) -> dict:
    case_id = str(
        case.get(
            "id",
            case.get("case_id", "UNKNOWN"),
        )
    )

    query = get_query(case)

    target = get_expected_target(case)

    # Hybrid -> RRF -> candidate pool

    candidates = retrieve_hybrid_raw(
        retriever,
        query,
        candidate_k=CANDIDATE_K,
    )

    # Target rank sebelum reranking.
    raw_rank = find_target_rank(
        candidates,
        target,
    )

    # BGE reranking

    if retriever._reranker is None:
        raise RuntimeError(
            "BGE reranker belum tersedia. "
            "Pastikan reranker.enabled=true dan model berhasil diload."
        )

    reranked = retriever._rerank(
        query=query,
        candidates=candidates,
        top_k=TOP_K,
    )

    # Karena _rerank hanya mengembalikan TOP_K,
    # kita ukur apakah target berhasil masuk TOP_K.
    rerank_rank = find_target_rank(
        reranked,
        target,
    )

    # Metrics

    result = {
        "id": case_id,
        "query": query,
        "candidate_count": len(candidates),
        "raw_rank": raw_rank,
        "rerank_rank": rerank_rank,
        "raw_mrr": reciprocal_rank(raw_rank),
        "rerank_mrr": reciprocal_rank(rerank_rank),
        "raw_hit@1": hit_at_k(raw_rank, 1),
        "raw_hit@3": hit_at_k(raw_rank, 3),
        "raw_hit@5": hit_at_k(raw_rank, 5),
        "rerank_hit@1": hit_at_k(rerank_rank, 1),
        "rerank_hit@3": hit_at_k(rerank_rank, 3),
        "rerank_hit@5": hit_at_k(rerank_rank, 5),
        "improved": (
            raw_rank > 0
            and rerank_rank > 0
            and rerank_rank < raw_rank
        ),
        "target_found_candidate": raw_rank > 0,
        "target_found_topk": rerank_rank > 0,
    }

    # Simpan diagnostic TOP results

    result["raw_top5"] = [
        {
            "rank": i + 1,
            "source": d.get("source"),
            "text": short_text(d.get("text", "")),
        }
        for i, d in enumerate(candidates[:5])
    ]

    result["rerank_top5"] = [
        {
            "rank": i + 1,
            "score": d.get("reranker_score"),
            "source": d.get("source"),
            "text": short_text(d.get("text", "")),
        }
        for i, d in enumerate(reranked[:5])
    ]

    return result


# SUMMARY


def safe_mean(results: list[dict], key: str) -> float:
    values = [
        float(r[key])
        for r in results
        if key in r
    ]

    if not values:
        return 0.0

    return mean(values)


def safe_sd(results: list[dict], key: str) -> float:
    values = [
        float(r[key])
        for r in results
        if key in r
    ]

    if len(values) < 2:
        return 0.0

    return stdev(values)


def print_metric_comparison(
    results: list[dict],
    raw_key: str,
    rerank_key: str,
    label: str,
) -> None:
    raw = safe_mean(results, raw_key)
    rerank = safe_mean(results, rerank_key)

    delta = rerank - raw

    print(
        f"{label:<10} "
        f"Hybrid={raw:.3f}   "
        f"Hybrid+BGE={rerank:.3f}   "
        f"Δ={delta:+.3f}"
    )


def print_summary(results: list[dict]) -> None:
    print_header("SUMMARY")

    print(f"Jumlah kasus           : {len(results)}")

    found = sum(
        r["target_found_candidate"]
        for r in results
    )

    print(
        f"Target masuk candidate : "
        f"{found}/{len(results)} "
        f"({found / len(results):.1%})"
    )

    print()

    print("METRICS")
    print("-" * 70)

    print_metric_comparison(
        results,
        "raw_hit@1",
        "rerank_hit@1",
        "Hit@1",
    )

    print_metric_comparison(
        results,
        "raw_hit@3",
        "rerank_hit@3",
        "Hit@3",
    )

    print_metric_comparison(
        results,
        "raw_hit@5",
        "rerank_hit@5",
        "Hit@5",
    )

    print_metric_comparison(
        results,
        "raw_mrr",
        "rerank_mrr",
        "MRR",
    )

    print()

    print("RANKING")
    print("-" * 70)

    raw_ranks = [
        r["raw_rank"]
        for r in results
        if r["raw_rank"] > 0
    ]

    rerank_ranks = [
        r["rerank_rank"]
        for r in results
        if r["rerank_rank"] > 0
    ]

    if raw_ranks:
        print(
            f"Mean target rank sebelum BGE : "
            f"{mean(raw_ranks):.2f}"
        )

    if rerank_ranks:
        print(
            f"Mean target rank setelah BGE  : "
            f"{mean(rerank_ranks):.2f}"
        )

    improved = sum(
        r["improved"]
        for r in results
    )

    worsened = sum(
        1
        for r in results
        if (
            r["raw_rank"] > 0
            and r["rerank_rank"] > 0
            and r["rerank_rank"] > r["raw_rank"]
        )
    )

    unchanged = sum(
        1
        for r in results
        if (
            r["raw_rank"] > 0
            and r["rerank_rank"] > 0
            and r["rerank_rank"] == r["raw_rank"]
        )
    )

    print()
    print(f"Kasus rank membaik        : {improved}")
    print(f"Kasus rank memburuk       : {worsened}")
    print(f"Kasus rank tidak berubah  : {unchanged}")

    print()

    print("STANDARD DEVIATION")
    print("-" * 70)

    print(
        f"MRR Hybrid     : "
        f"{safe_mean(results, 'raw_mrr'):.3f} "
        f"+/- {safe_sd(results, 'raw_mrr'):.3f}"
    )

    print(
        f"MRR Hybrid+BGE : "
        f"{safe_mean(results, 'rerank_mrr'):.3f} "
        f"+/- {safe_sd(results, 'rerank_mrr'):.3f}"
    )


# CASE DETAILS


def print_case_table(results: list[dict]) -> None:
    print_header("PER-CASE RANK MOVEMENT")

    print(
        f"{'ID':<10}"
        f"{'RAW':>8}"
        f"{'BGE':>8}"
        f"{'Δ':>8}"
        f"{'STATUS':>14}"
    )

    print("-" * 55)

    for r in results:
        raw = r["raw_rank"]
        bge = r["rerank_rank"]

        if raw == 0:
            raw_display = "-"
        else:
            raw_display = str(raw)

        if bge == 0:
            bge_display = "-"
        else:
            bge_display = str(bge)

        if raw > 0 and bge > 0:
            delta = bge - raw

            if delta < 0:
                status = "IMPROVED"
            elif delta > 0:
                status = "WORSE"
            else:
                status = "UNCHANGED"

            delta_display = f"{delta:+d}"

        elif raw == 0 and bge > 0:
            status = "RECOVERED"
            delta_display = "NEW"

        elif raw > 0 and bge == 0:
            status = "LOST"
            delta_display = "LOST"

        else:
            status = "MISS"
            delta_display = "-"

        print(
            f"{r['id']:<10}"
            f"{raw_display:>8}"
            f"{bge_display:>8}"
            f"{delta_display:>8}"
            f"{status:>14}"
        )


# MAIN


def main() -> None:

    print_header(
        "BGE RERANKER ABLATION — HYBRID vs HYBRID + BGE"
    )

    print(f"Persist dir      : {PERSIST_DIR}")
    print(f"Collection       : {COLLECTION_NAME}")
    print(f"Retrieval mode   : {RETRIEVAL_MODE}")
    print(f"Top-K            : {TOP_K}")
    print(f"Candidate-K      : {CANDIDATE_K}")
    print(f"Dataset          : {DATASET_PATH}")

    # Load retriever

    print_header("INITIALIZING RETRIEVER")

    retriever = MMRRetriever(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embed_provider=EMBED_PROVIDER,
        retrieval_mode=RETRIEVAL_MODE,
    )

    print(
        f"Reranker enabled : "
        f"{retriever.reranker_enabled}"
    )

    print(
        f"Reranker model   : "
        f"{retriever.reranker_model}"
    )

    print(
        f"Candidate-K cfg  : "
        f"{retriever.reranker_candidate_k}"
    )

    print(
        f"Reranker loaded  : "
        f"{retriever._reranker is not None}"
    )

    if retriever._reranker_error:
        print(
            f"Reranker error   : "
            f"{retriever._reranker_error}"
        )

    if retriever._reranker is None:
        raise RuntimeError(
            "\nBGE reranker belum berhasil diload.\n"
            "Pastikan config.yaml memiliki:\n\n"
            "rag:\n"
            "  reranker:\n"
            "    enabled: true\n"
            "    model: BAAI/bge-reranker-v2-m3\n"
            "    candidate_k: 50\n"
        )

    # Load cases

    cases = load_cases()

    print()
    print(
        f"Kasus ditemukan : {len(cases)}"
    )

    # Evaluasi

    print_header("RUNNING ABLATION")

    results = []

    for i, case in enumerate(cases, start=1):

        case_id = str(
            case.get(
                "id",
                case.get("case_id", i),
            )
        )

        print(
            f"[{i}/{len(cases)}] "
            f"Evaluating {case_id} ...",
            end=" ",
            flush=True,
        )

        try:
            result = evaluate_case(
                retriever,
                case,
            )

            results.append(result)

            raw = result["raw_rank"]
            bge = result["rerank_rank"]

            print(
                f"raw={raw or '-'} "
                f"BGE={bge or '-'}"
            )

        except Exception as exc:
            print(
                f"ERROR: {type(exc).__name__}: {exc}"
            )

    if not results:
        raise RuntimeError(
            "Tidak ada kasus yang berhasil dievaluasi."
        )

    # Summary

    print_case_table(results)
    print_summary(results)

    # Save JSON

    output_dir = ROOT / "results"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir /
        "reranker_ablation.json"
    )

    output = {
        "experiment": (
            "Hybrid retrieval versus "
            "Hybrid retrieval + BGE reranker"
        ),
        "config": {
            "persist_dir": PERSIST_DIR,
            "collection_name": COLLECTION_NAME,
            "embed_provider": EMBED_PROVIDER,
            "retrieval_mode": RETRIEVAL_MODE,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "reranker_enabled": retriever.reranker_enabled,
            "reranker_model": retriever.reranker_model,
        },
        "description": {
            "baseline": (
                "BM25 + Vector/MMR -> RRF -> Top-K"
            ),
            "reranked": (
                "BM25 + Vector/MMR -> RRF -> "
                "Candidate Pool -> BGE -> Top-K"
            ),
        },
        "n_cases": len(results),
        "summary": {
            "hybrid": {
                "hit@1": safe_mean(
                    results,
                    "raw_hit@1",
                ),
                "hit@3": safe_mean(
                    results,
                    "raw_hit@3",
                ),
                "hit@5": safe_mean(
                    results,
                    "raw_hit@5",
                ),
                "mrr": safe_mean(
                    results,
                    "raw_mrr",
                ),
            },
            "hybrid_bge": {
                "hit@1": safe_mean(
                    results,
                    "rerank_hit@1",
                ),
                "hit@3": safe_mean(
                    results,
                    "rerank_hit@3",
                ),
                "hit@5": safe_mean(
                    results,
                    "rerank_hit@5",
                ),
                "mrr": safe_mean(
                    results,
                    "rerank_mrr",
                ),
            },
        },
        "cases": results,
    }

    output_path.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print(
        f"Hasil disimpan ke:\n{output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()