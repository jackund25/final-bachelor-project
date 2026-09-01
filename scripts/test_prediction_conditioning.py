"""
Ablation experiment:
Standard RAG vs Prediction-Conditioned RAG

Tujuan:
Menguji apakah penambahan konteks klinis hasil prediction/classification
ke query retrieval benar-benar meningkatkan retrieval.

Dua kondisi:

A. STANDARD
   pertanyaan asli
       -> Hybrid retrieval
       -> RRF
       -> BGE reranker

B. PREDICTION-CONDITIONED
   pertanyaan + konteks klinis hasil prediction/classification
       -> Hybrid retrieval
       -> RRF
       -> BGE reranker

Metrik:
- Hit@1
- Hit@3
- Hit@5
- MRR

Jalankan:
    python test_prediction_conditioning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean


# PROJECT ROOT

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


from src.rag.retriever import MMRRetriever


# CONFIG

PERSIST_DIR = "models/chroma_db_eval"
COLLECTION_NAME = "diabetes_kb_eval"

EMBED_PROVIDER = "sentence-transformers"
RETRIEVAL_MODE = "hibrida"

TOP_K = 5
CANDIDATE_K = 50

DATASET_PATH = (
    ROOT / "evaluation" / "ragas_dataset.json"
)


# DISPLAY

def header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def short_text(text: str, n: int = 180) -> str:
    text = " ".join(str(text).split())

    if len(text) > n:
        return text[:n] + "..."

    return text


# DATASET

def load_dataset() -> list[dict]:

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if isinstance(data, dict):

        if "kasus" in data:
            return data["kasus"]

        raise ValueError(
            "Dataset object tidak memiliki key 'kasus'."
        )

    if isinstance(data, list):
        return data

    raise ValueError(
        "Format dataset tidak dikenali."
    )


# QUERY

def get_question(case: dict) -> str:

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
        f"Pertanyaan tidak ditemukan pada kasus "
        f"{case.get('id', '?')}"
    )


# PREDICTION / CLASSIFICATION CONTEXT

def build_prediction_context(case: dict) -> str:
    """
    Mengambil konteks klinis yang memang berasal dari hasil
    prediction/classification kasus.

    Fungsi dibuat fleksibel karena struktur dataset dapat berbeda.
    """

    parts = []

    # Prediction value
    prediction = None

    for key in (
        "prediction",
        "prediksi",
        "predicted_glucose",
        "predicted_value",
    ):

        if case.get(key) is not None:

            prediction = case.get(key)
            break

    if prediction is not None:

        parts.append(
            f"Prediksi glukosa: {prediction} mg/dL"
        )

    # Risk / classification
    risk = None

    for key in (
        "risk_level",
        "risk",
        "status_risiko",
        "klasifikasi",
        "classification",
    ):

        if case.get(key):

            risk = case.get(key)
            break

    if risk:

        parts.append(
            f"Status klinis: {risk}"
        )

    # Context explicitly provided by dataset
    context = None

    for key in (
        "clinical_context",
        "konteks_klinis",
        "context",
        "konteks",
    ):

        if case.get(key):

            context = case.get(key)
            break

    if context:

        parts.append(
            f"Konteks klinis: {context}"
        )

    # Jika dataset tidak menyediakan field eksplisit,
    # kita tidak mengarang informasi.

    if not parts:

        return ""

    return " ".join(parts)


# TARGET

def target_matches(
    doc: dict,
    case: dict,
) -> bool:

    text = str(
        doc.get("text", "")
    ).lower()

    # Exact target text

    for key in (
        "target_text",
        "expected_text",
        "reference_text",
        "target_chunk",
        "relevant_text",
    ):

        value = case.get(key)

        if value:

            target = " ".join(
                str(value).split()
            ).lower()

            if target in " ".join(
                text.split()
            ):

                return True

    # Reference text

    for key in (
        "reference",
        "ground_truth",
        "jawaban_referensi",
        "expected_answer",
    ):

        value = case.get(key)

        if value:

            reference = " ".join(
                str(value).split()
            ).lower()

            # Hanya gunakan jika reference cukup spesifik.
            if len(reference) >= 30:

                # cek potongan awal reference
                fragment = reference[:100]

                if fragment in " ".join(
                    text.split()
                ):

                    return True

    return False


# RETRIEVAL

def hybrid_candidates(
    retriever: MMRRetriever,
    query: str,
) -> list[dict]:
    """
    Hybrid retrieval sampai candidate pool.

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

    docs_vector = (
        retriever
        ._vector_store
        .max_marginal_relevance_search(
            query,
            k=retriever.rrf_pool,
            fetch_k=max(
                retriever.rrf_pool,
                retriever.fetch_k,
            ),
            lambda_mult=retriever.lambda_mult,
            filter=None,
        )
    )

    vector_rank = [
        retriever._bm25_peta[d.page_content]
        for d in docs_vector
        if d.page_content in retriever._bm25_peta
    ]

    rrf_rank = retriever._gabung_rrf(
        [
            vector_rank,
            bm25_rank,
        ]
    )

    candidates = retriever._hasil_dari_indeks(
        rrf_rank[:CANDIDATE_K]
    )

    return candidates


# RERANK

def retrieve_final(
    retriever: MMRRetriever,
    query: str,
) -> list[dict]:

    candidates = hybrid_candidates(
        retriever,
        query,
    )

    if retriever._reranker is None:

        raise RuntimeError(
            "BGE reranker tidak tersedia."
        )

    return retriever._rerank(
        query=query,
        candidates=candidates,
        top_k=TOP_K,
    )


# RANK

def target_rank(
    docs: list[dict],
    case: dict,
) -> int:

    for rank, doc in enumerate(
        docs,
        start=1,
    ):

        if target_matches(
            doc,
            case,
        ):

            return rank

    return 0


def mrr(rank: int) -> float:

    if rank <= 0:
        return 0.0

    return 1.0 / rank


def hit(rank: int, k: int) -> int:

    return int(
        rank > 0
        and rank <= k
    )


# EVALUATE ONE CASE

def evaluate_case(
    retriever: MMRRetriever,
    case: dict,
) -> dict:

    case_id = str(
        case.get(
            "id",
            "UNKNOWN",
        )
    )

    question = get_question(case)

    # STANDARD

    standard_docs = retrieve_final(
        retriever,
        question,
    )

    standard_rank = target_rank(
        standard_docs,
        case,
    )

    # PREDICTION-CONDITIONED

    clinical_context = (
        build_prediction_context(case)
    )

    if clinical_context:

        conditioned_query = (
            question
            + " "
            + clinical_context
        )

    else:

        conditioned_query = question

    conditioned_docs = retrieve_final(
        retriever,
        conditioned_query,
    )

    conditioned_rank = target_rank(
        conditioned_docs,
        case,
    )

    return {
        "id": case_id,

        "question": question,

        "clinical_context": clinical_context,

        "standard_rank": standard_rank,

        "conditioned_rank": conditioned_rank,

        "standard_mrr": mrr(
            standard_rank
        ),

        "conditioned_mrr": mrr(
            conditioned_rank
        ),

        "standard_hit@1": hit(
            standard_rank,
            1,
        ),

        "standard_hit@3": hit(
            standard_rank,
            3,
        ),

        "standard_hit@5": hit(
            standard_rank,
            5,
        ),

        "conditioned_hit@1": hit(
            conditioned_rank,
            1,
        ),

        "conditioned_hit@3": hit(
            conditioned_rank,
            3,
        ),

        "conditioned_hit@5": hit(
            conditioned_rank,
            5,
        ),

        "standard_top5": [
            {
                "rank": i + 1,
                "score": d.get(
                    "reranker_score"
                ),
                "text": short_text(
                    d.get("text", "")
                ),
            }

            for i, d in enumerate(
                standard_docs[:5]
            )
        ],

        "conditioned_top5": [
            {
                "rank": i + 1,
                "score": d.get(
                    "reranker_score"
                ),
                "text": short_text(
                    d.get("text", "")
                ),
            }

            for i, d in enumerate(
                conditioned_docs[:5]
            )
        ],
    }


# SUMMARY

def avg(
    results: list[dict],
    key: str,
) -> float:

    values = [
        float(x[key])
        for x in results
    ]

    return (
        mean(values)
        if values
        else 0.0
    )


def print_comparison(
    results: list[dict],
    standard_key: str,
    conditioned_key: str,
    label: str,
) -> None:

    standard = avg(
        results,
        standard_key,
    )

    conditioned = avg(
        results,
        conditioned_key,
    )

    delta = conditioned - standard

    print(
        f"{label:<10}"
        f"Standard={standard:.3f}   "
        f"Conditioned={conditioned:.3f}   "
        f"Δ={delta:+.3f}"
    )


# MAIN

def main():

    header(
        "PREDICTION-CONDITIONED RETRIEVAL ABLATION"
    )

    print(
        "Standard:"
    )

    print(
        "Question → Hybrid → RRF → BGE"
    )

    print()

    print(
        "Prediction-Conditioned:"
    )

    print(
        "Question + Clinical Context → "
        "Hybrid → RRF → BGE"
    )

    print()

    # Retriever

    header(
        "INITIALIZING RETRIEVER"
    )

    retriever = MMRRetriever(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embed_provider=EMBED_PROVIDER,
        retrieval_mode=RETRIEVAL_MODE,
    )

    print(
        "Reranker enabled :",
        retriever.reranker_enabled,
    )

    print(
        "Reranker model   :",
        retriever.reranker_model,
    )

    print(
        "Reranker loaded  :",
        retriever._reranker is not None,
    )

    if retriever._reranker is None:

        raise RuntimeError(
            "BGE reranker belum tersedia."
        )

    # Dataset

    cases = load_dataset()

    print(
        "Jumlah kasus     :",
        len(cases),
    )

    # Evaluate

    header(
        "RUNNING EXPERIMENT"
    )

    results = []

    for i, case in enumerate(
        cases,
        start=1,
    ):

        case_id = str(
            case.get(
                "id",
                i,
            )
        )

        print(
            f"[{i}/{len(cases)}] "
            f"{case_id}",
            end=" ... ",
            flush=True,
        )

        try:

            result = evaluate_case(
                retriever,
                case,
            )

            results.append(
                result
            )

            print(
                f"Standard={result['standard_rank'] or '-'} "
                f"Conditioned={result['conditioned_rank'] or '-'}"
            )

        except Exception as exc:

            print(
                "ERROR:",
                type(exc).__name__,
                exc,
            )

    if not results:

        raise RuntimeError(
            "Tidak ada kasus yang berhasil."
        )

    # Per case

    header(
        "RANK MOVEMENT"
    )

    print(
        f"{'ID':<10}"
        f"{'STANDARD':>12}"
        f"{'CONDITIONED':>14}"
        f"{'DELTA':>10}"
        f"{'STATUS':>14}"
    )

    print("-" * 62)

    for r in results:

        s = r["standard_rank"]
        c = r["conditioned_rank"]

        s_display = (
            str(s)
            if s > 0
            else "-"
        )

        c_display = (
            str(c)
            if c > 0
            else "-"
        )

        if s > 0 and c > 0:

            delta = c - s

            if delta < 0:
                status = "IMPROVED"

            elif delta > 0:
                status = "WORSE"

            else:
                status = "UNCHANGED"

            delta_display = (
                f"{delta:+d}"
            )

        elif s == 0 and c > 0:

            status = "RECOVERED"
            delta_display = "NEW"

        elif s > 0 and c == 0:

            status = "LOST"
            delta_display = "LOST"

        else:

            status = "MISS"
            delta_display = "-"

        print(
            f"{r['id']:<10}"
            f"{s_display:>12}"
            f"{c_display:>14}"
            f"{delta_display:>10}"
            f"{status:>14}"
        )

    # Metrics

    header(
        "METRIC COMPARISON"
    )

    print_comparison(
        results,
        "standard_hit@1",
        "conditioned_hit@1",
        "Hit@1",
    )

    print_comparison(
        results,
        "standard_hit@3",
        "conditioned_hit@3",
        "Hit@3",
    )

    print_comparison(
        results,
        "standard_hit@5",
        "conditioned_hit@5",
        "Hit@5",
    )

    print_comparison(
        results,
        "standard_mrr",
        "conditioned_mrr",
        "MRR",
    )

    # Aggregate interpretation

    improved = sum(
        1
        for r in results
        if (
            r["conditioned_rank"] > 0
            and (
                r["standard_rank"] == 0
                or r["conditioned_rank"]
                < r["standard_rank"]
            )
        )
    )

    worsened = sum(
        1
        for r in results
        if (
            r["standard_rank"] > 0
            and r["conditioned_rank"] > 0
            and r["conditioned_rank"]
            > r["standard_rank"]
        )
    )

    unchanged = sum(
        1
        for r in results
        if (
            r["standard_rank"] > 0
            and r["conditioned_rank"] > 0
            and r["standard_rank"]
            == r["conditioned_rank"]
        )
    )

    print()

    print(
        "Rank improved/recovered :",
        improved,
    )

    print(
        "Rank worsened           :",
        worsened,
    )

    print(
        "Rank unchanged          :",
        unchanged,
    )

    # Save

    output_dir = (
        ROOT / "results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "prediction_conditioning_ablation.json"
    )

    output = {
        "experiment": (
            "Standard RAG vs "
            "Prediction-Conditioned RAG"
        ),

        "pipeline": {
            "standard": (
                "Question -> Hybrid -> RRF -> BGE"
            ),

            "prediction_conditioned": (
                "Question + clinical prediction/"
                "classification context -> "
                "Hybrid -> RRF -> BGE"
            ),
        },

        "config": {
            "persist_dir": PERSIST_DIR,
            "collection_name": COLLECTION_NAME,
            "retrieval_mode": RETRIEVAL_MODE,
            "top_k": TOP_K,
            "candidate_k": CANDIDATE_K,
            "reranker_model": (
                retriever.reranker_model
            ),
        },

        "summary": {
            "standard": {
                "hit@1": avg(
                    results,
                    "standard_hit@1",
                ),
                "hit@3": avg(
                    results,
                    "standard_hit@3",
                ),
                "hit@5": avg(
                    results,
                    "standard_hit@5",
                ),
                "mrr": avg(
                    results,
                    "standard_mrr",
                ),
            },

            "prediction_conditioned": {
                "hit@1": avg(
                    results,
                    "conditioned_hit@1",
                ),
                "hit@3": avg(
                    results,
                    "conditioned_hit@3",
                ),
                "hit@5": avg(
                    results,
                    "conditioned_hit@5",
                ),
                "mrr": avg(
                    results,
                    "conditioned_mrr",
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

    # Final

    header(
        "RESULT SAVED"
    )

    print(
        output_path
    )


if __name__ == "__main__":
    main()