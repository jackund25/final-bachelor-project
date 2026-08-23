from src.rag.retriever import MMRRetriever


PERSIST_DIR = "models/chroma_db_eval"
COLLECTION = "diabetes_kb_eval"
TARGET = "turunkan dosis 4 unit"

retriever = MMRRetriever(
    persist_dir=PERSIST_DIR,
    collection_name=COLLECTION,
    embed_provider="sentence-transformers",
    retrieval_mode="hibrida",
)

queries = {
    "A - Pertanyaan murni": (
        "Apa yang harus dilakukan terhadap dosis insulin basal "
        "bila pasien mengalami hipoglikemia?"
    ),

    "B - Query produksi": (
        "Prediksi glukosa 30 menit ke depan: 58.0 mg/dL "
        "(dari 58.0 mg/dL, perubahan +0.0 mg/dL, tren stabil). "
        "Status risiko prediksi: BAHAYA - Hipoglikemia. "
        "Apa yang harus dilakukan terhadap dosis insulin basal "
        "bila pasien mengalami hipoglikemia?"
    ),

    "C - Context pendek": (
        "Apa yang harus dilakukan terhadap dosis insulin basal "
        "bila pasien mengalami hipoglikemia? "
        "Konteks: glukosa 58 mg/dL, hipoglikemia."
    ),
}


def retrieve_and_rerank(query):
    bm25 = retriever._peringkat_bm25(query, 50)

    docs_v = retriever._vector_store.max_marginal_relevance_search(
        query,
        k=50,
        fetch_k=50,
        lambda_mult=retriever.lambda_mult,
    )

    vector_indices = [
        retriever._bm25_peta[d.page_content]
        for d in docs_v
        if d.page_content in retriever._bm25_peta
    ]

    rrf_indices = retriever._gabung_rrf(
        [vector_indices, bm25]
    )

    candidates = retriever._hasil_dari_indeks(
        rrf_indices[:50]
    )

    ranked = retriever._rerank(
        query=query,
        candidates=candidates,
        top_k=50,
    )

    return ranked


for name, query in queries.items():

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    ranked = retrieve_and_rerank(query)

    target_rank = None
    target_score = None

    for i, item in enumerate(ranked, start=1):
        if TARGET in item.get("text", ""):
            target_rank = i
            target_score = item.get("reranker_score")
            break

    print("TARGET RANK :", target_rank)
    print("BGE SCORE   :", target_score)

    print()
    print("TOP 5:")

    for i, item in enumerate(ranked[:5], start=1):
        is_target = TARGET in item.get("text", "")

        print(
            f"{i}. "
            f"BGE={item.get('reranker_score', 0):.6f} "
            f"TARGET={is_target}"
        )

        print(
            "   ",
            item.get("text", "")[:250]
            .replace("\n", " ")
        )