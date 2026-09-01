from src.rag.retriever import MMRRetriever


PERSIST_DIR = "models/chroma_db_eval"
COLLECTION_NAME = "diabetes_kb_eval"

QUERY = (
    "Apa yang harus dilakukan terhadap dosis insulin basal "
    "bila pasien mengalami hipoglikemia? "
    "Konteks klinis: hipoglikemia."
)

TARGET = "turunkan dosis 4 unit"


def find_rank(items, target_idx):
    try:
        return items.index(target_idx) + 1
    except ValueError:
        return None


def main():
    print("=" * 70)
    print("E02 — RETRIEVAL / RRF DIAGNOSTIC")
    print("=" * 70)

    r = MMRRetriever(
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        embed_provider="sentence-transformers",
        retrieval_mode="hibrida",
    )

    # Cari target chunk di seluruh corpus
    target_idx = None

    for idx, text in enumerate(r._bm25_teks):
        if TARGET in text.lower():
            target_idx = idx
            break

    if target_idx is None:
        print("\nTARGET TIDAK DITEMUKAN DI CORPUS")
        return

    print("\nTARGET CHUNK")
    print("-" * 70)
    print("Index :", target_idx)
    print("Target:", TARGET)

    # BM25
    bm25_indices = r._peringkat_bm25(QUERY, 50)
    bm25_rank = find_rank(bm25_indices, target_idx)

    print("\nBM25")
    print("-" * 70)
    print("Target rank:", bm25_rank)

    # Vector / MMR
    docs_vector = r._vector_store.max_marginal_relevance_search(
        QUERY,
        k=50,
        fetch_k=50,
        lambda_mult=r.lambda_mult,
    )

    vector_indices = [
        r._bm25_peta[d.page_content]
        for d in docs_vector
        if d.page_content in r._bm25_peta
    ]

    vector_rank = find_rank(vector_indices, target_idx)

    print("\nVECTOR / MMR")
    print("-" * 70)
    print("Jumlah hasil :", len(vector_indices))
    print("Target rank  :", vector_rank)

    # RRF
    rrf_indices = r._gabung_rrf(
        [
            vector_indices,
            bm25_indices,
        ]
    )

    rrf_rank = find_rank(rrf_indices, target_idx)

    print("\nRRF")
    print("-" * 70)
    print("Jumlah kandidat:", len(rrf_indices))
    print("Target rank    :", rrf_rank)

    # TOP 20 RRF
    print("\nTOP 20 RRF")
    print("-" * 70)

    for rank, idx in enumerate(rrf_indices[:20], start=1):
        text = r._bm25_teks[idx]

        bm_rank = find_rank(bm25_indices, idx)
        vec_rank = find_rank(vector_indices, idx)

        marker = " <<< TARGET" if idx == target_idx else ""

        print(
            f"{rank:2d}. "
            f"BM25={str(bm_rank):>3} "
            f"VECTOR={str(vec_rank):>3}"
            f"{marker}"
        )

        print(
            "    ",
            text[:220].replace("\n", " "),
        )

    # Candidate pool yang benar-benar dikirim ke BGE
    candidate_k = max(
        5,
        r.reranker_candidate_k,
    )

    candidate_indices = rrf_indices[:candidate_k]

    candidate_rank = find_rank(
        candidate_indices,
        target_idx,
    )

    print("\nCANDIDATE POOL")
    print("-" * 70)
    print("candidate_k :", candidate_k)
    print("Target masuk:", candidate_rank is not None)

    if candidate_rank is not None:
        print("Target rank :", candidate_rank)

    # BGE Reranker
    if r.reranker_enabled and r._reranker is not None:

        candidates = r._hasil_dari_indeks(
            candidate_indices
        )

        reranked = r._rerank(
            query=QUERY,
            candidates=candidates,
            top_k=len(candidates),
        )

        bge_target_rank = None
        bge_target_score = None

        for rank, row in enumerate(reranked, start=1):
            if TARGET in row.get("text", "").lower():
                bge_target_rank = rank
                bge_target_score = row.get(
                    "reranker_score"
                )
                break

        print("\nBGE RERANKER")
        print("-" * 70)
        print("Target rank :", bge_target_rank)
        print("Target score:", bge_target_score)

        print("\nTOP 10 BGE")
        print("-" * 70)

        for rank, row in enumerate(
            reranked[:10],
            start=1,
        ):
            is_target = TARGET in row.get(
                "text", ""
            ).lower()

            marker = " <<< TARGET" if is_target else ""

            print(
                f"{rank:2d}. "
                f"BGE={row.get('reranker_score', 0):.6f}"
                f"{marker}"
            )

            print(
                "    ",
                row.get(
                    "text",
                    "",
                )[:220].replace("\n", " "),
            )

    else:
        bge_target_rank = None

        print("\nBGE RERANKER")
        print("-" * 70)
        print("BGE tidak tersedia.")

    # SUMMARY
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("BM25 target rank     :", bm25_rank)
    print("Vector target rank   :", vector_rank)
    print("RRF target rank      :", rrf_rank)
    print("Candidate pool rank  :", candidate_rank)
    print("BGE target rank      :", bge_target_rank)

    print("\nINTERPRETASI")

    if bm25_rank is None:
        print(
            "[!] Target tidak ditemukan oleh BM25."
        )

    elif candidate_rank is None:
        print(
            "[!] Target ditemukan BM25 tetapi "
            "tidak masuk candidate pool RRF."
        )

    elif bge_target_rank is not None:
        if bge_target_rank <= 5:
            print(
                "[OK] Target berhasil masuk TOP-5 "
                "setelah reranking."
            )
        else:
            print(
                "[!] Target sudah masuk candidate pool "
                "tetapi BGE belum mengangkatnya ke TOP-5."
            )

    else:
        print(
            "[!] Target tidak ditemukan setelah "
            "candidate generation."
        )


if __name__ == "__main__":
    main()