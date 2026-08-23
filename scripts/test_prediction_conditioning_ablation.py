from dotenv import load_dotenv
load_dotenv()
from src.rag.retriever import MMRRetriever

PERSIST_DIR = "models/chroma_db_eval"
COLLECTION = "diabetes_kb_eval"

CASES = [
    ("E01", "Berapa dosis awal insulin basal pada terapi insulin rawat jalan?", "Konteks klinis hasil klasifikasi: terapi insulin basal rawat jalan.", "10 unit"),
    ("E02", "Apa yang harus dilakukan terhadap dosis insulin basal bila pasien mengalami hipoglikemia?", "Prediksi glukosa: 58 mg/dL. Status klinis: BAHAYA - Hipoglikemia.", "turunkan dosis 4 unit"),
    ("E03", "Berapa ambang kadar glukosa darah untuk hipoglikemia Level 1 dan Level 2?", "Status klinis hasil klasifikasi: BAHAYA - Hipoglikemia.", "70 mg/dL"),
    ("E04", "Apa yang dimaksud dengan hipoglikemia berat atau Level 3?", "Status klinis hasil klasifikasi: BAHAYA - Hipoglikemia berat.", "membutuhkan bantuan orang lain"),
    ("E05", "Berapa laju infus insulin intravena pada krisis hiperglikemia dan berapa penurunan glukosa darah yang diharapkan?", "Status klinis hasil klasifikasi: BAHAYA - Hiperglikemia.", "5"),
    ("E06", "Apa yang harus dilakukan terhadap dosis infus insulin bila kadar glukosa darah sudah turun di bawah 250 mg/dL?", "Prediksi glukosa: 240 mg/dL. Status klinis: PERHATIAN - Glukosa masih tinggi tetapi menurun.", "50% dari dosis sebelumnya"),
    ("E07", "Berapa dosis awal insulin intramuskular pada krisis hiperglikemia bila drip insulin tidak tersedia?", "Status klinis hasil klasifikasi: BAHAYA - Krisis hiperglikemia.", "0,1 unit/kgBB"),
    ("E08", "Bagaimana menyesuaikan kecepatan drip insulin bila glukosa darah hanya menurun 0 sampai 49 mg/dL per jam?", "Status klinis hasil klasifikasi: BAHAYA - Hiperglikemia dengan respons insulin tidak adekuat.", "naikkan drip insulin 25"),
    ("E09", "Mengapa pemberian insulin subkutan pada krisis hiperglikemia lebih berisiko dibandingkan intramuskular?", "Status klinis hasil klasifikasi: BAHAYA - Krisis hiperglikemia.", "subkutan"),
    ("E10", "Bila insulin basal diganti menjadi insulin premixed dua kali sehari, bagaimana pembagian dosisnya?", "Konteks klinis hasil klasifikasi: terapi insulin basal akan dialihkan ke insulin premixed dua kali sehari.", "2/3"),
]

def doc_rank(r, indices, target):
    for i, idx in enumerate(indices, 1):
        text = r._hasil_dari_indeks([idx])[0].get("text", "")
        if target.lower() in text.lower():
            return i
    return None

def evaluate(r, query, target):
    bm25 = r._peringkat_bm25(query, r.rrf_pool)
    docs_v = r._vector_store.max_marginal_relevance_search(
        query, k=r.rrf_pool, fetch_k=max(r.rrf_pool, r.fetch_k), lambda_mult=r.lambda_mult
    )
    vector = [r._bm25_peta[d.page_content] for d in docs_v if d.page_content in r._bm25_peta]
    rrf = r._gabung_rrf([vector, bm25])
    candidates = r._hasil_dari_indeks(rrf[:50])
    ranked = r._rerank(query=query, candidates=candidates, top_k=len(candidates))
    target_rank = next((i+1 for i,x in enumerate(ranked) if target.lower() in x.get("text", "").lower()), None)
    score = ranked[target_rank-1].get("reranker_score") if target_rank else None
    return doc_rank(r,bm25,target), doc_rank(r,vector,target), target_rank, score

def main():
    print("="*78)
    print("PREDICTION-CONDITIONING ABLATION")
    print("STANDARD QUERY vs PREDICTION-CONDITIONED QUERY")
    print("="*78)
    r = MMRRetriever(persist_dir=PERSIST_DIR, collection_name=COLLECTION, embed_provider="sentence-transformers", retrieval_mode="hibrida")
    results=[]
    for cid, question, context, target in CASES:
        std = evaluate(r, question, target)
        con = evaluate(r, question + " " + context, target)
        results.append((cid,std,con))
        print(f"\n{cid} | {question}\nCONTEXT: {context}\nTARGET: {target}")
        print(f"STANDARD             BM25={std[0]} VECTOR={std[1]} BGE={std[2]} SCORE={std[3]}")
        print(f"PREDICTION-CONDITION BM25={con[0]} VECTOR={con[1]} BGE={con[2]} SCORE={con[3]}")
        if std[2] and con[2]: print(f"DELTA BGE RANK = {std[2]-con[2]} (positif = membaik)")
    print("\n"+"="*78+"\nSUMMARY\n"+"="*78)
    def hit(vals,k): return sum(v is not None and v<=k for v in vals)/len(vals)
    def mrr(vals): return sum(1/v if v else 0 for v in vals)/len(vals)
    sr=[x[1][2] for x in results]; cr=[x[2][2] for x in results]
    print(f"Standard         Hit@1={hit(sr,1):.3f} Hit@3={hit(sr,3):.3f} Hit@5={hit(sr,5):.3f} MRR={mrr(sr):.3f}")
    print(f"Prediction-cond. Hit@1={hit(cr,1):.3f} Hit@3={hit(cr,3):.3f} Hit@5={hit(cr,5):.3f} MRR={mrr(cr):.3f}")
    print("\nConditioning hanya menggunakan output prediksi/klasifikasi klinis; patient_state tidak dimasukkan.")

if __name__ == "__main__": main()
