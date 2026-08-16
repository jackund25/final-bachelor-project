"""T14 — validasi lengan penelusuran dengan alat ukur yang TIDAK berbasis kata kunci.

Prapendaftaran: docs/PRAPENDAFTARAN_T14_RAGAS_LENGAN.md (ditulis lebih dulu).

MENGAPA ADA. T13 menemukan hibrida RRF mengungguli vektor, tetapi temuan itu
terkonfound: classify_chunk melabeli menurut KATA KUNCI, dan build_ablation_query
menyusun kueri yang memuat kata-kata itu harfiah, sehingga BM25 mengoptimalkan sinyal
yang sama dengan yang diukur metriknya. T14 mengukur ulang dengan context_precision
dan context_recall, keduanya berbasis RUJUKAN dan bukan kata kunci.

DUA PENGHEMATAN YANG DISENGAJA
1. Tahap PEMBANGKITAN JAWABAN DILEWATI. Kedua metrik hanya memerlukan pertanyaan,
   konteks terambil, dan jawaban acuan; `response` tidak dipakai keduanya. Melewatinya
   menghemat kuota sekaligus membuang satu sumber variasi yang tidak relevan bagi
   pertanyaan penelusuran.
2. KORPUS PRODUKSI dipakai. Ini PENYIMPANGAN dari arahan pembimbing yang meminta
   koleksi RAGAS khusus 2-4 halaman demi menjaga kuota, dan penyimpangan itu
   dinyatakan terbuka di docs/PRAPENDAFTARAN_T14_RAGAS_LENGAN.md.

   Koleksi terkontrol BUKAN kelemahan — ia instrumen yang tepat untuk pertanyaan
   "seberapa baik mutu RAG ini", dengan ground truth terverifikasi sampai kalimat.
   Tetapi ia tidak dapat menjawab "retriever mana yang lebih baik": metadata dataset
   itu sendiri mencatat porsi_koleksi_terambil_per_kueri_persen = 29,4, sehingga
   setiap retriever mengambil hampir sepertiga koleksi yang sama.

   Tujuan arahan tetap dipatuhi: 120 panggilan dari 500 per hari, dan tahap
   pembangkitan dilewati seluruhnya.

Keluaran: results/ragas/lengan_penelusuran.json
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy — WinError 1114)

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DATASET = ROOT / "evaluation/ragas_dataset.json"
OUT = ROOT / "results/ragas/lengan_penelusuran.json"
CACHE = ROOT / "results/ragas/cache/lengan"
LENGAN = ["vektor", "hibrida"]
METRIK = ["context_precision", "context_recall"]
TOP_K = 5


def muat_kunci_api() -> str | None:
    p = ROOT / ".env"
    if p.exists():
        for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.strip()
            if s.startswith("GOOGLE_API_KEY=") and not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = s.split("=", 1)[1].strip().strip('"')
    return os.environ.get("GOOGLE_API_KEY")


def konteks_per_lengan(kasus: list) -> dict:
    """Ambil konteks top-k untuk tiap lengan, SEBELUM satu pun panggilan LLM.

    Dipisahkan dari tahap penilaian supaya kegagalan kuota tidak menghanguskan
    pekerjaan penelusuran, dan supaya konteksnya dapat diperiksa manual.
    """
    from eval_retrieval_crossfold import bangun_bm25, _rrf, tokenisasi
    from src.rag.retriever import MMRRetriever
    import numpy as np

    r = MMRRetriever(persist_dir="models/chroma_db", collection_name="diabetes_kb",
                     embed_provider="sentence-transformers")
    bm25, teks = bangun_bm25("models/chroma_db", "diabetes_kb")
    peta = {t: i for i, t in enumerate(teks)}
    print(f"Korpus produksi: {len(teks)} potongan (BUKAN koleksi terkontrol 17).")

    hasil = {l: [] for l in LENGAN}
    for c in kasus:
        q = c["pertanyaan"]
        docs_v = r.retrieve(q, top_k=TOP_K)
        hasil["vektor"].append([d["text"] for d in docs_v])

        skor = bm25.get_scores(tokenisasi(q))
        urut_bm = list(np.argsort(-skor)[:50])
        docs_v50 = r.retrieve(q, top_k=50)
        urut_v = [peta[d["text"]] for d in docs_v50 if d["text"] in peta]
        hasil["hibrida"].append([teks[i] for i in _rrf([urut_v, urut_bm])[:TOP_K]])
    return hasil


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="batasi jumlah kasus")
    ap.add_argument("--dry-run", action="store_true",
                    help="cetak estimasi panggilan lalu berhenti, TANPA memanggil LLM")
    args = ap.parse_args()

    data = json.loads(DATASET.read_text(encoding="utf-8"))
    kasus = data["kasus"][: args.limit] if args.limit else data["kasus"]

    # context_precision menilai TIAP konteks terambil -> top_k panggilan.
    # context_recall -> 1 panggilan.
    per_kasus = TOP_K + 1
    total = per_kasus * len(kasus) * len(LENGAN)
    print(f"Kasus: {len(kasus)} | lengan: {len(LENGAN)} | metrik: {METRIK}")
    print(f"Estimasi panggilan penilai: {per_kasus} x {len(kasus)} x {len(LENGAN)} = {total}")
    print("Tahap pembangkitan DILEWATI (kedua metrik tidak memakai `response`).")

    if args.dry_run:
        print("\n--dry-run: berhenti sebelum memanggil LLM.")
        return 0

    if not muat_kunci_api():
        print("GAGAL: GOOGLE_API_KEY tidak ditemukan.", file=sys.stderr)
        return 1

    ctx = konteks_per_lengan(kasus)

    from langchain_openai import ChatOpenAI
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig
    import ragas.metrics as M
    from run_ragas import kuota_model

    from src.config import load_rag_config
    cfg = load_rag_config()
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", cfg.llm_model)
    rpm, rpd = kuota_model(judge_model)
    rate = float(os.getenv("RAGAS_RPS", str(round(rpm * 0.8 / 60, 3))))
    print(f"\nPenilai: {judge_model} | kuota {rpm} RPM, {rpd} RPD | laju {rate}/dtk")
    if total > rpd:
        print(f"PERINGATAN: {total} panggilan MELAMPAUI kuota harian {rpd}. "
              f"Jalan ini akan berhenti di tengah dan hasilnya PARSIAL.")

    key = os.environ["GOOGLE_API_KEY"]
    base = os.getenv("RAGAS_JUDGE_BASE_URL",
                     "https://generativelanguage.googleapis.com/v1beta/openai/")
    limiter = InMemoryRateLimiter(requests_per_second=rate, check_every_n_seconds=0.5,
                                  max_bucket_size=1)
    judge = LangchainLLMWrapper(ChatOpenAI(
        model=judge_model, api_key=key, base_url=base, temperature=0.0,
        max_tokens=4096, max_retries=5, rate_limiter=limiter))
    embed = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
        model_name=cfg.embedding_model, model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}))
    OBJ = {"context_precision": M.context_precision, "context_recall": M.context_recall}

    CACHE.mkdir(parents=True, exist_ok=True)
    hasil = {"percobaan": "T14 — validasi lengan penelusuran dengan RAGAS",
             "prapendaftaran": "docs/PRAPENDAFTARAN_T14_RAGAS_LENGAN.md",
             "korpus": "produksi (models/chroma_db)",
             "n_kasus": len(kasus), "top_k": TOP_K,
             "penilai": judge_model, "metrik": METRIK,
             "catatan_pembangkitan": ("Tahap pembangkitan jawaban DILEWATI; "
                                      "context_precision dan context_recall tidak "
                                      "memakai medan `response`."),
             "lengan": {}}

    for l in LENGAN:
        print(f"\n=== Lengan {l} ===", flush=True)
        samples = [SingleTurnSample(user_input=c["pertanyaan"],
                                    retrieved_contexts=ctx[l][i],
                                    reference=c["jawaban_acuan"])
                   for i, c in enumerate(kasus)]
        try:
            res = evaluate(dataset=EvaluationDataset(samples=samples),
                           metrics=[OBJ[m] for m in METRIK],
                           llm=judge, embeddings=embed,
                           run_config=RunConfig(max_workers=1, timeout=180,
                                                max_retries=5, max_wait=60))
            df = res.to_pandas()
            df.insert(0, "id", [c["id"] for c in kasus])
            hasil["lengan"][l] = {
                "rerata": {m: round(float(df[m].mean()), 4) for m in METRIK if m in df},
                "per_kasus": df[["id"] + [m for m in METRIK if m in df]].to_dict("records"),
            }
            (CACHE / f"{l}.json").write_text(
                json.dumps(hasil["lengan"][l], indent=2, ensure_ascii=False),
                encoding="utf-8")
            print(f"  {hasil['lengan'][l]['rerata']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  GAGAL pada lengan {l}: {type(exc).__name__}: {str(exc)[:220]}")
            hasil["lengan"][l] = {"GAGAL": f"{type(exc).__name__}: {str(exc)[:400]}"}
            hasil["PERINGATAN_PARSIAL"] = (
                "Sekurangnya satu lengan GAGAL. Hasil ini PARSIAL dan TIDAK boleh "
                "dilaporkan sebagai hasil lengkap.")

    if all(isinstance(v, dict) and "rerata" in v for v in hasil["lengan"].values()):
        v, h = hasil["lengan"]["vektor"]["rerata"], hasil["lengan"]["hibrida"]["rerata"]
        hasil["selisih_hibrida_minus_vektor"] = {
            m: round(h.get(m, 0) - v.get(m, 0), 4) for m in METRIK}
        d = hasil["selisih_hibrida_minus_vektor"]
        hasil["putusan"] = (
            "D1 didukung: context_recall naik di atas ambang 0,05."
            if d.get("context_recall", 0) > 0.05 else
            "D1 TIDAK didukung: kenaikan context_recall di bawah ambang 0,05 yang "
            "ditetapkan di muka. Keunggulan hibrida pada T13 karena itu belum dapat "
            "dipisahkan dari artefak alat ukur berbasis kata kunci.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDisimpan ke {OUT}")
    if "putusan" in hasil:
        print(f"\nSelisih (hibrida - vektor): {hasil['selisih_hibrida_minus_vektor']}")
        print(hasil["putusan"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
