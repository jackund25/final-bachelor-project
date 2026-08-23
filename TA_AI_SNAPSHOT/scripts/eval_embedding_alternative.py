"""Uji model embedding alternatif (multilingual) terhadap all-MiniLM-L6-v2.

Hipotesis awal: karena korpus pedoman klinis berbahasa Indonesia, model embedding
multilingual (paraphrase-multilingual-MiniLM-L12-v2) semestinya me-retrieve lebih baik
daripada all-MiniLM-L6-v2 yang dilatih untuk Bahasa Inggris.

Skrip ini menguji hipotesis tersebut secara langsung: korpus yang sama diindeks ulang
memakai model alternatif ke koleksi ChromaDB terpisah, lalu ablation korpus-penuh
(RAG standar vs PC-RAG) dijalankan ulang di atasnya dengan kasus uji, metrik, dan
top_k yang identik. Yang berubah HANYA model embedding.

Hasil pada laporan: hipotesis TIDAK terbukti — Hit@1 PC-RAG jatuh dari 83,3% ke 0% dan
MRR dari 0,889 ke 0,319, sehingga all-MiniLM-L6-v2 dipertahankan.

Keluaran: results/baseline_ablation_fullkb/embedding_alternative.json

Catatan: model alternatif (~470 MB) diunduh sekali oleh sentence-transformers.
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy/pandas — WinError 1114)
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ALT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
BASE_MODEL = "all-MiniLM-L6-v2"
ALT_PERSIST = "models/chroma_db_multilingual"
COLLECTION = "diabetes_kb"

# Tag korpus untuk penamaan keluaran, sama seperti skrip evaluasi lain. Hipotesis
# multilingual pernah diuji pada korpus LAMA (14 PDF PERKENI/ADA) dan gagal — MRR
# PC-RAG 0,889 -> 0,319. Korpus baru berkomposisi bahasa berbeda (8 dari 12 dokumen
# berbahasa Inggris), sehingga itu eksperimen yang berbeda dan hasil lama TIDAK boleh
# dipakai untuk menyimpulkannya. Sufiks ini menjaga keduanya berdampingan.
CORPUS_TAG = os.environ.get("CORPUS_TAG", "kb12")
OUT = ROOT / f"results/baseline_ablation_fullkb_{CORPUS_TAG}/embedding_alternative.json"


def ingest_with(model_name: str, persist_dir: str) -> int:
    """Indeks ulang korpus yang sama memakai model embedding tertentu.

    Memanggil scripts/reingest_kb.py — jalur ingest produksi — agar korpus pedoman
    dan parameter chunking benar-benar identik; yang berbeda hanya embedding.
    """
    import subprocess

    env = dict(os.environ, HF_EMBED_MODEL=model_name, PYTHONPATH=str(ROOT))
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts/reingest_kb.py"),
         "--persist", persist_dir, "--collection", COLLECTION],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"reingest gagal:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")

    import chromadb
    count = chromadb.PersistentClient(path=persist_dir).get_collection(COLLECTION).count()
    if count == 0:
        raise RuntimeError("Indeks alternatif kosong — perbandingan tidak sah.")
    return count


# ── Bahasa dokumen (B1) ───────────────────────────────────────────────────────
# Manifest tidak memuat kolom bahasa, tetapi pemetaannya tegas dari lembaga
# penerbitnya: IDAI dan PERKENI menerbitkan dalam Bahasa Indonesia, ADA/EASD, ATTD,
# dan ISPAD dalam Bahasa Inggris. Diverifikasi terhadap teks korpus oleh
# `periksa_pemetaan_bahasa()` supaya pemetaan ini tidak sekadar diasumsikan.
BAHASA_ID = {"KB-01", "KB-02", "KB-03", "KB-04"}
BAHASA_EN = {"KB-05", "KB-06", "KB-07", "KB-08", "KB-09", "KB-10", "KB-11", "KB-12"}

# Kata fungsi yang sangat sering muncul dan hampir tidak beririsan antar-kedua bahasa.
_STOP_ID = (" yang ", " dan ", " pada ", " dengan ", " untuk ", " tidak ", " dapat ", " adalah ")
_STOP_EN = (" the ", " and ", " with ", " for ", " that ", " should ", " are ", " been ")


def bahasa_chunk(meta: dict) -> str:
    """Bahasa dokumen asal sebuah chunk, dari kb_id pada metadata."""
    kb = str(meta.get("kb_id") or "").strip().upper()[:5]
    if kb in BAHASA_ID:
        return "indonesia"
    if kb in BAHASA_EN:
        return "inggris"
    return "tak_diketahui"


def deteksi_bahasa_teks(teks: str) -> str:
    """Tebakan bahasa dari rasio kata fungsi. Hanya untuk MEMVERIFIKASI pemetaan kb_id."""
    t = " " + teks.lower() + " "
    return "indonesia" if sum(t.count(w) for w in _STOP_ID) >= sum(t.count(w) for w in _STOP_EN) \
        else "inggris"


def ablate_on(persist_dir: str, model_name: str) -> dict:
    """Jalankan ablation korpus-penuh (identik dengan ablation_rag_fullkb) pada indeks tertentu."""
    os.environ["HF_EMBED_MODEL"] = model_name
    for mod in [m for m in list(sys.modules) if m.startswith("src.rag")]:
        del sys.modules[mod]

    # WAJIB: sejak Tugas 4, nama model embedding dibaca lewat load_rag_config() yang
    # ber-lru_cache. Menghapus modul src.rag saja TIDAK cukup — cache config tetap
    # memegang nilai lama, sehingga indeks alternatif akan dikueri memakai embedding
    # model yang salah dan seluruh perbandingan menjadi tidak sah tanpa error apa pun.
    from src.config import clear_cache, load_rag_config
    clear_cache()
    efektif = load_rag_config().embedding_model
    if efektif != model_name:
        raise RuntimeError(
            f"Model embedding efektif ({efektif}) != yang diminta ({model_name}). "
            f"Perbandingan dibatalkan karena tidak sah."
        )

    from src.rag.retriever import MMRRetriever

    import importlib
    ab = importlib.import_module("ablation_rag_fullkb")

    r = MMRRetriever(persist_dir=persist_dir, collection_name=COLLECTION,
                     embed_provider="sentence-transformers")

    hasil = {"ablasi_6_kasus": {}, "realcases": {}, "waktu_kueri_dtk": None}
    t0 = time.time()

    # (a) Enam kasus divergen — melanjutkan pelaporan sebelumnya.
    for mode in ("standard", "prediction_conditioned"):
        kasus = []
        for case in ab.TEST_CASES:
            q = ab.build_query(case, mode)
            docs = r.retrieve(q, top_k=ab.TOP_K)
            kasus.append(skor_satu_kueri(docs, case["expected"], ab))
        hasil["ablasi_6_kasus"][mode] = agregat(kasus, ab.TOP_K)

    # (b) Kasus nyata pasien hold-out — jauh lebih banyak (n=120 per himpunan) sehingga
    # pemecahan per bahasa punya dasar yang layak. Kuerinya DIPUTAR ULANG dari nilai
    # glukosa yang sudah tersimpan, jadi tidak perlu menjalankan RF lagi dan kasusnya
    # persis sama dengan yang dipakai evaluasi utama.
    for himpunan, kueri in muat_kueri_realcases().items():
        kasus = [skor_satu_kueri(r.retrieve(q, top_k=ab.TOP_K), exp, ab) for q, exp in kueri]
        hasil["realcases"][himpunan] = agregat(kasus, ab.TOP_K)

    hasil["waktu_kueri_dtk"] = round(time.time() - t0, 1)
    return hasil


def skor_satu_kueri(docs, expected: str, ab) -> dict:
    """Skor satu kueri, sekaligus dipecah menurut bahasa dokumen sumbernya."""
    topics = [ab.classify_chunk(d["text"]) for d in docs]
    langs = [bahasa_chunk(d.get("metadata", {})) for d in docs]

    def rank_of(mask) -> int:
        for i, ok in enumerate(mask):
            if ok:
                return i + 1
        return 0

    relevan = [t == expected for t in topics]
    baris = {"rank": rank_of(relevan), "n_docs": len(docs)}
    # Per bahasa: relevan HANYA bila topiknya benar DAN dokumennya berbahasa tersebut.
    # Ini menjawab pertanyaan B1 secara langsung: apakah model mampu memunculkan
    # dokumen berbahasa Indonesia ke peringkat atas.
    for bhs in ("indonesia", "inggris"):
        baris[f"rank_{bhs}"] = rank_of([r and l == bhs for r, l in zip(relevan, langs)])
    baris["n_indonesia"] = sum(1 for l in langs if l == "indonesia")
    return baris


def agregat(kasus: list[dict], top_k: int) -> dict:
    """Hit@1, Hit@3, Hit@k, dan MRR — keseluruhan dan per bahasa dokumen sasaran."""
    n = max(len(kasus), 1)

    def metrik(kunci: str) -> dict:
        ranks = [k[kunci] for k in kasus]
        return {
            "hit@1(%)": round(100 * sum(1 for x in ranks if x == 1) / n, 1),
            "hit@3(%)": round(100 * sum(1 for x in ranks if 0 < x <= 3) / n, 1),
            f"hit@{top_k}(%)": round(100 * sum(1 for x in ranks if 0 < x <= top_k) / n, 1),
            "mrr": round(sum((1.0 / x) if x else 0.0 for x in ranks) / n, 3),
        }

    return {
        "n": len(kasus),
        "keseluruhan": metrik("rank"),
        "dokumen_indonesia": metrik("rank_indonesia"),
        "dokumen_inggris": metrik("rank_inggris"),
        "rerata_chunk_indonesia_di_top_k": round(
            sum(k["n_indonesia"] for k in kasus) / n, 2),
    }


def muat_kueri_realcases() -> dict:
    """Putar ulang kueri kasus nyata dari hasil yang sudah tersimpan.

    Memakai `query_glucose` (nilai yang benar-benar dipakai membentuk kueri) dan
    `expected` (kondisi yang BENAR-BENAR terjadi), sehingga himpunan kasusnya identik
    dengan evaluasi utama tanpa perlu menjalankan RF lagi.
    """
    import csv

    from src.rag.ablation_query import build_ablation_query

    keluar = {}
    for berkas, nama in (("per_case_divergen.csv", "divergen"),
                         ("per_case_natural.csv", "natural")):
        p = ROOT / f"results/retrieval_realcases_{CORPUS_TAG}/{berkas}"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("mode") == "pc_rag"]
        keluar[nama] = [(build_ablation_query(float(r["query_glucose"])), r["expected"])
                        for r in rows]
    return keluar


def periksa_pemetaan_bahasa(persist_dir: str) -> dict:
    """Verifikasi pemetaan kb_id -> bahasa terhadap isi teks, bukan sekadar diasumsikan."""
    import chromadb

    col = chromadb.PersistentClient(path=persist_dir).get_collection(COLLECTION)
    got = col.get(limit=4000, include=["documents", "metadatas"])
    per_kb: dict[str, list[int]] = {}
    for teks, meta in zip(got["documents"], got["metadatas"]):
        kb = str((meta or {}).get("kb_id") or "")[:5].upper()
        if not kb:
            continue
        per_kb.setdefault(kb, []).append(deteksi_bahasa_teks(teks) == "indonesia")

    laporan = {}
    for kb, flags in sorted(per_kb.items()):
        rasio_id = sum(flags) / max(len(flags), 1)
        terdeteksi = "indonesia" if rasio_id >= 0.5 else "inggris"
        dipetakan = "indonesia" if kb in BAHASA_ID else (
            "inggris" if kb in BAHASA_EN else "tak_diketahui")
        laporan[kb] = {"n_chunk": len(flags), "rasio_indonesia": round(rasio_id, 3),
                       "terdeteksi": terdeteksi, "dipetakan": dipetakan,
                       "cocok": terdeteksi == dipetakan}
    return laporan


def cetak_blok(judul: str, res: dict) -> None:
    print(f"\n  {judul} (n={res['n']})")
    print(f"    {'sasaran':<22}{'Hit@1':>8}{'Hit@3':>8}{'MRR':>8}")
    for label, kunci in (("keseluruhan", "keseluruhan"),
                         ("dokumen Indonesia", "dokumen_indonesia"),
                         ("dokumen Inggris", "dokumen_inggris")):
        m = res[kunci]
        print(f"    {label:<22}{m['hit@1(%)']:>7.1f}%{m['hit@3(%)']:>7.1f}%{m['mrr']:>8.3f}")
    print(f"    rerata chunk Indonesia di top-k: {res['rerata_chunk_indonesia_di_top_k']}")


def main() -> None:
    print(f"[1/4] Indeks ulang korpus dengan {ALT_MODEL} ...")
    t0 = time.time()
    n_chunks_alt = ingest_with(ALT_MODEL, ALT_PERSIST)
    t_ingest_alt = round(time.time() - t0, 1)
    print(f"      {n_chunks_alt} chunk terindeks ke {ALT_PERSIST} dalam {t_ingest_alt} dtk")

    print("[2/4] Verifikasi pemetaan bahasa terhadap isi teks ...")
    pemetaan = periksa_pemetaan_bahasa(ALT_PERSIST)
    menyimpang = [k for k, v in pemetaan.items() if not v["cocok"]]

    print(f"[3/4] Evaluasi pada indeks multilingual ...")
    alt = ablate_on(ALT_PERSIST, ALT_MODEL)

    print("[4/4] Evaluasi pada indeks produksi (all-MiniLM) ...")
    base = ablate_on("models/chroma_db", BASE_MODEL)

    import chromadb
    n_chunks_base = chromadb.PersistentClient(path="models/chroma_db") \
        .get_collection(COLLECTION).count()

    out = {
        "catatan": ("Korpus, kasus uji, metrik, dan top_k identik. Hanya model embedding "
                    "yang berbeda. Kueri memakai pembentuk simetris pasca-A1."),
        "pemetaan_bahasa": pemetaan,
        "pemetaan_bahasa_menyimpang": menyimpang,
        "ukuran_koleksi": {BASE_MODEL: n_chunks_base, ALT_MODEL: n_chunks_alt},
        "waktu_indeks_ulang_dtk": {ALT_MODEL: t_ingest_alt},
        BASE_MODEL: base,
        ALT_MODEL: alt,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    if menyimpang:
        print(f"\nPERINGATAN: pemetaan bahasa menyimpang dari deteksi isi untuk {menyimpang}")
    else:
        print("      Pemetaan bahasa kb_id COCOK dengan deteksi isi untuk seluruh dokumen.")

    for nama, res in ((BASE_MODEL, base), (ALT_MODEL, alt)):
        print(f"\n=== {nama} ===")
        cetak_blok("ablasi 6 kasus divergen (PC-RAG)",
                   res["ablasi_6_kasus"]["prediction_conditioned"])
        for h, v in res["realcases"].items():
            cetak_blok(f"realcases {h}", v)

    print(f"\nUkuran koleksi : {BASE_MODEL} {n_chunks_base} chunk | "
          f"{ALT_MODEL} {n_chunks_alt} chunk")
    print(f"Waktu indeks ulang multilingual: {t_ingest_alt} dtk")
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
