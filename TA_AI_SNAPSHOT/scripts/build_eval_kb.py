#!/usr/bin/env python3
"""Bangun koleksi ChromaDB TERPISAH untuk evaluasi RAGAS.

Mengapa terpisah dari koleksi produksi:
  1. Ground truth dapat diverifikasi manual — hanya 2 halaman, sehingga setiap
     jawaban acuan dapat dilacak ke kalimat sumbernya.
  2. Konsumsi kuota LLM tertahan — RAGAS memanggil LLM penilai untuk tiap sampel
     tiap metrik, jadi korpus kecil menjaga jumlah panggilan tetap terkendali.

Koleksi ini memakai persist_dir dan collection_name yang berbeda dari produksi;
keduanya TIDAK PERNAH bercampur.

Sumber: KB-03 (PERKENI, Pedoman Petunjuk Praktis Terapi Insulin 2021),
halaman cetak 18 dan 23.

Mengapa KB-03 dan bukan KB-09 (ISPAD hipoglikemia): KB-09 berbahasa Inggris
sedangkan keluaran sistem berbahasa Indonesia, sehingga metrik RAGAS akan
mengukur mutu penelusuran DAN mutu penerjemahan sekaligus — skor rendah menjadi
tidak dapat ditafsirkan. KB-03 berbahasa Indonesia sepenuhnya.

Halaman yang dipakai (T1.2, disetujui pembimbing 6 Agustus 2026): cetak 23, 44, 45, 48.

  - Hal. 23 — "Gambar III.2 Algoritma Strategi Umum Terapi Insulin Rawat Jalan":
    halaman terpadat angka tata laksana di seluruh dokumen. Dosis awal 5-10
    unit/hari, penyesuaian 10-15% atau 2-4 unit, dan respons hipoglikemia
    eksplisit (turunkan 4 unit atau 10-20%).
  - Hal. 48 — "B. HIPOGLIKEMIA" + Tabel VIII.1 Klasifikasi Hipoglikemia. Memuat
    ambang <=70 mg/dL (Level 1) dan <54 mg/dL (Level 2), yaitu SUMBER LANGSUNG
    nilai GLUCOSE_LOW dan GLUCOSE_CRITICAL_LOW di src/constants.py.
  - Hal. 44 — infus insulin IV pada krisis hiperglikemia: 5-7 unit/jam menurunkan
    50-75 mg/dL/jam; bila GD < 250 mg/dL dosis dikurangi 50%.
  - Hal. 45 — insulin subkutan pada krisis hiperglikemia dan peringatan late
    hypoglycemia; melengkapi hal. 44.

Mengapa EMPAT halaman dan bukan dua: sistem mengklasifikasikan ke TIGA kelas
(hipoglikemia, normal, hiperglikemia). Dengan hanya halaman hipoglikemia, kasus uji
hiperglikemia tidak akan punya jawaban acuan yang sah dan context_recall bernilai nol
karena dokumennya memang tidak ada — bukan karena sistemnya gagal. Halaman 44 dan 45
menutup jalur hiperglikemia.

Halaman 18 (dipakai versi Tugas 6) DIGANTI: isinya inisiasi dosis yang sebagian
tumpang tindih dengan hal. 23, sedangkan kuota halaman lebih berguna untuk menutup
kondisi hiperglikemia yang sebelumnya tidak terwakili sama sekali.

Jalankan:
    PYTHONPATH=. python scripts/build_eval_kb.py
"""
from __future__ import annotations

import torch  # noqa: F401  (Windows: torch sebelum numpy — WinError 1114)

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from reingest_kb import _clean_text, _load_manifest, _page_metadata, _read_pdf_pages  # noqa: E402

# Halaman CETAK (bukan indeks PDF) yang menjadi basis evaluasi.
EVAL_PAGES = {"KB-03_PERKENI-2021_Terapi-Insulin.pdf": [23, 44, 45, 48]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Bangun koleksi ChromaDB evaluasi RAGAS")
    ap.add_argument("--pdf-dir", default="data/knowledge_base/books")
    ap.add_argument("--manifest", default="data/knowledge_base/manifest.csv")
    ap.add_argument("--persist", default="models/chroma_db_eval",
                    help="HARUS berbeda dari koleksi produksi (models/chroma_db)")
    ap.add_argument("--collection", default="diabetes_kb_eval")
    ap.add_argument("--embed", default="sentence-transformers")
    args = ap.parse_args()

    from src.config import load_rag_config
    from src.rag.knowledge_base import MedicalKnowledgeBase

    cfg = load_rag_config()
    if Path(args.persist).resolve() == Path(cfg.persist_dir).resolve():
        print(f"GAGAL: --persist ({args.persist}) sama dengan koleksi produksi "
              f"({cfg.persist_dir}). Koleksi evaluasi WAJIB terpisah.", file=sys.stderr)
        return 1
    if args.collection == cfg.collection_name:
        print(f"GAGAL: --collection ({args.collection}) sama dengan koleksi produksi "
              f"({cfg.collection_name}). Koleksi evaluasi WAJIB terpisah.", file=sys.stderr)
        return 1

    manifest = _load_manifest(Path(args.manifest))

    docs = []
    for nama_berkas, halaman_cetak_list in EVAL_PAGES.items():
        if nama_berkas not in manifest:
            print(f"GAGAL: {nama_berkas} tidak terdaftar di manifest.", file=sys.stderr)
            return 1
        entry = manifest[nama_berkas]
        pdf_path = Path(args.pdf_dir) / nama_berkas
        # Pilih berdasarkan halaman CETAK, bukan indeks PDF — inilah gunanya offset.
        want = set(halaman_cetak_list)
        found = set()
        for halaman_pdf, raw in _read_pdf_pages(pdf_path):
            cetak = halaman_pdf + entry.offset
            if cetak not in want:
                continue
            found.add(cetak)
            text = _clean_text(raw)
            meta = _page_metadata(entry, halaman_pdf)
            docs.append({
                "text": text,
                "source": entry.nama_berkas,
                "topic": entry.judul_lengkap,
                "metadata": meta,
            })
            print(f"  + {entry.kb_id} halaman cetak {cetak} (PDF {halaman_pdf}): {len(text)} char")

        hilang = want - found
        if hilang:
            print(f"GAGAL: halaman cetak {sorted(hilang)} tidak ditemukan di {nama_berkas}.",
                  file=sys.stderr)
            return 1

    if not docs:
        print("GAGAL: tidak ada halaman terpilih.", file=sys.stderr)
        return 1

    persist = Path(args.persist)
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
        print(f"Koleksi evaluasi lama dihapus (fresh rebuild): {persist}")

    kb = MedicalKnowledgeBase(
        kb_dir="data/knowledge_base", persist_dir=str(persist),
        collection_name=args.collection, embed_provider=args.embed,
    )
    chunks = kb.chunk_documents(documents=docs)
    if not kb.save_to_chroma(chunks=chunks, reset_collection=False):
        print("GAGAL menyimpan ke Chroma.", file=sys.stderr)
        return 1

    import chromadb
    col = chromadb.PersistentClient(path=str(persist)).get_collection(args.collection)
    n_chunk = col.count()
    print(f"\nKoleksi evaluasi '{args.collection}' berisi {n_chunk} chunk "
          f"dari {len(docs)} halaman.")
    print(f"Lokasi: {persist}  (produksi tetap di {cfg.persist_dir}, tidak tersentuh)")

    # Proporsi koleksi yang terambil tiap kueri. Diminta pembimbing dinyatakan
    # KUANTITATIF, bukan sekadar disebut "sempit", karena angka ini yang dipakai
    # menyatakan keterbatasan context_precision di Bab VI.
    top_k = cfg.top_k
    porsi = 100.0 * top_k / max(n_chunk, 1)
    print(f"\n=== Proporsi koleksi terambil per kueri ===")
    print(f"  top_k produksi           : {top_k}")
    print(f"  jumlah chunk koleksi     : {n_chunk}")
    print(f"  terambil per kueri       : {top_k}/{n_chunk} = {porsi:.1f}%")
    print(f"  fetch_k (kolam kandidat) : {cfg.fetch_k} = "
          f"{100.0 * min(cfg.fetch_k, n_chunk) / max(n_chunk, 1):.1f}% koleksi")
    N_PRODUKSI = 2061
    porsi_produksi = 100.0 * top_k / N_PRODUKSI
    if porsi >= 5:
        print(f"\n  CATATAN untuk Bab VI: {porsi:.1f}% koleksi terambil tiap kueri, versus")
        print(f"  {porsi_produksi:.2f}% pada korpus produksi ({N_PRODUKSI} chunk) — yaitu")
        print(f"  {porsi / porsi_produksi:.0f}x lebih besar.")
        print(f"  context_precision pada korpus sekecil ini cenderung OPTIMISTIS: peluang")
        print(f"  chunk relevan masuk top-k jauh lebih besar. Angka RAGAS di sini mengukur")
        print(f"  mutu PEMBANGKITAN pada konteks yang hampir pasti memadai, BUKAN mutu")
        print(f"  penelusuran pada korpus penuh, dan TIDAK dapat digeneralisasi ke sana.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
