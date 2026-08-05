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

Mengapa halaman 18 dan 23:
  - Hal. 18 memuat "E. STRATEGI PRAKTIS TERAPI INSULIN" dengan pernyataan dosis
    tunggal dan konkret (insulin basal awal 0,2 unit/kgBB) — ground truth yang
    dapat diverifikasi tanpa ambiguitas.
  - Hal. 23 memuat "Gambar III.2 Algoritma Strategi Umum Terapi Insulin Rawat
    Jalan": halaman terpadat angka tata laksana di seluruh dokumen, dan
    satu-satunya yang memuat algoritma respons hipoglikemia eksplisit.
  Bersama-sama keduanya mencakup inisiasi, titrasi, dan respons hipoglikemia —
  memetakan langsung ke tiga kondisi terprediksi sistem.

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
EVAL_PAGES = {"KB-03_PERKENI-2021_Terapi-Insulin.pdf": [18, 23]}


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
    print(f"\nKoleksi evaluasi '{args.collection}' berisi {col.count()} chunk "
          f"dari {len(docs)} halaman.")
    print(f"Lokasi: {persist}  (produksi tetap di {cfg.persist_dir}, tidak tersentuh)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
