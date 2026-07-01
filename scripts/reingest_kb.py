#!/usr/bin/env python3
"""T7 — Re-ingest knowledge base ke ChromaDB (idempoten) + verifikasi.

Meng-ingest KB kurasi (manual_kb.json) DAN dokumen riil (PDF PERKENI/ADA/Kemenkes di
data/knowledge_base/additional_docs/) ke ChromaDB, dengan pelacakan sumber & jumlah chunk.

Idempoten: koleksi Chroma dibangun ulang dari nol setiap run (folder persist dihapus dulu),
sehingga hasil selalu deterministik dan aman dijalankan berulang.

Jalankan:
    python scripts/reingest_kb.py
    python scripts/reingest_kb.py --min-chars 300   # ambang teks PDF (lewati hasil scan)
"""
from __future__ import annotations

# torch harus sebelum numpy (Windows/conda c10.dll) — sentence-transformers memuat torch.
import torch  # noqa: F401

import os
# Model embedding sudah ter-cache lokal; paksa offline agar tidak memanggil HF Hub
# (menghindari error intermiten "client has been closed" saat init embedding).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import re
import shutil
from collections import Counter
from pathlib import Path


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _topic_from_name(path: Path) -> str:
    name = re.sub(r"[_\-]+", " ", path.stem)
    return re.sub(r"\s+", " ", name).strip().title() or "General"


def _read_pdf(path: Path) -> str:
    import PyPDF2

    pages = []
    with path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for page in reader.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def main() -> int:
    parser = argparse.ArgumentParser(description="T7: idempotent KB re-ingest + verify")
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument("--pdf-dir", default="data/knowledge_base/additional_docs")
    parser.add_argument("--persist", default="models/chroma_db")
    parser.add_argument("--collection", default="diabetes_kb")
    parser.add_argument("--embed", default="sentence-transformers")
    parser.add_argument("--min-chars", type=int, default=300,
                        help="Lewati PDF dengan teks terekstrak < ambang ini (indikasi hasil scan)")
    args = parser.parse_args()

    from src.rag.knowledge_base import MedicalKnowledgeBase

    kb = MedicalKnowledgeBase(
        kb_dir=args.kb_dir, persist_dir=args.persist,
        collection_name=args.collection, embed_provider=args.embed,
    )

    # 1) Dokumen kurasi (manual_kb.json)
    manual_docs = kb.load_manual_kb("manual_kb.json") or []
    docs = list(manual_docs)
    print(f"[1] manual_kb.json : {len(manual_docs)} dokumen")

    # 2) Dokumen riil (PDF)
    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    added, skipped = [], []
    for p in pdfs:
        try:
            text = _normalize(_read_pdf(p))
            if len(text) < args.min_chars:
                skipped.append((p.name, f"teks {len(text)} char — kemungkinan hasil scan/among gambar"))
                continue
            docs.append({
                "text": text, "source": p.name, "topic": _topic_from_name(p),
                "metadata": {"doc_id": re.sub(r'[^a-z0-9]+', '_', p.stem.lower())},
            })
            added.append((p.name, len(text)))
        except Exception as exc:  # noqa: BLE001
            skipped.append((p.name, f"ERROR: {exc}"))

    print(f"[2] PDF ditemukan  : {len(pdfs)} | ter-ekstrak: {len(added)} | dilewati: {len(skipped)}")
    for name, n in added:
        print(f"    + {name}: {n:,} char")
    for name, why in skipped:
        print(f"    - {name}: {why}")

    if not docs:
        print("Tidak ada dokumen untuk di-ingest.")
        return 1

    # 3) Rebuild Chroma dari nol (idempoten) — hapus folder persist yang mungkin korup/versi lama
    persist = Path(args.persist)
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
        print(f"[3] Folder chroma lama dihapus (fresh rebuild): {persist}")

    chunks = kb.chunk_documents(documents=docs)  # 900/120 default
    ok = kb.save_to_chroma(chunks=chunks, reset_collection=False)
    if not ok:
        print("GAGAL menyimpan ke Chroma.")
        return 1

    by_source = Counter(c["source"] for c in chunks)
    print(f"\n[4] Total chunk: {len(chunks)} | chunk per sumber:")
    for src, n in by_source.most_common():
        print(f"    {src}: {n} chunk")

    # 5) Verifikasi: buka ulang koleksi & hitung
    import chromadb
    client = chromadb.PersistentClient(path=args.persist)
    col = client.get_collection(args.collection)
    print(f"\n[5] Verifikasi ChromaDB: koleksi '{args.collection}' berisi {col.count()} chunk.")
    print("    Status: OK" if col.count() == len(chunks) else "    Status: MISMATCH!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
