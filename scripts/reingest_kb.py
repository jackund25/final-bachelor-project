#!/usr/bin/env python3
"""Re-ingest knowledge base ke ChromaDB dengan metadata halaman yang dapat ditelusuri.

Meng-ingest KB kurasi (manual_kb.json) DAN dokumen pedoman (KB-01..KB-12 di
data/knowledge_base/books/) ke ChromaDB.

PERUBAHAN PENTING (Tugas 1, 4 Agt 2026)
---------------------------------------
Versi sebelumnya menggabungkan seluruh halaman PDF dengan "\\n\\n".join(pages)
SEBELUM chunking, sehingga batas halaman hilang permanen dan setiap sitasi
tampil sebagai "Hal. N/A". Versi ini:

- Ekstraksi PDF dilakukan PER HALAMAN; setiap halaman menjadi satu "dokumen"
  bagi chunk_documents(), sehingga tidak ada chunk yang melintasi batas halaman.
- Metadata dokumen dibaca dari data/knowledge_base/manifest.csv.
- halaman_cetak = halaman_pdf + offset (offset bisa negatif untuk dokumen yang
  memulai penomoran ulang setelah front matter, atau besar untuk artikel jurnal
  dengan penomoran berkelanjutan).
- Halaman dengan halaman_cetak < 1 (front matter: sampul, daftar isi) TIDAK
  diindeks: isinya padat kata kunci topik tanpa isi berguna sehingga mencemari
  hasil retrieval.
- PDF yang tidak terdaftar di manifest MENGHENTIKAN proses; tidak ada skip diam.

Idempoten: koleksi Chroma dibangun ulang dari nol setiap run.

Jalankan:
    python scripts/reingest_kb.py
    python scripts/reingest_kb.py --min-page-chars 150
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
import csv
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────
# Manifest
# ──────────────────────────────────────────────────────────────

MANIFEST_COLUMNS = [
    "kb_id", "source_id", "nama_berkas", "lembaga",
    "tahun", "judul_lengkap", "hlm_total", "offset",
]


@dataclass(frozen=True)
class ManifestEntry:
    """Satu baris manifest.csv — metadata bibliografis satu dokumen pedoman."""

    kb_id: str
    source_id: str
    nama_berkas: str
    lembaga: str
    tahun: int
    judul_lengkap: str
    hlm_total: int
    offset: int


class ManifestError(RuntimeError):
    """Manifest tidak lengkap / tidak konsisten dengan berkas di disk."""


def _load_manifest(manifest_path: Path) -> Dict[str, ManifestEntry]:
    """Baca manifest.csv menjadi dict berkunci nama_berkas."""
    if not manifest_path.exists():
        raise ManifestError(
            f"Manifest tidak ditemukan: {manifest_path}\n"
            f"Manifest wajib ada — ia satu-satunya sumber metadata halaman."
        )

    # utf-8-sig: Excel di Windows kerap menulis BOM di awal berkas CSV.
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing_cols = [c for c in MANIFEST_COLUMNS if c not in (reader.fieldnames or [])]
        if missing_cols:
            raise ManifestError(
                f"Kolom manifest tidak lengkap. Hilang: {', '.join(missing_cols)}\n"
                f"Kolom wajib: {', '.join(MANIFEST_COLUMNS)}"
            )

        entries: Dict[str, ManifestEntry] = {}
        for lineno, row in enumerate(reader, start=2):
            if not (row.get("kb_id") or "").strip():
                continue  # lewati baris kosong di akhir berkas
            try:
                entry = ManifestEntry(
                    kb_id=row["kb_id"].strip(),
                    source_id=row["source_id"].strip(),
                    nama_berkas=row["nama_berkas"].strip(),
                    lembaga=row["lembaga"].strip(),
                    tahun=int(row["tahun"]),
                    judul_lengkap=row["judul_lengkap"].strip(),
                    hlm_total=int(row["hlm_total"]),
                    offset=int(row["offset"]),
                )
            except (ValueError, KeyError) as exc:
                raise ManifestError(
                    f"Baris {lineno} manifest tidak dapat diparse: {exc}\n  isi: {row}"
                ) from exc
            if entry.nama_berkas in entries:
                raise ManifestError(
                    f"Baris {lineno}: nama_berkas duplikat '{entry.nama_berkas}'"
                )
            entries[entry.nama_berkas] = entry

    if not entries:
        raise ManifestError(f"Manifest kosong: {manifest_path}")
    return entries


def _printed_page(halaman_pdf: int, offset: int) -> Optional[int]:
    """halaman_cetak = halaman_pdf + offset; None bila < 1 (front matter).

    Offset negatif terjadi pada dokumen yang memulai penomoran ulang setelah
    front matter (mis. KB-03 offset -15: PDF hal. 16 = halaman cetak 1).
    Offset besar terjadi pada artikel jurnal dengan penomoran berkelanjutan
    (mis. KB-09 offset 1321: PDF hal. 1 = halaman cetak 1322).
    """
    cetak = halaman_pdf + offset
    return cetak if cetak >= 1 else None


# ──────────────────────────────────────────────────────────────
# Ekstraksi & pembersihan teks
# ──────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_LIGATURES = {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi",
              "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st"}


def _clean_text(text: str) -> str:
    """Buang noise ekstraksi PDF: ligatur, karakter kontrol, baris daftar-isi (dot leader),
    dan boilerplate header/footer jurnal (ADA/diabetesjournals). Menaikkan kualitas chunk.

    Diterapkan PER HALAMAN — header/footer memang artefak per halaman, sehingga
    penyaringan tidak lagi bocor antar halaman seperti pada versi teks-gabungan.
    """
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    # karakter kontrol non-printable (mis. \x83) → spasi
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", " ", text)
    kept = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if re.search(r"\.{4,}\s*\d+\s*$", s):                       # daftar isi: "Bab .... 12"
            continue
        if re.search(r"diabetesjournals\.org|Downloaded from http|article-pdf|by guest on", s, re.I):
            continue
        if re.search(r"Diabetes Care Volume \d+,\s*Supplement", s, re.I):  # footer ADA
            continue
        kept.append(s)
    return _normalize("\n".join(kept))


def _read_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """Ekstrak PDF menjadi daftar (halaman_pdf 1-based, teks mentah).

    MENGGANTIKAN _read_pdf() lama yang menggabungkan halaman dengan "\\n\\n".join()
    dan menghancurkan batas halaman.
    """
    import PyPDF2

    pages: List[Tuple[int, str]] = []
    with path.open("rb") as fh:
        reader = PyPDF2.PdfReader(fh)
        for idx, page in enumerate(reader.pages, start=1):
            pages.append((idx, page.extract_text() or ""))
    return pages


def _pdf_page_count(path: Path) -> int:
    import PyPDF2

    with path.open("rb") as fh:
        return len(PyPDF2.PdfReader(fh).pages)


# ──────────────────────────────────────────────────────────────
# Metadata & dokumen per halaman
# ──────────────────────────────────────────────────────────────

def _page_metadata(entry: ManifestEntry, halaman_pdf: int) -> Dict[str, Any]:
    """Metadata datar & aman-Chroma untuk satu halaman.

    Seluruh nilai skalar (str/int/bool) — ChromaDB menolak list/dict/None.
    Halaman tanpa nomor cetak dinyatakan dengan pasangan sentinel
    halaman_cetak=0 + halaman_cetak_valid=False, BUKAN None.
    """
    cetak = _printed_page(halaman_pdf, entry.offset)
    return {
        "kb_id": entry.kb_id,
        "source_id": entry.source_id,
        "nama_dokumen": entry.nama_berkas,
        "lembaga": entry.lembaga,
        "tahun": entry.tahun,
        "judul_lengkap": entry.judul_lengkap,
        "halaman_pdf": halaman_pdf,
        "halaman_cetak": cetak if cetak is not None else 0,
        "halaman_cetak_valid": cetak is not None,
        # doc_id unik PER HALAMAN: chunk_documents() me-reset part_idx per dokumen,
        # sehingga tanpa ini chunk_id halaman 1 dan halaman 2 akan bertabrakan.
        "doc_id": f"{entry.kb_id}_p{halaman_pdf:04d}",
        "domain": "buku_panduan",
    }


def _build_page_docs(
    pdf_path: Path,
    entry: ManifestEntry,
    min_page_chars: int,
) -> Tuple[List[Dict[str, Any]], int, int, int]:
    """Bangun dokumen-per-halaman untuk satu PDF.

    Returns:
        (docs, n_front_matter, n_too_short, total_chars)
    """
    docs: List[Dict[str, Any]] = []
    n_front = n_short = 0
    total_chars = 0

    for halaman_pdf, raw in _read_pdf_pages(pdf_path):
        text = _clean_text(raw)
        total_chars += len(text)

        if _printed_page(halaman_pdf, entry.offset) is None:
            # Front matter: sampul & daftar isi. Padat kata kunci topik tanpa isi
            # berguna → mencemari retrieval. Tidak diindeks.
            n_front += 1
            continue

        if len(text) < min_page_chars:
            n_short += 1
            continue

        meta = _page_metadata(entry, halaman_pdf)
        docs.append({
            "text": text,
            "source": entry.nama_berkas,
            "topic": entry.judul_lengkap,
            "metadata": meta,
        })

    return docs, n_front, n_short, total_chars


def _validate_corpus(
    pdf_dir: Path,
    manifest: Dict[str, ManifestEntry],
) -> List[Tuple[Path, ManifestEntry]]:
    """Pastikan manifest dan isi direktori cocok persis. Abort bila tidak.

    Tidak ada skip diam-diam: mengindeks 11 dari 12 dokumen tanpa peringatan
    sama berbahayanya dengan mengindeks dokumen yang salah.
    """
    if not pdf_dir.exists():
        raise ManifestError(f"Direktori PDF tidak ada: {pdf_dir}")

    on_disk = {p.name: p for p in sorted(pdf_dir.glob("*.pdf"))}

    tak_terdaftar = sorted(set(on_disk) - set(manifest))
    if tak_terdaftar:
        raise ManifestError(
            "PDF berikut ada di disk tetapi TIDAK terdaftar di manifest:\n"
            + "\n".join(f"  - {n}" for n in tak_terdaftar)
            + "\nTambahkan barisnya ke manifest.csv (termasuk offset halaman), "
              "atau pindahkan berkasnya keluar dari direktori korpus."
        )

    tak_ada_berkas = sorted(set(manifest) - set(on_disk))
    if tak_ada_berkas:
        raise ManifestError(
            "Baris manifest berikut TIDAK punya berkas di disk:\n"
            + "\n".join(f"  - {n}" for n in tak_ada_berkas)
            + f"\nDirektori yang diperiksa: {pdf_dir}"
        )

    pasangan: List[Tuple[Path, ManifestEntry]] = []
    salah_hitung = []
    for nama, path in on_disk.items():
        entry = manifest[nama]
        aktual = _pdf_page_count(path)
        if aktual != entry.hlm_total:
            salah_hitung.append(f"  - {nama}: manifest={entry.hlm_total}, aktual={aktual}")
        pasangan.append((path, entry))

    if salah_hitung:
        raise ManifestError(
            "Jumlah halaman tidak cocok dengan manifest:\n"
            + "\n".join(salah_hitung)
            + "\nJumlah halaman yang meleset berarti offset diturunkan dari revisi "
              "berkas yang berbeda — SELURUH halaman_cetak dokumen itu akan salah."
        )

    return pasangan


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-ingest KB ke ChromaDB dengan metadata halaman (idempoten)"
    )
    parser.add_argument("--kb-dir", default="data/knowledge_base")
    parser.add_argument(
        "--n-cadangan-per-dokumen", type=int, default=3,
        help=("Jumlah potongan terpanjang per dokumen yang diekspor ke "
              "fallback_chunks.json sebagai cadangan penelusuran KNF-04."))
    parser.add_argument("--pdf-dir", default="data/knowledge_base/books")
    parser.add_argument("--manifest", default="data/knowledge_base/manifest.csv")
    parser.add_argument("--persist", default="models/chroma_db")
    parser.add_argument("--collection", default="diabetes_kb")
    parser.add_argument("--embed", default="sentence-transformers")
    parser.add_argument("--min-page-chars", type=int, default=100,
                        help="Lewati halaman dengan teks terekstrak < ambang ini")
    parser.add_argument("--min-doc-chars", type=int, default=300,
                        help="Abort bila total teks satu PDF < ambang ini (indikasi hasil scan)")
    parser.add_argument("--min-chunk-chars", type=int, default=80,
                        help="Buang chunk lebih pendek dari ambang ini (fragmen ekor halaman)")
    # Ditambahkan untuk B4. Tanpa ini ukuran potongan hanya dapat diubah dengan menyunting
    # config.yaml, sehingga sapuan parameter akan meninggalkan config dalam keadaan berubah
    # bila skripnya gagal di tengah. Default None = ikut config, jadi perilaku lama utuh.
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="Timpa rag.chunk_size untuk satu jalannya (dipakai sapuan B4)")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="Timpa rag.chunk_overlap untuk satu jalannya (dipakai sapuan B4)")
    # Ditambahkan untuk T7 (strategi chunking). Default keduanya = perilaku lama,
    # supaya indeks yang angkanya sudah dilaporkan tidak berubah diam-diam.
    parser.add_argument("--pemisah-kalimat", action="store_true",
                        help="Sisipkan '. ', '! ', '? ', '; ' sebelum spasi pada daftar "
                             "pemisah, sehingga potongan berhenti di batas kalimat "
                             "(Gao dkk. 2023 §V.A.1)")
    parser.add_argument("--satuan-panjang", choices=["karakter", "token"],
                        default="karakter",
                        help="Satuan pengukur chunk_size. 'token' menakar dengan "
                             "tokenizer model embedding sehingga pemotongan senyap "
                             "256 token menjadi mustahil")
    parser.add_argument("--max-seq-length", type=int, default=None,
                        help="Timpa jendela token model embedding (bawaan MiniLM 256, "
                             "sedangkan BERT di bawahnya mendukung 512). Menaikkannya "
                             "menghapus pemotongan senyap TANPA mengubah batas potongan")
    args = parser.parse_args()

    # Disetel SEBELUM src.rag diimpor: RagConfig di-cache lru_cache pada pembacaan
    # pertama, sehingga menyetelnya belakangan tidak akan terbaca dan jendelanya
    # diam-diam kembali ke 256. Lewat environment agar proses ini DAN retriever
    # mana pun yang dijalankan dengan env sama memakai jendela yang sama — dokumen
    # dan kueri wajib diwakili dengan aturan yang sama.
    if args.max_seq_length:
        os.environ["EMBED_MAX_SEQ_LENGTH"] = str(args.max_seq_length)

    from src.rag.knowledge_base import MedicalKnowledgeBase

    kb_dir = Path(args.kb_dir)
    pdf_dir = Path(args.pdf_dir)
    manifest_path = Path(args.manifest)

    kb = MedicalKnowledgeBase(
        kb_dir=str(kb_dir), persist_dir=args.persist,
        collection_name=args.collection, embed_provider=args.embed,
    )

    # 1) Manifest + validasi korpus (abort sebelum pekerjaan mahal apa pun)
    try:
        manifest = _load_manifest(manifest_path)
        pasangan = _validate_corpus(pdf_dir, manifest)
    except ManifestError as exc:
        print(f"\nGAGAL: {exc}\n", file=sys.stderr)
        return 1

    print(f"[1] Manifest      : {len(manifest)} dokumen terdaftar ({manifest_path})")
    print(f"    Korpus        : {pdf_dir} — cocok, jumlah halaman terverifikasi")

    # 2) TIDAK ADA lagi dokumen kurasi manual.
    #
    # MENGAPA DICABUT (17 Agustus 2026). manual_kb.json berisi prosa yang disusun sendiri
    # oleh peneliti, bukan kutipan dari pedoman klinis terbitan resmi. Ia menyumbang 15
    # potongan ke indeks, dan potongan-potongan itu TIDAK memiliki nomor halaman sumber,
    # sehingga citations.py menampilkannya sebagai "Hal. tidak tercatat".
    #
    # Akibatnya klaim KNF-08 pada laporan — bahwa tiap potongan menyimpan identitas dokumen
    # beserta nomor halamannya — TIDAK benar selama potongan itu ada di dalam indeks.
    # Mencabutnya membuat seluruh isi korpus tertelusur ke pedoman terbitan resmi tanpa
    # kecuali, dan itu memperkuat, bukan mengurangi, dasar sistem ini.
    #
    # Berkasnya dipindahkan ke arsip dan TIDAK dipakai lagi oleh jalur mana pun.
    docs: List[Dict[str, Any]] = []
    print("[2] Dokumen kurasi manual: DICABUT — korpus murni pedoman terbitan resmi")

    # 3) Dokumen pedoman — ekstraksi PER HALAMAN
    print(f"[3] Ekstraksi per halaman:")
    total_front = total_short = 0
    ringkas_front: List[Tuple[str, int, int]] = []
    for pdf_path, entry in pasangan:
        page_docs, n_front, n_short, total_chars = _build_page_docs(
            pdf_path, entry, args.min_page_chars
        )
        if total_chars < args.min_doc_chars:
            print(
                f"\nGAGAL: {entry.nama_berkas} hanya menghasilkan {total_chars} karakter "
                f"(< {args.min_doc_chars}).\nKemungkinan PDF hasil scan tanpa lapisan teks; "
                f"perlu OCR sebelum dapat diindeks.\n",
                file=sys.stderr,
            )
            return 1

        docs.extend(page_docs)
        total_front += n_front
        total_short += n_short
        ringkas_front.append((entry.kb_id, n_front, len(page_docs)))
        cetak_lo = _printed_page(1, entry.offset) or 1
        cetak_hi = entry.hlm_total + entry.offset
        print(
            f"    {entry.kb_id}  {entry.hlm_total:3d} hal PDF  "
            f"-> {len(page_docs):3d} diindeks  "
            f"(front matter {n_front:2d}, terlalu pendek {n_short:2d})  "
            f"cetak {cetak_lo}-{cetak_hi}"
        )

    print(f"    TOTAL: {total_front} halaman depan tersaring, "
          f"{total_short} halaman terlalu pendek, "
          f"{len(docs)} halaman diindeks")

    if not docs:
        print("Tidak ada dokumen untuk di-ingest.", file=sys.stderr)
        return 1

    # 4) Rebuild Chroma dari nol (idempoten)
    persist = Path(args.persist)
    if persist.exists():
        shutil.rmtree(persist, ignore_errors=True)
        print(f"[4] Folder chroma lama dihapus (fresh rebuild): {persist}")

    chunks = kb.chunk_documents(documents=docs, chunk_size=args.chunk_size,
                                chunk_overlap=args.chunk_overlap,
                                # None = ikut rag.pemisah_kalimat pada config.
                                # Meneruskan False saat argumen tidak diberikan akan
                                # MENIMPA config dan diam-diam mengindeks dengan
                                # pemisah lama meski config menyalakannya.
                                pemisah_kalimat=(True if args.pemisah_kalimat else None),
                                satuan_panjang=args.satuan_panjang)
    if args.chunk_size is not None or args.chunk_overlap is not None:
        print(f"    (chunk_size={args.chunk_size or 'config'}, "
              f"chunk_overlap={args.chunk_overlap or 'config'} — timpaan dari argumen)")
    # Nilai EFEKTIF, bukan nilai argumen. Mencetak args.pemisah_kalimat akan
    # melaporkan False padahal config menyalakannya — persis jenis penyimpangan
    # senyap antara yang dilaporkan dan yang dijalankan yang menjadi pokok T7.
    pemisah_efektif = bool(args.pemisah_kalimat or getattr(kb.cfg, "pemisah_kalimat", False))
    print(f"    (satuan_panjang={args.satuan_panjang}, "
          f"pemisah_kalimat={pemisah_efektif}, "
          f"embedding={kb.cfg.embedding_model}, "
          f"max_seq_length={kb.cfg.embedding_max_seq_length or 'bawaan model'})")

    # Fragmen ekor halaman: pemecahan per halaman menghasilkan potongan pendek
    # yang dulu tersembunyi oleh penggabungan antar halaman.
    sebelum = len(chunks)
    chunks = [c for c in chunks if len(c["text"].strip()) >= args.min_chunk_chars]
    dibuang = sebelum - len(chunks)

    ok = kb.save_to_chroma(chunks=chunks, reset_collection=False)
    if not ok:
        print("GAGAL menyimpan ke Chroma.", file=sys.stderr)
        return 1

    by_source = Counter(c["source"] for c in chunks)
    print(f"\n[5] Total chunk: {len(chunks)} "
          f"({dibuang} fragmen < {args.min_chunk_chars} char dibuang) | chunk per sumber:")
    for src, n in by_source.most_common():
        print(f"    {src}: {n} chunk")

    # 6) Cadangan penelusuran (KNF-04) — diekspor DARI korpus pedoman.
    #
    # MENGAPA ADA. Bila ChromaDB tidak dapat dibuka, pipeline mundur ke
    # SimpleKeywordRetriever, dan penelusur itu memerlukan potongan dalam memori.
    # Sebelumnya sumbernya manual_kb.json, yang kini dicabut. Menggantinya dengan ekspor
    # dari korpus pedoman menjaga DUA klaim sekaligus tetap benar: KNF-04 (cadangan
    # terkendali tersedia) dan KNF-08 (tiap potongan tertelusur ke dokumen dan halamannya).
    #
    # Yang diekspor adalah potongan TERPANJANG per dokumen, sebagai wakil isi yang paling
    # berinformasi, dengan batas per dokumen supaya berkasnya tetap kecil dan dapat dibaca.
    per_dokumen: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        per_dokumen[c["source"]].append(c)
    cadangan: List[Dict[str, Any]] = []
    for src in sorted(per_dokumen):
        teratas = sorted(per_dokumen[src], key=lambda c: len(c["text"]), reverse=True)
        cadangan.extend(teratas[: args.n_cadangan_per_dokumen])
    jalur_cadangan = Path(args.kb_dir) / "fallback_chunks.json"
    jalur_cadangan.write_text(
        json.dumps(cadangan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[6] Cadangan KNF-04: {len(cadangan)} potongan dari {len(per_dokumen)} dokumen "
          f"-> {jalur_cadangan}")

    # 7) Verifikasi: buka ulang koleksi & hitung
    import chromadb
    client = chromadb.PersistentClient(path=args.persist)
    col = client.get_collection(args.collection)
    print(f"\n[7] Verifikasi ChromaDB: koleksi '{args.collection}' berisi {col.count()} chunk.")
    print("    Status: OK" if col.count() == len(chunks) else "    Status: MISMATCH!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
