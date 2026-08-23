from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reingest_kb import _load_manifest, _build_page_docs
from src.rag.knowledge_base import MedicalKnowledgeBase

PDF_DIR = ROOT / "data/knowledge_base/books"
MANIFEST = ROOT / "data/knowledge_base/manifest.csv"

manifest = _load_manifest(MANIFEST)

# Ambil satu dokumen dulu agar eksperimen kecil.
entry = next(iter(manifest.values()))
pdf_path = PDF_DIR / entry.nama_berkas

docs, _, _, _ = _build_page_docs(
    pdf_path,
    entry,
    min_page_chars=100,
)

print(f"Dokumen : {entry.nama_berkas}")
print(f"Halaman : {len(docs)}")

kb = MedicalKnowledgeBase(
    kb_dir=str(ROOT / "data/knowledge_base"),
    persist_dir=str(ROOT / "models/_splitter_test"),
    collection_name="test",
    embed_provider="sentence-transformers",
)

v0 = kb.chunk_documents(
    documents=docs,
    chunk_size=900,
    chunk_overlap=120,
    pemisah_kalimat=False,
    satuan_panjang="karakter",
)

v1 = kb.chunk_documents(
    documents=docs,
    chunk_size=900,
    chunk_overlap=120,
    pemisah_kalimat=True,
    satuan_panjang="karakter",
)

print()
print("=== HASIL ===")
print("V0 chunks:", len(v0))
print("V1 chunks:", len(v1))

beda = sum(
    a["text"] != b["text"]
    for a, b in zip(v0, v1)
)

print("Chunk pada posisi sama yang berbeda:", beda)

print()
print("=== CONTOH PERBEDAAN ===")

for i, (a, b) in enumerate(zip(v0, v1)):
    if a["text"] != b["text"]:
        print(f"\n--- CHUNK {i} ---")
        print("V0:")
        print(a["text"][-250:])
        print("\nV1:")
        print(b["text"][-250:])
        break