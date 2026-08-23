import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import chromadb
from collections import defaultdict
from src.rag.citations import _kandidat_batas_kalimat

ROOT = ROOT_DIR / "models" / "_chunk_strategi"

hasil = {}

for varian in ["v0", "v1"]:
    client = chromadb.PersistentClient(path=str(ROOT / varian))
    col = client.get_collection("diabetes_kb")
    data = col.get(include=["documents", "metadatas"])

    per_kb = defaultdict(lambda: [0, 0])

    for text, meta in zip(data["documents"], data["metadatas"]):
        kb_id = meta.get("kb_id", "?")
        per_kb[kb_id][0] += 1

        teks = (text or "").rstrip()
        if teks and _kandidat_batas_kalimat(teks + " "):
            per_kb[kb_id][1] += 1

    hasil[varian] = per_kb


print("\n=== PERBANDINGAN BOUNDARY V0 vs V1 ===")

semua_kb = sorted(set(hasil["v0"]) | set(hasil["v1"]))

for kb in semua_kb:
    v0_n, v0_ok = hasil["v0"].get(kb, [0, 0])
    v1_n, v1_ok = hasil["v1"].get(kb, [0, 0])

    v0_pct = 100 * v0_ok / max(v0_n, 1)
    v1_pct = 100 * v1_ok / max(v1_n, 1)

    print(
        f"{kb}: "
        f"V0 {v0_n:4d} {v0_pct:5.1f}% | "
        f"V1 {v1_n:4d} {v1_pct:5.1f}% | "
        f"delta {v1_pct - v0_pct:+5.1f} pp"
    )