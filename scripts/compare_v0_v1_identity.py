import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import chromadb

BASE = ROOT / "models" / "_chunk_strategi"

for kb_id in ["KB-01", "KB-02", "KB-03", "KB-04", "KB-05",
              "KB-06", "KB-07", "KB-08", "KB-09", "KB-10",
              "KB-11", "KB-12"]:

    hasil = {}

    for varian in ["v0", "v1"]:
        col = chromadb.PersistentClient(
            path=str(BASE / varian)
        ).get_collection("diabetes_kb")

        data = col.get(
            where={"kb_id": kb_id},
            include=["documents"]
        )

        hasil[varian] = data["documents"]

    v0 = hasil["v0"]
    v1 = hasil["v1"]

    sama = sum(
        a == b for a, b in zip(v0, v1)
    )

    dibandingkan = min(len(v0), len(v1))

    print(
        f"{kb_id}: "
        f"V0={len(v0):4d} "
        f"V1={len(v1):4d} "
        f"posisi berbeda={dibandingkan - sama:4d}"
    )