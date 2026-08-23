"""T2.2 — agreement dua annotator manusia.

Membandingkan dua anotasi independen Putaran 2.
CSV asli tidak dimodifikasi; variasi ejaan label dinormalisasi hanya
untuk keperluan analisis.

Output:
- raw agreement
- Cohen's kappa
- confusion matrix
- daftar disagreement
"""

from __future__ import annotations

import csv
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]

PENILAI_1 = ROOT / "evaluation/verifikasi_relevansi_putaran2_penilai1.csv"
PENILAI_2 = ROOT / "evaluation/verifikasi_relevansi_putaran2_penilai2.csv"

LABEL = ["hipoglikemia", "hiperglikemia", "normal", "lain"]


def baca_csv(path: Path) -> dict[int, str]:
    """Baca data anotasi setelah menemukan header sebenarnya."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    header = "id,penilaian_manusia,teks_potongan"
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip() == header)
    except StopIteration:
        raise RuntimeError(f"Header tidak ditemukan: {path}")

    rows = csv.DictReader(lines[start:])

    hasil = {}
    for row in rows:
        raw_id = (row.get("id") or "").strip()
        if not raw_id.isdigit():
            continue

        item_id = int(raw_id)
        raw_label = (row.get("penilaian_manusia") or "").strip()

        hasil[item_id] = normalisasi_label(raw_label)

    return hasil


def normalisasi_label(label: str) -> str:
    """Normalisasi variasi ejaan label annotator."""
    x = label.strip().lower()

    if x in {
        "hipoglikemia",
        "hipoglikemi",
    }:
        return "hipoglikemia"

    if x in {
        "hiperglikemia",
        "hiperglikemi",
        "hiperglikemik",
    }:
        return "hiperglikemia"

    if x in {
        "normal",
    }:
        return "normal"

    if x in {
        "lain",
        "lainnya",
    }:
        return "lain"

    raise ValueError(f"Label tidak dikenali: {label!r}")


def main() -> None:
    a = baca_csv(PENILAI_1)
    b = baca_csv(PENILAI_2)

    ids_a = set(a)
    ids_b = set(b)

    if ids_a != ids_b:
        raise SystemExit(
            "ID kedua penilai tidak identik.\n"
            f"Hanya Penilai 1: {sorted(ids_a - ids_b)}\n"
            f"Hanya Penilai 2: {sorted(ids_b - ids_a)}"
        )

    ids = sorted(ids_a)

    y1 = [a[i] for i in ids]
    y2 = [b[i] for i in ids]

    agreement = accuracy_score(y1, y2)
    kappa = cohen_kappa_score(y1, y2, labels=LABEL)
    cm = confusion_matrix(y1, y2, labels=LABEL)

    print("\n=== T2.2 HUMAN-HUMAN AGREEMENT ===")
    print(f"Jumlah item          : {len(ids)}")
    print(f"Raw agreement        : {agreement * 100:.1f}%")
    print(f"Cohen's kappa        : {kappa:.3f}")

    print("\nConfusion matrix")
    print("Baris = Penilai 1 | Kolom = Penilai 2")
    print("             " + " ".join(f"{x:>15}" for x in LABEL))

    for label, row in zip(LABEL, cm):
        print(f"{label:<12}" + " ".join(f"{x:>15}" for x in row))

    beda = [
        (i, a[i], b[i])
        for i in ids
        if a[i] != b[i]
    ]

    print(f"\nJumlah disagreement : {len(beda)}")

    if beda:
        print("\n=== DISAGREEMENT ===")
        for item_id, l1, l2 in beda:
            print(
                f"ID {item_id:>2}: "
                f"Penilai 1={l1:<15} "
                f"Penilai 2={l2:<15}"
            )


if __name__ == "__main__":
    main()