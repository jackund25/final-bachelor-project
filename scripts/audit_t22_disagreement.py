import csv
from pathlib import Path

P = Path("evaluation")

FILES = [
    P / "verifikasi_relevansi_putaran2_penilai1.csv",
    P / "verifikasi_relevansi_putaran2_penilai2.csv",
]


def normalize(label):
    x = label.strip().lower()

    if x in {"hipoglikemi", "hipoglikemia"}:
        return "hipoglikemia"

    if x in {"hiperglikemi", "hiperglikemia", "hiperglikemik"}:
        return "hiperglikemia"

    if x == "normal":
        return "normal"

    if x in {"lain", "lainnya"}:
        return "lain"

    return x


def load(path):
    lines = path.read_text(encoding="utf-8-sig").splitlines()

    header = "id,penilaian_manusia,teks_potongan"
    start = next(
        i for i, line in enumerate(lines)
        if line.strip() == header
    )

    rows = csv.DictReader(lines[start:])

    return {
        int(row["id"]): row
        for row in rows
        if row.get("id", "").strip().isdigit()
    }


a = load(FILES[0])
b = load(FILES[1])

print(f"Penilai 1: {len(a)} item")
print(f"Penilai 2: {len(b)} item")

print("\n=== 12 DISAGREEMENT ===")

jumlah = 0

for item_id in sorted(a):
    label_a = normalize(a[item_id]["penilaian_manusia"])
    label_b = normalize(b[item_id]["penilaian_manusia"])

    if label_a != label_b:
        jumlah += 1

        print(f"\n=== ID {item_id} ===")
        print(f"Penilai 1 : {label_a}")
        print(f"Penilai 2 : {label_b}")
        print("TEKS:")
        print(a[item_id]["teks_potongan"])

print(f"\nTotal disagreement: {jumlah}")
