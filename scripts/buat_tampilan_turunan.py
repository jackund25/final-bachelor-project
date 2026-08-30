"""Bangkitkan ulang dua tampilan turunan dari dataset terpadu OhioT1DM.

Latar
-----
``data/raw/ohio_t1dm_merged.csv`` dan ``data/raw/ohio_t1dm_smbg.csv`` dibaca oleh
belasan skrip evaluasi (Bab VI), tetapi tidak ada satu pun jalur kode yang
menulisnya: helper ``_merge_and_write`` pada parser terdefinisi namun tidak
pernah dipanggil. Akibatnya kedua berkas itu tampak seperti artefak tanpa
asal-usul, dan setiap perubahan skema pada parser membuat keduanya melenceng
diam-diam.

Keduanya sebenarnya TURUNAN MURNI dari ``data/raw/ohio_t1dm.csv``:

    merged = baris glucose_source == "CGM"          (166.533 baris)
    smbg   = baris glucose_source == "FINGER_STICK" (4.566 baris)

keduanya tanpa kolom ``glucose_source`` dan ``dataset_split``. Skrip ini
menyatakan derivasi tersebut secara eksplisit sehingga skema ketiga berkas
selalu bergerak bersama.

Pemakaian
---------
Dijalankan tepat setelah ``process_ohio_dataset``::

    python scripts/buat_tampilan_turunan.py

Mode verifikasi membandingkan hasil terhadap berkas yang ada tanpa menulis
apa pun. Dipakai untuk membuktikan derivasinya benar SEBELUM skema parser
diubah; keluar dengan kode tidak nol bila ada yang tidak cocok::

    python scripts/buat_tampilan_turunan.py --verifikasi
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SUMBER = ROOT / "data/raw/ohio_t1dm.csv"

# Kolom yang hanya ada pada berkas terpadu. Keduanya justru yang MEMBEDAKAN
# kedua tampilan, sehingga tidak ikut dibawa ke dalam tampilan itu sendiri.
KOLOM_PEMBEDA = ["glucose_source", "dataset_split"]

TURUNAN = {
    "data/raw/ohio_t1dm_merged.csv": "CGM",
    "data/raw/ohio_t1dm_smbg.csv": "FINGER_STICK",
}


def bangun(uni: pd.DataFrame, sumber: str) -> pd.DataFrame:
    return (
        uni[uni["glucose_source"] == sumber]
        .drop(columns=KOLOM_PEMBEDA)
        .reset_index(drop=True)
    )


def sebagai_bita(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False, lineterminator="\r\n")
    return buf.getvalue().encode("utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Bangkitkan tampilan turunan OhioT1DM.")
    ap.add_argument(
        "--verifikasi",
        action="store_true",
        help="bandingkan terhadap berkas yang ada, jangan tulis apa pun",
    )
    args = ap.parse_args()

    # dtype=str: menjaga representasi angka persis seperti pada berkas sumber,
    # sehingga tampilan turunan tidak diam-diam membulatkan atau menambah desimal.
    uni = pd.read_csv(SUMBER, dtype=str, keep_default_na=False)
    print(f"sumber: {SUMBER.relative_to(ROOT)} ({len(uni):,} baris)")

    galat = 0
    for relatif, sumber in TURUNAN.items():
        path = ROOT / relatif
        df = bangun(uni, sumber)
        bita = sebagai_bita(df)

        if args.verifikasi:
            if not path.exists():
                print(f"  {relatif}: TIDAK ADA")
                galat += 1
                continue
            sama = bita == path.read_bytes()
            print(f"  {relatif}: {len(df):,} baris — {'identik' if sama else 'BERBEDA'}")
            galat += 0 if sama else 1
        else:
            path.write_bytes(bita)
            print(f"  -> {relatif} ({len(df):,} baris)")

    if args.verifikasi:
        raise SystemExit(galat)


if __name__ == "__main__":
    main()
