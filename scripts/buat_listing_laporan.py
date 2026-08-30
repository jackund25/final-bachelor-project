"""Ekstrak potongan source code menjadi berkas kutipan untuk \\lstinputlisting.

Dijalankan dari akar repositori:
    conda run -n diabetes-ta python scripts/buat_listing_laporan.py

Prinsip: kutipan di Bab V dihasilkan ULANG dari source code, tidak pernah
disalin-tempel, sehingga kutipan di laporan dapat diperbarui bila kodenya berubah
(Aturan 25). Setiap potongan dicatat rentang barisnya agar dapat diaudit balik.

PERINGATAN: rentang baris di bawah menjadi usang bila berkas sumbernya disunting.
Setelah menjalankan skrip ini, PERIKSA keluarannya di
docs/laporan_TA/TA-STI-template-1.0/listings/ sebelum kompilasi.
"""

import io
import os

REPO = os.environ.get("TA_REPO") or os.getcwd()
OUT = os.path.join(REPO, "docs", "laporan_TA", "TA-STI-template-1.0", "listings")

ELIPSIS = "# (docstring dipangkas untuk keringkasan)"

# nama berkas keluaran -> (berkas sumber, [(awal, akhir), ...], sisipan elipsis)
# Rentang baris bersifat 1-indexed dan inklusif.
# Kunci sisipan adalah INDEKS RENTANG; elipsis disisipkan SESUDAH rentang itu.
SPEC = {
    "bm25-index.tex": (
        "src/rag/retriever.py",
        [(260, 260), (278, 294)],
        {0: ELIPSIS},
    ),
    "bm25-degradasi.tex": (
        "src/rag/retriever.py",
        [(295, 309)],
        {},
    ),
    "resolusi-sitasi.tex": (
        "src/rag/citations.py",
        [(24, 55)],
        {},
    ),
    "penjagaan-riwayat.tex": (
        "backend/services/prediction_service.py",
        [(398, 424)],
        {},
    ),
}


def dedent(baris):
    """Buang indentasi bersama supaya potongan tidak terdorong ke kanan."""
    isi = [b for b in baris if b.strip()]
    if not isi:
        return baris
    n = min(len(b) - len(b.lstrip()) for b in isi)
    return [b[n:] if b.strip() else b for b in baris]


def main():
    if not os.path.isdir(OUT):
        raise SystemExit("Direktori keluaran tidak ada: " + OUT)

    for nama, (sumber, rentang, sisipan) in SPEC.items():
        path = os.path.join(REPO, sumber)
        with io.open(path, "r", encoding="utf-8") as f:
            src = f.read().splitlines()

        keluaran = []
        for i, (a, b) in enumerate(rentang):
            keluaran.extend(src[a - 1:b])
            if i in sisipan:
                # Elipsis memakai indentasi baris berikutnya agar tetap sah sebagai
                # Python dan jelas terbaca sebagai bagian yang sengaja dihilangkan.
                lanjut = src[rentang[i + 1][0] - 1] if i + 1 < len(rentang) else ""
                spasi = " " * (len(lanjut) - len(lanjut.lstrip()))
                keluaran.append(spasi + sisipan[i])

        keluaran = dedent(keluaran)

        tujuan = os.path.join(OUT, nama)
        with io.open(tujuan, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(keluaran) + "\n")

        jejak = ", ".join("%d-%d" % r for r in rentang)
        print("%-24s <- %s baris %s  (%d baris)"
              % (nama, sumber, jejak, len(keluaran)))


if __name__ == "__main__":
    main()
