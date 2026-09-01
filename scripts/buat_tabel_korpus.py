#!/usr/bin/env python3
"""Bangkitkan tabel korpus pedoman klinis untuk lampiran laporan.

Sumber tunggal: ``data/knowledge_base/manifest.csv``, yaitu manifes yang sama
yang dipakai proses pengindeksan. Dengan begitu daftar dokumen pada laporan
tidak pernah disalin tangan dan selalu sama dengan korpus yang benar-benar
diindeks.

Jalankan:
    conda run -n diabetes-ta python scripts/buat_tabel_korpus.py
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMBER = ROOT / "data/knowledge_base/manifest.csv"
KELUAR = (ROOT / "docs/laporan_TA/TA-STI-template-1.0/tables"
          / "tabel_D1_korpus.tex")
BS = chr(92)
RS = BS + BS


def rapikan(s: str) -> str:
    """Amankan karakter yang bermakna khusus pada LaTeX."""
    for a, b in (("&", BS + "&"), ("%", BS + "%"), ("#", BS + "#"),
                 ("_", BS + "_")):
        s = s.replace(a, b)
    return s


def main() -> None:
    with open(SUMBER, encoding="utf-8") as f:
        baris = list(csv.DictReader(f))

    rr = ">{" + BS + "raggedright" + BS + "arraybackslash}"
    rl = ">{" + BS + "raggedleft" + BS + "arraybackslash}"
    spec = ("@{}" + rr + "p{1.400cm}" + rr + "p{2.350cm}" + rl + "p{1.000cm}"
            + rr + "p{6.525cm}" + rl + "p{1.600cm}@{}")

    kepala = ["Kode", "Lembaga", "Tahun", "Judul dokumen", "Halaman"]
    hdr = " & ".join(BS + "textbf{%s}" % k for k in kepala) + " " + RS
    lanjutan = (BS + "multicolumn{5}{@{}l}{" + BS + "small" + BS
                + "emph{Tabel " + BS + "thetable{} (lanjutan)}} " + RS)

    isi = []
    for x in baris:
        isi.append(" & ".join([
            rapikan(x["kb_id"]),
            rapikan(x["lembaga"]),
            rapikan(x["tahun"]),
            rapikan(x["judul_lengkap"]),
            rapikan(x["hlm_total"]),
        ]))

    total_hlm = sum(int(x["hlm_total"]) for x in baris)

    out = [
        BS + "begingroup",
        BS + "setlength{" + BS + "LTcapwidth}{" + BS + "textwidth}",
        BS + "small",
        BS + "captionsetup{font=normalsize}",
        BS + "setlength{" + BS + "tabcolsep}{4pt}",
        BS + "begin{longtable}{" + spec + "}",
        BS + "caption{Dokumen pedoman klinis penyusun basis pengetahuan.}",
        BS + "label{tab:tabel_D1_korpus} " + RS,
        BS + "toprule", hdr, BS + "midrule", BS + "endfirsthead", "",
        lanjutan, BS + "toprule", hdr, BS + "midrule", BS + "endhead", "",
        BS + "bottomrule", BS + "endlastfoot", "",
        (" " + RS + "\n\n").join(isi) + " " + RS,
        BS + "end{longtable}",
        "",
        BS + "noindent" + BS + "tabnotestyle Seluruhnya berjumlah %d dokumen "
        "dengan total %d halaman cetak. Kolom halaman menyatakan jumlah halaman "
        "dokumen sumber, bukan jumlah potongan hasil pemenggalan."
        % (len(baris), total_hlm) + BS + "normalsize",
        BS + "endgroup",
    ]

    KELUAR.write_text(
        ("\n".join(out) + "\n").replace("\n", "\r\n"), encoding="utf-8")
    print("ditulis:", KELUAR.relative_to(ROOT))
    print("%d dokumen, %d halaman" % (len(baris), total_hlm))


if __name__ == "__main__":
    main()
