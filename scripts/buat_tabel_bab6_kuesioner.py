#!/usr/bin/env python3
"""Bangkitkan tabel LaTeX Bab VI untuk evaluasi ahli putaran n=12.

Angka diambil langsung dari ``results/hasil_form/terbaru/statistik.json`` yang
dihasilkan ``scripts/analisis_hasil_form_csv.py``, sehingga tabel pada laporan
tidak pernah disalin tangan dan selalu dapat ditelusuri ke CSV tanggapannya.

Format longtable mengikuti tab3.1/tab3.2: booktabs, kepala berulang, dan
\\LTcapwidth selebar teks.

Jalankan:
    conda run -n diabetes-ta python scripts/buat_tabel_bab6_kuesioner.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMBER = ROOT / "results/hasil_form/terbaru/statistik.json"
TABEL = ROOT / "docs/laporan_TA/TA-STI-template-1.0/tables/bab6"
BS = chr(92)
RS = BS + BS


def ind(x, desimal=3):
    """Angka gaya Indonesia: koma desimal."""
    return ("%.*f" % (desimal, x)).replace(".", ",")


def persen(cacah, n):
    return ind(100.0 * cacah / n, 1) + BS + "%"


def longtable(spec, caption, label, kepala, baris, catatan=None, ncol=None):
    ncol = ncol or len(kepala)
    lanjutan = (BS + "multicolumn{%d}{@{}l}{" % ncol + BS + "small" + BS
                + "emph{Tabel " + BS + "thetable{} (lanjutan)}} " + RS)
    hdr = " & ".join(BS + "textbf{%s}" % k for k in kepala) + " " + RS
    out = [
        BS + "begingroup",
        BS + "setlength{" + BS + "LTcapwidth}{" + BS + "textwidth}",
        BS + "begin{longtable}{%s}" % spec,
        BS + "caption{%s}" % caption,
        BS + "label{%s} " % label + RS,
        BS + "toprule", hdr, BS + "midrule", BS + "endfirsthead", "",
        lanjutan, BS + "toprule", hdr, BS + "midrule", BS + "endhead", "",
        BS + "bottomrule", BS + "endlastfoot", "",
    ]
    out.append((" " + RS + "\n\n").join(" & ".join(b) for b in baris) + " " + RS)
    out.append(BS + "end{longtable}")
    if catatan:
        out += ["", BS + "noindent" + BS + "tabnotestyle " + catatan + BS + "normalsize"]
    out.append(BS + "endgroup")
    return "\n".join(out) + "\n"


def tulis(nama, isi):
    p = TABEL / nama
    p.write_text(isi.replace("\r\n", "\n").replace("\n", "\r\n"), encoding="utf-8")
    print("ditulis:", p.relative_to(ROOT))


def main() -> None:
    d = json.loads(SUMBER.read_text(encoding="utf-8"))
    n = d["n_responden"]
    rl = ">{" + BS + "raggedleft" + BS + "arraybackslash}"
    rr = ">{" + BS + "raggedright" + BS + "arraybackslash}"

    # ---------- Tabel karakteristik responden ----------
    prof = d["profil"]["profesi"]["cacah"]
    cdss = d["profil"]["pengalaman_cdss"]["cacah"]
    cgm = d["profil"]["pengalaman_cgm"]["cacah"]
    urut_prof = ["Dokter Spesialis", "Dokter", "Dokter Muda",
                 "Mahasiswa kedokteran/kesehatan"]
    baris = [[k, str(prof[k]), persen(prof[k], n)] for k in urut_prof if k in prof]
    baris += [
        ["Pernah menggunakan CDSS/aplikasi kesehatan digital",
         str(cdss.get("Sudah", 0)), persen(cdss.get("Sudah", 0), n)],
        ["Belum pernah menggunakan CDSS/aplikasi kesehatan digital",
         str(cdss.get("Belum", 0)), persen(cdss.get("Belum", 0), n)],
        ["Pernah menangani pasien diabetes dengan CGM",
         str(cgm.get("Ya", 0)), persen(cgm.get("Ya", 0), n)],
        ["Jarang menangani pasien diabetes dengan CGM",
         str(cgm.get("Jarang", 0)), persen(cgm.get("Jarang", 0), n)],
        ["Belum pernah menangani pasien diabetes dengan CGM",
         str(cgm.get("Belum Pernah", 0)), persen(cgm.get("Belum Pernah", 0), n)],
    ]
    okupasi = {k: v for k, v in d["profil"]["bidang_okupasi"]["cacah"].items()
               if k not in ("-",)}
    n_okupasi = sum(okupasi.values())
    ket = ", ".join("%d %s" % (v, k) for k, v in
                    sorted(okupasi.items(), key=lambda x: (-x[1], x[0])))
    tulis("tab6.16_karakteristik_responden.tex", longtable(
        "@{}" + rr + "p{8.756cm}" + rl + "p{1.800cm}" + rl + "p{2.600cm}@{}",
        "Karakteristik responden evaluasi ahli.",
        "tab:tab6.16_karakteristik_responden",
        ["Karakteristik", "Cacah", "Persentase"], baris,
        catatan=("Bidang okupasi diisi %d dari %d responden: %s."
                 % (n_okupasi, n, ket))))

    # ---------- Tabel penilaian skenario ----------
    baris = []
    for label, v in d["butir_skala4"].items():
        baris.append([label, ind(v["rerata"]), ind(v["sb"]),
                      ind(v["persen_min_baik"], 1) + BS + "%"])
    tulis("tab6.17_evaluasi_skenario_cdss.tex", longtable(
        "@{}" + rr + "p{7.400cm}" + rl + "p{1.500cm}" + rl + "p{1.500cm}"
        + rl + "p{2.194cm}@{}",
        "Ringkasan penilaian skenario penggunaan CDSS.",
        "tab:tab6.17_evaluasi_skenario_cdss",
        ["Aspek", "Rerata", "SB", "\\% skor $" + BS + "geq$ 3"], baris,
        catatan="Skala 1--4; $n = %d$ responden." % n))

    # ---------- Tabel kesesuaian rekomendasi ----------
    peta = [
        ("Sesuai pedoman dan aman diikuti", "Sesuai pedoman & aman diikuti"),
        ("Sesuai pedoman, tetapi perlu verifikasi dokter",
         "Sesuai pedoman, tapi perlu verifikasi dokter"),
        ("Kurang sesuai, berpotensi berisiko",
         "Kurang sesuai pedoman, berpotensi berisiko"),
        ("Tidak sesuai pedoman", "Tidak sesuai pedoman"),
    ]
    b3, c3 = d["rekomendasi"]["3B"], d["rekomendasi"]["3C"]
    baris = []
    for tampil, kunci in peta:
        cb, cc = b3["cacah"].get(kunci, 0), c3["cacah"].get(kunci, 0)
        baris.append([tampil,
                      "%d (%s)" % (cb, persen(cb, b3["n_menjawab"])),
                      "%d (%s)" % (cc, persen(cc, c3["n_menjawab"]))])
    isi = [
        BS + "begin{table}[H]",
        BS + "centering",
        BS + "caption{Penilaian kesesuaian rekomendasi terhadap pedoman.}",
        BS + "label{tab:tab6.18_kesesuaian_rekomendasi}",
        "    " + BS + "begin{tabularx}{" + BS + "textwidth}{@{}L r r@{}}",
        "    " + BS + "hline",
        "    " + BS + "textbf{Kategori penilaian} & " + BS
        + "textbf{3B Hipoglikemia} & " + BS + "textbf{3C Hiperglikemia} " + RS,
        "    " + BS + "hline",
    ]
    isi += ["    " + " & ".join(b) + " " + RS for b in baris]
    isi += [
        "    " + BS + "hline",
        "    " + BS + "end{tabularx}",
        "",
        "    " + BS + "noindent" + BS + "tabnotestyle $n = %d$ responden pada "
        "masing-masing kondisi." % b3["n_menjawab"] + BS + "normalsize",
        BS + "end{table}",
    ]
    tulis("tab6.18_kesesuaian_rekomendasi.tex", "\n".join(isi) + "\n")

    # ---------- Tabel SUS ----------
    s = d["sus"]
    baris = []
    for i, (label, v) in enumerate(s["per_butir"].items(), start=1):
        r = v["rerata"]
        sumbangan = (r - 1) * 2.5 if v["arah"] == "positif" else (5 - r) * 2.5
        baris.append([str(i), ind(r), v["arah"].capitalize(), ind(sumbangan, 2)])
    catatan = ("Skala butir 1--5. Rerata SUS = %s / 100 (SB %s; median %s; "
               "rentang %s--%s; selang kepercayaan 95\\%% %s--%s). Skor dihitung "
               "per responden dengan algoritma Brooke dari $n = %d$ baris "
               "tanggapan, sehingga sebaran antarresponden teridentifikasi."
               % (ind(s["rerata"], 2), ind(s["sb"], 2), ind(s["median"], 2),
                  ind(s["min"], 1), ind(s["maks"], 1),
                  ind(s["ki95"][0], 2), ind(s["ki95"][1], 2), s["n"]))
    tulis("tab6.19_ringkasan_sus.tex", longtable(
        "@{}" + rr + "p{1.400cm}" + rl + "p{2.200cm}" + rr + "p{3.000cm}"
        + rl + "p{3.400cm}@{}",
        "Ringkasan hasil System Usability Scale pada dua belas responden.",
        "tab:tab6.19_ringkasan_sus",
        ["Butir", "Rerata", "Arah", "Sumbangan SUS"], baris, catatan=catatan))


if __name__ == "__main__":
    main()
