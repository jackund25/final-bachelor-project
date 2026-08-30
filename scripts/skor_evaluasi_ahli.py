"""Penyekoran instrumen evaluasi ahli (Bab VI, Subbab 6.13).

Membaca berkas tanggapan Google Forms apa adanya dan menghasilkan seluruh angka
yang dilaporkan pada naskah: skor SUS per penilai (algoritma Brooke 1996),
median dan rentang butir skala empat titik, cacah butir kategoris, serta
penanda straight-lining.

Jalankan:
    conda run -n diabetes-ta python scripts/skor_evaluasi_ahli.py
"""

import csv
import io
import json
import os
import statistics as st

CSV = os.path.join(
    "docs",
    "arsip",
    "Form Penggunaan dan Evaluasi Sistem Diabetes Clinical Decision Support "
    " (Jawaban) - Form Responses 3.csv",
)
KELUARAN = os.path.join("results", "evaluasi_ahli", "ringkasan.json")

# Indeks kolom pada berkas tanggapan. Dipatok karena judul kolomnya memuat
# karakter en dash dan spasi ganda yang tidak stabil antar-ekspor.
KOL = {
    "profesi": 4,
    "okupasi": 5,
    "pernah_cdss": 6,
    "pernah_cgm": 8,
    "s1_cgm": 9,
    "s2_fingerstick": 10,
    "s3a": [11, 12, 13, 14],
    "s3b": [15, 16, 17, 18],
    "s3b_pedoman": 19,
    "s3c": [20, 21, 22, 23],
    "s3c_pedoman": 24,
    "s4_penolakan": 26,
    "s4_kepastian": 27,
    "sus": list(range(29, 39)),  # butir 1..10 berurutan
    "b11_populasi": 39,
    "b11_gap": 40,
    "b11_otonomi": 41,
    "b11_consent": 42,
    "b11_dasar": 43,
    "b11_alur": 44,
    "masukan": 45,
}
KRITERIA = ["Relevansi", "Kejelasan", "Kecukupan konteks", "Batasan keputusan"]


def angka(sel):
    """'4 Sangat Setuju' -> 4 ; '3' -> 3 ; '' -> None."""
    sel = (sel or "").strip()
    if not sel:
        return None
    return int(sel.split()[0])


def sus(nilai):
    """Algoritma Brooke: ganjil n-1, genap 5-n, dijumlahkan, dikali 2,5."""
    assert len(nilai) == 10 and all(v is not None for v in nilai)
    mentah = sum(
        (v - 1) if (i % 2 == 0) else (5 - v) for i, v in enumerate(nilai)
    )
    return mentah * 2.5


def rangkum(vals):
    v = [x for x in vals if x is not None]
    return {"n": len(v), "median": st.median(v), "min": min(v), "maks": max(v)}


def main():
    rows = list(csv.reader(io.open(CSV, encoding="utf-8")))
    resp = rows[1:]
    print(f"Penilai: {len(resp)}\n")

    hasil = {"n_penilai": len(resp), "penilai": [], "agregat": {}}

    semua_sus = []
    for k, r in enumerate(resp, start=1):
        nilai_sus = [angka(r[i]) for i in KOL["sus"]]
        skor = sus(nilai_sus)
        semua_sus.append(skor)
        seragam = len(set(nilai_sus)) == 1
        hasil["penilai"].append(
            {
                "kode": f"P{k}",
                "profesi": r[KOL["profesi"]].strip(),
                "okupasi": r[KOL["okupasi"]].strip(),
                "pernah_cdss": r[KOL["pernah_cdss"]].strip(),
                "pernah_cgm": r[KOL["pernah_cgm"]].strip(),
                "sus": skor,
                "sus_butir": nilai_sus,
                "sus_seragam": seragam,
                "masukan": r[KOL["masukan"]].strip(),
                "gap_populasi": r[KOL["b11_gap"]].strip(),
            }
        )
        tanda = "  <-- seluruh butir bernilai sama (straight-lining)" if seragam else ""
        print(f"P{k}  {r[KOL['profesi']].strip():18s} SUS = {skor:5.1f}{tanda}")

    print(f"\nRerata SUS = {st.mean(semua_sus):.1f}   "
          f"(rentang {min(semua_sus):.1f}-{max(semua_sus):.1f})")
    hasil["agregat"]["sus"] = {
        "rerata": round(st.mean(semua_sus), 2),
        "min": min(semua_sus),
        "maks": max(semua_sus),
        "per_penilai": semua_sus,
    }

    print("\n--- Butir skala 1-4 (median [min-maks]) ---")
    butir = {
        "Skenario 1, konsistensi prediksi CGM": [KOL["s1_cgm"]],
        "Skenario 2, prediksi dan batas data finger-stick": [KOL["s2_fingerstick"]],
        "Skenario 4, kejelasan penolakan memprediksi": [KOL["s4_penolakan"]],
        "Skenario 4, pengendalian kesan kepastian": [KOL["s4_kepastian"]],
        "Keterwakilan populasi sumber data": [KOL["b11_populasi"]],
        "Otonomi dokter dan bias": [KOL["b11_otonomi"]],
        "Persetujuan dan keamanan data": [KOL["b11_consent"]],
        "Kejelasan dasar rekomendasi": [KOL["b11_dasar"]],
        "Kelayakan alur pada praktik klinis": [KOL["b11_alur"]],
    }
    hasil["agregat"]["butir"] = {}
    for nama, kols in butir.items():
        vals = [angka(r[c]) for r in resp for c in kols]
        s = rangkum(vals)
        hasil["agregat"]["butir"][nama] = s
        print(f"  {nama:52s} {s['median']:.1f} [{s['min']}-{s['maks']}]")

    print("\n--- Kisi empat kriteria per skenario advisory ---")
    hasil["agregat"]["kisi"] = {}
    for sk in ["s3a", "s3b", "s3c"]:
        hasil["agregat"]["kisi"][sk] = {}
        baris = []
        for j, krit in enumerate(KRITERIA):
            vals = [angka(r[KOL[sk][j]]) for r in resp]
            s = rangkum(vals)
            hasil["agregat"]["kisi"][sk][krit] = s
            baris.append(f"{krit} {s['median']:.1f}[{s['min']}-{s['maks']}]")
        print(f"  {sk.upper():4s} " + "  ".join(baris))

    print("\n--- Butir kategoris kesesuaian pedoman ---")
    hasil["agregat"]["pedoman"] = {}
    for sk, kol in [("3B (hipoglikemia)", KOL["s3b_pedoman"]),
                    ("3C (hiperglikemia)", KOL["s3c_pedoman"])]:
        cacah = {}
        for r in resp:
            v = r[kol].strip()
            cacah[v] = cacah.get(v, 0) + 1
        hasil["agregat"]["pedoman"][sk] = cacah
        print(f"  {sk}:")
        for v, n in sorted(cacah.items(), key=lambda x: -x[1]):
            print(f"      {n} x  {v}")

    os.makedirs(os.path.dirname(KELUARAN), exist_ok=True)
    with io.open(KELUARAN, "w", encoding="utf-8") as f:
        json.dump(hasil, f, ensure_ascii=False, indent=2)
    print(f"\nRingkasan ditulis ke {KELUARAN}")


if __name__ == "__main__":
    main()
