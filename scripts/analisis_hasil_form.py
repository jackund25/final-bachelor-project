#!/usr/bin/env python3
"""Replikasi hasil Google Forms evaluasi ahli putaran n=10: statistik dan diagram.

Sumber tunggal: ``results/hasil_form/data_mentah.json``, hasil transkripsi
diagram pada ekspor PDF ringkasan analytics Google Forms (17 halaman, dicetak
31 Agustus 2026, ``results/hasil_form/hasil_form_update.pdf``). PDF itu hanya
memuat DISTRIBUSI MARGINAL per pertanyaan, bukan baris per responden, sehingga:

  * statistik per BUTIR (rerata, simpangan baku, median, modus, kuartil) tetap
    eksak, karena cacah marginal satu butir sudah merupakan seluruh sampel
    butir tersebut;
  * skor SUS per RESPONDEN tidak dapat direkonstruksi, karena pemasangan nilai
    antar-butir untuk responden yang sama tidak terekam pada marginal. Yang
    dihitung di sini adalah rerata SUS (eksak, karena algoritma Brooke linear
    terhadap nilai butir) beserta batas bawah dan batas atas skor individu yang
    masih mungkin terjadi. Simpangan baku SUS antar-responden sengaja TIDAK
    dilaporkan karena tidak teridentifikasi dari data marginal.

Skrip juga menjalankan uji konsistensi terhadap
``results/evaluasi_ahli/ringkasan.json`` (putaran terdahulu, n=4, berbasis CSV
per responden). Bila keempat responden lama benar merupakan himpunan bagian
dari kesepuluh responden sekarang, selisih cacah per butir harus tak negatif
dan berjumlah tepat ``n_responden - 4`` pada setiap butir SUS.

Jalankan:
    conda run -n diabetes-ta python scripts/analisis_hasil_form.py
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUMBER = ROOT / "results/hasil_form/data_mentah.json"
ARSIP_N4 = ROOT / "results/evaluasi_ahli/ringkasan.json"
OUT_DIR = ROOT / "results/hasil_form"
GAMBAR_DIR = OUT_DIR / "gambar"

# Palet Google Forms, disamakan dengan tampilan pada PDF sumber.
HIJAU = "#48A14D"          # batang tunggal
PALET_PIE = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58",
             "#AB47BC", "#00ACC1", "#FF7043"]
PALET_KISI = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58"]
LABEL_KISI = ["1 Sangat Tidak Setuju", "2", "3", "4 Sangat Setuju"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#666666",
    "axes.labelcolor": "#333333",
    "text.color": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "figure.dpi": 150,
})


# --------------------------------------------------------------------------
# Statistik
# --------------------------------------------------------------------------
def bentangkan(dist: dict) -> list[int]:
    """{'3': 4, '4': 3} -> [3,3,3,3,4,4,4]. Marginal satu butir = sampel penuh."""
    nilai: list[int] = []
    for k in sorted(dist, key=int):
        nilai.extend([int(k)] * int(dist[k]))
    return nilai


def kuartil(v: list[int]) -> tuple[float, float]:
    """Kuartil metode median-of-halves (Tukey), konsisten untuk n kecil."""
    s = sorted(v)
    n = len(s)
    bawah = s[: n // 2]
    atas = s[(n + 1) // 2:]
    return st.median(bawah), st.median(atas)


def ringkas_skala(dist: dict, skala: list[int], arah: str) -> dict:
    v = bentangkan(dist)
    n = len(v)
    q1, q3 = kuartil(v)
    ambang = 3 if skala[1] == 4 else 4          # "setuju" pada masing-masing skala
    n_setuju = sum(1 for x in v if x >= ambang)
    hasil = {
        "n": n,
        "rerata": round(st.mean(v), 3),
        "simpangan_baku": round(st.stdev(v), 3) if n > 1 else 0.0,
        "median": st.median(v),
        "modus": st.multimode(v),
        "min": min(v),
        "maks": max(v),
        "q1": q1,
        "q3": q3,
        "ambang_setuju": ambang,
        "n_setuju": n_setuju,
        "persen_setuju": round(100 * n_setuju / n, 1),
        "distribusi": {str(k): int(dist.get(str(k), 0))
                       for k in range(skala[0], skala[1] + 1)},
        "persen": {str(k): round(100 * int(dist.get(str(k), 0)) / n, 1)
                   for k in range(skala[0], skala[1] + 1)},
    }
    if arah == "negatif":
        hasil["tafsir"] = ("butir bernada negatif: skor tinggi berarti penilaian "
                           "yang tidak menguntungkan sistem")
    return hasil


def ringkas_kategorik(opsi: dict, n: int) -> dict:
    return {
        "n": n,
        "cacah": {k: int(v) for k, v in opsi.items()},
        "persen": {k: (round(100 * int(v) / n, 1) if n else 0.0)
                   for k, v in opsi.items()},
        "modus": max(opsi, key=lambda k: opsi[k]) if n else None,
    }


def periksa_cacah(data: dict) -> list[str]:
    """Setiap distribusi harus berjumlah n. Mencegah salah salin dari PDF."""
    galat = []
    for p in data["pertanyaan"]:
        if p["tipe"] == "skala":
            total = sum(int(x) for x in p["distribusi"].values())
            if total != p["n"]:
                galat.append(f"{p['id']}: jumlah {total} != n {p['n']}")
        elif p["tipe"] == "kisi":
            for baris, d in p["baris"].items():
                total = sum(int(x) for x in d.values())
                if total != p["n"]:
                    galat.append(f"{p['id']}/{baris}: jumlah {total} != n {p['n']}")
        elif p["tipe"] == "kategorik":
            total = sum(int(x) for x in p["opsi"].values())
            if total != p["n"]:
                galat.append(f"{p['id']}: jumlah {total} != n {p['n']}")
        elif p["tipe"] == "teks":
            if len(p["jawaban"]) != p["n"]:
                galat.append(f"{p['id']}: {len(p['jawaban'])} teks != n {p['n']}")
    return galat


def hitung_sus(butir: list[dict]) -> dict:
    """Brooke (1996). Ganjil menyumbang (x-1), genap menyumbang (5-x), lalu x2,5.

    Rerata SUS eksak karena transformasinya linear: rerata dari jumlah sama
    dengan jumlah dari rerata. Skor individu tidak eksak, hanya dibatasi.
    """
    sumbangan_rerata, batas_min, batas_maks = [], [], []
    rincian = []
    for p in butir:
        v = bentangkan(p["distribusi"])
        rerata = st.mean(v)
        positif = p["arah"] == "positif"
        if positif:
            sumbangan_rerata.append(rerata - 1)
            batas_min.append(min(v) - 1)
            batas_maks.append(max(v) - 1)
        else:
            sumbangan_rerata.append(5 - rerata)
            batas_min.append(5 - max(v))
            batas_maks.append(5 - min(v))
        rincian.append({
            "butir": p["sus_butir"],
            "teks": p["teks"],
            "arah": p["arah"],
            "rerata": round(rerata, 3),
            "simpangan_baku": round(st.stdev(v), 3),
            "sumbangan_sus": round(2.5 * sumbangan_rerata[-1], 2),
        })
    return {
        "rerata": round(2.5 * sum(sumbangan_rerata), 2),
        "rentang_mungkin_individu": [round(2.5 * sum(batas_min), 2),
                                     round(2.5 * sum(batas_maks), 2)],
        "simpangan_baku_antar_responden": None,
        "catatan": ("Rerata bersifat eksak. Skor per responden dan simpangan "
                    "bakunya tidak teridentifikasi dari distribusi marginal; "
                    "rentang di atas adalah batas terendah dan tertinggi yang "
                    "masih dapat dicapai satu responden pada marginal ini."),
        "acuan_sauro_lewis": {
            "rerata_industri": 68,
            "posisi": None,      # diisi di bawah
            "peringkat_huruf": None,
        },
        "butir": rincian,
    }


def peringkat_sus(skor: float) -> str:
    """Kurva peringkat Sauro & Lewis (2016)."""
    for ambang, huruf in [(84.1, "A+"), (80.8, "A"), (78.9, "A-"),
                          (77.2, "B+"), (74.1, "B"), (72.6, "B-"),
                          (71.1, "C+"), (65.0, "C"), (62.7, "C-"),
                          (51.7, "D")]:
        if skor >= ambang:
            return huruf
    return "F"


def uji_konsistensi_n4(data: dict) -> dict:
    """Selisih marginal putaran ini terhadap 4 responden berbasis CSV pada arsip.

    Bila arsip benar merupakan himpunan bagian, setiap selisih tak boleh negatif
    dan setiap butir SUS harus menyisakan tepat ``n_responden - 4`` tanggapan.
    Jumlah sisa diturunkan dari metadata, bukan ditanam sebagai tetapan, supaya
    uji ini tetap sahih ketika putaran pengumpulan data bertambah.
    """
    if not ARSIP_N4.exists():
        return {"status": "arsip tidak ditemukan"}
    arsip = json.loads(ARSIP_N4.read_text(encoding="utf-8"))
    n_total = int(data["meta"]["n_responden"])
    n_arsip = len(arsip["penilai"])
    n_baru = n_total - n_arsip
    butir_sus = {p["sus_butir"]: p for p in data["pertanyaan"]
                 if p.get("sus_butir")}
    sisa, konsisten = {}, True
    for idx in range(1, 11):
        d = dict(butir_sus[idx]["distribusi"])
        for pen in arsip["penilai"]:
            nilai = str(pen["sus_butir"][idx - 1])
            d[nilai] = int(d.get(nilai, 0)) - 1
        if any(v < 0 for v in d.values()) or sum(d.values()) != n_baru:
            konsisten = False
        sisa[f"butir_{idx}"] = {k: v for k, v in sorted(d.items()) if v}
    return {
        "status": "konsisten" if konsisten else "TIDAK konsisten",
        "arti": (f"Keempat responden pada results/evaluasi_ahli/ringkasan.json "
                 f"merupakan himpunan bagian dari {n_total} responden di sini; "
                 f"sisa di bawah adalah distribusi {n_baru} responden baru."
                 if konsisten else
                 f"Selisih memuat cacah negatif atau tidak berjumlah {n_baru}: "
                 f"arsip n={n_arsip} bukan himpunan bagian dari marginal "
                 f"n={n_total}, atau ada salah salin."),
        "n_responden_arsip": n_arsip,
        "n_responden_baru": n_baru,
        "sisa_distribusi_sus": sisa,
        "sus_arsip_n4": arsip["agregat"]["sus"],
    }


# --------------------------------------------------------------------------
# Diagram
# --------------------------------------------------------------------------
def _judul(ax, teks: str, lebar: int = 62, pad: int = 12):
    """Judul dibungkus manual agar kalimat pertanyaan yang panjang tetap muat."""
    baris, kini = [], ""
    for kata in teks.split():
        if len(kini) + len(kata) + 1 > lebar:
            baris.append(kini)
            kini = kata
        else:
            kini = f"{kini} {kata}".strip()
    baris.append(kini)
    ax.set_title("\n".join(baris), fontsize=10, loc="left", pad=pad)


def gambar_batang(dist: dict, skala: list[int], n: int, teks: str, keluar: Path):
    """Batang hijau vertikal, label 'cacah (persen)' di atas batang."""
    kunci = [str(k) for k in range(skala[0], skala[1] + 1)]
    nilai = [int(dist.get(k, 0)) for k in kunci]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(kunci, nilai, color=HIJAU, width=0.55)
    for i, v in enumerate(nilai):
        ax.text(i, v + max(nilai) * 0.03, f"{v} ({100 * v / n:.1f}%)".replace(".", ","),
                ha="center", va="bottom", fontsize=8,
                color="white" if v > max(nilai) * 0.5 else "#333333")
        if v > max(nilai) * 0.5:                      # label di dalam batang
            ax.texts[-1].set_position((i, v - max(nilai) * 0.08))
            ax.texts[-1].set_va("top")
    ax.set_ylim(0, max(nilai) * 1.18)
    ax.set_yticks(range(0, max(nilai) + 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    _judul(ax, teks)
    ax.text(0, 1.02, f"{n} jawaban", transform=ax.transAxes,
            fontsize=8, color="#777777")
    fig.tight_layout()
    fig.savefig(keluar, bbox_inches="tight")
    plt.close(fig)


def gambar_batang_kategorik(opsi: dict, n: int, teks: str, keluar: Path):
    pakai = {k: v for k, v in opsi.items() if v}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(list(pakai), list(pakai.values()), color=HIJAU, width=0.5)
    puncak = max(pakai.values())
    for i, v in enumerate(pakai.values()):
        ax.text(i, v - puncak * 0.08, f"{v} ({100 * v / n:.0f}%)",
                ha="center", va="top", fontsize=8, color="white")
    ax.set_ylim(0, puncak * 1.15)
    ax.set_yticks(range(0, puncak + 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    _judul(ax, teks)
    ax.text(0, 1.02, f"{n} jawaban", transform=ax.transAxes,
            fontsize=8, color="#777777")
    fig.tight_layout()
    fig.savefig(keluar, bbox_inches="tight")
    plt.close(fig)


def gambar_pie(opsi: dict, n: int, teks: str, keluar: Path):
    """Irisan nol tetap muncul di legenda, meniru perilaku Google Forms."""
    label = list(opsi)
    nilai = [int(v) for v in opsi.values()]
    warna = PALET_PIE[: len(label)]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    irisan = [(v, w) for v, w in zip(nilai, warna) if v]
    ax.pie([v for v, _ in irisan], colors=[w for _, w in irisan],
           startangle=90, counterclock=False,
           autopct=lambda p: f"{p:.1f}%".replace(".", ","),
           textprops={"color": "white", "fontsize": 9})
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="",
                                  color=w, markersize=7, label=lb)
                       for lb, w in zip(label, warna)],
              loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, fontsize=8)
    _judul(ax, teks)
    ax.text(0, 1.02, f"{n} jawaban", transform=ax.transAxes,
            fontsize=8, color="#777777")
    fig.tight_layout()
    fig.savefig(keluar, bbox_inches="tight")
    plt.close(fig)


def gambar_kisi(baris: dict, n: int, teks: str, keluar: Path):
    """Batang berkelompok: satu kelompok per kriteria, satu batang per skor."""
    kriteria = list(baris)
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    lebar = 0.19
    for j, skor in enumerate(["1", "2", "3", "4"]):
        posisi = [i + (j - 1.5) * lebar for i in range(len(kriteria))]
        tinggi = [int(baris[k].get(skor, 0)) for k in kriteria]
        ax.bar(posisi, tinggi, width=lebar, color=PALET_KISI[j],
               label=LABEL_KISI[j])
    ax.set_xticks(range(len(kriteria)))
    ax.set_xticklabels(kriteria, fontsize=8)
    puncak = max(max(int(v) for v in d.values()) for d in baris.values())
    ax.set_ylim(0, puncak + 1)
    ax.set_yticks(range(0, puncak + 2, 2 if puncak >= 4 else 1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    # Legenda diletakkan tepat di atas bidang gambar, judul di atas legenda.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=4, frameon=False, fontsize=8)
    _judul(ax, teks, pad=30)
    fig.tight_layout()
    fig.savefig(keluar, bbox_inches="tight")
    plt.close(fig)


def gambar_ringkas_sus(sus: dict, keluar: Path):
    """Sumbangan tiap butir terhadap skor SUS, maksimum 10 poin per butir."""
    butir = sus["butir"]
    label = [f"B{b['butir']}" for b in butir]
    nilai = [b["sumbangan_sus"] for b in butir]
    warna = ["#4285F4" if b["arah"] == "positif" else "#DB4437" for b in butir]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(label, nilai, color=warna, width=0.6)
    ax.axhline(5.0, color="#999999", linestyle="--", linewidth=1)
    for i, v in enumerate(nilai):
        ax.text(i, v + 0.15, f"{v:.2f}".replace(".", ","),
                ha="center", fontsize=7.5)
    ax.set_ylim(0, 10)
    ax.set_ylabel("sumbangan skor SUS (maks 10)")
    ax.set_title(f"Sumbangan per butir SUS, rerata total "
                 f"{sus['rerata']:.1f}".replace(".", ",") + " dari 100",
                 fontsize=10, loc="left", pad=10)
    ax.legend(handles=[plt.Line2D([], [], marker="s", linestyle="",
                                  color="#4285F4", markersize=8,
                                  label="butir bernada positif"),
                       plt.Line2D([], [], marker="s", linestyle="",
                                  color="#DB4437", markersize=8,
                                  label="butir bernada negatif"),
                       plt.Line2D([], [], linestyle="--", color="#999999",
                                  label="titik tengah butir (5,0)")],
              loc="upper right", frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(keluar, bbox_inches="tight")
    plt.close(fig)


def gambar_ringkas_butir4(stat: dict, data: dict, keluar: Path):
    """Rerata seluruh butir skala 1-4 beserta galat baku, diurutkan menaik."""
    nama = {
        "s1_cgm": "S1 konsistensi prediksi (CGM)",
        "s2_fingerstick": "S2 relevansi dan batas finger-stick",
        "s4_penolakan": "S4 kejelasan penolakan memprediksi",
        "s4_kepastian": "S4 pengendalian kesan kepastian",
        "b11_populasi": "Keterwakilan populasi sumber data",
        "b11_otonomi": "Otonomi dokter dan bias",
        "b11_consent": "Persetujuan dan keamanan data",
        "b11_dasar": "Kejelasan dasar rekomendasi",
        "b11_alur": "Kelayakan alur pada praktik klinis",
    }
    item = sorted(((nama[k], stat["butir"][k]) for k in nama),
                  key=lambda t: t[1]["rerata"])
    label = [t[0] for t in item]
    rerata = [t[1]["rerata"] for t in item]
    galat = [t[1]["simpangan_baku"] / math.sqrt(t[1]["n"]) for t in item]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(label, rerata, xerr=galat, color="#4285F4", height=0.6,
            error_kw={"ecolor": "#555555", "capsize": 3, "linewidth": 1})
    ax.axvline(3.0, color="#DB4437", linestyle="--", linewidth=1)
    ax.text(3.04, len(label) - 0.45, "ambang setuju (3)",
            fontsize=7.5, color="#DB4437")
    for i, v in enumerate(rerata):
        ax.text(v + galat[i] + 0.05, i, f"{v:.2f}".replace(".", ","),
                va="center", fontsize=8)
    ax.set_xlim(1, 4.35)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("rerata skor (skala 1-4), galat = galat baku rerata")
    ax.set_title("Rerata butir skala 1-4, n = 7 penilai", fontsize=10,
                 loc="left", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e8e8e8", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(keluar, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Laporan Markdown
# --------------------------------------------------------------------------
def koma(x) -> str:
    return f"{x}".replace(".", ",")


def tulis_markdown(data: dict, stat: dict, gambar: list[tuple[str, str]]) -> str:
    b = []
    b.append(f"# Hasil {data['meta']['judul']}\n")
    b.append(f"Sumber: {data['meta']['sumber']}.  ")
    b.append(f"Jumlah responden: **{data['meta']['n_responden']}**.\n")
    b.append("> Ekspor PDF Google Forms hanya memuat distribusi marginal per "
             "pertanyaan. Statistik per butir di bawah bersifat eksak; skor SUS "
             "per responden tidak dapat direkonstruksi dari data ini.\n")

    b.append("\n## 1. Profil responden\n")
    for pid in ["profesi", "okupasi", "pernah_cdss", "pernah_cgm"]:
        s = stat["kategorik"][pid]
        teks = next(p["teks"] for p in data["pertanyaan"] if p["id"] == pid)
        b.append(f"\n**{teks}** (n = {s['n']})\n")
        b.append("\n| Kategori | Cacah | Persen |")
        b.append("| --- | ---: | ---: |")
        for k, v in s["cacah"].items():
            b.append(f"| {k} | {v} | {koma(s['persen'][k])}% |")
        b.append("")

    b.append("\n## 2. Butir skala 1-4 (skenario dan penilaian menyeluruh)\n")
    b.append("| Butir | Rerata | SB | Median | Modus | Min-Maks | % skor >= 3 |")
    b.append("| --- | ---: | ---: | ---: | ---: | :---: | ---: |")
    for pid, s in stat["butir"].items():
        teks = next(p["teks"] for p in data["pertanyaan"] if p["id"] == pid)
        potong = teks if len(teks) <= 78 else teks[:75] + "..."
        b.append(f"| {potong} | {koma(s['rerata'])} | {koma(s['simpangan_baku'])} "
                 f"| {koma(s['median'])} | {'/'.join(map(str, s['modus']))} "
                 f"| {s['min']}-{s['maks']} | {koma(s['persen_setuju'])}% |")

    b.append("\n## 3. Kisi penilaian advisory (Skenario 3A-3C)\n")
    b.append("| Skenario | Kriteria | Rerata | SB | Median | Min-Maks | % skor >= 3 |")
    b.append("| --- | --- | ---: | ---: | ---: | :---: | ---: |")
    for kid, baris in stat["kisi"].items():
        judul = kid.replace("kisi_", "").upper()
        for kriteria, s in baris.items():
            b.append(f"| {judul} | {kriteria} | {koma(s['rerata'])} "
                     f"| {koma(s['simpangan_baku'])} | {koma(s['median'])} "
                     f"| {s['min']}-{s['maks']} | {koma(s['persen_setuju'])}% |")

    b.append("\n## 4. Kesesuaian rekomendasi terhadap pedoman\n")
    b.append("| Opsi | 3B hipoglikemia | 3C hiperglikemia |")
    b.append("| --- | ---: | ---: |")
    o3b, o3c = stat["kategorik"]["pedoman_3b"], stat["kategorik"]["pedoman_3c"]
    for k in o3b["cacah"]:
        b.append(f"| {k} | {o3b['cacah'][k]} ({koma(o3b['persen'][k])}%) "
                 f"| {o3c['cacah'][k]} ({koma(o3c['persen'][k])}%) |")

    sus = stat["sus"]
    b.append("\n## 5. System Usability Scale\n")
    b.append(f"- Rerata SUS: **{koma(sus['rerata'])}** dari 100 "
             f"(peringkat {sus['acuan_sauro_lewis']['peringkat_huruf']}, "
             f"{sus['acuan_sauro_lewis']['posisi']} rerata industri 68).")
    b.append(f"- Rentang skor individu yang masih mungkin: "
             f"{koma(sus['rentang_mungkin_individu'][0])} sampai "
             f"{koma(sus['rentang_mungkin_individu'][1])}.")
    b.append(f"- Simpangan baku antar-responden: tidak dilaporkan. {sus['catatan']}\n")
    b.append("\n| # | Butir | Nada | Rerata | SB | Sumbangan SUS (maks 10) |")
    b.append("| ---: | --- | :---: | ---: | ---: | ---: |")
    for r in sus["butir"]:
        potong = r["teks"] if len(r["teks"]) <= 70 else r["teks"][:67] + "..."
        b.append(f"| {r['butir']} | {potong} | {r['arah']} | {koma(r['rerata'])} "
                 f"| {koma(r['simpangan_baku'])} | {koma(r['sumbangan_sus'])} |")

    uji = stat["uji_konsistensi_n4"]
    b.append("\n## 6. Uji konsistensi terhadap putaran n=4\n")
    b.append(f"Status: **{uji['status']}**. {uji['arti']}\n")
    if uji["status"] == "konsisten":
        b.append(f"Rerata SUS putaran n=4 adalah "
                 f"{koma(uji['sus_arsip_n4']['rerata'])}; putaran n=7 "
                 f"{koma(sus['rerata'])}.\n")

    b.append("\n## 7. Masukan terbuka\n")
    for p in data["pertanyaan"]:
        if p["tipe"] == "teks" and p["jawaban"]:
            b.append(f"\n**{p['teks']}** ({p['n']} jawaban)\n")
            for j in p["jawaban"]:
                b.append(f"- {j}")
    kosong = [p["teks"] for p in data["pertanyaan"]
              if p["tipe"] == "teks" and not p["jawaban"]]
    if kosong:
        b.append("\nTanpa jawaban: " + "; ".join(kosong) + "\n")

    b.append("\n## 8. Diagram\n")
    b.append("| Berkas | Isi |")
    b.append("| --- | --- |")
    for berkas, isi in gambar:
        b.append(f"| `gambar/{berkas}` | {isi} |")
    b.append("")
    return "\n".join(b)


# --------------------------------------------------------------------------
def main() -> None:
    data = json.loads(SUMBER.read_text(encoding="utf-8"))
    GAMBAR_DIR.mkdir(parents=True, exist_ok=True)

    galat = periksa_cacah(data)
    if galat:
        raise SystemExit("Transkripsi tidak konsisten:\n  " + "\n  ".join(galat))
    print(f"Verifikasi cacah: seluruh distribusi berjumlah n. "
          f"n responden = {data['meta']['n_responden']}\n")

    stat = {"meta": data["meta"], "butir": {}, "kisi": {}, "kategorik": {}}
    gambar: list[tuple[str, str]] = []
    urut = 1

    for p in data["pertanyaan"]:
        nomor = f"G{urut:02d}"
        if p["tipe"] == "skala":
            s = ringkas_skala(p["distribusi"], p["skala"], p["arah"])
            if p.get("sus_butir"):
                nama = f"{nomor}_{p['id']}.png"
            else:
                stat["butir"][p["id"]] = s
                nama = f"{nomor}_{p['id']}.png"
            gambar_batang(p["distribusi"], p["skala"], p["n"], p["teks"],
                          GAMBAR_DIR / nama)
            gambar.append((nama, p["teks"]))
            urut += 1
        elif p["tipe"] == "kisi":
            stat["kisi"][p["id"]] = {
                k: ringkas_skala(d, p["skala"], p["arah"])
                for k, d in p["baris"].items()
            }
            nama = f"{nomor}_{p['id']}.png"
            gambar_kisi(p["baris"], p["n"], p["teks"], GAMBAR_DIR / nama)
            gambar.append((nama, p["teks"]))
            urut += 1
        elif p["tipe"] == "kategorik":
            stat["kategorik"][p["id"]] = ringkas_kategorik(p["opsi"], p["n"])
            nama = f"{nomor}_{p['id']}.png"
            # Meniru PDF sumber: consent dan okupasi memakai batang, sisanya pie.
            if p["id"] in ("consent", "okupasi"):
                gambar_batang_kategorik(p["opsi"], p["n"], p["teks"],
                                        GAMBAR_DIR / nama)
            else:
                gambar_pie(p["opsi"], p["n"], p["teks"], GAMBAR_DIR / nama)
            gambar.append((nama, p["teks"]))
            urut += 1

    butir_sus = sorted((p for p in data["pertanyaan"] if p.get("sus_butir")),
                       key=lambda p: p["sus_butir"])
    sus = hitung_sus(butir_sus)
    sus["acuan_sauro_lewis"]["peringkat_huruf"] = peringkat_sus(sus["rerata"])
    sus["acuan_sauro_lewis"]["posisi"] = (
        "di atas" if sus["rerata"] > 68 else "di bawah")
    stat["sus"] = sus
    stat["uji_konsistensi_n4"] = uji_konsistensi_n4(data)

    gambar_ringkas_sus(sus, GAMBAR_DIR / "R01_ringkas_sus.png")
    gambar.append(("R01_ringkas_sus.png",
                   "Ringkasan: sumbangan tiap butir terhadap skor SUS"))
    gambar_ringkas_butir4(stat, data, GAMBAR_DIR / "R02_ringkas_butir4.png")
    gambar.append(("R02_ringkas_butir4.png",
                   "Ringkasan: rerata seluruh butir skala 1-4 dengan galat baku"))

    (OUT_DIR / "statistik.json").write_text(
        json.dumps(stat, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT_DIR / "statistik.md").write_text(
        tulis_markdown(data, stat, gambar), encoding="utf-8")

    print(f"SUS rerata           : {sus['rerata']:.2f} "
          f"(peringkat {sus['acuan_sauro_lewis']['peringkat_huruf']})")
    print(f"SUS rentang individu : {sus['rentang_mungkin_individu']}")
    print(f"Uji konsistensi n=4  : {stat['uji_konsistensi_n4']['status']}")
    print(f"\nDiagram tertulis     : {len(gambar)} berkas di {GAMBAR_DIR}")
    print(f"Statistik            : {OUT_DIR / 'statistik.json'}")
    print(f"Laporan              : {OUT_DIR / 'statistik.md'}")


if __name__ == "__main__":
    main()
