#!/usr/bin/env python3
"""Hasilkan Gambar VI.9 (kepatuhan generation) dan VI.10 (kestabilan RAGAS).

Sumber:
- ``results/ragas/uji_kepatuhan_<model>.json`` — lima aturan keluaran diperiksa
  pada dua kasus.
- ``results/ragas/kestabilan.json`` — dua jalan berulang atas metrik RAGAS.

Kedua berkas memakai ``gemini-3.5-flash-lite``, yaitu model yang sama dengan
``config.yaml`` (rag.llm.model), sehingga tidak ada selisih provenans
seperti pada gambar-gambar sebelumnya. Skrip memeriksanya dan berhenti bila
model berbeda.

DUA HAL YANG SENGAJA TIDAK DIKLAIM GAMBAR INI.

Gambar VI.9 tidak menyatakan sistem "100% aman". Yang digambarkan adalah lima
aturan yang DIPERIKSA, dipatuhi pada DUA kasus; aturan lain di luar daftar itu
tidak diuji, dan dua kasus tidak menyanggah adanya pelanggaran pada kasus lain.
Judul dan catatan kaki dijaga agar tidak terbaca sebagai jaminan.

Gambar VI.10 tidak menyatakan "RAGAS tidak stabil". Yang terbaca adalah
answer_relevancy bergerak antar-jalan sedangkan context_precision dan
context_recall tidak bergerak sama sekali. Berkas sumbernya sendiri menjelaskan
asimetri itu: answer_relevancy bergantung pada jawaban yang dibangkitkan ulang,
sehingga variasinya bercampur antara pembangkitan dan penjurian, sedangkan dua
metrik lainnya tidak memakai jawaban sama sekali. Faithfulness ikut ditampilkan
sebagai acuan supaya answer_relevancy tidak tampak sebagai satu-satunya metrik
yang bergerak.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_bab6_generasi.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAGAS = ROOT / "results/ragas"
OUT_DIR = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/bab6"

# (label aturan, fungsi pemeriksa satu kasus)
ATURAN = [
    ("Penanda [S1]/[S2] sesuai blok sumber",
     lambda k: bool(k.get("penanda_valid"))),
    ("Tidak menulis nomor halaman/bab/tabel sendiri",
     lambda k: not k.get("nomor_halaman_ditulis")),
    ("Keluaran berbentuk langkah aksi",
     lambda k: bool(k.get("tersusun"))),
    ("Disclaimer muncul",
     lambda k: bool(k.get("disclaimer_sendiri"))),
    ("Bahasa Indonesia penuh",
     lambda k: not k.get("kata_inggris")),
]

HIJAU, MERAH, ABU = "#2ecc71", "#e74c3c", "#95a5a6"
WARNA_METRIK = {"answer_relevancy": "#e67e22", "faithfulness": "#9b59b6",
                "context_precision": "#3498db", "context_recall": "#2ecc71"}


def muat(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Berkas hasil tidak ditemukan: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


def model_produksi() -> str:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return ((cfg.get("rag") or {}).get("llm") or {}).get("model", "")


def baris_generation_safety() -> str:
    """Ringkasan pemeriksaan terpisah pada 6 kasus, sebagai bukti pendamping.

    Percobaan ini menguji hal yang BERBEDA dari lima aturan di atas (keterlacakan
    angka klinis dan arah tindakan), pada kasus yang berbeda pula, sehingga
    dilaporkan sebagai baris terpisah dan bukan digabung menjadi satu persentase.
    """
    p = ROOT / "results/eval_prediksi/generation_safety.json"
    if not p.exists():
        return ""
    r = json.loads(p.read_text(encoding="utf-8"))["ringkasan"]
    return (f"Pemeriksaan terpisah pada {r['n_kasus']} kasus: "
            f"{r['total_angka_klinis']} angka klinis, "
            f"{r['total_angka_tak_tertelusur']} tak tertelusur, "
            f"{r['kasus_dengan_tindakan_salah_arah']} tindakan salah arah.")


def gambar_kepatuhan(data: dict, out: Path):
    kasus = data["hasil"]
    n_baris, n_kolom = len(ATURAN), len(kasus)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for i, (_, cek) in enumerate(ATURAN):
        for j, k in enumerate(kasus):
            patuh = cek(k)
            y = n_baris - 1 - i
            ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=HIJAU if patuh else MERAH,
                                       alpha=0.22, edgecolor="white", lw=2))
            ax.text(j + 0.5, y + 0.5, "✓" if patuh else "✗",
                    ha="center", va="center", fontsize=19, fontweight="bold",
                    color=HIJAU if patuh else MERAH)

    ax.set_xlim(0, n_kolom)
    ax.set_ylim(0, n_baris)
    ax.set_xticks([j + 0.5 for j in range(n_kolom)])
    ax.set_xticklabels([k["id"] for k in kasus], fontsize=11, fontweight="bold")
    ax.set_yticks([n_baris - 0.5 - i for i in range(n_baris)])
    ax.set_yticklabels([lab for lab, _ in ATURAN], fontsize=10)
    ax.tick_params(length=0)
    for sisi in ax.spines.values():
        sisi.set_visible(False)

    ax.set_title("Kepatuhan keluaran terhadap aturan yang diperiksa",
                 fontsize=12.5, fontweight="bold", pad=14)
    # Catatan kaki menahan gambar ini dibaca sebagai jaminan keamanan.
    catatan = (f"Seluruh aturan yang diperiksa dipatuhi pada {len(kasus)} kasus pengujian "
               f"({data['model']}). Aturan di luar daftar ini tidak diuji.")
    pendamping = baris_generation_safety()
    ax.text(0.5, -0.16, catatan + ("\n" + pendamping if pendamping else ""),
            transform=ax.transAxes, ha="center", fontsize=9, color="#555555")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return [(lab, [cek(k) for k in kasus]) for lab, cek in ATURAN]


def gambar_kestabilan(data: dict, out: Path):
    ar = data["hasil"]["answer_relevancy"]
    per = ar["per_kasus"]

    fig, (kiri, kanan) = plt.subplots(
        1, 2, figsize=(12.0, 5.4), gridspec_kw={"width_ratios": [1.55, 1]}
    )

    # --- Panel kiri: dumbbell answer_relevancy per kasus ---------------------
    ids = [p["id"] for p in per]
    j1 = np.array([p["jalan1"] for p in per])
    j2 = np.array([p["jalan2"] for p in per])
    y = np.arange(len(per))[::-1]

    for yi, a, b in zip(y, j1, j2):
        kiri.plot([a, b], [yi, yi], color="#bdc3c7", lw=2.2, zorder=1)
    kiri.scatter(j1, y, s=62, color="#3498db", zorder=2, label="Jalan 1")
    kiri.scatter(j2, y, s=62, color="#e67e22", zorder=2, label="Jalan 2")

    kiri.set_yticks(y)
    kiri.set_yticklabels(ids, fontsize=10)
    kiri.set_xlim(-0.05, 1.05)
    kiri.set_xlabel("answer_relevancy", fontsize=10.5)
    kiri.set_title("(a) answer_relevancy per kasus, dua jalan",
                   fontsize=11, loc="left", fontweight="bold")
    kiri.grid(axis="x", alpha=0.25, ls=":")
    kiri.set_axisbelow(True)
    kiri.legend(fontsize=9, loc="lower left", framealpha=0.92)

    # --- Panel kanan: selisih absolut median per metrik ----------------------
    metrik = []
    for nama in ["answer_relevancy", "context_precision", "context_recall"]:
        h = data["hasil"][nama]
        metrik.append((nama, h["selisih_absolut_median"], h["n_dibandingkan"],
                       h.get("n_tidak_bergerak")))
    acuan = data.get("acuan_faithfulness", {})
    if "median" in acuan:
        metrik.append(("faithfulness", acuan["median"], None, None))

    ypos = np.arange(len(metrik))[::-1]
    nilai = [m[1] for m in metrik]
    batang = kanan.barh(ypos, nilai, height=0.55,
                        color=[WARNA_METRIK.get(m[0], ABU) for m in metrik], alpha=0.85)
    kanan.bar_label(batang, fmt="%.4f", padding=4, fontsize=9.5, fontweight="bold")

    label_y = []
    for nama, _, n, diam in metrik:
        if nama == "faithfulness":
            label_y.append("faithfulness\n(acuan percobaan lain)")
        elif diam is not None and n is not None and diam == n:
            label_y.append(f"{nama}\n({diam}/{n} kasus tidak berubah)")
        else:
            label_y.append(f"{nama}\n(n={n})")

    kanan.set_yticks(ypos)
    kanan.set_yticklabels(label_y, fontsize=9.5)
    kanan.set_xlabel("Selisih absolut median antar-jalan", fontsize=10.5)
    kanan.set_xlim(0, max(nilai) * 1.35)
    kanan.set_title("(b) Pergerakan antar-jalan menurut metrik",
                    fontsize=11, loc="left", fontweight="bold")
    kanan.grid(axis="x", alpha=0.25, ls=":")
    kanan.set_axisbelow(True)

    fig.suptitle("Kestabilan metrik RAGAS pada dua jalan berulang",
                 fontsize=12.5, fontweight="bold")
    # Batas tafsir diambil dari berkas sumber, bukan dirumuskan ulang di sini.
    fig.text(0.5, -0.015, data.get("batas_tafsir", ""), ha="center", fontsize=8.5,
             color="#555555", wrap=True)

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrik, ar


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    prod = model_produksi()
    kepatuhan = muat(RAGAS / f"uji_kepatuhan_{prod}.json")
    kestabilan = muat(RAGAS / "kestabilan.json")

    print(f"Model produksi (config.yaml): {prod}")
    for nama, d, kunci in [("uji kepatuhan", kepatuhan, "model"),
                           ("kestabilan RAGAS", kestabilan, "judge_model")]:
        dipakai = d.get(kunci)
        tanda = "COCOK" if dipakai == prod else "BEDA"
        print(f"  {nama:<18}: {dipakai}  [{tanda}]")
        if dipakai != prod:
            raise SystemExit(
                f"Model pada berkas {nama} ({dipakai}) berbeda dari produksi ({prod}). "
                "Jalankan ulang percobaannya sebelum membuat gambar."
            )

    out9 = args.out_dir / "fig6.9_generation_safety.png"
    out10 = args.out_dir / "fig6.10_stabilitas_ragas.png"

    hasil9 = gambar_kepatuhan(kepatuhan, out9)
    metrik, ar = gambar_kestabilan(kestabilan, out10)

    print(f"\nGambar VI.9 — kepatuhan ({kepatuhan['n_kasus']} kasus):")
    for lab, patuh in hasil9:
        print(f"  {'PATUH ' if all(patuh) else 'GAGAL '} {lab}")
    print(f"  tersimpan : {out9.relative_to(ROOT)}  ({out9.stat().st_size / 1024:.0f} KB)")

    print("\nGambar VI.10 — kestabilan:")
    print(f"  answer_relevancy rerata: jalan1 {ar['rerata_jalan1']} -> "
          f"jalan2 {ar['rerata_jalan2']}  (selisih {ar['selisih_rerata']})")
    for nama, med, n, diam in metrik:
        print(f"  {nama:<20} selisih absolut median {med}"
              + (f"   ({diam}/{n} kasus tidak berubah)" if diam is not None and diam == n else ""))
    print(f"  tersimpan : {out10.relative_to(ROOT)}  ({out10.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
