"""T7 — diagnosis POST-HOC: apakah alat ukur MRR sah dipakai lintas varian chunking?

DIJALANKAN SESUDAH HASIL DILIHAT. Karena itu seluruh isinya berstatus
PENGAMATAN, bukan uji hipotesis. Tidak ada nilai-p yang sah dihitung di sini.
Ditulis terpisah dari scripts/eval_strategi_chunking.py supaya batas antara yang
diprapendaftarkan dan yang tidak tetap terlihat.

LATAR. T7 mengukur MRR dengan cara: ambil 5 potongan teratas, beri label topik
tiap potongan dengan classify_chunk(), lalu cari peringkat potongan yang labelnya
sama dengan label yang diharapkan. Varian V3 (sadar-token) anjlok ke MRR 0,150
dari 0,492 — penurunan yang terlalu besar untuk sekadar pertukaran mutu.

DUGAAN YANG DIPERIKSA. classify_chunk() memberi label DARI TEKS POTONGAN ITU
SENDIRI, memakai kata kunci berbobot dengan max(). Yang justru diubah percobaan
ini adalah batas potongan. Bila peluang sebuah potongan menangkap kata kunci
bergantung pada panjangnya, maka alat ukurnya bergerak mengikuti perlakuan, dan
perbandingan MRR antarvarian tidak sah.

Keluaran: results/eval_rag/diagnosis_metrik_chunking.json
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ablation_rag_fullkb import KW, classify_chunk  # noqa: E402

TMP = ROOT / "models/_chunk_strategi"
OUT = ROOT / "results/eval_rag/diagnosis_metrik_chunking.json"
VARIAN = ["v0", "v1", "v2", "v3"]
BIN = [(0, 300, "<300"), (300, 600, "300-600"), (600, 900, "600-900"), (900, 10**9, ">=900")]


def muat(v: str):
    import chromadb
    col = chromadb.PersistentClient(path=str(TMP / v)).get_collection("diabetes_kb")
    return col.get(include=["documents"])["documents"]


def bias_panjang(teks_semua) -> dict:
    """Peluang sebuah potongan mendapat label topik (bukan 'lain'), per kelas panjang.

    Bila angkanya naik seiring panjang, alat ukurnya memihak potongan panjang —
    dan panjang potongan persis yang diubah percobaan ini.
    """
    hasil = {}
    for lo, hi, nama in BIN:
        anggota = [t for t in teks_semua if lo <= len(t) < hi]
        if not anggota:
            hasil[nama] = None
            continue
        berlabel = sum(classify_chunk(t) != "lain" for t in anggota)
        hasil[nama] = {
            "n": len(anggota),
            "persen_dapat_label_topik": round(100 * berlabel / len(anggota), 1),
        }
    return hasil


def main() -> None:
    tersedia = [v for v in VARIAN if (TMP / v).exists()]
    if not tersedia:
        raise SystemExit(f"Indeks varian tidak ditemukan di {TMP}. "
                         "Jalankan scripts/eval_strategi_chunking.py lebih dulu.")

    per_varian = {}
    for v in tersedia:
        teks = muat(v)
        per_varian[v] = {
            "n_chunk": len(teks),
            "char_rerata": round(sum(len(t) for t in teks) / len(teks), 1),
            "sebaran_label_korpus": dict(Counter(classify_chunk(t) for t in teks)),
            "bias_panjang": bias_panjang(teks),
        }

    # Apakah biasnya monoton naik di dalam setiap varian?
    monoton = {}
    for v, d in per_varian.items():
        seri = [d["bias_panjang"][n]["persen_dapat_label_topik"]
                for _, _, n in BIN if d["bias_panjang"].get(n)]
        monoton[v] = {
            "seri": seri,
            "naik_monoton": all(a <= b for a, b in zip(seri, seri[1:])) and len(seri) > 1,
        }

    hasil = {
        "percobaan": "T7 — diagnosis post-hoc kesahihan alat ukur MRR lintas varian chunking",
        "STATUS": ("POST-HOC. Dijalankan SESUDAH hasil dilihat. Seluruh isinya "
                   "PENGAMATAN, bukan uji hipotesis. Tidak ada nilai-p yang sah."),
        "sumber_angka_MRR": "results/eval_rag/strategi_chunking.json",
        "alat_ukur_yang_diperiksa": {
            "fungsi": "classify_chunk() pada scripts/ablation_rag_fullkb.py",
            "cara_kerja": ("kata kunci berbobot per kelas, dipilih dengan max(); "
                           "bila skor tertinggi 0 maka labelnya 'lain'"),
            "n_kata_kunci_per_kelas": {k: len(v) for k, v in KW.items()},
            "bobot_maks_per_kelas": {k: max(w for _, w in v) for k, v in KW.items()},
        },
        "per_varian": per_varian,
        "monotonisitas_bias_panjang": monoton,
        "TEMUAN": (
            "Peluang sebuah potongan mendapat label topik naik seiring panjangnya, "
            "di SETIAP varian. Sebabnya mekanis: makin banyak teks dalam satu "
            "potongan, makin besar peluangnya memuat salah satu frasa kata kunci. "
            "Karena percobaan ini justru mengubah panjang dan batas potongan, label "
            "yang menjadi dasar MRR ikut berubah bersama perlakuan."
        ),
        "AKIBAT_BAGI_KEPUTUSAN": (
            "Perbandingan MRR ANTARVARIAN tidak sah dipakai untuk mengadopsi atau "
            "menolak sebuah strategi chunking, karena kelas pembanding bukan besaran "
            "tetap. Aturan 4 pada prapendaftaran mengandaikan alat ukurnya invarian "
            "lintas lengan; pengandaian itu TIDAK terpenuhi. Penolakan V3 karena itu "
            "DITANGGUHKAN, bukan disimpulkan."
        ),
        "BUKTI_KUALITATIF": (
            "Untuk kueri kondisi normal, dua potongan teratas V3 adalah 'Target "
            "Glukosa Darah Berdasarkan ISPAD dan IDF' dan 'Target Glukosa Darah "
            "Untuk Penyandang Diabetes Melitus Gestational' — keduanya justru tabel "
            "sasaran glikemik, yakni isi yang paling tepat. Keduanya dilabeli 'lain' "
            "karena daftar kata kunci kelas 'normal' menuntut frasa 'target glikemik' "
            "atau 'kontrol glikemik', sedangkan potongan itu menulis 'Target Glukosa "
            "Darah'. Alat ukurnya menghukum potongan yang benar."
        ),
        "YANG_TIDAK_DAPAT_DISIMPULKAN": (
            "Temuan ini TIDAK membuktikan V3 lebih baik. Ia hanya membuktikan "
            "perbandingan ini tidak dapat memutuskan. Bias panjang juga tidak "
            "menjelaskan seluruhnya: V2 berpotongan lebih pendek daripada V0 namun "
            "ber-MRR lebih tinggi, sehingga ada faktor lain yang belum dipisahkan."
        ),
        "LANGKAH_YANG_DIPERLUKAN": [
            "Ganti label kebenaran agar tidak diturunkan dari teks potongan — "
            "misalnya label di tingkat DOKUMEN/HALAMAN yang tetap meski potongannya "
            "berubah, atau penilaian relevansi oleh manusia.",
            "Sampai itu tersedia, keputusan adopsi V3 ditangguhkan.",
            "V1 (pemisah kalimat) tetap dapat diadopsi: dasarnya integritas kalimat "
            "yang terukur langsung, bukan MRR.",
        ],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Bias panjang alat ukur — % potongan yang mendapat label topik (bukan 'lain'):")
    print(f"{'varian':8}" + "".join(f"{n:>16}" for _, _, n in BIN))
    for v, d in per_varian.items():
        baris = ""
        for _, _, n in BIN:
            b = d["bias_panjang"].get(n)
            baris += f"{(str(b['persen_dapat_label_topik']) + '% (n=' + str(b['n']) + ')') if b else '-':>16}"
        print(f"{v:8}{baris}")
    print("\nNaik monoton per varian:",
          {v: m["naik_monoton"] for v, m in monoton.items()})
    print(f"\nDisimpan ke {OUT}")


if __name__ == "__main__":
    main()
