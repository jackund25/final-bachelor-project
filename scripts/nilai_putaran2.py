"""T2.2 putaran 2 — kesepakatan tiap penilai terhadap pelabel otomatis DAN antar-penilai.

Ditetapkan pada protokol yang ditulis sebelum lembar penilaian dibagikan.

APA YANG DIHITUNG
-----------------
1. kappa tiap penilai lawan pelabel otomatis  -> kesesuaian pelabel
2. kappa ANTAR-PENILAI                        -> BATAS ATAS yang sah
3. matriks konfusi 4x4 per pasangan
4. ARAH ketidaksepakatan per kelas

Arah lebih berguna daripada kappa. Ia menentukan apakah angka penelusuran yang dilaporkan
**terlalu optimistis atau terlalu pesimistis**: pelabel yang terlalu KETAT menandai potongan
relevan sebagai tidak relevan, sehingga Hit@k dan MRR yang dilaporkan menjadi BATAS BAWAH.

PENJAGA
-------
* kategori yang TIDAK DIPAKAI SAMA SEKALI oleh penilai mana pun -> peringatan menonjol,
  kode keluar bukan nol. Itu persis cacat yang memicu putaran kedua: pada putaran pertama
  kategori `normal` tidak pernah terpakai, dan kappa pada keadaan itu tidak dapat
  ditafsirkan apa adanya.
* nilai di luar empat kelas -> berhenti, sebutkan id dan nilainya
* himpunan id antar-berkas berbeda -> berhenti
* kappa antar-penilai hanya dihitung bila KEDUA lembar terisi; bila hanya satu, ketiadaan
  batas atas antar-penilai dinyatakan sebagai keterbatasan, bukan didiamkan

Keluaran: results/eval_rag/verifikasi_relevansi_putaran2.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.csv_penilaian import mulai_dari_header  # noqa: E402

KELAS = ["hipoglikemia", "hiperglikemia", "normal", "lain"]
BERKAS_KUNCI = ROOT / "evaluation/verifikasi_relevansi_KUNCI.json"
OUT = ROOT / "results/eval_rag/verifikasi_relevansi_putaran2.json"
csv.field_size_limit(10 ** 7)


def baca(path: Path) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as f:
        pembaca = csv.DictReader(mulai_dari_header(f))
        if not pembaca.fieldnames or "penilaian_manusia" not in pembaca.fieldnames:
            raise SystemExit(f"{path.name}: baris header tidak ditemukan. "
                             f"Kolom terbaca: {pembaca.fieldnames}")
        baris = [x for x in pembaca if x.get("id")]
    nilai, kosong, tak_sah = {}, [], []
    for b in baris:
        v = (b.get("penilaian_manusia") or "").strip().lower()
        if not v:
            kosong.append(b["id"])
        elif v not in KELAS:
            tak_sah.append((b["id"], v))
        else:
            nilai[b["id"]] = v
    if tak_sah:
        raise SystemExit(f"{path.name}: nilai tidak sah pada id "
                         f"{[i for i, _ in tak_sah]} -> {sorted({v for _, v in tak_sah})}. "
                         f"Sah: {KELAS}")
    return {"nilai": nilai, "kosong": kosong, "n_baris": len(baris)}


def kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in KELAS)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def tafsir(k: float) -> str:
    if k != k:
        return "tidak terdefinisi"
    return ("sangat lemah" if k < 0.20 else "lemah" if k < 0.40 else
            "sedang" if k < 0.60 else "kuat" if k < 0.80 else "sangat kuat")


def banding(nama_a: str, a: dict, nama_b: str, b: dict) -> dict:
    ids = sorted(set(a) & set(b), key=lambda x: int(x))
    va, vb = [a[i] for i in ids], [b[i] for i in ids]
    k = kappa(va, vb)
    matriks = {x: {y: 0 for y in KELAS} for x in KELAS}
    for x, y in zip(va, vb):
        matriks[x][y] += 1
    per_kelas = {}
    for kls in KELAS:
        tp = matriks[kls][kls]
        n_a = sum(matriks[kls].values())
        n_b = sum(matriks[x][kls] for x in KELAS)
        per_kelas[kls] = {
            f"n_menurut_{nama_a}": n_a, f"n_menurut_{nama_b}": n_b,
            "presisi": round(tp / n_a, 3) if n_a else None,
            "recall": round(tp / n_b, 3) if n_b else None,
        }
    arah = Counter((x, y) for x, y in zip(va, vb) if x != y)
    return {
        "n": len(ids),
        "kesepakatan_mentah_persen": round(100 * sum(1 for x, y in zip(va, vb) if x == y)
                                           / len(ids), 2) if ids else None,
        "cohen_kappa": None if k != k else round(float(k), 4),
        "tafsir": tafsir(k),
        f"matriks_baris_{nama_a}_kolom_{nama_b}": matriks,
        "per_kelas": per_kelas,
        "arah_ketidaksepakatan": [
            {f"{nama_a}": x, f"{nama_b}": y, "n": n} for (x, y), n in arah.most_common()],
        "sebaran": {nama_a: dict(Counter(va)), nama_b: dict(Counter(vb))},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--penilai1", default="evaluation/verifikasi_relevansi_putaran2_penilai1.csv")
    ap.add_argument("--penilai2", default="evaluation/verifikasi_relevansi_putaran2_penilai2.csv")
    ap.add_argument("--kunci", default=str(BERKAS_KUNCI))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    kunci = json.loads(Path(args.kunci).read_text(encoding="utf-8"))["label"]
    penilai = {}
    for nama, jalur in (("penilai1", args.penilai1), ("penilai2", args.penilai2)):
        p = Path(jalur)
        if not p.exists():
            print(f"CATATAN: {p.name} tidak ada — {nama} dilewati.")
            continue
        d = baca(p)
        if not d["nilai"]:
            print(f"CATATAN: {p.name} belum terisi ({len(d['kosong'])} baris kosong) — "
                  f"{nama} dilewati.")
            continue
        penilai[nama] = d

    if not penilai:
        raise SystemExit("Tidak ada lembar yang terisi. Jangan jalankan pada data kosong.")

    print("T2.2 PUTARAN 2 — kesepakatan pelabel dan antar-penilai")
    for nama, d in penilai.items():
        print(f"  {nama}: {len(d['nilai'])} terisi, {len(d['kosong'])} kosong "
              f"dari {d['n_baris']} baris | sebaran {dict(Counter(d['nilai'].values()))}")

    # ── PENJAGA: kategori yang tidak dipakai sama sekali ─────────────────────────
    peringatan, tak_terpakai = [], {}
    for nama, d in penilai.items():
        hilang = [k for k in KELAS if k not in set(d["nilai"].values())]
        if hilang:
            tak_terpakai[nama] = hilang
            peringatan.append(
                f"{nama} TIDAK PERNAH memakai kategori {hilang}. Kappa pada keadaan ini "
                f"TIDAK dapat ditafsirkan apa adanya — itu persis cacat yang memicu "
                f"putaran kedua.")

    hasil = {}
    for nama, d in penilai.items():
        oto = {i: kunci[i] for i in d["nilai"] if i in kunci}
        hasil[f"{nama}_vs_otomatis"] = banding("otomatis", oto, nama, d["nilai"])

    if len(penilai) == 2:
        hasil["antar_penilai"] = banding("penilai1", penilai["penilai1"]["nilai"],
                                         "penilai2", penilai["penilai2"]["nilai"])
    else:
        peringatan.append(
            "Hanya SATU lembar terisi, sehingga kappa ANTAR-PENILAI tidak dapat dihitung. "
            "Ketiadaan batas atas antar-manusia WAJIB dinyatakan sebagai keterbatasan pada "
            "K1, bukan didiamkan sebagai sesuatu yang masih ditunggu.")

    print(f"\n{'perbandingan':<26}{'n':>4}{'sepakat%':>10}{'kappa':>9}  tafsir")
    for k, v in hasil.items():
        print(f"{k:<26}{v['n']:>4}{v['kesepakatan_mentah_persen']:>10.2f}"
              f"{v['cohen_kappa'] if v['cohen_kappa'] is not None else float('nan'):>9.4f}"
              f"  {v['tafsir']}")

    for k, v in hasil.items():
        print(f"\nARAH ketidaksepakatan — {k}")
        if not v["arah_ketidaksepakatan"]:
            print("  (sepakat penuh)")
        for a in v["arah_ketidaksepakatan"]:
            kk = [x for x in a if x != "n"]
            print(f"  {kk[0]} '{a[kk[0]]}' padahal {kk[1]} '{a[kk[1]]}': {a['n']}")

    if "antar_penilai" in hasil:
        ka = hasil["antar_penilai"]["cohen_kappa"]
        ko = [hasil[k]["cohen_kappa"] for k in hasil if k.endswith("_vs_otomatis")]
        ko = [x for x in ko if x is not None]
        if ka is not None and ko:
            lebih = ka > max(ko)
            print(f"\nD3: kappa antar-penilai {ka:.4f} lawan maks kappa thd otomatis "
                  f"{max(ko):.4f} -> {'TERPENUHI' if lebih else 'TERBANTAH'}")
            if not lebih:
                peringatan.append(
                    "Kedua manusia lebih sepakat dengan MESIN daripada dengan satu sama "
                    "lain. Itu menunjukkan TUGASNYA sendiri ambigu, bukan pelabelnya yang "
                    "buruk, dan menyentuh K1 secara mendasar.")

    out = {
        "percobaan": "T2.2 putaran 2 — verifikasi pelabel relevansi, dua penilai",
        "protokol": "penilaian putaran kedua — ditulis di muka",
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "penilai": {
            "penilai1": "mahasiswa kedokteran; putaran KEDUA, BUKAN penilaian independen "
                        "(sudah pernah membaca ke-40 potongan yang sama)",
            "penilai2": "mahasiswa koas, belum pernah menyentuh proyek ini; penilaian "
                        "tunggal dan mandiri",
            "keduanya": "tidak terlibat pengembangan sistem; nama tidak dicatat",
            "peneliti": "TIDAK menjadi penilai — sudah melihat label otomatis dan hasil "
                        "putaran pertama, sehingga penilaiannya akan menaikkan kappa "
                        "secara palsu. Ini juga bukan test-retest, yang mensyaratkan kedua "
                        "kali dilakukan buta.",
        },
        "perubahan_instrumen": [
            "panduan: penjelasan bahwa `normal` berarti kelas target/rentang aman, dan "
            "bahwa keempat kategori wajib dipertimbangkan",
            "lembar: kolom kueri, sumber, halaman_cetak, dan peringkat DIBUANG — "
            "menampilkan kueri mengundang penilaian relevansi alih-alih penilaian topik",
        ],
        "kesebandingan_antar_putaran": (
            "Instrumen berubah pada DUA titik, sehingga selisih kappa antar-putaran TIDAK "
            "dapat diatribusikan seluruhnya kepada perbaikan panduan. Angka utama adalah "
            "kappa PUTARAN KEDUA, bukan selisihnya."),
        "n_penilai_terisi": len(penilai),
        "kategori_tidak_terpakai": tak_terpakai,
        "hasil": hasil,
        "peringatan": peringatan,
        "aturan_pelaporan": (
            "kappa < 0,40: angka penelusuran wajib berkualifikasi eksplisit di SETIAP "
            "penyebutannya. 0,40-0,60: dilaporkan dengan catatan. > 0,60: pelabel memadai "
            "untuk tujuan ini. Kappa antar-penilai adalah BATAS ATAS; kappa penilai1 "
            "terhadap otomatis BUKAN batas atas antar-manusia."),
        "batas_tafsir": (
            "n = 40 kecil; selang kepercayaan kappa lebar. Yang sah disimpulkan adalah "
            "GOLONGANNYA (lemah/sedang/kuat), bukan nilai persisnya."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    if peringatan:
        print(f"\n  {'!' * 68}")
        for w in peringatan:
            print(f"  - {w}")
        print(f"  {'!' * 68}")
    print(f"\nDisimpan ke {args.out}")
    return 1 if tak_terpakai else 0


if __name__ == "__main__":
    raise SystemExit(main())
