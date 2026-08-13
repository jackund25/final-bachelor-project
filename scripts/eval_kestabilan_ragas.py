"""Kestabilan tiga metrik RAGAS yang belum pernah diuji (keputusan #7).

Prapendaftaran: docs/PRAPENDAFTARAN_T_RAGAS_STABIL.md, ditulis sebelum skrip ini ada.

APA YANG DIKERJAKAN, DAN APA YANG TIDAK
---------------------------------------
Skrip ini **tidak memanggil LLM**. Ia membandingkan dua himpunan skor yang sudah ada:

  jalan 1  results/ragas/kestabilan_jalan1.json  (disalin dari cache T1.2 sebelum jalan 2)
  jalan 2  results/ragas/cache/                  (ditulis run_ragas.py --no-cache)

Pemanggilan LLM dikerjakan `run_ragas.py`, yang sudah punya rate limiter, estimasi kuota,
dan cache per kasus — cache itu SEKALIGUS checkpoint-nya: bila jalan kedua terputus di
tengah, kasus yang sudah dinilai tidak dinilai ulang.

MENGAPA PERBANDINGANNYA DIPISAH DARI PEMANGGILANNYA
---------------------------------------------------
Karena jalan kedua menimpa cache. Kalau perbandingan dikerjakan di dalam proses yang sama,
satu-satunya salinan jalan pertama akan hilang begitu jalan kedua menulis, dan pengukuran
kestabilan justru menghancurkan datanya sendiri.

PENJAGA
-------
Empat hal ditolak, bukan dihitung diam-diam:
  1. snapshot jalan 1 tidak ada
  2. himpunan kasus kedua jalan berbeda
  3. model juri atau top_k berbeda
  4. skor jalan 2 identik SELURUHNYA dengan jalan 1 -> tanda cache tidak benar-benar
     dilewati, sehingga "stabil sempurna" adalah artefak, bukan temuan

NaN tidak diperlakukan sebagai nol; kasusnya dikeluarkan dan dilaporkan terpisah.

Keluaran: results/ragas/kestabilan.json
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JALAN1 = ROOT / "results/ragas/kestabilan_jalan1.json"
CACHE = ROOT / "results/ragas/cache"
JALAN2 = ROOT / "results/ragas/kestabilan_jalan2.json"
OUT = ROOT / "results/ragas/kestabilan.json"

METRIK = ["answer_relevancy", "context_precision", "context_recall"]
# Acuan faithfulness dari dua jalan penuh (T1.2). Dipakai sebagai pembanding, bukan ambang.
FAITHFULNESS_ACUAN = {"median": 0.18, "maks": 0.56, "selisih_rerata": 0.003}


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def baca_jalan2(sumber: Path) -> dict:
    """Terima snapshot JSON maupun direktori cache.

    Snapshot lebih disukai: cache DIPULIHKAN ke keadaan jalan 1 setelah pengukuran, karena
    run_ragas.py menimpa summary.json dan per_sample.csv dengan angka jalan kedua — dan
    aturan pra-registrasi melarang angka jalan pertama berubah. Tanpa snapshot, hasil
    pengukuran ini tidak dapat dihitung ulang.
    """
    if sumber.is_file():
        return json.loads(sumber.read_text(encoding="utf-8"))["skor"]
    skor: dict = {}
    for p in sumber.glob("*.json"):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if "metric" in j and "id" in j:
            skor.setdefault(j["metric"], {})[j["id"]] = j.get("score")
    return skor


def median(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def banding(m: str, a: dict, b: dict) -> dict:
    kasus = sorted(set(a) & set(b))
    pasang, nan_ids = [], []
    for k in kasus:
        if _nan(a[k]) or _nan(b[k]):
            nan_ids.append(k)
            continue
        pasang.append((k, float(a[k]), float(b[k])))

    selisih = [abs(y - x) for _, x, y in pasang]
    berbalik = [k for k, x, y in pasang if abs(y - x) >= 0.999]
    rerata1 = sum(x for _, x, _ in pasang) / len(pasang) if pasang else None
    rerata2 = sum(y for _, _, y in pasang) / len(pasang) if pasang else None

    return {
        "n_dibandingkan": len(pasang),
        "n_dikeluarkan_karena_NaN": len(nan_ids),
        "kasus_NaN": nan_ids,
        "selisih_absolut_median": round(median(selisih), 4) if selisih else None,
        "selisih_absolut_maks": round(max(selisih), 4) if selisih else None,
        "rerata_jalan1": round(rerata1, 4) if rerata1 is not None else None,
        "rerata_jalan2": round(rerata2, 4) if rerata2 is not None else None,
        "selisih_rerata": (round(rerata2 - rerata1, 4)
                           if rerata1 is not None and rerata2 is not None else None),
        "n_berbalik_penuh": len(berbalik),
        "kasus_berbalik_penuh": berbalik,
        "n_tidak_bergerak": sum(1 for d in selisih if d < 1e-9),
        "per_kasus": [{"id": k, "jalan1": round(x, 4), "jalan2": round(y, 4),
                       "selisih": round(y - x, 4)} for k, x, y in pasang],
        "lebih_stabil_daripada_faithfulness": (
            bool(selisih and median(selisih) < FAITHFULNESS_ACUAN["median"]
                 and max(selisih) < FAITHFULNESS_ACUAN["maks"])),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jalan1", default=str(JALAN1))
    ap.add_argument("--cache", default=str(JALAN2),
                    help=("Sumber jalan 2: berkas snapshot ATAU direktori cache. "
                          "Bawaannya snapshot, karena cache DIPULIHKAN ke keadaan jalan 1 "
                          "setelah pengukuran — run_ragas.py menimpa summary.json dan "
                          "per_sample.csv, dan aturan pra-registrasi melarang angka jalan "
                          "pertama berubah."))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    p1 = Path(args.jalan1)
    if not p1.exists():
        raise SystemExit(
            f"PENJAGA 1: snapshot jalan 1 tidak ada di {p1}. Ia HARUS disalin sebelum "
            f"jalan kedua dijalankan; setelah cache tertimpa, jalan pertama tidak dapat "
            f"dipulihkan.")
    j1 = json.loads(p1.read_text(encoding="utf-8"))
    s1 = j1["skor"]
    s2 = baca_jalan2(Path(args.cache))

    print("KESTABILAN TIGA METRIK RAGAS (keputusan #7)")
    print(f"  jalan 1 : {p1.name}  (disalin {j1.get('disalin')})")
    print(f"  jalan 2 : {args.cache}")
    print(f"  juri    : {j1.get('judge_model')} | top_k {j1.get('top_k')}\n")

    hasil, peringatan = {}, []
    for m in METRIK:
        a, b = s1.get(m, {}), s2.get(m, {})
        if not a or not b:
            peringatan.append(f"{m}: metrik tidak lengkap (jalan1 {len(a)}, jalan2 {len(b)})")
            hasil[m] = {"_status": "TIDAK LENGKAP", "n_jalan1": len(a), "n_jalan2": len(b)}
            continue
        if set(a) != set(b):
            hanya1, hanya2 = sorted(set(a) - set(b)), sorted(set(b) - set(a))
            peringatan.append(f"{m}: himpunan kasus BERBEDA (hanya jalan1 {hanya1}, "
                              f"hanya jalan2 {hanya2}) — hanya irisannya dibandingkan")
        hasil[m] = banding(m, a, b)

    # PENJAGA 4: skor identik seluruhnya menandakan cache tidak benar-benar dilewati.
    identik = [m for m, r in hasil.items()
               if r.get("n_dibandingkan") and r["n_tidak_bergerak"] == r["n_dibandingkan"]]
    if len(identik) == len([m for m in METRIK if hasil[m].get("n_dibandingkan")]):
        peringatan.append(
            "PENJAGA 4: SELURUH skor identik pada seluruh metrik. Itu tanda cache TIDAK "
            "benar-benar dilewati (--no-cache tidak berlaku), bukan tanda metrik stabil "
            "sempurna. Hasil ini TIDAK SAH sebagai pengukuran kestabilan.")

    print(f"{'metrik':<20}{'n':>4}{'NaN':>5}{'median':>9}{'maks':>8}"
          f"{'d-rerata':>10}{'berbalik':>10}{'diam':>6}")
    for m in METRIK:
        r = hasil[m]
        if r.get("_status"):
            print(f"{m:<20}  {r['_status']}")
            continue
        print(f"{m:<20}{r['n_dibandingkan']:>4}{r['n_dikeluarkan_karena_NaN']:>5}"
              f"{r['selisih_absolut_median']:>9.4f}{r['selisih_absolut_maks']:>8.4f}"
              f"{r['selisih_rerata']:>+10.4f}{r['n_berbalik_penuh']:>10}"
              f"{r['n_tidak_bergerak']:>6}")
    print(f"\n{'faithfulness (acuan)':<20}{'10':>4}{'0':>5}"
          f"{FAITHFULNESS_ACUAN['median']:>9.4f}{FAITHFULNESS_ACUAN['maks']:>8.4f}"
          f"{FAITHFULNESS_ACUAN['selisih_rerata']:>+10.4f}")

    out = {
        "percobaan": "Kestabilan answer_relevancy, context_precision, context_recall",
        "keputusan": "#7 pada docs/KEPUTUSAN_DIAMBIL.md",
        "prapendaftaran": "docs/PRAPENDAFTARAN_T_RAGAS_STABIL.md",
        "waktu": datetime.now().isoformat(timespec="seconds"),
        "sifat": ("DIAGNOSTIK. Tidak satu pun angka RAGAS yang sudah ada diubah atau "
                  "diganti. Yang dihasilkan adalah pernyataan kestabilan yang menyertai "
                  "angka-angka itu di Bab VI. Rerata dua jalan BUKAN angka baru."),
        "judge_model": j1.get("judge_model"), "top_k": j1.get("top_k"),
        "acuan_faithfulness": FAITHFULNESS_ACUAN,
        "hasil": hasil,
        "peringatan": peringatan,
        "asimetri_sumber_variasi": (
            "answer_relevancy bergantung pada JAWABAN, yang dibangkitkan ulang pada jalan "
            "kedua, sehingga variasinya bercampur antara pembangkitan dan penjurian. "
            "context_precision dan context_recall tidak memakai jawaban sama sekali, "
            "sehingga variasinya adalah variasi JURI murni (penelusuran deterministik)."),
        "batas_tafsir": (
            "n = 10 kasus terlalu kecil untuk menyatakan BESARAN ketidakstabilan secara "
            "presisi. Yang sah disimpulkan adalah ADA atau TIDAKNYA ketidakstabilan "
            "tingkat kasus. NaN tidak diperlakukan sebagai nol."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    if peringatan:
        print("\nPERINGATAN:")
        for w in peringatan:
            print(f"  - {w}")
    print(f"\nDisimpan ke {args.out}")
    return 2 if any(p.startswith("PENJAGA 4") for p in peringatan) else 0


if __name__ == "__main__":
    raise SystemExit(main())
