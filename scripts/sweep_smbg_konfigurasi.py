"""Sapuan konfigurasi FINGER_STICK: panjang riwayat x horizon target.

Motivasi
--------
Konfigurasi produksi ``8 observasi -> ~4 jam`` ditetapkan berdasarkan audit
kelayakan temporal, yaitu berapa banyak sampel yang tersisa untuk tiap kombinasi.
Audit itu menjawab "kombinasi mana yang datanya cukup", bukan "kombinasi mana yang
kinerjanya terbaik". Sapuan ini melengkapinya dengan pengukuran, sehingga pilihan
konfigurasi bersandar pada bukti dan bukan pada asumsi.

Yang sudah diketahui sebelum sapuan (Tahap 1, ``results/eval_prediksi/
smbg_persistence_4h.json``): pada horizon ~4 jam model produksi MENGUNGGULI
baseline persistence secara meyakinkan — RMSE 68,24 lawan 95,69 dan Clarke A+B
52,52% lawan 38,85%. Model belajar sinyal nyata; yang membatasi adalah sukarnya
horizon 4 jam. Pertanyaan sapuan ini karena itu menjadi: apakah horizon yang lebih
pendek menembus ambang layak-klinis, dan berapa harga yang dibayar dalam jumlah
sampel.

Setiap kombinasi disertai baseline persistence-nya sendiri. Tanpa itu angka
antar-horizon tidak dapat dibandingkan: horizon pendek selalu tampak lebih baik
pada RMSE semata-mata karena glukosa bergerak lebih sedikit.

Jendela target mengikuti audit ``src/data/smbg_training_sample_audit.py``:

    ~1 jam :  45-75  menit
    ~2 jam :  90-150 menit
    ~4 jam : 180-300 menit

Subproses per kombinasi
-----------------------
Tiap kombinasi dijalankan pada subproses tersendiri. ``train_gbm_from_config``
memuat ulang berkas dataset penuh (171 ribu baris) setiap kali dipanggil, dan
menjalankan sembilan kombinasi dalam satu proses membuat pemakaian memori menumpuk
sampai gagal alokasi pada mesin kelas konsumen. Subproses mengembalikan memori
sepenuhnya di antara putaran.

Artefak produksi TIDAK tersentuh: seluruh pelatihan memakai ``save_artifacts=False``.

Keluaran:
    results/eval_prediksi/smbg_sweep.json
    results/eval_prediksi/smbg_sweep.csv
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUMBER = "FINGER_STICK"

RIWAYAT = [5, 8, 12]

# (label, nominal, batas_bawah, batas_atas) — sepadan dengan audit temporal.
HORIZON = [
    ("1j", 60.0, 45.0, 75.0),
    ("2j", 120.0, 90.0, 150.0),
    ("4j", 240.0, 180.0, 300.0),
]

DEST_JSON = ROOT / "results/eval_prediksi/smbg_sweep.json"
DEST_CSV = ROOT / "results/eval_prediksi/smbg_sweep.csv"

# Konfigurasi yang sedang berlaku di produksi, untuk ditandai pada tabel hasil.
PRODUKSI = (8, "4j")


# ===================================================================== worker
def worker(seq: int, nominal: float, bawah: float, atas: float) -> None:
    """Latih satu kombinasi + hitung persistence-nya, cetak JSON ke stdout.

    Dijalankan sebagai subproses oleh ``main``. Seluruh impor berat berada di
    sini agar proses induk tetap ringan.
    """
    import yaml

    from scripts.eval_smbg_persistence_4h import jendela, siapkan
    from src.models.gbm_model import train_gbm_from_config
    from src.utils.metrics import calculate_all_metrics

    hasil = train_gbm_from_config(
        source=SUMBER,
        horizon_min=nominal,
        sequence_length=seq,
        min_target_horizon_min=bawah,
        max_target_horizon_min=atas,
        save_artifacts=False,          # models/ tidak boleh tersentuh
    )
    m = hasil[f"h{int(nominal)}m"]

    # Persistence pada himpunan jendela yang sama persis.
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pre, d = siapkan(config, SUMBER)
    source_cfg = {
        "target_tolerance_min": 0.0,
        "max_history_gap_min": 720,
        "min_history_interval_min": 5.0,
        "min_target_horizon_min": bawah,
        "max_target_horizon_min": atas,
    }
    y, anchor = jendela(pre, d, source_cfg, seq, nominal, "test")
    pm = calculate_all_metrics(y, anchor) if len(y) else None

    print("<<<JSON>>>" + json.dumps({"model": m, "persistence": pm,
                                     "n_uji_persistence": int(len(y))}))


# ======================================================================= main
def jalankan_kombinasi(seq, nominal, bawah, atas):
    """Panggil worker sebagai subproses; kembalikan dict hasil atau None."""
    perintah = [
        sys.executable, str(Path(__file__).resolve()),
        "--worker",
        "--seq", str(seq),
        "--nominal", str(nominal),
        "--bawah", str(bawah),
        "--atas", str(atas),
    ]
    proses = subprocess.run(
        perintah, capture_output=True, text=True, cwd=str(ROOT)
    )
    for garis in proses.stdout.splitlines():
        if garis.startswith("<<<JSON>>>"):
            return json.loads(garis[len("<<<JSON>>>"):])

    ekor = (proses.stderr or proses.stdout).strip().splitlines()
    print(f"  gagal: {ekor[-1] if ekor else 'tanpa keluaran'}")
    return None


def main() -> None:
    baris, rinci = [], {}

    for seq in RIWAYAT:
        for label, nominal, bawah, atas in HORIZON:
            kunci = f"seq{seq}_{label}"
            print(f"\n=== {kunci} (riwayat {seq}, target {bawah:g}-{atas:g} menit) ===")

            hasil = jalankan_kombinasi(seq, nominal, bawah, atas)
            if hasil is None:
                baris.append({"konfigurasi": kunci, "riwayat": seq,
                              "horizon": label, "catatan": "gagal"})
                continue

            m, pm = hasil["model"], hasil["persistence"]
            r = {
                "konfigurasi": kunci,
                "riwayat": seq,
                "horizon": label,
                "target_min": f"{bawah:g}-{atas:g}",
                "n_latih": m["train_samples"],
                "n_uji": m["test_samples"],
                "RMSE": round(m["RMSE"], 2),
                "MAE": round(m["MAE"], 2),
                "Clarke_A": round(m["Clarke_A"], 2),
                "Clarke_A+B": round(m["Clarke_A+B"], 2),
                "Clarke_E": round(m["Clarke_E"], 2),
                "pers_RMSE": round(pm["RMSE"], 2) if pm else None,
                "pers_A+B": round(pm["Clarke_A"] + pm["Clarke_B"], 2) if pm else None,
                "pers_E": round(pm["Clarke_E"], 2) if pm else None,
                "produksi": (seq, label) == PRODUKSI,
            }
            # Selisih terhadap baseline: inilah besaran yang benar-benar
            # sebanding antar-horizon, sebab RMSE mentah selalu mengecil pada
            # horizon pendek hanya karena glukosa bergerak lebih sedikit.
            if pm:
                r["unggul_RMSE"] = round(pm["RMSE"] - m["RMSE"], 2)
                r["unggul_A+B"] = round(
                    m["Clarke_A+B"] - (pm["Clarke_A"] + pm["Clarke_B"]), 2
                )

            baris.append(r)
            rinci[kunci] = hasil
            print(f"  model       RMSE {r['RMSE']:7.2f} | A+B {r['Clarke_A+B']:6.2f}% "
                  f"| E {r['Clarke_E']:5.2f}% | n latih {r['n_latih']} uji {r['n_uji']}")
            if pm:
                print(f"  persistence RMSE {r['pers_RMSE']:7.2f} | A+B {r['pers_A+B']:6.2f}% "
                      f"| E {r['pers_E']:5.2f}%")

    tab = pd.DataFrame(baris)
    DEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    DEST_JSON.write_text(
        json.dumps(
            {
                "catatan": (
                    "Sapuan konfigurasi FINGER_STICK. Setiap kombinasi disertai "
                    "baseline persistence pada himpunan jendela yang sama. "
                    "Artefak produksi tidak tersentuh (save_artifacts=False)."
                ),
                "sumber": SUMBER,
                "riwayat_diuji": RIWAYAT,
                "horizon_diuji": [
                    {"label": l, "nominal_min": n, "batas": [b, a]}
                    for l, n, b, a in HORIZON
                ],
                "konfigurasi_produksi": {"riwayat": PRODUKSI[0], "horizon": PRODUKSI[1]},
                "ringkas": baris,
                "rinci": rinci,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    tab.to_csv(DEST_CSV, index=False)

    print("\n" + "=" * 100)
    kolom = ["konfigurasi", "n_latih", "n_uji", "RMSE", "pers_RMSE", "unggul_RMSE",
             "Clarke_A+B", "pers_A+B", "unggul_A+B", "Clarke_E", "produksi"]
    print(tab[[k for k in kolom if k in tab.columns]].to_string(index=False))
    print(f"\nDisimpan ke {DEST_JSON.relative_to(ROOT)} dan {DEST_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sapuan konfigurasi FINGER_STICK.")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--seq", type=int)
    ap.add_argument("--nominal", type=float)
    ap.add_argument("--bawah", type=float)
    ap.add_argument("--atas", type=float)
    args = ap.parse_args()

    if args.worker:
        worker(args.seq, args.nominal, args.bawah, args.atas)
    else:
        main()
