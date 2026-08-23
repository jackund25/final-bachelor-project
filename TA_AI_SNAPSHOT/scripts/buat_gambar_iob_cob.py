#!/usr/bin/env python3
"""Hasilkan Gambar II.5 — pencatatan kejadian sesaat lawan besaran meluruh.

Gambar ini BUKAN hasil EDA. EDA (notebooks/01_data_exploration.ipynb §5)
menghasilkan bar chart persentase baris non-nol per kanal — agregat, tanpa
sumbu waktu, sehingga tidak dapat memperlihatkan peluruhan. Gambar ini
memperlihatkan KELUARAN rekayasa fitur pada sumbu waktu.

PENTING — kolom yang dipakai. Panel atas memakai ``bolus_dose``, BUKAN
``insulin``. Kanal ``insulin`` memuat basal yang diinfus terus-menerus sehingga
bernilai non-nol pada 99,92% baris; menggambarkannya sebagai batang kejadian
akan membantah teks yang menyertainya. ``bolus_dose`` bernilai non-nol pada
1,94% baris dan memang diskret. Pilihan ini konsisten dengan
``preprocessor.engineer_features()`` yang menghitung IOB dari ``bolus_dose``
bila kolom itu tersedia.

Jalankan:
    PYTHONPATH=. python scripts/buat_gambar_iob_cob.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataPreprocessor  # noqa: E402

OUT = ROOT / "docs/laporan_TA/TA-STI-template-1.0/images/Gambar_II5_IOBCOB.png"
JAM_TAMPIL = 14          # panjang jendela yang digambar
MIN_BOLUS = 3            # syarat minimum agar penumpukan terlihat
MIN_MEAL = 3


def pilih_jendela(df: pd.DataFrame) -> pd.DataFrame:
    """Cari jendela yang memuat cukup banyak bolus DAN makan agar penumpukan terlihat."""
    lebar = pd.Timedelta(hours=JAM_TAMPIL)
    for pid, g in df.groupby("patient_id", sort=False):
        g = g.sort_values("timestamp").reset_index(drop=True)
        bolus_t = g.loc[g["bolus_dose"] > 0, "timestamp"]
        for t0 in bolus_t:
            sub = g[(g["timestamp"] >= t0 - pd.Timedelta(hours=1)) &
                    (g["timestamp"] <= t0 - pd.Timedelta(hours=1) + lebar)]
            if (sub["bolus_dose"] > 0).sum() >= MIN_BOLUS and (sub["carbs"] > 0).sum() >= MIN_MEAL:
                return sub.reset_index(drop=True)
    raise SystemExit("Tidak ditemukan jendela yang memenuhi syarat.")


def main() -> int:
    df = pd.read_csv(ROOT / "data/raw/ohio_t1dm_merged.csv", parse_dates=["timestamp"])
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)

    # Rekayasa fitur dijalankan pada SELURUH deret pasien, bukan pada potongan,
    # agar nilai IOB/COB di awal jendela sudah memuat riwayat sebelumnya.
    pre = DataPreprocessor({})
    fe = pre.engineer_features(df, insulin_tau_min=240.0, carbs_tau_min=180.0, trend_steps=3)

    jendela = pilih_jendela(fe)
    pid = jendela["patient_id"].iloc[0]
    t = jendela["timestamp"]
    jam = (t - t.iloc[0]).dt.total_seconds() / 3600.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True,
                                   gridspec_kw={"hspace": 0.18})

    # ---- Panel atas: kejadian diskret -----------------------------------
    b = jendela["bolus_dose"].to_numpy()
    c = jendela["carbs"].to_numpy()
    mb, mc = b > 0, c > 0
    ax1.vlines(jam[mb], 0, b[mb], color="#1f4e79", lw=2.4, label="Bolus insulin (unit)")
    ax1.plot(jam[mb], b[mb], "o", color="#1f4e79", ms=5)
    ax1b = ax1.twinx()
    ax1b.vlines(jam[mc], 0, c[mc], color="#b35c00", lw=2.4, label="Asupan karbohidrat (g)")
    ax1b.plot(jam[mc], c[mc], "s", color="#b35c00", ms=5)

    ax1.set_ylabel("Bolus insulin (unit)", color="#1f4e79")
    ax1b.set_ylabel("Karbohidrat (g)", color="#b35c00")
    ax1.tick_params(axis="y", labelcolor="#1f4e79")
    ax1b.tick_params(axis="y", labelcolor="#b35c00")
    ax1.set_ylim(bottom=0)
    ax1b.set_ylim(bottom=0)
    ax1.set_title("(a) Pencatatan sesaat: bernilai nol di luar titik kejadian",
                  loc="left", fontsize=10.5)
    ax1.grid(alpha=0.25, ls=":")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.9)

    # ---- Panel bawah: besaran meluruh -----------------------------------
    ax2.plot(jam, jendela["iob"], color="#1f4e79", lw=2.0, label="IOB (unit)")
    ax2.fill_between(jam, 0, jendela["iob"], color="#1f4e79", alpha=0.13)
    ax2b = ax2.twinx()
    ax2b.plot(jam, jendela["cob"], color="#b35c00", lw=2.0, label="COB (g)")
    ax2b.fill_between(jam, 0, jendela["cob"], color="#b35c00", alpha=0.13)

    # tandai titik kejadian agar keterkaitan antarpanel terbaca
    for x in jam[mb]:
        ax2.axvline(x, color="#1f4e79", alpha=0.18, lw=0.9, ls="--")
    for x in jam[mc]:
        ax2.axvline(x, color="#b35c00", alpha=0.18, lw=0.9, ls="--")

    ax2.set_ylabel("Insulin on board (unit)", color="#1f4e79")
    ax2b.set_ylabel("Carbs on board (g)", color="#b35c00")
    ax2.tick_params(axis="y", labelcolor="#1f4e79")
    ax2b.tick_params(axis="y", labelcolor="#b35c00")
    ax2.set_ylim(bottom=0)
    ax2b.set_ylim(bottom=0)
    ax2.set_xlabel("Waktu sejak awal jendela (jam)")
    ax2.set_title(r"(b) Besaran meluruh: $\tau_{\rm insulin}$ = 240 menit, "
                  r"$\tau_{\rm karbohidrat}$ = 180 menit",
                  loc="left", fontsize=10.5)
    ax2.grid(alpha=0.25, ls=":")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8.5, framealpha=0.9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"pasien        : {pid}")
    print(f"jendela       : {t.iloc[0]}  ->  {t.iloc[-1]}  ({jam.iloc[-1]:.1f} jam)")
    print(f"bolus / makan : {int(mb.sum())} / {int(mc.sum())} kejadian")
    print(f"IOB puncak    : {jendela['iob'].max():.2f} unit")
    print(f"COB puncak    : {jendela['cob'].max():.1f} g")
    print(f"tersimpan     : {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
