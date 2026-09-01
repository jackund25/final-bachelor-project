"""Gambar komparatif model prakiraan — Clarke Error Grid dan batang metrik.

Skrip ini TIDAK MELATIH apa pun. Ia menggambar dari vektor prediksi yang disimpan
masing-masing modul model pada saat pelatihan.

MENGAPA BEGITU (24 Agustus 2026). Sebelumnya skrip ini melatih ulang ketiga model,
terpisah dari pelatihan yang menghasilkan tabel metrik. Dua jalur pelatihan untuk satu
klaim membuka celah yang pernah benar-benar terjadi di proyek ini: gambar Clarke Error
Grid memperlihatkan model yang BUKAN model produksi, dan itu tidak ketahuan karena
angkanya tak pernah dibandingkan langsung. Dengan menggambar dari vektor prediksi yang
tersimpan, gambar dan tabel dijamin berasal dari model yang sama — dan gambarnya dapat
dibuat ulang kapan pun tanpa biaya komputasi.

Prasyarat: latih lebih dulu lengan yang ingin digambar.

    python -m src.models.gbm_model  --source CGM
    python -m src.models.rf_model   --source CGM
    python -m src.models.lstm_model --source CGM

    python scripts/eval_rf_lstm.py --source CGM

Output per horizon, di results/eval_prediksi/h<menit>m/:
    comparison_metrics.csv
    clarke_grid_gbm.png     <- model produksi
    clarke_grid_rf.png
    clarke_grid_lstm.png
    comparison_bar.png
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


# Clarke Error Grid plotting

def _clarke_zones_to_color(zone: str) -> str:
    return {"A": "#2ecc71", "B": "#f1c40f", "C": "#e67e22", "D": "#e74c3c", "E": "#8e44ad"}[zone]


def plot_clarke_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    title: str,
    out_path: Path,
) -> None:
    """Render Clarke Error Grid scatter with zone boundaries and save to PNG."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel("Actual Glucose (mg/dL)", fontsize=12)
    ax.set_ylabel("Predicted Glucose (mg/dL)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Zone boundary lines (approximate standard Clarke grid)
    # Zone A upper boundary
    ax.plot([0, 58.33], [0, 70], "k--", lw=0.8)
    ax.plot([58.33, 400], [70, 400 * (70 / 58.33)], "k--", lw=0.8)  # not real; use standard pts
    # Reference diagonal
    ax.plot([0, 400], [0, 400], "k-", lw=1.0, alpha=0.4)
    # ±20% bands
    ax.fill_between([70, 400], [58.33, 320], [84, 480], alpha=0.06, color="green")

    # Colour-code by zone
    # Ambang Clarke dari src/constants.py (CLARKE_*, sengaja terpisah dari ambang
    # kebijakan risiko agar metrik terbitan ini tidak ikut berubah).
    from src.constants import CLARKE_HIGH, CLARKE_LOW

    zone_colors = []
    zones_per_point = []
    for tv, pv in zip(y_true, y_pred):
        if (tv < CLARKE_LOW and pv < CLARKE_LOW) or abs(tv - pv) <= 0.2 * tv:
            zones_per_point.append("A")
        elif CLARKE_LOW <= tv <= CLARKE_HIGH and CLARKE_LOW <= pv <= CLARKE_HIGH:
            zones_per_point.append("B")
        elif (tv < CLARKE_LOW and pv > CLARKE_HIGH) or (tv > CLARKE_HIGH and pv < CLARKE_LOW):
            zones_per_point.append("C")
        elif (tv < CLARKE_LOW and CLARKE_LOW <= pv <= CLARKE_HIGH) or (tv > CLARKE_HIGH and CLARKE_LOW <= pv <= CLARKE_HIGH):
            zones_per_point.append("D")
        else:
            zones_per_point.append("E")

    for zone in ["E", "D", "C", "B", "A"]:
        mask = np.array(zones_per_point) == zone
        ax.scatter(
            y_true[mask], y_pred[mask],
            c=_clarke_zones_to_color(zone), s=8, alpha=0.5, label=zone, zorder=3
        )

    # Stats box
    stat_text = (
        f"RMSE : {metrics['RMSE']:.2f} mg/dL\n"
        f"MAE  : {metrics['MAE']:.2f} mg/dL\n"
        f"Zone A: {metrics['Clarke_A']:.1f}%\n"
        f"A+B  : {metrics['Clarke_A+B']:.1f}%"
    )
    ax.text(
        0.03, 0.97, stat_text,
        transform=ax.transAxes, fontsize=9,
        verticalalignment="top", family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    patches = [mpatches.Patch(color=_clarke_zones_to_color(z), label=f"Zone {z}") for z in "ABCDE"]
    ax.legend(handles=patches, loc="lower right", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Clarke grid saved: {out_path}")


def plot_comparison_bar(hasil: "list[tuple[str, dict]]", out_path: Path,
                        judul: str = "Perbandingan Metrik Antar-Model") -> None:
    """Grafik batang berdampingan untuk N model.

    Sebelumnya fungsi ini menerima tepat dua model (RF dan LSTM). Ia digeneralkan
    menjadi daftar ``(nama, metrik)`` supaya model produksi *gradient boosting* dapat
    ikut ditampilkan tanpa menulis fungsi kedua yang isinya nyaris sama.
    """
    keys = ["RMSE", "MAE", "Clarke_A", "Clarke_A+B"]
    labels = ["RMSE (mg/dL)", "MAE (mg/dL)", "Clarke A (%)", "Clarke A+B (%)"]
    warna = ["#2ecc71", "#3498db", "#e74c3c", "#9b59b6"]

    x = np.arange(len(keys))
    n = len(hasil)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (nama, m) in enumerate(hasil):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, [m[k] for k in keys], width,
                      label=nama, color=warna[i % len(warna)], alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(judul, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison chart saved: {out_path}")


# Main evaluation routine

LENGAN = [
    ("gbm", "Gradient Boosting"),
    ("rf", "Random Forest"),
    ("lstm", "LSTM"),
]


def _gambar_satu_horizon(source, horizon_min, out_root):
    """Gambar Clarke grid tiap lengan pada satu horizon, dari prediksi tersimpan.

    Lengan yang belum dilatih DILEWATI dengan peringatan alih-alih menggagalkan
    skrip: seseorang boleh saja hanya ingin menggambar model produksi.
    """
    from src.models.persiapan_data import berkas_prediksi
    from src.utils.metrics import calculate_all_metrics

    minutes = horizon_min
    out_dir = out_root / f"h{int(horizon_min)}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"{source} +{minutes:g} menit")
    print("=" * 60)

    hasil = []
    jumlah = set()

    for kunci, nama in LENGAN:
        jalur = berkas_prediksi(kunci, source, horizon_min)

        if not jalur.exists():
            print(f"  [lewat] {nama}: prediksi belum ada -> {jalur.name}")
            continue

        with np.load(jalur) as d:
            y_true, y_pred = d["y_true"], d["y_pred"]

        jumlah.add(len(y_true))

        m = calculate_all_metrics(y_true, y_pred)
        hasil.append((nama, m))

        print(
            f"  {nama:18s} n={len(y_true):6d}  RMSE={m['RMSE']:.3f}  "
            f"MAE={m['MAE']:.3f}  Clarke-A+B={m['Clarke_A+B']:.2f}%"
        )

        plot_clarke_grid(
            y_true, y_pred, m,
            f"Clarke Error Grid — {nama} ({source} +{minutes:g} mnt)",
            out_dir / f"clarke_grid_{kunci}.png",
        )

    if not hasil:
        print("  Tidak ada lengan yang dapat digambar pada horizon ini.")
        return []

    # Seluruh lengan WAJIB dinilai pada jumlah jendela yang sama. Bila tidak, tabel
    # dan gambarnya membandingkan himpunan uji yang berbeda — persis jenis
    # ketidaksebandingan senyap yang menjadi alasan berkas ini ditulis ulang.
    if len(jumlah) > 1:
        print(
            f"  PERINGATAN: jumlah jendela berbeda antar-lengan {sorted(jumlah)}. "
            "Angka ini TIDAK sebanding; latih ulang dari konfigurasi yang sama."
        )

    plot_comparison_bar(
        hasil, out_dir / "comparison_bar.png",
        judul=f"Perbandingan Metrik Antar-Model ({source} +{minutes:g} mnt)",
    )

    rows = [
        {"metric": k, **{nama: m[k] for nama, m in hasil}}
        for k in ["RMSE", "MAE", "MAPE", "Clarke_A", "Clarke_B",
                  "Clarke_C", "Clarke_D", "Clarke_E", "Clarke_A+B"]
    ]
    pd.DataFrame(rows).to_csv(
        out_dir / "comparison_metrics.csv", index=False, float_format="%.4f"
    )
    return hasil


def gambar_perbandingan(config_path: str = "config.yaml", source: str = "CGM") -> None:
    from src.models.persiapan_data import resolve_profil_sumber

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    profil = resolve_profil_sumber(config, source)

    print(f"Modalitas   : {source}")
    print(f"Horizon     : {', '.join(f'+{h:g} mnt' for h in profil.horizons_min)}")
    print("Sumber angka: vektor prediksi tersimpan (TIDAK melatih ulang)")

    out_root = Path("results") / "eval_prediksi"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for h in profil.horizons_min:
        hasil = _gambar_satu_horizon(source, h, out_root)
        for model_name, m in hasil:
            summary.append({
                "source": source, "horizon_min": h, "model": model_name,
                "RMSE": round(m["RMSE"], 3), "MAE": round(m["MAE"], 3), "MAPE": round(m["MAPE"], 3),
                "Clarke_A": round(m["Clarke_A"], 2), "Clarke_A+B": round(m["Clarke_A+B"], 2),
                "split": "official_dataset_split (temporal within-patient)",
            })

    if not summary:
        raise SystemExit(
            "Tidak ada prediksi tersimpan. Latih lengan yang diinginkan lebih dulu, "
            "misalnya `python -m src.models.gbm_model --source CGM`."
        )

    summary_df = pd.DataFrame(summary)
    summary_path = out_root / f"summary_{source.lower()}.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("RINGKASAN SEMUA HORIZON")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nRingkasan -> {summary_path}")
    print(f"Detail per horizon -> {out_root}/h<menit>m/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gambar perbandingan GBM, RF, dan LSTM dari prediksi tersimpan")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--source",
        default="CGM",
        choices=["CGM", "FINGER_STICK"],
        help="Modalitas observasi glukosa",
    )
    args = parser.parse_args()
    gambar_perbandingan(args.config, args.source)
