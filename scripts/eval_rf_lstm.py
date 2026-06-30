"""T4: Evaluasi komparatif RF vs LSTM — RMSE, MAE, Clarke Error Grid.

Jalankan:
    python scripts/eval_rf_lstm.py                  # pakai ohio_t1dm (default)
    python scripts/eval_rf_lstm.py --smbg           # downsample ke SMBG cadence dulu

Output:
    results/eval_prediksi/comparison_metrics.csv
    results/eval_prediksi/clarke_grid_rf.png
    results/eval_prediksi/clarke_grid_lstm.png
    results/eval_prediksi/clarke_grid_comparison.png
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


# ---------------------------------------------------------------------------
# Clarke Error Grid plotting
# ---------------------------------------------------------------------------

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
    zone_colors = []
    from src.utils.metrics import clarke_error_grid
    zones_per_point = []
    for tv, pv in zip(y_true, y_pred):
        if (tv < 70 and pv < 70) or abs(tv - pv) <= 0.2 * tv:
            zones_per_point.append("A")
        elif 70 <= tv <= 180 and 70 <= pv <= 180:
            zones_per_point.append("B")
        elif (tv < 70 and pv > 180) or (tv > 180 and pv < 70):
            zones_per_point.append("C")
        elif (tv < 70 and 70 <= pv <= 180) or (tv > 180 and 70 <= pv <= 180):
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


def plot_comparison_bar(rf_metrics: dict, lstm_metrics: dict, out_path: Path) -> None:
    """Side-by-side bar chart: RMSE, MAE, Clarke-A for RF vs LSTM."""
    keys = ["RMSE", "MAE", "Clarke_A", "Clarke_A+B"]
    labels = ["RMSE (mg/dL)", "MAE (mg/dL)", "Clarke A (%)", "Clarke A+B (%)"]
    rf_vals = [rf_metrics[k] for k in keys]
    lstm_vals = [lstm_metrics[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars_rf = ax.bar(x - width / 2, rf_vals, width, label="Random Forest", color="#3498db", alpha=0.85)
    bars_lstm = ax.bar(x + width / 2, lstm_vals, width, label="LSTM", color="#e74c3c", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("RF vs LSTM — Metric Comparison", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.bar_label(bars_rf, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(bars_lstm, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Comparison chart saved: {out_path}")


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def evaluate_both_models(config_path: str = "config.yaml", smbg: bool = False) -> None:
    from src.data.loader import DiabetesDataLoader
    from src.data.preprocessor import DataPreprocessor
    from src.models.rf_model import RandomForestGlucoseModel
    from src.models.lstm_model import LSTMGlucoseModel
    from src.utils.metrics import calculate_all_metrics

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DiabetesDataLoader(config["data"]["output_dir"])
    primary = config.get("data", {}).get("primary_source", "ohio_t1dm")
    fallback = config.get("data", {}).get("fallback_source", "latest_generated")
    df, used_source = loader.load_preferred_dataset(primary, fallback)
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    print(f"Data source   : {used_source}")

    preprocessor = DataPreprocessor(config)
    df = preprocessor.handle_missing_values(df)

    if smbg:
        interval = config.get("data", {}).get("smbg_interval_min", 240)
        df = preprocessor.downsample_smbg(df, interval_minutes=interval)
        seq_len = config["model"].get("smbg_sequence_length", 6)
        mode_tag = "smbg"
    else:
        seq_len = config["model"].get("sequence_length", 12)
        mode_tag = "cgm"

    patient_ids = sorted(df["patient_id"].unique().tolist())
    test_patients = patient_ids[-2:]
    train_df, test_df = preprocessor.split_by_patient(df, test_patients)

    X_train, y_train = preprocessor.create_sequences(train_df, sequence_length=seq_len)
    X_test, y_test = preprocessor.create_sequences(test_df, sequence_length=seq_len)
    X_train_s, X_test_s = preprocessor.normalize_data(X_train, X_test)

    out_dir = Path("results") / "eval_prediksi"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- RF ----
    print("\n[1/2] Training Random Forest...")
    rf = RandomForestGlucoseModel(config)
    rf.train(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_metrics = calculate_all_metrics(y_test, rf_pred)
    print(f"  RF  RMSE={rf_metrics['RMSE']:.3f}  MAE={rf_metrics['MAE']:.3f}  Clarke-A={rf_metrics['Clarke_A']:.2f}%")
    plot_clarke_grid(y_test, rf_pred, rf_metrics, "Clarke Error Grid — Random Forest",
                     out_dir / f"clarke_grid_rf_{mode_tag}.png")

    # ---- LSTM ----
    print("\n[2/2] Training LSTM...")
    lstm = LSTMGlucoseModel(config)
    lstm.train(X_train_s, y_train, X_test_s, y_test)
    lstm_pred = lstm.predict(X_test_s)
    lstm_metrics = calculate_all_metrics(y_test, lstm_pred)
    print(f"  LSTM RMSE={lstm_metrics['RMSE']:.3f}  MAE={lstm_metrics['MAE']:.3f}  Clarke-A={lstm_metrics['Clarke_A']:.2f}%")
    plot_clarke_grid(y_test, lstm_pred, lstm_metrics, "Clarke Error Grid — LSTM",
                     out_dir / f"clarke_grid_lstm_{mode_tag}.png")

    # ---- Comparison chart ----
    plot_comparison_bar(rf_metrics, lstm_metrics, out_dir / f"clarke_grid_comparison_{mode_tag}.png")

    # ---- CSV ----
    rows = []
    for k in ["RMSE", "MAE", "MAPE", "Clarke_A", "Clarke_B", "Clarke_C", "Clarke_D", "Clarke_E", "Clarke_A+B"]:
        rows.append({"metric": k, "RF": rf_metrics[k], "LSTM": lstm_metrics[k]})
    df_cmp = pd.DataFrame(rows)
    csv_path = out_dir / f"comparison_metrics_{mode_tag}.csv"
    df_cmp.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Comparison CSV: {csv_path}")

    # ---- JSON for each model ----
    with open(out_dir / f"rf_metrics_{mode_tag}.json", "w") as f:
        json.dump({k: float(v) for k, v in rf_metrics.items()}, f, indent=2)
    with open(out_dir / f"lstm_metrics_{mode_tag}.json", "w") as f:
        json.dump({k: float(v) for k, v in lstm_metrics.items()}, f, indent=2)

    # ---- Interpretation ----
    print("\n" + "=" * 60)
    print("INTERPRETASI")
    print("=" * 60)
    if rf_metrics["RMSE"] <= lstm_metrics["RMSE"]:
        diff = lstm_metrics["RMSE"] - rf_metrics["RMSE"]
        print(f"RF lebih unggul: RMSE RF={rf_metrics['RMSE']:.3f} vs LSTM={lstm_metrics['RMSE']:.3f} (diff {diff:.3f} mg/dL)")
        print("Justifikasi empiris: RF dipilih sebagai model utama karena lebih akurat")
        print("pada data sparse OhioT1DM yang di-surrogate-kan untuk skenario SMBG.")
    else:
        diff = rf_metrics["RMSE"] - lstm_metrics["RMSE"]
        print(f"LSTM lebih unggul: RMSE LSTM={lstm_metrics['RMSE']:.3f} vs RF={rf_metrics['RMSE']:.3f} (diff {diff:.3f} mg/dL)")
    print("=" * 60)

    print(f"\nSemua hasil tersimpan di: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T4: Evaluate RF vs LSTM")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--smbg", action="store_true",
                        help="Downsample to SMBG cadence before evaluation")
    args = parser.parse_args()
    evaluate_both_models(args.config, args.smbg)
