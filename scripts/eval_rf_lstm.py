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

def _evaluate_one_horizon(config, df_clean, seq_len, horizon, out_root, feature_list, predict_delta):
    """Latih & evaluasi RF vs LSTM pada satu horizon. Return (rf_metrics, lstm_metrics)."""
    from src.data.preprocessor import DataPreprocessor
    from src.models.rf_model import RandomForestGlucoseModel
    from src.models.lstm_model import LSTMGlucoseModel
    from src.utils.metrics import calculate_all_metrics

    minutes = horizon * 5
    print("\n" + "=" * 60)
    print(f"HORIZON {horizon} langkah (+{minutes} menit) | fitur={len(feature_list)} | delta={predict_delta}")
    print("=" * 60)

    prep = DataPreprocessor(config)  # scaler baru per horizon
    prep.feature_columns = list(feature_list)
    pids = sorted(df_clean["patient_id"].unique().tolist())
    train_df, test_df = prep.split_by_patient(df_clean, pids[-2:])

    X_train, y_train, anc_train = prep.create_sequences(train_df, seq_len, horizon, return_anchor=True)
    X_test, y_test, anc_test = prep.create_sequences(test_df, seq_len, horizon, return_anchor=True)
    X_train_s, X_test_s = prep.normalize_data(X_train, X_test)

    # Target: absolut, atau delta (selisih dari glukosa terakhir di window) lalu direkonstruksi
    y_train_fit = (y_train - anc_train) if predict_delta else y_train
    y_test_fit = (y_test - anc_test) if predict_delta else y_test

    out_dir = out_root / f"h{horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Random Forest...")
    rf = RandomForestGlucoseModel(config)
    rf.train(X_train_s, y_train_fit)
    rf_pred = rf.predict(X_test_s)
    if predict_delta:
        rf_pred = rf_pred + anc_test
    rf_m = calculate_all_metrics(y_test, rf_pred)
    print(f"  RF   RMSE={rf_m['RMSE']:.3f}  MAE={rf_m['MAE']:.3f}  Clarke-A+B={rf_m['Clarke_A+B']:.2f}%")
    plot_clarke_grid(y_test, rf_pred, rf_m, f"Clarke Error Grid — RF (+{minutes} mnt)",
                     out_dir / "clarke_grid_rf.png")

    print("[2/2] LSTM...")
    lstm = LSTMGlucoseModel(config)
    lstm.train(X_train_s, y_train_fit, X_test_s, y_test_fit)
    lstm_pred = lstm.predict(X_test_s)
    if predict_delta:
        lstm_pred = lstm_pred + anc_test
    lstm_m = calculate_all_metrics(y_test, lstm_pred)
    print(f"  LSTM RMSE={lstm_m['RMSE']:.3f}  MAE={lstm_m['MAE']:.3f}  Clarke-A+B={lstm_m['Clarke_A+B']:.2f}%")
    plot_clarke_grid(y_test, lstm_pred, lstm_m, f"Clarke Error Grid — LSTM (+{minutes} mnt)",
                     out_dir / "clarke_grid_lstm.png")

    plot_comparison_bar(rf_m, lstm_m, out_dir / "comparison_bar.png")

    rows = [{"metric": k, "RF": rf_m[k], "LSTM": lstm_m[k]}
            for k in ["RMSE", "MAE", "MAPE", "Clarke_A", "Clarke_B", "Clarke_C", "Clarke_D", "Clarke_E", "Clarke_A+B"]]
    pd.DataFrame(rows).to_csv(out_dir / "comparison_metrics.csv", index=False, float_format="%.4f")
    return rf_m, lstm_m


def evaluate_both_models(config_path: str = "config.yaml") -> None:
    from src.data.loader import DiabetesDataLoader
    from src.data.preprocessor import DataPreprocessor

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    loader = DiabetesDataLoader(config["data"]["output_dir"])
    primary = config.get("data", {}).get("primary_source", "ohio_t1dm")
    fallback = config.get("data", {}).get("fallback_source", "latest_generated")
    df, used_source = loader.load_preferred_dataset(primary, fallback)
    df = df.sort_values(["patient_id", "timestamp"]).reset_index(drop=True)
    print(f"Data source   : {used_source}")

    df_clean = DataPreprocessor(config).handle_missing_values(df)
    seq_len = config["model"].get("sequence_length", 12)
    horizons = config["model"].get("prediction_horizons", [config["model"].get("default_horizon", 1)])

    # Feature engineering (opsional, dari config)
    use_eng = config["model"].get("use_engineered", False)
    predict_delta = config["model"].get("predict_delta", False)
    if use_eng:
        fe = config["model"].get("feature_engineering", {})
        df_clean = DataPreprocessor(config).engineer_features(df_clean, **fe)
        feature_list = config["model"]["engineered_features"]
        print(f"Mode fitur    : ENGINEERED ({len(feature_list)} fitur) | predict_delta={predict_delta}")
    else:
        feature_list = config["model"]["features"]
        print(f"Mode fitur    : BASELINE ({len(feature_list)} fitur) | predict_delta={predict_delta}")

    out_root = Path("results") / "eval_prediksi"
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for h in horizons:
        rf_m, lstm_m = _evaluate_one_horizon(config, df_clean, seq_len, h, out_root, feature_list, predict_delta)
        for model_name, m in [("RF", rf_m), ("LSTM", lstm_m)]:
            summary.append({
                "horizon_steps": h, "horizon_min": h * 5, "model": model_name,
                "RMSE": round(m["RMSE"], 3), "MAE": round(m["MAE"], 3), "MAPE": round(m["MAPE"], 3),
                "Clarke_A": round(m["Clarke_A"], 2), "Clarke_A+B": round(m["Clarke_A+B"], 2),
            })

    summary_df = pd.DataFrame(summary)
    summary_path = out_root / "summary_all_horizons.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("RINGKASAN SEMUA HORIZON")
    print("=" * 60)
    print(summary_df.to_string(index=False))
    print(f"\nRingkasan -> {summary_path}")
    print(f"Detail per horizon -> {out_root}/h<langkah>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T4: Evaluate RF vs LSTM across horizons")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate_both_models(args.config)
