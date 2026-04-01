import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data.loader import DiabetesDataLoader


st.set_page_config(page_title="Prediction", page_icon="📈", layout="wide")
st.title("📈 Glucose Prediction")
st.caption("Prediksi 1 langkah ke depan menggunakan baseline Random Forest")


@st.cache_data
def load_metrics(metrics_path: str):
	path = Path(metrics_path)
	if not path.exists():
		return None
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


@st.cache_resource
def load_inference_artifacts(bundle_path: str, model_path: str):
	bundle_file = Path(bundle_path)
	if bundle_file.exists():
		with open(bundle_file, "rb") as f:
			bundle = pickle.load(f)
		return {
			"mode": "bundle",
			"model": bundle["model"],
			"scaler": bundle.get("scaler"),
			"features": bundle.get("features", ["glucose", "carbs", "insulin", "activity", "stress"]),
			"sequence_length": int(bundle.get("sequence_length", 12)),
		}

	model_file = Path(model_path)
	if model_file.exists():
		with open(model_file, "rb") as f:
			model = pickle.load(f)
		inferred_features = ["glucose", "carbs", "insulin", "activity", "stress"]
		n_features_in = getattr(model, "n_features_in_", len(inferred_features) * 12)
		seq_len = max(1, int(n_features_in / len(inferred_features)))
		return {
			"mode": "model_only",
			"model": model,
			"scaler": None,
			"features": inferred_features,
			"sequence_length": seq_len,
		}

	return None


metrics = load_metrics("models/rf_baseline_metrics.json")
artifacts = load_inference_artifacts("models/rf_inference_bundle.pkl", "models/rf_baseline.pkl")

if artifacts is None:
	st.error("Model belum tersedia. Jalankan training dulu dari CLI:")
	st.code("python -m src.models.rf_model --config config.yaml")
	st.stop()

if artifacts["mode"] == "model_only":
	st.warning(
		"Terbaca hanya model tanpa scaler. Prediksi masih bisa jalan, "
		"tapi sebaiknya retrain sekali lagi agar bundle inference dibuat."
	)

if metrics:
	st.subheader("Baseline Metrics")
	cols = st.columns(4)
	cols[0].metric("RMSE", f"{metrics['RMSE']:.4f}")
	cols[1].metric("MAE", f"{metrics['MAE']:.4f}")
	cols[2].metric("MAPE", f"{metrics['MAPE']:.2f}%")
	cols[3].metric("Clarke A+B", f"{metrics['Clarke_A+B']:.2f}%")

st.subheader("Run Prediction")

loader = DiabetesDataLoader("data/raw")
try:
	df = loader.load_latest_dataset().sort_values(["patient_id", "timestamp"])
except FileNotFoundError:
	st.error("Dataset tidak ditemukan di data/raw")
	st.stop()

patient_ids = sorted(df["patient_id"].unique().tolist())
if not patient_ids:
	st.error("Dataset kosong")
	st.stop()

selected_patient = st.selectbox("Pilih Patient ID", patient_ids)
patient_df = df[df["patient_id"] == selected_patient].copy()

features = artifacts["features"]
sequence_length = artifacts["sequence_length"]

if len(patient_df) < sequence_length:
	st.error(f"Data pasien {selected_patient} kurang dari {sequence_length} baris")
	st.stop()

window_df = patient_df.tail(sequence_length).copy().reset_index(drop=True)

st.write("Data window terbaru yang dipakai model")
st.dataframe(window_df[["timestamp"] + features], use_container_width=True)

st.markdown("### Optional Override Input Terakhir")
last_row = window_df.iloc[-1].copy()
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
	last_row["glucose"] = st.number_input("Glucose", value=float(last_row["glucose"]), min_value=40.0, max_value=400.0)
with col2:
	last_row["carbs"] = st.number_input("Carbs", value=float(last_row["carbs"]), min_value=0.0, max_value=200.0)
with col3:
	last_row["insulin"] = st.number_input("Insulin", value=float(last_row["insulin"]), min_value=0.0, max_value=30.0)
with col4:
	last_row["activity"] = st.number_input("Activity", value=float(last_row["activity"]), min_value=0.0, max_value=180.0)
with col5:
	last_row["stress"] = st.number_input("Stress", value=float(last_row["stress"]), min_value=1.0, max_value=10.0)

window_df.loc[len(window_df) - 1, features] = [last_row[f] for f in features]

if st.button("Predict Next Glucose", type="primary"):
	X = window_df[features].values.astype(float)
	if artifacts["scaler"] is not None:
		X = artifacts["scaler"].transform(X)

	X_flat = X.reshape(1, -1)
	pred = float(artifacts["model"].predict(X_flat)[0])

	st.success(f"Predicted next glucose: {pred:.2f} mg/dL")

	if pred < 70:
		st.error("Risk level: HYPOGLYCEMIA")
	elif pred > 180:
		st.warning("Risk level: HYPERGLYCEMIA")
	else:
		st.info("Risk level: IN RANGE")

	history = window_df["glucose"].tolist() + [pred]
	chart_df = pd.DataFrame({"step": np.arange(len(history)), "glucose": history})
	st.line_chart(chart_df.set_index("step"))
