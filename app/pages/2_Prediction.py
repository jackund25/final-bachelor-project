import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

from src.data.loader import DiabetesDataLoader
from src.rag import RAGPipeline


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


@st.cache_data
def load_data_policy(config_path: str = "config.yaml"):
	path = Path(config_path)
	if not path.exists():
		return {"primary_source": "ohio_t1dm", "fallback_source": "latest_generated"}
	with open(path, "r", encoding="utf-8") as f:
		cfg = yaml.safe_load(f) or {}
	data_cfg = cfg.get("data", {})
	return {
		"primary_source": data_cfg.get("primary_source", "ohio_t1dm"),
		"fallback_source": data_cfg.get("fallback_source", "latest_generated"),
	}


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


@st.cache_resource
def load_rag_pipeline(knowledge_base_dir: str = "data/knowledge_base"):
	pipeline = RAGPipeline(kb_dir=knowledge_base_dir, llm_provider="template")
	pipeline.build()
	return pipeline


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
manual_logbook_path = Path("data/raw/manual_logbook.csv")
data_policy = load_data_policy("config.yaml")

try:
	df, resolved_source = loader.load_preferred_dataset(
		data_policy["primary_source"],
		data_policy["fallback_source"],
	)
	df = df.sort_values(["patient_id", "timestamp"])
	st.info(
		f"Auto data source aktif: primary='{data_policy['primary_source']}', "
		f"fallback='{data_policy['fallback_source']}', digunakan='{resolved_source}'."
	)
except FileNotFoundError as exc:
	st.error(f"Dataset tidak ditemukan: {exc}")
	st.stop()

patient_ids = sorted(df["patient_id"].unique().tolist())
if not patient_ids:
	st.error("Dataset kosong")
	st.stop()

source_mode = st.radio(
    "Data Source",
	["Auto (OhioT1DM -> Generated)", "Generated dataset", "Manual logbook"],
    horizontal=True,
)

if source_mode == "Manual logbook" and manual_logbook_path.exists():
    source_df = loader.load_csv("manual_logbook.csv").sort_values(["patient_id", "timestamp"])
    patient_ids = sorted(source_df["patient_id"].unique().tolist()) or patient_ids
    st.info("Prediction akan memprioritaskan data dari manual logbook.")
elif source_mode == "Generated dataset":
	source_df = loader.load_latest_dataset().sort_values(["patient_id", "timestamp"])
	st.info("Prediction menggunakan generated dataset.")
else:
    source_df = df
    if source_mode == "Manual logbook":
        st.warning("Manual logbook belum ada. Menggunakan generated dataset sebagai fallback.")

selected_patient = st.selectbox("Pilih Patient ID", patient_ids)


def build_prediction_window(source_data: pd.DataFrame, fallback_data: pd.DataFrame, patient_id: str, seq_len: int) -> pd.DataFrame:
	"""Build a sequence window prioritizing source data and falling back to generated data if needed."""
	source_patient = source_data[source_data["patient_id"] == patient_id].copy()
	if not source_patient.empty:
		source_patient = source_patient.sort_values("timestamp")

	if len(source_patient) >= seq_len:
		return source_patient.tail(seq_len).reset_index(drop=True)

	fallback_patient = fallback_data[fallback_data["patient_id"] == patient_id].copy().sort_values("timestamp")
	combined = pd.concat([fallback_patient, source_patient], ignore_index=True)
	combined = combined.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

	if len(combined) < seq_len:
		return combined.reset_index(drop=True)

	return combined.tail(seq_len).reset_index(drop=True)


def build_patient_state(row: pd.Series) -> dict:
	"""Build a compact patient state for the RAG explanation layer."""
	return {
		"current_glucose": float(row.get("glucose", 100.0)),
		"insulin_on_board": float(row.get("insulin", 0.0)),
		"carbs_on_board": float(row.get("carbs", 0.0)),
		"activity_level": int(float(row.get("activity", 0))),
		"stress_level": int(float(row.get("stress", 5))),
	}


patient_df = build_prediction_window(source_df, df, selected_patient, artifacts["sequence_length"])

features = artifacts["features"]
sequence_length = artifacts["sequence_length"]

if len(patient_df) < sequence_length:
	st.warning(
		f"Data pasien {selected_patient} belum cukup {sequence_length} baris. "
		"Prediction akan memakai data yang tersedia, atau fallback dari generated dataset bila ada."
	)
	st.dataframe(patient_df[["timestamp"] + [c for c in features if c in patient_df.columns]], use_container_width=True)
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

	# Chart data from native pandas (Narwhals wrapper stripped at loader level)
	history_values = window_df["glucose"].tolist() + [float(pred)]
	fig, ax = plt.subplots(figsize=(8, 3))
	ax.plot(range(1, len(history_values) + 1), history_values, marker="o", linewidth=2)
	ax.set_title("Glucose Trajectory")
	ax.set_xlabel("Step")
	ax.set_ylabel("mg/dL")
	ax.grid(alpha=0.25)
	st.pyplot(fig, clear_figure=True)

	st.markdown("### Clinical Explanation")
	rag_pipeline = load_rag_pipeline("data/knowledge_base")
	patient_state = build_patient_state(window_df.iloc[-1])
	rag_result = rag_pipeline.answer(patient_state=patient_state, prediction=pred)

	st.info(rag_result["explanation"])
	with st.expander("Retrieved medical context"):
		for doc in rag_result["retrieved_docs"]:
			st.markdown(f"**Rank {doc['rank']}** - source: `{doc['source']}` - similarity: {doc['similarity']:.2f}")
			st.write(doc["text"])
			st.markdown("---")

st.caption("Jika memilih Manual logbook, prediction akan memprioritaskan input pasien terbaru dari data/raw/manual_logbook.csv.")
