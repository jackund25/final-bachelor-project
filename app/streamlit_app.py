from pathlib import Path
import json

import pandas as pd
import streamlit as st


st.set_page_config(
	page_title="Diabetes Digital Twin",
	page_icon="🩺",
	layout="wide",
	initial_sidebar_state="expanded",
)


def _file_exists(path: str) -> bool:
	return Path(path).exists()


def _load_metrics(path: str):
	metrics_path = Path(path)
	if not metrics_path.exists():
		return None
	with open(metrics_path, "r", encoding="utf-8") as f:
		return json.load(f)


def _dataset_summary(path: str):
	csv_path = Path(path)
	if not csv_path.exists():
		return None

	df = pd.read_csv(csv_path, nrows=5000)
	summary = {
		"rows_previewed": len(df),
		"patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
		"glucose_mean": float(df["glucose"].mean()) if "glucose" in df.columns else None,
		"glucose_min": float(df["glucose"].min()) if "glucose" in df.columns else None,
		"glucose_max": float(df["glucose"].max()) if "glucose" in df.columns else None,
	}
	return summary


st.markdown(
	"""
	<style>
	.hero {
		padding: 1.1rem 1.3rem;
		border-radius: 16px;
		background: linear-gradient(120deg, #0e3b43 0%, #1d6f84 45%, #f2a35e 100%);
		color: #f8fbfd;
		margin-bottom: 1rem;
	}
	.hero h1 {
		margin: 0;
		font-size: 2rem;
		line-height: 1.2;
		letter-spacing: 0.3px;
	}
	.hero p {
		margin: 0.35rem 0 0 0;
		font-size: 1.02rem;
		opacity: 0.95;
	}
	.tile {
		border: 1px solid rgba(27, 48, 58, 0.15);
		border-radius: 14px;
		padding: 0.9rem 1rem;
		background: #fbfdff;
		min-height: 115px;
	}
	.tile h4 {
		margin: 0 0 0.3rem 0;
		color: #1d3642;
	}
	.tile p {
		margin: 0;
		color: #2d4a57;
		font-size: 0.95rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
	<div class="hero">
	  <h1>Diabetes Digital Twin Dashboard</h1>
	  <p>Platform patient-centered untuk prediksi glukosa, simulasi skenario, dan dukungan keputusan harian.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

st.subheader("Module Navigation")
left, mid, right = st.columns(3)

with left:
	st.markdown(
		"""
		<div class="tile">
		  <h4>1. Input Logbook</h4>
		  <p>Catat data harian pasien: glukosa, karbohidrat, insulin, aktivitas, dan stres.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

with mid:
	st.markdown(
		"""
		<div class="tile">
		  <h4>2. Prediction</h4>
		  <p>Prediksi glukosa langkah berikutnya dengan baseline model Random Forest.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

with right:
	st.markdown(
		"""
		<div class="tile">
		  <h4>3. What-If Simulator</h4>
		  <p>Eksplorasi dampak perubahan makan, aktivitas, atau insulin sebelum intervensi nyata.</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

st.markdown("---")
st.subheader("System Status")

metrics = _load_metrics("models/rf_baseline_metrics.json")
dataset_info = _dataset_summary("data/raw/training_data_complete.csv")

status_cols = st.columns(4)
status_cols[0].metric("Model File", "Ready" if _file_exists("models/rf_baseline.pkl") else "Missing")
status_cols[1].metric("Inference Bundle", "Ready" if _file_exists("models/rf_inference_bundle.pkl") else "Missing")
status_cols[2].metric("Metrics JSON", "Ready" if _file_exists("models/rf_baseline_metrics.json") else "Missing")
status_cols[3].metric("Dataset CSV", "Ready" if _file_exists("data/raw/training_data_complete.csv") else "Missing")

if metrics is not None:
	st.markdown("### Baseline Performance")
	m1, m2, m3, m4 = st.columns(4)
	m1.metric("RMSE", f"{metrics.get('RMSE', 0):.4f}")
	m2.metric("MAE", f"{metrics.get('MAE', 0):.4f}")
	m3.metric("MAPE", f"{metrics.get('MAPE', 0):.2f}%")
	m4.metric("Clarke A+B", f"{metrics.get('Clarke_A+B', 0):.2f}%")

if dataset_info is not None:
	st.markdown("### Dataset Snapshot")
	d1, d2, d3, d4 = st.columns(4)
	d1.metric("Rows (sampled)", f"{dataset_info['rows_previewed']:,}")
	d2.metric("Patients", dataset_info["patients"])
	d3.metric("Mean Glucose", f"{dataset_info['glucose_mean']:.1f} mg/dL" if dataset_info["glucose_mean"] is not None else "N/A")
	d4.metric(
		"Range",
		f"{dataset_info['glucose_min']:.1f} - {dataset_info['glucose_max']:.1f}"
		if dataset_info["glucose_min"] is not None and dataset_info["glucose_max"] is not None
		else "N/A",
	)

st.info("Gunakan menu sidebar untuk membuka halaman Input Logbook, Prediction, atau What-If Simulator.")
