import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data.loader import DiabetesDataLoader
from src.digital_twin import PatientDigitalTwin, WhatIfSimulator
from src.rag import RAGPipeline


st.set_page_config(page_title="What-If Simulator", page_icon="🧪", layout="wide")
st.title("🧪 What-If Simulator")
st.caption("Simulasikan dampak intervensi sebelum diterapkan pada pasien")


@st.cache_data
def load_dataset():
	loader = DiabetesDataLoader("data/raw")
	return loader.load_latest_dataset().sort_values(["patient_id", "timestamp"])


@st.cache_resource
def load_rag_pipeline(knowledge_base_dir: str = "data/knowledge_base"):
	pipeline = RAGPipeline(kb_dir=knowledge_base_dir, llm_provider="template")
	pipeline.build()
	return pipeline


def create_twin_from_latest_row(patient_df: pd.DataFrame) -> PatientDigitalTwin:
	latest = patient_df.iloc[-1]

	initial_state = {
		"current_glucose": float(latest["glucose"]),
		"insulin_on_board": float(latest.get("insulin", 0.0)),
		"carbs_on_board": float(latest.get("carbs", 0.0)),
		"activity_level": int(latest.get("activity", 0)),
		"stress_level": int(latest.get("stress", 5)),
	}

	return PatientDigitalTwin(patient_id=str(latest["patient_id"]), initial_state=initial_state)


def build_patient_state(twin: PatientDigitalTwin) -> dict:
	return {
		"current_glucose": float(twin.state["current_glucose"]),
		"insulin_on_board": float(twin.state["insulin_on_board"]),
		"carbs_on_board": float(twin.state["carbs_on_board"]),
		"activity_level": int(twin.state["activity_level"]),
		"stress_level": int(twin.state["stress_level"]),
	}


try:
	df = load_dataset()
except FileNotFoundError:
	st.error("Dataset belum ditemukan di data/raw. Generate data dulu dari CLI.")
	st.code("python -m src.data.generator")
	st.stop()

if df.empty:
	st.error("Dataset kosong.")
	st.stop()

patient_ids = sorted(df["patient_id"].unique().tolist())
selected_patient = st.selectbox("Pilih Patient ID", patient_ids)

patient_df = df[df["patient_id"] == selected_patient].copy()
twin = create_twin_from_latest_row(patient_df)
simulator = WhatIfSimulator(twin)

state_col1, state_col2, state_col3, state_col4 = st.columns(4)
state_col1.metric("Current Glucose", f"{twin.state['current_glucose']:.1f} mg/dL")
state_col2.metric("Insulin On Board", f"{twin.state['insulin_on_board']:.2f} u")
state_col3.metric("Carbs On Board", f"{twin.state['carbs_on_board']:.1f} g")
state_col4.metric("Stress Level", f"{twin.state['stress_level']}/10")

st.markdown("---")
st.subheader("Scenario Builder")

left, right = st.columns([1.5, 1])

with left:
	time_horizon = st.slider("Time Horizon (minutes)", min_value=30, max_value=240, value=60, step=15)
	carbs_delta = st.slider("Additional Carbs (g)", min_value=0, max_value=120, value=30, step=5)
	insulin_delta = st.slider("Additional Insulin (units)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
	activity_delta = st.slider("Additional Activity (minutes)", min_value=0, max_value=120, value=15, step=5)
	stress_delta = st.slider("Stress Change", min_value=-5, max_value=5, value=0, step=1)

	scenario = {
		"carbs_delta": float(carbs_delta),
		"insulin_delta": float(insulin_delta),
		"activity_delta": int(activity_delta),
		"stress_delta": int(stress_delta),
		"time_horizon": int(time_horizon),
	}

	if st.button("Run What-If Simulation", type="primary"):
		result = twin.simulate_scenario(scenario)

		current_glucose = float(result["current_glucose"])
		predicted_glucose = float(result["predicted_glucose"])
		glucose_change = float(result["glucose_change"])

		out1, out2, out3 = st.columns(3)
		out1.metric("Current", f"{current_glucose:.1f} mg/dL")
		out2.metric("Predicted", f"{predicted_glucose:.1f} mg/dL")
		out3.metric("Change", f"{glucose_change:+.1f} mg/dL")

		if predicted_glucose < 70:
			st.error(f"Risk: {result['risk_level']}")
		elif predicted_glucose > 180:
			st.warning(f"Risk: {result['risk_level']}")
		else:
			st.success(f"Risk: {result['risk_level']}")

		# Simple bar chart without Narwhals conversion issues - use simple dict format
		labels = ["Current", "Predicted"]
		values = [float(current_glucose), float(predicted_glucose)]
		fig, ax = plt.subplots(figsize=(6, 3))
		ax.bar(labels, values, color=["#4F8BF9", "#F97316"])
		ax.set_title("What-If Glucose Comparison")
		ax.set_ylabel("mg/dL")
		ax.grid(axis="y", alpha=0.25)
		st.pyplot(fig, clear_figure=True)

		st.markdown("### Clinical Explanation")
		rag_pipeline = load_rag_pipeline("data/knowledge_base")
		rag_result = rag_pipeline.answer(
			patient_state=build_patient_state(twin),
			prediction=predicted_glucose,
			query=(
				f"Simulasi what-if untuk pasien diabetes: karbohidrat +{scenario['carbs_delta']} g, "
				f"insulin +{scenario['insulin_delta']} unit, aktivitas +{scenario['activity_delta']} menit, "
				f"stres {scenario['stress_delta']:+d}, horizon {scenario['time_horizon']} menit."
			),
		)

		st.info(rag_result["explanation"])
		with st.expander("Retrieved medical context"):
			for doc in rag_result["retrieved_docs"]:
				st.markdown(f"**Rank {doc['rank']}** - source: `{doc['source']}` - similarity: {doc['similarity']:.2f}")
				st.write(doc["text"])
				st.markdown("---")

with right:
	st.markdown("### Quick Simulation")

	stress_reduction = st.slider("Stress Reduction", min_value=1, max_value=5, value=3)
	stress_result = simulator.simulate_stress_reduction(stress_reduction)
	st.write(
		f"Relaksasi {-stress_reduction} point stres: "
		f"{stress_result['glucose_change']:+.1f} mg/dL"
	)

	exercise_duration = st.slider("Exercise Duration", min_value=10, max_value=90, value=30, step=5)
	exercise_result = simulator.simulate_exercise(exercise_duration)
	st.write(
		f"Olahraga {exercise_duration} menit: "
		f"{exercise_result['glucose_change']:+.1f} mg/dL"
	)

	meal_carbs = st.slider("Meal Carbs for Dose Recommendation", min_value=15, max_value=120, value=60, step=5)
	dose_result = simulator.find_optimal_insulin_dose(meal_carbs=meal_carbs, target_glucose=120)
	st.info(
		f"Rekomendasi insulin: {dose_result['recommended_dose']:.1f} unit "
		f"(prediksi {dose_result['predicted_glucose']:.1f} mg/dL)"
	)

st.markdown("---")
st.subheader("Decision Support")
decision = simulator.generate_decision_tree()

for rec in decision["recommendations"]:
	priority = rec.get("priority", "LOW")
	if priority == "HIGH":
		st.error(f"{priority} | {rec['action']} - {rec['reason']}")
	elif priority == "MEDIUM":
		st.warning(f"{priority} | {rec['action']} - {rec['reason']}")
	else:
		st.success(f"{priority} | {rec['action']} - {rec['reason']}")

st.caption("Catatan: hasil simulasi adalah decision support, bukan pengganti keputusan klinis.")
