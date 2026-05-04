import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data.loader import DiabetesDataLoader
from src.digital_twin import DigitalTwinStateManager, PatientDigitalTwin, WhatIfSimulator


st.set_page_config(page_title="Digital Twin Dashboard", page_icon="🫀", layout="wide")
st.title("🫀 Digital Twin Dashboard")
st.caption("Monitoring status pasien, alert klinis, dan rekomendasi aksi berbasis digital twin")


def _compute_meal_type(carbs: float) -> str:
    if carbs >= 70:
        return "dinner"
    if carbs >= 50:
        return "lunch"
    if carbs >= 25:
        return "breakfast"
    if carbs > 0:
        return "snack"
    return "none"


def _risk_badge(glucose_value: float) -> tuple[str, str]:
    if glucose_value < 70:
        return "BAHAYA - Hipoglikemia", "#b91c1c"
    if glucose_value > 180:
        return "HATI-HATI - Hiperglikemia", "#b45309"
    return "AMAN", "#15803d"


def _build_initial_state(row: pd.Series) -> dict:
    return {
        "current_glucose": float(row.get("glucose", 100.0)),
        "insulin_on_board": float(row.get("insulin", 0.0)),
        "carbs_on_board": float(row.get("carbs", 0.0)),
        "activity_level": int(float(row.get("activity", 0))),
        "stress_level": int(float(row.get("stress", 5))),
        "last_meal_time": row.get("timestamp", None),
        "timestamp": datetime.now().isoformat(),
    }


def _load_source_data() -> tuple[pd.DataFrame, str]:
    loader = DiabetesDataLoader("data/raw")

    manual_path = Path("data/raw/manual_logbook.csv")
    if manual_path.exists():
        manual_df = loader.load_csv("manual_logbook.csv").sort_values(["patient_id", "timestamp"])
        if not manual_df.empty:
            return manual_df, "manual_logbook"

    latest_df = loader.load_latest_dataset().sort_values(["patient_id", "timestamp"])
    return latest_df, "latest_generated"


def _render_risk_gauge(glucose_value: float):
    fig, ax = plt.subplots(figsize=(7, 1.4))
    ax.axhspan(0, 1, xmin=0.0, xmax=0.175, color="#ef4444", alpha=0.25)
    ax.axhspan(0, 1, xmin=0.175, xmax=0.45, color="#22c55e", alpha=0.25)
    ax.axhspan(0, 1, xmin=0.45, xmax=1.0, color="#f59e0b", alpha=0.25)

    clamped = max(40.0, min(400.0, glucose_value))
    x = (clamped - 40.0) / (400.0 - 40.0)
    ax.axvline(x=x, color="#111827", linewidth=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([0.0, 0.175, 0.45, 1.0])
    ax.set_xticklabels(["40", "70", "180", "400"])
    ax.set_yticks([])
    ax.set_title("Risk Band Gauge (mg/dL)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig, clear_figure=True)


try:
    data_df, source_name = _load_source_data()
except FileNotFoundError:
    st.error("Tidak ada dataset yang bisa dipakai. Buat data dulu dengan generator atau isi logbook manual.")
    st.code("python -m src.data.generator")
    st.stop()

if data_df.empty:
    st.error("Dataset kosong.")
    st.stop()

patient_ids = sorted(data_df["patient_id"].unique().tolist())
selected_patient = st.selectbox("Pilih Patient ID", patient_ids)

patient_df = data_df[data_df["patient_id"] == selected_patient].copy().sort_values("timestamp")
latest_row = patient_df.iloc[-1]

initial_state = _build_initial_state(latest_row)
twin = PatientDigitalTwin(patient_id=selected_patient, initial_state=initial_state)
simulator = WhatIfSimulator(twin)

state_manager = DigitalTwinStateManager(storage_file="data/processed/patient_states.json")
state_manager.load()
if selected_patient not in state_manager.list_patients():
    state_manager.create_state(selected_patient, initial_state)
else:
    state_manager.update_state(selected_patient, initial_state)
state_manager.save()

risk_text, risk_color = _risk_badge(float(twin.state["current_glucose"]))

st.markdown(
    f"""
<div style="padding: 0.8rem 1rem; border-radius: 10px; background: #f8fafc; border: 1px solid #e2e8f0;">
  <b>Data source aktif:</b> {source_name} &nbsp; | &nbsp; <b>Risk saat ini:</b>
  <span style="color: {risk_color}; font-weight: 700;">{risk_text}</span>
</div>
""",
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Current Glucose", f"{float(twin.state['current_glucose']):.1f} mg/dL")
m2.metric("Insulin On Board", f"{float(twin.state['insulin_on_board']):.2f} u")
m3.metric("Carbs On Board", f"{float(twin.state['carbs_on_board']):.1f} g")
m4.metric("Activity", f"{int(twin.state['activity_level'])} min")
m5.metric("Stress", f"{int(twin.state['stress_level'])}/10")

left, right = st.columns([1.25, 1])

with left:
    st.markdown("### Glucose Timeline")
    timeline_df = patient_df[["timestamp", "glucose"]].tail(120)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.plot(timeline_df["timestamp"], timeline_df["glucose"], color="#2563eb", linewidth=2)
    ax.axhline(70, color="#dc2626", linestyle="--", linewidth=1)
    ax.axhline(180, color="#d97706", linestyle="--", linewidth=1)
    ax.set_ylabel("mg/dL")
    ax.set_xlabel("timestamp")
    ax.grid(alpha=0.2)
    ax.set_title("Last 120 Readings")
    plt.xticks(rotation=20)
    st.pyplot(fig, clear_figure=True)

    st.markdown("### Risk Gauge")
    _render_risk_gauge(float(twin.state["current_glucose"]))

with right:
    st.markdown("### Quick Actions")

    action_choice = st.selectbox(
        "Simulasi Cepat",
        [
            "Stress reduction",
            "Exercise",
            "Meal bolus recommendation",
        ],
    )

    if action_choice == "Stress reduction":
        stress_drop = st.slider("Turunkan stress", min_value=1, max_value=5, value=3)
        if st.button("Run stress simulation"):
            result = simulator.simulate_stress_reduction(stress_drop)
            st.info(f"Perubahan prediksi: {result['glucose_change']:+.1f} mg/dL")

    if action_choice == "Exercise":
        duration = st.slider("Durasi olahraga (menit)", min_value=10, max_value=120, value=30, step=5)
        if st.button("Run exercise simulation"):
            result = simulator.simulate_exercise(duration)
            st.info(f"Perubahan prediksi: {result['glucose_change']:+.1f} mg/dL")

    if action_choice == "Meal bolus recommendation":
        meal_carbs = st.slider("Karbohidrat makan (gram)", min_value=15, max_value=120, value=60, step=5)
        target = st.slider("Target glucose", min_value=90, max_value=150, value=120, step=5)
        if st.button("Find insulin dose"):
            result = simulator.find_optimal_insulin_dose(meal_carbs=float(meal_carbs), target_glucose=float(target))
            st.success(
                f"Rekomendasi insulin: {result['recommended_dose']:.1f} u | "
                f"Prediksi: {result['predicted_glucose']:.1f} mg/dL"
            )

st.markdown("---")
st.markdown("### Decision Support")

decision = simulator.generate_decision_tree()
for rec in decision["recommendations"]:
    priority = rec.get("priority", "LOW")
    message = f"{priority} | {rec['action']} - {rec['reason']}"
    if priority == "HIGH":
        st.error(message)
    elif priority == "MEDIUM":
        st.warning(message)
    else:
        st.success(message)

st.markdown("---")
st.markdown("### Persisted Twin State")
current_record = state_manager.get_state(selected_patient)
state_preview = {
    "timestamp": current_record.get("timestamp"),
    "current_glucose": current_record.get("current_glucose"),
    "insulin_on_board": current_record.get("insulin_on_board"),
    "carbs_on_board": current_record.get("carbs_on_board"),
    "activity_level": current_record.get("activity_level"),
    "stress_level": current_record.get("stress_level"),
    "meal_type_estimate": _compute_meal_type(float(current_record.get("carbs_on_board", 0.0))),
}
st.json(state_preview)
