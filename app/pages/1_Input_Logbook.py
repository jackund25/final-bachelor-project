from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Input Logbook", page_icon="📝", layout="wide")
st.title("📝 Input Logbook")
st.caption("Masukkan data harian pasien untuk mendukung digital twin dan prediksi glukosa.")


LOGBOOK_PATH = Path("data/raw/manual_logbook.csv")


def load_logbook() -> pd.DataFrame:
	if not LOGBOOK_PATH.exists():
		return pd.DataFrame(
			columns=[
				"timestamp",
				"patient_id",
				"glucose",
				"carbs",
				"insulin",
				"activity",
				"stress",
				"sleep",
				"work",
				"illness",
				"meal_type",
				"notes",
				"source",
			]
		)

	df = pd.read_csv(LOGBOOK_PATH, parse_dates=["timestamp"])
	# BRUTAL FIX: Reconstruct from dict to strip Narwhals wrapper, then sort
	df_native = pd.DataFrame(df.to_dict('list'))
	return df_native.sort_values("timestamp", ascending=False).reset_index(drop=True)


def save_entry(entry: dict) -> None:
	LOGBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
	df = load_logbook()
	new_row = pd.DataFrame([entry])

	if df.empty:
		combined = new_row
	else:
		combined = pd.concat([new_row, df], ignore_index=True)

	combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
	combined = combined.sort_values("timestamp", ascending=False)
	combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
	combined.to_csv(LOGBOOK_PATH, index=False)


left, right = st.columns([1.1, 0.9])

with left:
	st.subheader("Add New Entry")
	with st.form("logbook_form", clear_on_submit=False):
		patient_id = st.text_input("Patient ID", value="adult#001")
		date_value = st.date_input("Date")
		time_value = st.time_input("Time")

		c1, c2, c3 = st.columns(3)
		with c1:
			glucose = st.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=110.0, step=1.0)
			carbs = st.number_input("Carbs (g)", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
			insulin = st.number_input("Insulin (units)", min_value=0.0, max_value=30.0, value=3.0, step=0.1)
		with c2:
			activity = st.number_input("Activity (minutes)", min_value=0, max_value=240, value=15, step=5)
			stress = st.slider("Stress Level", min_value=1, max_value=10, value=5)
			sleep = st.checkbox("Sleep", value=False)
		with c3:
			work = st.checkbox("Work", value=True)
			illness = st.checkbox("Illness", value=False)
			meal_type = st.selectbox("Meal Type", ["none", "breakfast", "lunch", "dinner", "snack"])

		notes = st.text_area("Notes", placeholder="Contoh: setelah makan siang, gula cenderung naik...")

		submitted = st.form_submit_button("Save Logbook Entry")

	if submitted:
		timestamp = datetime.combine(date_value, time_value)
		entry = {
			"timestamp": pd.to_datetime(timestamp),
			"patient_id": patient_id.strip(),
			"glucose": float(glucose),
			"carbs": float(carbs),
			"insulin": float(insulin),
			"activity": int(activity),
			"stress": int(stress),
			"sleep": int(sleep),
			"work": int(work),
			"illness": int(illness),
			"meal_type": meal_type,
			"notes": notes.strip(),
			"source": "manual",
		}

		save_entry(entry)
		st.success(f"Entry saved for {patient_id} at {entry['timestamp']}")

with right:
	st.subheader("Logbook Status")
	current_logbook = load_logbook()

	total_entries = len(current_logbook)
	total_patients = current_logbook["patient_id"].nunique() if total_entries else 0
	latest_timestamp = current_logbook["timestamp"].max() if total_entries else None

	stat1, stat2 = st.columns(2)
	stat1.metric("Total Entries", f"{total_entries:,}")
	stat2.metric("Patients", f"{total_patients}")

	if latest_timestamp is not None:
		st.info(f"Latest entry: {latest_timestamp}")
	else:
		st.warning("Belum ada data logbook tersimpan.")

	if total_entries:
		st.markdown("### Recent Entries")
		st.dataframe(current_logbook.head(10), use_container_width=True)

		st.markdown("### Glucose Snapshot")
		chart_df = current_logbook[["timestamp", "glucose"]].sort_values("timestamp").tail(100)
		glucose_values = chart_df["glucose"].tolist()
		fig, ax = plt.subplots(figsize=(8, 3))
		ax.plot(range(1, len(glucose_values) + 1), glucose_values, linewidth=2)
		ax.set_title("Recent Glucose")
		ax.set_xlabel("Recent Point")
		ax.set_ylabel("mg/dL")
		ax.grid(alpha=0.25)
		st.pyplot(fig, clear_figure=True)

st.markdown("---")
st.caption("Logbook ini disimpan ke data/raw/manual_logbook.csv sebagai sumber input manual pasien.")
