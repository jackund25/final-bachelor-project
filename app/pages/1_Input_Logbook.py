import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))  # folder app (untuk ui)

import pandas as pd
import streamlit as st

from ui import app_header, glucose_zone_chart, zone_legend, disclaimer_footer

st.set_page_config(page_title="Input Logbook", page_icon="📝", layout="wide")
app_header("Input Logbook", "Catat data harian pasien untuk mendukung digital twin & prediksi", "📝")

LOGBOOK_PATH = Path("data/raw/manual_logbook.csv")
COLUMNS = ["timestamp", "patient_id", "glucose", "carbs", "insulin", "activity",
           "stress", "sleep", "work", "illness", "meal_type", "notes", "source"]


def load_logbook() -> pd.DataFrame:
    if not LOGBOOK_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(LOGBOOK_PATH, parse_dates=["timestamp"])
    df = pd.DataFrame(df.to_dict("list"))  # strip wrapper
    return df.sort_values("timestamp", ascending=False).reset_index(drop=True)


def save_entry(entry: dict) -> None:
    LOGBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = load_logbook()
    combined = pd.DataFrame([entry]) if df.empty else pd.concat([pd.DataFrame([entry]), df], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
    combined = combined.sort_values("timestamp", ascending=False)
    combined["timestamp"] = combined["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    combined.to_csv(LOGBOOK_PATH, index=False)


left, right = st.columns([1.1, 0.9])

with left:
    st.subheader("Tambah Catatan Baru")
    with st.form("logbook_form", clear_on_submit=False):
        patient_id = st.text_input("ID Pasien", value=st.session_state.get("patient_id", "adult#001"))
        dcol, tcol = st.columns(2)
        date_value = dcol.date_input("Tanggal")
        time_value = tcol.time_input("Waktu")

        c1, c2, c3 = st.columns(3)
        with c1:
            glucose = st.number_input("Glukosa (mg/dL)", 40.0, 400.0, 110.0, 1.0)
            carbs = st.number_input("Karbohidrat (g)", 0.0, 200.0, 30.0, 1.0)
            insulin = st.number_input("Insulin (unit)", 0.0, 30.0, 3.0, 0.1)
        with c2:
            activity = st.number_input("Aktivitas (menit)", 0, 240, 15, 5)
            stress = st.slider("Tingkat Stres", 1, 10, 5)
            sleep = st.checkbox("Tidur", value=False)
        with c3:
            work = st.checkbox("Kerja", value=True)
            illness = st.checkbox("Sakit", value=False)
            meal_type = st.selectbox("Jenis Makan", ["none", "sarapan", "makan siang", "makan malam", "camilan"])

        notes = st.text_area("Catatan", placeholder="Contoh: setelah makan siang gula cenderung naik...")
        submitted = st.form_submit_button("💾 Simpan Catatan", type="primary", use_container_width=True)

    if submitted:
        ts = datetime.combine(date_value, time_value)
        entry = {
            "timestamp": pd.to_datetime(ts), "patient_id": patient_id.strip(),
            "glucose": float(glucose), "carbs": float(carbs), "insulin": float(insulin),
            "activity": int(activity), "stress": int(stress), "sleep": int(sleep),
            "work": int(work), "illness": int(illness), "meal_type": meal_type,
            "notes": notes.strip(), "source": "manual",
        }
        save_entry(entry)
        st.session_state["patient_id"] = patient_id.strip()
        st.success(f"✅ Tersimpan untuk {patient_id} pada {entry['timestamp']}")

with right:
    st.subheader("Status Logbook")
    lb = load_logbook()
    total = len(lb)
    s1, s2 = st.columns(2)
    s1.metric("Total Catatan", f"{total:,}")
    s2.metric("Pasien", f"{lb['patient_id'].nunique() if total else 0}")

    if total:
        st.caption(f"Catatan terakhir: {lb['timestamp'].max()}")
        st.markdown("**Catatan Terbaru**")
        st.dataframe(lb.head(8), use_container_width=True, height=200)

        st.markdown("**Snapshot Glukosa**")
        chart_df = lb[["timestamp", "glucose"]].sort_values("timestamp").tail(100)
        fig = glucose_zone_chart(
            x=list(range(1, len(chart_df) + 1)), y=chart_df["glucose"].tolist(),
            title="Glukosa Terkini (logbook)", height=300,
        )
        st.plotly_chart(fig, use_container_width=True)
        zone_legend()
    else:
        st.info("Belum ada data logbook. Tambahkan catatan pertama di sebelah kiri.")

st.caption("Logbook disimpan ke `data/raw/manual_logbook.csv` sebagai sumber input manual pasien.")
disclaimer_footer()
