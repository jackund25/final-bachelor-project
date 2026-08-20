import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))  # folder app (untuk ui)

import pandas as pd
import streamlit as st

from ui import app_header, glucose_zone_chart, zone_legend, disclaimer_footer

st.set_page_config(page_title="Input Logbook", page_icon="📝", layout="wide")
app_header("Input Logbook", "Catat data harian pasien untuk mendukung prediksi glukosa", "📝")

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
        # step=60 detik supaya menit mana pun dapat dipilih. Bawaan Streamlit mengunci
        # pilihan ke kelipatan 15 menit dari 00:00, sehingga pembacaan nyata pukul 16.42
        # terpaksa dibulatkan ke 16.45 dan catatannya menjadi tidak lagi jujur terhadap
        # waktu pengukurannya.
        time_value = tcol.time_input("Waktu", step=60)

        c1, c2, c3 = st.columns(3)
        with c1:
            glucose = st.number_input("Glukosa (mg/dL)", 40.0, 400.0, 110.0, 1.0)
            carbs = st.number_input("Karbohidrat (g)", 0.0, 200.0, 30.0, 1.0)
            insulin = st.number_input("Insulin (unit)", 0.0, 30.0, 3.0, 0.1)
        with c2:
            # Skala INTENSITAS 0-10, bukan menit. Kolom activity pada data pelatihan berasal
            # dari atribut "intensity" peristiwa exercise OhioT1DM (src/data/ohio_parser.py),
            # yang sebarannya 0 sampai 10. Label lama "Aktivitas (menit)" dengan rentang
            # 0-240 membuat dokter memberi model angka sampai 24 kali di luar sebaran
            # pelatihan tanpa peringatan apa pun. Bawaannya 0 karena hanya 0,12% baris
            # pelatihan yang bukan-nol: tidak berolahraga adalah keadaan yang normal.
            activity = st.number_input("Aktivitas (intensitas 0–10)", 0, 10, 0, 1)
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
        # Halaman Konsultasi men-cache logbook (@st.cache_data). Tanpa pembersihan ini,
        # catatan yang baru disimpan tidak akan terlihat di jalur prediksi sampai aplikasi
        # dimuat ulang — dan dokter akan mengira catatannya tidak tersimpan.
        st.cache_data.clear()
        st.success(f"✅ Tersimpan untuk {patient_id} pada {entry['timestamp']}")
        st.info("Catatan ini dapat disertakan ke jendela prediksi lewat kotak centang "
                "**Sertakan catatan logbook** di sidebar halaman Konsultasi. "
                "Catatan hanya dipakai bila jaraknya terhadap pembacaan di sekitarnya "
                "masih serapat jendela yang dipakai melatih model.")

with right:
    # Legenda variabel. Sebelum ini tidak ada satu pun keterangan di layar mengenai arti,
    # rentang, maupun satuan tiap variabel — dan yang paling menyesatkan, tidak ada
    # keterangan bahwa stres, tidur, kerja, sakit, dan jenis makan TIDAK dibaca model.
    # Dokter yang mengisi tingkat stres 9 akan mengira prediksinya memperhitungkan stres.
    st.subheader("Arti dan Rentang Variabel")
    with st.expander("Masuk ke model dan menggerakkan prediksi", expanded=True):
        st.markdown(
            "**Glukosa** — 40 sampai 400 mg/dL  \n"
            "Kadar hasil pengukuran saat itu. Patokan klinis: di bawah **54** hipoglikemia "
            "berat, di bawah **70** hipoglikemia, **70–180** rentang sasaran, di atas "
            "**180** hiperglikemia, di atas **250** hiperglikemia berat.\n\n"
            "**Karbohidrat** — 0 sampai 200 g  \n"
            "Jumlah karbohidrat pada **satu asupan**, bukan akumulasi sehari. Patokan: "
            "sepiring nasi ±40 g, sepotong roti ±15 g, satu pisang ±25 g. Pengaruhnya "
            "meluruh dengan tetapan waktu **3 jam**.\n\n"
            "**Insulin** — 0 sampai 30 unit  \n"
            "Dosis **bolus** yang diberikan saat itu, bukan laju basal. Pengaruhnya meluruh "
            "dengan tetapan waktu **4 jam**, mengikuti lama kerja insulin analog kerja cepat "
            "4 sampai 6 jam menurut PERKENI.\n\n"
            "**Aktivitas** — intensitas 0 sampai 10  \n"
            "Seberapa **berat** kegiatan fisiknya, **bukan berapa lama**. "
            "0 tidak beraktivitas; 1–3 ringan seperti jalan santai atau pekerjaan rumah; "
            "4–6 sedang seperti jalan cepat atau bersepeda santai; 7–8 berat seperti lari "
            "atau berenang; 9–10 sangat berat seperti lari cepat atau angkat beban."
        )
        st.caption(
            "Keempatnya menjadi tujuh fitur yang dibaca model: glukosa, tren glukosa, "
            "insulin aktif, karbohidrat aktif, aktivitas, dan dua komponen waktu dalam hari."
        )

    with st.expander("Dicatat sebagai rekam jejak, TIDAK dibaca model", expanded=False):
        st.markdown(
            "**Stres, Tidur, Kerja, Sakit, Jenis Makan, dan Catatan** tersimpan pada berkas "
            "logbook untuk keperluan rekam jejak klinis, tetapi **tidak menjadi masukan "
            "model**. Prediksi tidak berubah sedikit pun karena nilainya."
        )
        st.caption(
            "Model produksi dilatih tanpa kolom-kolom itu, sehingga menambahkannya saat "
            "prediksi akan membuat bentuk masukan tidak cocok dengan penskala. Pada data "
            "pelatihan pun kolom stres hanya bukan-nol pada 6 dari 166.533 baris, sehingga "
            "tidak ada pola yang dapat dipelajari darinya."
        )

    with st.expander("Syarat agar catatan dapat diprediksi", expanded=False):
        st.markdown(
            "Prediksi menuntut **12 catatan** dengan jarak antar-catatan **tidak lebih dari "
            "30 menit**. Bila syarat itu tidak terpenuhi, halaman konsultasi menolak memakai "
            "logbook dan menyatakan alasannya, bukan menambal jeda dengan nilai karangan."
        )
        st.caption(
            "Perhitungan insulin aktif dan karbohidrat aktif mengandaikan catatan berjarak "
            "5 menit. Catatan yang lebih renggang membuat kedua nilai aktif itu tampak lebih "
            "besar daripada semestinya."
        )

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

st.caption("Logbook disimpan ke `data/raw/manual_logbook.csv`. Hanya `glucose`, `carbs`, "
           "`insulin`, dan `activity` yang menjadi fitur model; `stress`, `sleep`, `work`, "
           "`illness`, `meal_type`, dan `notes` disimpan sebagai rekam jejak klinis karena "
           "model produksi dilatih tanpa kolom-kolom tersebut.")
disclaimer_footer()
