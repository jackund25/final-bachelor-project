import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # app (ui)

import json

import pandas as pd
import streamlit as st

from ui import app_header, disclaimer_footer

st.set_page_config(page_title="Tentang & Validasi", page_icon="ℹ️", layout="wide")
app_header("Tentang Sistem & Validasi Model", "Transparansi untuk klinisi — dasar & bukti validasi", "ℹ️")

st.markdown(
    """
Sistem ini adalah **alat bantu keputusan klinis (decision support)** untuk pengelolaan glukosa,
menggabungkan **prediksi glukosa** (Random Forest) dengan **rekomendasi berbasis panduan medis**
(PERKENI/ADA) yang **dikondisikan pada nilai prediksi** — memberi saran *antisipatif* terhadap
kondisi yang akan datang, bukan hanya kondisi saat ini.

> **Status: prototipe riset (Tugas Akhir).** Digunakan secara *doctor-mediated* — keputusan
> medis final tetap pada dokter. Belum melalui uji klinis; divalidasi pada dataset **OhioT1DM**
> (surrogate) untuk pembuktian teknis.
"""
)

st.divider()
st.subheader("Validasi Akurasi Prediksi")
st.caption("Disajikan sebagai konteks kepercayaan, bukan target penggunaan harian.")


def _clarke_sentence(clarke_ab: float) -> str:
    return (f"**{clarke_ab:.0f}%** prediksi berada di **zona aman klinis** "
            f"(Clarke Error Grid A+B) — artinya sebagian besar prediksi tidak akan menyebabkan "
            f"keputusan terapi yang keliru.")


summary_path = Path("results/eval_prediksi/summary_all_horizons.csv")
metrics_path = Path("models/rf_baseline_metrics.json")

if summary_path.exists():
    df = pd.read_csv(summary_path)
    rf = df[df["model"] == "RF"]
    for _, r in rf.iterrows():
        st.markdown(f"**Horizon +{int(r['horizon_min'])} menit** — "
                    f"rata-rata meleset ±{r['RMSE']:.0f} mg/dL; " + _clarke_sentence(r["Clarke_A+B"]))
    with st.expander("Tabel lengkap (RF vs LSTM, semua horizon) — untuk laporan"):
        st.dataframe(df, use_container_width=True)
    st.caption("RMSE = rata-rata simpangan prediksi (mg/dL). Clarke A+B = % prediksi di zona aman.")
elif metrics_path.exists():
    m = json.load(open(metrics_path, encoding="utf-8"))
    st.markdown(f"Rata-rata meleset ±{m.get('RMSE', 0):.0f} mg/dL. " + _clarke_sentence(m.get("Clarke_A+B", 0)))
else:
    st.info("Data validasi belum tersedia. Jalankan `python scripts/eval_rf_lstm.py`.")

st.divider()
st.subheader("Cara Kerja Singkat")
st.markdown(
    """
1. **Prediksi:** model mempelajari pola glukosa, insulin, karbohidrat, aktivitas dari riwayat pasien,
   lalu memperkirakan glukosa 30–60 menit ke depan.
2. **Rekomendasi terkondisi:** nilai prediksi dipakai untuk menarik panduan klinis yang relevan
   dengan kondisi *yang akan datang* (mis. antisipasi hipoglikemia), lalu diringkas menjadi saran.
3. **Konseling & keputusan:** dokter dapat mensimulasikan intervensi (what-if) dan mencatat keputusan.
"""
)

st.subheader("Keterbatasan")
st.markdown(
    """
- Dataset **OhioT1DM** adalah Diabetes **Tipe 1** (surrogate); perlu validasi pada data Tipe 2 Indonesia.
- Embedding pencarian panduan belum dioptimalkan untuk Bahasa Indonesia.
- Belum ada uji klinis / evaluasi keselamatan pada pasien nyata.
"""
)

disclaimer_footer()
