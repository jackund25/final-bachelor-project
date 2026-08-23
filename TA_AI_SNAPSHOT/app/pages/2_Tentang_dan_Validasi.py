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
Sistem ini adalah **alat bantu keputusan klinis (decision support)** untuk pengelolaan
**diabetes melitus tipe 1**, menggabungkan **prediksi glukosa** (Gradient Boosting) dengan
**rekomendasi berbasis panduan klinis** yang **dikondisikan pada nilai prediksi** — memberi
saran *antisipatif* terhadap kondisi yang akan datang, bukan hanya kondisi saat ini.

> **Status: prototipe riset (Tugas Akhir).** Digunakan secara *doctor-mediated* — keputusan
> medis final tetap pada dokter. Belum melalui uji klinis; divalidasi pada dataset **OhioT1DM**,
> yang berisi data penyandang diabetes tipe 1 sehingga sesuai populasi sasaran.
"""
)

st.divider()
st.subheader("Validasi Akurasi Prediksi")
st.caption("Disajikan sebagai konteks kepercayaan, bukan target penggunaan harian.")


def _clarke_sentence(clarke_ab: float) -> str:
    return (f"**{clarke_ab:.0f}%** prediksi berada di **zona aman klinis** "
            f"(Clarke Error Grid A+B) — artinya sebagian besar prediksi tidak akan menyebabkan "
            f"keputusan terapi yang keliru.")


# Metrik yang ditampilkan HARUS berasal dari model yang benar-benar dipakai aplikasi.
# Sebelumnya halaman ini membaca summary_all_horizons.csv lalu menyaring baris "RF",
# sehingga setelah prediktor berganti ke GBM angka yang ditampilkan kepada dokter
# berasal dari model yang tidak lagi dijalankan.
summary_path = Path("results/eval_prediksi/summary_all_horizons.csv")
metrik_horizon = [(h, Path(f"models/gbm_metrics_h{h}.json")) for h in (6, 12)]
tersedia = [(h, p) for h, p in metrik_horizon if p.exists()]

if tersedia:
    for h, p in tersedia:
        m = json.load(open(p, encoding="utf-8"))
        st.markdown(f"**Horizon +{h * 5} menit** — rata-rata meleset "
                    f"±{m['RMSE']:.0f} mg/dL; " + _clarke_sentence(m["Clarke_A+B"]))
    st.caption("RMSE = rata-rata simpangan prediksi (mg/dL). Clarke A+B = % prediksi di zona aman. "
               "Diukur pada dua pasien uji yang tidak pernah dilihat model saat pelatihan.")
    if summary_path.exists():
        with st.expander("Pembanding: Random Forest dan LSTM, semua horizon"):
            st.dataframe(pd.read_csv(summary_path), use_container_width=True)
            st.caption("Tabel pembanding dari evaluasi terdahulu. Model produksi saat ini "
                       "adalah Gradient Boosting; baris di sini disajikan sebagai konteks.")
else:
    st.info("Data validasi belum tersedia. Jalankan "
            "`python -m src.models.gbm_model --config config.yaml --data_source ohio_t1dm`.")

st.divider()
st.subheader("Cara Kerja Singkat")
st.markdown(
    """
1. **Prediksi:** model mempelajari pola glukosa, insulin, karbohidrat, aktivitas dari riwayat pasien,
   lalu memperkirakan glukosa 30–60 menit ke depan.
2. **Rekomendasi terkondisi:** nilai prediksi dipakai untuk menarik panduan klinis yang relevan
   dengan kondisi *yang akan datang* (mis. antisipasi hipoglikemia), lalu diringkas menjadi saran.
3. **Konseling & keputusan:** dokter meninjau rekomendasi beserta rujukan halamannya, lalu mencatat
   keputusannya sebagai rekam jejak (alur *doctor-mediated*).
"""
)

st.subheader("Keterbatasan")
st.markdown(
    """
- Dataset **OhioT1DM** sesuai populasi sasaran (Diabetes **Tipe 1**), tetapi seluruh kontributornya
  memakai pompa insulin dan CGM, sedangkan sasaran penerapan memakai suntikan harian dan tusuk jari.
- Embedding pencarian panduan belum dioptimalkan untuk Bahasa Indonesia.
- Belum ada uji klinis / evaluasi keselamatan pada pasien nyata.
"""
)

disclaimer_footer()
