# Metodologi: Data & Horizon Prediksi

## 1. Sumber Dataset

Dataset yang digunakan adalah **OhioT1DM** (Marling & Bunescu, 2020), kumpulan
data Diabetes Tipe 1 dari 12 pasien (kohort 2018: 6 pasien, kohort 2020: 6 pasien)
selama ±8 minggu per pasien.  Data ini dipilih karena:
- Merupakan dataset T1DM multimodal publik dengan CGM, insulin, makanan, dan gaya hidup.
- Diakui sebagai benchmark standar riset prediksi glukosa (bdk. Martinsson et al., 2020).
- Tersedia secara bebas untuk penelitian akademik.

Parser XML tersedia di `src/data/ohio_parser.py`; output digabung menjadi
`data/raw/ohio_t1dm_merged.csv` melalui `process_ohio_dataset()`.

## 2. Cadence Asli dan Surrogate SMBG

### 2.1 Data Mentah: CGM 5-menit

OhioT1DM merekam glukosa setiap **5 menit** (Continuous Glucose Monitor / CGM),
menghasilkan ±288 titik/hari per pasien.  Cadence ini jauh lebih padat daripada
kondisi klinis yang menjadi target sistem ini: pasien Tipe 2 tanpa CGM yang
menggunakan **Self-Monitoring of Blood Glucose** (SMBG) — pengukuran mandiri
3–6 kali per hari.

### 2.2 Justifikasi Penggunaan Data CGM untuk Pelatihan RF

Model Random Forest dilatih pada **data CGM asli (5-menit)** dengan alasan:

1. **Kualitas sinyal** — window CGM memberikan resolusi temporal yang cukup untuk
   menangkap tren glukosa postprandial dan nocturnal yang sangat informatif bagi
   model prediktif.
2. **Ketersediaan OhioT1DM** — satu-satunya dataset publik dengan label multimodal
   (karbohidrat, insulin, aktivitas, stres) yang dibutuhkan oleh kelima fitur model;
   tidak tersedia dataset SMBG publik dengan kelengkapan fitur yang setara.
3. **Kinerja empiris** — RF mencapai RMSE 6.25 mg/dL dan Clarke-A 99.5% pada data
   CGM, yang menjadi *baseline* valid untuk perbandingan LSTM (T4).

`sequence_length = 12` pada cadence 5-menit = **jendela look-back 1 jam**,
cukup untuk menangkap dinamika postprandial (puncak glukosa umumnya 45–90 menit
setelah makan, ADA 2023).

### 2.3 Simulasi Cadence SMBG (Surrogate Downsampling)

Untuk mensimulasikan kondisi pasien SMBG, tersedia fungsi eksplisit:

```python
from src.data.preprocessor import DataPreprocessor

prep = DataPreprocessor(config)
df_smbg = prep.downsample_smbg(df_cgm, interval_minutes=240)
# → ~6 readings/hari, step=48 baris CGM per pembacaan SMBG
```

| Parameter | CGM (training RF) | SMBG surrogate |
|---|---|---|
| `interval_minutes` | 5 | 240 (4 jam) |
| `readings_per_day` | 288 | ~6 |
| `sequence_length` | 12 (= 1 jam) | 6 (= ~1 hari) |
| `look_back_window` | 60 menit | ~24 jam |

Downsampling dilakukan **per pasien** (sorted by timestamp, step=N) sehingga
urutan temporal terjaga.  Fungsi ini digunakan pada eksperimen LSTM (T3) untuk
memverifikasi bahwa RF tetap unggul bahkan pada data yang lebih sparse.

## 3. Horizon Prediksi

Sistem memprediksi **nilai glukosa satu langkah ke depan** (next-step prediction):
- Pada CGM: +5 menit dari titik terakhir window
- Pada SMBG: +1 pengukuran ke depan (horizon ~4 jam)

Horizon ini dipilih secara sengaja agar:
- Rekomendasi RAG dapat dikondisikan pada nilai **yang akan datang** dalam
  jangka waktu klinis yang bermakna (sebelum makan berikutnya / sebelum tidur).
- Menghindari error kumulatif yang membesar pada multi-step forecast.

## 4. Train/Test Split

Pembagian data dilakukan **per pasien** (bukan per baris) untuk menghindari
data leakage temporal:
- **Train**: 80% pasien (8 dari 10 yang digunakan)
- **Test**: 20% pasien (2 pasien hold-out)

Fungsi: `DataPreprocessor.split_by_patient(df, test_patients=[...])`.

Normalisasi (`StandardScaler`) di-*fit* hanya pada data training dan di-*transform*
pada data test.

## 5. Reproduksi

```bash
# 1. Parse OhioT1DM XML → CSV
python -m src.data.ohio_parser

# 2. (Opsional) Inspeksi dataset
python -c "
from src.data.loader import DiabetesDataLoader
dl = DiabetesDataLoader()
df, src = dl.load_preferred_dataset()
print(df.shape, src)
"

# 3. (Opsional) Downsample ke SMBG cadence
python -c "
import yaml, pandas as pd
from src.data.loader import DiabetesDataLoader
from src.data.preprocessor import DataPreprocessor
cfg = yaml.safe_load(open('config.yaml'))
df, _ = DiabetesDataLoader().load_preferred_dataset()
prep = DataPreprocessor(cfg)
df_smbg = prep.downsample_smbg(df, interval_minutes=240)
df_smbg.to_csv('data/processed/ohio_smbg_240min.csv', index=False)
print(df_smbg.shape)
"
```

## Referensi

- Marling, C. & Bunescu, R. (2020). *The OhioT1DM Dataset for Blood Glucose Level
  Prediction.* CEUR Workshop Proceedings, KDH@ECAI.
- Martinsson, J. et al. (2020). *Blood glucose prediction with variance estimation
  using recurrent neural networks.* Journal of Healthcare Informatics Research.
- American Diabetes Association (2023). *Standards of Medical Care in Diabetes.*
  Diabetes Care, 46(Suppl. 1).
