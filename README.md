# Prediction-Conditioned RAG untuk Pengelolaan Diabetes

Tugas Akhir — Institut Teknologi Bandung

Penulis: Daffari Adiyatma (18222003)

Pembimbing:
- Prof. Dr. Ir. Suhono Harso Supangkat, M.Eng.
- Ir. Devi Willieam Anggara S.T., M.Phil., Ph.D

## Ringkasan

Sistem pendukung keputusan klinis (*decision support*) untuk pengelolaan glukosa darah yang
menggabungkan prediksi glukosa jangka pendek dengan *Retrieval-Augmented Generation* (RAG).
Kontribusi utama adalah **Prediction-Conditioned RAG**: kueri pengambilan panduan medis
dibangun dari **nilai glukosa yang diprediksi** (bukan hanya kondisi saat ini), sehingga
rekomendasi bersifat antisipatif terhadap kondisi yang akan datang.

Sistem diposisikan sebagai *simplified data-driven digital twin* (bukan digital twin
fisiologis penuh) dan digunakan secara *doctor-mediated*: keputusan medis final tetap pada
dokter.

## Komponen

- **Prediksi glukosa**: Random Forest (model utama) dengan pembanding LSTM, pada horizon
  standar BGLP Challenge (+30 dan +60 menit).
- **Prediction-Conditioned RAG**: pembangun kueri berbasis prediksi + pengambilan dari basis
  pengetahuan klinis (ChromaDB) + ringkasan oleh LLM (Google Gemini), dengan mode pembanding
  *standard* untuk membuktikan kontribusi.
- **Basis pengetahuan**: dokumen panduan klinis (PERKENI, ADA) di-*ingest* ke ChromaDB.
- **Digital twin sederhana**: proyeksi state metabolik (insulin/karbohidrat aktif) untuk
  simulasi *what-if*.
- **Aplikasi Streamlit**: antarmuka konsultasi klinis (status pasien, prediksi & risiko,
  rekomendasi, simulasi, pencatatan keputusan).

## Dataset

**OhioT1DM** (Marling & Bunescu, 2020) — data CGM Diabetes Tipe 1 dari 12 pasien, digunakan
sebagai *surrogate* untuk pembuktian teknis. Parser menghasilkan dua berkas:
- `data/raw/ohio_t1dm_merged.csv` — timeline CGM 5-menit (pelatihan model).
- `data/raw/ohio_t1dm_smbg.csv` — timeline `finger_stick` (SMBG nyata) untuk skenario SMBG.

Data OhioT1DM tunduk pada *Data Use Agreement* dan tidak disertakan dalam repositori.

## Tech Stack

Python 3.11, scikit-learn 1.3.0 (Random Forest), TensorFlow 2.15 (LSTM), LangChain +
Google Gemini, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Streamlit, pandas, Plotly.

## Penyiapan Lingkungan

Proyek dikembangkan pada environment conda `diabetes-ta` (Python 3.11). Versi paket dikunci
(mis. scikit-learn 1.3.0) agar artefak model yang tersimpan dapat dimuat dengan benar.

```bash
conda create -n diabetes-ta python=3.11 -y
conda activate diabetes-ta
pip install -r requirements.txt
```

Salin `.env.example` menjadi `.env` dan isi `GOOGLE_API_KEY` (Google AI Studio) untuk fitur
advisory LLM.

## Hasil Utama

| Evaluasi | Hasil |
| --- | --- |
| Prediksi CGM, +30 menit (RF) | RMSE 22,60 mg/dL · Clarke A+B 94,35% |
| Prediksi CGM, +60 menit (RF) | RMSE 34,24 mg/dL · Clarke A+B 86,94% |
| Prediksi CGM, +30 menit (LSTM, pembanding) | RMSE 22,04 mg/dL · Clarke A+B 94,60% |
| Skenario SMBG (`finger_stick` nyata), +30 menit | RMSE 27,09 mg/dL · Clarke A+B 92,09% |
| *Retrieval*: RAG standar → **PC-RAG** | MRR 0,225 → **0,889** · Hit@1 0% → **83,3%** |
| Interval konformal (nominal 95%) | cakupan empiris 96,5% |

Selisih RF vs LSTM (~0,44 mg/dL) signifikan secara statistik namun tidak bermakna klinis,
sehingga RF dipilih karena interpretabel dan berjalan tanpa GPU.

**Catatan pada skenario SMBG:** pada horizon +30 menit model praktis setara dengan *baseline
persistence* (RMSE 27,09 vs 27,33 mg/dL) — jarak antar-pembacaan SMBG (median 124,8 menit)
jauh lebih panjang daripada horizon prediksinya. Keunggulan yang jelas baru muncul pada
+60 menit (42,86 vs 47,87 mg/dL).

## Reproduksi

Jalankan dari root proyek dengan environment `diabetes-ta` aktif (`set PYTHONPATH=.` pada
Windows CMD atau `$env:PYTHONPATH="."` pada PowerShell):

```bash
# 1. Parse OhioT1DM XML -> CSV (CGM + SMBG finger_stick)
python -m src.data.ohio_parser

# 2. Latih Random Forest (menghasilkan bundle inferensi untuk aplikasi)
python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm

# 3. Verifikasi: apakah artefak model mereproduksi angka yang dilaporkan?
python scripts/verify_prediction_artifacts.py
```

Evaluasi yang menghasilkan angka pada tabel Hasil Utama:

```bash
# Prediksi: RF vs LSTM (+30/+60 menit), validasi silang lintas-pasien, uji signifikansi
python scripts/eval_rf_lstm.py --config config.yaml
python scripts/crossval_rf_vs_lstm.py

# Keamanan klinis: kalibrasi konformal & deteksi hipoglikemia
python scripts/conformal_calibration.py
python scripts/eval_hypo_uncertainty.py
python scripts/improve_hypo_detection.py

# Skenario deployment SMBG (timeline finger_stick nyata, bukan downsampling CGM)
python scripts/eval_smbg_deployment.py

# RAG: ingest basis pengetahuan, lalu ablation PC-RAG vs RAG standar
python scripts/reingest_kb.py
python scripts/ablation_rag_fullkb.py      # kebaruan pada korpus penuh
python scripts/sensitivity_analysis.py     # sensitivitas top_k & ukuran chunk
python scripts/eval_generation_novelty.py  # kualitas keluaran LLM

# Aplikasi
streamlit run app/streamlit_app.py
```

Seluruh keluaran evaluasi tersimpan di `results/` (JSON, CSV, dan gambar).

## Pengujian

```bash
pytest tests/ -q
```

## Struktur Proyek

```
app/               Aplikasi Streamlit (konsultasi klinis) + komponen UI
src/data/          Parser OhioT1DM, praproses, rekayasa fitur, kontrak data
src/models/        Random Forest (utama), LSTM (pembanding)
src/rag/           Pipeline RAG, prediction-conditioned query, retriever, knowledge base
src/digital_twin/  Patient twin, simulator what-if, state manager
src/utils/         Metrik evaluasi, logging, visualisasi
scripts/           Skrip evaluasi & utilitas (mereproduksi seluruh angka laporan)
tests/             Uji unit & integrasi
results/           Keluaran evaluasi (metrik, gambar)
models/            Metrik model terlatih (berkas .pkl tidak di-commit)
docs/laporan_TA/   Sumber LaTeX laporan tugas akhir
config.yaml        Konfigurasi terpusat (data, model, RAG, evaluasi)
run_app.py         Entry point aplikasi
```

## Keterbatasan

- OhioT1DM adalah dataset Diabetes Tipe 1 (*surrogate*); diperlukan validasi pada data
  Tipe 2 Indonesia sebelum penggunaan klinis.
- Pada skenario SMBG, prediksi +30 menit belum melampaui *baseline persistence* (lihat
  Hasil Utama); manfaat prediktif pada data *logbook* yang jarang baru nyata pada horizon
  yang lebih panjang.
- Sensitivitas deteksi hipoglikemia pada ambang standar 70 mg/dL masih rendah (14%);
  aplikasi memakai ambang peringatan dini 85 mg/dL (sensitivitas 59%, PPV 31%).
- Digital twin bersifat sederhana (berbasis formula), bukan model fisiologis multi-skala.
- Belum ada uji klinis; evaluasi bersifat teknis (akurasi prediksi dan kualitas retrieval).
- Evaluasi *retrieval* dan generasi memakai enam kasus terkurasi.
- Embedding pencarian belum dioptimalkan (*fine-tune*) untuk Bahasa Indonesia.

## Lisensi

Penggunaan akademik — Tugas Akhir Institut Teknologi Bandung.

## Acknowledgments

- Marling & Bunescu (Ohio University) atas dataset OhioT1DM.
- PERKENI dan ADA atas panduan klinis diabetes.
