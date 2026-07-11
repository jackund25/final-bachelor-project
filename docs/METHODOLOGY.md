# Metodologi: Data & Horizon Prediksi

## 1. Sumber Dataset

Dataset yang digunakan adalah **OhioT1DM** (Marling & Bunescu, 2020), kumpulan
data Diabetes Tipe 1 dari 12 pasien (kohort 2018: 6 pasien, kohort 2020: 6 pasien)
selama ±8 minggu per pasien.  Data ini dipilih karena:
- Merupakan dataset T1DM multimodal publik dengan CGM, insulin, makanan, dan gaya hidup.
- Diakui sebagai benchmark standar riset prediksi glukosa (bdk. Martinsson et al., 2020).
- Tersedia secara bebas untuk penelitian akademik.

Parser XML (`src/data/ohio_parser.py`, `process_ohio_dataset()`) menghasilkan **dua** CSV:
- `data/raw/ohio_t1dm_merged.csv` — timeline **CGM 5-menit** (untuk pelatihan model).
- `data/raw/ohio_t1dm_smbg.csv` — timeline **`finger_stick` (SMBG nyata)** (untuk skenario SMBG).

Parser meng-*align* event (insulin bolus, basal step-function, makanan, dll.) ke grid CGM
dengan toleransi ±2.5 menit, sehingga fitur `insulin`, `carbs`, dan `basal_rate` benar-benar
terisi (multimodal). Detail perbaikan parser & justifikasinya ada di `docs/journey.md`.

## 2. Cadence: CGM (Pelatihan) dan SMBG Nyata (finger_stick)

### 2.1 Data Mentah: CGM 5-menit

OhioT1DM merekam glukosa setiap **5 menit** (Continuous Glucose Monitor / CGM),
menghasilkan ±288 titik/hari per pasien.  Cadence ini jauh lebih padat daripada
kondisi klinis yang menjadi target sistem ini: pasien tanpa CGM yang menggunakan
**Self-Monitoring of Blood Glucose** (SMBG) — pengukuran mandiri beberapa kali per hari.

### 2.2 Justifikasi Penggunaan Data CGM untuk Pelatihan

Model dilatih pada **data CGM (5-menit)** dengan alasan:

1. **Kualitas sinyal** — resolusi temporal CGM cukup untuk menangkap tren glukosa
   postprandial dan nocturnal yang informatif bagi model prediktif.
2. **Volume data** — ±166 ribu baris memberi cukup contoh untuk melatih model;
   timeline SMBG jauh lebih sedikit (±4.5 ribu baris).
3. **Kinerja empiris** — pada horizon standar BGLP, RF mencapai RMSE **22.7 mg/dL**
   (+30 menit) dan **34.5 mg/dL** (+60 menit), sebanding dengan literatur OhioT1DM,
   menjadi *baseline* untuk perbandingan LSTM (T4).

`sequence_length = 12` pada cadence 5-menit = **jendela look-back 1 jam**,
cukup untuk menangkap dinamika postprandial (puncak glukosa umumnya 45–90 menit
setelah makan, ADA 2023).

### 2.3 SMBG Nyata (finger_stick), Bukan Simulasi

OhioT1DM **sudah memuat pembacaan SMBG nyata** melalui kanal `<finger_stick>`
(±397 pembacaan/pasien). Maka skenario SMBG memakai **data nyata ini** langsung
(`ohio_t1dm_smbg.csv`), bukan hasil downsampling CGM yang artifisial. Keputusan ini diambil
karena: bila data nyata tersedia, mengartifisialkannya tidak menambah validitas dan justru
menambah asumsi.

> Catatan: fungsi `DataPreprocessor.downsample_smbg()` tetap tersedia sebagai **utilitas
> opsional** (mis. uji robustness), tetapi **bukan basis metodologi** lagi.

| Aspek | CGM (pelatihan) | SMBG nyata (finger_stick) |
|---|---|---|
| Sumber | `<glucose_level>` | `<finger_stick>` |
| Jumlah baris | ±166.533 | ±4.566 |
| Cadence | 5 menit | tidak teratur (beberapa per hari) |
| Peran | Melatih & menguji model | Skenario deployment SMBG |

## 3. Pemilihan Fitur Prediktor

Fitur model = `[glucose, carbs, insulin, activity]` (stres dikeluarkan; lihat bawah).
Pemilihan ini berbasis **dua landasan**: fisiologi/literatur dan bukti empiris dari data.

### 3.1 Landasan Fisiologis & Literatur

Dinamika glukosa darah secara klasik dimodelkan dari interaksi **glukosa–insulin–karbohidrat**:
- **Minimal Model** (Bergman) dan simulator **UVA/Padova** (basis *artificial pancreas*)
  memodelkan glukosa sebagai fungsi insulin & asupan karbohidrat.
- Karya prediksi glukosa berbasis OhioT1DM (mis. Mirshekarian et al., 2017/2019) memakai
  input CGM + insulin + makanan.
- **Aktivitas fisik** didukung sebagai faktor sekunder (olahraga menurunkan glukosa).

> Catatan: sitasi di atas perlu diverifikasi penulis sebelum masuk laporan final.

### 3.2 Bukti Empiris (Feature Importance RF, per horizon)

Importance RF (agregat per fitur) mengonfirmasi urutan yang sesuai fisiologi, dan
menunjukkan kontribusi fitur **meningkat pada horizon lebih panjang**:

| Horizon | glucose | insulin | carbs | activity |
|---|---|---|---|---|
| +5 menit | 99.77% | 0.19% | 0.04% | 0.01% |
| +30 menit | 96.49% | 2.65% | 0.76% | 0.10% |
| +60 menit | 90.05% | 7.53% | 2.27% | 0.15% |

Pada +5 menit, glukosa terakhir mendominasi (tugas ≈ *persistence*), sehingga fitur lain
nyaris tak berkontribusi — salah satu alasan horizon diperpanjang (lihat §4).

### 3.2.1 Fitur Engineered (konfigurasi final)

Fitur mentah per-bin (`carbs`, `insulin`) menyimpan efek yang tertunda & tersebar, sehingga
kontribusinya kecil. Maka fitur final memakai versi **berbasis fisiologi**:
`[glucose, glucose_delta, iob, cob, activity, hour_sin, hour_cos]` + target **Δglukosa**.
- **iob** = Insulin-on-Board (peluruhan ~4 jam), **cob** = Carbs-on-Board (~3 jam) —
  model glukosa-insulin-karbohidrat (Bergman; UVA/Padova).
- **glucose_delta** = tren; **hour_sin/cos** = pola diurnal.

**Dampak akurasi:** ~tidak berubah (RMSE +30 RF 22.70→22.60; LSTM 21.94→22.04) — akurasi
lintas-pasien dibatasi autokorelasi glukosa.

**Dampak struktur model (penting):** importance jadi benar-benar multimodal —

| Fitur | Baseline (+30) | Engineered (+30) |
|---|---|---|
| glukosa (level) | 96.49% | 23.08% |
| glucose_delta | — | 39.53% |
| iob (insulin) | 2.65% | 11.90% |
| cob (karbohidrat) | 0.76% | 10.38% |
| hour_sin+cos | — | 14.68% |
| activity | 0.10% | 0.44% |

Kontribusi insulin+karbohidrat naik dari ~3.4% → **~22.3%** → klaim "prediksi multimodal"
terbukti empiris.

### 3.3 Stres Dikeluarkan

OhioT1DM hanya memuat **7 event stressor** di seluruh 12 pasien → kolom `stress`
nol-varians dan tak informatif. Karena itu `stress` **dikeluarkan dari fitur model**, namun
**tetap dipertahankan** sebagai variabel Digital Twin / logbook (relevan saat deployment).
Sinyal *sensor band* (heart rate, GSR, suhu kulit, step count) **sengaja tidak dipakai**
(kompleksitas + ketersediaan berbeda antar-kohort) → dicatat sebagai future work.

## 4. Horizon Prediksi

Sistem memprediksi glukosa pada **horizon +30 menit dan +60 menit** — standar
**BGLP Challenge** (OhioT1DM) dan mayoritas literatur prediksi glukosa.

Alasan **tidak** memakai +5 menit:
- +5 menit ≈ *persistence* (glukosa praktis tak berubah) → tugas trivial, fitur multimodal
  tak berkontribusi (lihat tabel §3.2).
- Premis novelty "Prediction-Conditioned RAG" menuntut prediksi yang benar-benar
  **menatap masa depan**: +30/60 menit memberi jendela antisipasi klinis yang bermakna
  (cukup waktu mencegah hipo/hiper — sesuai motivasi paper OhioT1DM).

Konfigurasi: `prediction_horizons: [6, 12]` (langkah 5-menit), `default_horizon: 6`
(bundle aplikasi memakai +30 menit). Untuk SMBG (finger_stick), horizon mengikuti
pembacaan berikutnya.

## 5. Train/Test Split

Pembagian data dilakukan **per pasien** (bukan per baris) untuk menghindari
data leakage temporal:
- **Train**: 80% pasien (8 dari 10 yang digunakan)
- **Test**: 20% pasien (2 pasien hold-out)

Fungsi: `DataPreprocessor.split_by_patient(df, test_patients=[...])`.

Normalisasi (`StandardScaler`) di-*fit* hanya pada data training dan di-*transform*
pada data test.

## 6. Reproduksi

```bash
# 1. Parse OhioT1DM XML → 2 CSV (CGM merged + SMBG finger_stick)
python -m src.data.ohio_parser

# 2. Latih model RF (bundle inferensi untuk aplikasi)
python -m src.models.rf_model --config config.yaml --data_source ohio_t1dm

# 3. Evaluasi komparatif RF vs LSTM (RMSE, MAE, Clarke Error Grid)
python scripts/eval_rf_lstm.py --config config.yaml
# Hasil → results/eval_prediksi/
```

## Referensi

> Tanda **[✓PDF]** = dirujuk pada daftar pustaka paper OhioT1DM (`docs/OhioT1DM-dataset-paper.pdf`).
> Tanda **[verifikasi]** = dari pengetahuan umum bidang; cek sitasi persis sebelum laporan final.

**Dataset & horizon (30/60 menit):**
- Marling, C. & Bunescu, R. (2020). *The OhioT1DM Dataset for Blood Glucose Level
  Prediction: Update 2020.* CEUR Workshop Proceedings, KDH@ECAI 2020. *(tersedia di
  `docs/OhioT1DM-dataset-paper.pdf`; konvensi BGLP Challenge memakai PH 30 & 60 menit)*
- Mirshekarian, S. et al. (2017). *Using LSTMs to learn physiological models of blood
  glucose behavior.* EMBC 2017. **[✓PDF, ref 7]**
- Mirshekarian, S. et al. (2019). *LSTMs and neural attention models for blood glucose
  prediction.* EMBC 2019. **[✓PDF, ref 8]**

**Pemilihan fitur (glukosa–insulin–karbohidrat, aktivitas):**
- Bunescu, R. et al. (2013). *Blood glucose level prediction using physiological models
  and support vector regression.* ICMLA 2013. **[✓PDF, ref 1]**
- Bergman, R.N. et al. (1979). *Quantitative estimation of insulin sensitivity (minimal
  model).* Am. J. Physiology. **[verifikasi]**
- Dalla Man, C. et al. (2014). *The UVA/PADOVA Type 1 Diabetes Simulator: New Features.*
  J. Diabetes Sci. Technol., 8(1), 26–34. **[verifikasi]**
- Oviedo, S. et al. (2017). *A review of personalized blood glucose prediction.* Int. J.
  Numer. Method Biomed. Eng. **[verifikasi]**
- Riddell, M.C. et al. (2017). *Exercise management in type 1 diabetes: a consensus
  statement.* Lancet Diabetes & Endocrinol., 5(5), 377–390. **[verifikasi]**

**Standar klinis:**
- American Diabetes Association (2023). *Standards of Care in Diabetes.* Diabetes Care,
  46(Suppl. 1). **[verifikasi]**
