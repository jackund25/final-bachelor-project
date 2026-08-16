# Prapendaftaran T12 — menyelaraskan model kalibrasi conformal dengan model produksi

> **Ditulis 16 Agustus 2026, SEBELUM satu pun angka cakupan baru dilihat.**
> Cabang `refaktor-tujuh-tugas`.
>
> Diverifikasi saat penulisan: `results/eval_prediksi/conformal_h6.json` dan
> `conformal_h12.json` masih berisi angka hasil pembagian 8/2/2, dan
> `models/gbm_inference_bundle_h6.pkl` masih model 10 pasien.

---

## 1. Cacatnya, dinyatakan persis

| | Melatih pada | Sumber |
|---|---|---|
| Bundel produksi | **10 pasien** (`patient_ids[-2:]` sebagai uji) | `src/models/gbm_model.py:246` |
| Kalibrasi conformal | **8 pasien** (`pids[:-4]`) | `scripts/conformal_calibration.py:105` |

Kuantil `q` dihitung atas model yang **berbeda** dari model yang membuat prediksi.

### Ini cacat RUNTIME, bukan sekadar cacat pelaporan

Sempat diduga `q` hanya masuk laporan. **Tidak.** Jalurnya:

```
scripts/conformal_calibration.py  ->  results/eval_prediksi/conformal_h{N}.json
src/conformal.py:load_calibration()   membaca berkas itu
src/conformal.py:conformal_factor()   mengambil levels.95.conformal_normalized.q
src/conformal.py:prediction_interval() margin = q * prediction_std
app/streamlit_app.py:296              menampilkan interval itu kepada dokter
```

Jadi **interval ketidakpastian yang dilihat dokter** memakai `q` yang dikalibrasi
untuk model 8 pasien, sedangkan angka prediksinya berasal dari model 10 pasien.
Jaminan cakupan 95% **tidak berlaku** bagi model yang sebenarnya dipakai.

---

## 2. Mengapa ini tidak dapat diperbaiki dengan cara yang murah

Split conformal menuntut himpunan kalibrasi yang **tidak pernah dilihat** model.
Dengan 12 pasien, pilihan yang tersedia:

| Pilihan | Sah? | Sebab |
|---|---|---|
| Latih 10, kalibrasi pada 2 pasien uji | Sebagian | `q` menjadi sah, tetapi tidak tersisa himpunan ketiga untuk melaporkan cakupan empiris secara jujur |
| Latih 10, kalibrasi pada pasien yang sudah dilihat | **Tidak** | Bocor; interval menjadi terlalu sempit |
| **Latih 8, kalibrasi 2, uji 2** | **Ya** | Tiga himpunan terpisah: `q` sah, dan cakupan empiris dapat dilaporkan pada himpunan yang tidak dipakai keduanya |
| CV+ / Jackknife+ | **Tidak dipakai** | Lihat catatan di bawah |

**Catatan CV+/Jackknife+ — dan mengapa ditolak.** Metode ini memungkinkan seluruh
data dipakai untuk melatih sekaligus mengkalibrasi, sehingga secara teori
menyelesaikan masalah tanpa mengecilkan himpunan latih. Angelopoulos & Bates (2021)
§6.2 hal. 28 menyebutnya, **tetapi tidak memerikan algoritmanya** — teksnya
berbunyi bahwa pembaca dirujuk ke [41] dan [42] untuk uraian yang tepat. Kedua
rujukan itu **belum dibaca dan tidak ada di koleksi**. Aturan proyek melarang
mengutip yang belum dibaca, sehingga CV+ tidak dipakai. Yang diperikan penuh oleh
paper yang ada adalah **split conformal**, dan itulah yang dipakai.

**Keputusan: pembagian 8/2/2 dipakai SERAGAM**, dan bundel produksi dilatih ulang
pada 8 pasien yang sama dengan yang dipakai kalibrasi conformal.

---

## 3. Biaya yang harus disadari sebelum menyetujui

Ini bukan perbaikan yang terisolasi. Bundel produksi berpindah dari 10 pasien ke 8,
sehingga **seluruh angka prediksi yang dilaporkan berubah**:

- RMSE, MAE, MAPE, dan Clarke Error Grid pada h6 dan h12
- Sensitivitas/spesifisitas/PPV hipoglikemia
- Angka pengklasifikasi kondisi
- Skenario deployment SMBG
- Seluruh angka itu muncul di Abstrak dan Bab VI

Arahnya dapat diperkirakan tetapi besarnya tidak: model dengan data latih lebih
sedikit **biasanya** sedikit lebih buruk.

---

## 4. Dugaan yang didaftarkan di muka

- **D1 — keselarasan.** Sesudah perbaikan, jumlah pasien latih pada
  `conformal_calibration.py` WAJIB sama persis dengan pada `gbm_model.py`, dan
  keduanya wajib menuliskan daftar pasiennya ke dalam artefak. Ini pemeriksaan
  mekanis, bukan statistik; bila gagal, tidak ada angka yang dilaporkan.
- **D2 — arah RMSE.** Melatih pada 8 pasien alih-alih 10 diperkirakan **menaikkan**
  RMSE h6 (memburuk), dengan kenaikan **kurang dari 3 mg/dL**. Bila kenaikannya
  melampaui itu, yang dicurigai lebih dulu adalah pipeline pelatihannya, bukan
  kesimpulan bahwa data lebih sedikit merugikan.
- **D3 — arah cakupan TIDAK didugakan.** Sengaja. Cakupan conformal dijamin secara
  teori pada tingkat yang diminta, sehingga cakupan empiris seharusnya mendekati
  95% **baik sebelum maupun sesudah** perbaikan. Yang berubah adalah **keabsahan**
  jaminannya, bukan angkanya. Justru itu sebabnya cacat ini berbahaya: **angka
  cakupan yang salah pun akan tetap terlihat wajar.**
- **D4 — keterlacakan.** Bundel produksi saat ini **tidak merekam** pasien mana
  yang dipakai melatih (diperiksa langsung: kunci bundel hanya `model`, `scaler`,
  `features`, `sequence_length`, `prediction_horizon`, `use_engineered`,
  `predict_delta`, `feature_engineering`, `std_method`, `std_models`,
  `std_quantiles`, `model_family`). Sesudah perbaikan, bundel WAJIB memuat
  `train_patients`, `calibration_patients`, dan `test_patients`, sehingga
  ketidakselarasan seperti ini dapat terdeteksi otomatis di kemudian hari,
  bukan lewat pembacaan kode secara manual.

---

## 5. Aturan penafsiran yang mengikat

1. **Bila D1 gagal:** berhenti, tidak ada angka yang dilaporkan.
2. **Bila RMSE memburuk lebih dari 3 mg/dL:** periksa pipeline lebih dulu; jangan
   langsung menyimpulkan "data lebih sedikit memang lebih buruk".
3. **Perbaikan ini TIDAK boleh dibatalkan hanya karena angkanya jadi kurang
   menguntungkan.** Interval yang tidak sah lebih buruk daripada interval yang
   lebih lebar. Membatalkannya berarti memilih metode berdasarkan hasil hilirnya —
   kesalahan yang sama yang dihindari prapendaftaran lain di proyek ini.
4. **Angka lama TIDAK dihapus.** Disimpan sebagai `*_10pasien_arsip.json` beserta
   catatan bahwa `q`-nya tidak selaras, agar perubahannya dapat ditelusuri.

---

## 6. Keluaran

- `models/gbm_inference_bundle_h6.pkl` dan `_h12.pkl` — dilatih ulang pada 8 pasien,
  memuat medan keterlacakan baru.
- `results/eval_prediksi/conformal_h6.json` dan `conformal_h12.json` — diperbarui.
- Arsip angka lama.
- Seluruh angka prediksi hilir dihitung ulang, lalu Bab VI dan Abstrak menyusul.
