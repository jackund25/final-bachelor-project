# Prapendaftaran T4.1 — Gradient Boosting sebagai pembanding ketiga

> **Ditulis 11 Agustus 2026 pada commit `55bf85a`.**
> **T4.1 BELUM dijalankan.** `results/eval_prediksi/gradient_boosting_h6.json` dan
> `_h12.json` belum ada, dan tidak satu pun angka GBM sudah dilihat. Skrip
> `scripts/eval_gradient_boosting.py` sudah ada tetapi belum pernah dieksekusi; perubahan
> yang akan dilakukan padanya (checkpoint dan metrik hipoglikemia) dicatat di bawah dan
> ditulis **setelah** berkas ini disimpan.

## Aturan keputusan, ditetapkan sebelum melihat hasil

**Model produksi TIDAK diganti pada sesi ini, apa pun hasilnya.** Bukan karena hasilnya
diperkirakan tidak menarik, melainkan karena mengganti prediktor membatalkan seluruh angka
hilir sekaligus: crossfold retrieval, realcases, T3.1, T3.2, T3.3, kalibrasi konformal, dan
RAGAS yang terkunci kuota 20 panggilan per hari untuk sebagian model.

Bila GBM unggul, ia masuk Bab VI sebagai **pembanding** dan sebagai **saran pengembangan
dengan besaran terukur** — bukan sebagai perubahan yang dikerjakan sekarang. Aturan ini
ditulis di muka supaya tidak ada godaan menafsirkan hasil sebagai mandat mengganti sesuatu.

**Hiperparameter dibiarkan bawaan.** RF produksi (200/20/5) juga tidak pernah disetel
ekstensif. Menyetel GBM tetapi tidak menyetel RF membuat perbandingan berat sebelah. Akibat
yang dicatat terus terang: hasil ini adalah **batas BAWAH** kemampuan gradient boosting.
Bila GBM menang pada keadaan tak-tersetel, keunggulannya lebih meyakinkan; bila kalah,
kesimpulannya hanya berlaku untuk keadaan tak-tersetel.

## Angka acuan yang sudah diketahui

6 fold lintas-pasien, jendela tersegmentasi `max_gap_steps` 6, aturan fold `pids[i::6]`.

| | RF | LSTM |
|---|---|---|
| RMSE +30 mnt | 21,08 ± 1,15 | 20,60 ± 1,33 |
| RMSE +60 mnt | 34,78 ± 1,50 | 34,50 ± 1,45 |
| Clarke A+B +30 | 94,35 ± 0,55% | 94,89 ± 0,52% |
| ukuran model | 322 MB | 0,124 MB |
| **sensitivitas hipo +30** | **21,68 ± 4,72%** | **32,11 ± 6,21%** |
| **sensitivitas hipo +60** | **4,36 ± 1,97%** | **10,56 ± 6,75%** |
| bias pada hipo +30 / +60 | +21,03 / +42,66 | +17,81 / +38,03 |
| bias keseluruhan | +0,21 | −0,04 |

## Yang diperkirakan, dinyatakan sebelum diukur

### D1 — RMSE GBM setara RF, selisihnya di bawah SD antar-fold

`HistGradientBoostingRegressor` bawaan berarti 100 iterasi, `learning_rate` 0,1, dan 31
daun per pohon, atas 84 fitur (12 langkah × 7 fitur) yang memprediksi delta.

**Perkiraan: RMSE GBM +30 mnt berada di rentang 20,5–22,0 mg/dL, dan |RMSE_RF − RMSE_GBM|
lebih kecil daripada SD antar-fold RF (1,15).** Konsekuensinya selisihnya **tidak layak
dinarasikan sebagai keunggulan** menurut aturan pelaporan yang berlaku sejak T1.1, betapa
pun kecilnya p tingkat sampel.

Perkiraan untuk +60 mnt: 34,0–36,0 mg/dL, dengan kesimpulan pelaporan yang sama.

### D2 — GBM jauh lebih kecil daripada RF, seordinal LSTM

100 iterasi × 31 daun adalah struktur yang jauh lebih ringkas daripada 200 pohon berkedalaman
20 yang mencapai batas kedalamannya (T1.1b mengukur 200/200 pohon menyentuh `max_depth`).

**Perkiraan: ukuran berkas GBM berada di rentang 0,1–1,0 MB**, yaitu dua sampai tiga orde
lebih kecil daripada RF 322 MB dan seordinal LSTM 0,124 MB.

**Perkiraan waktu latih: kurang dari sepertiga waktu latih RF per fold.**

Bila D2 benar, ia mengubah bentuk argumen keterterapan pada Bab VI: bukan lagi "RF besar
lawan LSTM kecil", melainkan "hanya RF yang besar, dan dua keluarga model lain memberi
akurasi setara dengan berkas seukuran LSTM".

### D3 — GBM TIDAK memperbaiki sensitivitas hipoglikemia, dan alasannya mekanistik

Ini perkiraan yang paling menentukan bagi keputusan menggantung #5, dan yang paling ingin
saya lihat terbantah.

RF menyusut ke tengah pada kelas langka karena ia meminimalkan galat kuadrat; T2.1
mengukurnya sebagai bias **+21,03 mg/dL pada kejadian hipoglikemia** padahal bias
keseluruhannya **+0,21**. Gradient boosting bawaan sklearn meminimalkan **galat kuadrat yang
sama**. Mekanismenya karena itu tidak berubah oleh pergantian keluarga model — ia melekat
pada fungsi kerugiannya, bukan pada cara pohonnya dibangun.

**Perkiraan:**

| besaran | perkiraan GBM | pembanding |
|---|---|---|
| sensitivitas hipo +30 | **15–30%** | RF 21,68% · LSTM 32,11% |
| sensitivitas hipo +60 | **2–10%** | RF 4,36% · LSTM 10,56% |
| bias pada hipo +30 | **+15 sampai +25 mg/dL** | RF +21,03 · LSTM +17,81 |
| bias keseluruhan | **|bias| < 1,5 mg/dL** | RF +0,21 · LSTM −0,04 |

Artinya: **GBM diperkirakan berada di dekat RF, bukan di dekat LSTM**, pada aspek yang
paling kritis secara klinis.

**Bila D3 terbantah dan GBM justru mendekati atau melampaui LSTM**, penjelasan "fungsi
kerugian yang menentukan" runtuh, dan yang harus dicari adalah apa pada LSTM yang
memberinya sensitivitas lebih tinggi — kemungkinan besar penanganan urutan waktunya, bukan
kerugiannya. Itu akan menjadi temuan yang lebih berharga daripada D3 yang terpenuhi.

### D4 — reproduksibilitas RF terpenuhi

RF dilatih ulang di dalam skrip T4.1, bukan diambil dari berkas, supaya galat per sampel
tersedia untuk uji berpasangan. **Perkiraan: RMSE RF per fold menyimpang kurang dari 0,05
mg/dL dari `crossval_rf_vs_lstm_h{6,12}.json`.**

Bila menyimpang lebih dari itu, ada sesuatu yang berubah sejak T1.1, angka LSTM tersimpan
**tidak sepadan**, dan seluruh perbandingan tiga model **dihentikan dan dilaporkan** alih-alih
disandingkan. Penjaga ini diverifikasi menyala lebih dulu dengan angka yang sengaja
disalahkan, seperti penjaga D1 pada T3.3.

## Perubahan yang akan dilakukan pada skrip, dicatat sekarang

1. **Checkpoint per fold** ke `.cache/`, bukan `results/`, menyimpan |galat| per sampel RF
   dan GBM (tanpa itu Wilcoxon tidak dapat dilanjutkan dan checkpoint-nya tidak berguna),
   beserta metadata horizon, `max_gap_steps`, dan pembagian fold — supaya checkpoint yang
   tidak sepadan **ditolak, bukan dipakai diam-diam**. Pelajaran T1.1b: sepuluh konfigurasi
   dan ~40 menit hilang karena keluaran hanya ditulis di akhir.
2. **Metrik hipoglikemia** untuk GBM dan RF, memakai ulang `metrik_hipo()` dari
   `eval_hipoglikemia.py` **tanpa menulis ulang definisinya**, sehingga ambang 70 dan 54,
   penanganan pembagi nol, dan konvensi tanda bias identik dengan T2.1. Angka LSTM diambil
   dari `hipoglikemia_h{6,12}.json` yang aturan foldnya sudah identik.
3. **Dijalankan untuk h6 DAN h12.** Bukti terkuat yang menantang RF (sensitivitas hipo 4,4%
   lawan 10,6%) ada di +60 menit; membandingkan GBM hanya di +30 menit meninggalkan horizon
   paling menentukan tanpa pembanding ketiga.

Ketiganya menambah keluaran dan penjagaan; **tidak satu pun mengubah cara model dilatih
atau dinilai**, sehingga angka yang dihasilkan tetap sepadan dengan T1.1 dan T2.1.

## Aturan penafsiran

1. **Selisih yang lebih kecil daripada SD antar-fold tidak dinarasikan sebagai keunggulan**,
   berapa pun p tingkat sampelnya. Uji tingkat sampel pada n>150.000 akan menolak H₀ untuk
   perbedaan yang tidak berarti secara klinis.
2. **Uji GBM lawan LSTM hanya sah di tingkat fold.** LSTM tidak dilatih ulang di sini,
   sehingga galat per sampelnya tidak tersedia dan uji tingkat sampel tidak dapat dilakukan.
3. **Tiga model bukan tiga kandidat setara.** RF dan LSTM sudah pernah dinilai penuh; GBM
   tak-tersetel. Perbandingannya adalah "apakah keluarga model yang lazim menang pada data
   tabular memberi sesuatu yang belum terlihat", bukan pemilihan juara.
4. Hasil T4.1 **tidak memutuskan** keputusan menggantung #5. Ia melengkapi datanya, dan
   ringkasan keputusannya disiapkan untuk pembimbing.

## Kaitan

- `docs/PRAPENDAFTARAN_T3.3.md` — pola yang diikuti, termasuk pencatatan dugaan yang terbantah
- `results/eval_prediksi/crossval_rf_vs_lstm_h{6,12}.json` — acuan RMSE dan sumber D4
- `results/eval_prediksi/hipoglikemia_h{6,12}.json` — sumber angka LSTM untuk D3
- Keputusan menggantung #5 pada `docs/HANDOFF.md`, yang bobotnya dinaikkan T3.2: jarak ke
  oracle +0,2646 menunjukkan tuas terbesar ada di sisi prediksi
