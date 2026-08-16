# Argumen pemilihan Gradient Boosting sebagai prediktor produksi

Disusun 16 Agustus 2026 sebagai bahan bimbingan. Setiap angka dapat ditelusuri ke
berkas di `results/eval_prediksi/`; jalur berkasnya disebut di tiap tabel.

**Cara memakai dokumen ini.** Bagian 1–3 adalah argumennya. Bagian 4 adalah
**kelemahan yang harus Anda sebut lebih dulu sebelum ditanya** — argumen yang
menyembunyikan kelemahan akan runtuh begitu penguji menemukannya, sedangkan
argumen yang menyebutkannya duluan menunjukkan Anda menguasai datanya. Bagian 5
adalah pertanyaan yang paling mungkin diajukan beserta jawabannya.

---

## 1. Landasan: kriteria pemilihannya bukan RMSE

Prediktor dalam penelitian ini adalah **komponen, bukan kontribusi**. Kebaruannya
ada pada penyusunan kueri *retrieval* dari kondisi yang **diprediksi**. Karena itu
yang menentukan mutu sistem bukan seberapa kecil galat regresi, melainkan
**seberapa sering kondisi masa depan tertebak benar** — sebab kondisi itulah yang
menyusun kueri.

Konsekuensinya: RMSE hanya proksi, dan proksi yang lemah. Model dengan RMSE lebih
kecil dapat saja lebih buruk dalam menebak kelas kondisi, dan kelas itulah yang
dipakai sistem.

**Ini bukan pembenaran yang dibuat belakangan.** Ia sudah tertulis sebagai
identitas penelitian sejak awal, dan konsisten dengan pemilihan metrik pada Bab VI
yang menempatkan sensitivitas hipoglikemia dan akurasi kondisi di atas RMSE.

---

## 2. Pada sumbu yang menentukan, GBM unggul

Sumber: `results/eval_prediksi/condition_classifier.json` dan
`condition_classifier_RF_arsip.json`, horizon +30 menit, n uji 26.445.

### 2.1 Akurasi kondisi pada kasus divergen — sumbu paling menentukan

Kasus **divergen** adalah kasus yang kondisi sekarang berbeda dari kondisi masa
depan. Di sanalah PC-RAG bermakna; pada kasus non-divergen, RAG standar sudah
cukup.

| Jalur | RF | GBM | Selisih |
|---|---:|---:|---:|
| Regresi lalu ambang | 36,5% | 36,8% | +0,3 |
| **Pengklasifikasi kondisi** | 29,0% | **35,4%** | **+6,4** |

### 2.2 Sensitivitas hipoglikemia — kelas paling berbahaya

| Jalur (h6) | RF | GBM |
|---|---:|---:|
| Regresi lalu ambang | 15,4% | **17,3%** |
| **Pengklasifikasi kondisi** | 45,2% | **67,2%** |

Kenaikan **+22 poin** pada kelas yang kegagalannya paling berbahaya.

### 2.3 Regresi: setara, sedikit lebih baik

Sumber: `results/eval_prediksi/gradient_boosting_h6.json` dan `_h12.json`,
rerata ± SD lintas 6 fold.

| Metrik | RF | GBM |
|---|---:|---:|
| RMSE +30 mnt | 21,079 ± 1,152 | **20,721 ± 1,163** |
| RMSE +60 mnt | 34,784 ± 1,499 | **33,808 ± 1,443** |
| Clarke A+B +30 mnt | 94,350% | **94,686%** |
| Clarke A+B +60 mnt | 85,343% | **85,827%** |

Selisihnya kecil dan berada di dalam SD antar-fold. **Jangan mengklaim GBM lebih
akurat**; klaim yang benar adalah **setara**.

### 2.4 Biaya komputasi — memenuhi batasan desain, bukan sekadar nyaman

Sumber: medan `durasi_detik` pada `results/eval_prediksi/conformal_h*.json` dan
arsip RF-nya. Beban kerja identik, mesin sama.

| Horizon | RF | GBM | Rasio |
|---|---:|---:|---:|
| +30 mnt | 545,6 dtk | **17,4 dtk** | **31×** |
| +60 mnt | 509,0 dtk | **17,7 dtk** | **29×** |

Ukuran artefak: `rf_inference_bundle_h6.pkl` **334 MB** lawan
`gbm_inference_bundle_h6.pkl` **1,45 MB** — sekitar **230× lebih kecil**.

Ini penting karena batasan desain penelitian **sudah menyatakan** sistem harus
berjalan tanpa GPU pada layanan bersumber daya terbatas. Jadi ia **memenuhi
batasan yang dinyatakan sendiri**, bukan keuntungan sampingan yang dicari-cari.

---

## 3. Argumen terkuat: yang berubah bukan modelnya, melainkan CARA memprediksi kondisi

Angka paling penting di seluruh dokumen ini:

| Cara memprediksi kondisi hipoglikemia (h6) | Sensitivitas |
|---|---:|
| Regresi RF lalu ambang | 21,7% |
| Regresi GBM lalu ambang | 23,9% |
| Regresi LSTM lalu ambang | 32,1% |
| **Pengklasifikasi kondisi GBM, sadar-biaya** | **67,2%** |

Pengklasifikasi kondisi mengungguli **setiap** jalur regresi, termasuk LSTM yang
paling akurat regresinya, dengan selisih besar.

**Maknanya:** memperbaiki regresi bukan jalan untuk memperbaiki deteksi kondisi.
Menurunkan kondisi dari regresi selalu kehilangan informasi, karena regresi
dioptimalkan untuk galat kuadrat rata-rata, bukan untuk batas kelas yang penting
secara klinis. Pertanyaan "RF atau GBM" karena itu **bukan pertanyaan terpenting**
— pertanyaan terpentingnya adalah "regresi atau klasifikasi", dan jawabannya
klasifikasi.

GBM dipilih karena ia keluarga model yang menyediakan **kedua jalur** dengan biaya
komputasi yang memenuhi batasan desain.

---

## 4. Kelemahan yang harus disebut lebih dulu

### 4.1 GBM lebih buruk pada sensitivitas hipoglikemia +60 menit

Sumber: `results/eval_prediksi/gradient_boosting_h12.json`.

| | RF | GBM |
|---|---:|---:|
| Sensitivitas hipo +60 mnt | **4,355 ± 1,972%** | 2,419 ± 2,034% |
| Hipo BERAT terlewat | **933** | 957 |
| PPV hipo +60 mnt | **57,963%** | 50,465% |

**Status statistiknya bercabang, dan Anda harus tahu keduanya:**

| Kriteria | Nilai |
|---|---|
| `wilcoxon_p` | **0,031** — bermakna pada 0,05 |
| `melampaui_sd.selisih_berpasangan` | true |
| `melampaui_sd.maks_per_model` | false |
| `melampaui_sd.gabungan_definisi_T2.1` | false |
| `sd_sepakat` | **false** |
| `layak_dinarasikan_konservatif` | **false** |

**Cara menyebutnya yang jujur:** *"Pada horizon 60 menit GBM melewatkan 24 kasus
hipoglikemia berat lebih banyak daripada RF (957 lawan 933). Uji Wilcoxon tingkat
fold memberi p = 0,031, tetapi ketiga definisi SD pada aturan pelaporan kami tidak
sepakat, sehingga menurut KEPUTUSAN #8 selisih ini belum layak dinarasikan sebagai
temuan. Kami mencatatnya sebagai kelemahan yang diketahui, bukan sebagai selisih
yang dapat diabaikan."*

**Jangan** mengatakan selisihnya tidak bermakna. Wilcoxon-nya bermakna.

### 4.2 Pengklasifikasi GBM menukar ketepatan demi kepekaan

| Ukuran (h6, jalur pengklasifikasi) | RF | GBM |
|---|---:|---:|
| Akurasi keseluruhan | **88,2%** | 86,1% |
| PPV hipoglikemia | **40,9%** | 31,9% |
| Sensitivitas kelas `normal` | **91,2%** | 85,6% |
| Sensitivitas hipoglikemia | 45,2% | **67,2%** |

Ini pertukaran, dan harus disebut sebagai **pilihan yang disengaja**, bukan
kebetulan yang menguntungkan. Alasannya klinis: pada hipoglikemia, **negatif palsu
jauh lebih berbahaya daripada positif palsu**. Yang terlewat dapat berujung kejang
dan penurunan kesadaran; yang berlebih berujung pemeriksaan glukosa ulang. Sistem
ini pun bersifat **pendukung keputusan dengan dokter di dalam alur**, sehingga
positif palsu tersaring oleh penilaian dokter, sedangkan negatif palsu tidak
pernah sampai kepadanya untuk disaring.

### 4.3 Klaim di dokumen sendiri yang harus dicabut

`docs/HANDOFF.md:26` menyatakan **"GBM > RF di tiap dimensi"**. **Itu salah** dan
dibantah oleh 4.1 dan 4.2. Cabut sebelum bimbingan — penguji dapat menemukannya.

---

## 5. Pertanyaan yang paling mungkin diajukan

**"LSTM lebih baik pada beberapa metrik. Mengapa tidak LSTM?"**
Benar, dan sebut sendiri angkanya: LSTM unggul pada RMSE +30 mnt (20,602 lawan
20,721), Clarke A+B (94,891% lawan 94,686%), dan terutama sensitivitas hipoglikemia
regresi (32,1% lawan 23,9%). Tiga jawaban:
1. Sensitivitas hipoglikemia terbaik **tidak dicapai oleh regresi mana pun**,
   melainkan oleh pengklasifikasi kondisi (67,2%). Jadi keunggulan LSTM pada
   sumbu itu tidak menyelesaikan masalah yang sedang diselesaikan.
2. Selisih RMSE-nya (0,119 mg/dL) jauh di dalam SD antar-fold (±1,1–1,3).
3. Batasan desain menuntut tanpa GPU. LSTM bertentangan dengan batasan yang sudah
   dinyatakan dan diverifikasi sebagai bagian dari kontribusi.

**"Mengapa tidak pakai RF saja, toh setara?"**
Karena setara pada regresi tetapi **tidak setara** pada sumbu yang menentukan:
akurasi kondisi divergen +6,4 poin dan sensitivitas hipoglikemia pengklasifikasi
+22 poin, dengan biaya komputasi 31× lebih murah dan artefak 230× lebih kecil.

**"Apakah hiperparameternya disetel?"**
**Tidak, sengaja.** GBM memakai hiperparameter bawaan agar sebanding dengan RF
pada T4.1. Menyetel salah satu saja akan membuat perbandingannya tidak adil. Ini
sekaligus berarti angka GBM di sini adalah **batas bawah** — bukan hasil terbaik
yang mungkin.

**"Bagaimana Anda tahu ini bukan pemilihan model berdasarkan hasil hilirnya?"**
Karena kriteria pemilihan (akurasi kelas kondisi, bukan RMSE) ditetapkan **sebelum**
perbandingannya dijalankan, dan tercatat pada prapendaftaran
`docs/PRAPENDAFTARAN_T6_GBM_RETRIEVAL.md`, yang juga mendugakan besaran
perubahannya di muka. Dugaan D2 memperkirakan `pc_rag` divergen 0,433 ± 0,02;
hasilnya 0,435. Dugaan D3 memperkirakan kenaikan +0,01…+0,06; hasilnya +0,042.

---

## 6. Ringkasan satu paragraf

Gradient Boosting dipilih bukan karena regresinya lebih akurat — pada sumbu itu ia
setara dengan Random Forest dan sedikit di bawah LSTM. Ia dipilih karena kriteria
pemilihan dalam penelitian ini adalah **akurasi kelas kondisi**, sebab kelas itulah
yang menyusun kueri *retrieval*, dan pada sumbu itu GBM unggul +6,4 poin pada kasus
divergen serta +22 poin pada sensitivitas hipoglikemia lewat pengklasifikasi
sadar-biaya, dengan biaya komputasi 31× lebih murah dan artefak 230× lebih kecil
sehingga memenuhi batasan desain tanpa GPU. Kelemahannya diketahui dan tidak
disangkal: pada horizon 60 menit GBM melewatkan 24 kasus hipoglikemia berat lebih
banyak daripada RF, dan pengklasifikasinya menukar ketepatan demi kepekaan —
pertukaran yang disengaja karena negatif palsu pada hipoglikemia jauh lebih
berbahaya daripada positif palsu.
