# Konteks untuk sesi lanjutan — 16 Agustus 2026

Dokumen ini adalah serah-terima. Dibuat saat sesi sebelumnya berhenti tepat di
tengah penyelidikan *chunking*. Bagian 5 memuat temuan yang **belum ditindaklanjuti**
dan merupakan titik lanjut yang sebenarnya.

Cara pakai: buka chat baru, minta Claude membaca berkas ini lebih dahulu, lalu
lanjutkan dari Bagian 6.

---

## 1. Aturan kerja yang berlaku (jangan dilanggar)

| Aturan | Rinci |
|---|---|
| **Git lokal saja** | Commit boleh. `push`, `remote add`, atau kirim ke repo jarak jauh **dilarang**. |
| **Controlled Folder Access aktif** | Folder `Documents` diproteksi Windows. Bila operasi berkas diblokir: **berhenti setelah satu diagnosis**, minta pengguna melakukan manual. Jangan iterasi mencoba jalur lain. Tercatat pernah memblokir penulisan ke `results/`; gunakan Python (`shutil`) bukan `cp`. |
| **Temuan ≠ izin memperbaiki** | Bila menemukan sesuatu yang tampak layak diperbaiki di luar tugas yang diminta: **laporkan dan berhenti**, jangan kerjakan. |
| **Restart Streamlit setelah edit `src/`** | Streamlit hanya memuat ulang skrip halaman, bukan modul `src/`. Pernah menimbulkan `KeyError` yang menyesatkan. |
| **Bahasa** | Naskah dan komentar kode berbahasa Indonesia. Istilah teknis boleh tetap Inggris (*chunking*, *retrieval*, *recall*). |

### Standar penulisan dari pembimbing — 5W+1H

Setiap kali menulis atau mengubah sesuatu, harus terjawab:

- **What** — apa yang dibuat/diterapkan/disesuaikan
- **Who** — pada komponen atau bagian apa
- **Where** — di fungsi/berkas/tempat mana persisnya
- **When** — kapan dipanggil atau digunakan
- **Why** — apa dasarnya (**wajib bersandar paper**, bukan intuisi)
- **How** — bagaimana penerapannya

Berlaku juga untuk pesan commit dan komentar kode.

---

## 2. Identitas penelitian

Tugas Akhir STI ITB. Sistem pendukung keputusan klinis **RAG terkondisi-prediksi**
untuk **Diabetes Melitus Tipe 1**.

Inti kebaruan: kueri *retrieval* disusun dari **kondisi glukosa yang DIPREDIKSI**,
bukan kondisi saat ini. Pembanding utama Kresevic dkk. (2024) mengoptimalkan sisi
**korpus**; penelitian ini mengubah sisi **kueri** — komplementer, bukan bersaing.

Konsekuensi penting: prediktor adalah **komponen, bukan kontribusi**. Kriteria
pemilihan model karena itu **akurasi kelas kondisi** (hipo/normal/hiper), bukan RMSE.

**Istilah yang SUDAH DICABUT** dan tidak boleh muncul di naskah aktif:
*digital twin*, *what-if*, *surrogate*, `T2DM`/"tipe 2" non-kontrastif,
*Prediction-Conditioned* (bentuk Inggrisnya), FKTP.

---

## 3. Yang selesai di sesi ini

### 3a. Migrasi LaTeX Bab I–III

Lokasi: `docs/laporan_TA/TA-STI-template-1.0/`

Bab I, II, III sudah ditulis ulang penuh dalam LaTeX. Tabel Bab II/III dibuat baru
(`tabel_II1_perbandinganmodel`, `tabel_II2_terkait`, `tabel_II3_potongan`,
`tabel_II4_metrik`, `tabel_III1_KF`, `tabel_III2_KNF`, `tabel_III3_pemetaan`,
`tabel_III4_korpus`, `tabel_III5_matriks`, `tabel_III6_alasan`).

Status verifikasi terakhir:

```
sitasi          : 49/49 terselesaikan
rujukan silang  : 48/48 terselesaikan
\input          : 10/10 ada
gambar Bab I-III:  9/9  ada
istilah tercabut:  0     pada seluruh berkas terpakai
```

Catatan: `multirow` TIDAK dimuat di `TA.tex` — jangan pakai.

### 3b. Gambar

Sembilan gambar Bab I–III lengkap. Dua dibuat/dipindahkan di sesi ini:

- **`Gambar_II5_IOBCOB.png`** — dihasilkan `scripts/buat_gambar_iob_cob.py`.
  Memakai kolom **`bolus_dose`** (1,94% non-nol), **BUKAN `insulin`** (99,48% non-nol
  karena memuat basal). Salah kolom akan membuat gambarnya membantah teksnya sendiri.
- **`Gambar_III3_SparsityKanal.png`** — dari `results/eda/04_event_sparsity.png`.
  Dipasang di Bab III §III.3.2.

### 3c. Daftar pustaka

`daftar-pustaka.bib`: 41 → 72 entri. Perbaikan yang pernah salah dan sudah dikoreksi:
IDF tahun 2024→2025; `rad2024` penulis Farida→**Fatemeh**, Xinyuan→**Xinyi**.
Koleksi PDF rujukan kini bernomor 01–43 tanpa lompatan.

### 3d. Arsip artefak lama

Dibuat `docs/laporan_TA/TA-STI-template-1.0/arsip/` berisi 19 berkas + `README.md`
(`bab/` 3, `tables/` 12, `images/` 4). Penentuan "tidak dipakai" lewat penelusuran
**transitif `\input`/`\includegraphics` dari `TA.tex`**, bukan pola nama.

Sengaja **tidak** diarsipkan: berkas contoh bawaan template (`tabel1`, `longtable1`,
`gambar1`, `binary-search-*`), berkas struktural yang sengaja dikomentari
(`7a Daftar Lampiran.tex` di `TA.tex:369`, `Lampiran-B.tex`), dan berkas **baru yang
belum dipakai** (`Gambar_IV1_AlurRinciSistem.png`, `listings/conditioned-query.tex`,
`listings/engineer-features.tex`).

Sembilan berkas terlacak git dipindah dengan `git mv` (sudah ter-*stage*, **belum
di-commit**); sepuluh sisanya tak terlacak.

Lima berkas `_ARSIP` di `arsip/tables/` **byte-identik** dengan pasangannya —
duplikat mubazir, aman dihapus, menunggu keputusan pengguna.

### 3e. Model prediksi: RF → GBM

`config.yaml` kini `model.name: "GradientBoosting"`. Hiperparameter **sengaja
default** agar sebanding dengan T4.1.

`src/models/gbm_model.py` (baru) melatih model titik + dua model kuantil.
σ diperoleh dari sebaran kuantil, bukan varian pohon:

```
σ = (q0.975 − q0.025) / 3.92        # GAUSS_95_WIDTH
```

Alasannya `HistGradientBoostingRegressor` tidak punya `estimators_`, sehingga
metode varian-pohon milik RF tidak tersedia. Bundle menyimpan `std_method`,
`std_models`, `std_quantiles`, `model_family` supaya aplikasi bisa memilih jalur
yang benar.

Angka GBM terverifikasi: h6 RMSE 20,81 / MAE 14,09 / Clarke A+B 95,19%;
h12 RMSE 32,24 / MAE 23,46 / 88,07%.

Uji: 137 → 145 (`tests/test_gbm_model.py` 4, `tests/test_citations_snippet.py` 4).

---

## 4. Utang yang belum dibayar (perlu diputuskan/dikerjakan)

### 4a. Menunggu keputusan pembimbing — RF lawan GBM

Tiga tempat ditandai `[tertahan]` di naskah: kalimat "model utama" §II.4.3,
baris "Kesesuaian" Tabel II.1, dan baris "Model prediksi" Tabel III.5/III.6.

**Jangan diisi sebelum ada keputusan.** GBM **tidak** unggul di segala dimensi,
sehingga klaim "GBM > RF di tiap dimensi" di `docs/HANDOFF.md:26` memang terlalu
kuat dan perlu dicabut.

> **DIVERIFIKASI ULANG 16 Agustus 2026 (sore): angka di atas BENAR.**
> Sempat diduga salah baca dari tabel RF-lawan-LSTM di `docs/HANDOFF.md:178`.
> Dugaan itu KELIRU. Pemeriksaan langsung ke
> `results/eval_prediksi/gradient_boosting_h12.json` menegaskan:
> - `rerata_lintas_fold.RF.sensitivitas_persen` = **4,355%**
> - `RF_vs_GBM__sensitivitas_persen.selisih_rerata` = **1,936**, sehingga
>   GBM = 4,355 − 1,936 = **2,42%**
> - `basis_kejadian.hipo_berat_terlewat` = RF **933**, GBM **957**
>
> **Status statistiknya bercabang, dan ini yang harus dipahami sebelum bimbingan:**
>
> | Kriteria | Nilai |
> |---|---|
> | `wilcoxon_p` | **0,031** (bermakna pada 0,05) |
> | `melampaui_sd.selisih_berpasangan` | **true** |
> | `melampaui_sd.maks_per_model` | false |
> | `melampaui_sd.gabungan_definisi_T2.1` | false |
> | `sd_sepakat` | **false** — ketiga definisi SD tidak sepakat |
> | `layak_dinarasikan_konservatif` | **false** |
>
> Jadi kelemahan GBM pada hipoglikemia h12 **nyata dan tidak boleh disangkal**,
> tetapi menurut aturan pelaporan proyek sendiri (KEPUTUSAN #8) ia **tidak layak
> dinarasikan sebagai temuan** karena ketiga definisi SD tidak sepakat. Cara
> menyebutnya yang jujur: *"GBM kehilangan 24 kasus hipoglikemia berat lebih banyak
> daripada RF pada horizon 60 menit (957 lawan 933); Wilcoxon p=0,031, namun uji
> konsistensi proyek ini sendiri menyatakan selisihnya belum layak dinarasikan."*
>
> Kelemahan GBM lain yang juga nyata, lihat `docs/ARGUMEN_GBM.md`: akurasi
> keseluruhan pengklasifikasi turun 88,2% → 86,1%, PPV hipoglikemia turun
> 40,9% → 31,9%, sensitivitas kelas `normal` turun 91,2% → 85,6%.

Karena keputusan ini menggantung, `results/eval_prediksi/*_RF_arsip.json`
(4 berkas) adalah **data pembanding hidup**, bukan sisa — jangan diarsipkan.

### 4b. Eksperimen yang belum dijalankan ulang dengan GBM

> **KOREKSI 16 Agustus 2026 (sore).** Daftar di bawah ini **SALAH** ketika ditulis:
> dua dari tiga butirnya sebenarnya SUDAH dijalankan. Kekeliruan ini sempat
> menyesatkan sesi lanjutan sampai diperiksa langsung ke `results/`.
> **Pelajaran: periksa `results/` lebih dulu, jangan percaya daftar ini.**

| Butir | Status sebenarnya |
|---|---|
| Crossfold *retrieval* + *realcases* | **SUDAH** — `results/retrieval_realcases_kb12_gbm/` (14 Agu). D1 lolos: ketiga kontrol tereproduksi persis, termasuk `n_divergen` per fold. D2, D3, D4 juga lolos. |
| Skenario *deployment* SMBG | **SUDAH** — `results/eval_prediksi/smbg_deployment.json` sudah berkeluarga `HistGradientBoostingRegressor`. |
| T3.1–T3.3 | Belum diperiksa ulang. |

Hasil crossfold GBM (kasus divergen, prapendaftaran
`docs/PRAPENDAFTARAN_T6_GBM_RETRIEVAL.md`):
`standard` 0,382 · `pc_rag` 0,435 · `pc_rag_classifier` 0,506 · `oracle` 0,829;
selisih `pc_rag − standard` **+0,0527**, positif **6/6 fold**.

Sesudah chunking diubah (T7), crossfold dijalankan lagi pada konfigurasi produksi
baru — `results/retrieval_realcases_kb12_gbm_kalimat/`: selisih **+0,0832**, tetap
positif **6/6**. **Kontribusi tereplikasi di dua konfigurasi.**

### 4c. Cacat terverifikasi yang belum diperbaiki

| Cacat | Lokasi | Catatan |
|---|---|---|
| **Kalibrasi conformal melatih model 8-pasien BARU** (`pids[:-4]`) sementara produksi melatih 10 (`patient_ids[-2:]`) | `scripts/conformal_calibration.py` | `q` mengkalibrasi model yang **berbeda** dari yang dipakai. Tidak terdokumentasi di mana pun. Paling serius di daftar ini. |
| Sitasi horizon 30/60 menit **tidak ada dasarnya** | Bab I Batasan 3, §II.4.3, §III.3.3 | Sapuan seluruh halaman `docs/OhioT1DM-dataset-paper.pdf` → **nol kecocokan**. Perlu pengganti (kandidat: Ghimire 2024, Woldaregay 2019, prosiding BGLP). |
| `_metadata_tags()` tidak pernah sampai ke retriever | `src/rag/` | Kode mati; nilai `dm_tipe2` di-*hardcode*. |
| Parser membuang `duration` dari `<exercise>` | `src/data/ohio_parser.py:121` | |
| Docstring basi | `scripts/build_eval_kb.py:13` ("hal. 18 dan 23" vs `EVAL_PAGES` [23,44,45,48]); `scripts/run_ragas.py:5` ("2 halaman") | |
| Bab V & VI masih memuat istilah tercabut | `Gambar_V5_UI_WhatIf.png` **masih dipakai** Bab V | Belum dimigrasi. |
| `.env` memuat 3 kunci API asli di komentar | `.env` | Perlu dicabut dan dihapus. |
| README memuat klaim basi | `README.md` | `digital twin`, `surrogate`, "tipe 2", latensi 11,4 s. |

---

## 5. TITIK LANJUT — masalah *chunking* (belum selesai)

Permintaan pengguna: perbaiki *chunking* agar teks tidak terpotong; pastikan
potongan berhenti di batas kalimat, bukan di tengah; pastikan kalimat sebelum dan
sesudah kutipan masih berkaitan; **gunakan metode yang terverifikasi paper**.

### Temuan kunci: ini DUA masalah berbeda, jangan disatukan

**Masalah A — pemotongan senyap di sisi vektor (LEBIH SERIUS).**

`all-MiniLM-L6-v2` punya `max_seq_length` = **256 token**. Token ke-257 dan
seterusnya **dibuang diam-diam** — tanpa peringatan, tanpa error. `manifest.csv`
dan jumlah chunk tetap tampak wajar.

Sudah diukur — `results/eval_rag/distribusi_token.json` (T5.1, 10 Agu 2026):

| chunk_size | n_chunk | rerata token | p95 | chunk melewati 256 | token hilang |
|---|---|---|---|---|---|
| **900 (produksi)** | 2186 | 214,5 | 337 | **766 (35,04%)** | **8,23%** |
| 500 | 3745 | 125,8 | 191 | 4 (0,11%) | 0,01% |

Terburuk per dokumen pada produksi: `KB-04_PERKENI-2023_Tatalaksana-Hiperglikemia-RS.pdf`
**67,19%** chunk terpotong; `KB-03_PERKENI-2021_Terapi-Insulin.pdf` 48,84%;
`KB-01_IDAI-2015_Konsensus-DMT1.pdf` 44,87%.

Artinya **8,23% korpus tidak pernah masuk vektor** dan tidak akan pernah terambil
betapa pun relevannya.

> **Ketegangan yang harus diselesaikan, jangan diabaikan.**
> `config.yaml:135` menyatakan chunk_size **SENGAJA** tetap 900: menggabungkan
> lambda 0,0 dengan chunk 500 justru **menurunkan** MRR ke 0,464 dan ditolak uji
> Wilcoxon terhadap chunk 500 sendirian (p=0,029) —
> `results/baseline_ablation_fullkb_kb12_sym/kombinasi_b4b5.json`.
> Sapuan B4: MRR 300→0,306 · 500→**0,541** · 900→0,425 · 1400→0,313 · 2000→0,302.
> Hit@1: 500→40,0% · 900→26,2%.
>
> Jadi menurunkan ke 500 **tidak otomatis benar**. Kesimpulan T5.1 sendiri
> mengusulkan jalan ketiga: *"Pertahankan chunk_size 900 tetapi hilangkan
> pemotongannya — indeks ulang"* (`scripts/eval_distribusi_token.py:265`).
> Opsi ini belum pernah diuji.

**Masalah B — pemotongan tampilan di UI (yang dilihat pengguna).**

`src/rag/citations.py`: `SNIPPET_CHARS_DEFAULT = 700`, fungsi `_truncate()`
(baris 73–89) memotong pada **batas kata**, bukan batas kalimat. Inilah sebabnya
kutipan di Streamlit berhenti di tengah kalimat.

Sudah tersedia: `teks_lengkap`, `n_char`, `snippet_terpotong` di keluarannya, dan
UI `app/streamlit_app.py` sudah punya sakelar `utuh` untuk menampilkan teks penuh.

### Penyebab potongan bukan "titik ke titik"

`src/rag/knowledge_base.py:193` — `RecursiveCharacterTextSplitter` dipanggil dengan:

```python
separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
```

**Tidak ada pemisah kalimat.** Setelah `"\n"` gagal, ia langsung jatuh ke spasi `" "`,
yang berarti memotong di sembarang kata. Menambahkan `". "` (dan varian `.\n`)
sebelum `" "` adalah perubahan paling kecil yang menangani ini.

### Yang perlu diperiksa dan BELUM diperiksa

1. **Apakah jalur Gemini memakai chunking yang sama?** Pengguna menanyakan ini
   eksplisit. `config.yaml` menyediakan `embedding_provider` alternatif
   (`google`/`ollama`) dan `google_embedding_model: models/embedding-001`.
   `models/embedding-001` punya batas token **jauh lebih besar** dari 256 —
   bila korpus diindeks ulang dengan provider berbeda, besaran masalah A berubah
   total. **Belum diverifikasi provider mana yang benar-benar dipakai produksi.**
2. `manual_kb` memakai chunk_size 350/overlap 40 terpisah — apakah ikut terdampak?
3. Apakah `chunk_overlap: 120` cukup menjamin kalimat di batas chunk tidak
   kehilangan konteks tetangganya (pertanyaan "kalimat sebelum-sesudah" dari pengguna).

### Kandidat metode bersandar paper (perlu diverifikasi, JANGAN dikutip sebelum dibaca)

Pengguna mensyaratkan dasar paper. Arah yang relevan:

- ***Sentence-aware / sentence-window chunking*** — potong di batas kalimat,
  simpan tetangga sebagai konteks. Menjawab langsung dua keluhan pengguna.
- ***Semantic chunking*** — batas ditentukan kemiripan embedding antarkalimat.
- ***Late chunking*** — embed dokumen panjang dulu, baru potong representasinya.
- **Gao dkk.** sudah dirujuk di `scripts/eval_chunk_size.py:3` sebagai dasar bahwa
  chunk_size tidak boleh ditetapkan tanpa pengujian — sudah ada di koleksi.

Berkas relevan: `src/rag/knowledge_base.py` (`chunk_documents`, baris 175+),
`scripts/reingest_kb.py`, `scripts/eval_chunk_size.py`, `scripts/eval_distribusi_token.py`,
`src/rag/citations.py`.

---

## 6. Urutan yang disarankan untuk sesi baru

1. **Verifikasi provider embedding produksi** (`sentence-transformers` atau `google`).
   Segala keputusan chunking bergantung pada ini, dan jawabannya mengubah besaran
   masalah A.
2. Perbaiki **Masalah B** lebih dulu — potongan UI ke batas kalimat. Murah,
   terisolasi di `citations.py`, tidak menuntut indeks ulang, dan langsung
   menjawab apa yang pengguna lihat di layar.
3. Baru tangani **Masalah A**. Buat prapendaftaran seperti pola
   `docs/PRAPENDAFTARAN_*.md` **sebelum** menjalankan, karena ini mengubah angka
   retrieval yang sudah dilaporkan. Uji ketiga opsi, jangan pilih satu:
   (i) 900 + pemisah kalimat, (ii) 500, (iii) 900 tanpa pemotongan (usulan T5.1).
4. Baru menulis Bab IV.

**Jangan** menulis Bab IV sebelum langkah 1–3 selesai: Bab IV memerikan rancangan,
dan rancangan chunking-nya sedang berubah.

---

## 7. Peta berkas seperlunya

```
config.yaml                          blok rag: baris 100-145
src/rag/knowledge_base.py            chunk_documents() baris 175+
src/rag/citations.py                 SNIPPET_CHARS_DEFAULT 37, _truncate() 73-89
src/models/gbm_model.py              prediktor produksi
app/streamlit_app.py                 KELUARGA_BUNDLE, predict_uncertainty, blok snippet
scripts/eval_distribusi_token.py     T5.1 -- pengukur pemotongan token
scripts/eval_chunk_size.py           sapuan B4
scripts/conformal_calibration.py     CACAT 8 vs 10 pasien ada di sini
results/eval_rag/distribusi_token.json
results/baseline_ablation_fullkb_kb12_sym/kombinasi_b4b5.json
docs/ALUR_SISTEM_FINAL.md            spesifikasi beku: invarian I1-I8, batas B1-B8
docs/METHODOLOGY.md                  §7.0 kebaruan, §8 rujukan tiga lapis
docs/PRAPENDAFTARAN_T6_GBM_RETRIEVAL.md
docs/laporan_TA/TA-STI-template-1.0/ naskah LaTeX
```

`docs/ALUR_SISTEM_FINAL.md` adalah **spesifikasi beku**. Perubahan yang menyentuh
invarian I1–I8 harus dikonsultasikan lebih dulu, bukan dikerjakan.
