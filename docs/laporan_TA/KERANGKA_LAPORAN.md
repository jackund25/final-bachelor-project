# Kerangka dan Handoff Laporan TA

Ditulis ulang **25 Agustus 2026, sesi sore**. Menggantikan versi 25 Agustus pagi.

Dokumen ini **handoff**: keadaan naskah, apa yang sudah selesai, apa yang masih
terbuka, dan keputusan apa yang menunggu penulis. Baca berurutan; bagian 1 dan 2
menentukan apakah bagian selanjutnya boleh dikerjakan.

**Acuan aturan adalah `Aturan_Penulisan_Laporan_TA.md`** (52 aturan + 12 Golden
Rules), bukan dokumen ini. Bila keduanya berbeda, aturan itu yang menang. Acuan
finalisasi visual adalah `Checklist_Perbaikan_Formatting_TA_Final.md`.

---

## 0. Keadaan naskah per 25 Agustus 2026 (sore)

Kompilasi terakhir bersih:

```
latexmk -lualatex   ->  exit 0
0 galat · 194 halaman · 0 rujukan tak terdefinisi
56 entri bib · 35 berkas tabel · 7 berkas listing · 33 berkas gambar
```

Peringatan `LaTeX Warning: There were undefined references` **tetap muncul dan itu
wajar**. Ia berasal dari `biblatex-chicago` (`Empty Ibidem citation`), bukan dari
rujukan yang hilang. Sudah diverifikasi ulang: nol.

**Belum ada commit.** Seluruh pekerjaan ada di *working tree* (68 berkas termodifikasi).

| Bab | section | subsection | gambar | tabel | listing | Catatan |
|---|:--:|:--:|:--:|:--:|:--:|---|
| I | 7 | 0 | 1 | 0 | 0 | selesai |
| II | 8 | 23 | 5 | 6 | 0 | selesai |
| III | 4 | 13 | 2 | 6 | 0 | selesai |
| IV | 8 | 27 | 6 | 5 | 0 | selesai; **belum ada rancangan basis data** |
| V | 10 | **5** | **8** | 3 | **4** | ✅ gambar UI dan potongan kode masuk sesi ini |
| VI | 14 | 9 | 3 | 13 | 0 | **lihat bagian 1 — angkanya bermasalah** |
| VII | 3 | **0** | 0 | 0 | 0 | §7.2 perlu jadi tabel |

Lampiran A memuat 2 listing tambahan (rekayasa fitur, kueri terkondisi).

---

## 1. ⛔ MASALAH TERBESAR: Bab VI tidak dapat direproduksi

**Belum berubah sejak sesi pagi. Ini yang harus diputuskan lebih dulu.**

### Apa yang terjadi

API penyiapan data diganti pada 23–24 Agustus, **sesudah** seluruh angka Bab VI
dihasilkan. Metode `DataPreprocessor.create_sequences` **dicabut** dan diganti
`create_time_horizon_sequences`. Lihat `src/models/persiapan_data.py:20`.

Semantiknya berubah pada tiga hal:

1. jendela dibentuk dari **waktu nyata (menit)**, bukan jumlah langkah;
2. tiap modalitas punya profil sendiri (`config.model.source_profiles`);
3. pembagian latih/uji mengikuti **pembagian resmi OhioT1DM** yang bersifat
   temporal dalam-pasien — bukan lintas-pasien.

### Skrip yang terkena

**14 skrip** memakai API yang dicabut. Tujuh menghasilkan angka Bab VI:
`crossval_rf_vs_lstm`, `conformal_calibration`, `eval_hipoglikemia`,
`improve_hypo_detection`, `eval_feature_importance`, `eval_retrieval_crossfold`
(**MRR 6-fold, p = 0,031 — bukti kebaruan**), `eval_retrieval_realcases`.

Selain itu **9 skrip** masih membaca `data/raw/ohio_t1dm_merged.csv` yang sudah
disuperseded. Diverifikasi 25 Agustus: subset CGM pada `ohio_t1dm.csv` **identik**
dengan isi berkas lama, jadi mengganti masukan bersifat **netral terhadap hasil**;
yang tidak netral adalah perubahan API.

### Risikonya tidak merata

- **Angka CGM — risiko rendah.** Pada sampling seragam 5 menit, jendela berbasis
  langkah dan berbasis waktu praktis setara.
- **Angka SMBG — risiko tinggi.** Menyentuh RMSE 28,56 dan Clarke 90,68% yang
  muncul di **Abstrak** dan **Bab VII**.
- **Apa pun yang memakai pembagian bawaan — risiko tinggi.**

### 🔴 KEPUTUSAN YANG MENUNGGU

1. Migrasikan semua 14 skrip, jalankan ulang seluruh Bab VI.
2. **Migrasikan yang menopang klaim utama saja** *(disarankan)* —
   `eval_retrieval_crossfold`, `eval_retrieval_realcases`, `eval_smbg_deployment`.
3. Tidak menjalankan ulang, tetapi menyatakan terbuka di Bab VI bahwa angka itu
   dihasilkan pada API penyiapan data terdahulu.

---

## 2. ✅ Yang dikerjakan sesi 25 Agustus sore

Seluruhnya sudah lulus kompilasi. Rujukan rencana:
`~/.claude/plans/baca-aturan-penulisan-laporan-ta-kerangk-goofy-cookie.md`.

### 2.1 Perbaikan layout P0 (Checklist Tahap 1)

**Penomoran di Checklist sudah usang.** Sasaran dipetakan menurut isi dan label,
bukan nomor. Peta yang berlaku sekarang:

| Sebutan di Checklist | Berkas / label sebenarnya |
|---|---|
| "Gambar IV.2/IV.3 alur rinci" | `fig:alur-rinci`, terbit sebagai **Gambar IV.3** |
| "Gambar IV.5 sequence" | `fig:sequence-pcrag`, terbit sebagai **Gambar IV.5** |
| "Tabel V.2 collision" | `tables/tabel_V2_dependensi.tex` (`tab:dependensi`) |
| "Tabel VI.12 overflow" | `tables/tabel_VI9_benchmark.tex` (`tab:benchmark`) |

Yang dikerjakan:

- **`tabel_V2_dependensi`** — kolom `3,4/3,3/5,3` → `2,9/4,0/5,1` cm, ketiganya
  `>{\raggedright\arraybackslash}p{}`, `\allowbreak` disisipkan pada
  `@supabase/supabase-js`, `langchain-google-genai`, `sentence-transformers`,
  `python-dotenv`. **Nol `Overfull \hbox`** — tabrakan teks hilang.
- **`tabel_VI9_benchmark`** — `tabular{llr}` (kolom `l` tidak pernah membungkus,
  itu sebab overflow-nya) → `tabularx` selebar `\textwidth`. Caption dipangkas jadi
  "Hasil pengukuran kelayakan penerapan sistem."; spesifikasi laptop dipindah ke
  paragraf sebelum tabel. **Nol `Overfull \hbox`**.
- **Gambar IV.3 alur rinci** — caption 9 baris dipangkas jadi "Alur rinci sistem
  luring dan daring."; penjelasan empat kontribusi metodologis dan lima kotak
  bertanda dipindah menjadi dua paragraf sesudah gambar. Float jadi `[p]`
  (halaman khusus) dengan `width=\textwidth,height=0.95\textheight`.
  **Landscape sengaja TIDAK dipakai**: gambarnya 4443×7806 (rasio tinggi/lebar
  1,76), sehingga landscape justru memperkecilnya.
- **Gambar IV.5 sekuens** — 6051×3768 (rasio 0,62, lebar), jadi landscape memang
  menolong. Dipindah ke `sidewaysfigure`, ukuran linear naik ~1,6×.
- **Gambar VI.1–VI.3** — sumbernya sudah benar sejak awal (paragraf substantif ada
  di antara ketiganya); penumpukan di PDF murni **artefak penempatan float**.
  Diperbaiki dengan `[!ht]` + dua `\clearpage`. Hasil: hal. 103 / 104 / 105,
  masing-masing dengan paragrafnya. Caption ketiganya dipangkas; angka 94,70% dan
  87,32% sudah ada di paragraf sekitarnya sehingga tidak ada informasi hilang.
  Ditambahkan satu paragraf baru yang **membaca** Gambar VI.3 (Aturan 39).

### 2.2 Delapan tangkapan layar Bab V (Checklist Tahap 5)

Bab V sebelumnya **nol gambar**. Dipilih 8 dari 19 tangkapan yang tersedia; prinsip
pemilihannya adalah **klaim mana di §5.7 yang menuntut bukti visual**, bukan gambar
mana yang bagus.

| Gbr | Berkas di `images/` | Sumber di `images/SS Sistem/` | Membuktikan |
|---|---|---|---|
| V.1 | `Gambar_V1_PencatatanMandiri.jpg` | `Input logbook.jpg` | KF-01 |
| V.2 | `Gambar_V2_PrediksiGlukosa.jpg` | `Prediksi glukosa.jpg` | KF-03 |
| V.3 | `Gambar_V3_RekomendasiSBAR.jpg` | `Clinical assessment.jpg` | urutan SBAR |
| V.4 | `Gambar_V4_BatasPengetahuan.jpg` | `lanjutan clinical assessment.jpg` | batas pengetahuan + *evidence status* |
| V.5 | `Gambar_V5_PanelRujukanSumber.jpg` | `Keterlacakan sumber.jpg` | KF-05, KNF-08 |
| V.6 | `Gambar_V6_PenurunanAnggun.jpg` | `tampilan awal untuk pasien yang datanya kurang memadai.jpg` | KNF-04 |
| V.7 | `Gambar_V8_TampilanDesktop.png` | tangkapan desktop di `images/` | responsivitas PWA |
| V.8 | `Gambar_V7_RiwayatPenilaian.jpg` | `tampilan awal page history.jpg` | riwayat penilaian |

> ⚠️ **Nama berkas V.7 dan V.8 tertukar terhadap nomor gambarnya.** Desktop
> disisipkan sebelum paragraf "Riwayat penilaian", sehingga LaTeX menomorinya V.7
> padahal berkasnya bernama `Gambar_V8_...`. Tidak terlihat pembaca, tetapi
> menyesatkan saat menyunting. **Belum diperbaiki.**

Pasangan responsivitas adalah **V.6 (mobile) → paragraf → V.7 (desktop)**, keduanya
menampilkan P008 · `INSUFFICIENT_HISTORY` · 142,0 mg/dL, sehingga syarat konsistensi
identitas pasien dan state UI (Checklist §9) terpenuhi tanpa tangkapan ulang.

Sebelas tangkapan lain **sengaja tidak dipakai** (panel arti variabel ×3, logbook
awal/preview/history, sebelum-prediksi, finger stick, line chart, lanjutan pasien
kurang memadai, pemilih pasien, dashboard mobile P001). Alasannya tercatat di berkas
rencana. Berkas aslinya tetap ada di `images/SS Sistem/` sebagai arsip.

§5.7 dipecah menjadi **5 `\subsection`** (V.7.1–V.7.5), menutup butir bimbingan GB.1
yang sebelumnya ⚠️ untuk Bab V. Blok komentar penanda posisi gambar sudah dihapus.

### 2.3 Empat potongan kode di Bab V

Aturan 24 mengizinkan, Aturan 25 mewajibkan klaim implementasi dapat ditunjukkan di
source code. Infrastrukturnya sudah ada sejak awal (`listings` + `pythonstyle` +
halaman Daftar Listing).

| Listing | Berkas kutipan | Sumber | Menutup klaim |
|---|---|---|---|
| V.1 | `listings/bm25-index.tex` | `src/rag/retriever.py:260,278–294` | indeks BM25 dibangun dari korpus yang **sama** dengan jalur vektor |
| V.2 | `listings/bm25-degradasi.tex` | `src/rag/retriever.py:295–309` | penurunan mode dinyatakan eksplisit (KNF-04) |
| V.3 | `listings/resolusi-sitasi.tex` | `src/rag/citations.py:24–55` | nomor halaman diresolusi dari metadata, tidak ditebak (KF-05) |
| V.4 | `listings/penjagaan-riwayat.tex` | `backend/services/prediction_service.py:398–424` | penolakan memprediksi ditegakkan di lapis layanan |

Listing V.4 berpasangan langsung dengan Gambar V.6: pesan
`Insufficient historical {source} observations: {n}/{required}` pada kode adalah
teks yang terbaca pada tangkapan layar.

**Kutipan dihasilkan skrip, bukan salin-tempel.** Skripnya ada di scratchpad sesi
(`buat_listing.py`); bila kode berubah, kutipan harus dihasilkan ulang. Spesifikasi
rentang barisnya tercatat di dalam skrip itu. **Salin skrip ini ke `scripts/` bila
ingin dipertahankan** — scratchpad sesi tidak permanen.

### 2.4 Kontradiksi angka Bab VI yang diperbaiki

**Di luar rencana semula, tetapi wajib.** Dua paragraf sesudah tabel kelayakan masih
memuat angka yang sudah dicabut dan **berlawanan dengan tabelnya sendiri**:

| Lokasi | Sebelum (usang) | Sesudah (sesuai tabel) |
|---|---|---|
| `Bab VI:330` | BM25 **menaikkan** beban lokal 21,2 → 284,3 ms | BM25 **menurunkan**; 11,1 ms lawan 284,3 ms jalur vektor |
| `Bab VI:332` | LLM 11,3 s, total 11,4 s, 98% | LLM 2,5 s, total 2,6 s, ~97% |

Diperiksa: tidak ada lagi kemunculan `11,3` / `11,4` / `21,2` / `284,3` sebagai
klaim aktif di seluruh naskah.

### 2.5 Satu perbaikan kode frontend

`frontend/src/app/history/page.tsx:151–153` menampilkan float tanpa pembulatan
(`119.304023661237 mg/dL`). Ditambahkan `.toFixed(1)` sesuai konvensi yang sudah
dipakai seluruh berkas lain. `tsc --noEmit` lolos. **Tangkapan layar V.8 masih
memakai versi lama, jadi perlu diambil ulang.**

---

## 3. Yang sudah diverifikasi aman

Jangan ragukan ulang yang berikut; sudah ditelusuri ke artefaknya.

**MRR utama sahih.** `results/retrieval_realcases_kb12/crossfold.json` mencatat
`chunk_size: 900, chunk_overlap: 120` — konfigurasi korpus produksi. MRR
0,168 / 0,065 / 0,833 dan *p* = 0,031 diukur pada korpus yang benar.

**Korpus = 2.233 potongan**, dihitung dari `models/chroma_db/chroma.sqlite3`. Bukan 4.038.

**Reranking tidak aktif.** `config.yaml:374` → `reranker.enabled: false`.

**Angka +0,158 MRR: nol kemunculan.** Larangan K1 dipatuhi.

**Korpus 12 dokumen, 543 halaman**, sesuai `data/knowledge_base/manifest.csv`.
IDAI-2015, tiga PERKENI, ADA–EASD-2021, ATTD-2019, enam bab ISPAD-2022.

### Benchmark KNF-01 — sudah dijalankan ulang (18 → 25 Agustus)

Angka lama dicabut: diukur pada korpus 4.038 potongan **dan** jalur vektor (MMR),
dua-duanya bukan konfigurasi final.

| | Lama (4.038 · MMR) | **Baru (2.233 · BM25)** |
|---|---|---|
| Muat artefak | 16.545,9 ms | **1.600,1 ms** |
| Rekayasa fitur | 23,0 ms | 45,5 ms |
| Prediksi | 7,5 ms | 8,5 ms |
| Penelusuran | 284,3 ms | **11,1 ms** |
| **Total lokal** | 314,8 ms | **65,1 ms** |
| Memori | 641,2 MB | **484,2 MB** |
| Satu rekomendasi utuh | ≈9,6 s | **≈2,6 s** |

**Satu kesimpulan lama terbalik:** BM25 **26× lebih cepat** daripada jalur vektor,
bukan lebih lambat. Sebabnya model *embedding* tidak pernah dimuat pada mode BM25.
Narasi Bab VI sudah diselaraskan (lihat 2.4).

### RAGAS lengan penelusuran — sudah dijalankan ulang (25 Agustus)

Konfigurasi produksi (2.233 potongan, 900/120), penilai `gemini-3.5-flash-lite`
— **sama** dengan jalan terdahulu.

| Lengan | *precision* lama → **baru** | *recall* lama → **baru** |
|---|---|---|
| Padat (vektor) | 0,167 → **0,025** | 0,200 → **0,100** |
| **Leksikal (BM25)** | 0,642 → **0,803** | 0,800 → **0,900** |
| Hibrida (RRF) | 0,259 → **0,542** | 0,550 → **0,700** |

Seluruh angka berubah, **kesimpulannya utuh**: BM25 > hibrida > vektor, dan
`aturan4_bm25_mengungguli_hibrida` tetap `True`. Dengan sepuluh kasus, angkanya
tidak boleh dibaca sebagai presisi tiga desimal; yang kokoh adalah **urutannya**.

---

## 4. 🔴 Yang masih terbuka — prioritas 1

### 4.1 Delapan tabel Bab VI keluar margin

Terukur langsung dari `TA.log` sesudah kompilasi terakhir. **Ini pekerjaan paling
mendesak berikutnya** dan seluruhnya murni formatting.

| Tabel | Overfull | Perkiraan |
|---|---:|---|
| `tabel_VI10_sintesis` | **124,6 pt** | ≈4,4 cm keluar margin |
| `tabel_VI7_realcases` | 77,0 pt | ≈2,7 cm |
| `tabel_VI13_chunking` | 61,1 pt | ≈2,1 cm |
| `tabel_VI1_prediksi` | 55,5 pt | ≈2,0 cm |
| `tabel_VI2_crossval` | 55,4 pt | ≈2,0 cm |
| `tabel_VI6_literatur` | 48,7 pt | ≈1,7 cm |
| `tabel_VI3_retrieval` | 27,1 pt | ≈1,0 cm |
| `tabel_VI11_crossfold` | 22,9 pt | ≈0,8 cm |

Resep yang **terbukti bekerja** pada `tabel_VI9_benchmark`: ganti kolom `l`/`c` yang
memuat kalimat menjadi `tabularx` dengan `>{\raggedright\arraybackslash}X`, dan
jangan memperkecil font. Cara mengulang pengukurannya:

```bash
grep -n "Overfull" TA.log
```

lalu telusuri ~25 baris ke atas untuk menemukan berkas `./tables/...` pemiliknya.

Sisa overfull kecil (`tabel_III5_matriks` 0,7 pt, `tabel_V1_parameter` 17,2 pt,
`tabel_VI12_lengan` 11,9 pt) menyusul sesudahnya.

### 4.2 Dua cacat isi pada gambar yang sudah tercetak

Keduanya **sudah masuk PDF** dan akan terlihat penguji:

1. **Gambar V.3** menampilkan `insulin aktif 45.50 unit, karbohidrat aktif
   646.3 gram` — nilai IOB/COB yang tidak masuk akal secara klinis. Indikasi
   akumulasi yang tidak meluruh; bila benar, ia menyentuh klaim rekayasa fitur
   Bab IV §4.4 dan Persamaan IOB/COB. **Periksa sebelum sidang.**
2. **Gambar V.5** menampilkan potongan PERKENI Hiperglikemia-RS yang membahas
   **DM tipe 2**, padahal kondisi terprediksinya AMAN. Dua risiko: penelusuran
   terlihat tidak relevan, dan istilah "DM2" muncul pada penelitian yang ruang
   lingkupnya DM Tipe 1. Pertimbangkan mengambil ulang pada kasus hipoglikemia
   atau hiperglikemia.

### 4.3 Gambar V.8 perlu diambil ulang

Masih memakai versi sebelum perbaikan pembulatan (lihat 2.5).

### 4.4 §7.2 jadi tabel

18 butir KF/KNF masih `description`. Sebagai tabel (Kebutuhan · Status · Bukti ·
Lokasi) ia memaksa kolom Bukti terisi, sekaligus memberi Bab VII artefak pertamanya.
Bab VII masih **nol `\subsection`, nol gambar, nol tabel**.

### 4.5 Rancangan basis data Bab IV

Aturan 14 mewajibkannya; Bab IV punya 6 gambar tetapi **nol rancangan basis data**.

| Artefak | Bab | Isi |
|---|---|---|
| ERD konseptual | **IV** | 7 entitas + relasi + kardinalitas, tanpa tipe data. **Tanpa `stress_events`** |
| Skema fisik | **V** | Tangkapan Supabase apa adanya, 8 tabel, plus satu kalimat mengapa `stress_events` masih ada tetapi tidak lagi ditulisi |

Pemisahan ini menyelesaikan soal `stress_events` **tanpa menyentuh basis data**.

---

## 5. 🟠 Prioritas 2 — klaim tanpa penopang, asetnya sudah ada

`results/eda/` punya 8 gambar menganggur. Tiga menutup klaim yang kini tak terbukti:

| Gambar | Menutup klaim | Lokasi |
|---|---|---|
| `02_diurnal_pattern.png` | *"Kadar glukosa mengikuti pola harian"* — diasersikan telanjang | Bab IV §4.4.3 |
| `08_cgm_vs_smbg.png` | dasar asumsi penyetaraan kanal | Bab III §3.3 |
| `07_per_patient_boxplot.png` | dasar validasi silang lintas-pasien | Bab VI §6.3 |

---

## 6. 🟡 Prioritas 3

- **Tabel RAGAS §6.7** — melaporkan 7 angka dalam prosa, tanpa tabel. Angka §6.7
  berasal dari `results/ragas/summary.json` (22 Ags), **bukan** dari
  `lengan_penelusuran.json` yang sudah dijalankan ulang. Statusnya belum diperiksa.
- **Tabel kalibrasi konformal** — §6.5 menyebut 95,5% / 91,2% / q′ = 3,3 tanpa tabel.
- **Istilah "prakiraan" vs "prediksi" bercampur** (Aturan 27). Perlu satu istilah resmi.
- **`tab:uji-statistik` tak pernah di-`\ref`** (Aturan 38).
- **Tabel *reference audit*** yang diminta Aturan 43 belum ada.
- **Nama berkas V.7/V.8 tertukar** (lihat 2.2).
- **Gambar IV.3 masih padat.** Sudah semaksimal mungkin secara tipografis; sisa
  pilihannya menyederhanakan diagram atau memindahkan detail ke lampiran —
  keduanya menuntut menggambar ulang.

---

## 7. 🙋 Butuh penulis, bukan agen

- **URL GitHub di Lampiran A.** `github.com/jackund25/final-bachelor-project`
  dicantumkan sebagai tempat kode lengkap. Repositori ini **lokal saja** dan belum
  ada commit. Pastikan repo publiknya ada, atau ubah pernyataannya.
- **Hapus manual `docs/laporan_TA/TA-STI-template-1.0/_ui_lama.bak`** — penghapusan
  oleh agen ditolak sistem.
- **Hapus atau arsipkan `Gambar_V1..V6_UI_*.png` lama** (tangkapan Streamlit
  20 Agustus). Masih ada di `images/` dan **tidak dipakai** naskah mana pun, tetapi
  namanya mirip dengan gambar baru sehingga rawan tertukar.
- **Evaluasi dokter** masih berjalan. §6.2, §6.3, dan Bab VII ditahan sampai selesai.

---

## 8. Basis data: jangan DROP dulu

Membuang `stress_events` **akan merusak sistem**. Tabel itu masih dibaca setiap
penilaian klinis di `backend/services/supabase_data_service.py` baris 105, 108, dan
133. Yang berhenti pada 24 Agustus hanyalah **penulisan**.

| Tindakan | Akibat |
|---|---|
| `DROP TABLE stress_events` | 🔴 penilaian klinis mati (HTTP 500) |
| `DROP COLUMN stress_level` | 🔴 sama saja |
| `DELETE FROM stress_events` | ✅ aman |

Kolom yang benar-benar mati dan aman dibuang: `glucose_sources.started_at` dan
`glucose_sources.ended_at`. Kolom hanya-tulis (`insulin_type`, `meal_type`,
`activity_type`, `duration_min`) **disarankan dipertahankan**.

---

## 9. KF-08 diturunkan jadi "Tercapai sebagian"

`ClinicalDecisionLog` ada di `src/clinical_state/decision_log.py` tetapi
**satu-satunya** yang mengimpornya adalah `tests/test_decision_log.py`.

Yang **terwujud**: dokter sebagai penentu akhir — tidak ada tindakan otonom,
penyangkalan terpasang di tampilan dan *system prompt*, tiap klaim tertelusur
sampai potongan sumbernya. Yang **tidak terwujud**: pencatatan keputusan sebagai
jejak audit.

Ini sudah dinyatakan terbuka di §5.7 dan diperkuat sesi ini: paragraf Gambar V.8
menegaskan halaman riwayat merekam **keluaran sistem**, bukan tindakan dokter.

> **Catatan.** Validasi lewat form evaluasi dokter **bukan** pengganti KF-08.
> KF-08 adalah fitur sistem; form evaluasi adalah instrumen penelitian yang
> mengukur sistem dari luar. Menyamakannya melanggar Aturan 21, 34, dan 31.

---

## 10. Prinsip perangkaian (tetap berlaku)

Bab I, II, dan III **tidak boleh** menyebut sistem yang akan dibangun. Sistem
pertama kali muncul di **Bab IV**.

| Bab | Menjawab | Sumber kebenaran |
|---|---|---|
| I | mengapa penelitian perlu | data + literatur |
| II | apa kata literatur | sumber referensi |
| III | apa yang dibutuhkan masalah | Bab III sendiri |
| IV | bagaimana dirancang | Bab IV sendiri |
| V | bagaimana diwujudkan | **source code / artefak** |
| VI | apakah klaimnya terbukti | **output eksperimen** |
| VII | apa jawabannya | hasil evaluasi |

Yang paling sering terlewat:

- Bab II hanya melaporkan literatur — **tidak menetapkan solusi**;
- Bab IV **tidak boleh** menyebut nama fungsi, berkas, kelas, atau model aktual
  (`\texttt` di Bab IV kini **nol**);
- Bab IV **tidak boleh** memakai angka hasil evaluasi untuk membenarkan rancangan;
- Bab V harus dapat dijawab dengan *"tunjukkan buktinya di source code"* — kini
  ditopang 4 listing;
- bedakan **dirancang / diimplementasikan / diuji / terbukti**.

---

## 11. Catatan teknis — WAJIB DIBACA SEBELUM MENYUNTING

**MiKTeX menggantung pada paket yang belum terpasang.** Menambah `\usepackage`
yang belum ada membuat `latexmk` menggantung **tanpa batas** dalam mode
non-interaktif, dan `kpsewhich <paket>.sty` pun ikut menggantung sehingga tidak
dapat dipakai memeriksa. Terkonfirmasi 25 Agustus: `placeins` dan `pdflscape`
**tidak terpasang**; `rotating`, `lscape`, `array`, `tabularx`, `longtable`,
`caption`, `subcaption`, `floatrow`, `threeparttable` **terpasang**. Periksa lewat
filesystem:

```bash
find "C:/Users/Asus/AppData/Local/Programs/MiKTeX/tex" -name "<paket>.sty"
```

Bila menggantung, matikan dengan `taskkill //F //IM lualatex.exe //T`.

**Heredoc merusak backslash.** `<<'PY'` meruntuhkan `\\` jadi `\`, lalu Python
membaca `\b`/`\t` sebagai karakter kendali. Ini **sudah menyebabkan kegagalan lagi
pada sesi ini**. Untuk menyunting LaTeX: pakai `python -c` dengan `chr(92)`, atau
tulis skrip ke berkas lebih dulu. Jangan pernah heredoc.

**Akhiran baris.** Bab I ber-LF, sisanya CRLF; berkas tabel ber-LF. Skrip penyunting
harus membaca dan menulis dengan `newline=''`.

**Interpreter.** Pakai env proyek: `conda run -n diabetes-ta python <skrip>`.
Anaconda base menghasilkan kegagalan palsu. `conda run` **tidak menerima**
`python -c` yang memuat baris baru.

**Hambatan sistem.** `cp` untuk membuat berkas baru di `images/` **ditolak**.
Yang berhasil: skrip Python via `conda run` (`shutil.copyfile`). Bila tertolak,
berhenti setelah satu diagnosis.

**Memeriksa PDF secara visual.** Nomor halaman di `TA.lof`/`TA.lot` adalah nomor
**cetak**; PDF bergeser **+32** halaman karena bagian depan bernomor Romawi.

```bash
pdftoppm -r 68 -f <halaman_pdf> -l <halaman_pdf> -png TA.pdf keluaran
```

**Git lokal saja.** Commit boleh; `push` dan `remote add` **tidak pernah**.

---

## Lampiran: peta butir bimbingan → lokasi

| Butir | Isi ringkas | Ditutup di | Status |
|---|---|---|---|
| 20/8 #1 | mengapa 3 model | §2.4.3, §4.7.1 | ✅ |
| 20/8 #2 | DRM vs DSRM | §1.5 | ✅ |
| 20/8 #3 | prompt engineering / fine-tuning | §2.6.6, §3.4, §4.8.2 | ✅ |
| 20/8 #4 | alur jelas, letak kebaruan | §4.1 + Bab IV | ✅ |
| GB.1 | tiap subbab ≥ 2 section | Bab IV–V | ✅ **Bab V ditutup sesi ini** |
| GB.2 | tiap proses sebut tools/method/rumus | Bab IV–V | ✅ |
| GB.3 | hapus sarang online/offline | §4.1, §4.8.1 | ✅ |
| Obj 1 | perjelas 12 dokumen | §4.2.2, §4.5.1 | ✅ |
| Obj 2 | penamaan preprocessing | §4.3 | ✅ |
| Obj 3 | diurnal features | §4.4.3 | ⚠️ klaim tanpa penopang |
| Obj 4 | bentuk feature vector | §4.4.5 | ✅ |
| Obj 5 | mengapa LSTM/GBM/RF | §2.4.3, §4.7.1 | ✅ |
| Obj 6 | pemilihan GBM harus ilmiah | §3.4, §4.7.3, §6.2.1 | ✅ |
| Obj 7 | letak prompt vs chunking | §4.3.2, §4.5.2, §4.8.2 | ✅ |
| Obj 8 | classifier pakai apa | §4.7.4 | ✅ |
| Obj 9 | masukan PC-Transform | §4.8.1 | ✅ |
| Obj 10 | jelaskan BM25 | §2.6.7, §4.6.2 | ✅ **plus Listing V.1 di §5.6** |
| Obj 11 | chunk berhenti di titik | §4.5.2 | ✅ **plus Listing V.3** |
| Suhono #1 | justifikasi dataset terbuka | §3.3.1, §2.8.3 | ✅ |
| Suhono #2 | etika, keamanan, privasi | §2.8 | ✅ |
| Suhono #3 | opini 3 pakar | §6.1.5, §6.2.7 | 🙋 evaluasi dokter berjalan |
| K1 | cabut angka +0,158 | §6.3 | ✅ nol kemunculan |
| K2 | reranking = Inactive | §4.6.3 | ✅ terverifikasi |
| K3 | 206 tes, bukan 177 | §5.9 | ✅ |

---

## Urutan kerja yang disarankan untuk sesi berikutnya

1. **Putuskan bagian 1** (reproduksibilitas Bab VI). Tanpa ini, §6.2/§6.3/Bab VII
   tidak boleh disentuh.
2. **Delapan tabel Bab VI yang keluar margin** (4.1) — resepnya sudah terbukti,
   tinggal diulang.
3. **Periksa dua cacat isi pada Gambar V.3 dan V.5** (4.2), ambil ulang bila perlu.
4. **§7.2 jadi tabel** (4.4) dan **ERD Bab IV** (4.5).
5. Prioritas 2 dan 3.
6. Compile akhir, periksa Daftar Gambar / Tabel / Listing, telusuri seluruh
   `Overfull \hbox` sampai nol pada tabel.
