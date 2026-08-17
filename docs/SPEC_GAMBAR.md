# Spesifikasi gambar Bab IV

Dokumen ini memerikan **apa yang harus ada di dalam tiap gambar** yang digambar manual pada
Bab IV: isi kotak, label persis, arah panah, dan angka. Tujuannya agar gambar dapat dibuat
ulang tanpa menebak, dan agar isinya tidak menyimpang dari kode maupun naskah.

Pendamping `docs/METHODOLOGY.md`, yang memerikan **keputusan** di balik sistem. Dokumen ini
memerikan **tampilannya**.

---

## 0. Aturan yang berlaku bagi SELURUH gambar

### 0.1 Enam istilah yang DILARANG muncul

Keenamnya sudah dicabut dari penelitian. Bila salah satu muncul di gambar, gambar itu
bertentangan dengan naskah:

| Dilarang | Gantinya |
|---|---|
| *digital twin*, *twin*, *DT* | **kondisi klinis terstruktur** |
| *what-if*, simulasi pengandaian | — (komponennya dicabut, jangan digambar) |
| *Random Forest* sebagai model utama | **Gradient Boosting** |
| *MMR* / *vector search* sebagai jalur produksi | **Okapi BM25** (leksikal) |
| `gemini-2.5-flash-lite` | **`gemini-3.5-flash-lite`** |
| *surrogate*, *tipe 2*, *T2DM* | — (penelitian ini pada diabetes tipe 1) |

### 0.2 Angka yang WAJIB dipakai

Seluruhnya dibaca dari `config.yaml` dan `results/` pada 17 Agustus 2026. Jangan memakai
angka lain, dan jangan menambah angka yang tidak ada di daftar ini.

| Besaran | Nilai |
|---|---|
| Panjang jendela masukan | 12 langkah (± 1 jam) untuk CGM; 6 langkah untuk SMBG |
| Jumlah fitur | 7 |
| Daftar fitur | `glucose`, `glucose_delta`, `iob`, `cob`, `activity`, `hour_sin`, `hour_cos` |
| Konstanta peluruhan | IOB τ = 240 menit; COB τ = 180 menit |
| Horizon prediksi | +30 menit (6 langkah) dan +60 menit (12 langkah) |
| Model produksi | `HistGradientBoostingRegressor` |
| Sumber sigma | dua model kuantil, 0,025 dan 0,975; σ = (q₀,₉₇₅ − q₀,₀₂₅) / 3,92 |
| Potongan korpus | **2.233** |
| Ukuran potongan | 900 karakter, tumpang-tindih 120, dipotong pada **batas kalimat** |
| Model *embedding* | `all-MiniLM-L6-v2`, 384 dimensi, berjalan pada prosesor |
| Cara penelusuran produksi | **`bm25`** (Okapi BM25) |
| Potongan diambil | `top_k` = **5** |
| Model bahasa | `gemini-3.5-flash-lite`, suhu 0,2, maks 700 token |
| Waktu siklus lokal | 194,2 ms |
| Waktu penelusuran | 161,4 ms |
| Waktu model bahasa | ± 8,3 detik |
| Jejak memori | 632 MB |

### 0.3 Nama berkas: ada satu tabrakan nama, dan tidak ada Gambar IV.2

**Nomor gambar pada PDF dihasilkan LaTeX secara berurutan, bukan dari nama berkas.** Bab IV
memuat **enam** gambar, sehingga pada naskah tercetak bernomor **Gambar IV.1 sampai
Gambar IV.6, tanpa lompatan.** Yang melompat hanyalah *nama berkasnya*, dan itu tidak
terlihat pembaca.

| Nama berkas | Nomor pada PDF | Bagian spesifikasi |
|---|---|---|
| `Gambar_IV1_Arsitektur.png` | Gambar IV.1 | §1 |
| `Gambar_IV3_PipelinePCRAG.png` | Gambar IV.2 | §2 |
| **`Gambar_IV1_AlurRinciSistem.png`** | **Gambar IV.3** | **§6 — WAJIB, arahan pembimbing** |
| `Gambar_IV4_ClassDiagram.png` | Gambar IV.4 | §3 |
| `Gambar_IV5_SequencePCRAG.png` | Gambar IV.5 | §4 |
| `Gambar_IV6_Wireframe.png` | Gambar IV.6 | §5 |

**Mengapa tidak ada berkas `Gambar_IV2_*`.** Berkas itu dahulu bernama
`Gambar_IV2_DuaMesin.png` dan memerikan rancangan dua mesin, yakni satu model untuk
prakiraan dan formula farmakokinetik untuk simulasi pengandaian. Rancangan itu **dicabut**
bersama seluruh komponen *what-if*, dan gambarnya dipindahkan ke `docs/arsip/gambar/`.
Penomoran berkas sengaja **tidak** dirapikan, supaya nama berkas tetap cocok dengan riwayat
dan dengan hasil evaluasi lama yang menyebutnya.

**Tabrakan nama yang perlu diwaspadai.** Ada dua berkas yang keduanya berawalan
`Gambar_IV1`, dan **keduanya dirujuk naskah**:

- `Gambar_IV1_Arsitektur.png` → tercetak **Gambar IV.1**, arsitektur berlapis lima lapis (§1)
- `Gambar_IV1_AlurRinciSistem.png` → tercetak **Gambar IV.3**, alur rinci luring dan daring (§6)

Keduanya **berbeda bentuk dan berbeda tujuan**, dan awalan nama yang sama itu warisan
penamaan lama. Selalu sebut nama berkas lengkapnya, jangan hanya "Gambar IV.1".

### 0.4 Konvensi visual

- **Bahasa Indonesia** untuk seluruh label, kecuali nama kelas, nama berkas, nama pustaka,
  dan nama model — keempatnya ditulis apa adanya.
- **Empat kelompok warna**, dipakai konsisten di seluruh gambar:
  biru = data dan masukan · hijau = prakiraan · kuning/jingga = kondisi klinis dan
  kebaruan · ungu = penelusuran dan pembangkitan · merah/hijau tua = validasi dokter.
- **Kotak berlatar kuning menandai kontribusi metodologis.** Tepatnya empat: pengklasifikasi
  kondisi, perakitan `PatientState`, transformasi kueri, dan resolusi sitasi dari metadata.
- Ukuran ekspor: PNG, skala 2×, latar putih (bukan transparan), lebar muat pada
  `0.95\textwidth`.

---

## 1. `Gambar_IV1_Arsitektur.png` — Arsitektur Berlapis

*Tercetak sebagai **Gambar IV.1**.*

**Label naskah:** `fig:arsitektur`

**Yang diklaim keterangan gambar:** *"Arsitektur berlapis solusi: dari data logbook menuju
prediksi, ringkasan keadaan pasien, penalaran berbasis dokumen, dan rekomendasi tertelusur
yang divalidasi dokter."*

**Bentuk:** lima lapis bertumpuk, digambar dari bawah ke atas atau kiri ke kanan. Tiap lapis
satu kotak besar berisi komponennya.

| Lapis | Isi kotak |
|---|---|
| 1. Data | Catatan *logbook*: glukosa, karbohidrat, insulin, aktivitas, stres · Dataset OhioT1DM (12 pasien) |
| 2. Prakiraan | Rekayasa fitur berbasis fisiologi (IOB, COB, tren, pola diurnal) · Gradient Boosting +30/+60 menit · Interval konformal · Pengklasifikasi kondisi |
| 3. Kondisi klinis | `PatientState`: tingkat risiko, arah tren, kegentingan · Peringatan divergensi |
| 4. Penalaran berbasis dokumen | Pembentuk kueri terkondisi-prediksi · Penelusuran BM25 atas 2.233 potongan · Model bahasa · Resolusi sitasi dari metadata |
| 5. Validasi dokter | Antarmuka konsultasi · Persetujuan/penyesuaian/penolakan · Jejak audit keputusan |

**Panah:** satu arah menaik antar-lapis, ditambah satu panah balik dari lapis 5 ke lapis 1
berlabel *"keputusan tercatat"*.

**Yang harus terbaca:** lapis 3 menerima masukan dari lapis 2 dan **nilai yang dipakainya
adalah nilai terprediksi**, bukan nilai terkini. Beri label pada panah lapis 2 → 3:
**"nilai TERPREDIKSI"**.

---

## 2. `Gambar_IV3_PipelinePCRAG.png` — Pipeline RAG Terkondisi-Prediksi

*Tercetak sebagai **Gambar IV.2**.*

**Label naskah:** `fig:pipeline-pcrag`

**Status gambar lama: BASI.** Terverifikasi memuat "Random Forest — predict()",
"MMR Retriever … top_k 4", dan `gemini-2.5-flash-lite`. Ketiganya salah.

**Bentuk:** alur tegak satu jalur, **sebelas kotak** sesuai tabel di bawah.

| # | Kotak | Isi label |
|---|---|---|
| 1 | Masukan | Catatan *logbook*: glukosa, karbohidrat, insulin, aktivitas, stres |
| 2 | Prapemrosesan | Jendela 12 × 5 menit · 7 fitur: `glucose`, `glucose_delta`, `iob`, `cob`, `activity`, `hour_sin`, `hour_cos` |
| 3 | Prakiraan | **Gradient Boosting** · target Δglukosa lalu direkonstruksi ke nilai absolut · horizon +30 / +60 menit |
| 4 | Ketidakpastian | Dua model kuantil (0,025 dan 0,975) → σ → **interval konformal 95%** |
| 5 | Kondisi *(kuning)* | **Pengklasifikasi kondisi** sadar-biaya → hipoglikemia / normal / hiperglikemia |
| 6 | Kontrak data *(kuning)* | `PatientState`: `risk_level`, `trend_direction`, `urgency` — **diturunkan dari nilai TERPREDIKSI** |
| 7 | Kebaruan *(kuning, garis tebal)* | **Pembentuk kueri terkondisi-prediksi** — kueri disusun dari kondisi masa depan |
| 8 | Penelusuran | **Okapi BM25** atas **2.233 potongan** ChromaDB · lima potongan teratas |
| 9 | Pembangkitan | `gemini-3.5-flash-lite`, suhu 0,2, maks 700 token · rekomendasi dibumikan pada potongan |
| 10 | Sitasi *(kuning)* | Nomor halaman diresolusi **dari metadata potongan**, bukan dari teks model |
| 11 | Validasi | Dokter menyetujui / menyesuaikan / menolak → jejak audit |

**Dua kotak catatan di samping (bergaris putus-putus):**

1. **Kontras terhadap RAG konvensional** — *"RAG konvensional menyusun kueri dari glukosa
   saat ini. Pada kasus divergen, yaitu ketika kondisi kini normal tetapi prediksi menuju
   hipoglikemia atau hiperglikemia, RAG konvensional menargetkan kondisi normal sehingga
   buta terhadap bahaya yang akan datang."* Sambungkan ke kotak 7.

2. **Cadangan berlapis (KNF-04)** — tiga baris:
   - Data: OhioT1DM → CSV → logbook manual
   - Penelusuran: BM25 → jalur padat, **disertai pernyataan penurunan**
   - Pembangkitan: model bahasa → templat berbasis potongan

**Yang harus terbaca:** kotak 4 (interval) **tidak** memberi masukan ke kotak 7. Gambarkan
panah dari kotak 4 langsung ke kotak 11, berlabel *"peringatan klinis"*.

Klaim ini **sudah diverifikasi pada kode**, bukan disimpulkan dari rancangan.
`PatientState` dan `_primary_query` memang *mampu* memperluas kueri dengan kondisi yang
tercakup interval, tetapi `app/streamlit_app.py` **tidak meneruskan** `predicted_lower`
maupun `predicted_upper`, sehingga keduanya `None` dan cabang itu dilewati. Kemampuan itu
hanya diaktifkan lengan `pc_rag_interval` pada evaluasi, yang terbukti terbaik pada kasus
divergen tetapi runtuh pada distribusi natural — sebab itulah ia tidak diadopsi. Rinciannya
pada `docs/METHODOLOGY.md` §7.0.

---

## 3. `Gambar_IV4_ClassDiagram.png` — Diagram Kelas

*Tercetak sebagai **Gambar IV.3**.*

**Label naskah:** `fig:class-diagram`

**Status gambar lama: BASI.** Terverifikasi memuat `DigitalTwinStateManager`,
`WhatIfSimulator`, dan `RandomForestGlucoseModel` sebagai model utama. Kelas
`digital_twin` **sudah dihapus dari kode**; direktorinya kosong.

**Bentuk:** diagram kelas UML sederhana, tiga kolom dari kiri ke kanan.

### Kolom 1 — Lapisan data dan prakiraan *(hijau)*

| Kelas | Isi |
|---|---|
| `OhioParser` *(`src/data/ohio_parser.py`)* | `+ process_ohio_dataset(): DataFrame`<br>`+ parse_ohio_fingerstick(): DataFrame` |
| `DataPreprocessor` *(`src/data/preprocessor.py`)* | `+ handle_missing_values(df)`<br>`+ engineer_features(df)`<br>`+ create_sequences(df, w, h)`<br>`+ split_by_patient(df, uji)`<br>`+ normalize_data(Xtr, Xte)` |
| `BaseGlucoseModel` «antarmuka» *(garis putus-putus)* | `+ train(X, y)`<br>`+ predict(X)`<br>`+ save(path) / load(path)` |
| **`GBMGlucoseModel`** «produksi» *(garis tebal)* | `- model: HistGradientBoostingRegressor`<br>`- model_q025, model_q975`<br>`+ train(X, y): dict`<br>`+ predict(X)`<br>`+ predict_std(X)`<br>σ = (q₀,₉₇₅ − q₀,₀₂₅) / 3,92 |
| `RandomForestGlucoseModel` «pembanding» | *(kotak satu baris, garis putus-putus)* |
| `LSTMGlucoseModel` «pembanding» | *(kotak satu baris, garis putus-putus)* |

### Kolom 2 — Kontrak data dan kondisi klinis *(kuning/jingga)*

| Kelas | Isi |
|---|---|
| **`PatientState`** «kontrak data» *(garis tebal, kuning)* | `+ current_glucose, predicted_glucose`<br>`+ insulin_on_board, carbs_on_board`<br>`+ activity_level, stress_level`<br>*— diturunkan dari nilai TERPREDIKSI —*<br>`+ risk_level`: hipo \| normal \| hiper<br>`+ trend_direction, trend_rate`<br>`+ urgency`: critical \| high \| medium \| low<br>`+ to_rag_context(): dict`<br>`+ from_model_output(...)` |
| `alerts` «modul» *(`src/alerts.py`)* | `+ evaluate_divergence(kini, prediksi, menit)`<br>peringatan saat kondisi kini aman tetapi kondisi terprediksi tidak |
| `ClinicalDecisionLog` *(`src/clinical_state/decision_log.py`)* | `+ record(StateRecord)`<br>`+ load() / save()` → JSON, jejak audit |

### Kolom 3 — Lapisan RAG *(ungu)*

| Kelas | Isi |
|---|---|
| **`PredictionConditionedQueryBuilder`** *(garis tebal)* | `+ build(state): query`<br>kueri disusun dari kondisi TERPREDIKSI |
| `RAGPipeline` *(`src/rag/pipeline.py`)* | `+ _build_query(state, prediksi)`<br>`+ _retrieve(query, top_k=5)`<br>`+ answer(...)`<br>`+ _ensure_disclaimer(teks)` |
| **`MMRRetriever`** *(garis tebal)* | `- retrieval_mode`: vektor \| **bm25** \| hibrida<br>`+ retrieve(query, top_k)`<br>`- _peringkat_bm25(query, n)`<br>`- _gabung_rrf(daftar)`<br>**bm25 = jalur produksi** |
| `SimpleKeywordRetriever` «cadangan, KNF-04» | *(kotak satu baris, garis putus-putus)* |
| `MedicalKnowledgeBase` | `+ chunk_documents()` → 900/120, batas kalimat<br>`+ save_to_chroma()` |
| `RAGGenerator` | `gemini-3.5-flash-lite`, suhu 0,2, maks 700 token<br>`+ generate(context, query)`<br>`+ _template_answer()` «cadangan» |
| `citations` «modul» *(`src/rag/citations.py`)* | `+ potong_batas_kalimat(teks, batas)`<br>kutipan berhenti di akhir kalimat (KNF-08) |
| ChromaDB *(bentuk silinder)* | 2.233 potongan · `all-MiniLM-L6-v2`, 384 dimensi |

### Hubungan antar-kelas

| Dari | Ke | Jenis panah | Label |
|---|---|---|---|
| `OhioParser` | `DataPreprocessor` | panah penuh | — |
| `DataPreprocessor` | `BaseGlucoseModel` | panah penuh | — |
| `GBMGlucoseModel`, `RandomForestGlucoseModel`, `LSTMGlucoseModel` | `BaseGlucoseModel` | panah kosong putus-putus (realisasi) | — |
| `GBMGlucoseModel` | `PatientState` | panah penuh | **"prediksi + σ"** |
| `PatientState` | `alerts` | putus-putus | — |
| `PatientState` | `ClinicalDecisionLog` | putus-putus | — |
| `PatientState` | `PredictionConditionedQueryBuilder` | panah penuh | **"kontrak data"** |
| `RAGPipeline` | `PredictionConditionedQueryBuilder` | belah ketupat penuh (komposisi) | — |
| `RAGPipeline` | `MMRRetriever` | belah ketupat penuh | — |
| `RAGPipeline` | `RAGGenerator` | belah ketupat penuh | — |
| `MMRRetriever` | `SimpleKeywordRetriever` | putus-putus | **"bila indeks gagal"** |
| `MMRRetriever` | ChromaDB | panah penuh | **"baca potongan"** |
| `MMRRetriever` | `citations` | putus-putus | — |
| `MedicalKnowledgeBase` | ChromaDB | panah penuh | — |

**Yang harus terbaca:** `PatientState` adalah **satu-satunya** jalur yang menghubungkan
lapisan prakiraan dengan lapisan RAG. Tidak boleh ada panah langsung dari
`GBMGlucoseModel` ke `RAGPipeline` atau ke `MMRRetriever`.

---

## 4. `Gambar_IV5_SequencePCRAG.png` — Diagram Sekuens

*Tercetak sebagai **Gambar IV.4**.*

**Label naskah:** `fig:sequence-pcrag`

**Yang diklaim keterangan gambar:** *"Diagram sekuens satu siklus RAG terkondisi-prediksi,
dari permintaan dokter hingga rekomendasi tertelusur."*

**Pelaku (dari kiri ke kanan):**

1. Dokter
2. Antarmuka Streamlit
3. `GBMGlucoseModel`
4. Pengklasifikasi kondisi
5. `PatientState`
6. `PredictionConditionedQueryBuilder`
7. `MMRRetriever`
8. ChromaDB
9. `RAGGenerator` (`gemini-3.5-flash-lite`)
10. `citations`
11. `ClinicalDecisionLog`

**Urutan pesan:**

| # | Dari → Ke | Pesan |
|---|---|---|
| 1 | Dokter → Antarmuka | Pilih pasien, masukkan catatan *logbook* |
| 2 | Antarmuka → Antarmuka | Bangun jendela 12 baris terakhir, hitung 7 fitur |
| 3 | Antarmuka → `GBMGlucoseModel` | `predict(X)` |
| 4 | `GBMGlucoseModel` → Antarmuka | Δglukosa → nilai absolut, σ dari lebar antar-kuantil |
| 5 | Antarmuka → Antarmuka | Terapkan faktor konformal → interval 95% |
| 6 | Antarmuka → Pengklasifikasi | `predict_condition(X)` |
| 7 | Pengklasifikasi → Antarmuka | hipoglikemia / normal / hiperglikemia |
| 8 | Antarmuka → `PatientState` | `from_model_output(pred, interval, kondisi)` |
| 9 | `PatientState` → Antarmuka | `risk_level`, `trend_direction`, `urgency` |
| 10 | Antarmuka → Dokter | **Peringatan divergensi** bila kondisi kini aman tetapi prediksi tidak |
| 11 | Antarmuka → Pembentuk kueri | `build(state)` |
| 12 | Pembentuk kueri → `MMRRetriever` | `primary_query` — **dari kondisi TERPREDIKSI** |
| 13 | `MMRRetriever` → ChromaDB | Ambil dokumen, bangun peringkat BM25 |
| 14 | ChromaDB → `MMRRetriever` | Lima potongan teratas beserta metadata |
| 15 | `MMRRetriever` → `RAGGenerator` | Konteks bertanda `[S1..Sn]`, **tanpa nomor halaman** |
| 16 | `RAGGenerator` → Antarmuka | Teks rekomendasi berbahasa Indonesia |
| 17 | Antarmuka → `citations` | `build_source_list(potongan)` |
| 18 | `citations` → Antarmuka | `page_label` dari metadata, kutipan dipotong batas kalimat |
| 19 | Antarmuka → Dokter | Prediksi, interval, peringatan, rekomendasi, sumber |
| 20 | Dokter → `ClinicalDecisionLog` | Setujui / sesuaikan / tolak |

**Dua hal yang harus terbaca:**

- Pesan 15 **tidak** membawa nomor halaman; nomor halaman baru muncul pada pesan 18. Beri
  catatan di samping: *"model bahasa tidak pernah melihat nomor halaman, sehingga tidak
  dapat mengarangnya"*.
- Pesan 12 berlabel tebal **"dari kondisi TERPREDIKSI"** — inilah kebaruannya.

---

## 5. `Gambar_IV6_Wireframe.png` — Wireframe Antarmuka

*Tercetak sebagai **Gambar IV.5**.*

**Label naskah:** `fig:wireframe`

**Status gambar lama: BASI.** Terverifikasi memuat bilah navigasi
`[ Input Logbook ] [ Prediksi ] [ What-If ] [ Twin Dashboard ]`. Dua tab terakhir **tidak
ada** di aplikasi; aplikasi sebenarnya punya **dua tab**: *Rekomendasi Klinis* dan
*Catat Keputusan*.

**Bentuk:** satu kerangka jendela, gaya kotak-kawat (bukan tangkapan layar).

**Susunan dari atas ke bawah:**

1. **Bilah judul** — `Konsol Konsultasi Diabetes — Pasien: P001`
2. **Bilah navigasi** — `[ Input Logbook ]  [ Konsultasi ]` (hanya dua)
3. **Kolom kiri, panel biru — INPUT LOGBOOK**
   ```
   Glukosa (mg/dL):  [______]
   Karbohidrat (g):  [______]
   Insulin (u):      [______]
   Aktivitas:        [______]
   Stres (0–10):     [______]

   [ Simpan & Prediksi ]
   ```
4. **Kolom kanan atas, panel kuning — PRAKIRAAN**
   ```
   +30 mnt:  168 mg/dL   [interval 150–186]
   +60 mnt:  205 mg/dL   [interval 178–232]

   Risiko: HIPERGLIKEMIA (menuju)
   Tren: NAIK ↑     Kegentingan: SEDANG
   ```
5. **Kolom kanan, panel merah — PERINGATAN DIVERGENSI**
   > ⚠ Kondisi terkini masih normal (150), tetapi kondisi terprediksi menuju
   > hiperglikemia (205) dalam 60 menit.
6. **Kolom kanan, panel ungu — REKOMENDASI KLINIS** (antisipatif, dibumikan pada dokumen)
   ```
   •  [teks nasihat berbahasa Indonesia …]

   Sumber: PERKENI-2021, hlm. XX  ·  ADA-SOC-2025, hlm. XX
   ```
7. **Panel lebar, hijau — RUJUKAN PANDUAN MEDIS** (potongan yang benar-benar ditelusur)
   ```
   1.  PERKENI-2021 · hlm. 42 — "…kutipan berhenti pada batas kalimat…"
   2.  ADA-SOC-2025 · hlm. 118 — "…"

   Tiap potongan menyimpan identitas dokumen dan nomor halamannya
   ```
8. **Bilah bawah, hijau — KEPUTUSAN DOKTER**
   `[ ✓ Setujui ]   [ ✎ Sesuaikan ]   [ ✗ Tolak ]  →  dicatat sebagai jejak audit;
   diteruskan hanya setelah validasi dokter`
9. **Catatan kaki kecil, kelabu** — *"Nilai pada wireframe hanya ilustrasi tata letak."*

**Yang harus terbaca:** panel 7 memperlihatkan **nomor halaman** dan kutipan yang
**berhenti di akhir kalimat**. Keduanya bukti KNF-08, dan itulah sebabnya panel ini harus
tampil utuh, bukan disingkat.

---

## 6. `Gambar_IV1_AlurRinciSistem.png` — Alur Rinci Sistem

**Tercetak sebagai Gambar IV.3.** **WAJIB** — arahan pembimbing, dan sudah dirujuk
`Bab IV - Perancangan.tex` dengan label `fig:alur-rinci`.

**Yang diklaim keterangan gambar:** *"Alur rinci sistem, memisahkan tahap luring yang
dijalankan sekali di luar aplikasi dari tahap daring yang dijalankan pada tiap konsultasi."*

**Bentuk:** alur bercabang dengan **dua wadah bertanda** (`subgraph`), luring di atas dan
daring di bawah. Ini gambar terbesar di Bab IV; format mendatar (*landscape*) boleh dipakai.

### 6.1 Wadah LURING — dijalankan sekali, di luar aplikasi

| Kotak | Label | Ke |
|---|---|---|
| A | OhioT1DM XML · 12 pasien | B |
| B | `ohio_parser.py` · selaraskan *event* ±2,5 mnt | C |
| C | `ohio_t1dm_merged.csv` · CGM 5 menit | D |
| D | `preprocessor` · `engineer_features` → `create_sequences` | E, E2, E3 |
| E | Latih **Gradient Boosting** · *bundle* `.pkl` per horizon | — |
| E2 | Kalibrasi konformal · `conformal_h6/h12.json` | — |
| E3 | **Pengklasifikasi kondisi** · hipo / normal / hiper *(kuning)* | — |
| F | 12 PDF pedoman · KB-01…KB-12 + `manifest.csv` | G |
| G | Pecah 900/120 pada **batas kalimat** · `all-MiniLM-L6-v2` | H |
| H | **ChromaDB `diabetes_kb` · 2.233 potongan** *(bentuk silinder)* | — |

### 6.2 Wadah DARING — dijalankan tiap konsultasi

| Kotak | Label | Waktu | Ke |
|---|---|---|---|
| I | Dokter memilih pasien, memasukkan *logbook* | — | J |
| J | Bangun jendela 12 baris terakhir · hitung 7 fitur | 25,2 ms | K, M |
| K | `predict` · Δglukosa lalu rekonstruksi ke nilai absolut | 7,6 ms | L, O |
| L | σ dari lebar antar-kuantil → **interval konformal 95%** | — | N, U |
| M | **Pengklasifikasi kondisi** *(kuning)* | — | O |
| N | `evaluate_divergence` · peringatan bila kondisi kini aman tetapi prediksi tidak | — | U |
| O | **`PatientState`** *(kuning)* · risiko, tren, kegentingan — dari **nilai TERPREDIKSI** | — | P |
| P | **Pembentuk kueri terkondisi-prediksi** *(kuning, garis tebal)* | — | Q |
| Q | **`MMRRetriever` mode `bm25`** · Okapi BM25 → `top_k` 5 | 161,4 ms | R |
| R | `gemini-3.5-flash-lite` · suhu 0,2 · maks 700 token | ± 8,3 s | S |
| S | `_ensure_disclaimer` | — | T |
| T | **`build_source_list`** *(kuning)* · nomor halaman dari **METADATA** | — | U |
| U | Layar dokter · prediksi, interval, peringatan, rekomendasi, sumber | — | V |
| V | `ClinicalDecisionLog` · jejak audit | — | — |

### 6.3 Panah antar-wadah

Digambar **putus-putus**, sebab menandai artefak yang dipakai ulang, bukan aliran data
dalam satu jalan:

- E ⇢ K *(bundle model)* · E2 ⇢ L *(faktor konformal)* · E3 ⇢ M *(pengklasifikasi)*
- H ⇢ Q *(indeks leksikal dibangun dari potongan ChromaDB)*

### 6.4 Empat kotak kuning, tidak lebih dan tidak kurang

**E3/M** pengklasifikasi kondisi · **O** perakitan `PatientState` · **P** transformasi
kueri · **T** resolusi sitasi dari metadata.

### 6.5 Kotak keterangan waktu, di sudut gambar

```
Subtotal lokal (J + K + Q)   : 194,2 ms
Model bahasa (R)             : ± 8,3 s
Total satu rekomendasi       : ± 8,5 s
Jejak memori                 : 632 MB
Muat artefak, sekali di awal : 18,6 s
```

### 6.6 Tiga hal yang HARUS terbaca

Ketiganya klaim penelitian; bila hilang, gambar ini kehilangan gunanya.

1. **Kotak O memakai nilai TERPREDIKSI**, bukan nilai terkini. Beri label pada panah K → O:
   **"nilai terprediksi"**. Bila kotak ini memakai nilai sekarang, seluruh sifat antisipatif
   sistem batal.
2. **Kotak R tidak pernah menerima nomor halaman.** Beri catatan di samping: *"konteks
   bertanda `[S1..Sn]` tanpa nomor halaman; nomor halaman baru muncul pada T dari metadata,
   sehingga model bahasa tidak dapat mengarangnya"*.
3. **Kotak V menyimpan sumber yang persis dilihat dokter**, bukan hasil penelusuran mentah,
   sehingga keputusan dapat diaudit ke halaman dokumen di kemudian hari.

### 6.7 Yang TIDAK boleh digambar

- Panah dari **L (interval)** ke **P (pembentuk kueri)**. Interval hanya menuju N dan U.
  Alasannya diverifikasi pada kode; lihat §2 dan `docs/METHODOLOGY.md` §7.0.
- Tahap pelatihan atau pengindeksan di dalam wadah daring. Seluruhnya luring — justru
  pemisahan itulah yang membuat batasan komputasi dapat dipenuhi.

**Rujukan tambahan:** tabel kontrak data 14 tahap pada `docs/METHODOLOGY.md` §7.0 memerikan
objek apa yang berpindah antar-tahap beserta tempat memeriksanya di kode.

---

## 7. Daftar periksa sebelum gambar dianggap selesai

- [ ] Tidak ada satu pun istilah dari daftar larangan pada §0.1
- [ ] Seluruh angka cocok dengan tabel §0.2
- [ ] Empat kotak kontribusi berlatar kuning, tidak lebih dan tidak kurang
- [ ] Label berbahasa Indonesia, kecuali nama kelas, berkas, pustaka, dan model
- [ ] Nama berkas dan kelas ditulis persis seperti di kode
- [ ] Diekspor PNG skala 2×, latar putih
- [ ] Nama berkas keluaran persis sama dengan yang dirujuk naskah, **keenamnya**:
      `Gambar_IV1_Arsitektur.png`, `Gambar_IV3_PipelinePCRAG.png`,
      **`Gambar_IV1_AlurRinciSistem.png`**, `Gambar_IV4_ClassDiagram.png`,
      `Gambar_IV5_SequencePCRAG.png`, `Gambar_IV6_Wireframe.png`
- [ ] Tidak ada panah dari kotak interval ke kotak pembentuk kueri, pada gambar mana pun
