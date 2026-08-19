# Isi gambar Bab IV

Dokumen ini memerikan **informasi apa yang harus ada di dalam tiap gambar** Bab IV: kotak apa
saja, label persisnya, apa yang menghubungkan apa, dan angka mana yang dipakai. Bentuk visual,
warna, dan tata letak **tidak diatur di sini** — itu diserahkan kepada pembuat gambar.

Pendamping `docs/METHODOLOGY.md`, yang memerikan keputusan di balik sistem.

---

## 0. Berlaku bagi SELURUH gambar

### 0.1 Istilah yang DILARANG muncul

Keenam kelompok ini sudah dicabut dari penelitian. Bila salah satu muncul, gambarnya
bertentangan dengan naskah.

| Dilarang | Gantinya |
|---|---|
| *digital twin*, *twin*, *DT* | **kondisi klinis terstruktur** |
| *what-if*, simulasi pengandaian | — komponennya dicabut, jangan digambar |
| *Random Forest* sebagai model utama | **Gradient Boosting** |
| *MMR* / pencarian vektor sebagai jalur produksi | **Okapi BM25** (leksikal) |
| `gemini-2.5-flash-lite` | **`gemini-3.5-flash-lite`** |
| *surrogate*, *tipe 2*, *T2DM* | — penelitian ini pada diabetes tipe 1 |

*Random Forest* dan *MMR* **boleh** muncul bila jelas dilabeli **pembanding**, bukan produksi.

### 0.2 Angka yang WAJIB dipakai

Dibaca dari `config.yaml` dan ChromaDB, **18 Agustus 2026, korpus V2**, dan kecocokannya
diverifikasi terprogram. Jangan memakai angka lain, jangan menambah angka di luar daftar ini.

**Satu pengecualian:** nilai contoh pada wireframe (§6) — kadar glukosa, interval, nomor
halaman kutipan — adalah **ilustrasi tata letak**, bukan hasil pengukuran. Ia dipakai persis
seperti tertulis di §6 dan tidak perlu dicocokkan ke tabel ini.

| Besaran | Nilai |
|---|---|
| Panjang jendela masukan | 12 langkah (± 1 jam) untuk CGM; 6 langkah untuk SMBG |
| Jumlah fitur | 7 |
| Daftar fitur | `glucose`, `glucose_delta`, `iob`, `cob`, `activity`, `hour_sin`, `hour_cos` |
| Konstanta peluruhan | IOB τ = 240 menit; COB τ = 180 menit |
| Horizon prediksi | +30 menit (6 langkah) dan +60 menit (12 langkah) |
| Model produksi | `HistGradientBoostingRegressor` (Gradient Boosting) |
| Sumber sigma | dua model kuantil 0,025 dan 0,975; σ = (q₀,₉₇₅ − q₀,₀₂₅) / 3,92 |
| Potongan korpus | **4.038** |
| Ukuran potongan | **500** karakter, tumpang-tindih **67**, dipotong pada **batas kalimat** |
| Model *embedding* | `all-MiniLM-L6-v2`, 384 dimensi, berjalan pada prosesor |
| Cara penelusuran produksi | **`bm25`** (Okapi BM25) |
| Potongan diambil | `top_k` = **5** |
| Model bahasa (konfigurasi) | `gemini-3.5-flash-lite`, suhu 0,2, maks 700 token |
| Rekayasa fitur | 23,0 ms |
| Prediksi | 7,5 ms |
| Penelusuran | 284,3 ms |
| Subtotal lokal | **314,8 ms** |
| Waktu pembangkitan | ± 9,3 detik |
| Total satu rekomendasi | ± 9,6 detik |
| Muat artefak, sekali di awal | 16,5 detik |
| Jejak memori | 641 MB |
| Korpus | 12 dokumen pedoman, KB-01…KB-12 |
| Halaman dokumen | 543 |
| Halaman diindeks | **494** — selisih 49 = 45 halaman depan + 4 halaman terlalu pendek |

### 0.3 Nama berkas dan nomor tercetak

Nomor gambar dihasilkan LaTeX dari urutan kemunculan, **bukan** dari nama berkas. Bab IV
memuat enam gambar, tercetak **IV.1 sampai IV.6 tanpa lompatan**. Nama berkasnya melompat
karena warisan penamaan lama, dan itu tidak terlihat pembaca.

| Nama berkas keluaran | Tercetak sebagai | Bagian |
|---|---|---|
| `Gambar_IV1_Arsitektur.png` | Gambar IV.1 | §1 |
| `Gambar_IV2_PipelinePCRAG.png` | Gambar IV.2 | §2 |
| `Gambar_IV3_AlurRinciSistem.png` | Gambar IV.3 | §3 |
| `Gambar_IV4_ClassDiagram.png` | Gambar IV.4 | §4 |
| `Gambar_IV5_SequencePCRAG.png` | Gambar IV.5 | §5 |
| `Gambar_IV6_Wireframe.png` | Gambar IV.6 | §6 |

Nomor pada nama berkas kini **sama** dengan nomor tercetak, jadi tidak ada lagi lompatan
maupun dua berkas berawalan `Gambar_IV1`. Penamaan lama (`Gambar_IV3_PipelinePCRAG.png` untuk
Gambar IV.2 dan `Gambar_IV1_AlurRinciSistem.png` untuk Gambar IV.3) sudah tidak dipakai.

### 0.4 Empat TAHAP yang ditandai sebagai kontribusi

1. Pengklasifikasi kondisi
2. Perakitan `PatientState`
3. Transformasi kueri terkondisi-prediksi
4. Resolusi sitasi dari metadata

**Yang dihitung empat adalah TAHAPnya, bukan kotaknya.** Satu tahap boleh muncul lebih dari
satu kotak bila memang begitu adanya di sistem, dan kotak-kotak itu semuanya ditandai.

Satu-satunya kasus demikian ada di §3: pengklasifikasi kondisi muncul sebagai **E3** di sisi
luring (pelatihannya) dan **M** di sisi daring (pemakaiannya). **Keduanya ditandai**, dan
keterangan gambar menyebutkan keduanya satu komponen. Jadi §3 memuat lima kotak bertanda
untuk empat tahap, dan itu benar.

**Ditandai hanya pada §2 dan §3.** Gambar arsitektur berlapis (§1) memang memuat keempat
komponen itu, tetapi penandaannya **tidak dipakai di sana**. Alasannya: §1 berperan sebagai
orientasi, lapisannya berupa pengelompokan dan bukan tahapan, sehingga menandai komponen di
dalam lapisan justru memecah pembacaan lapisannya. Yang harus menonjol di §1 hanya satu, yaitu
label **nilai TERPREDIKSI** pada hubungan lapis 2 → lapis 3.

### 0.5 Satu larangan yang berlaku di semua gambar

**Tidak boleh ada panah dari kotak interval ketidakpastian ke kotak pembentuk kueri.**

Sudah diverifikasi pada kode: `app/streamlit_app.py` tidak meneruskan `predicted_lower`
maupun `predicted_upper`, sehingga cabang yang memakainya dilewati. Interval hanya menuju
peringatan divergensi dan layar dokter.

---

## 1. `Gambar_IV1_Arsitektur.png` — Arsitektur Berlapis

Tercetak **Gambar IV.1**. Label naskah `fig:arsitektur`.

**Keterangan gambar di naskah:** *"Arsitektur berlapis solusi: dari data logbook menuju
prediksi, ringkasan keadaan pasien, penalaran berbasis dokumen, dan rekomendasi tertelusur
yang divalidasi dokter."*

**Lima lapis, berurutan:**

| Lapis | Isi |
|---|---|
| 1. Data | Catatan *logbook*: glukosa, karbohidrat, insulin, aktivitas, stres · Dataset OhioT1DM, 12 pasien |
| 2. Prakiraan | Rekayasa fitur berbasis fisiologi (IOB, COB, tren, pola diurnal) · Gradient Boosting +30/+60 menit · Interval konformal · Pengklasifikasi kondisi |
| 3. Kondisi klinis | `PatientState`: tingkat risiko, arah tren, kegentingan · Peringatan divergensi |
| 4. Penalaran berbasis dokumen | Pembentuk kueri terkondisi-prediksi · Penelusuran BM25 atas 4.038 potongan · Model bahasa · Resolusi sitasi dari metadata |
| 5. Validasi dokter | Antarmuka konsultasi · Setujui / sesuaikan / tolak · Jejak audit keputusan |

**Hubungan:** satu arah menaik lapis 1 → 5, ditambah satu panah balik dari lapis 5 ke lapis 1
berlabel **"keputusan tercatat"**.

**Yang harus terbaca:** panah lapis 2 → 3 diberi label **"nilai TERPREDIKSI"**. Lapis 3
memakai nilai yang diprediksi, bukan nilai yang sedang berlaku.

---

## 2. `Gambar_IV3_PipelinePCRAG.png` — Pipeline RAG Terkondisi-Prediksi

Tercetak **Gambar IV.2**. Label naskah `fig:pipeline-pcrag`.

**Gambar lama BASI** — terverifikasi memuat "Random Forest — predict()", "MMR Retriever …
top_k 4", dan `gemini-2.5-flash-lite`; tidak memuat interval konformal maupun pengklasifikasi
kondisi.

**Sebelas kotak, satu jalur berurutan:**

| # | Kotak | Isi label |
|---|---|---|
| 1 | Masukan | Catatan *logbook*: glukosa, karbohidrat, insulin, aktivitas, stres |
| 2 | Prapemrosesan | Jendela 12 × 5 menit · 7 fitur: `glucose`, `glucose_delta`, `iob`, `cob`, `activity`, `hour_sin`, `hour_cos` |
| 3 | Prakiraan | **Gradient Boosting** · target Δglukosa lalu direkonstruksi ke nilai absolut · horizon +30 / +60 menit |
| 4 | Ketidakpastian | Dua model kuantil (0,025 dan 0,975) → σ → **interval konformal 95%** |
| 5 | Kondisi ★ | **Pengklasifikasi kondisi** sadar-biaya → hipoglikemia / normal / hiperglikemia |
| 6 | Kontrak data ★ | `PatientState`: `risk_level`, `trend_direction`, `urgency` — diturunkan dari **nilai TERPREDIKSI** |
| 7 | Kebaruan ★ | **Pembentuk kueri terkondisi-prediksi** — kueri disusun dari kondisi masa depan |
| 8 | Penelusuran | **Okapi BM25** atas **4.038 potongan** ChromaDB · lima potongan teratas |
| 9 | Pembangkitan | `gemini-3.5-flash-lite`, suhu 0,2, maks 700 token · rekomendasi dibumikan pada potongan |
| 10 | Sitasi ★ | Nomor halaman diresolusi **dari metadata potongan**, bukan dari teks model |
| 11 | Validasi | Dokter setujui / sesuaikan / tolak → jejak audit |

★ = tahap kontribusi (§0.4).

**Dua kotak keterangan terpisah:**

1. **Kontras terhadap RAG konvensional**, disambungkan ke kotak 7:
   > RAG konvensional menyusun kueri dari glukosa saat ini. Pada kasus divergen, yaitu ketika
   > kondisi kini normal tetapi prediksi menuju hipoglikemia atau hiperglikemia, RAG
   > konvensional menargetkan kondisi normal sehingga buta terhadap bahaya yang akan datang.

2. **Cadangan berlapis (KNF-04)**, tiga baris:
   - Data: OhioT1DM → CSV → logbook manual
   - Penelusuran: BM25 → jalur padat, **disertai pernyataan penurunan**
   - Pembangkitan: model bahasa → templat berbasis potongan

**Hubungan khusus:** kotak 4 tersambung langsung ke kotak 11 berlabel **"peringatan klinis"**,
dan **tidak** tersambung ke kotak 7 (§0.5).

---

## 3. `Gambar_IV1_AlurRinciSistem.png` — Alur Rinci Sistem

Tercetak **Gambar IV.3**. Label naskah `fig:alur-rinci`. **Arahan pembimbing.**

**Keterangan gambar di naskah:** *"Alur rinci sistem, memisahkan tahap luring yang dijalankan
sekali di luar aplikasi dari tahap daring yang dijalankan pada tiap konsultasi."*

Gambar terbesar di Bab IV. Isinya terbagi **dua kelompok bertanda**: LURING dan DARING.

### 3.1 Kelompok LURING — dijalankan sekali, di luar aplikasi

| Kode | Isi label | Menuju |
|---|---|---|
| A | OhioT1DM XML · 12 pasien | B |
| B | `ohio_parser.py` · selaraskan *event* ±2,5 menit | C |
| C | `ohio_t1dm_merged.csv` · CGM 5 menit | D |
| D | `preprocessor` · `engineer_features` → `create_sequences` | E, E2, E3 |
| E | Latih **Gradient Boosting** · *bundle* `.pkl` per horizon | — |
| E2 | Kalibrasi konformal · `conformal_h6.json`, `conformal_h12.json` | — |
| E3 ★ | **Pengklasifikasi kondisi** · hipoglikemia / normal / hiperglikemia | — |
| F | 12 PDF pedoman · KB-01…KB-12 + `manifest.csv` · 494 halaman diindeks | G |
| G | Pecah 500/67 pada **batas kalimat** · `all-MiniLM-L6-v2` | H |
| H | **ChromaDB `diabetes_kb` · 4.038 potongan** | — |

### 3.2 Kelompok DARING — dijalankan tiap konsultasi

| Kode | Isi label | Waktu | Menuju |
|---|---|---|---|
| I | Dokter memilih pasien, memasukkan *logbook* | — | J |
| J | Bangun jendela 12 baris terakhir · hitung 7 fitur | 25,2 ms | K, M |
| K | `predict` · Δglukosa lalu rekonstruksi ke nilai absolut | 7,6 ms | L, O |
| L | σ dari lebar antar-kuantil → **interval konformal 95%** | — | N, U |
| M ★ | **Pengklasifikasi kondisi** | — | O |
| N | `evaluate_divergence` · peringatan bila kondisi kini aman tetapi prediksi tidak | — | U |
| O ★ | **`PatientState`** · risiko, tren, kegentingan — dari **nilai TERPREDIKSI** | — | P |
| P ★ | **Pembentuk kueri terkondisi-prediksi** | — | Q |
| Q | **`MMRRetriever` mode `bm25`** · Okapi BM25 → `top_k` 5 | 284,3 ms | R |
| R | `gemini-3.5-flash-lite` · suhu 0,2 · maks 700 token | ± 8,3 s | S |
| S | `_ensure_disclaimer` | — | T |
| T ★ | **`build_source_list`** · nomor halaman dari **METADATA** | — | U |
| U | Layar dokter · prediksi, interval, peringatan, rekomendasi, sumber | — | V |
| V | `ClinicalDecisionLog` · jejak audit | — | — |

★ = tahap kontribusi (§0.4). Perhatikan pengklasifikasi kondisi muncul dua kali, sebagai E3
di sisi luring (pelatihannya) dan M di sisi daring (pemakaiannya) — itu satu komponen.

### 3.3 Hubungan antar-kelompok

Empat hubungan berikut menandai **artefak yang dipakai ulang**, bukan aliran data dalam satu
jalan, sehingga sebaiknya dibedakan dari panah biasa:

- E → K *(bundle model)*
- E2 → L *(faktor konformal)*
- E3 → M *(pengklasifikasi)*
- H → Q *(indeks leksikal dibangun dari potongan ChromaDB)*

### 3.4 Kotak keterangan waktu

```
Subtotal lokal (J + K + Q)   : 314,8 ms
Model bahasa (R)             : ± 9,3 detik
Total satu rekomendasi       : ± 9,6 detik
Jejak memori                 : 641 MB
Muat artefak, sekali di awal : 16,5 detik
```

### 3.5 Tiga hal yang HARUS terbaca

1. **Kotak O memakai nilai TERPREDIKSI**, bukan nilai terkini. Beri label pada hubungan
   K → O: **"nilai terprediksi"**. Bila kotak ini memakai nilai sekarang, seluruh sifat
   antisipatif sistem batal.
2. **Kotak R tidak pernah menerima nomor halaman.** Beri keterangan: *"konteks bertanda
   `[S1..Sn]` tanpa nomor halaman; nomor halaman baru muncul pada T dari metadata, sehingga
   model bahasa tidak dapat mengarangnya"*.
3. **Kotak V menyimpan potongan sumber yang persis dilihat dokter**, bukan hasil penelusuran
   mentah, sehingga keputusan dapat diaudit ke halaman dokumen di kemudian hari.

### 3.6 Yang TIDAK boleh ada

- Hubungan dari L (interval) ke P (pembentuk kueri) — lihat §0.5.
- Tahap pelatihan atau pengindeksan di dalam kelompok daring. Seluruhnya luring; pemisahan
  itulah yang membuat batasan komputasi dapat dipenuhi.

**Rujukan tambahan:** `docs/METHODOLOGY.md` §7.0 memuat tabel kontrak data 14 tahap, yakni
objek apa yang berpindah antar-tahap beserta tempat memeriksanya di kode.

---

## 4. `Gambar_IV4_ClassDiagram.png` — Diagram Kelas

Tercetak **Gambar IV.4**. Label naskah `fig:class-diagram`.

**Gambar lama BASI** — terverifikasi memuat `DigitalTwinStateManager` dan `WhatIfSimulator`,
yang kodenya **sudah dihapus** (direktori `src/digital_twin/` kosong), serta Random Forest
sebagai model utama.

**Keterangan gambar di naskah:** *"Diagram kelas komponen inti sistem, memisahkan domain logic
(prediksi, perakitan keadaan pasien, retrieval, generasi) dari lapisan antarmuka."*

Notasi UML. Tiga kelompok.

### 4.1 Kelompok data dan prakiraan

| Kelas | Isi |
|---|---|
| `ohio_parser` «modul» — `src/data/ohio_parser.py` | `+ parse_ohio_xml(xml_path): DataFrame`<br>`+ parse_ohio_fingerstick(xml_path): DataFrame`<br>`+ process_ohio_dataset(...)` menulis CSV |
| `DataPreprocessor` — `src/data/preprocessor.py` | `+ handle_missing_values(df)`<br>`+ engineer_features(df)`<br>`+ create_sequences(df, w, h)`<br>`+ split_by_patient(df, uji)`<br>`+ normalize_data(Xtr, Xte)` |
| `BaseGlucoseModel` «antarmuka» | `+ train(X, y)`<br>`+ predict(X)`<br>`+ save(path) / load(path)` |
| `GBMGlucoseModel` «produksi» | `- model: HistGradientBoostingRegressor`<br>`- model_q_low, model_q_high`<br>`+ train(X, y): dict`<br>`+ predict(X)`<br>`+ predict_std(X)`<br>σ = (q₀,₉₇₅ − q₀,₀₂₅) / 3,92 |
| `RandomForestGlucoseModel` «pembanding» | — |
| `LSTMGlucoseModel` «pembanding» | — |

### 4.2 Kelompok kontrak data dan kondisi klinis

| Kelas | Isi |
|---|---|
| `PatientState` «kontrak data» | `+ current_glucose, predicted_glucose`<br>`+ insulin_on_board, carbs_on_board`<br>`+ activity_level, stress_level`<br>*— diturunkan dari nilai TERPREDIKSI —*<br>`+ risk_level`: hipo \| normal \| hiper<br>`+ trend_direction, trend_rate`<br>`+ urgency`: critical \| high \| medium \| low<br>`+ to_rag_context(): dict`<br>`+ from_model_output(...)` |
| `alerts` «modul» — `src/alerts.py` | `+ evaluate_divergence(current_glucose, predicted_glucose, horizon_minutes)`<br>peringatan saat kondisi kini aman tetapi kondisi terprediksi tidak |
| `DivergenceAlert` «dataclass» — `src/alerts.py` | `from_class`, `to_class`, `from_label`, `to_label`<br>keluaran `evaluate_divergence()`; `None` bila kedua kategori sama |
| `ClinicalDecisionLog` — `src/clinical_state/decision_log.py` | `+ update_state(patient_id, updates)`<br>`+ log_intervention(patient_id, jenis, ringkasan)`<br>`+ get_events(patient_id)`<br>`+ save() / load()` → JSON, jejak audit |
| `StateRecord` «dataclass» | `patient_id`, `state`, `events` — wadah simpanan `ClinicalDecisionLog` |

### 4.3 Kelompok RAG

| Kelas | Isi |
|---|---|
| `PredictionConditionedQueryBuilder` | `+ build(state): query`<br>kueri disusun dari kondisi TERPREDIKSI |
| `RAGPipeline` — `src/rag/pipeline.py` | `+ _build_query(state, prediksi)`<br>`+ _retrieve(query, top_k=5)`<br>`+ answer(...)`<br>`+ _ensure_disclaimer(teks)` |
| `MMRRetriever` — `src/rag/retriever.py` | `- retrieval_mode`: vektor \| **bm25** \| hibrida<br>`+ retrieve(query, top_k)`<br>`- _peringkat_bm25(query, n)`<br>`- _gabung_rrf(daftar)`<br>**bm25 = jalur produksi** |
| `SimpleKeywordRetriever` «cadangan, KNF-04» | — |
| `MedicalKnowledgeBase` — `src/rag/knowledge_base.py` | `+ chunk_documents()` → 500/67, batas kalimat<br>`+ save_to_chroma()` |
| `RAGGenerator` — `src/rag/generator.py` | `gemini-3.5-flash-lite`, suhu 0,2, maks 700 token<br>`+ generate_advisory(...)`<br>`+ generate_explanation(...)`<br>`- _template_answer(...)` «cadangan» |
| `DiabetesAdvisorChain` — `src/rag/advisor_chain.py` | rantai LangChain yang benar-benar memanggil model bahasa<br>dibuat `RAGGenerator` pada `__init__` |
| `RetrievedDocument` «dataclass» — `src/rag/pipeline.py` | `rank`, `text`, `source`, `similarity`, `metadata`<br>kontrak keluaran penelusuran; **di sinilah metadata sitasi mengalir** |
| `citations` «modul» — `src/rag/citations.py` | `+ potong_batas_kalimat(teks, batas)`<br>kutipan berhenti di akhir kalimat (KNF-08) |
| ChromaDB «penyimpan» | 4.038 potongan · `all-MiniLM-L6-v2`, 384 dimensi |

### 4.4 Hubungan antar-kelas

| Dari | Ke | Jenis | Label |
|---|---|---|---|
| `ohio_parser` | `DataPreprocessor` | kebergantungan | — |
| `DataPreprocessor` | `BaseGlucoseModel` | asosiasi | — |
| `GBMGlucoseModel` | `BaseGlucoseModel` | realisasi | — |
| `RandomForestGlucoseModel` | `BaseGlucoseModel` | realisasi | — |
| `LSTMGlucoseModel` | `BaseGlucoseModel` | realisasi | — |
| `GBMGlucoseModel` | `PatientState` | asosiasi | **"prediksi + σ"** |
| `PatientState` | `alerts` | kebergantungan | — |
| `PatientState` | `ClinicalDecisionLog` | kebergantungan | — |
| `PatientState` | `PredictionConditionedQueryBuilder` | asosiasi | **"kontrak data"** |
| `PatientState` | `DivergenceAlert` | kebergantungan | lewat `evaluate_divergence()` |
| `RAGPipeline` | `PredictionConditionedQueryBuilder` | **kebergantungan** | builder dibuat lokal di dalam `_build_query()`, **bukan** disimpan sebagai atribut |
| `RAGPipeline` | `MMRRetriever` | komposisi | disimpan pada `self.retriever` |
| `RAGPipeline` | `RAGGenerator` | komposisi | disimpan pada `self.generator` |
| `RAGPipeline` | `SimpleKeywordRetriever` | kebergantungan | **"bila indeks gagal"** — pemilihan cadangan terjadi di `RAGPipeline.build()`, **bukan** di dalam `MMRRetriever` |
| `RAGPipeline` | `RetrievedDocument` | kebergantungan | keluaran `_retrieve()` |
| `RAGGenerator` | `DiabetesAdvisorChain` | komposisi | dibuat pada `__init__` |
| `MMRRetriever` | ChromaDB | asosiasi | **"baca potongan"** |
| `MedicalKnowledgeBase` | ChromaDB | asosiasi | — |
| Lapisan antarmuka | `citations` | kebergantungan | `app/streamlit_app.py` memanggil `build_source_list()`; `advisor_chain.py` memanggil `format_page_label()` |

**Yang harus terbaca:** `PatientState` adalah **satu-satunya** penghubung antara kelompok
prakiraan dan kelompok RAG. **Tidak boleh ada** hubungan langsung dari `GBMGlucoseModel` ke
`RAGPipeline` maupun ke `MMRRetriever`.

**Tiga panah yang PERNAH salah pada spesifikasi terdahulu, jangan diulang.** Ketiganya sudah
diperiksa ulang langsung ke kode, bukan dikira-kira:

1. **Tidak ada** panah `MMRRetriever` → `citations`. Berkas `src/rag/retriever.py` tidak pernah
   mengimpor `citations` sama sekali. Pemakainya adalah lapisan antarmuka dan `advisor_chain`.
2. **Tidak ada** panah `MMRRetriever` → `SimpleKeywordRetriever`. Pemilihan retriever cadangan
   terjadi di `RAGPipeline.build()`, sehingga panahnya berasal dari `RAGPipeline`.
3. Hubungan `RAGPipeline` → `PredictionConditionedQueryBuilder` adalah **kebergantungan**,
   bukan komposisi. Buildernya dibuat sesaat di dalam `_build_query()` lalu dibuang; ia tidak
   pernah menjadi atribut `RAGPipeline`. Yang benar-benar komposisi hanya `MMRRetriever` dan
   `RAGGenerator`, sebab keduanya disimpan pada `self`.

---

## 5. `Gambar_IV5_SequencePCRAG.png` — Diagram Sekuens

Tercetak **Gambar IV.5**. Label naskah `fig:sequence-pcrag`.

**Keterangan gambar di naskah:** *"Diagram sekuens satu siklus RAG terkondisi-prediksi, dari
permintaan dokter hingga rekomendasi tertelusur."*

**Sebelas pelaku, berurutan:** Dokter · Antarmuka Streamlit · `GBMGlucoseModel` ·
Pengklasifikasi kondisi · `PatientState` · `PredictionConditionedQueryBuilder` ·
`MMRRetriever` · ChromaDB · `RAGGenerator` · `citations` · `ClinicalDecisionLog`

**Dua puluh pesan:**

| # | Dari → Ke | Pesan |
|---|---|---|
| 1 | Dokter → Antarmuka | Pilih pasien, masukkan catatan *logbook* |
| 2 | Antarmuka → Antarmuka | Bangun jendela 12 baris terakhir, hitung 7 fitur |
| 3 | Antarmuka → `GBMGlucoseModel` | `predict(X)` |
| 4 | `GBMGlucoseModel` → Antarmuka | Δglukosa → nilai absolut, σ dari lebar antar-kuantil |
| 5 | Antarmuka → Antarmuka | Terapkan faktor konformal → interval 95% |
| 6 | Antarmuka → Pengklasifikasi | `predict_condition(X)` |
| 7 | Pengklasifikasi → Antarmuka | hipoglikemia / normal / hiperglikemia |
| 8 | Antarmuka → `PatientState` | `from_model_output(pred, kondisi)` |
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
| 20 | Dokter → `ClinicalDecisionLog` | Setujui / sesuaikan / tolak → `log_intervention(...)` |

**Dua hal yang harus terbaca:**

- Pesan 15 **tidak** membawa nomor halaman; nomor halaman baru muncul pada pesan 18. Beri
  keterangan: *"model bahasa tidak pernah melihat nomor halaman, sehingga tidak dapat
  mengarangnya"*.
- Pesan 12 ditegaskan: **"dari kondisi TERPREDIKSI"** — inilah kebaruannya.

---

## 6. `Gambar_IV6_Wireframe.png` — Wireframe Antarmuka

Tercetak **Gambar IV.6**. Label naskah `fig:wireframe`.

**Gambar lama BASI** — terverifikasi memuat bilah navigasi
`[ Input Logbook ] [ Prediksi ] [ What-If ] [ Twin Dashboard ]`. Dua tab terakhir **tidak
ada**; aplikasi sebenarnya punya **dua tab**: *Rekomendasi Klinis* dan *Catat Keputusan*.

**Keterangan gambar di naskah:** *"Rancangan wireframe antarmuka alat konsultasi klinis: panel
ringkasan keadaan pasien, prediksi dengan interval terkalibrasi, rekomendasi tertelusur, dan
pencatatan keputusan."*

**Sembilan bagian:**

1. **Bilah judul** — `Konsol Konsultasi Diabetes — Pasien: P001`

2. **Bilah navigasi** — `[ Input Logbook ]  [ Konsultasi ]` — hanya dua

3. **Panel INPUT LOGBOOK**
   ```
   Glukosa (mg/dL):  [______]
   Karbohidrat (g):  [______]
   Insulin (u):      [______]
   Aktivitas:        [______]
   Stres (0–10):     [______]

   [ Simpan & Prediksi ]
   ```

4. **Panel PRAKIRAAN**
   ```
   +30 mnt:  168 mg/dL   [interval 150–186]
   +60 mnt:  205 mg/dL   [interval 178–232]

   Risiko: HIPERGLIKEMIA (menuju)
   Tren: NAIK ↑     Kegentingan: SEDANG
   ```

5. **Panel PERINGATAN DIVERGENSI**
   > ⚠ Kondisi terkini masih normal (142), tetapi kondisi terprediksi menuju hiperglikemia
   > (205) dalam 60 menit.

6. **Panel REKOMENDASI KLINIS** — antisipatif, dibumikan pada dokumen
   ```
   •  [teks nasihat berbahasa Indonesia …]

   Sumber: PERKENI-2021, hlm. XX  ·  ADA-SOC-2025, hlm. XX
   ```

7. **Panel RUJUKAN PANDUAN MEDIS** — potongan yang benar-benar ditelusur
   ```
   1.  PERKENI-2021 · hlm. 42 — "…kutipan berhenti pada batas kalimat…"
   2.  ADA-SOC-2025 · hlm. 118 — "…"

   Tiap potongan menyimpan identitas dokumen dan nomor halamannya
   ```

8. **Bilah KEPUTUSAN DOKTER**
   `[ ✓ Setujui ]   [ ✎ Sesuaikan ]   [ ✗ Tolak ]  →  dicatat sebagai jejak audit;
   diteruskan hanya setelah validasi dokter`

9. **Catatan kaki** — *"Nilai pada wireframe hanya ilustrasi tata letak."*

**Yang harus terbaca:** bagian 7 memperlihatkan **nomor halaman** dan kutipan yang **berhenti
di akhir kalimat**. Keduanya bukti KNF-08, sehingga bagian ini harus tampil utuh, jangan
disingkat.

---

## 7. Daftar periksa isi

- [ ] Tidak ada istilah dari daftar larangan §0.1, kecuali sebagai pembanding yang dilabeli
- [ ] Seluruh angka cocok dengan tabel §0.2, dan tidak ada angka tambahan
- [ ] Empat TAHAP kontribusi ditandai (§0.4), hanya pada §2 dan §3; pada §3 itu berarti
      lima kotak, karena pengklasifikasi kondisi muncul dua kali (E3 luring, M daring)
- [ ] Angka ilustrasi wireframe §6 dipakai apa adanya, tidak dicocokkan ke §0.2
- [ ] Tidak ada hubungan dari kotak interval ke kotak pembentuk kueri (§0.5)
- [ ] Nama kelas, berkas, pustaka, dan model ditulis persis seperti di kode
- [ ] Nama berkas keluaran persis seperti tabel §0.3, keenamnya
- [ ] Pada §3 dan §5: nomor halaman tidak pernah sampai ke tahap pembangkitan
- [ ] Pada §4: tidak ada hubungan langsung model prakiraan → `RAGPipeline` / `MMRRetriever`
