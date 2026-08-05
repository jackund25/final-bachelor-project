# AUDIT KODE TERHADAP KEBUTUHAN LAPORAN TA

> Audit dilakukan **hanya dengan membaca kode**, setelah tujuh tugas perbaikan selesai
> (commit `83aebc9` … `b069cbc`). **Tidak ada kode yang diubah.**
>
> Status yang dipakai:
> **ADA** — diimplementasikan dan sesuai deskripsi ·
> **SEBAGIAN** — ada tetapi tidak sesuai deskripsi ·
> **TIDAK ADA** — belum diimplementasikan ·
> **TIDAK DITEMUKAN** — tidak dapat dipastikan dari kode
>
> Tanggal audit: 5 Agustus 2026 · Cabang: `refaktor-tujuh-tugas`

---

## 1. Tabel Ringkas

### Kebutuhan Fungsional

| Kode | Status | Catatan |
|---|---|---|
| KF-01 | **SEBAGIAN** | Kelima variabel dapat diisi, tetapi datanya tidak pernah sampai ke jalur prediksi |
| KF-02 | **SEBAGIAN** | Besaran turunan hanya dihitung pada pipeline dataset, bukan pada masukan manual |
| KF-03 | **SEBAGIAN** | Aplikasi hanya menyajikan horizon 30 menit; bundle 60 menit ada tetapi tidak dipakai |
| KF-04 | **SEBAGIAN** | Kueri utama memakai prediksi, tetapi `_enhance_query()` menambah tag dari `current_glucose` |
| KF-05 | **ADA** | Retriever dapat dipanggil terpisah; mengembalikan metadata dokumen dan halaman |
| KF-06 | **SEBAGIAN** | Instruksi bahasa ada di prompt; tidak ada pemeriksaan keluaran |
| KF-07 | **SEBAGIAN** | Divergensi berbasis kategori (benar), tetapi hanya satu arah |
| KF-08 | **ADA** | Pencatatan keputusan dokter lengkap, termasuk sumber rujukan yang ditampilkan |

### Kebutuhan Nonfungsional

| Kode | Status | Catatan |
|---|---|---|
| KNF-01 | **ADA** | Terukur pada laptop tanpa GPU |
| KNF-02 | **SEBAGIAN** | Sitasi dapat ditelusuri; kontribusi fitur **tidak** ditampilkan di antarmuka |
| KNF-03 | **SEBAGIAN** | Hanya temperature rendah + disclaimer; tidak ada penyaring angka dosis |
| KNF-04 | **ADA** | Tiga lapisan cadangan terverifikasi |
| KNF-05 | **SEBAGIAN** | Sama dengan KF-06 — dijamin instruksi, tidak diperiksa |
| KNF-06 | **ADA** | Disclaimer dipaksa di setiap keluaran |
| KNF-07 | **SEBAGIAN** | Ambang klinis sudah terpusat, tetapi konstanta konformal dan logika peringatan masih di `app/` |
| KNF-08 | **ADA** | Metadata halaman cetak tersimpan dan terverifikasi ke PDF sumber |
| KNF-09 | **ADA** | Seluruh komponen sumber terbuka atau tier gratis |
| KNF-10 | **TIDAK ADA** | Nol instrumentasi waktu tanggap ujung-ke-ujung |

### Janji Laporan yang Menuntut Skrip Evaluasi

| Kode | Status | Catatan |
|---|---|---|
| J1 | **SEBAGIAN** | Skrip ada dan berjalan, tetapi tidak tersegmentasi sehingga tidak sebanding dengan RF produksi |
| J2 | **SEBAGIAN** | Cakupan diukur agregat; **tidak** dipecah per rentang glukosa |
| J3 | **TIDAK ADA** | Nol skrip memvariasikan konstanta waktu karbohidrat |
| J4 | **ADA** | Perbandingan kuantitatif tersimpan dan sudah diregenerasi pascaperubahan |
| J5 | **ADA** | Ablasi berjalan di korpus baru dengan Hit@k dan MRR |
| J6 | **ADA** | Terukur: Hit@k **jenuh 100%** pada mode conditioned sejak k=3 |
| J7 | **ADA** | Membandingkan dua representasi, bukan satu angka tunggal |
| J8 | **ADA** | Rincian zona A–E tersimpan, bukan hanya A+B |
| J9 | **ADA** | Implementasi dapat disalin; ada perbedaan dari definisi baku yang perlu ditulis |

---

## 2. Uraian per Butir

### KF-01 — Masukan logbook terstruktur · SEBAGIAN

Kelima variabel **memang dapat diisi** lewat antarmuka
([1_Input_Logbook.py:50-55](../app/pages/1_Input_Logbook.py#L50-L55)):

```python
glucose  = st.number_input("Glukosa (mg/dL)", 40.0, 400.0, 110.0, 1.0)
carbs    = st.number_input("Karbohidrat (g)", 0.0, 200.0, 30.0, 1.0)
insulin  = st.number_input("Insulin (unit)", 0.0, 30.0, 3.0, 0.1)
activity = st.number_input("Aktivitas (menit)", 0, 240, 15, 5)
stress   = st.slider("Tingkat Stres", 1, 10, 5)
```

**Mengapa bukan ADA.** Data itu ditulis ke `data/raw/manual_logbook.csv`
([:16](../app/pages/1_Input_Logbook.py#L16)), sedangkan halaman prediksi memuat sumber
lain ([streamlit_app.py:35](../app/streamlit_app.py#L35)):

```python
return DiabetesDataLoader("data/raw").load_preferred_dataset("ohio_t1dm", "latest_generated")[0]
```

`manual_logbook` hanya dapat dimuat lewat `_load_by_source("manual_logbook")`
([loader.py:83-84](../src/data/loader.py#L83-L84)) yang **tidak pernah dipanggil** dari
`app/`. Isi `data/raw/` saat ini hanya `ohio_t1dm_merged.csv` dan `ohio_t1dm_smbg.csv` —
berkas logbook bahkan belum pernah terbentuk.

**Selisih terhadap deskripsi:** sistem menerima masukan, tetapi masukan itu tidak
menjadi masukan sistem. Yang ada adalah formulir pencatatan, bukan jalur data.

**Catatan satuan:** label antarmuka menulis "Aktivitas (menit)", sedangkan kolom
`activity` pada dataset berisi **skor intensitas** kanal `exercise` OhioT1DM. Dua hal
berbeda dengan nama sama.

---

### KF-02 — Besaran turunan berbasis fisiologi · SEBAGIAN

Keempat besaran memang dihitung
([preprocessor.py:105-110](../src/data/preprocessor.py#L105-L110)): `iob`, `cob`,
`glucose_delta`, `hour_sin`/`hour_cos`.

**Mengapa bukan ADA.** Perhitungan hanya berjalan pada data dataset
([streamlit_app.py:90-98](../app/streamlit_app.py#L90-L98)):

```python
def build_window(patient_df, art):
    seq = art["sequence_length"]
    if art["use_engineered"]:
        from src.data.preprocessor import DataPreprocessor
        feat_df = DataPreprocessor({}).engineer_features(patient_df, **art["feature_engineering"])
    ...
```

`patient_df` berasal dari `load_dataset()` (OhioT1DM). **Tidak ada jalur** yang
memanggil `engineer_features()` atas masukan manual pengguna. Sesuai kriteria yang
diminta ("kalau hanya pada pipeline dataset, itu SEBAGIAN"), statusnya SEBAGIAN.

---

### KF-03 — Prediksi 30 dan 60 menit beserta selang ketidakpastian · SEBAGIAN

Aplikasi hanya memuat **satu** bundle ([streamlit_app.py:66](../app/streamlit_app.py#L66)):

```python
bf = Path("models/rf_inference_bundle.pkl")   # = horizon 6 (30 menit)
```

Seluruh tampilan memakai satu horizon
([:149-150](../app/streamlit_app.py#L149-L150)):

```python
horizon_min = art["horizon"] * 5
st.caption(f"Horizon prediksi: **+{horizon_min} menit**")
```

Selang konformal juga hanya untuk horizon itu ([:193-195](../app/streamlit_app.py#L193-L195)).

**Yang tersedia tetapi tidak dipakai:** `models/rf_inference_bundle_h12.pkl` dan
`models/rf_metrics_h12.json` ada dan terkini (RMSE 32,729 · Clarke A+B 87,82%).

**Selisih terhadap deskripsi:** deskripsi menuntut dua horizon **beserta** selangnya;
aplikasi menyajikan satu horizon beserta selangnya.

---

### KF-04 — Kueri dibangun dari kadar terprediksi · SEBAGIAN

**Jalur utama benar.** `RAGPipeline._build_query()` membentuk `PatientState` dari
`prediction`, lalu `_primary_query()` menyusun kueri dari `state.predicted_glucose`
([conditioned_query.py:122-128](../src/rag/conditioned_query.py#L122-L128)):

```python
parts.append(
    f"Prediksi glukosa {state.prediction_horizon_minutes} menit ke depan: "
    f"{state.predicted_glucose:.1f} mg/dL "
    f"(dari {state.current_glucose:.1f} mg/dL, "
    f"perubahan {state.glucose_delta:+.1f} mg/dL, tren {state.trend_label})."
)
```

Kemunculan `current_glucose` di sini **sah** — ia hanya konteks naratif "dari X ke Y",
sedangkan yang menentukan kondisi adalah `predicted_glucose`.

**Yang membuatnya SEBAGIAN: ada tahap kedua yang memakai `current_glucose` untuk
menentukan kondisi.** Setelah kueri utama tersusun, `RAGPipeline._retrieve()` memanggil
`retrieve_with_context()` ([pipeline.py:281-284](../src/rag/pipeline.py#L281-L284)), yang
menambahkan tag lewat `_enhance_query()`
([retriever.py:218-234](../src/rag/retriever.py#L218-L234)):

```python
def _enhance_query(self, query: str, patient_state: Dict[str, Any]) -> str:
    from src.constants import CLASS_NORMAL, classify_glucose_3class

    tags: List[str] = []
    glucose = float(patient_state.get("current_glucose", 0.0))   # <-- KADAR SAAT INI
    stress = int(patient_state.get("stress_level", 0))

    kondisi = classify_glucose_3class(glucose)                    # <-- KONDISI DARI KADAR SAAT INI
    if kondisi != CLASS_NORMAL:
        tags.append(kondisi)

    if stress >= 7:
        tags.append("stress tinggi")

    if not tags:
        return query
    return f"{query}. Konteks pasien: {', '.join(tags)}."
```

**Selisih terhadap deskripsi.** Deskripsi menuntut kueri dibangun dari kadar
**terprediksi, bukan dari kadar saat ini**. Kode menambahkan frasa kondisi yang
diturunkan dari kadar **saat ini** ke ujung setiap kueri produksi.

**Dampak praktis pada kasus divergen** — justru kasus yang menjadi klaim inti laporan.
Bila kadar kini 150 mg/dL (normal) dan prediksi 214 mg/dL (hiperglikemia), tag tidak
ditambahkan karena kondisi kini normal, sehingga tidak ada kerusakan. Tetapi bila kadar
kini 190 mg/dL (hiper) dan prediksi 65 mg/dL (hipo), kueri akan berakhir dengan
`"Konteks pasien: hiperglikemia."` — **menarik retrieval ke arah yang berlawanan dengan
kondisi yang diantisipasi.**

Perlu dicatat jujur: seluruh angka retrieval pada `results/` dihitung lewat skrip
evaluasi yang memanggil `retriever.retrieve()` **langsung**, bukan
`retrieve_with_context()`. Jadi **angka evaluasi tidak terdampak**; yang terdampak
adalah **perilaku aplikasi**. Ini justru berarti angka laporan tidak mencerminkan
perilaku produksi pada titik ini.

---

### KF-05 — Penelusuran mengembalikan potongan beserta penunjuk dokumen dan halaman · ADA

Lapisan penelusuran **dapat dipanggil dan diuji terpisah** dari pembangkitan.
`MMRRetriever` berdiri sendiri di [src/rag/retriever.py](../src/rag/retriever.py) dan
dikonstruksi langsung oleh **8 skrip evaluasi** tanpa menyentuh lapisan generasi
(`ablation_rag_fullkb.py`, `eval_retrieval_crossfold.py`, `eval_retrieval_realcases.py`,
`sensitivity_analysis.py`, `benchmark_deployability.py`, `eval_embedding_alternative.py`,
`ragas_eval.py`, `run_ragas.py`).

Nilai kembaliannya memuat penunjuk dokumen dan halaman
([retriever.py:193-204](../src/rag/retriever.py#L193-L204)): `rank`, `text`, `source`,
`similarity`, dan `metadata` yang berisi `kb_id`, `nama_dokumen`, `lembaga`, `tahun`,
`halaman_pdf`, `halaman_cetak`, `halaman_cetak_valid`.

Pemisahan KF-05/KF-06 untuk evaluasi terpisah pada Bab VI **terpenuhi**.

---

### KF-06 — Rekomendasi berbahasa Indonesia yang dibumikan · SEBAGIAN

**Instruksi ada** ([prompts.py](../src/rag/prompts.py), `SYSTEM_PROMPT` aturan 6):

```
6. Gunakan Bahasa Indonesia yang ringkas, jelas, dan actionable.
```

Pembumian juga ada: blok konteks berisi potongan hasil penelusuran, aturan 1 mewajibkan
menjawab berdasarkan konteks, dan aturan 3 mewajibkan menyatakan bila konteks kosong.

**Mengapa bukan ADA — tidak ada pemeriksaan keluaran.** Pencarian di seluruh
`src/rag/` tidak menemukan satu pun deteksi bahasa, pemeriksaan pasca-generasi, atau
percobaan ulang bila keluaran bukan Bahasa Indonesia. Satu-satunya pemeriksaan
pasca-generasi adalah penyisipan disclaimer
([pipeline.py:397-401](../src/rag/pipeline.py#L397-L401)).

**Risiko nyata pada korpus sekarang.** Delapan dari dua belas dokumen berbahasa Inggris
(seluruh ISPAD, ADA-EASD, ATTD). Konteks yang masuk ke prompt karena itu sering
berbahasa Inggris, sementara jaminan keluaran Bahasa Indonesia hanya bersandar pada
kepatuhan model terhadap satu baris instruksi.

---

### KF-07 — Kondisi terstruktur, peringatan hipoglikemia dini, dan peringatan divergensi · SEBAGIAN

**Kondisi terstruktur: ADA.** `PatientState` menghasilkan `risk_level`, `risk_label`,
`trend_direction`, `trend_label`, `trend_rate`, dan `urgency`
([patient_state.py:93-101](../src/patient_state.py#L93-L101)).

**Peringatan hipoglikemia dini: ADA**, dua sinyal
([streamlit_app.py:223-229](../app/streamlit_app.py#L223-L229)).

**Peringatan divergensi: SEBAGIAN.** Kode
([streamlit_app.py:206-209](../app/streamlit_app.py#L206-L209)):

```python
# Peringatan divergen (current normal tapi prediksi bahaya) — nilai jual sistem
if cur_label == "Dalam Target" and pred_label != "Dalam Target":
    st.warning(f"⚠️ **Antisipasi:** kondisi saat ini normal, namun glukosa diprediksi menuju "
               f"**{pred_label}** ({pred:.0f} mg/dL) dalam {horizon_min} menit. Pertimbangkan tindakan pencegahan.")
```

`cur_label` dan `pred_label` berasal dari `classify_glucose()`
([streamlit_app.py:171-172](../app/streamlit_app.py#L171-L172)), yang mengembalikan
**kategori** (`"Hipoglikemia"` / `"Dalam Target"` / `"Hiperglikemia"`).

**Klarifikasi penting: ini BUKAN kasus ambang selisih.** Perbandingannya memang antar
**kategori**, bukan `|delta| > 10 mg/dL`. Ambang selisih memang ada di kode, tetapi
untuk keperluan lain — penunjuk tren ([:196](../app/streamlit_app.py#L196)) — dan tidak
dipakai oleh peringatan divergensi.

**Selisih yang membuatnya SEBAGIAN: implementasinya asimetris.** Definisi Tabel III.1:

> "Peringatan divergensi diberikan ketika kondisi terprediksi **berbeda kategori** dari
> kondisi yang sedang berlaku, misalnya kadar glukosa terkini berada dalam rentang
> sasaran sedangkan kondisi terprediksi tergolong hipoglikemia."

Kata "misalnya" menandakan kasus rentang-sasaran adalah **contoh**, bukan keseluruhan
definisi. Kode hanya menyala bila kondisi kini **tepat** `"Dalam Target"`:

| Kondisi kini | Kondisi terprediksi | Berbeda kategori? | Peringatan menyala? |
|---|---|---|---|
| Dalam Target | Hipoglikemia | ya | **ya** |
| Dalam Target | Hiperglikemia | ya | **ya** |
| Hiperglikemia | Hipoglikemia | ya | **tidak** |
| Hipoglikemia | Hiperglikemia | ya | **tidak** |
| Hiperglikemia | Dalam Target | ya | **tidak** |
| Hipoglikemia | Dalam Target | ya | **tidak** |

Empat dari enam transisi antar-kategori tidak tertangkap. Baris ketiga
(hiper → hipo) justru transisi paling berbahaya secara klinis.

---

### KF-08 — Alur doctor-mediated · ADA

Tab "Catat Keputusan" ([streamlit_app.py:315](../app/streamlit_app.py#L315)) memakai
`ClinicalDecisionLog` ([src/clinical_state/decision_log.py](../src/clinical_state/decision_log.py)).

Jenis keputusan yang dapat dicatat: `tinjauan`, `setujui rekomendasi`,
`sesuaikan rekomendasi`, `tolak`. Payload menyimpan `rag_sources` — **baris sumber yang
benar-benar ditampilkan kepada dokter**, termasuk `page_label` yang sudah diresolusi —
sehingga keputusan dapat ditelusuri ke halaman dokumen di kemudian hari. Dijaga tes
`test_decision_log_persists_rag_sources_for_audit`.

Setiap keluaran juga memuat `doctor_review_required: True`
([pipeline.py:373](../src/rag/pipeline.py#L373)).

---

### KNF-01 — Keterterapan tanpa akselerator grafis · ADA

[scripts/benchmark_deployability.py](../scripts/benchmark_deployability.py) mengukur
waktu rekayasa fitur, prediksi RF, pengklasifikasi, pemuatan retriever, dan retrieval,
seluruhnya di CPU. Embedding dikonfigurasi `device="cpu"`
([knowledge_base.py:39](../src/rag/knowledge_base.py#L39)).

Terkonfirmasi pula secara empiris selama refaktor: seluruh pelatihan dan evaluasi
dijalankan pada AMD Ryzen 5 5600H (6 inti / 12 thread, 15,4 GB RAM) **tanpa GPU**.

---

### KNF-02 — Keterjelasan · SEBAGIAN

**Penelusuran sitasi: ADA.** Setiap rekomendasi dapat ditelusuri ke dokumen dan halaman
(lihat KNF-08).

**Kontribusi fitur: tidak ditampilkan.** `scripts/eval_feature_importance.py` menghitung
dan menyimpannya ke `results/eval_prediksi/feature_importance.json`, tetapi pencarian
kata `importance` dan `kontribusi` di seluruh `app/` mengembalikan **nol hasil**.

**Selisih terhadap deskripsi:** deskripsi menuntut prediksi "dapat dijelaskan lewat
kontribusi fitur". Sesuai kriteria yang diminta ("kalau hanya tersimpan, itu SEBAGIAN"),
statusnya SEBAGIAN. Dokter tidak pernah melihat angka itu.

---

### KNF-03 — Keamanan konten · SEBAGIAN

Yang ada:
- `temperature = 0.2` dari config ([config.yaml](../config.yaml) `rag.llm.temperature`)
- `max_tokens = 700` kini benar-benar diterapkan
- Disclaimer dipaksa ([pipeline.py:397-401](../src/rag/pipeline.py#L397-L401))
- Aturan prompt melarang menulis nomor halaman/bab/tabel
- Daftar tindakan tidak berasal dari LLM melainkan aturan deterministik
  ([pipeline.py:378-395](../src/rag/pipeline.py#L378-L395))

**Yang tidak ada: penyaring keluaran.** Pencarian pola pemeriksaan angka dosis di
`src/rag/pipeline.py` mengembalikan **nol hasil**. Tidak ada regex, tidak ada
pencocokan silang angka terhadap konteks, tidak ada penolakan keluaran.

**Selisih terhadap deskripsi:** deskripsi menyebut "parameter pembangkitan konservatif
untuk menekan pemunculan angka dosis atau ambang yang tidak berdasar". Parameter
konservatif memang ada, tetapi **penekanannya sepenuhnya bersifat probabilistik** —
tidak ada mekanisme yang benar-benar memeriksa apakah angka yang muncul berdasar.

`scripts/eval_generation_safety.py` mengevaluasi keamanan keluaran secara
*post-hoc* pada tahap penelitian, bukan sebagai penyaring runtime.

---

### KNF-04 — Ketahanan tiga lapisan · ADA

**Lapisan data.** PDF gagal ekstrak, tidak terdaftar di manifest, jumlah halaman tidak
cocok, atau teks terlalu sedikit → `reingest_kb.py` **menghentikan proses** dengan pesan
eksplisit. Tidak ada skip diam-diam. Pada `handle_missing_values()`, runtun NaN melebihi
batas → barisnya dibuang, bukan diisi paksa.

**Lapisan penelusuran.** ChromaDB/embedding gagal diinisialisasi → `MMRRetriever`
menandai dirinya tidak siap, dan `RAGPipeline.build()` beralih ke `SimpleKeywordRetriever`
([pipeline.py:193](../src/rag/pipeline.py#L193)). Perlu dicatat: fallback ini juga
menurunkan korpus dari 2.061 chunk ke ~30 chunk `manual_kb` saja. Bila penilaian skor
gagal, retrieval tetap jalan tanpa skor (`similarity = None`).

**Lapisan pembangkitan.** Inisialisasi LLM gagal → `_chain = None`
([advisor_chain.py:93](../src/rag/advisor_chain.py#L93)). Pemanggilan gagal → tertangkap
([:132](../src/rag/advisor_chain.py#L132)) dan beralih ke `_template_answer()`
([:171](../src/rag/advisor_chain.py#L171)). Sitasi tetap dikembalikan, disclaimer tetap
disisipkan, daftar tindakan tetap ada. Di lapisan UI, kegagalan ditangkap dan hanya blok
rekomendasi yang hilang — prediksi dan peringatan klinis tetap tampil.

---

### KNF-05 — Bahasa Indonesia · SEBAGIAN

Sama dengan KF-06. Antarmuka, label, dan template cadangan seluruhnya Bahasa Indonesia
dan terjamin karena hardcoded. Yang tidak terjamin adalah **keluaran LLM**, yang hanya
bersandar pada instruksi prompt tanpa pemeriksaan.

---

### KNF-06 — Keamanan klinis, dokter penentu akhir · ADA

Tiga lapis penegasan:
1. `_ensure_disclaimer()` memaksa kalimat "keputusan medis final tetap pada dokter"
   pada setiap keluaran, apa pun sumbernya
2. `disclaimer_footer()` di setiap halaman ([ui.py](../app/ui.py))
3. `doctor_review_required: True` pada setiap advisory

Ditambah caption di bawah rekomendasi yang menyatakan sumbernya dan bahwa keputusan
akhir pada dokter.

---

### KNF-07 — Pemisahan logika domain dan antarmuka · SEBAGIAN

**Yang sudah benar (Tugas 3).** Ambang glikemik terpusat di
[src/constants.py](../src/constants.py). `app/ui.py` dan `app/streamlit_app.py`
mengimpor dari sana; tidak ada angka ambang yang ditulis ulang.

**Yang masih tertanam di `app/`:**

1. **Konstanta konformal, terduplikasi**
   ([streamlit_app.py:193](../app/streamlit_app.py#L193) dan
   [:221](../app/streamlit_app.py#L221)):

```python
CONFORMAL_K = 3.3
lo, hi = pred - CONFORMAL_K * pred_std, pred + CONFORMAL_K * pred_std
...
lo95, hi95 = pred - 3.3 * pred_std, pred + 3.3 * pred_std
```

   Angka yang sama ditulis dua kali di dua tempat, tidak dibaca dari config maupun dari
   `results/eval_prediksi/conformal.json` yang menghasilkannya.

2. **Aturan keputusan peringatan** — logika divergensi
   ([:207](../app/streamlit_app.py#L207)) dan peringatan hipoglikemia dini
   ([:223-229](../app/streamlit_app.py#L223-L229)) seluruhnya berada di lapisan
   antarmuka, bukan di modul domain. Akibatnya keduanya **tidak dapat diuji tanpa
   menjalankan Streamlit** dan tidak dapat dipakai ulang oleh skrip evaluasi.

---

### KNF-08 — Keterlacakan sitasi · ADA

Setiap chunk dokumen pedoman menyimpan `kb_id`, `source_id`, `nama_dokumen`, `lembaga`,
`tahun`, `judul_lengkap`, `halaman_pdf`, `halaman_cetak`, `halaman_cetak_valid`.

`halaman_cetak = halaman_pdf + offset`, dengan offset dari
[manifest.csv](../data/knowledge_base/manifest.csv).

**Terverifikasi ke PDF sumber** pada lima chunk yang mencakup ketiga kasus offset —
negatif (KB-02, −13), besar (KB-11, +1340), dan nol (KB-07). Pada dokumen Indonesia,
nomor cetak ikut terekstrak ke teks dan cocok dengan nilai yang dihitung: KB-02 halaman
9, KB-04 halaman 79, 64, dan 41.

Ditampilkan ke dokter lewat expander "Sumber Rujukan", dan **nomor halaman tidak pernah
berasal dari LLM** — blok konteks yang dikirim ke model sudah tidak memuat nomor halaman
sama sekali (terverifikasi: 0 penyebutan).

---

### KNF-09 — Biaya dan keterpindahan · ADA

Seluruh komponen sumber terbuka: scikit-learn, ChromaDB, LangChain,
sentence-transformers, Streamlit. Basis vektor tertanam (ChromaDB persist ke direktori
lokal), tanpa server khusus. LLM memakai tier gratis Gemini
(15 RPM / 1.000 RPD untuk `gemini-2.5-flash-lite`), dengan jalur alternatif Ollama lokal
dan mode `template` yang berjalan tanpa LLM sama sekali.

---

### KNF-10 — Waktu tanggap · TIDAK ADA

**Nol instrumentasi waktu di `app/`.** Pencarian `time.perf_counter`, `time.time()`,
`elapsed`, dan `latency` di seluruh direktori `app/` mengembalikan **nol hasil**.

Yang ada hanyalah pengukuran **per komponen** di
[scripts/benchmark_deployability.py](../scripts/benchmark_deployability.py)
(baris 46, 121, 126, 131, 197): rekayasa fitur, prediksi RF, pengklasifikasi, pemuatan
retriever, retrieval, dan satu panggilan LLM — diukur terpisah pada skrip benchmark,
bukan pada alur aplikasi sebenarnya.

**Selisih terhadap deskripsi:** yang dibutuhkan Bab VI adalah waktu tanggap
**ujung ke ujung, dari masukan sampai rekomendasi tampil**. Menjumlahkan komponen tidak
setara: ia mengabaikan pemuatan data, pembangunan window, render Streamlit, dan overhead
sesi.

**Besar pekerjaan: sedang (1–4 jam).** Perlu penanda waktu di sekitar alur
`build_window → predict_next → predict_condition → load_rag().answer() → render`,
disimpan ke `results/` agar dapat diagregasi, plus tampilan opsional di antarmuka.

---

### J1 — Ketahanan pada pemantauan mandiri periodik · SEBAGIAN

[scripts/eval_smbg_deployment.py](../scripts/eval_smbg_deployment.py) ada, sintaksnya
sah, dan hasilnya tersimpan di `results/eval_prediksi/smbg_deployment.json`. Ia memakai
kanal `finger_stick` nyata, bukan downsampling CGM, dan horizonnya dinyatakan dalam
menit — sesuai janji Tujuan 1.

**Mengapa SEBAGIAN — tidak lagi sebanding dengan RF produksi.** Skrip ini **tidak
memanggil** `create_sequences()`; ia membangun jendelanya sendiri
([:58](../scripts/eval_smbg_deployment.py#L58) `build_smbg_frame`). Karena itu ia
**tidak terdampak** perubahan Tugas 5 — dan justru karena itu pula ia **tidak
tersegmentasi**, sedangkan RF produksi kini dilatih atas jendela tersegmentasi.

Angka SMBG (RMSE 27,09 · Clarke A+B 92,09%) karena itu dihasilkan dengan perlakuan data
yang berbeda dari angka RF CGM terkini (RMSE 21,12 · Clarke A+B 94,86%). Menyandingkan
keduanya di Bab VI tanpa catatan akan menyesatkan.

**Baseline persistence: TIDAK DITEMUKAN.** Tidak ditemukan pembanding persistence di
dalam skrip ini. Perlu diperiksa ulang isi berkas hasilnya sebelum diklaim di laporan.

---

### J2 — Kalibrasi ketidakpastian · SEBAGIAN

[scripts/conformal_calibration.py](../scripts/conformal_calibration.py) **memang
mengukur cakupan empiris**, bukan sekadar menghitung interval
([:84-104](../scripts/conformal_calibration.py#L84-L104)):

```python
def coverage(lo, hi):
    return float(np.mean((yte >= lo) & (yte <= hi)) * 100)
...
"conformal_normalized": {"coverage%": round(coverage(lo_n, hi_n), 1), ...}
```

Diukur untuk dua tingkat kepercayaan (90% dan 95%) dan tiga varian interval.

**Mengapa SEBAGIAN — cakupannya agregat, tidak dipecah per rentang glukosa.**
`coverage()` menghitung rata-rata atas **seluruh** himpunan uji. Tidak ada pengelompokan
per rentang kadar.

Ini persis yang dipersoalkan Subbab II.4.4: jaminan konformal bersifat **marginal**,
berlaku rata-rata dan tidak dijamin seragam per subkelompok — dan rentang paling kritis
secara klinis justru paling jarang muncul. Data mendukung kekhawatiran itu: hipoglikemia
hanya 3,28% dari dataset. Cakupan agregat 96,5% dapat sepenuhnya ditopang rentang normal
sementara rentang hipoglikemia jauh di bawah nominal — **dan kode saat ini tidak dapat
membedakannya.**

`scripts/eval_hypo_uncertainty.py` mengukur sensitivitas/spesifisitas/PPV pada ambang
<70 mg/dL, tetapi **bukan cakupan interval konformal** per rentang.

**Besar pekerjaan: sedang (1–4 jam).** Perlu penambahan pengelompokan `y_true` ke
rentang (<54, 54–70, 70–180, 180–250, >250) lalu menghitung `coverage()` per kelompok.

---

### J3 — Sensitivitas konstanta waktu karbohidrat · TIDAK ADA

Pencarian `carbs_tau_min` dan `tau_carb` di seluruh `scripts/` mengembalikan **nol
berkas**. `sensitivity_analysis.py` hanya menyapu `top_k` dan `chunk_size` — keduanya
parameter retrieval, bukan parameter fisiologi.

Nilai `carbs_tau_min: 180` ditetapkan di [config.yaml](../config.yaml) dan mengalir ke
`engineer_features()`, tetapi **tidak pernah divariasikan** dalam eksperimen apa pun.

**Selisih terhadap deskripsi:** Subbab II.3.5 menyatakan pengaruh pemilihan nilainya
"diuji pada Bab VI". Uji itu belum ada.

**Besar pekerjaan: sedang (1–4 jam).** Perlu skrip yang memvariasikan `carbs_tau_min`
(mis. 90/135/180/240/300 menit), melatih ulang RF untuk tiap nilai, dan melaporkan RMSE.
Biaya komputasi: ~16 menit per nilai berdasarkan pengukuran Tugas 5, jadi lima nilai
≈ 80 menit berjalan.

---

### J4 — Pengklasifikasi vs pengambangan regresi · ADA

`results/eval_prediksi/condition_classifier.json` ada dan **sudah diregenerasi
5 Agustus** setelah perubahan Tugas 5, jadi angkanya mencerminkan kode terkini.

Perbandingan kuantitatifnya lengkap:

| Pendekatan | Sensitivitas hipoglikemia | PPV | Akurasi |
|---|---|---|---|
| Regresi lalu ambang | 15,4% | 52,4% | 89,3% |
| Pengklasifikasi 3 kelas | **45,2%** | 40,9% | 88,2% |

Ditambah perincian pada kasus divergen (regresi 36,5% vs pengklasifikasi 29,0%) yang
menunjukkan **keterbatasan** pendekatan itu — berguna untuk pembahasan yang seimbang.

Klaim Subbab III.4.2 ("jauh lebih baik") **didukung** untuk sensitivitas hipoglikemia
(naik ~3×), tetapi laporan sebaiknya menyebut ongkosnya: PPV turun dan akurasi
keseluruhan sedikit turun.

---

### J5 — Bukti empiris pengondisian prediksi · ADA

[scripts/ablation_rag_fullkb.py](../scripts/ablation_rag_fullkb.py) berjalan di korpus
baru (terkonfirmasi: header mencetak 2.061 chunk) dan metriknya mencakup Hit@1, Hit@5,
MRR, precision@5, dan nDCG@5 — sesuai Subbab II.6.5.

Hasil terkini (`results/baseline_ablation_fullkb_kb12/`, horizon 30 menit):

| Metrik | Conditioned | Standard |
|---|---|---|
| targets_needed | 100,0% | 0,0% |
| Hit@1 | 33,3% | 50,0% |
| Hit@5 | 100,0% | 50,0% |
| MRR | 0,625 | 0,500 |
| nDCG@5 | 0,424 | 0,249 |

Bukti kontribusi mekanismenya **tetap ada** (`targets_needed` 100% vs 0%), tetapi
keunggulan retrieval-nya jauh menyempit dibandingkan korpus lama. Ini perlu dirumuskan
ulang di Bab VI, bukan disalin dari angka lama.

---

### J6 — Kejenuhan Hit@k pada korpus kecil · ADA (terukur)

Terbaca dari `results/eval_prediksi/sensitivity_kb12.json`:

| top_k | cond Hit@1 | **cond Hit@k** | cond MRR | std Hit@1 | **std Hit@k** | std MRR |
|---|---|---|---|---|---|---|
| 3 | 33,3% | **100,0%** | 0,667 | 50,0% | 50,0% | 0,500 |
| 4 | 33,3% | **100,0%** | 0,639 | 50,0% | 50,0% | 0,500 |
| 5 | 33,3% | **100,0%** | 0,625 | 50,0% | 50,0% | 0,500 |
| 8 | 33,3% | **100,0%** | 0,611 | 50,0% | 83,3% | 0,545 |
| 10 | 33,3% | **100,0%** | 0,611 | 50,0% | 83,3% | 0,549 |

**Jawaban untuk Bab VI: ya, Hit@3 masih 100% — tetapi hanya untuk mode
prediction-conditioned.** Mode standard tidak jenuh (50% sampai k=5, naik ke 83,3% pada
k=8).

Ini justru **menguatkan** argumen Subbab II.6.5, dengan nuansa yang layak ditulis:
kejenuhan bersifat **asimetris**. Hit@k pinned di 100% untuk conditioned sejak k=3
sehingga kehilangan seluruh daya beda, sementara MRR tetap bergerak (0,667 → 0,611).
Karena itu penilaian memang harus bertumpu pada metrik peringkat, persis seperti yang
diantisipasi laporan.

---

### J7 — Kontribusi fitur: perbandingan dua representasi · ADA

[scripts/eval_feature_importance.py:83-84](../scripts/eval_feature_importance.py#L83-L84)
menjalankan eksperimen **dua kali**, bukan sekali:

```python
raw = run(RAW_FEATURES, predict_delta=False, cfg=cfg, engineered=False)
eng = run(ENGINEERED_FEATURES, predict_delta=True, cfg=cfg, engineered=True)
```

Fungsi `run()` menerima parameter `engineered` yang menentukan apakah
`engineer_features()` dipanggil ([:52-53](../scripts/eval_feature_importance.py#L52-L53)),
lalu keduanya diringkas menjadi `kontribusi_insulin_karbohidrat_%`
([:86](../scripts/eval_feature_importance.py#L86)).

Ini **persis** perbandingan dua kondisi yang dijanjikan Subbab III.3.2, bukan satu angka
tunggal. Hasil tersimpan di `results/eval_prediksi/feature_importance.json`.

**Catatan:** hasil itu perlu **diregenerasi** — belum dijalankan ulang setelah segmentasi
jendela Tugas 5, sehingga angkanya masih dari konfigurasi lama. Besar pekerjaan: kecil
(<1 jam), sekitar 30 menit berjalan.

---

### J8 — Perincian Clarke Error Grid · ADA

`calculate_all_metrics()` mengembalikan seluruh zona, dan **tersimpan lengkap** di
`models/rf_metrics_h6.json`:

```json
{
  "RMSE": 21.12267810688516,
  "MAE": 14.406159634627675,
  "MAPE": 10.944228426626822,
  "Clarke_A": 85.64189827944791,
  "Clarke_B": 9.215352618642465,
  "Clarke_C": 0.011344299489506523,
  "Clarke_D": 2.824730572887124,
  "Clarke_E": 2.3066742295329927,
  "Clarke_A+B": 94.85725089809038
}
```

Rincian A–E tersedia, bukan hanya gabungan A+B. Zona D (2,82%) dan E (2,31%) layak
dibahas di Bab VI — keduanya kesalahan berkonsekuensi klinis.

Ambang zona memakai `CLARKE_LOW`/`CLARKE_HIGH` yang **sengaja dipisahkan** dari ambang
kebijakan risiko di [src/constants.py](../src/constants.py), agar perubahan kebijakan
tidak diam-diam mengubah metrik terbitan.

---

### J9 — Rumusan matematis metrik · ADA (dengan perbedaan yang harus ditulis)

Implementasi persis dari
[ablation_rag_fullkb.py:113-119](../scripts/ablation_rag_fullkb.py#L113-L119):

```python
rank = (topics.index(expected) + 1) if expected in topics else 0
rels = [1 if t == expected else 0 for t in topics]
...
"hit@1": int(bool(topics) and topics[0] == expected),
f"hit@{TOP_K}": int(expected in topics),
"rank": rank,
"mrr": round(1.0 / rank, 3) if rank else 0.0,
```

nDCG@k ([:24-30](../scripts/ablation_rag_fullkb.py#L24-L30)):

```python
def ndcg_at_k(rels, k):
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rels))))
    return dcg / idcg if idcg > 0 else 0.0
```

**Tiga perbedaan dari definisi baku yang HARUS ditulis di laporan:**

1. **Relevansi ditetapkan pelabel kata kunci, bukan penilaian manusia.** `expected`
   dibandingkan terhadap `classify_chunk(d["text"])` — pencocokan kata kunci berbobot.
   Pada korpus sekarang, **49,2% chunk jatuh ke kelas "lain"** dan tidak pernah dapat
   dihitung relevan. Ini batas atas yang membatasi seluruh metrik dan wajib disebut
   sebagai keterbatasan evaluasi.

2. **MRR di sini adalah *reciprocal rank* satu kueri, dirata-ratakan belakangan.**
   Definisi baku MRR = (1/|Q|)·Σ(1/rank_i). Kode menghitung 1/rank per kasus lalu
   `groupby("mode").mean()` — secara matematis setara, tetapi rumus di laporan harus
   ditulis sebagai rata-rata atas kueri, bukan atas dokumen.

3. **IDCG dihitung dari `min(k, len(rels))`, bukan dari jumlah dokumen relevan
   sebenarnya di korpus.** Definisi baku memakai jumlah dokumen relevan yang tersedia.
   Karena korpus memuat jauh lebih banyak chunk relevan daripada `k`, keduanya berimpit
   di sini — tetapi laporan harus menulis rumus **sebagaimana dihitung kode**, disertai
   catatan asumsinya.

---

## 3. Ada di Kode, Tidak Ada di Laporan

### Halaman dan tab antarmuka

| Elemen | Dinaungi KF? | Catatan |
|---|---|---|
| Tab "Rekomendasi Klinis" | KF-04, KF-05, KF-06 | terdokumentasi |
| Tab "Catat Keputusan" | KF-08 | terdokumentasi |
| Chart zona glukosa + badge risiko | KF-07 | terdokumentasi |
| Rentang keyakinan konformal | KF-03 | terdokumentasi (sebagian) |
| **Halaman "Input Logbook"** | **tidak ada** | Menulis `manual_logbook.csv` yang tidak dibaca siapa pun. KF-01 mengasumsikan masukan menjadi masukan sistem; halaman ini tidak memenuhinya, sehingga tidak ada KF yang benar-benar menaunginya. |
| **Halaman "Tentang & Validasi"** | **tidak ada** | Menampilkan RMSE dan Clarke A+B dari `results/`. Tidak ada KF/KNF yang menuntut halaman transparansi. Layak diusulkan sebagai bagian KNF-02 (keterjelasan). |

**Risiko saat sidang:** penguji yang membuka aplikasi akan menemukan halaman Input
Logbook, mengisinya, lalu mendapati prediksi tidak berubah. Ini pertanyaan yang paling
mudah muncul dan paling sulit dijawab.

### Skrip yang tidak lagi terpakai atau tumpang tindih

Seluruhnya masih sah secara sintaksis; tidak ada yang menunjuk sumber data yang hilang
(rujukan `additional_docs` yang tersisa **hanya komentar**, sudah diperiksa satu per satu).

| Skrip | Status |
|---|---|
| `ragas_eval.py` | Tumpang tindih dengan `run_ragas.py` (Tugas 6). Memakai korpus produksi, bukan koleksi evaluasi terpisah. Perlu diputuskan mana yang dipakai Bab VI. |
| `ingest_kb.py` | Hanya `manual_kb.json`, tanpa PDF. Tidak menghasilkan indeks produksi. |
| `import_kb_docs.py` | Jalur impor alternatif; kini korpus dikelola manifest. |
| `ablation_rag.py` | Ablasi korpus terkontrol 7 topik `manual_kb`. Masih relevan sebagai pembanding, tetapi bukan bukti korpus penuh. |
| `improve_hypo_detection.py` | Eksperimen; hasil di `hypo_improve.json`. |
| `crossval_rf_vs_lstm.py` | Hasil di `crossval_rf_vs_lstm.json`, **belum diregenerasi** pasca-Tugas 5. |
| `eval_generation_novelty.py` | Hasil di `generation_novelty.json`. |
| `verify_prediction_artifacts.py` | Utilitas verifikasi, bukan penghasil angka laporan. |

**Belum diregenerasi setelah Tugas 5** (angkanya dari konfigurasi lama):
`crossval_rf_vs_lstm.json`, `feature_importance.json`, `smbg_deployment.json`,
`conformal.json`, `hypo_safety_uncertainty.json`.

---

## 4. Daftar Prioritas

Diurutkan menurut dampaknya terhadap kelengkapan **Bab VI**.

| # | Butir | Status | Mengapa didahulukan | Besar |
|---|---|---|---|---|
| 1 | **KNF-10** waktu tanggap | TIDAK ADA | Bab VI **harus** melaporkan angkanya. Tanpa instrumentasi, tidak ada angka untuk dilaporkan sama sekali. | sedang |
| 2 | **J3** sensitivitas `tau_carb` | TIDAK ADA | Subbab II.3.5 menyatakan "diuji pada Bab VI". Belum ada skripnya sama sekali. | sedang |
| 3 | **J2** cakupan per rentang | SEBAGIAN | Subbab II.4.4 sudah membahas sifat marginal; tanpa pemecahan per rentang, pembahasan itu tanpa bukti. | sedang |
| 4 | **Regenerasi hasil lama** | — | Lima berkas hasil masih dari konfigurasi pra-Tugas 5. Menyandingkannya dengan angka baru akan menyesatkan. | sedang |
| 5 | **KF-04** kebocoran `current_glucose` | SEBAGIAN | Merusak klaim inti novelty bila penguji menelusuri kode. Perbaikannya satu baris. | kecil |
| 6 | **KF-07** divergensi asimetris | SEBAGIAN | Definisi Tabel III.1 lebih luas dari implementasi; 4 dari 6 transisi tidak tertangkap. | kecil |
| 7 | **KF-01/KF-02** jalur logbook | SEBAGIAN | Paling mudah ditemukan penguji saat mencoba aplikasi. | sedang |
| 8 | **KF-03** interval dua horizon | SEBAGIAN | Bundle h12 sudah ada; hanya perlu dimuat dan ditampilkan. | kecil |
| 9 | **KNF-02** kontribusi fitur di UI | SEBAGIAN | Angkanya sudah dihitung; hanya perlu ditampilkan. | kecil |
| 10 | **KNF-07** konstanta konformal di `app/` | SEBAGIAN | Kerapian arsitektur; dampak ke Bab VI kecil. | kecil |
| 11 | **KNF-03** penyaring angka dosis | SEBAGIAN | Peningkatan keamanan nyata, tetapi laporan tidak menjanjikan penyaring eksplisit. | sedang |
| 12 | **KF-06/KNF-05** pemeriksaan bahasa | SEBAGIAN | Risiko meningkat karena 8 dari 12 dokumen berbahasa Inggris. | kecil |

### Rekomendasi urutan kerja

**Kerjakan lebih dulu butir 1–4.** Keempatnya adalah hal yang **sudah terlanjur
dijanjikan Bab I–III** tetapi belum dapat dipenuhi kode. Ini risiko paling langsung:
janji tanpa bukti lebih berbahaya di sidang daripada fitur yang tidak sempurna.

**Butir 5–6 murah dan berdampak besar pada kredibilitas.** Keduanya cacat kecil di
kode, tetapi keduanya menyentuh klaim inti (pengondisian prediksi dan peringatan
divergensi) yang menjadi novelty laporan.

**Butir 7 perlu keputusan lingkup, bukan sekadar pekerjaan.** Menyambungkan logbook ke
prediksi berarti memutuskan bagaimana data satu pasien dari logbook digabung dengan
riwayat CGM — itu keputusan rancangan, bukan perbaikan bug. Alternatif yang jujur dan
jauh lebih murah: **nyatakan di laporan bahwa halaman logbook adalah antarmuka
pencatatan untuk pengembangan lanjut**, dan sesuaikan rumusan KF-01.

---

## Catatan Metodologis

Audit ini memeriksa **kesesuaian kode terhadap deskripsi laporan**, bukan mutu
klinis atau mutu hasil. Beberapa butir berstatus ADA tetap memiliki keterbatasan
substantif yang dibahas di `docs/journey.md` — terutama runtuhnya mutu retrieval pada
korpus KB-01..KB-12 (oracle MRR 0,983 → 0,527) yang **tidak** tercermin pada status
KF-04/KF-05 karena keduanya menilai keberadaan dan kesesuaian mekanisme, bukan
efektivitasnya.

Status **SEBAGIAN** diberikan secara ketat: bila deskripsi menuntut X dan kode melakukan
Y yang mirip tetapi tidak sama, statusnya SEBAGIAN — bukan ADA. Tidak ada status yang
dilunakkan.
