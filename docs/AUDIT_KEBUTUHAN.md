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
> Tanggal audit awal: 5 Agustus 2026 · Cabang: `refaktor-tujuh-tugas`
>
> **REVISI 6 Agustus 2026** setelah Bagian A (perbaikan cacat) dan Bagian B (pengujian
> berbasis literatur) dari `docs/PROMPT_PERBAIKAN_DAN_REKAYASA.md` selesai dikerjakan.
> Status yang berubah ditandai **(A1)** … **(A4)** menurut butir yang mengubahnya.
>
> **Pembacaan penting untuk Bagian B:** seluruh butir B **hanya diukur, tidak diadopsi**.
> `config.yaml` dan kode produksi tidak disentuh oleh satu pun butir B. Karena itu status
> kebutuhan di bawah mencerminkan kode **setelah Bagian A saja**. Hasil Bagian B ada di
> Bab 5 dokumen ini sebagai bahan keputusan, bukan sebagai perubahan yang sudah berlaku.

---

## 1. Tabel Ringkas

### Kebutuhan Fungsional

| Kode | Status | Catatan |
|---|---|---|
| KF-01 | **SEBAGIAN** | Kelima variabel dapat diisi, tetapi datanya tidak pernah sampai ke jalur prediksi |
| KF-02 | **SEBAGIAN** | Besaran turunan hanya dihitung pada pipeline dataset, bukan pada masukan manual |
| KF-03 | **ADA** *(A3)* | Kedua horizon ditampilkan dengan interval konformalnya masing-masing |
| KF-04 | **ADA** *(A1)* | Sumber kondisi kini parameter eksplisit; tidak ada fallback ke `current_glucose` |
| KF-05 | **ADA** | Retriever dapat dipanggil terpisah; mengembalikan metadata dokumen dan halaman |
| KF-06 | **SEBAGIAN** | Instruksi bahasa ada di prompt; tidak ada pemeriksaan keluaran |
| KF-07 | **ADA** *(A2)* | Keenam perpindahan kategori tertangkap; pesan menyebut kategori asal dan tujuan |
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
| KNF-07 | **SEBAGIAN** *(A2, A3)* | Konstanta konformal dan logika divergensi sudah pindah ke `src/`; **masih tersisa** ambang tren literal dan logika peringatan hipoglikemia di `app/` |
| KNF-08 | **ADA** | Metadata halaman cetak tersimpan dan terverifikasi ke PDF sumber |
| KNF-09 | **ADA** | Seluruh komponen sumber terbuka atau tier gratis |
| KNF-10 | **ADA** *(A4)* | Waktu tanggap per tahap terinstrumentasi, ditampilkan ke dokter, dan ada skrip benchmark |

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

#### STATUS BARU: ADA (diperbaiki A3, commit `66c707c`)

`load_horizons()` memuat semua bundle horizon yang tersedia; aplikasi menampilkan prediksi
dan interval **per horizon** beserta cakupan yang benar-benar terukur, dan mengevaluasi
divergensi pada setiap horizon sambil menyebut yang mana memicunya.

Faktor konformal kini dibaca per horizon dari [`src/conformal.py`](../src/conformal.py),
menggantikan `CONFORMAL_K = 3.3` yang **terduplikasi di dua tempat** pada `app/`. Bila
suatu horizon belum dikalibrasi, aplikasi **tidak menampilkan interval sama sekali** dan
menyuruh menjalankan kalibrasi — interval dengan faktor tebakan tidak dapat dibedakan
dokter dari interval yang benar-benar terkalibrasi.

**Cacat ketiga yang tidak ada di audit awal:** `scripts/conformal_calibration.py` tidak
menerapkan `max_gap_steps`, tidak seperti pelatihan produksi, `train_condition_classifier`,
dan `eval_retrieval_crossfold` yang memakainya sejak Tugas 5. Faktor 3,3 karena itu
**tidak pernah** mengkalibrasi model produksi — bukan sekadar menjadi usang.

Kalibrasi ulang (ternormalisasi, target 95%):

| Kalibrasi | q | cakupan terukur | lebar rata-rata |
|---|---|---|---|
| lama (9 Juli, tanpa segmentasi, h6 saja) | 3,30 | 95,5% | 109,1 mg/dL |
| **baru h6 (+30 mnt)** | **3,31** | 95,3% | 100,0 mg/dL |
| **baru h12 (+60 mnt)** | **2,96** | 96,5% | 154,1 mg/dL |

**Yang harus dilaporkan jujur:** dugaan bahwa faktor lama meleset ternyata **tidak
terbukti**. q h6 yang baru adalah 3,31, praktis sama dengan 3,30. Interval +30 menit yang
selama ini ditampilkan **memang sudah benar**, meskipun berasal dari kalibrasi yang secara
metodologis tidak sepadan. Nilai A3 terletak pada **menampilkan horizon kedua** dan
**menghapus faktor hardcode**, bukan pada memperbaiki interval yang salah.

Catatan teknis untuk laporan: q h12 lebih **kecil** daripada q h6, tetapi interval h12
tetap lebih lebar (154,1 vs 100,0 mg/dL) karena std antar-pohon di h12 jauh lebih besar.
Yang mengecil hanya pengalinya. Inilah bukti langsung bahwa satu faktor untuk dua horizon
tidak sah.

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

#### STATUS BARU: ADA (diperbaiki A1, commit `61f0569`)

`_enhance_query()` kini menerima parameter eksplisit `condition_glucose`
([retriever.py:232-265](../src/rag/retriever.py#L232-L265)) dan **tidak punya fallback**:
bila pemanggil tidak menyebutkan sumbernya, tag kondisi tidak ditambahkan sama sekali.
Menebak sumbernya itulah cacat aslinya, sehingga menebak "dengan lebih baik" bukan
perbaikan. `RAGPipeline._retrieve()` meneruskan nilai prediksi
([pipeline.py:295-300](../src/rag/pipeline.py#L295-L300)).

Dikunci enam tes regresi di `tests/test_rag_retrieval.py`, termasuk satu yang memastikan
pipeline meneruskan glukosa terprediksi lewat retriever tiruan yang disuntikkan — sengaja
tidak bergantung pada retriever asli, karena versi pertama tes itu **ter-skip diam-diam**
pada suite penuh dan hijau tanpa menguji apa pun (diperbaiki di commit `f048e1a`).

**Temuan tambahan A1 yang tidak ada di audit awal.** Seluruh skrip ablasi memberi lengan
prediction-conditioned sufiks `" (prediksi 30 menit ke depan)"` yang tidak dimiliki lengan
standard, sehingga kedua kueri berbeda pada **dua** hal sekaligus dan selisih metriknya
tidak mengisolasi sumber pengondisian. Sufiks dihapus dari keduanya lewat
`src/rag/ablation_query.py` yang kini menjadi satu-satunya pembentuk kueri untuk lima
skrip.

Arah akibatnya **berlawanan dengan dugaan audit**: sufiks itu merugikan lengan PC-RAG,
sehingga angka PC-RAG yang pernah dilaporkan **terlalu rendah**, bukan terlalu tinggi.
Ablasi korpus-penuh MRR 0,625 -> **1,000**; oracle crossfold divergen 0,539 -> **0,794**.
Lengan standard tidak bergeser satu digit pun pada 15 sel independen, karena kuerinya
memang byte-identical sebelum dan sesudah.

**Konsekuensi untuk Bab VI:** diagnosis lama bahwa runtuhnya oracle disebabkan komposisi
topik korpus **harus dicabut**. Oracle pulih tanpa menyentuh korpus sama sekali.

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

#### STATUS BARU: ADA (diperbaiki A2, commit `2524d7d`)

Logika dipindahkan ke [`src/alerts.py`](../src/alerts.py) sebagai
`evaluate_divergence(current, predicted, horizon)` yang menangani **keenam** perpindahan
kategori dengan pesan yang menyebut kategori asal dan tujuan, dalam tiga tingkat
kegentingan: ayunan hipo↔hiper **kritis**, keluar dari target **waspada**, kembali ke
target **informatif**. `app/ui.py` hanya menyisakan `render_divergence_alert()` yang
memilih wadah Streamlit menurut kegentingan yang sudah ditetapkan modul logika.

Cacat kedua yang ikut diperbaiki: syarat lama membandingkan **string tampilan**
(`cur_label == "Dalam Target"`), sehingga mengubah satu kata di `app/ui.py` akan mematikan
peringatan tanpa satu pun error. Label kini berasal dari `condition_label_id()` di
`src/constants.py`, satu sumber untuk UI dan logika.

21 tes di `tests/test_alerts.py` menutup keenam perpindahan, kesetaraan ambang dengan
`src/constants.py`, dan kasus non-divergen.

**Besar cacatnya, diukur pada seluruh OhioT1DM** (163.310 pasangan +30 menit tanpa jeda
sensor): 21.585 divergensi kategori, aturan lama melewatkan **10.841 — tepat separuhnya**.

Tetapi kejujurannya harus lengkap: **99,8% dari yang terlewat adalah kembali ke rentang
target** (10.822 dari 10.841), yang informatif dan bukan berbahaya. Ayunan hipo↔hiper yang
memotivasi desain simetris ini ternyata **sangat langka: 19 kejadian dari 163.310 jendela
(0,012%)**.

Jadi A2 memperbaiki **selisih definisi yang nyata** terhadap Tabel III.1, tetapi **dampak
keselamatannya sedang, bukan besar**. Bab VI harus memakai kalimat itu, bukan "menutup
celah berbahaya".

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

#### STATUS BARU: TETAP SEBAGIAN (sebagian diperbaiki A2 dan A3)

**Yang sudah selesai:**

- Konstanta konformal terduplikasi **hilang seluruhnya**. `grep "3\.3\|CONFORMAL_K"
  app/*.py` tidak menghasilkan apa pun. Faktor kini dibaca per horizon dari
  `src/conformal.py`.
- Logika divergensi **pindah** ke `src/alerts.py` dan dapat diuji tanpa Streamlit
  (21 tes).

**Yang MASIH tersisa di `app/` — sebab status ini tidak dinaikkan menjadi ADA:**

1. **Ambang tren ditulis ulang secara literal**
   ([streamlit_app.py:247](../app/streamlit_app.py#L247)):

```python
trend = "↑ Meningkat" if delta > 10 else ("↓ Menurun" if delta < -10 else "→ Stabil")
```

   `src/constants.py` sudah mendefinisikan `TREND_STABLE_THRESHOLD_MGDL = 10.0` sebagai
   sumber tunggal, tetapi `app/` menulis ulang angkanya. **Mengubah konstanta itu tidak
   akan mengubah tampilan** — persis kelas cacat yang KNF-07 permasalahkan, dan persis
   pola yang diperbaiki Tugas 3 untuk ambang glikemik.

2. **Logika peringatan hipoglikemia dini masih di `app/`**
   ([streamlit_app.py:282-286](../app/streamlit_app.py#L282-L286)). Ambangnya sudah
   diimpor dengan benar, tetapi **keputusan klinisnya** berada di lapisan antarmuka —
   keadaan yang sama dengan divergensi sebelum A2, dan belum dipindahkan.

Keduanya berukuran kecil (masing-masing di bawah satu jam) dan **sengaja tidak dikerjakan**
karena berada di luar lingkup A1-A4 yang sudah ditutup dan dilaporkan. Dimasukkan ke daftar
prioritas.

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

#### STATUS BARU: ADA (diperbaiki A4, commit `7547dda`)

[`src/timing.py`](../src/timing.py) menyediakan `StageTimer` tanpa dependensi, cukup ringan
untuk dinyalakan di jalur produksi. `RAGPipeline.answer()` menerima `timer` opsional dan
mengembalikan `timings`; aplikasi mengukur rekayasa fitur, prediksi, dan kalibrasi lalu
menampilkan waktu tanggap beserta rinciannya kepada dokter.
[`scripts/benchmark_latency.py`](../scripts/benchmark_latency.py) melaporkan median, p95,
dan proporsi LLM. 15 tes di `tests/test_timing.py`.

Hasil pada Ryzen 5 5600H tanpa GPU:

| Tahap | median | p95 |
|---|---|---|
| rekayasa fitur | 0,244 dtk | 0,413 dtk |
| prediksi (2 horizon) | 0,056 dtk | 0,058 dtk |
| kalibrasi interval | 0,035 dtk | 0,037 dtk |
| penyusunan kueri | 0,000 dtk | 0,000 dtk |
| retrieval | 0,032 dtk | 0,038 dtk |
| **komputasi lokal** | **0,370 dtk** | **0,537 dtk** |
| generasi LLM | 2,94 dtk (11 sampel bersih) | tidak sah, lihat catatan |
| **TOTAL** | **3,44 dtk** | — |

**Proporsi menunggu LLM: median 89,6%.** Seluruh bagian yang dikuasai sistem selesai di
bawah satu detik; optimasi lokal tidak akan terasa oleh dokter.

**p95 LLM TIDAK sah dilaporkan.** Satu dari 12 permintaan menembus batas laju free-tier dan
menunggu retry 33,4 detik, sehingga p95 17,3 dtk mengukur waktu tunggu retry, bukan
kecepatan model. Upaya memperbaikinya dengan penjedaan justru lebih buruk karena kuota
harian sudah habis. Skrip kini mencetak peringatan otomatis bila ada permintaan >5x median.

**Kuota terukur langsung: 10 permintaan/menit** untuk `gemini-2.5-flash-lite`, bukan 15 RPM
seperti catatan Tugas 6. Angka lama diperlakukan sebagai tidak terverifikasi.

**TEMUAN TERBUKA yang belum diperbaiki.** Bila `GOOGLE_API_KEY` tidak terbaca, sistem
menjawab dengan template **tanpa satu pun error**, sambil tetap melaporkan
`llm_provider: "gemini"`. Dokter tidak diberi tahu bahwa jalur cadangan yang aktif.
Penjagaan sudah dipasang di skrip benchmark (membatalkan run bila rantai LLM tidak siap),
tetapi memperbaikinya di `RAGPipeline.answer()` menyentuh kontrak keluaran pipeline dan
aplikasi, sehingga berada di luar lingkup A4. **Perlu keputusan.**

Ini penting karena jalur cadangan template memangkas waktu tanggap dari ~3,44 ke ~0,37 dtk
— perbedaan yang justru terlihat seperti sistem bekerja sangat baik.

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

#### PEMUTAKHIRAN setelah A3: TETAP SEBAGIAN, tetapi dasarnya sudah diperbaiki

A3 **tidak** menambahkan pemecahan per rentang, sehingga status tidak berubah. Yang berubah
adalah kesahihan angka dasarnya:

- Kalibrasi dijalankan ulang dengan `max_gap_steps`, sehingga kini sepadan dengan model
  produksi. Sebelumnya **tidak pernah** sepadan.
- Keluarannya per horizon: `results/eval_prediksi/conformal_h6.json` dan `conformal_h12.json`
  menggantikan `conformal.json` tunggal yang kini **usang** dan tidak boleh dikutip.
- Cakupan terukur h6 95,3% dan h12 96,5% pada target 95%.

Kekhawatiran Subbab II.4.4 **tetap belum terjawab**: kedua angka itu masih agregat. Bila
Bab VI ingin membahas sifat marginal jaminan konformal, pemecahan per rentang tetap harus
dikerjakan.

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

**Pemutakhiran 6 Agustus:** `conformal.json` sudah **digantikan** `conformal_h6.json` dan
`conformal_h12.json` (A3), keduanya dihitung dengan `max_gap_steps` sehingga sepadan dengan
model produksi. Berkas `conformal.json` lama masih ada di direktori tetapi **tidak boleh
dikutip**. Empat berkas lainnya tetap belum diregenerasi.

Skrip baru dari Bagian A dan B, seluruhnya **pengukuran, bukan jalur produksi**:

| Skrip | Butir | Keterangan |
|---|---|---|
| `scripts/benchmark_latency.py` | A4 | waktu tanggap ujung-ke-ujung |
| `scripts/eval_reranking.py` | B2 | pemeringkatan ulang cross-encoder |
| `scripts/eval_reranking_bentuk_kueri.py` | B2 | diagnostik bentuk kueri (hasilnya confounded) |
| `scripts/eval_hybrid_bm25.py` | B3 | hibrida BM25 + RRF |
| `scripts/eval_chunk_size.py` | B4 | sapuan ukuran potongan |
| `scripts/eval_mmr_lambda.py` | B5 | sapuan `lambda_mult` |
| `scripts/tuning_protocol.py` | B4, B5 | pembagian set dan pencatatan konfigurasi (Bagian C) |

---

## 4. Daftar Prioritas

### 4.1 Daftar awal (5 Agustus) dan penyelesaiannya

| # | Butir | Status awal | Status 6 Agustus |
|---|---|---|---|
| 1 | **KNF-10** waktu tanggap | TIDAK ADA | **SELESAI (A4)** |
| 2 | **J3** sensitivitas `tau_carb` | TIDAK ADA | **belum dikerjakan** |
| 3 | **J2** cakupan per rentang | SEBAGIAN | **belum dikerjakan**; dasar angkanya diperbaiki A3 |
| 4 | **Regenerasi hasil lama** | — | sebagian: `conformal.json` digantikan; empat berkas lain belum |
| 5 | **KF-04** kebocoran `current_glucose` | SEBAGIAN | **SELESAI (A1)** |
| 6 | **KF-07** divergensi asimetris | SEBAGIAN | **SELESAI (A2)** |
| 7 | **KF-01/KF-02** jalur logbook | SEBAGIAN | belum; perlu keputusan lingkup |
| 8 | **KF-03** interval dua horizon | SEBAGIAN | **SELESAI (A3)** |
| 9 | **KNF-02** kontribusi fitur di UI | SEBAGIAN | belum |
| 10 | **KNF-07** konstanta konformal di `app/` | SEBAGIAN | **sebagian**; konstanta hilang, dua sisa lain tetap |
| 11 | **KNF-03** penyaring angka dosis | SEBAGIAN | belum |
| 12 | **KF-06/KNF-05** pemeriksaan bahasa | SEBAGIAN | belum |

Empat dari lima butir berstatus TIDAK ADA/SEBAGIAN yang paling menyentuh klaim inti sudah
tertutup. Yang tersisa didaftar ulang di bawah.

### 4.2 Daftar prioritas terbaru

Diurutkan menurut dampaknya terhadap kelengkapan **Bab VI**.

| # | Butir | Status | Mengapa didahulukan | Besar |
|---|---|---|---|---|
| 1 | **J3** sensitivitas `tau_carb` | TIDAK ADA | Subbab II.3.5 menyatakan "diuji pada Bab VI". Belum ada skripnya sama sekali. Satu-satunya janji laporan yang sepenuhnya kosong. | sedang |
| 2 | **J2** cakupan per rentang | SEBAGIAN | Subbab II.4.4 membahas sifat marginal jaminan konformal; tanpa pemecahan per rentang, pembahasan itu tanpa bukti. Dasar angkanya kini sudah sah (A3). | sedang |
| 3 | **Keputusan Bagian B** | — | Lima butir sudah diukur dan menunggu keputusan adopsi. Selama belum diputuskan, Bab VI tidak dapat menyatakan konfigurasi final. | keputusan |
| 4 | **Degradasi LLM diam-diam** | TEMUAN BARU | Sistem menjawab dengan template tanpa error sambil melaporkan `llm_provider: "gemini"`. Dokter tidak diberi tahu. Menyentuh KNF-04 dan KNF-06. | kecil |
| 5 | **Regenerasi 4 hasil lama** | — | `crossval_rf_vs_lstm`, `feature_importance`, `smbg_deployment`, `hypo_safety_uncertainty` masih pra-Tugas 5. | sedang |
| 6 | **KF-01/KF-02** jalur logbook | SEBAGIAN | Paling mudah ditemukan penguji saat mencoba aplikasi. | sedang |
| 7 | **KNF-07** dua sisa di `app/` | SEBAGIAN | Ambang tren literal + logika hipoglikemia dini. Kelas cacat yang sama dengan yang sudah diperbaiki. | kecil |
| 8 | **KNF-02** kontribusi fitur di UI | SEBAGIAN | Angkanya sudah dihitung; hanya perlu ditampilkan. | kecil |
| 9 | **KNF-03** penyaring angka dosis | SEBAGIAN | Peningkatan keamanan nyata, tetapi laporan tidak menjanjikan penyaring eksplisit. | sedang |
| 10 | **KF-06/KNF-05** pemeriksaan bahasa | SEBAGIAN | Risiko meningkat karena 8 dari 12 dokumen berbahasa Inggris. | kecil |

### Rekomendasi urutan kerja

**Butir 1–2 tetap yang paling mendesak.** Keduanya **sudah terlanjur dijanjikan Bab I–III**
tetapi belum dapat dipenuhi kode. Janji tanpa bukti lebih berbahaya di sidang daripada
fitur yang tidak sempurna.

**Butir 3 memblokir penulisan Bab VI**, bukan karena sulit melainkan karena menunggu
keputusan. Selama lima butir Bagian B belum diputuskan, tidak ada konfigurasi final yang
dapat ditulis sebagai hasil.

**Butir 4 kecil tetapi menyentuh keselamatan.** Sistem yang diam-diam turun ke jalur
cadangan tanpa memberi tahu dokter bertentangan dengan semangat KNF-06 (dokter penentu
akhir dengan informasi lengkap).

**Butir 6 perlu keputusan lingkup, bukan sekadar pekerjaan.** Menyambungkan logbook ke
prediksi berarti memutuskan bagaimana data satu pasien dari logbook digabung dengan
riwayat CGM — itu keputusan rancangan, bukan perbaikan bug. Alternatif yang jujur dan
jauh lebih murah: **nyatakan di laporan bahwa halaman logbook adalah antarmuka
pencatatan untuk pengembangan lanjut**, dan sesuaikan rumusan KF-01.

---


---

## 5. Hasil Bagian B — pengujian berbasis literatur

> **Seluruh butir di bawah HANYA DIUKUR, tidak diadopsi.** `config.yaml` dan kode produksi
> tidak disentuh oleh satu pun butir B. Bagian ini adalah bahan keputusan, bukan catatan
> perubahan yang sudah berlaku.

Seluruh pengukuran dijalankan **setelah A1**, sehingga berdiri di atas kueri yang sudah
simetris. Angka Bagian B apa pun yang pernah dihitung sebelum A1 **tidak berlaku**.

### 5.0 Keterbatasan yang berlaku untuk SELURUH Bagian B

Relevansi ditetapkan `classify_chunk()`, pelabel berbasis kata kunci berbobot — bukan
penilaian manusia. Tiga akibatnya wajib disebut di Bab VI:

1. **49,2% chunk korpus jatuh ke kelas "lain"** dan tidak pernah dapat dihitung relevan.
   Ini batas atas yang membatasi seluruh metrik.
2. **Cakupan pelabel tidak setara antar-bahasa**: chunk berbahasa Inggris 53,2% jatuh ke
   "lain" versus 41,7% untuk Bahasa Indonesia — bias 1,24x yang merugikan setiap teknik
   yang memunculkan lebih banyak dokumen Inggris.
3. **Sirkularitas terhadap BM25** (lihat 5.3): kueri memuat kata kunci yang dipakai
   pelabel, sehingga metode berbasis kecocokan istilah diuntungkan secara sistematis.

Untuk mengangkat keterbatasan ini diperlukan pelabel relevansi manusia, yang tidak
tersedia dalam lingkup TA ini.

### 5.1 B1 — Model embedding dwibahasa · HIPOTESIS TIDAK TERBUKTI

Uji `all-MiniLM-L6-v2` (produksi) versus `paraphrase-multilingual-MiniLM-L12-v2`,
246 kueri (6 ablasi + 240 kasus nyata).

| Himpunan | Metrik | all-MiniLM | multilingual |
|---|---|---|---|
| ablasi 6 kasus PC-RAG | MRR | **1,000** | 0,639 |
| realcases divergen | MRR | **0,393** | 0,385 |
| realcases natural | MRR | **0,506** | 0,381 |
| realcases natural | Hit@1 | **23,3%** | 22,5% |

Model lama unggul pada **8 dari 9 sel**, satu imbang, tidak ada sel tempat multilingual
unggul.

**Pemecahan menurut bahasa dokumen sasaran** (pemetaan `kb_id` ke bahasa diverifikasi
terhadap isi teks, cocok untuk seluruh 12 dokumen):

| Himpunan | sasaran | all-MiniLM MRR | multilingual MRR |
|---|---|---|---|
| realcases natural | dokumen Indonesia | **0,322** | 0,138 |
| realcases divergen | dokumen Indonesia | **0,356** | 0,339 |
| semua | dokumen Inggris | 0,000 | 0,000 |

**Mekanismenya berkebalikan dari dugaan.** Untuk kueri Bahasa Indonesia, `all-MiniLM`
mengambil hampir seluruhnya dokumen Indonesia (4,1 dari 5 chunk; pemeriksaan manual D1-D3
menunjukkan kelimanya dari KB-01..KB-04). Model monolingual memang memisahkan wilayah
ruang vektor per bahasa persis seperti kata Reimers dan Gurevych (2020) — tetapi karena
kuerinya **juga** Bahasa Indonesia, pemisahan itu berfungsi sebagai **penyaring bahasa yang
menguntungkan**. Model multilingual menghapus penyaring itu (chunk Indonesia turun ke ~2,5)
tanpa membawa dokumen yang lebih tepat topiknya.

**Biaya:** dimensi dan ukuran koleksi identik (384, 2.061 chunk), tetapi latensi penyematan
kueri naik dari **41,2 ke 116,0 ms median (2,8x)**.

**Rekomendasi: jangan adopsi.** Yang layak masuk laporan bukan penggantian modelnya
melainkan **temuannya** — pada korpus dwibahasa dengan kueri satu bahasa, model monolingual
dapat berperilaku sebagai penyaring bahasa yang menguntungkan. Ini justru **memperkuat**
rujukan Reimers dan Gurevych: mekanisme yang mereka jelaskan terbukti, hanya arah akibatnya
berbeda karena kuerinya monolingual.

### 5.2 B2 — Pemeringkatan ulang cross-encoder · BERGANTUNG MUTU KOLAM

Kolam kandidat `fetch_k=12` identik untuk kedua lengan; diterapkan pada **kedua mode**.

| Himpunan | baseline | ms-marco-L6 (EN) | mmarco-L12 (ML) |
|---|---|---|---|
| divergen PC-RAG | 0,393 | 0,348 | 0,365 |
| divergen standard | 0,327 | 0,209 | 0,229 |
| **natural PC-RAG** | 0,506 | **0,892** | 0,776 |
| **natural standard** | 0,495 | **0,885** | 0,824 |

Kesimpulannya bukan "membantu" atau "merusak". Pemeringkatan ulang hanya dapat memulihkan
dokumen yang **sudah ada di kolam tetapi bukan peringkat 1** — ukurannya selisih
Hit@5 - Hit@1:

| Himpunan | ruang perbaikan | dMRR (EN) |
|---|---|---|
| ablasi standard | 0,0% | -0,183 |
| divergen standard | 2,5% | -0,118 |
| divergen PC-RAG | 13,4% | -0,045 |
| natural PC-RAG | 68,4% | **+0,386** |
| natural standard | 69,2% | **+0,390** |

**Pearson r = 0,970 (EN).** Dengan n=6 himpunan ini indikasi, bukan bukti; yang
menguatkannya adalah mekanismenya, karena reranking secara definisi tidak dapat memunculkan
dokumen yang tidak ada di kolam.

**Biaya: median 2.317,8 ms (EN)**, menaikkan komputasi lokal dari 0,370 ke ~2,69 dtk (7x)
dan total dengan LLM +67%.

Model Inggris **mengungguli** multilingual pada lima dari enam himpunan sekaligus 1,5x
lebih cepat — konsisten dengan B1.

**Diagnostik bentuk kueri.** Menguji apakah kerugian pada himpunan divergen berasal dari
cross-encoder MS MARCO yang menerima pernyataan, bukan pertanyaan. Hasilnya **membalik
antar-himpunan**: divergen 0,348 -> 0,414 (menolong), natural 0,892 -> 0,592 (merusak).
Bentuk pernyataan yang berlaku sekarang jauh lebih baik. Diagnostik ini juga **cacat
rancangan**: templat pertanyaannya bukan parafrase token-per-token, sehingga tidak dapat
memisahkan "bentuk pertanyaan" dari "susunan kata berbeda". Yang tetap sah: susunan kata
sangat berpengaruh pada cross-encoder Inggris, sedangkan model multilingual hampir tidak
peka.

**Rekomendasi: jangan adopsi sekarang, jangan buang.** Perbaikan pada himpunan natural
terlalu besar untuk diabaikan, dan himpunan itulah yang mewakili distribusi penerapan
sesungguhnya. Tetapi biaya 2,3 dtk belum sepadan sebelum diuji apakah kolam dapat
diperbaiki lebih murah.

### 5.3 B3 — Penelusuran hibrida BM25 + RRF · ANGKA TIDAK SAH DIPAKAI

| Himpunan | baseline | BM25 saja | hibrida RRF |
|---|---|---|---|
| divergen PC-RAG | 0,393 | **0,444** | 0,418 |
| divergen standard | 0,327 | 0,290 | 0,272 |
| natural PC-RAG | 0,506 | **0,913** | 0,887 |
| natural standard | 0,495 | **0,904** | 0,876 |

Latensi BM25 **12,4 ms**, hibrida **24,8 ms** — **93x lebih murah** daripada reranking B2.
BM25 sendirian mengungguli hibrida pada seluruh himpunan besar, sehingga yang sepadan
justru **menghapus** penggabungan RRF.

**ANCAMAN VALIDITAS: sirkularitas.** Pelabel `classify_chunk()` berbasis kata kunci; BM25
juga. Diperiksa langsung, kueri **memuat kata kunci yang dipakai pelabel**:

| Kondisi | kata kunci pelabel yang muncul di kueri | bobot tercakup |
|---|---|---|
| hipoglikemia | `hipoglikemi`, `gula darah rendah`, `15-15` | **7 dari 18** |
| hiperglikemia | `hiperglikemi`, `gula darah tinggi` | 5 dari 26 |
| normal | `target kontrol glikemik`, `kontrol glikemik` | 5 dari 11 |

BM25 **mengoptimalkan langsung apa yang diukur**; retriever vektor tidak punya keuntungan
sejenis karena bekerja pada embedding, bukan kecocokan token harfiah.

**Karena itu angka 0,506 -> 0,913 TIDAK sah dibaca sebagai "BM25 hampir dua kali lebih
baik".** Besar bagian artefaknya tidak dapat diukur tanpa pelabel relevansi manusia.

**Rekomendasi: jangan adopsi berdasarkan angka ini.** BM25 murah dan menjanjikan, tetapi
bukti yang ada tidak dapat memisahkan keunggulan nyata dari artefak evaluasi.


### 5.4 B4 — Ukuran potongan dokumen · KANDIDAT ADOPSI KUAT

Mengikuti protokol Bagian C. Tumpang tindih dijaga proporsional (~13,3%) supaya yang diuji
benar-benar UKURAN, bukan campuran ukuran dan rasio tumpang tindih.

**Cacat produksi yang harus diperbaiki lebih dulu.** Jalannya pertama gagal total pada
ukuran 300: `Batch size of 5896 is greater than max batch size of 5461`. `save_to_chroma()`
memanggil `add_documents()` sekali untuk seluruh chunk. Korpus produksi 2.061 chunk sehingga
belum pernah menyentuh batas itu, tetapi korpus yang bertambah besar akan menabraknya.
Diperbaiki dengan pemecahan batch di
[knowledge_base.py:355-366](../src/rag/knowledge_base.py#L355-L366).

**Set penyetelan (n=240):**

| chunk_size | jumlah chunk | MRR | Hit@1 |
|---|---|---|---|
| 300 | 5.896 | 0,306 | 3,3% |
| **500** | 3.587 | **0,541** | **40,0%** |
| 900 (sekarang) | 2.061 | 0,425 | 26,2% |
| 1.400 | 1.404 | 0,313 | 16,2% |
| 2.000 | 1.014 | 0,302 | 13,8% |

Bentuknya **tidak monoton**: ada puncak di 500. Ukuran 900 yang berlaku sekarang berada di
sisi turun dari optimum itu.

**Set pelaporan (n=240, ukuran terkunci) — angka yang sah untuk Bab VI:**

| | MRR | Hit@1 | jumlah chunk |
|---|---|---|---|
| chunk_size=900 (sekarang) | 0,435 | 29,2% | 2.061 |
| **chunk_size=500 (terpilih)** | **0,475** | **34,6%** | 3.587 |

**Perbaikan +0,040 MRR.**

**Protokol Bagian C terbukti berfungsi.** Keunggulan chunk 500 menyusut dari **+0,116** pada
set penyetelan menjadi **+0,040** pada set pelaporan — hampir sepertiganya. Selisih itu
adalah optimisme yang akan terlaporkan sebagai hasil seandainya kedua tahap memakai himpunan
yang sama. Angka yang sah adalah +0,040.

**Pemeriksaan silang:** `chunk_size=900` menghasilkan MRR 0,425 dan Hit@1 26,2% — persis
sama dengan B5 pada `lambda=0,5` di set penyetelan yang sama. Dua skrip independen, angka
identik, membuktikan pembagian setnya deterministik.

**Tidak ada sirkularitas yang menguntungkan — justru sebaliknya.** Diukur langsung, makin
kecil potongan makin JARANG berhasil dilabeli: 300 -> 28,6% terlabeli, 500 -> 38,4%,
900 -> 50,8%, 1.400 -> 59,1%, 2.000 -> 66,0%. Jadi `chunk_size=500` mencapai MRR lebih tinggi
**meskipun** peluang chunk-nya dihitung relevan lebih rendah. Ini memperkuat hasilnya, dan
membedakannya secara tegas dari B3.

**Rekomendasi: kandidat adopsi kuat, setara B5.** Catatan bila diadopsi: mengubah
`chunk_size` **mewajibkan indeks ulang korpus** dan seluruh angka retrieval pada `results/`
harus dihitung ulang. Ini bukan perubahan satu angka seperti B5.

### 5.5 B5 — Parameter lambda_mult MMR · KANDIDAT ADOPSI TERKUAT

Mengikuti protokol Bagian C sepenuhnya:

| Ketentuan | Pemenuhan |
|---|---|
| Set penyetelan dan pelaporan tidak beririsan | n=240 dan n=240, benih tetap 42, dibagi per himpunan sebelum hasil dilihat |
| Kriteria ditetapkan di muka | `KRITERIA = "mrr"` konstanta modul, bukan argumen |
| Seluruh konfigurasi tercatat | 5 entri di `results/tuning_log.json` |
| Set evaluasi tidak diubah | tidak ada skenario dibuang |
| Jumlah konfigurasi dilaporkan | **5** |

Himpunan ablasi 6 kasus **dinyatakan tidak dapat disetel** (di bawah ambang untuk dibagi
dua), bukan dipaksakan.

**Set pelaporan (n=240, parameter terkunci) — angka yang sah untuk Bab VI:**

| | MRR | Hit@1 | dokumen unik di top-5 |
|---|---|---|---|
| lambda=0,5 (sekarang) | 0,435 | 29,2% | 3,73 |
| **lambda=0,0 (terpilih)** | **0,492** | 29,2% | 3,60 |

Perbaikan **+0,057 MRR** pada set yang tidak pernah dipakai menyetel.

**Arahnya berlawanan dengan dugaan pada prompt B5**: MRR menurun monoton seiring naiknya
bobot relevansi, sehingga keragaman justru **lebih** bernilai, bukan kurang.

**Pemeriksaan kewarasan yang lolos:** Hit@1 identik di seluruh nilai lambda. Itu memang
harus begitu — pilihan pertama MMR selalu dokumen paling relevan karena belum ada apa pun
untuk didiversifikasi.

**Keragaman sumber ternyata tidak dapat dikendalikan lewat `lambda_mult`**: jumlah dokumen
unik hanya bergerak 3,60-3,98 di seluruh rentang dan **tidak monoton** (puncak di
lambda=0,75). Sebabnya MMR mendiversifikasi ketidakmiripan **embedding**, bukan identitas
dokumen. Kekhawatiran "empat potongan dari satu dokumen yang sama" perlu penjaminan
eksplisit, bukan penyetelan parameter ini.

**Rekomendasi: kandidat adopsi terkuat.** Perbaikan terukur pada set pelaporan yang bersih,
biaya nol (satu angka di `config.yaml`), tanpa dependensi baru, tanpa tambahan latensi.
Tetap terkena keterbatasan pelabel pada 5.0.

### 5.6 Ringkasan keputusan yang menunggu

| Butir | Perbaikan terukur | Biaya | Ancaman validitas | Rekomendasi |
|---|---|---|---|---|
| **B1** embedding multilingual | kalah di 8 dari 9 sel | +182% latensi kueri | — | **jangan adopsi** |
| **B2** reranking cross-encoder | +0,39 MRR di natural; **-0,12 di divergen** | +2,3 dtk per kueri (7x komputasi lokal) | — | tunda sampai kolam diperbaiki |
| **B3** hibrida BM25 | +0,41 MRR di natural | +12 ms | **sirkularitas dengan pelabel** | **jangan adopsi atas angka ini** |
| **B4** chunk_size 500 | **+0,040 MRR** (set pelaporan) | indeks ulang wajib | pelabel justru merugikannya | **kandidat kuat** |
| **B5** lambda_mult 0,0 | **+0,057 MRR** (set pelaporan) | nol | — | **kandidat terkuat** |

**Dua butir yang paling layak diadopsi (B4 dan B5) adalah yang paling murah**, dan keduanya
lolos protokol Bagian C dengan set pelaporan yang tidak pernah dipakai menyetel.

**Peringatan bila B4 dan B5 diadopsi bersamaan:** keduanya diukur **secara terpisah**, masing-
masing dengan parameter lain tetap pada nilai sekarang. Perbaikan +0,040 dan +0,057 **tidak
dapat dijumlahkan**. Bila keduanya diadopsi, angkanya harus diukur ulang bersama-sama sebelum
dilaporkan di Bab VI.

**Peringatan tentang himpunan divergen.** Tidak ada satu pun teknik pada Bagian B yang
memperbaiki kedua mode pada himpunan divergen sekaligus. Himpunan itu tetap menjadi
keterbatasan sistem yang harus dinyatakan apa adanya di Bab VI.

---

## Catatan Metodologis

Audit ini memeriksa **kesesuaian kode terhadap deskripsi laporan**, bukan mutu
klinis atau mutu hasil. Beberapa butir berstatus ADA tetap memiliki keterbatasan
substantif yang dibahas di `docs/journey.md`.

Status **SEBAGIAN** diberikan secara ketat: bila deskripsi menuntut X dan kode melakukan
Y yang mirip tetapi tidak sama, statusnya SEBAGIAN — bukan ADA. Tidak ada status yang
dilunakkan. Pada revisi 6 Agustus, **KNF-07 sengaja TIDAK dinaikkan menjadi ADA** meskipun
sebagian besar keluhannya sudah selesai, karena dua sisa nyata masih ada di `app/`.

### Koreksi atas catatan metodologis versi 5 Agustus

Versi sebelumnya menyebut "runtuhnya mutu retrieval pada korpus KB-01..KB-12 (oracle MRR
0,983 → 0,527)" sebagai keterbatasan substantif. **Pernyataan itu harus dicabut.**

Setelah A1 memperbaiki asimetri pembentuk kueri, oracle pulih ke **0,794** (crossfold
divergen) dan **0,802** (realcases divergen) **tanpa menyentuh korpus sama sekali**. Sebagian
besar "runtuhnya oracle" ternyata artefak sufiks kueri, bukan sifat korpus. Komposisi korpus
mungkin masih berperan, tetapi tidak lagi dapat disebut sebagai penyebab utama.

### Keterbatasan yang berlaku untuk seluruh angka retrieval di dokumen ini

Relevansi ditetapkan pelabel kata kunci `classify_chunk()`, bukan penilaian manusia. 49,2%
chunk korpus jatuh ke kelas "lain" dan tidak pernah dapat dihitung relevan. Ini batas atas
yang membatasi **seluruh** metrik retrieval yang pernah dilaporkan proyek ini, termasuk
angka Bab VI.

Tiga akibat spesifik yang harus disebut di laporan:

1. Cakupan pelabel **tidak setara antar-bahasa** (Inggris 53,2% "lain" vs Indonesia 41,7%).
2. Pelabel **berbagi kosakata dengan kueri**, sehingga metode berbasis kecocokan istilah
   (B3) diuntungkan secara sistematis.
3. Pelabel **merugikan potongan pendek**, sehingga hasil B4 justru konservatif.

Mengangkat keterbatasan ini menuntut pelabel relevansi manusia, yang berada di luar lingkup
TA ini. Yang dapat dilakukan laporan adalah **menyatakannya secara eksplisit** alih-alih
menyajikan metrik seolah-olah bebas asumsi.

### Catatan tentang kejujuran proses

Selama Bagian B, kesimpulan sementara **empat kali** dinyatakan sebelum seluruh himpunan
selesai diukur, dan tiga di antaranya terbantah oleh himpunan berikutnya (rincian di
`docs/journey.md`). Ini dicatat karena relevan bagi pembacaan hasil: **setiap angka Bagian B
hanya sah dibaca setelah seluruh himpunan selesai**, dan pola "empat himpunan pertama
searah" terbukti bukan jaminan apa pun.
