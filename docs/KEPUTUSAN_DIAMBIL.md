# Keputusan yang Sudah Diambil

> **Diputuskan pembimbing 11 Agustus 2026 · commit `db4184f` · cabang `refaktor-tujuh-tugas`**
>
> **Berkas ini adalah rujukan tunggal.** `docs/RINGKASAN_KEPUTUSAN_PEMBIMBING.md` tetap
> disimpan sebagai bahan yang mendahuluinya — ia memuat pilihan yang tersedia beserta angka
> pendukungnya, sedangkan berkas ini memuat apa yang **dipilih** dan apa yang **tidak
> dikerjakan sebagai akibatnya**.
>
> Untuk tiap keputusan: **apa** yang diputuskan, **mengapa**, dan **apa yang tidak
> dikerjakan**. Yang terakhir sengaja ditulis eksplisit, karena pekerjaan yang tidak
> dikerjakan tidak meninggalkan jejak apa pun kecuali dicatat.

---

## #8 — Skala derau dan ambang kebermaknaan: DUA PERTANYAAN DIPISAH

### Yang diputuskan

| pertanyaan | alat ukur |
|---|---|
| Apakah efeknya **konsisten**? | **SD selisih berpasangan** `std(A−B)` + Wilcoxon tingkat fold |
| Apakah besarnya **bermakna**? | **Ambang orde besaran** dari ISO 15197:2013 — ±15 mg/dL untuk glukosa < 100 mg/dL |

Sumber ambang: `KB-02_PERKENI-2021_Pemantauan-Glukosa-Mandiri.pdf`, halaman cetak 25 —
korpus proyek ini sendiri.

### Mengapa

Aturan lama *"selisih harus melampaui SD antar-fold"* **mencampur dua pertanyaan yang
berbeda**, dan karena itu tidak dapat menjawab keduanya. Varians antar-fold mengukur
keragaman pasien, bukan ambang kebermaknaan; memakainya sebagai ambang adalah kekeliruan
kategori. Lima skala derau beredar di repo, empat dipakai tanpa pernah menyebut yang mana
(K14).

Untuk rancangan berpasangan, SD selisih adalah yang mengukur **kekonsistenan efek
antar-fold**; varians per model memuat keragaman pasien yang justru saling hapus dalam
pemasangan.

### DUA PILIHAN INI MEMPERKETAT, BUKAN MELONGGARKAN

Ini wajib dinyatakan, dan tidak boleh diperhalus.

**SD selisih menggugurkan klaim yang menguntungkan narasi penelitian ini.** Klaim
sensitivitas hipoglikemia LSTM di +60 menit (−6,203 poin persen) lolos di bawah definisi
lama (SD gabungan 5,862) tetapi **gugur** di bawah SD selisih (6,877). Klaim itu menopang
argumen bahwa keterbatasan prediktor bersifat klinis dan bukan sekadar statistik — dan
definisi yang dipilih justru mencabutnya.

**Ambang klinis menggugurkan ketiga klaim RMSE yang justru LOLOS di bawah SD selisih.**
Ketiganya konsisten antar-fold, dan ketiganya tidak bermakna secara klinis.

**Definisi yang dipilih merugikan pihak yang memilihnya.** Itu bukan kebetulan yang layak
dibanggakan; itu satu-satunya alasan pemilihan ini dapat dipertahankan sama sekali,
mengingat ia dilakukan setelah tabel auditnya terlihat.

### Dua kejujuran yang wajib menyertai ambang klinis

1. **Ia diadopsi SETELAH hasil terlihat.** Yang membuatnya tetap dapat dipertahankan: ia
   diturunkan dari literatur eksternal dan **tidak menyebut satu pun angka proyek ini**,
   sehingga tidak dapat dipilih demi meloloskan atau menggugurkan klaim tertentu.
2. **Ia analogi berdasar literatur, BUKAN penerapan ISO 15197 pada perbandingan model.**
   ISO mengatur akurasi alat ukur, bukan ambang kebermaknaan selisih antarmodel. Yang
   dipinjam hanya **skala besarannya**. Menyandingkan RMSE agregat dengan toleransi
   per-pengukuran 95% sebagai persentase tidak sepadan satuan statistiknya; yang sah adalah
   pernyataan orde besaran.

### Penerapan pada keempat klaim RAPUH

| klaim | selisih | SD selisih | konsisten? | % ambang | **rumusan yang dipakai** |
|---|---:|---:|---|---:|---|
| T4.1 h6 RMSE RF vs GBM | +0,358 | 0,135 | ya | **2,39%** | **konsisten tetapi tidak bermakna** |
| T4.1 h12 RMSE RF vs GBM | +0,976 | 0,388 | ya | **6,51%** | **konsisten tetapi tidak bermakna** |
| T1.1 h6 RMSE RF vs LSTM | +0,478 | 0,239 | ya | **3,19%** | **konsisten tetapi tidak bermakna** |
| T2.1 h12 sens RF vs LSTM | −6,203 | 6,877 | **tidak** | — | **RAPUH, tidak dinarasikan** |

**Untuk ketiga klaim RMSE, rumusannya sama di bawah ketiga definisi SD** — semuanya
menyatakan efek yang konsisten, dan ambang klinislah yang menggugurkan kebermaknaannya.
**Pilihan definisi SD karena itu tidak relevan bagi ketiganya**, dan perdebatan tentangnya
tidak perlu diteruskan untuk klaim RMSE.

**T2.1 h12** gugur pada pertanyaan konsistensi, bukan kebermaknaan. **Ambang mg/dL tidak
berlaku baginya** karena satuannya poin persen deteksi kejadian, bukan galat konsentrasi.

**T2.1 h6** (−10,429, SD selisih 4,727) **bertahan** — konsisten menurut ketiga definisi.
Keunggulan sensitivitas hipoglikemia LSTM karena itu **mantap di +30 menit dan rapuh di
+60 menit**, bukan "mantap pada kedua horizon".

### Pemeriksaan lanjutan: klaim inti penelitian ini SELAMAT

Seluruh repo disapu untuk mencari klaim yang memakai aturan lama tanpa menyebut definisinya.
Yang paling penting di antaranya adalah **klaim inti kontribusi** — PC-RAG lawan standard
pada crossfold — dan ia diperiksa terhadap ketiga definisi:

| himpunan | selisih | SD gab | SD maks | SD selisih | verdik |
|---|---:|---:|---:|---:|---|
| **divergen** | **+0,0505** | 0,0298 | 0,0209 | **0,0263** | **konsisten, ketiga definisi** |
| natural | +0,0017 | 0,0628 | 0,0636 | 0,0136 | tidak konsisten, ketiga definisi |

Pada kasus divergen selisihnya **positif di keenam fold** (0,048 · 0,070 · 0,067 · 0,084 ·
0,025 · 0,009). Pada kasus natural tandanya berganti-ganti (+ − − − + +).

**Keputusan #8 karena itu tidak melemahkan kontribusi penelitian ini**, dan juga tidak
menguatkannya secara artifisial: rumusan lama — *PC-RAG memberi perbaikan nyata pada kasus
divergen dan tidak memberi apa-apa pada kasus non-divergen* — bertahan utuh di bawah
definisi yang lebih ketat sekalipun.

### Batas yang wajib dinyatakan: ambang klinis TIDAK berlaku untuk metrik penelusuran

Ambang ±15 mg/dL hanya dapat diterapkan pada besaran bersatuan **mg/dL**. Untuk MRR, Hit@1,
nDCG@5, dan sensitivitas (poin persen), **tidak ada ambang kebermaknaan eksternal yang
ditetapkan** pada penelitian ini.

Akibatnya, bagi metrik penelusuran hanya pertanyaan **konsistensi** yang dapat dijawab.
Pertanyaan *"apakah +0,0505 MRR cukup besar untuk berarti bagi dokter?"* **tetap terbuka**,
dan tidak boleh dijawab diam-diam dengan uji signifikansi. Ini keterbatasan yang tersisa
setelah #8, bukan yang diselesaikannya.

### Apa yang TIDAK dikerjakan

- **Tidak satu pun angka lama diubah.** T1.1, T1.1b, T2.1, T3.x, T4.1 tetap sebagaimana
  tersimpan. Yang berubah hanya cara pelaporannya.
- **Ambang kebermaknaan untuk metrik penelusuran tidak ditetapkan** — tidak ada sumber
  eksternal yang sepadan dengan ISO 15197 bagi MRR.
- **Uji ulang dengan definisi baru tidak dijalankan** — SD selisih dihitung dari angka per
  fold yang sudah tersimpan, bukan dari pelatihan ulang.
- **Klaim T2.1 h12 tidak dicabut dari berkas hasil**, hanya ditandai rapuh dan dilarang
  dinarasikan.

---

## #5 — Pemilihan model: PILIHAN D, Random Forest tetap produksi

> ## ⚠ KEPUTUSAN INI SUDAH DILAMPAUI — 13 Agustus 2026
>
> **Prediktor produksi kini Gradient Boosting**, bukan Random Forest. `config.yaml`
> berbunyi `model.name: "GradientBoosting"`, dan `app/streamlit_app.py` memuat bundle GBM
> lebih dulu lewat `KELUARGA_BUNDLE`.
>
> Perubahan itu **dikerjakan pembimbing setelah** bagian di bawah ditulis, sehingga bagian
> ini menggambarkan keputusan **pada saat diambil**, bukan keadaan sistem sekarang.
>
> **Yang menjadi tidak berlaku:**
> - kalimat *"Random Forest **tetap** menjadi prediktor produksi"*
> - kalimat *"`config.yaml` tidak disentuh"* pada bagian "Apa yang TIDAK dikerjakan"
> - *"GBM masuk saran pengembangan"* — ia kini produksi, bukan saran
>
> **Yang TETAP berlaku, dan justru menjadi lebih penting:**
> - ketiga alasan di bawah tetap menggambarkan pertimbangannya
> - **batas yang wajib dinyatakan:** pada h12 GBM **KALAH** dari RF pada sensitivitas
>   hipoglikemia (2,42% lawan 4,36%; hipo berat terlewat 957 lawan 933). Klaim *"unggul di
>   setiap dimensi"* **hanya berlaku pada h6** — dan catatan itu kini sudah ada di
>   `config.yaml` sendiri
> - konsekuensi terhadap Bab II subbab II.4.1 **tidak berubah**: kedua alasan lama tetap
>   harus dibuang. Naskah pengganti di `docs/REVISI_BAB_II_4_1.md` perlu **disesuaikan**,
>   karena ia menulis prediktor tidak diganti — padahal akhirnya diganti
>
> **Yang belum diperiksa dan tidak boleh diasumsikan:** apakah seluruh angka hilir sudah
> dihitung ulang pada prediktor baru. Migrasi ini menyentuh crossfold retrieval, realcases,
> T3.1–T3.3, kalibrasi konformal, dan RAGAS. Sebagian artefak RF sudah diarsipkan
> (`*_RF_arsip.json`) dan ada `docs/PRAPENDAFTARAN_T6_GBM_RETRIEVAL.md` yang belum selesai,
> tetapi kelengkapannya **belum diverifikasi**.
>
> Alur sistem sebagaimana berjalan sekarang: `docs/METHODOLOGY.md` bagian 7.

### Yang diputuskan

Random Forest **tetap** menjadi prediktor produksi. GBM dan LSTM dilaporkan sebagai
**pembanding penuh** di Bab VI. GBM masuk **saran pengembangan dengan besaran terukur**.

### Mengapa — tiga alasan, tidak diperhalus

**1. Kontribusi penelitian ini adalah Prediction-Conditioned RAG. Prediktor adalah
komponen, bukan kontribusi.** Mengganti komponen di akhir penelitian demi keunggulan pada
komponen itu sendiri salah menempatkan apa yang sedang diteliti.

**2. Keunggulan akurasi GBM tidak bermakna secara klinis.** Menurut #8: +0,358 mg/dL
(2,39% ambang) dan +0,976 mg/dL (6,51% ambang). Keduanya konsisten antar-fold, dan keduanya
tidak dapat ditafsirkan secara fisik. **Menukar seluruh rantai evaluasi demi selisih yang
tak bermakna adalah keputusan yang buruk** — yang dibatalkan mencakup crossfold retrieval,
realcases, T3.1, T3.2, T3.3, kalibrasi konformal, dan RAGAS yang terkunci kuota.

**3. Keunggulan GBM yang NYATA adalah ukuran dan kecepatan — dan itu tidak membutuhkan
ambang klinis sama sekali.** 0,485 MB lawan 322 MB (**665×**), dan 3,4 detik lawan 708
detik (**208×**). Tidak ada perdebatan definisi yang dapat mengecilkan angka sebesar itu.
**Itulah isi saran pengembangan**, dan ia lebih kuat daripada klaim akurasi mana pun yang
dapat diajukan GBM.

### Naskah pengganti Bab II subbab II.4.1

Disimpan terpisah di **`docs/REVISI_BAB_II_4_1.md`**. Naskah bab **tidak** disunting
langsung.

Kedua alasan lama **dibuang**, bukan diperhalus atau ditambal:

| alasan lama | mengapa dibuang |
|---|---|
| keterjelasan kontribusi fitur | J7 menggugurkannya. Yang RF miliki bawaan adalah MDI, dan MDI **tidak boleh dilaporkan** karena bias kardinalitas. Angka yang dilaporkan adalah permutation importance, yang model-agnostik |
| kebutuhan komputasi rendah | **Terbalik secara faktual.** RF paling mahal dari ketiganya: 708 dtk dan 322 MB, lawan GBM 3,4 dtk / 0,485 MB dan LSTM 477 dtk / 0,124 MB |

### Paragraf penutup Bab VI

Disimpan di **`docs/NASKAH_PENUTUP_BAB_VI.md`**. Isinya: langit-langit sensitivitas
hipoglikemia melekat pada **fungsi kerugian**, bukan keluarga model — terbukti dengan
menguji dua keluarga pohon yang berbeda dan memperoleh hasil yang sama — disambungkan ke
jarak oracle T3.2 (+0,2646).

### Apa yang TIDAK dikerjakan

- **Model produksi tidak diganti.** `config.yaml` tidak disentuh.
- **GBM tidak disetel.** Angkanya tetap batas bawah kemampuan gradient boosting.
- **Kerugian berbobot kelas tidak diuji**, meskipun ia tuas yang paling menjanjikan bagi
  sensitivitas hipoglikemia. Masuk saran pengembangan, bukan pekerjaan sesi ini.
- **Naskah Bab II tidak disunting langsung** — hanya naskah penggantinya yang disiapkan.

---

## #2 — Bundle RF: TIDAK diganti ke 50/12

### Yang diputuskan

Bundle produksi tetap 200/20. Pertukaran 50/12 dilaporkan sebagai **saran pengembangan
bertaring angka**.

### Mengapa

**Ambang #8 menyelesaikannya.** Biaya 50/12 adalah **0,081 mg/dL = 0,54% dari ambang**,
untuk penghematan **−96,3% ukuran** (297,15 → 11,09 MB). Pertukaran itu konsisten dan
biayanya tidak bermakna secara klinis.

Bahwa ia tidak diganti sekarang adalah konsekuensi dari #5: prediktor tidak disentuh di
akhir penelitian. Bila #5 kelak memilih GBM, keputusan ini **gugur dengan sendirinya** —
GBM sudah 0,485 MB tanpa pengecilan apa pun.

### Yang TIDAK boleh dikatakan

**Jangan sebut kesetaraan.** T1.1b **tidak** membuktikan 50/12 setara dengan 200/20, dan
rancangannya memang tidak mampu membuktikannya: uji beda dipakai sebagai uji kesetaraan,
dan margin TOST tidak pernah ditetapkan di muka. Yang berdiri sendiri tanpa uji hipotesis
apa pun adalah **pertukarannya**: 0,081 mg/dL untuk 96,3% ukuran.

### Apa yang TIDAK dikerjakan

- Bundle tidak diperkecil; berkas 297 MB tetap dipakai.
- Uji kesetaraan TOST dengan margin yang ditetapkan di muka **tidak** dijalankan.

---

## #3 — `chunk_size` 900: TIDAK diubah

### Yang diputuskan

Tetap 900. Kerugiannya dinyatakan **penuh** sebagai keterbatasan dan saran pengembangan.

### Mengapa

Mengubahnya menuntut **indeks ulang korpus** dan **penghitungan ulang seluruh angka
penelusuran** — crossfold, realcases, T3.1, T3.2, T3.3, dan konsentrasi.

### Yang tetap dinyatakan penuh, tanpa diperhalus

- `chunk_size` 900 membuang **8,23% token korpus** akibat pemotongan diam-diam
  `all-MiniLM-L6-v2` pada **256 token** (bukan 512 — itu `model_max_length` BERT-nya).
- **KB-03 kehilangan 48,84% chunk-nya**, dan KB-03 adalah dokumen sumber evaluasi RAGAS.
- Keputusan mempertahankan 900 pada Tahap 0 diambil ketika **kedua fakta itu belum
  diketahui**. Ia dipertahankan sekarang atas alasan biaya penggantian, bukan atas alasan
  yang sama seperti dulu — dan perbedaan itu perlu tertulis.

### Apa yang TIDAK dikerjakan

- Korpus tidak diindeks ulang.
- `chunk_size` 500, yang MRR-nya 0,541 lawan 0,425 pada B4, tidak diadopsi.

---

## #4 — K12 (jangkauan 1,16%): TIDAK dikerjakan

### Yang diputuskan

Jangkauan penelusuran tidak diperluas. Batas klaimnya dinyatakan dengan tepat.

### Bingkai pertahanan — tulis persis begini

> Perbandingan PC-RAG lawan standard **valid secara internal**, karena kedua lengan
> beroperasi pada korpus, pelabel, dan jangkauan yang **sama persis**. Yang tidak diketahui
> adalah apakah keunggulan **+0,051 pada kasus divergen** bertahan pada retriever
> berjangkauan luas.

**Batasnya tepat sebesar itu — jangan lebih sempit, jangan lebih lebar.** Bukan "hasilnya
tidak sah" (ia sah secara internal), dan bukan pula "hasilnya berlaku umum" (jangkauannya
1,16%).

### Apa yang TIDAK dikerjakan

- `fetch_k` tidak dinaikkan dari 12.
- Frasa kondisi tidak diperbanyak dari tiga.
- **Tidak diuji apakah konsentrasi MENYEBABKAN** pola biner `context_precision`, jurang
  Hit@1/Hit@5, dan konteks tak relevan E01. Yang ada tetap satu sebab yang konsisten dengan
  ketiganya dan terukur besarnya.

---

## #6 — `hanya_kondisi`: TIDAK diadopsi

### Yang diputuskan

`build_ablation_query()` tidak diubah. Temuannya **diangkat sebagai saran pengembangan
spesifik**, dengan tempat perbaikan yang ditunjuk tepat.

### Alasan pokok

`build_ablation_query()` dipakai **12 skrip evaluasi dan NOL jalur aplikasi**. Mengadopsinya
mengubah **seluruh angka evaluasi tanpa mengubah perilaku sistem sedikit pun**.

Itu **kelas cacat `top_k` 4 lawan 5 dalam bentuk terbalik**: dulu evaluasi mengukur
kedalaman yang tidak pernah dipakai aplikasi; kali ini perubahan akan menggeser seluruh
angka evaluasi sementara aplikasi berjalan persis seperti sebelumnya. Keduanya berakar pada
hal yang sama — jalur evaluasi dan jalur aplikasi tidak berbagi pembentuk kueri.

### Yang DIANGKAT sebagai saran pengembangan

Temuannya sendiri kuat dan tidak boleh ikut terbuang bersama keputusan tidak mengadopsinya:

- **Numeral di teks kueri merugikan penelusuran.** Membuangnya menaikkan Hit@1 dari
  **42,2% ke 71,1%** (+68% relatif, p=0,00058).
- **Perbandingannya kebal kontaminasi K1**, karena produksi dan `hanya_kondisi` berbobot
  kata kunci pelabel **identik (17)**. Ini salah satu dari sedikit perbandingan penelusuran
  yang tidak terkena sirkularitas pelabel.
- **Tempat perbaikannya adalah `_primary_query()`** di `src/rag/conditioned_query.py` —
  jalur aplikasi — **bukan** jalur evaluasi.

**Pertukarannya wajib ikut disebut**, karena keduanya bergerak berlawanan:

| | produksi | `hanya_kondisi` |
|---|---:|---:|
| potongan unik terjangkau | **24** | 14 |
| teks kueri berbeda dari 73 nilai glukosa | **73** | **3** |
| porsi pengambilan ke 10 teratas | 75,07% | 93,42% |

Glukosa 55 dan 68 menghasilkan kueri **identik huruf demi huruf** di bawah `hanya_kondisi`.

### Apa yang TIDAK dikerjakan

- `build_ablation_query()` tidak diubah; `tests/test_rag_retrieval.py` tetap berlaku.
- `_primary_query()` **juga tidak** diubah — saran pengembangan, bukan pekerjaan sesi ini.
- Tidak diuji apakah membuang numeral dari `_primary_query()` memperbaiki penelusuran
  aplikasi. T3.3 hanya mengukur bentuk aplikasi apa adanya.

---

## #1 dan #7 — pekerjaan pengukuran, bukan keputusan tutup

**#1 (T2.2)** putaran pertama selesai: κ = 0,2505 (lemah). **Putaran kedua disiapkan** sesi
ini atas dasar cacat instrumen yang teridentifikasi sebelum κ dihitung. Lihat
`docs/PRAPENDAFTARAN_T2.2_PUTARAN_2.md`.

**#7 (kestabilan RAGAS)** diukur sesi ini. Lihat
`docs/PRAPENDAFTARAN_T_RAGAS_STABIL.md`.

Keduanya **diagnostik**: tidak satu pun angka yang sudah ada diubah atau diganti, apa pun
hasilnya.

---

## Ringkasan: apa yang berubah pada sistem

**Tidak ada.** Tidak satu pun keputusan di atas mengubah `config.yaml`, model produksi,
korpus, indeks, atau jalur aplikasi.

Yang berubah adalah **cara hasil dilaporkan**, dan itu memang yang diputuskan: empat klaim
keunggulan dirumuskan ulang, dua alasan pada Bab II dibuang, dan empat temuan diangkat
menjadi saran pengembangan bertaring angka.
