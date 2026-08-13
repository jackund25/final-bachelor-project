# Keputusan yang Menunggu Pembimbing

> **Disusun 11 Agustus 2026 · commit `f9e521d` · cabang `refaktor-tujuh-tugas`**
>
> Seluruh percobaan Bab VI **selesai**, termasuk T2.2 yang diisi 11 Agustus 2026. Berkas ini
> mengumpulkan **tujuh keputusan yang masih terbuka** — #1 sudah selesai dan disimpan sebagai
> rekaman — masing-masing dengan angka pendukungnya dan pilihan yang tersedia. Saya menyiapkan
> bahannya; keputusannya milik Anda.
>
> Angka final siap kutip: `results/ringkasan_untuk_bab6.json` (**24 dari 24 sumber ADA**).
> Rincian keterbatasan: `docs/DAFTAR_KETERBATASAN.md` (K1–K14).
> Rekaman kronologis: `docs/journey/part-8-uji-lanjutan-3-t33-t32-t41-t14.md`.

---

## Ikhtisar

| # | Keputusan | Sifat | Menghambat apa |
| - | --------- | ----- | -------------- |
| ~~1~~ | ~~Isi 40 pasangan T2.2~~ **SELESAI** | κ = **0,2505 (lemah)** | aturan κ<0,40 **TERPICU** — lihat di bawah |
| 2 | Ganti bundle RF ke 50/12? | pilihan rekayasa | narasi keterterapan |
| 3 | Tinjau ulang `chunk_size` 900? | pilihan rancangan | seluruh angka penelusuran |
| 4 | Kerjakan K12 (jangkauan 1,16%)? | pekerjaan kecil–sedang | K2, jurang Hit@1/Hit@5, E01 |
| 5 | **Pemilihan model: RF, LSTM, atau GBM?** | **pilihan pokok** | **Bab II II.4.1 dan Bab VI** |
| 6 | Adopsi `hanya_kondisi` ke produksi? | pilihan rancangan | jalur produksi |
| 7 | Uji kestabilan tiga metrik RAGAS? | ~90 panggilan LLM | status "belum diuji" |
| 8 | **Tetapkan skala derau dan ambang kebermaknaan** | **pilihan metodologis** | **empat klaim keunggulan** |

Dua yang paling menentukan isi laporan: **#5** dan **#8**.

> **#1 sudah selesai (11 Agustus 2026), dan hasilnya mengikat.** Cohen's κ = **0,2505
> (lemah)**; kesepakatan mentah 52,5%, **di bawah** tebakan konstan kelas mayoritas (77,5%).
> Pelabel **terlalu ketat**: recall hiperglikemia 0,452, presisi `lain` 0,200, presisi
> `normal` 0,000, dan 17 dari 19 ketidaksesuaian adalah kegagalan mengenali hiperglikemia.
>
> **Aturan yang kini berlaku — dan ia ditanam sebelum penilaian dilakukan, bukan dipilih
> sesudahnya:** κ < 0,40 → angka penelusuran **tidak boleh dilaporkan tanpa kualifikasi
> eksplisit di setiap penyebutannya**. Berlaku atas Hit@1, Hit@5, MRR, nDCG@5, T3.1, T3.2,
> T3.3, crossfold, realcases, dan K12.
>
> **Tidak satu pun angka ditarik.** Perbandingan antar-konfigurasi pada pelabel yang sama
> tetap sah. Yang belum diperiksa: kesalahan pelabel bersifat **sistematis**, dan kesalahan
> sistematis dapat mencondongkan perbandingan antar-konfigurasi bila keduanya berbeda dalam
> kelas apa yang mereka ambil. Rincian di **K1**.

---

## #1 — Isi 40 pasangan T2.2 — **SELESAI 11 Agustus 2026**

> Bagian ini disimpan sebagai rekaman prosedurnya. Hasil dan konsekuensinya ada di
> ikhtisar di atas dan di **K1**; jangan dibaca sebagai pekerjaan yang masih menunggu.

**Letak berkasnya:** `evaluation/verifikasi_relevansi.csv` — 40 baris, isi **hanya** kolom
`penilaian_manusia`. Nilai yang sah: `hipoglikemia` | `hiperglikemia` | `normal` | `lain`.

**Pertanyaan yang dijawab per baris:** *apa topik utama yang dibahas potongan ini?* — **bukan**
apakah potongan itu berguna bagi pasien pada kueri tersebut. `lain` dipakai bila potongan
tidak membahas ketiga kondisi itu.

**Jangan buka** `evaluation/verifikasi_relevansi_KUNCI.json` sebelum selesai; label otomatis
ada di sana dan melihatnya membatalkan sifat buta penilaian.

Setelah terisi: `python scripts/verifikasi_relevansi.py --nilai`

**Estimasi waktu.** Panjang teks potongan median ~1.000 karakter (dibatasi 1.200). Sekitar
**1–2 menit per pasangan** untuk membaca dan memutuskan, sehingga **45–80 menit** untuk 40
pasangan. Dapat dipecah beberapa duduk; skrip penilaiannya menghitung juga bila sebagian
terisi, tetapi ia **memperingatkan** bahwa kesepakatan atas sebagian data menyesatkan bila
yang dilewati justru kasus yang sulit.

> **Perhatian: jalur ini baru dapat dipakai sejak commit `4a13c86`.** Sebelum itu `--nilai`
> akan berhenti dengan *"Belum ada satu pun penilaian yang terisi"* **meskipun keempat puluh
> baris sudah diisi** — baris komentar berkoma dikutip `csv.writer` lalu terbaca sebagai
> header. Sudah diperbaiki dan dijaga empat tes regresi.

**Mengapa ini mengunci paling banyak.** Seluruh metrik penelusuran — Hit@1, Hit@5, MRR,
nDCG@5, T3.1, T3.2, T3.3 — memakai relevansi dari `classify_chunk()`, pelabel kata kunci
(K1). Kesepakatan yang diukur T2.2 adalah **batas atas** seberapa jauh angka-angka itu
mencerminkan relevansi sebenarnya. Aturan yang sudah tertanam di keluarannya: κ < 0,40 →
angka penelusuran tidak boleh dilaporkan tanpa kualifikasi eksplisit di setiap penyebutannya;
0,40–0,60 → dilaporkan dengan catatan; > 0,60 → pelabel memadai untuk tujuan ini.

---

## #5 — Pemilihan model: RF, LSTM, atau GBM

Ini keputusan yang paling banyak konsekuensinya, dan **naskah yang sudah ditulis bergantung
padanya.**

### Angka lengkap tiga model, 6 fold lintas-pasien, jendela tersegmentasi

| | RF | GBM | LSTM |
| --- | ---: | ---: | ---: |
| RMSE +30 mnt | 21,08 ± 1,15 | 20,72 ± 1,16 | **20,60 ± 1,33** |
| RMSE +60 mnt | 34,78 ± 1,50 | **33,81 ± 1,44** | 34,50 ± 1,45 |
| Clarke A+B +30 | 94,35% | 94,69% | **94,89%** |
| Clarke A+B +60 | 85,34% | **85,83%** | 85,50% |
| **ukuran model** | 322 / 390 MB | **0,485 MB** | **0,124 MB** |
| **waktu latih** | 708 / 690 dtk | **3,4 / 3,5 dtk** | 477 dtk |
| inferensi | 0,0049 ms | **0,0028 ms** | — |
| **sensitivitas hipo +30** | 21,68 ± 4,72% | 23,93 ± 5,85% | **32,11 ± 6,21%** |
| **sensitivitas hipo +60** | 4,36 ± 1,97% | 2,42 ± 2,03% | **10,56 ± 6,75%** |
| bias pada hipo +30 | +21,03 | +19,02 | **+17,81** |
| **hipo berat terlewat +30** | 588 | 495 | **410** |
| **hipo berat terlewat +60** | 933 | 957 | **743** |

Seluruh selisih RMSE antar-model **di bawah SD antar-fold** menurut definisi gabungan dan maks
per model, sehingga tidak dinarasikan sebagai keunggulan akurasi. Lihat #8 — di bawah definisi
SD selisih, tiga di antaranya justru lolos.

### Apa yang tiap angka dukung

**GBM mengalahkan RF pada setiap dimensi yang diukur**: RMSE lebih rendah di kedua horizon,
Clarke A+B lebih tinggi, **665× lebih kecil**, **208× lebih cepat dilatih**, 1,8× lebih cepat
inferensi, dan bias hipoglikemia +30 menit signifikan lebih baik (+2,01 mg/dL, p=0,031, ketiga
definisi SD sepakat).

**LSTM tetap memegang aspek paling kritis secara klinis.** Sensitivitas hipoglikemia +30
menit 32,11% lawan RF 21,68% dan GBM 23,93%. GBM lawan LSTM: −8,18 poin, p=0,031, **layak
dinarasikan menurut ketiga definisi SD**. Di +60 menit GBM bahkan **lebih buruk daripada RF**
(2,42%), dengan satu fold mencatat **0,0%**.

**Mekanismenya terkonfirmasi, dan itu membatasi harapan pada keluarga model pohon.** GBM
meminimalkan kerugian kuadrat yang sama dengan RF. Penyusutan ke tengah pada kelas langka
melekat pada **fungsi kerugian**, bukan pada keluarga model — karena itu memindahkan RF ke GBM
tidak menyelesaikan masalah hipoglikemia. Yang mungkin menyelesaikannya adalah **kerugian
berbobot kelas atau kerugian asimetris**, dan itu belum diuji.

### Konsekuensi terhadap naskah yang SUDAH ditulis — revisi yang menunggu

**Bab II subbab II.4.1 membenarkan pemilihan Random Forest atas dua alasan, dan percobaan
proyek ini melemahkan keduanya.**

**Alasan 1 — keterjelasan kontribusi fitur.** Yang sebenarnya dimiliki RF bukan
"dapat dijelaskan" (permutation importance berlaku untuk model apa pun) melainkan bahwa
kontribusi fitur tersedia **bawaan dan murah** lewat `feature_importances_`. **J7
menggugurkan tepat keunggulan itu:** `feature_importances_` adalah MDI, dan MDI **tidak boleh
dilaporkan** karena bias kardinalitas — `iob` punya 165.950 nilai unik lawan `glucose` 361,
dan MDI melebihkan kontribusi insulin/karbohidrat 21,67% lawan angka sebenarnya **15,92%**.
Angka yang dilaporkan adalah permutation importance, yang harus dihitung terpisah **pada model
apa pun, termasuk RF**. Jadi satu-satunya keterjelasan yang RF miliki bawaan adalah
keterjelasan yang ternyata **tidak dapat dipakai**.

**Alasan 2 — kebutuhan komputasi rendah.** **T1.1 sudah mencabutnya** (RF 708 dtk lawan LSTM
477 dtk; 322 MB lawan 0,124 MB), dan **T4.1 memperlebar jaraknya**: GBM 3,4 detik dan 0,485 MB.
RF adalah yang **paling mahal** dari ketiganya pada waktu latih dan ukuran.

**Yang perlu direvisi:** subbab II.4.1 tidak lagi dapat mempertahankan kedua alasannya. Ini
pekerjaan penulisan, bukan percobaan.

### Pilihan yang tersedia

| Pilihan | Yang didapat | Yang dibayar |
| ------- | ------------ | ------------ |
| **A. Pertahankan RF** | tidak ada angka hilir yang batal | II.4.1 harus dibenarkan ulang dengan alasan baru; RF paling mahal dan sensitivitas hiponya terendah di +30 mnt |
| **B. Pindah ke GBM** | 665× lebih kecil, 208× lebih cepat, RMSE dan Clarke lebih baik | **membatalkan seluruh angka hilir**: crossfold retrieval, realcases, T3.1–T3.3, kalibrasi konformal, RAGAS (terkunci kuota); GBM tak punya `feature_importances_` sama sekali |
| **C. Pindah ke LSTM** | sensitivitas hipoglikemia terbaik, model terkecil | membatalkan seluruh angka hilir; waktu latih 477 dtk; keterjelasan menuntut permutasi terpisah |
| **D. RF sekarang, GBM sebagai saran pengembangan** | tidak ada yang batal; angka GBM tetap dilaporkan sebagai pembanding | narasi harus jujur bahwa pembanding tak-tersetel mengalahkan produksi |

**Yang saya sudah lakukan sesuai prapendaftaran:** model produksi **tidak diubah** sesi ini,
karena mengganti prediktor membatalkan seluruh angka hilir. Pilihan D adalah keadaan sekarang
secara bawaan — tetapi ia **pilihan**, bukan kelambanan, dan perlu dinyatakan demikian.

**Catatan kejujuran yang wajib menyertai apa pun pilihannya:** hiperparameter GBM **bawaan,
tanpa penyetelan**, sehingga angkanya **batas bawah** kemampuan gradient boosting. RF juga
tidak pernah disetel ekstensif, sehingga perbandingannya sepadan — tetapi keduanya berarti
pemenang sebenarnya belum diketahui.

---

## #6 — Adopsi `hanya_kondisi` ke produksi

**Pertukarannya kini terukur pada kedua sisi**, yang sebelumnya belum pernah.

| | produksi | `hanya_kondisi` |
| --- | ---: | ---: |
| MRR (T3.1, set penyetelan) | 0,6211 | **0,7111** |
| Hit@1 | 0,4222 | **0,7111** (+68% relatif, p=0,00058) |
| potongan unik terjangkau | **24** | 14 |
| % korpus terjangkau | **1,16%** | 0,68% |
| % korpus tidak pernah terambil | 98,84% | **99,32%** |
| porsi pengambilan ke 10 teratas | **75,07%** | 93,42% |
| teks kueri berbeda dari 73 nilai glukosa | **73** | **3** |

Mengadopsinya menukar **41,7% dari jangkauan yang sudah hanya 1,16%** demi kenaikan peringkat.
Mekanismenya terbaca pada baris terakhir: 73 nilai glukosa runtuh menjadi **3 teks kueri**, dan
glukosa 55 dan 68 menghasilkan kueri **identik huruf demi huruf**.

**Perbandingan peringkatnya kebal kontaminasi K1** (produksi dan `hanya_kondisi` berbobot kata
kunci identik, 17), sehingga kenaikan Hit@1 itu sah. **PC-RAG tidak dibatalkan** olehnya —
frasa tetap dipilih `classify_glucose(prediksi)`; yang dibuang hanya numeral di teks kueri.

**Yang perlu diketahui sebelum memutuskan:** `build_ablation_query()` dipakai **12 skrip
evaluasi dan satu tes**, sedangkan **jalur aplikasi tidak memakainya sama sekali** (aplikasi
memakai `_primary_query()`). Mengubahnya karena itu **mengubah seluruh angka evaluasi tanpa
mengubah perilaku aplikasi satu pun**, dan membatalkan
`tests/test_rag_retrieval.py::test_build_ablation_query_simetris_antar_mode`.

| Pilihan | Konsekuensi |
| ------- | ----------- |
| **A. Adopsi** | peringkat naik; jangkauan turun ke 0,68%; seluruh angka penelusuran harus dihitung ulang; satu tes harus diperbarui; aplikasi tidak berubah |
| **B. Jangan adopsi** | jalur evaluasi tetap memakai susunan yang terbukti merugikan peringkat |
| **C. Adopsi bersama perbaikan K12** | naikkan `fetch_k` dan perbanyak frasa kondisi lebih dulu, supaya jangkauan tidak menjadi korban — menggabungkan #4 dan #6 |

Saya condong menyebut **C** paling koheren, karena #4 dan #6 menarik jangkauan ke arah
berlawanan dan memutuskannya terpisah berisiko saling membatalkan. Tetapi itu pengamatan,
bukan keputusan.

---

## #8 — Tetapkan skala derau dan ambang kebermaknaan (BARU)

Rincian penuh di **K14**. Ringkasnya: aturan pelaporan *"selisih yang lebih kecil daripada
simpangan baku antar-fold tidak boleh dinarasikan"* **mencampur dua pertanyaan berbeda**, dan
**lima skala derau** beredar di repo — empat di antaranya dipakai tanpa pernah menyebut yang
mana.

| | pertanyaan | dijawab oleh | TIDAK dijawab oleh |
| - | ---------- | ------------ | ------------------ |
| 1 | efeknya **konsisten**? | SD selisih berpasangan, Wilcoxon | — |
| 2 | besarnya **cukup berarti**? | ambang dari luar data | varians antar-kelompok |

**Empat klaim keunggulan berstatus RAPUH**, dan kerapuhannya memotong dua arah:

| klaim | selisih | SD gab | SD maks | SD selisih | keadaan sekarang |
| ----- | ------: | -----: | ------: | ---------: | ---------------- |
| T2.1 h12 sens RF vs LSTM | −6,203 | **5,862** | 6,754 | 6,877 | **dinarasikan**, lolos hanya definisi 1 |
| T4.1 h6 RMSE RF vs GBM | +0,358 | 1,171 | 1,163 | **0,135** | **ditahan**, akan lolos definisi 3 |
| T4.1 h12 RMSE RF vs GBM | +0,976 | 1,550 | 1,499 | **0,388** | **ditahan**, akan lolos definisi 3 |
| T1.1 h6 RMSE RF vs LSTM | +0,478 | 1,266 | 1,329 | **0,239** | **ditahan**, akan lolos definisi 3 |

Klaim yang menyatakan keunggulan LSTM pada hipoglikemia **mantap di +30 menit** (ketiga
definisi sepakat) tetapi **rapuh di +60 menit**. Kata *"kedua horizon"* pada HANDOFF sudah
dikualifikasi.

**Usulan, untuk diperiksa bukan ditelan:**
1. **Konsistensi** → SD selisih berpasangan (`std(A−B)`), karena untuk rancangan berpasangan
   itulah yang mengukur kekonsistenan efek antar-fold; varians per model memuat keragaman
   pasien yang justru saling hapus dalam pemasangan.
2. **Kebermaknaan** → ambang orde besaran dari ISO 15197:2013 (±15 mg/dL untuk glukosa
   < 100 mg/dL, dikutip KB-02 PERKENI-2021 hal. 25). Selisih RMSE antarmodel 0,358–0,976 mg/dL
   berada **dua orde besaran di bawah** toleransi per-pengukuran alat yang menghasilkan
   datanya.

**Dua kejujuran yang menyertai usulan 2.** Ia **analogi berdasar literatur, bukan penerapan
ISO pada perbandingan model** — ISO mengatur akurasi alat ukur. Dan ia **diadopsi setelah
hasil terlihat**; yang membuatnya tetap dapat dipertahankan adalah ia tidak menyebut satu pun
angka proyek ini, sehingga tidak dapat dipilih demi meloloskan klaim tertentu.

**Ambang mg/dL tidak berlaku untuk sensitivitas hipoglikemia**, karena satuannya poin persen
deteksi kejadian. Klaim sensitivitas LSTM berdiri atau jatuh murni pada pertanyaan konsistensi.

**Akarnya sama dengan T1.1b:** di sana margin kesetaraan tak pernah ditetapkan di muka, di sini
ambang kebermaknaan. Dua kejadian, satu akar.

**Tidak satu pun angka lama diubah.** Memilih definisi sekarang berarti memilih setelah tabel
auditnya terlihat, dan itu yang dilarang Bagian C — karena itu keputusannya diserahkan, bukan
saya ambil.

---

## #2, #3, #4, #7 — ringkas

**#2 Ganti bundle RF ke 50/12?** 297,15 → **11,09 MB** dengan biaya **+0,081 mg/dL** RMSE;
Clarke A+B justru naik 94,90% → 94,93%. **T1.1b TIDAK membuktikan kesetaraan** — uji beda
dipakai sebagai uji kesetaraan, dan margin TOST tak pernah ditetapkan. Yang berdiri sendiri
tanpa uji hipotesis: pertukaran 0,081 mg/dL untuk −96,3% ukuran. Bila #5 memilih GBM,
keputusan ini **gugur dengan sendirinya** (GBM 0,485 MB).

**#3 Tinjau ulang `chunk_size` 900?** Dipertahankan pada Tahap 0 karena selisih B4 tidak
signifikan (p=0,078). Saat itu **belum diketahui** bahwa 900 membuang **8,23% token korpus**
dan KB-03 — dokumen sumber evaluasi RAGAS — kehilangan **48,84%** chunk-nya akibat pemotongan
diam-diam pada 256 token. Mengubahnya menuntut **indeks ulang** dan penghitungan ulang seluruh
angka penelusuran.

**#4 Kerjakan K12?** Jangkauan produksi **1,16% korpus**; 98,84% tidak pernah terambil oleh
nilai glukosa mana pun. Tuasnya **keragaman kueri** dan **`fetch_k`** — keduanya parameter
tanpa indeks ulang, jadi pekerjaannya kecil–sedang. Kandidat perbaikan berdampak tertinggi yang
tersisa di sisi penelusuran. Berkaitan langsung dengan #6.

**#7 Uji kestabilan tiga metrik RAGAS?** ~90 panggilan LLM. Hanya `faithfulness` yang punya
data run berulang (labil per kasus: selisih median 0,18, maksimum 0,56; rerata stabil 0,003).
**SUDAH diukur 13 Agustus 2026 (keputusan #7).** `context_precision` dan `context_recall`
**reprodusibel sempurna**; `answer_relevancy` **labil dan reratanya bergeser 0,176**,
wajib dikutip sebagai rentang **0,587–0,763**. Lihat K10 dan K15. Kuota `gemini-3.5-flash-lite` 500 RPD memadai — satu hari kerja.

---

## Yang MASIH BELUM DIUKUR, dan sengaja dibiarkan demikian

Ditulis supaya tidak ada yang mengira hal-hal ini sudah terukur.

| Yang belum diukur | Mengapa dibiarkan | Di mana batasnya dinyatakan |
| ----------------- | ----------------- | --------------------------- |
| **Optimum `tau` di bawah 120/60** | RMSE naik monoton pada kedua sumbu, sehingga sapuan **tidak mengurung optimum**. Batas atas efeknya di seluruh ruang tau ~0,44 mg/dL, diturunkan dari biaya permutasi `iob` — **sedikit di atas** ambang peka 0,3936 | T1.4, `sensitivitas_tau_h6.json` |
| **T1.4 pada horizon +60 menit** | Hanya h6 diukur. Pada +60 mnt RMSE dasar naik ~21 → ~34 dan bobot relatif IOB/COB dapat berbeda. **Klaim ketidakpekaan dibatasi ke +30 menit** | T1.4, `batas_lingkup` |
| **Mutu JAWABAN pada bentuk kueri aplikasi** | T3.3 mengukur peringkat penelusuran, bukan mutu jawaban. Mutu jawaban diukur RAGAS dan tidak dijalankan pada bentuk aplikasi | K13; T3.3 `yang_TIDAK_diukur` |
| **Interaksi `insulin_tau` × `carbs_tau`** | Rancangan OAT tidak dapat menangkapnya. Kedua sumbu datar, sehingga interaksinya hampir pasti juga datar — tetapi itu penalaran, bukan pengukuran | T1.4, `rancangan.keterbatasan` |
| **Apakah GBM tersetel mengalahkan RF tersetel** | Keduanya sengaja tak-tersetel agar sepadan. Angka GBM adalah **batas bawah** | T4.1, `gbm.akibat` |
| **Kerugian berbobot kelas untuk hipoglikemia** | Mekanisme penyusutan terbukti melekat pada fungsi kerugian, bukan keluarga model — tetapi kerugian alternatif belum diuji sama sekali | T4.1, saran pengembangan |
| **Apakah K12 MENYEBABKAN tiga temuan lain** | Yang ada satu sebab yang konsisten dengan ketiganya dan terukur besarnya. Pembuktian menuntut menaikkan jangkauan lalu memeriksa apakah ketiga pola ikut berubah | K12; `konsentrasi_penelusuran.json` |
| **`lambda_mult` dan `chunk_size` pada berkas crossfold** | `crossfold.json` hanya merekam `top_k`. Bukti bahwa ia dari lambda 0,0 hanyalah **nama direktorinya**. Perbaikannya menuntut run ulang 2 jam 3 menit | K14 area; T5.2 `verifikasi_provenans_crossfold` |
| **Berapa persen catatan logbook nyata yang ditolak** | Belum ada pengguna nyata; hanya dapat diperoleh dari uji pakai, yang bagian dari K5 | K11 |
| **Validasi klinis apa pun** | Memerlukan partisipasi klinisi dan persetujuan etik | K5 — **di luar lingkup** |

---

## Bila ingin mengerjakan satu hal saja

**#8 — tetapkan skala derau dan ambang kebermaknaan.** Setelah T2.2 selesai, inilah yang
tersisa dan tidak menuntut komputasi sama sekali: ia **keputusan, bukan pekerjaan**. Ia
membebaskan empat klaim keunggulan dari status RAPUH, dan tanpanya Bab VI harus menyebut
definisi SD pada setiap klaim satu per satu.

Kalau yang dicari justru yang paling mengubah isi laporan, itu **#5** — dan #5 menuntut
keputusan tentang naskah II.4.1 yang sudah ditulis, bukan sekadar tentang model.
