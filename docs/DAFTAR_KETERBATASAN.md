# Daftar Keterbatasan — bahan langsung Bab VII

> Dikumpulkan sepanjang rangkaian percobaan Bagian A, Bagian B, dan Tahap 0-5.
> Untuk tiap butir dicatat: **apa** keterbatasannya, **mengapa** tidak diperbaiki,
> **dampaknya** terhadap kesimpulan yang dapat ditarik, dan **perkiraan pekerjaan**
> untuk mengatasinya.
>
> Besar pekerjaan: **kecil** < 1 jam · **sedang** 1-4 jam · **besar** > 4 jam ·
> **di luar lingkup** memerlukan sumber daya yang tidak tersedia dalam TA ini.
>
> Terakhir diperbarui: 7 Agustus 2026.

---

## K1. Pelabel relevansi berbasis kata kunci, bukan penilaian manusia

**Apa.** Seluruh metrik penelusuran (Hit@k, MRR, nDCG, precision) menetapkan relevansi
lewat `classify_chunk()` di `scripts/ablation_rag_fullkb.py` — pencocokan kata kunci
berbobot, bukan penilaian klinisi.

Tiga akibat terukur:

| Akibat | Angka |
|---|---|
| Chunk yang tidak pernah dapat dihitung relevan | **49,2%** jatuh ke kelas "lain" |
| Ketidaksetaraan antar-bahasa | Inggris 53,2% "lain" vs Indonesia 41,7% — bias **1,24x** |
| Sirkularitas dengan metode leksikal | Kueri memuat kata kunci pelabel: hipoglikemia **7 dari 18** bobot, hiperglikemia 5 dari 26, normal 5 dari 11 |

**Mengapa tidak diperbaiki.** Memerlukan pelabelan manual oleh klinisi atas ratusan
pasangan kueri-chunk. T2.2 mengukur *tingkat kesesuaian* pelabel, bukan menggantikannya.

### T2.2 SUDAH DIUKUR — dan hasilnya lemah (11 Agustus 2026)

40 pasangan dinilai manusia secara buta (label otomatis disembunyikan, urutan diacak
berbenih tetap, pasangan diambil dari retrieval sungguhan).

| | |
|---|---|
| kesepakatan mentah | **52,5%** |
| **Cohen's κ** | **0,2505 — lemah** |
| tebakan konstan "hiperglikemia" akan mencapai | **77,5%** |

**Pelabel otomatis kalah dari penebak konstan kelas mayoritas** pada sampel ini. Itu bukan
pengganti κ — κ sudah mengoreksi kesepakatan kebetulan lewat kedua marginal — melainkan cara
kedua membaca angka yang sama, dan arahnya sama-sama merugikan.

**Pola ketidaksesuaian: TERLALU KETAT, bukan terlalu longgar.** 17 dari 19 ketidaksesuaian
adalah pelabel gagal mengenali hiperglikemia:

| pelabel otomatis | penilaian manusia | n |
|---|---|---:|
| `lain` | hiperglikemia | 8 |
| `hipoglikemia` | hiperglikemia | 5 |
| `normal` | hiperglikemia | 4 |
| `hiperglikemia` | hipoglikemia | 2 |

| kelas | presisi otomatis | recall otomatis |
|---|---:|---:|
| hipoglikemia | 0,500 | 0,714 |
| hiperglikemia | 0,875 | **0,452** |
| normal | **0,000** | tak terdefinisi (manusia tak pernah memilihnya) |
| `lain` | **0,200** | 1,000 |

**Mekanismenya terukur.** Pelabel memberi kelas hanya bila kata kuncinya cocok; **10 dari 40
potongan berbobot kata kunci NOL** dan otomatis jatuh ke `lain`. Potongan yang tidak
disepakati berbobot lebih rendah (2,32 lawan 3,29), sehingga kegagalannya terkonsentrasi pada
teks klinis yang tidak memakai kosakata ambang yang diharapkan pelabel. Kesepakatan paling
buruk pada potongan yang dimunculkan kueri kondisi **normal**: 4 dari 13 (30,8%).

**Aturan yang kini TERPICU.** Aturan yang sudah ditanam pada keluaran T2.2 sejak sebelum
penilaian: **κ < 0,40 → angka penelusuran tidak boleh dilaporkan tanpa kualifikasi eksplisit
di setiap penyebutannya.** Berlaku untuk Hit@1, Hit@5, MRR, nDCG@5, T3.1, T3.2, T3.3,
crossfold, realcases, dan K12.

**Yang TETAP sah:** perbandingan **antar-konfigurasi yang memakai pelabel sama** tetap dapat
dibaca sebagai perbandingan relatif — dan itulah pemakaiannya di seluruh laporan.

**Yang menjadi LEBIH rapuh, dan belum diperiksa:** kesalahan pelabel bersifat **sistematis**
(gagal mengenali hiperglikemia), bukan acak. Kesalahan sistematis **dapat** mencondongkan
perbandingan antar-konfigurasi bila kedua konfigurasi berbeda dalam kelas apa yang mereka
ambil. Itu belum diukur, dan tidak boleh diasumsikan tidak terjadi.

**Kualifikasi atas angka κ itu sendiri.** Sebaran penilaian manusia sangat timpang pada
sampel ini — 31 hiperglikemia, 7 hipoglikemia, 2 `lain`, dan **nol `normal`** — sehingga baik
κ maupun perbandingan terhadap tebakan mayoritas bersifat **labil**. Tiga penilaian ditulis
sebagai teks bebas dan dikodekan atas konfirmasi eksplisit pembimbing; teks aslinya disimpan
verbatim di `evaluation/verifikasi_relevansi_CATATAN.json`. Salah satunya menunjuk
keterbatasan skema itu sendiri: satu potongan dapat membahas hiperglikemia **dan**
hipoglikemia sekaligus, dan skema empat kelas berlabel tunggal memaksa memilih satu.

**Dampak.** Menjadi **batas atas terukur** — bukan lagi asumsi — bagi seluruh angka
penelusuran yang pernah dilaporkan. Skor tidak boleh dibaca sebagai mutu penelusuran mutlak.
Sirkularitas membuat hasil B3 (BM25) tidak sah dipakai sebagai bukti keunggulan.

**Perkiraan pekerjaan.** Verifikasi sampel: **selesai**. Penggantian penuh dengan pelabel
manusia: **di luar lingkup**. Memperluas sampel agar κ lebih stabil, dan memeriksa apakah
kesalahan sistematis mencondongkan perbandingan antar-konfigurasi: **sedang**.

---

## K2. Korpus evaluasi RAGAS sempit — hanya context_precision yang optimistis

**Apa.** Koleksi evaluasi T1.2 hanya **17 chunk** dari 4 halaman KB-03.

| Besaran | Koleksi evaluasi | Korpus produksi |
|---|---|---|
| Jumlah chunk | 17 | 2.061 |
| `top_k=5` terambil | **29,4%** | **0,24%** — 121x lebih kecil |
| `fetch_k=12` kolam | 70,6% | 0,58% |

**Mengapa tidak diperbaiki.** Kuota Gemini free-tier. Korpus kecil dipilih agar ground
truth dapat diverifikasi manual dan konsumsi kuota tertahan. Lihat K7.

**Dampak.** `context_precision` **terbukti optimistis** — presisi tinggi karena kolam
sempit, bukan karena penelusurannya baik. Angka RAGAS mengukur mutu **pembangkitan** pada
korpus kecil dan **tidak dapat digeneralisasi** ke korpus produksi.

> **Koreksi penting.** Semula dinyatakan bahwa `context_recall` **juga** optimistis, dengan
> alasan kolam `fetch_k` mencakup 70,6% koleksi sehingga konteks acuan pasti terjangkau.
> **Terbantah oleh pengukuran**: `context_recall` = 0,450, sama rendahnya dengan precision.
> Yang menentukan bukan keberadaan konteks di kolam, melainkan apakah ia lolos ke `top_k`
> setelah MMR memilih. Peringatan untuk `context_recall` **dicabut**.

**Perkiraan pekerjaan.** Memperluas korpus evaluasi ke 20-30 halaman: **sedang**, tetapi
menuntut verifikasi manual ground truth yang sebanding.

---

## K3. Penanda `grounded` tidak mendeteksi konteks yang tidak relevan

**Apa.** `RAGPipeline.answer()` menetapkan `grounded = bool(retrieved_docs)` — bernilai
`True` selama **ada dokumen yang dikembalikan**, tanpa memeriksa apakah dokumen itu
relevan.

Terbukti pada Tahap D: kedua kasus negatif memperoleh `grounded = True` padahal kelima
konteksnya tidak memuat jawaban sama sekali.

**Mengapa tidak diperbaiki.** Ditemukan pada tahap akhir rangkaian percobaan; mengubahnya
menyentuh kontrak keluaran pipeline dan lapisan antarmuka.

**Dampak — dan ini yang paling perlu dinyatakan terus terang.**

Klaim Tugas 1B butir 7 tentang penanganan retrieval kosong hanya berlaku untuk kasus
**nol dokumen**, bukan kasus **dokumen tidak relevan** yang jauh lebih sering terjadi.
Pada korpus produksi, retrieval hampir selalu mengembalikan `top_k` dokumen, sehingga
`grounded` praktis selalu `True` dan tidak pernah menjadi pengaman.

**Sistem tetap LULUS kedua kasus negatif meskipun penanda ini tidak berfungsi sebagaimana
dimaksud.** Yang menyelamatkan adalah **kepatuhan model pada instruksi prompt** (aturan 3
dan 4 SYSTEM_PROMPT), bukan mekanisme deterministik di kode.

Konsekuensinya: **pengaman KNF-03 saat ini bertumpu pada perilaku model.** Untuk sistem
pendukung keputusan klinis, pengaman yang bergantung pada kepatuhan model lebih lemah
daripada pengaman yang dijamin kode — ia dapat berubah bila model diganti, versinya
diperbarui, atau promptnya disunting. Penggantian model dari 2.5 ke 3.5 pada proyek ini
justru memperlihatkan hal itu: 2.5 **mengarang** pada konteks yang salah, 3.5 mengakui
keterbatasan. Perilaku yang sama sekali berbeda dari kode yang sama.

**Arah pengembangan (jangan dikerjakan sekarang).** Penjagaan sebaiknya memakai **ambang
skor kemiripan minimum**, bukan sekadar jumlah dokumen bukan nol. `MMRRetriever.retrieve()`
sudah mengembalikan medan `similarity` per dokumen, sehingga bahannya tersedia. Yang perlu
diputuskan adalah nilai ambangnya, dan itu menuntut kalibrasi tersendiri.

**Perkiraan pekerjaan.** Mengganti `grounded` dengan ambang kemiripan: **kecil**.
Mengalibrasi ambangnya secara sah: **sedang**.

---

## K4. Ekstraksi PDF merusak label dekorasi gambar

**Apa.** `PyPDF2` meratakan tata letak menjadi teks linear. Label sumbu dan legenda diagram
menjadi barisan tanpa makna, contohnya
`1 2 3 TINGGI SEDANG RENDAH JUMLAH INJEKSI KOMPLEKSITAS FLEKSIBILITAS`.

**Cakupan terukur: 25 dari 2.061 chunk (1,2%)**, terkonsentrasi di KB-03 (6,2%). Delapan
dari dua belas dokumen nol.

**Yang TIDAK rusak — diperiksa langsung.** Tabel data **selamat** dengan pasangan
label-nilai utuh:

- Tabel VIII.1 klasifikasi hipoglikemia: `Level 1) <= 70 mg/dL ... Level 2) < 54 mg/dL`
- Algoritma dosis hal. 23: `Awal : 5 - 10 unit/hari`
- Tabel koreksi kalium hal. 45: `<3 : 75`, `3-4,5 : 50`

Dari 25 chunk yang ditandai heuristik, 15 memuat istilah dosis — dan seluruhnya terbaca
baik. Heuristik menandainya karena kepadatan kalimat rendah dan banyak angka, yang memang
**sifat tabel dosis**, bukan bukti kerusakan.

**Mengapa tidak diperbaiki.** Kerusakan terbatas pada dekorasi gambar yang tidak membawa
informasi klinis. Perbaikannya menuntut penggantian pustaka ekstraksi, **indeks ulang
korpus**, dan **penghitungan ulang seluruh angka penelusuran**.

**Dampak.** Kecil. Sumber ambang 70 dan 54 di `src/constants.py` tetap terlacak ke Tabel
VIII.1. Yang hilang hanya konteks visual diagram, dan tidak ada angka dosis yang kehilangan
labelnya.

**Perkiraan pekerjaan.** Beralih ke `pdfplumber.extract_tables()`: **sedang (2-4 jam)**,
ditambah indeks ulang dan penghitungan ulang seluruh angka penelusuran.

---

## K5. Tidak ada validasi klinis

**Apa.** Tidak satu pun keluaran sistem pernah dinilai oleh dokter. Seluruh penilaian
mutu bersandar pada metrik otomatis dan kesesuaian terhadap teks pedoman.

**Mengapa tidak diperbaiki.** Memerlukan partisipasi klinisi dan, untuk penilaian yang
sah, persetujuan etik.

**Dampak.** Sistem **tidak boleh** diklaim aman atau bermanfaat secara klinis. Seluruh
angka menunjukkan kesesuaian terhadap pedoman tertulis, bukan kelayakan pemakaian pada
pasien. Klaim yang sah terbatas pada: sistem mengambil rujukan yang tepat topik pada
sekian persen kasus, dan jawabannya didukung konteks pada skor sekian.

**Perkiraan pekerjaan.** **Di luar lingkup.**

---

## K6. Perlakuan pembuangan disclaimer sebelum penilaian RAGAS

**Apa.** Teks disclaimer dibuang dari jawaban **sebelum** dinilai RAGAS, baik sufiks yang
ditambahkan `_ensure_disclaimer()` maupun paragraf yang ditulis LLM sendiri.

**Mengapa demikian, dan mengapa ini bukan manipulasi skor.** `faithfulness` menilai apakah
pernyataan didukung konteks. Disclaimer adalah **artefak sistem yang tetap**, bukan
pernyataan tentang pasien, dan tidak akan pernah didukung konteks mana pun. Membiarkannya
berarti mengukur disclaimer, bukan mutu pembangkitan.

Sebagai gantinya, kepatuhan disclaimer diperiksa **terpisah** lewat pencocokan teks tanpa
panggilan LLM, pada dua tingkat:

| Tingkat | Makna | Hasil |
|---|---|---|
| MODEL | LLM menulis disclaimer sendiri | 9/10 dan 10/10 |
| SISTEM | frasa wajib ada setelah penegakan | 10/10 — **selalu 100% menurut konstruksi** |

**Dampak.** Skor `faithfulness` yang dilaporkan menggambarkan isi klinis jawaban, bukan
jawaban apa adanya. Ini **harus dinyatakan** di Bab VI agar tidak terbaca sebagai
penyaringan yang menguntungkan. Verifikasi otomatis mencatat 0 dari 10 jawaban menyisakan
disclaimer pada kedua tahap.

**Catatan proses.** Penghapus disclaimer memerlukan **empat iterasi**; versi ketiga
terlalu longgar dan sempat memotong kalimat sah "Jawab tanpa disclaimer sama sekali"
menjadi "Jawab tanpa" — merusak data tanpa tanda apa pun. Yang menangkapnya adalah
penjagaan sisa-disclaimer yang ditambahkan bersamaan.

**Perkiraan pekerjaan.** Tidak perlu diperbaiki; perlu **dinyatakan**.

---

## K7. Kuota LLM membatasi rancangan evaluasi

**Apa.** Kuota free-tier Gemini berbeda sampai **25 kali lipat** antargenerasi model:

| Model | RPM | RPD |
|---|---|---|
| gemini-3.5-flash-lite | 15 | **500** |
| gemini-2.5-flash-lite | 10 | **20** |

Kuota berlaku **per model**, bukan per akun.

**Mengapa menjadi keterbatasan.** Dengan 20 RPD, evaluasi RAGAS (~120 panggilan) menuntut
**6 hari** dan praktis tidak dapat dijalankan. Ini yang memaksa korpus evaluasi dibatasi
4 halaman (K2).

**Dampak.** Pemilihan model ditentukan **kuota, bukan mutu**. Rancangan evaluasi mengikuti
batas lingkungan, bukan sebaliknya. Perlu dinyatakan di Bab V sebagai pertimbangan
lingkungan implementasi.

**Catatan proses.** Angka 20/hari sudah tercatat benar pada 8 Juli dari pengalaman, lalu
keliru dibatalkan pada Tugas 6 berdasarkan pembacaan dokumentasi vendor. Kesalahan itu
menyebabkan rencana 120 panggilan disusun di atas asumsi 1.000 RPD. Aturan yang berlaku
sekarang: **bila dokumentasi vendor bertentangan dengan pengukuran proyek, yang berlaku
pengukuran.**

**Perkiraan pekerjaan.** Sudah diatasi dengan berpindah model. Untuk evaluasi berskala
besar: memerlukan tier berbayar, **di luar lingkup**.

---

## K8. Pemeriksa angka pada kasus negatif menghasilkan positif palsu

**Apa.** `scripts/eval_kasus_negatif.py` menandai angka spesifik yang tidak ditemukan di
konteks sebagai "tak bersumber". Pada kasus N2 ia menandai **"230,0 mg"**, padahal itu
**kadar glukosa pasien sendiri** yang diteruskan lewat `patient_state`, bukan dosis yang
dikarang.

**Mengapa tidak diperbaiki.** Kelulusan kasus negatif ditentukan oleh **pengakuan
keterbatasan**, bukan oleh kolom ini, sehingga kesimpulan 2/2 LULUS tetap sah.

**Dampak.** Kolom `angka_tak_bersumber` pada `results/ragas/kasus_negatif.json` **tidak
boleh dibaca apa adanya**. Ia perlu diperiksa manual sebelum dikutip.

**Perkiraan pekerjaan.** Menambahkan nilai `patient_state` ke daftar sumber sah:
**kecil**.

---

## K9. top_k produksi sempat berbeda dari top_k evaluasi

**Apa.** Sampai 7 Agustus 2026, produksi memakai `top_k=4` sedangkan seluruh skrip evaluasi
memakai `TOP_K=5` hardcoded. Setiap angka Hit@5, MRR, precision@5, dan nDCG@5 yang pernah
dilaporkan mengukur kedalaman yang tidak pernah dipakai aplikasi.

**Status: SUDAH DIPERBAIKI.** `rag.top_k_retrieval` dinaikkan ke 5 agar sesuai evaluasi,
diverifikasi sampai instance retriever.

**Dampak sisa.** Angka yang dihitung **sebelum** perbaikan tetap menggambarkan kedalaman 5
sementara aplikasi menampilkan 4. Angka setelah perbaikan konsisten. Dicatat di sini agar
pembaca laporan tidak menemukan ketidakcocokan itu tanpa penjelasan.

**Perkiraan pekerjaan.** Selesai.

---

## K10. Kesahihan metrik faithfulness — bukan ukuran mutu, dan labil per kasus

Terpisah dari K3: K3 menyangkut **kerapuhan pengaman**, K10 menyangkut **kesahihan metrik**.

**Apa.** `faithfulness` tidak mengukur mutu jawaban dan tidak monoton terhadap kebenaran.
Dua mekanisme:

**(a) Ia hanya menguji dukungan konteks, bukan kesesuaian dengan pertanyaan.** Kutipan
verbatim dari konteks yang **salah** tetap memperoleh skor sempurna.

**(b) Ia labil per kasus.** Dua run dengan konfigurasi identik:

| Besaran | Nilai |
|---|---|
| Selisih absolut per kasus, median | **0,18** |
| Selisih absolut per kasus, maksimum | **0,56** (E04: 0,56 → 0,00) |
| Selisih rerata 10 kasus | **0,003** (0,681 vs 0,678) |

**Tiga bukti.**

1. Skor **1,000** pada dua jawaban yang **salah topik** (Tahap A awal).
2. Skor **1,000** pada model yang **mengarang** (2.5) dan model yang **mengakui
   keterbatasan** (3.5), pada kasus E01 yang sama. Metrik tidak membedakan keduanya.
3. Ketidakstabilan run-ke-run di atas.

**Bukti keempat DICABUT.** Sempat diusulkan bahwa metrik "menghukum panjang jawaban"
sehingga jawaban terpendek paling aman. Diuji pada 10 kasus: korelasi panjang karakter
**+0,163** dan jumlah kalimat **−0,089** — praktis nol, dan yang panjang bahkan berarah
positif. E08 (298 karakter) memperoleh 1,000 sedangkan E04 (396 karakter) memperoleh 0,000.
Klaim itu berasal dari satu pasang pengamatan pada E02 yang digeneralisasi tanpa diuji;
setelah bukti 3 diketahui, selisih E02 itu tidak memerlukan penjelasan mekanisme apa pun.
Pencabutan dicatat, tidak dihapus.

**Jumlah pernyataan tidak dapat diperoleh.** `faithfulness` adalah rasio
pernyataan-didukung terhadap total pernyataan, sehingga penafsirannya menuntut kedua angka
itu. RAGAS 0.2.6 **tidak memaparkannya**: medan `traces` dan `ragas_traces` pada
`EvaluationResult` ada tetapi tidak terisi daftar pernyataan (0 dari 10 sampel). Kolom
`n_pernyataan_faithfulness` disimpan kosong sebagai catatan bahwa angka dicari dan tidak
ditemukan. **Tanpa angka itu, penafsiran skor terbatas.**

**Kestabilan tiga metrik lain BELUM DIUJI.**

| Metrik | Run berulang | Status |
|---|---|---|
| faithfulness | ada (2 run) | labil per kasus |
| answer_relevancy | tidak ada | **belum diuji** |
| context_precision | tidak ada | **belum diuji** |
| context_recall | tidak ada | **belum diuji** |

Ketiganya **tidak boleh diasumsikan stabil**. Mengujinya memerlukan run ulang dengan cache
dikosongkan, sekitar 90 panggilan.

**Aturan pelaporan yang mengikat** — ditanamkan pada `results/ragas/summary.json` dan
`results/ragas/contoh_kasus_bab6.json`:

1. Skor `faithfulness` **per kasus tidak boleh dikutip**.
2. Perbandingan antarmodel pada **satu kasus tidak sah**, termasuk E02 (selisih 0,55, masih
   di dalam rentang variasi run).
3. Hanya **rerata atas sepuluh kasus** yang layak dilaporkan.

**Dampak.** Skor `faithfulness` 0,681 hanya sah dikutip sebagai **rerata**, dan hanya
sebagai bukti bahwa jawaban tidak mengarang **di luar konteks** — bukan sebagai bukti
kebenaran jawaban. Yang mengukur kesesuaian dengan pertanyaan adalah `answer_relevancy`
(0,763), dan yang mengukur mutu konteks adalah kedua metrik konteks.

**Perkiraan pekerjaan.** Menguji kestabilan tiga metrik lain: **sedang** (~90 panggilan
LLM). Memperoleh jumlah pernyataan: memerlukan pemanggilan prompt ekstraksi RAGAS secara
terpisah, **sedang**, atau menunggu versi pustaka yang memaparkannya.

---

## K11. Logbook manual tersambung, tetapi sebagian besar catatan akan ditolak

**Apa keterbatasannya.** T4.2 menyambungkan `data/raw/manual_logbook.csv` ke jalur
prediksi, sehingga masukan dokter kini benar-benar dapat membentuk prediksi. Dua hal tetap
tidak terselesaikan.

*Pertama, lima dari sembilan variabel logbook tidak menjadi fitur.* `stress`, `sleep`,
`work`, `illness`, dan `meal_type` disimpan sebagai rekam jejak klinis tetapi tidak masuk
model, karena model produksi dilatih tanpa kolom-kolom itu dan menambahkannya saat
inferensi akan membuat bentuk masukan tidak cocok dengan scaler. Deskripsi KF-01 menyebut
kelima variabel dapat "mendukung prediksi"; yang benar adalah keempat variabel numerik
(`glucose`, `carbs`, `insulin`, `activity`) yang mendukungnya.

*Kedua, dan lebih menentukan: catatan manual sering tidak layak dipakai.* Model dilatih
hanya pada jendela yang jarak antar-barisnya ≤ 30 menit (`max_gap_steps`, Tugas 5).
Catatan logbook dimasukkan pada waktu bebas. Sebuah catatan yang jaraknya berjam-jam dari
pembacaan CGM terdekat menghasilkan jendela di luar sebaran pelatihan, dan
`periksa_kelayakan()` menolaknya.

**Mengapa tidak diperbaiki.** Jalan keluar yang tersedia adalah menginterpolasi jeda
supaya jendelanya rapat. Itu ditolak: menginterpolasi jeda delapan jam menghasilkan dua
belas baris yang tampak sah bagi model **dan bagi dokter**, padahal sebelas di antaranya
karangan. Kekeliruan itulah yang membuat segmentasi jeda sensor diperlukan pada Tugas 5,
dan mengulanginya di sisi antarmuka tidak lebih dapat dibenarkan.

**Dampak pada kesimpulan.** KF-01 dan KF-02 naik dari SEBAGIAN menjadi
**SEBAGIAN-tersambung**, bukan ADA. Klaim yang boleh dibuat: masukan manual dapat masuk ke
jalur prediksi dan sistem menyatakan dengan jelas kapan ia dipakai dan kapan ditolak.
Klaim yang **tidak** boleh dibuat: bahwa sistem memanfaatkan seluruh variabel logbook, atau
bahwa dokter dapat mengandalkan catatan manual sebagai sumber masukan sehari-hari.

**Angka yang belum ada.** Berapa persen catatan nyata yang akan ditolak belum terukur,
karena belum ada pengguna nyata. Angka itu hanya dapat diperoleh dari uji pakai, yang
merupakan bagian dari K5.

**Perkiraan pekerjaan.** Memasukkan kelima variabel sisa sebagai fitur: **besar** —
menuntut pelatihan ulang seluruh model beserta kalibrasi konformalnya, sementara OhioT1DM
nyaris tidak memuat data stres (7 kejadian di seluruh dataset), sehingga fiturnya akan
nyaris nol-varians. Menurunkan angka penolakan: **sedang** — memerlukan kebijakan
interpolasi terbatas yang dibenarkan secara fisiologis, bukan sekadar mengendurkan ambang.

---

## K12. Jangkauan penelusuran hanya 1,16% korpus

**Apa keterbatasannya.** Seluruh rentang glukosa 40–400 mg/dL disapu langkah 5 — 73 nilai,
mencakup setiap angka yang dapat dihasilkan prediktor — dan konfigurasi produksi hanya
menjangkau **24 dari 2.061 chunk**. **98,84% korpus tidak pernah terambil oleh nilai
glukosa mana pun.** Sepuluh potongan teratas menyerap 75,07% seluruh pengambilan.

**Mekanismenya terukur, dan bukan `lambda_mult`.** Dua batas keras bekerja bersamaan:

1. `build_ablation_query()` meruntuhkan seluruh rentang glukosa menjadi **tiga frasa
   kondisi**. Angka glukosa nyaris tidak menggeser embedding: frasa saja menjangkau 14
   potongan, ditambah angka menjadi 24 — angka hanya menyumbang 10.
2. `fetch_k` = 12 membatasi kolam kandidat menjadi ~3 × 12 = 36 potongan; 31 yang
   benar-benar masuk. **MMR tidak dapat mengembalikan potongan di luar kolam**, betapa pun
   relevannya.

`lambda_mult` 0,0 lawan 0,5 menghasilkan 24 lawan 22 potongan — nilai produksi sudah yang
lebih baik dari keduanya, dan selisihnya terlalu kecil untuk menjadi tuas.

**Mengapa ini menaikkan bobot tiga temuan lain.** Pola biner `context_precision` (K2),
jurang Hit@1 23,3% lawan Hit@5 91,7%, dan konteks tidak relevan pada kasus E01 selama ini
dilaporkan sebagai tiga hal terpisah. Ketiganya konsisten dengan **satu sebab**: kolam
kandidat yang membeku membuat peringkat membeku pula.

**Dampak pada kesimpulan.** Setiap angka penelusuran di laporan — Hit@k, MRR, nDCG@5, T3.1,
T3.2 — diukur pada sistem yang hanya menyentuh 1,16% korpusnya sendiri. Angka-angka itu sah
sebagai gambaran **sistem sebagaimana dikonfigurasi**, tetapi **tidak** boleh dibaca sebagai
gambaran mutu korpus atau mutu model embedding. Klaim "korpus 12 pedoman klinis" perlu
disertai keterangan bahwa jalur penelusuran produksi hanya pernah menyentuh sebagian sangat
kecil darinya.

**Yang belum dibuktikan.** Bahwa konsentrasi ini *menyebabkan* ketiga temuan itu. Yang ada
adalah satu sebab yang konsisten dengan ketiganya dan terukur besarnya. Pembuktian menuntut
menaikkan jangkauan lalu memeriksa apakah ketiga pola ikut berubah.

**Perkiraan pekerjaan.** Memperbanyak frasa kondisi (mis. per rentang glukosa, bukan per
tiga kelas) dan menaikkan `fetch_k`: **kecil–sedang**, keduanya parameter, tanpa indeks
ulang. Mengukur ulang seluruh metrik penelusuran sesudahnya: **sedang**. Ini kandidat
perbaikan berdampak tertinggi yang tersisa pada sisi penelusuran.

---

## K13. Angka penelusuran diukur pada bentuk kueri yang tidak dipakai aplikasi

**Apa.** Seluruh metrik penelusuran di laporan — Hit@k, MRR, nDCG@5, T3.1, T3.2, T3.3 —
membangun kuerinya dengan `build_ablation_query()` di `src/rag/ablation_query.py`:

```
Kadar glukosa darah 58 mg/dL. Hipoglikemia, gula darah rendah di bawah 70 mg/dL.
Penyebab, gejala, dan penanganan segera (aturan 15-15).
```

Kueri penelusuran yang benar-benar dijalankan aplikasi dibangun
`PredictionConditionedQueryBuilder._primary_query()` di `src/rag/conditioned_query.py`, dan
bentuknya jauh berbeda — numeral empat kali, klausa tren, faktor kontribusi, dan ditutup
kalimat tanya:

```
Prediksi glukosa 30 menit ke depan: 58.0 mg/dL (dari 112.0 mg/dL, perubahan -54.0 mg/dL,
tren menurun). Status risiko prediksi: BAHAYA - Hipoglikemia. Faktor kontribusi: tren cepat
menurun, aktivitas fisik rendah. Berikan penilaian risiko, tindakan pencegahan, dan
protokol pemantauan untuk kondisi prediksi glukosa 58 mg/dL (BAHAYA - Hipoglikemia) dalam
30 menit ke depan. Sertakan rekomendasi yang bisa dilakukan dokter maupun pasien.
```

**Cakupan pemakaiannya terukur.** `build_ablation_query()` dipakai **12 skrip evaluasi dan
satu tes** (`tests/test_rag_retrieval.py` menegaskan stringnya persis berikut angkanya),
sedangkan **jalur aplikasi tidak memakainya sama sekali**. Konsekuensinya berarah dua:
mengubah `build_ablation_query()` mengubah **seluruh angka evaluasi** tanpa mengubah
**perilaku aplikasi** satu pun.

**Konsekuensinya kini terukur, dan tidak merugikan.** T3.3 menjalankan bentuk aplikasi pada
kerangka T3.1, set penyetelan, 90 kasus:

| lengan | MRR | Hit@1 | bobot kata kunci pelabel |
|---|---:|---:|---:|
| `aplikasi_delta0` | 0,6611 | 0,5444 | 6 |
| `aplikasi_penuh` | 0,6556 | 0,5333 | 6 |
| produksi (`build_ablation_query`) | 0,6211 | 0,4222 | 17 |

**Yang sah dinyatakan hanya bahwa bentuk aplikasi TIDAK LEBIH BURUK.** Selisih +0,0400
**tidak signifikan** (Wilcoxon p=0,441), **dan** bobot kata kunci pelabelnya berbeda (6 lawan
17) sehingga perbandingannya tidak kebal kontaminasi K1. Dua alasan terpisah, masing-masing
sudah cukup untuk melarang klaim "bentuk aplikasi lebih baik".

**Mengapa tidak diperbaiki.** Menyatukan kedua bentuk menuntut memilih salah satunya. Memakai
bentuk aplikasi di evaluasi membatalkan seluruh angka penelusuran yang sudah dihitung;
memakai bentuk evaluasi di aplikasi mengubah perilaku produksi tanpa bukti bahwa itu lebih
baik — T3.3 justru menunjukkan sebaliknya, walau tidak signifikan.

**Dampak.** Angka penelusuran sah sebagai **perbandingan antar-konfigurasi pada bentuk kueri
yang sama**, dan itulah pemakaiannya di seluruh laporan. Yang **tidak** sah adalah membaca
Hit@1 atau MRR sebagai gambaran mutu penelusuran yang dialami pengguna aplikasi. Sesudah
T3.3, batas itu diperlunak: bentuk aplikasi **tidak lebih buruk**, sehingga angka evaluasi
bukan gambaran yang terlalu optimistis — tetapi ia tetap gambaran sistem lain.

**Perkiraan pekerjaan.** Menjalankan seluruh metrik penelusuran ulang pada bentuk aplikasi:
**sedang**. Menyatukan kedua jalur: **sedang–besar**, menuntut keputusan mana yang menang.

---

## K14. Aturan pelaporan efek mencampur dua pertanyaan, dan skala derau tidak pernah ditetapkan

**Apa.** Aturan pelaporan yang berlaku sejak T1.1 berbunyi: *"selisih yang lebih kecil
daripada simpangan baku antar-fold tidak boleh dinarasikan sebagai temuan."* Aturan itu
**mencampur dua pertanyaan yang berbeda**, dan karena itu tidak dapat menjawab keduanya.

| | pertanyaan | dijawab oleh | TIDAK dijawab oleh |
|---|---|---|---|
| 1 | Apakah efeknya **konsisten** antar-kelompok? | SD selisih berpasangan, Wilcoxon tingkat fold | — |
| 2 | Apakah besarnya **cukup untuk berarti**? | ambang dari luar data | varians antar-kelompok |

Varians antar-fold mengukur **keragaman pasien**, bukan ambang kebermaknaan. Memakainya
sebagai ambang adalah **kekeliruan kategori**. Karena keduanya tercampur, "melampaui SD"
kadang berarti *konsisten* dan kadang berarti *besar*, bergantung definisi mana yang dipakai
— dan itulah sebabnya verdiknya berpindah-pindah.

### Lima skala derau beredar di proyek ini

| # | skala | rumus | dipakai di | status |
|---|---|---|---|---|
| 1 | SD gabungan | `np.std(concat([A, B]))` | T2.1 `eval_hipoglikemia.py:245` | dipakai tanpa disebut |
| 2 | SD maks per model | `max(SD_A, SD_B)` | T4.1 RMSE `eval_gradient_boosting.py:414` | dipakai tanpa disebut |
| 3 | SD selisih berpasangan | `np.std(A − B)` | T4.1, dilaporkan sejak temuan ini | paling dapat dipertahankan untuk rancangan berpasangan |
| 4 | SD satu model | `SD_RF` saja | part-6, pencabutan klaim zona D | dipakai tanpa disebut, **dan keliru** |
| 5 | SD bootstrap RMSE | 1.000 resample | T1.4 | **SAH** — ditetapkan di muka pada prapendaftaran, karena T1.4 tidak punya struktur fold |

Nomor 5 tidak menjadi masalah: ia dideklarasikan sebelum hasil terlihat, beserta alasannya.
Yang menjadi masalah adalah **nomor 1 sampai 4 dipakai tanpa pernah menyebut yang mana.**

### Tabel audit — sepakat atau rapuh

Dihitung ulang untuk setiap klaim keunggulan yang bersandar pada aturan ini. **Tidak satu pun
angka lama diubah**; yang dikerjakan hanya menandai.

| klaim | selisih | SD gab | SD maks | SD selisih | verdik |
|---|---:|---:|---:|---:|---|
| T2.1 h6 sens RF vs LSTM | −10,429 | 7,591 | 6,209 | 4,727 | **sepakat layak** |
| T2.1 h6 MAE pada hipo | +2,954 | 2,271 | 2,170 | 1,384 | **sepakat layak** |
| T2.1 h6 bias pada hipo | +3,216 | 2,312 | 2,092 | 1,383 | **sepakat layak** |
| **T2.1 h12 sens RF vs LSTM** | **−6,203** | **5,862** | **6,754** | **6,877** | **RAPUH** |
| T4.1 h6 GBM vs LSTM sens | −8,180 | 7,288 | 6,209 | 3,465 | **sepakat layak** |
| T4.1 h12 GBM vs LSTM sens | −8,139 | 6,437 | 6,754 | 6,641 | **sepakat layak** |
| T4.1 h6 RF vs GBM bias hipo | +2,010 | 1,670 | 1,552 | 0,725 | **sepakat layak** |
| **T4.1 h6 RMSE RF vs GBM** | **+0,358** | 1,171 | 1,163 | **0,135** | **RAPUH** |
| **T4.1 h12 RMSE RF vs GBM** | **+0,976** | 1,550 | 1,499 | **0,388** | **RAPUH** |
| **T1.1 h6 RMSE RF vs LSTM** | **+0,478** | 1,266 | 1,329 | **0,239** | **RAPUH** |
| T1.1 h6 Clarke D RF vs LSTM | +0,699 | 0,594 | 0,602 | 0,460 | **sepakat layak** |
| T1.1 h12 Clarke D RF vs LSTM | −0,065 | 1,056 | 1,130 | 0,379 | sepakat tidak layak |
| T1.1 h12 RMSE RF vs LSTM | +0,288 | 1,483 | 1,499 | 0,476 | sepakat tidak layak |

**Kerapuhannya memotong dua arah, dan itu yang membuatnya serius.** Empat klaim rapuh: satu
(T2.1 h12) sekarang **dinarasikan** dan hanya lolos di bawah definisi 1; tiga (T4.1 h6, T4.1
h12, T1.1 h6) sekarang **ditahan** dan justru akan **lolos** di bawah definisi 3. Pilihan
definisi bukan soal ketat lawan longgar — ia mengubah **isi** kesimpulan.

**Satu klaim yang alasannya sudah diperbaiki.** Pencabutan "RF menghasilkan 27% lebih banyak
kesalahan zona D" semula beralasan SD, memakai definisi 4. Terhitung ulang, 0,699 melampaui
ketiga definisi. **Pencabutannya tetap sah** karena alasan keduanya — di h12 zona D berbalik
tanda (7,66 vs 7,72), tidak tereplikasi antar-horizon. Yang diperbaiki alasannya, bukan
keputusannya.

### Ambang kebermaknaan: memisahkan pertanyaan 2, dengan ambang dari luar

Pertanyaan 2 memerlukan ambang yang tidak berasal dari data ini. Yang dipakai, dan sumbernya
ada di **korpus proyek ini sendiri**:

> ISO 15197:2013 mensyaratkan 95% pemeriksaan glukometer berada pada kisaran **±15 mg/dL**
> bila glukosa < 100 mg/dL, dan ±15% bila ≥ 100 mg/dL.
> — `KB-02_PERKENI-2021_Pemantauan-Glukosa-Mandiri.pdf`, halaman cetak 25

**Bingkainya argumen ORDE BESARAN, bukan penerapan standar.** Selisih RMSE antarmodel
0,358–0,976 mg/dL berada **dua orde besaran di bawah** toleransi per-pengukuran alat yang
menghasilkan datanya. Angka sekecil itu tidak dapat ditafsirkan secara fisik, karena variabel
yang diukurnya sendiri tidak pernah diketahui pada ketelitian tersebut.

Yang **tidak** diklaim: bahwa ISO 15197 menetapkan ambang kebermaknaan bagi selisih
antarmodel. Ia mengatur akurasi alat ukur. Yang dipinjam hanya skala besarannya. Menyandingkan
RMSE agregat dengan toleransi per-pengukuran sebagai persentase — seperti versi pertama
catatan T1.4 melakukannya — **tidak sepadan satuan statistiknya** dan sudah dikoreksi.

**Ambang mg/dL TIDAK berlaku untuk sensitivitas hipoglikemia**, karena satuannya **poin persen
deteksi kejadian**, bukan galat konsentrasi. Klaim sensitivitas LSTM karena itu tidak tergugur
oleh ambang ini; ia berdiri atau jatuh murni pada pertanyaan konsistensi — dan di situ ia
**sepakat layak di +30 menit** tetapi **rapuh di +60 menit**.

**Pengakuan yang wajib.** Ambang ini **diadopsi setelah hasil terlihat**. Yang membuatnya tetap
dapat dipertahankan: ia diturunkan dari literatur eksternal dan **tidak menyebut satu pun
angka proyek ini**, sehingga tidak dapat dipilih demi meloloskan atau menggugurkan klaim
tertentu. Perbedaan itu ditulis, bukan disembunyikan. MARD CGM dicari di seluruh teks
terekstrak proyek ini dan **tidak ditemukan**, sehingga angkanya tidak dikutip.

### Akar yang sama dengan T1.1b

Ini **bukan** kekeliruan baru. Ia kekeliruan yang sama dengan T1.1b, dan lebih berharga
dilaporkan sebagai satu daripada dua catatan terpisah:

| | T1.1b | K14 |
|---|---|---|
| pertanyaan yang diajukan | apakah 50/12 **setara** dengan 200/20? | apakah selisih ini **cukup besar** untuk berarti? |
| instrumen yang dipakai | uji beda (Wilcoxon) | varians antar-fold |
| yang seharusnya dipakai | uji kesetaraan (TOST) dengan **margin** ditetapkan di muka | **ambang kebermaknaan** ditetapkan di muka |
| apa yang tidak pernah ditetapkan | margin kesetaraan | ambang kebermaknaan |

Keduanya: **pertanyaan tentang besaran dijawab dengan instrumen tentang perbedaan**, karena
angka pembandingnya tidak pernah ditetapkan sebelum hasil terlihat.

### DIPUTUSKAN 11 Agustus 2026 — dua pertanyaan dipisah (keputusan #8)

| pertanyaan | alat ukur yang ditetapkan |
|---|---|
| Apakah efeknya **konsisten**? | **SD selisih berpasangan** `std(A−B)` + Wilcoxon tingkat fold |
| Apakah besarnya **bermakna**? | **Ambang orde besaran** ISO 15197:2013, ±15 mg/dL untuk glukosa < 100 mg/dL |

Sumber ambang: `KB-02_PERKENI-2021_Pemantauan-Glukosa-Mandiri.pdf` halaman cetak 25 — korpus
proyek ini sendiri.

#### Kedua pilihan ini MEMPERKETAT, bukan melonggarkan

Ini wajib dinyatakan apa adanya, karena pemilihannya dilakukan **setelah tabel audit di atas
terlihat**, dan hanya satu hal yang membuatnya dapat dipertahankan sama sekali.

**SD selisih menggugurkan klaim yang menguntungkan narasi penelitian ini.** Sensitivitas
hipoglikemia LSTM di +60 menit (−6,203) lolos di bawah definisi lama (SD gabungan 5,862) dan
**gugur** di bawah SD selisih (6,877). Klaim itu menopang argumen bahwa keterbatasan
prediktor bersifat klinis, bukan sekadar statistik — dan definisi yang dipilih mencabutnya.

**Ambang klinis menggugurkan ketiga klaim RMSE yang justru LOLOS di bawah SD selisih.**
Ketiganya konsisten antar-fold; ketiganya tidak bermakna.

**Definisi yang dipilih merugikan pihak yang memilihnya.** Itu bukan kebetulan yang layak
dibanggakan — itu satu-satunya alasan pemilihan pasca-hoc ini dapat dipertahankan.

#### Dua kejujuran yang wajib menyertai ambang klinis

1. **Diadopsi SETELAH hasil terlihat.** Yang membuatnya dapat dipertahankan: ia diturunkan
   dari literatur eksternal dan **tidak menyebut satu pun angka proyek ini**, sehingga tidak
   dapat dipilih demi meloloskan atau menggugurkan klaim tertentu.
2. **Analogi berdasar literatur, BUKAN penerapan ISO pada perbandingan model.** ISO mengatur
   akurasi alat ukur. Yang dipinjam hanya skala besarannya. Menyandingkan RMSE agregat
   dengan toleransi per-pengukuran 95% sebagai persentase tidak sepadan satuan
   statistiknya; yang sah adalah pernyataan **orde besaran**.

#### Rumusan yang berlaku bagi keempat klaim rapuh

| klaim | selisih | SD selisih | konsisten? | % ambang | rumusan |
|---|---:|---:|---|---:|---|
| T4.1 h6 RMSE RF vs GBM | +0,358 | 0,135 | ya | **2,39%** | **konsisten tetapi tidak bermakna** |
| T4.1 h12 RMSE RF vs GBM | +0,976 | 0,388 | ya | **6,51%** | **konsisten tetapi tidak bermakna** |
| T1.1 h6 RMSE RF vs LSTM | +0,478 | 0,239 | ya | **3,19%** | **konsisten tetapi tidak bermakna** |
| T2.1 h12 sens RF vs LSTM | −6,203 | 6,877 | **tidak** | — | **RAPUH, tidak dinarasikan** |

**Bagi ketiga klaim RMSE, rumusannya sama di bawah ketiga definisi SD.** Semuanya menyatakan
efek yang konsisten, dan ambang klinislah yang menggugurkan kebermaknaannya — sehingga
**pilihan definisi SD tidak relevan bagi ketiganya**.

**T2.1 h12** gugur pada pertanyaan konsistensi, bukan kebermaknaan; **ambang mg/dL tidak
berlaku baginya** karena satuannya poin persen deteksi. **T2.1 h6** (−10,429, SD selisih
4,727) bertahan menurut ketiga definisi. Keunggulan sensitivitas hipoglikemia LSTM karena
itu **mantap di +30 menit dan rapuh di +60 menit** — bukan "mantap pada kedua horizon".

**Yang tetap berlaku:** setiap klaim keunggulan wajib **menyebut definisi yang dipakainya**,
dan ketiga definisi tetap dilaporkan di keluaran skrip beserta medan `sd_sepakat`. Menetapkan
satu definisi tidak menghapus kewajiban menunjukkan ketiganya.

**Tidak satu pun angka lama diubah.** T1.1, T1.1b, T2.1, T3.x, dan T4.1 tetap sebagaimana
tersimpan; yang berubah hanya cara pelaporannya.

**Dampak setelah keputusan.** Tiga klaim RMSE dirumuskan sebagai *konsisten tetapi tidak
bermakna*; satu klaim sensitivitas ditarik dari narasi. Nol klaim keunggulan akurasi yang
tersisa dapat dikutip Bab VI sebagai keunggulan yang berarti secara klinis.

**Perkiraan pekerjaan.** Melaporkan ketiga definisi di seluruh skrip: **selesai** untuk T4.1
dan T1.4. Menerapkan rumusan baru ke naskah Bab VI: **penulisan, bukan pengukuran**.

---

## Ringkasan untuk Bab VII

| Kode | Keterbatasan | Dampak | Pekerjaan |
|---|---|---|---|
| K1 | Pelabel relevansi kata kunci | **Tinggi** — batas atas seluruh metrik penelusuran | di luar lingkup |
| K2 | Korpus RAGAS sempit | Sedang — `context_precision` optimistis | sedang |
| K3 | `grounded` tidak deteksi konteks tak relevan | **Tinggi** — pengaman KNF-03 bertumpu perilaku model | kecil-sedang |
| K4 | Ekstraksi merusak dekorasi gambar | Rendah — tabel dosis selamat | sedang |
| K5 | Tidak ada validasi klinis | **Tinggi** — batas klaim kelayakan | di luar lingkup |
| K6 | Pembuangan disclaimer | Rendah — perlu dinyatakan, bukan diperbaiki | — |
| K7 | Kuota LLM membatasi rancangan | Sedang — memaksa korpus kecil | selesai/di luar lingkup |
| K8 | Positif palsu pemeriksa angka | Rendah — satu kolom tidak dapat dibaca langsung | kecil |
| K9 | `top_k` produksi vs evaluasi | Rendah — sudah diperbaiki | selesai |
| K10 | `faithfulness` bukan ukuran mutu, labil per kasus | **Tinggi** — membatasi cara metrik RAGAS dikutip | sedang |
| K11 | Logbook tersambung tetapi catatan sering ditolak | Sedang — KF-01/KF-02 tidak menjadi ADA penuh | sedang-besar |
| K12 | Jangkauan penelusuran hanya 1,16% korpus | **Tinggi** — satu sebab bagi K2, jurang Hit@1/Hit@5, dan E01 | kecil-sedang |
| K13 | Angka penelusuran diukur pada bentuk kueri yang tidak dipakai aplikasi | Sedang — konsekuensinya terukur dan tidak merugikan; bentuk aplikasi tidak lebih buruk | sedang |
| **K14** | Aturan pelaporan efek mencampur konsistensi dengan kebermaknaan; skala derau tak pernah ditetapkan | **Tinggi** — empat klaim keunggulan berstatus rapuh | keputusan, bukan pekerjaan |

**Lima yang paling menentukan batas klaim laporan: K1, K3, K5, K10, dan K14.** Kelimanya
bukan cacat implementasi melainkan batas metodologis, dan seluruhnya harus dinyatakan
sebelum angka apa pun dikutip sebagai bukti kelayakan klinis.

**K14 berbeda sifatnya dari empat lainnya**: ia bukan batas pada apa yang dapat diukur,
melainkan pada cara hasil pengukuran ditafsirkan. Ia karena itu satu-satunya yang dapat
diselesaikan tanpa data baru — dengan menetapkan ambang, yang merupakan keputusan
pembimbing (#8).

---

## Butir yang TIDAK lagi menjadi keterbatasan

### KNF-10 waktu tanggap — SUDAH TERUKUR, tidak lagi menunggu pengukuran

Audit awal menempatkan KNF-10 sebagai satu-satunya butir berstatus **TIDAK ADA**: nol
instrumentasi waktu tanggap. Status itu **sudah tidak berlaku**.

Nilai acuan pada `gemini-3.5-flash-lite`, Ryzen 5 5600H tanpa GPU:

| Besaran | Median | p95 |
|---|---|---|
| Komputasi lokal | **0,339 dtk** | 0,529 dtk |
| Generasi LLM | 2,060 dtk | 2,750 dtk |
| **TOTAL ujung-ke-ujung** | **2,474 dtk** | **3,163 dtk** |
| Proporsi menunggu LLM | 85,0% | 88,7% |

**Kerapatan median terhadap p95 justru informatif.** Selisihnya hanya **0,689 detik**
(2,474 → 3,163), yang berarti waktu tanggap **konsisten** dan tidak punya ekor panjang.
Dua belas dari dua belas permintaan berjalan tanpa satu pun galat 429.

Bandingkan dengan `gemini-2.5-flash-lite`, yang p95-nya **17,3 detik** — bukan karena sifat
model, melainkan karena satu permintaan menunggu **33,4 detik** akibat menembus batas kuota
20 RPD. p95 itu mengukur **waktu tunggu retry**, bukan kecepatan model, dan dinyatakan
tidak sah.

Kerapatan pada model baru inilah yang membedakan keduanya: p95 yang rapat menandakan
pengukuran bersih, sedangkan p95 yang jauh dari median pada model lama menandakan
pencemaran kuota. **Angka 2,474 dtk median dan 3,163 dtk p95 menjadi nilai acuan KNF-10
yang selama ini kosong.**

Sisa yang masih terbuka pada KNF-10 bukan angkanya, melainkan temuan A4 bahwa sistem tidak
memberi tahu dokter ketika jalur cadangan template yang aktif — dan itu tercakup K3.
