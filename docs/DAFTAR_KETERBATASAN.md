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
pasangan kueri-chunk. T2.2 menyiapkan verifikasi sampel 30-50 pasangan, tetapi itu
mengukur *tingkat kesesuaian* pelabel, bukan menggantikannya.

**Dampak.** Menjadi **batas atas** bagi seluruh angka penelusuran yang pernah dilaporkan.
Skor tidak boleh dibaca sebagai mutu penelusuran mutlak, hanya sebagai perbandingan
antar-konfigurasi **yang memakai pelabel sama**. Sirkularitas membuat hasil B3 (BM25) tidak
sah dipakai sebagai bukti keunggulan.

**Perkiraan pekerjaan.** Verifikasi sampel: **sedang**. Penggantian penuh dengan pelabel
manusia: **di luar lingkup**.

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

**Tiga yang paling menentukan batas klaim laporan: K1, K3, dan K5.** Ketiganya bukan cacat
implementasi melainkan batas metodologis, dan ketiganya harus dinyatakan sebelum angka apa
pun dikutip sebagai bukti kelayakan klinis.
