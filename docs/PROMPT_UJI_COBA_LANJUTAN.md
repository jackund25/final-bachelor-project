# Rangkaian Percobaan untuk Bab VI

> Disimpan ke repo 6 Agustus 2026. Prompt sebelumnya sempat hilang dari konteks setelah
> pemadatan sehingga Bagian B tidak dapat dikerjakan; berkas ini mencegah itu terulang.

## Ketentuan Umum

1. Seluruh pemanggilan model bahasa menggunakan Google Gemini. Jangan memakai Ollama
   atau model lokal lain untuk peran apa pun, termasuk sebagai penilai pada evaluasi
   RAGAS. Jalur Ollama boleh tetap ada di kode sebagai mekanisme cadangan, tetapi
   tidak dipakai menghasilkan angka yang dilaporkan.

2. Satu percobaan satu commit. Tulis ringkasan ke docs/journey/ setiap kali selesai,
   memuat apa yang diuji, angka hasilnya, dan kesimpulannya.

3. Untuk percobaan yang bersifat penyetelan parameter, ikuti protokol pada
   docs/PROMPT_PERBAIKAN_DAN_REKAYASA.md Bagian C.

4. Percobaan yang hasilnya negatif tetap dilaporkan lengkap. Jangan menyembunyikan
   atau memperhalus.

5. Setelah setiap percobaan, jalankan pytest dan bandingkan dengan baseline.

6. Setelah setiap tahap selesai, laporkan ringkasan angkanya lalu berhenti agar
   pembimbing dapat menilai sebelum melanjutkan.

## Catatan git (WAJIB)

Seluruh commit bersifat **LOKAL saja**. Jangan pernah menjalankan `git push`,
`git remote add`, atau perintah lain yang mengirim perubahan ke repositori jarak jauh.
Repo ini punya remote `origin`, tetapi tidak boleh disentuh. Kalau sesuatu perlu didorong
ke remote, tanyakan dulu.

Commit lokal tetap dikerjakan seperti biasa, satu percobaan satu commit, karena riwayat
git dipakai menyusun timeline pengerjaan di laporan.

---

## Tahap 0. Adopsi Hasil Bagian B — SELESAI

> **HASIL AKHIR TAHAP 0 (keputusan pembimbing, 6 Agustus 2026).**
>
> Versi awal tahap ini meminta `chunk_size` diubah menjadi 500 DAN `lambda_mult` menjadi
> 0,0. Pengukuran butir 4 menunjukkan **gabungan keduanya lebih buruk daripada
> masing-masing sendirian** (MRR 0,464 vs 0,475 dan 0,492), dan uji Wilcoxon menolak
> konfigurasi gabungan (p=0,029).
>
> **Keputusan: adopsi B5 saja. `chunk_size` TETAP 900, `lambda_mult` menjadi 0,0.**
> B4 tidak diadopsi. Konfigurasi gabungan tidak diadopsi.
>
> **Rumusan yang wajib dipakai:** B4 dan B5 **setara secara statistik** (p=0,078);
> konfigurasi gabungan ditolak (p=0,029); B5 dipilih atas **pertimbangan di luar MRR**
> (keragaman dokumen 3,60 vs 3,15, dan keunggulan yang lebih bertahan di bawah protokol
> set penyetelan/pelaporan). **Jangan tulis bahwa B5 terbukti lebih baik daripada B4.**
>
> Pertimbangan berlawanan arah yang tidak boleh disembunyikan: **B4 unggul pada Hit@1
> (34,6% vs 29,2%)**, dan lambda 0,0 juga menurunkan keragaman dokumen dari 3,73 ke 3,60.

1. ~~Ubah `chunk_size` menjadi 500~~ **DIBATALKAN.** `chunk_size` tetap **900**.
2. Ubah `lambda_mult` menjadi **0.0** sesuai pengujian B5. **Dikerjakan.**
3. Jangan adopsi B1, B2, dan B3. Kodenya tetap ada tetapi tidak dipakai jalur produksi.
4. Angka gabungan sudah diukur pada set PELAPORAN; tabel empat baris beserta uji Wilcoxon
   tersimpan di `results/interaksi_parameter_b4b5.json` untuk subbab pembahasan Bab VI.
5. Seluruh percobaan Tahap 1-5 dijalankan pada **`chunk_size` 900 dan `lambda_mult` 0,0**.

---

## Tahap 1. Keabsahan dan Janji Laporan

### T1.1 Samakan basis perbandingan RF versus LSTM (prioritas tertinggi)

1. Periksa apakah LSTM sudah dilatih ulang di atas jendela tersegmentasi, atau angkanya
   masih dari jendela lama.
2. Kalau belum, latih ulang LSTM dengan pembagian data, penormalan, dan segmentasi
   jendela yang persis sama dengan Random Forest.
3. Laporkan tabel RMSE, MAE, MAPE, dan rincian zona Clarke A-E untuk kedua model pada
   kedua horizon.
4. Laporkan waktu latih, waktu inferensi, dan ukuran model (KNF-01, Subbab III.4.2).

### T1.2 Evaluasi RAGAS pada korpus terbatas dengan Gemini

1. Bangun koleksi evaluasi terpisah (collection_name dan persist_directory berbeda).
2. Isi dengan 2-5 halaman dari KB-03 PERKENI Terapi Insulin, memuat tata laksana
   konkret bukan pendahuluan. **Laporkan halaman mana dan alasannya sebelum membangun.**
3. Susun kasus uji pada `evaluation/ragas_dataset.json`: pertanyaan, kondisi_terprediksi,
   jawaban_acuan, konteks_acuan, halaman_sumber. Diambil dari isi halaman, bukan dikarang.
4. Jalankan RAGAS dengan penilai Gemini. Uji ulang keempat metrik; faithfulness dan
   context_precision pernah gagal terurai. Kalau gagal, laporkan galat lengkap dan coba
   perbaiki. Kalau tetap gagal, laporkan sebagai keterbatasan.
5. Terapkan rate limiter dan cache. Simpan mentah ke `results/ragas/`.
6. Cetak estimasi jumlah panggilan sebelum eksekusi dan minta konfirmasi.
7. Kalau kuota habis, lanjutkan keesokan harinya dari cache. Jangan beralih ke model lokal.
8. Laporkan skor tiap metrik beserta jumlah kasus yang berhasil dinilai.

### T1.3 Cakupan empiris interval konformal per rentang glukosa

1. Ukur cakupan empiris terhadap tingkat kepercayaan yang ditetapkan, pada data uji.
2. Pecah menurut rentang kadar sebenarnya, memakai zona `src/constants.py`.
3. Laporkan lebar rata-rata interval per rentang.
4. Kerjakan untuk kedua horizon.

Hipotesis: cakupan pada rentang hipoglikemia lebih rendah daripada agregat karena
sampelnya paling sedikit.

### T1.4 Sensitivitas konstanta waktu peluruhan

1. Variasikan konstanta waktu karbohidrat di sekitar nilai sekarang.
2. Variasikan pula konstanta waktu insulin.
3. Untuk tiap nilai, latih ulang RF dan ukur RMSE serta kontribusi fitur IOB dan COB.
4. Ikuti protokol Bagian C.
5. Laporkan apakah nilai sekarang dekat optimum dan seberapa peka hasilnya.

---

## Tahap 2. Memperkuat Hasil yang Sudah Ada

### T2.1 Evaluasi khusus kejadian hipoglikemia

1. Ukur galat terpisah untuk <70 mg/dL, rentang normal, dan >180 mg/dL.
2. Ukur sensitivitas dan spesifisitas deteksi hipoglikemia.
3. Laporkan berapa kejadian hipoglikemia yang terlewat sepenuhnya.
4. Kaitkan dengan zona Clarke D dan E.
5. Kerjakan untuk Random Forest dan LSTM.

### T2.2 Verifikasi pelabel relevansi terhadap penilaian manual

1. Ambil sampel acak 30-50 pasangan kueri dan chunk dari hasil penelusuran.
2. Siapkan berkas untuk dilabeli manual: kueri, teks chunk, kolom kosong.
   **Sembunyikan label otomatis** agar tidak memengaruhi penilaian.
3. Setelah dilabeli, hitung tingkat kesesuaian pelabel otomatis vs manual.
4. Laporkan pola ketidaksesuaian: terlalu longgar atau terlalu ketat.

Memerlukan pelabelan manual dari pembimbing. Siapkan berkasnya lalu lanjut ke percobaan
berikutnya sambil menunggu.

---

## Tahap 3. Memperkuat Kontribusi Utama

### T3.1 Susunan kalimat kueri terkondisi-prediksi

Tombol paling sensitif pada kontribusi penelitian. Sinyal awal: mengubah bentuk kueri
dari pernyataan menjadi pertanyaan mengubah MRR 0,892 -> 0,592 pada himpunan natural.

1. Susun varian: hanya nilai numerik prediksi; hanya kategori risiko; nilai + kategori;
   nilai + kategori + arah tren; susunan sekarang sebagai pembanding.
2. Ikuti protokol Bagian C sepenuhnya.
3. Terapkan varian yang sama pada kedua mode ablasi.
4. Laporkan MRR dan Hit@1 tiap varian pada set penyetelan DAN set pelaporan.

### T3.2 Horizon mana yang lebih baik untuk penelusuran

1. Jalankan penelusuran dengan kueri dari prediksi 30 menit, lalu 60 menit, pada
   himpunan kasus yang sama.
2. Bandingkan MRR dan Hit@1.
3. Uji penggabungan keduanya.
4. Laporkan berapa kasus yang kategori terprediksinya berbeda antar-horizon.

---

## Tahap 4. Melengkapi Alternatif yang Disebut di Laporan

### T4.1 Gradient Boosting sebagai pembanding ketiga

1. Uji XGBoost atau LightGBM dengan protokol persis sama seperti RF dan LSTM.
2. Laporkan RMSE, MAE, MAPE, rincian zona Clarke A-E.
3. Laporkan waktu latih, waktu inferensi, ukuran model, ketersediaan kontribusi fitur.
4. **Jangan menyetel hyperparameter secara ekstensif.** Pakai konfigurasi baku yang
   wajar dan sebutkan bahwa penyetelan tidak dilakukan.

### T4.2 Hubungkan halaman input logbook ke jalur prediksi

1. Sambungkan keluaran halaman input logbook ke jalur rekayasa fitur dan prediksi.
2. Pastikan IOB dan COB berjalan pada masukan manual.
3. Tangani riwayat yang terlalu pendek, dan nyatakan batas minimum riwayat.

---

## Tahap 5. Bahan Pembahasan Komparatif Bab VI

### T5.1 Konversi ukuran potongan ke satuan token

1. Hitung distribusi token per chunk memakai tokenizer yang benar-benar dipakai model
   embedding, bukan perkiraan.
2. Laporkan median, rata-rata, persentil ke-5 dan ke-95, untuk **900 karakter sebagai
   konfigurasi produksi** dan **500 karakter sebagai pembanding**.
3. Laporkan terpisah untuk dokumen Indonesia dan Inggris.

> **Penyesuaian 6 Agustus.** Sebelumnya butir ini menempatkan 500 sebagai konfigurasi
> produksi. Setelah keputusan Tahap 0, produksinya adalah **900**; 500 tetap dihitung
> sebagai pembanding karena pembahasan Bab VI menjadi lebih menarik: **ukuran optimum
> untuk penelusuran ternyata bukan ukuran terkecil**, dan nuansa itu tidak dibahas Gao
> dkk. (2023) yang hanya menyebut 100, 256, dan 512 token sebagai ukuran lazim.

### T5.2 Kumpulkan angka pembanding untuk pembahasan

Susun `results/ringkasan_untuk_bab6.json` memuat seluruh angka final Bab VI: metrik
prediksi tiap model tiap horizon termasuk zona Clarke, metrik hipoglikemia (T2.1),
cakupan konformal per rentang (T1.3), metrik penelusuran tiap mode (Hit@k, MRR, nDCG),
skor RAGAS (T1.2), waktu tanggap per tahap, jumlah chunk dan dokumen dan distribusi
token (T5.1), serta daftar konfigurasi yang diuji beserta jumlahnya.

Sertakan medan tanggal pengukuran dan commit hash.

---

## Urutan Pengerjaan

Tahap 0, lalu T1.1, T1.2, T1.3, T1.4, lalu T2.1, T2.2, lalu T3.1, T3.2, lalu T4.1,
T4.2, terakhir T5.1 dan T5.2.
