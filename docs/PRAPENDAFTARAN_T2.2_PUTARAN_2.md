# Prapendaftaran T2.2 putaran kedua — perbaikan instrumen, bukan pemilihan hasil

> **Ditulis 13 Agustus 2026 pada commit `8315927`.**
> **Penilaian putaran kedua BELUM dilakukan.** Belum ada satu pun lembar yang terisi.
>
> Berkas ini ditulis **sebelum** lembar dibagikan, karena inti persoalannya adalah
> **urutan**: perbaikan instrumen yang sah dan pemilihan hasil yang tidak sah menghasilkan
> berkas yang tampak sama bila urutannya tidak tercatat.

---

## 1. Alasan, dan urutannya

### Cacat instrumen bersifat objektif

Pada putaran pertama, Penilai 1 memakai `hiperglikemia` untuk **31 dari 40 baris** dan
**tidak pernah sekali pun** memakai kategori `normal`.

Kategori `normal` bukan tidak terpakai karena tidak ada isinya. Baris **id 17** memuat
potongan bertabel target glikemik — GDP 80–130 mg/dL dan HbA1c < 7% — dan pelabel otomatis
menandainya `normal`, sedangkan penilai menandainya `hiperglikemia`. Potongan yang isinya
**definisi rentang target** dinilai sebagai hiperglikemia.

Sebaran akhirnya: **31 hiperglikemia · 7 hipoglikemia · 2 `lain` · 0 normal**.

### Urutan tanggal — inilah yang membedakan perbaikan dari pemilihan

| tanggal | peristiwa |
|---|---|
| **11 Agustus 2026** | Sebaran timpang dan kategori `normal` yang tak pernah terpakai **dilaporkan**, beserta peringatan bahwa κ akan labil karenanya — **sebelum κ dihitung** |
| **11 Agustus 2026, sesudahnya** | κ dihitung: **0,2505** |
| **13 Agustus 2026** | Putaran kedua diputuskan |

Peringatan itu tercatat pada percakapan sesi dan pada `results/eval_rag/verifikasi_relevansi.json`
medan `PERBANDINGAN_TERHADAP_TEBAKAN_MAYORITAS.kualifikasi`, yang menyebut sebaran timpang
membuat κ labil — ditulis bersamaan dengan κ-nya, bukan sesudah hasilnya tidak menyenangkan.

**Yang membuat perbaikan ini sah bukan bahwa κ-nya rendah**, melainkan bahwa cacat
instrumennya dapat ditunjuk tanpa merujuk κ sama sekali: satu kategori tidak pernah terpakai
padahal ada baris yang isinya persis kategori itu.

---

## 2. Komitmen

**Kedua putaran dilaporkan, apa pun hasilnya, termasuk bila κ TURUN.**

Putaran pertama **tidak dihapus dan tidak ditimpa**. Ia diarsipkan bertanggal di
`evaluation/arsip/` sebelum apa pun disiapkan:

```
evaluation/arsip/verifikasi_relevansi_PUTARAN1_2026-08-13.csv
evaluation/arsip/hasil_PUTARAN1_2026-08-13.json
evaluation/arsip/catatan_PUTARAN1_2026-08-13.json
```

Bila κ putaran kedua lebih rendah, itu dilaporkan sebagai hasil dan **tidak** ditafsirkan
sebagai kegagalan penilai. κ yang turun setelah instrumen diperbaiki adalah temuan tentang
**pelabel otomatis**, bukan tentang manusianya.

---

## 3. Perubahan pada instrumen — DUA, bukan satu

Instruksi menyebut satu perubahan. Ada **dua**, dan keduanya wajib dinyatakan karena
keduanya memengaruhi kesebandingan antar-putaran.

### Perubahan 1 — panduan, pada satu titik

Ditambahkan penjelasan bahwa `normal` berarti **kelas target atau rentang aman**, dan bahwa
**keempat kategori wajib dipertimbangkan** pada tiap baris. Tidak ada perubahan lain pada
definisi, contoh, maupun urutan.

### Perubahan 2 — lembar, kolom yang ditampilkan

| kolom | putaran 1 | putaran 2 |
|---|---|---|
| `id`, `penilaian_manusia`, `teks_potongan` | ada | **ada** |
| `kueri` | ada | **dibuang** |
| `sumber`, `halaman_cetak`, `peringkat` | ada | **dibuang** |
| label otomatis | tidak pernah ada | tidak ada |

**Alasannya, dan ia bukan kosmetik.** Pertanyaan yang diajukan adalah *"apa topik utama yang
dibahas potongan ini?"* — **bukan** *"apakah potongan ini berguna bagi kueri tersebut"*.
Menampilkan kueri mengundang penilaian **relevansi** alih-alih penilaian **topik**, yaitu
persis kekeliruan yang panduannya sendiri larang di baris keempat. Membuang kolomnya membuat
lembar itu cocok dengan tugas yang dinyatakannya.

**Konsekuensi yang wajib diakui:** putaran 1 dan putaran 2 **tidak sepenuhnya sebanding**.
Selisih κ antar-putaran tidak dapat diatribusikan seluruhnya kepada perbaikan panduan, karena
lembarnya juga berubah. Perbandingan antar-putaran karena itu dilaporkan sebagai
**indikatif**, dan yang menjadi angka utama adalah **κ putaran kedua**, bukan selisihnya.

**Jawaban putaran pertama tidak ditunjukkan kepada siapa pun.**

---

## 4. Susunan penilai — status berbeda, tidak boleh disamarkan

| | latar | tugas | status metodologis |
|---|---|---|---|
| **Penilai 1** | mahasiswa kedokteran | putaran **kedua**, instrumen diperbaiki | **BUKAN independen** — sudah pernah membaca ke-40 potongan yang sama |
| **Penilai 2** | mahasiswa koas, belum pernah menyentuh proyek ini | penilaian **tunggal dan mandiri** | independen |
| **Peneliti** | — | **TIDAK menjadi penilai** | lihat di bawah |

Keduanya **tidak terlibat pengembangan sistem**. Nama tidak dicatat di berkas mana pun.

### Mengapa peneliti tidak menjadi penilai

Peneliti sudah melihat **label otomatis** (berkas kunci) dan **hasil putaran pertama**.
Penilaiannya karena itu akan tertarik ke arah pelabel dan **menaikkan κ secara palsu**.

Ini juga **bukan** test-retest: test-retest mensyaratkan kedua kali dilakukan **buta**, dan
kebutaan itu sudah hilang secara permanen bagi peneliti.

### Apa yang κ Penilai 1 putaran kedua BOLEH dan TIDAK BOLEH dipakai

**Boleh:** dibandingkan dengan pelabel otomatis, sebagai ukuran kesesuaian setelah instrumen
diperbaiki.

**Tidak boleh:** dipakai sebagai batas atas antar-manusia. Untuk itu hanya κ **Penilai 1
lawan Penilai 2** yang sah, dan hanya bila Penilai 2 benar-benar ada.

### Batas waktu pencarian Penilai 2: DUA HARI

Ditetapkan **sekarang**, sebelum pencarian dimulai, supaya ia tidak menyandera penulisan
Bab VI.

Bila setelah dua hari Penilai 2 tidak tersedia: **jalankan dengan Penilai 1 saja**, dan
nyatakan **ketiadaan batas atas antar-penilai sebagai keterbatasan** pada K1 — bukan sebagai
sesuatu yang masih ditunggu. Keputusan ini tidak ditinjau ulang setelah melihat hasil
Penilai 1.

---

## 5. Larangan selama penilaian

**Kedua penilai tidak membahas isi lembar sampai keduanya selesai.** Risikonya nyata karena
keduanya berada dalam satu lingkaran pertemanan; satu kalimat *"yang nomor 17 itu maksudnya
apa ya?"* sudah cukup menghapus kemandirian yang justru sedang diukur.

**Diskusi sesudahnya dianjurkan** dan dicatat sebagai **diskusi pasca-penilaian**, terpisah
dari data. Ketidaksepakatan yang mereka bahas sendiri sering menunjukkan ambiguitas definisi
yang tidak terlihat dari angka.

---

## 6. Yang diperkirakan, dinyatakan sebelum diukur

### D1 — kategori `normal` akan terpakai, sekurang-kurangnya sekali

Dasar: id 17 memuat tabel target GDP 80–130 dan HbA1c < 7%; pelabel otomatis menandai
**4 baris** sebagai `normal`.

**Perkiraan: sekurang-kurangnya 3 dari 40 baris dinilai `normal` oleh masing-masing penilai.**

Bila `normal` **tetap** tidak terpakai meski panduannya sudah menjelaskannya, maka sebabnya
bukan panduan melainkan **korpusnya** — potongan yang terambil memang didominasi materi
hiperglikemia — dan itu temuan tentang K12, bukan tentang instrumen.

### D2 — κ Penilai 1 putaran kedua NAIK di atas 0,2505

Dasar: 17 dari 19 ketidaksesuaian putaran pertama adalah pelabel gagal mengenali
hiperglikemia, sedangkan penilai **melebihkan** hiperglikemia. Mengurangi keduanya sekaligus
seharusnya menaikkan kesepakatan.

**Perkiraan: κ berada di rentang 0,35–0,60.**

**Perkiraan ini yang paling mudah salah, dan sengaja didaftarkan.** Bila κ justru turun,
tafsirnya bukan "penilai memburuk" melainkan bahwa kesepakatan putaran pertama sebagian
berasal dari **kebetulan menebak kelas mayoritas** — dan itu memperkuat, bukan melemahkan,
kesimpulan bahwa pelabel otomatis tidak memadai.

### D3 — κ antar-penilai LEBIH TINGGI daripada κ mana pun terhadap pelabel otomatis

Dasar: dua manusia berbagi pemahaman klinis yang tidak dimiliki pencocokan kata kunci.

**Perkiraan: κ(Penilai 1, Penilai 2) > κ(Penilai 1, otomatis) dan > κ(Penilai 2, otomatis).**

Bila terbantah — yaitu bila kedua manusia lebih sepakat dengan mesin daripada dengan satu
sama lain — itu temuan besar: berarti tugasnya sendiri **ambigu**, dan bukan pelabelnya yang
buruk. Konsekuensinya menyentuh K1 secara mendasar, karena "relevansi" yang sedang diukur
tidak akan punya acuan manusia yang stabil.

### D4 — arah ketidaksepakatan tetap sama

**Perkiraan: pelabel otomatis tetap terlalu KETAT** — recall hiperglikemianya tetap di bawah
0,70, dan `lain` tetap kelebihan dipakai relatif terhadap penilaian manusia.

Arah lebih berguna daripada κ, karena ia menentukan apakah angka penelusuran **terlalu
optimistis atau terlalu pesimistis**. Bila pelabel terlalu ketat, ia menandai potongan
relevan sebagai tidak relevan, sehingga Hit@k dan MRR yang dilaporkan adalah **batas bawah**.

---

## 7. Aturan penafsiran, ditetapkan sekarang

1. **Angka utama adalah κ putaran kedua**, bukan selisihnya terhadap putaran pertama —
   karena instrumennya berubah pada dua titik, bukan satu.
2. **κ Penilai 1 bukan batas atas antar-manusia.** Hanya κ antar-penilai yang sah untuk itu.
3. **n = 40 tetap kecil.** Selang kepercayaan κ pada n=40 lebar; yang sah disimpulkan adalah
   golongannya (lemah / sedang / kuat), bukan nilai persisnya.
4. **Aturan pelaporan K1 yang berlaku tidak berubah:** κ < 0,40 → angka penelusuran wajib
   berkualifikasi eksplisit di setiap penyebutannya; 0,40–0,60 → dengan catatan; > 0,60 →
   pelabel memadai untuk tujuan ini.
5. **Bila kategori mana pun tidak terpakai oleh penilai mana pun**, skrip memberi peringatan
   menonjol dan keluar dengan kode bukan nol — κ pada kondisi itu tidak dapat ditafsirkan
   apa adanya.

---

## Kaitan

- `docs/DAFTAR_KETERBATASAN.md` K1 — yang diukur percobaan ini
- `evaluation/arsip/` — putaran pertama, diarsipkan bertanggal
- `docs/KEPUTUSAN_DIAMBIL.md` #1 — putaran pertama dan konsekuensinya
- `docs/PRAPENDAFTARAN_T_RAGAS_STABIL.md` — pola prapendaftaran, termasuk aturan baru:
  periksa apakah kode atau konfigurasi yang ada sudah menentukan jawabannya
