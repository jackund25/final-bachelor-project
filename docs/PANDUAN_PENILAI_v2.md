# Panduan Penilai — versi 2

> Untuk `evaluation/verifikasi_relevansi_putaran2_penilai1.csv` dan `..._penilai2.csv`.
> Perkiraan waktu: **45–80 menit** untuk 40 baris.
>
> Versi 1 berbeda pada satu hal: penjelasan kategori `normal` di bawah belum ada di sana.

---

## Apa yang Anda kerjakan

Anda membaca **40 potongan teks** yang diambil dari pedoman klinis diabetes, dan untuk tiap
potongan menuliskan **satu** kategori pada kolom `penilaian_manusia`.

Isi **hanya** kolom itu. Jangan mengubah kolom lain, jangan menambah baris, jangan mengubah
urutan.

## Pertanyaan yang dijawab

> **Apa topik utama yang dibahas potongan teks ini?**

Bukan *"apakah potongan ini berguna bagi pasien tertentu"*. Bukan *"apakah potongan ini
menjawab suatu pertanyaan"*. Hanya: **teks ini membahas apa**.

## Empat kategori — keempatnya wajib dipertimbangkan

| kategori | dipakai bila potongan membahas… |
|---|---|
| `hipoglikemia` | gula darah **rendah** — gejalanya, penyebabnya, penanganannya, ambangnya, glukagon, aturan 15-15 |
| `hiperglikemia` | gula darah **tinggi** — ketoasidosis, hiperosmolar, koreksi insulin, keton, krisis hiperglikemia |
| **`normal`** | **kelas target atau rentang aman** — sasaran kontrol glikemik, target GDP, target HbA1c, *time in range*, pemantauan rutin |
| `lain` | **tidak** membahas ketiga kondisi itu — definisi diabetes, alat suntik, rumus laboratorium, gizi umum, prosedur administratif |

**Pertimbangkan keempatnya pada tiap baris.** Jangan memutuskan hanya antara dua yang paling
sering muncul.

### Catatan khusus untuk `normal`

Potongan yang isinya **angka sasaran** — misalnya *"GDP 80–130 mg/dL"* atau *"HbA1c < 7%"* —
membahas **rentang target**, sehingga topiknya `normal`. Ia **bukan** `hiperglikemia`
meskipun angkanya berkaitan dengan pengendalian gula darah tinggi, dan **bukan** `lain`
meskipun ia berupa tabel.

Pertanyaannya selalu *"teks ini membahas apa"*, bukan *"kondisi apa yang sedang dialami
pasien"*.

## Bila ragu

**Pilih tetap satu.** Skema ini mengizinkan satu label saja, dan itu keterbatasannya yang
sudah dicatat.

Bila sebuah potongan membahas **dua** kondisi sekaligus, pilih yang **paling banyak
porsinya** dalam teks itu. Bila benar-benar seimbang, pilih yang muncul lebih dahulu.

Jangan menulis catatan, penjelasan, atau tanda tanya di kolom `penilaian_manusia` — hanya
salah satu dari empat kata. Bila ada baris yang ingin Anda komentari, catat nomornya di
kertas terpisah dan sampaikan **setelah** selesai.

## Larangan selama pengerjaan

- **Jangan membahas isi lembar ini dengan penilai lain sampai keduanya selesai.** Satu
  kalimat pun sudah cukup menghapus kemandirian yang sedang diukur.
- **Jangan membuka** `evaluation/verifikasi_relevansi_KUNCI.json` atau berkas apa pun di
  `evaluation/arsip/`.
- Jangan mencari potongan itu di dokumen aslinya. Nilai **teks yang tampak saja**.

Diskusi **sesudah** keduanya selesai justru dianjurkan, dan akan dicatat terpisah sebagai
diskusi pasca-penilaian.

## Setelah selesai

Simpan berkasnya sebagai CSV dengan nama yang sama, lalu serahkan. Bila dibuka dengan Excel,
menyimpannya kembali sebagai CSV tidak masalah — pembacanya sudah tahan terhadap perubahan
tata letak yang ditimbulkan Excel.

---

## Untuk pencatatan (bukan bagian penilaian)

Yang dicatat pada berkas metode hanyalah **latar** kedua penilai: mahasiswa kedokteran dan
mahasiswa koas, keduanya **tidak terlibat pengembangan sistem**. Nama tidak dicatat di berkas
mana pun.

Mengapa penilaian ini dibutuhkan: seluruh angka penelusuran pada laporan memakai relevansi
yang ditetapkan **pencocokan kata kunci otomatis**. Penilaian Anda mengukur seberapa jauh
pelabel otomatis itu sesuai dengan penilaian manusia — dan itu menjadi **batas atas** bagi
seberapa jauh angka-angka tersebut mencerminkan relevansi yang sebenarnya.
