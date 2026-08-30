# Tindak Lanjut Umpan Balik Dokter Penilai

Dokumen ini mencatat empat butir umpan balik dari dokter penilai pada sesi evaluasi
sistem, beserta tindak lanjut yang sudah dikerjakan. Dipakai sebagai bahan Bab VI
(evaluasi ahli) dan sebagai daftar periksa sebelum sesi evaluasi berikutnya.

---

## Butir 1 — Sistem belum bisa menerima pasien baru

> "Belum bisa input output pasien baru. Terus untuk di formnya bisa ditambahkan
> input (pasien baru/pasien lama), kalau pasien lama bisa pilih gitu."

**Sebab.** Pendaftaran pasien tidak pernah tersedia di antarmuka. Satu-satunya cara
menambah pasien adalah menyisipkan baris langsung ke Supabase, karena kunci publik
yang dipakai peramban tunduk pada RLS dan tidak berhak menulis ke tabel `patients`.

**Tindak lanjut.**

- Endpoint baru `POST /api/patients` (`backend/routes/patients.py`). Penulisan
  dilakukan lewat backend karena backend memegang `SUPABASE_SECRET_KEY`.
- `SupabaseDataService.create_patient()` membuat baris pasien **beserta** kanal
  glukosanya dalam satu pemanggilan. Kanal harus ikut dibuat: `save_logbook`
  menolak entri yang kanalnya belum ada, sehingga pasien tanpa kanal akan tampak
  terdaftar tetapi menolak setiap pencatatan.
- Komponen `PatientSwitcher` memisahkan dua maksud secara eksplisit, persis seperti
  yang diminta: **Pasien lama (sudah terdaftar)** dengan daftar pilihan, dan
  **Pasien baru (daftarkan sekarang)** dengan isian kode pasien serta pilihan kanal
  CGM / finger-stick.
- Kode pasien dibatasi huruf, angka, tanda hubung, dan garis bawah (2–32 karakter),
  lalu dinormalkan ke huruf kapital. Nama pasien tidak diminta, dan hal itu
  dinyatakan di bawah isian.

**Catatan yang ikut ditampilkan.** Setelah pasien baru dibuat, prediksi belum
langsung tersedia — sistem memerlukan 12 pencatatan CGM atau 8 pencatatan
finger-stick lebih dulu. Keterangan ini muncul di panel supaya penilai tidak
menyimpulkan sistem rusak ketika prediksi belum keluar.

---

## Butir 2 — Perlu penjelasan apakah alat ini benar-benar dicoba pada pasien nyata

> "Perlu penjelasan apakah alat ini beneran dicoba dengan pasien beneran atau
> tidak."

**Jawabannya: belum, dan itu harus terbaca di dalam aplikasi.** Selama ini keterangan
tersebut hanya ada di laporan (Batasan Masalah #2, #8, #9), sedangkan dokter menilai
lewat antarmuka.

**Tindak lanjut.** Komponen `CatatanSumberData` ditampilkan di Dashboard dan Logbook,
memuat empat keterangan:

1. **Asal data** — dataset publik OhioT1DM: 12 orang dewasa DMT1 di Amerika Serikat,
   seluruhnya pengguna pompa insulin dan sensor CGM. Bukan pasien Indonesia, bukan
   rekam medis.
2. **Status pengujian** — penelitian bersifat *proof-of-concept*; belum ada uji
   klinis terkontrol, belum ada pasien nyata yang memakai sistem ini, dan
   keluarannya belum pernah dipakai mengambil keputusan terapi pada pasien
   sungguhan.
3. **Pasien yang dibuat penilai sendiri** — data uji coba; dilarang memasukkan data
   pasien sungguhan atau identitas yang dapat mengenali orang.
4. **Peran sistem** — pendukung keputusan beralur *doctor-mediated*; seluruh keluaran
   wajib ditinjau dokter.

Baris ringkasnya selalu terlihat ("Purwarupa penelitian. Sistem ini belum pernah
diuji pada pasien nyata di layanan klinis."), rinciannya dibuka lewat tombol
"Selengkapnya".

---

## Butir 3 — Tombol ganti pasien perlu diperjelas

> "Terus butuh untuk tombol ganti pasiennya diperjelas."

**Sebab.** Ada dua masalah terpisah.

1. Di Dashboard, pemilih pasien hanya berupa `<select>` berlabel "Patient" yang
   terselip **di dalam** bagian "Glucose source", tanpa judul sendiri.
2. Di Logbook **tidak ada pemilih pasien sama sekali**. Halaman itu memakai pasien
   terakhir dari `localStorage`, sehingga dokter yang membuka Logbook lebih dulu
   tidak punya cara mengganti pasien.

Ada pula cacat yang belum dilaporkan tetapi searah: lencana pasien di Navbar
bertuliskan `Patient P001` secara *hardcoded*, sehingga ia menyebut pasien yang salah
setiap kali pasien diganti.

**Tindak lanjut.**

- `PatientSwitcher` berdiri sebagai kartu tersendiri berjudul **PASIEN**, memuat kode
  pasien aktif berukuran besar dan tombol **"Ganti / tambah pasien"**. Kartu ini
  dipasang di Dashboard **dan** Logbook.
- Modul `lib/patientState.ts` menjadikan `localStorage` satu-satunya sumber kebenaran
  dan menyiarkan perubahannya, sehingga Dashboard, Logbook, dan lencana Navbar tidak
  pernah berselisih menyebut pasien.
- Lencana Navbar kini mengikuti pasien aktif yang sesungguhnya.

---

## Butir 4 — Istilah "CDSS" pada formulir perlu dijelaskan

> "Terus di form yang pertanyaan CDSS, jelasin apa itu CDSS dan contohnya yang
> populer dipakai di Indonesia maupun global, karena ga semua dokter mengerti
> istilah-istilah teknologi ini."

Formulir evaluasi adalah Google Form, sehingga perubahannya dilakukan manual.
Berikut teks siap tempel untuk **deskripsi pertanyaan** tersebut.

### Teks untuk pertanyaan lama

> **Pertanyaan:** Apakah Anda pernah menggunakan sistem CDSS atau aplikasi kesehatan
> digital sebelumnya?

**Deskripsi yang ditambahkan di bawah pertanyaan:**

> **Apa itu CDSS?**
> *Clinical Decision Support System* (CDSS) adalah perangkat lunak yang menyajikan
> informasi terkait pasien untuk membantu tenaga kesehatan mengambil keputusan
> klinis — misalnya peringatan, pengingat, perhitungan risiko, atau rujukan pedoman
> yang muncul saat menangani pasien. CDSS **tidak** mengambil keputusan sendiri;
> keputusan tetap ada pada dokter.
>
> **Contoh yang lazim ditemui secara global:**
> • **UpToDate** — rujukan klinis yang menyarankan tata laksana sesuai kondisi pasien.
> • **Peringatan pada rekam medis elektronik** (mis. *Best Practice Advisory* pada
>   Epic, peringatan serupa pada Cerner/Oracle Health) — muncul otomatis saat dokter
>   memesan obat atau tindakan.
> • **Pemeriksa interaksi obat** pada e-resep (mis. Lexicomp, Medscape Drug
>   Interaction Checker).
> • **Kalkulator klinis** seperti MDCalc (skor Wells, CHA₂DS₂-VASc, dan sejenisnya).
> • **Alat bantu diagnosis** seperti Isabel atau VisualDx.
>
> **Contoh yang lazim ditemui di Indonesia:**
> • **Peringatan interaksi obat dan alergi pada SIMRS / e-resep** rumah sakit.
> • **Fitur skrining dan pengingat pada aplikasi layanan primer**, misalnya penapisan
>   risiko penyakit tidak menular pada aplikasi Kemenkes (ASIK/Posbindu) dan
>   pengingat kontrol pada Prolanis BPJS Kesehatan.
> • **Pemeriksa gejala pada aplikasi telemedisin** seperti Halodoc atau Alodokter
>   (ditujukan bagi pasien, bukan dokter, tetapi memakai prinsip yang sama).
> • Sebagai catatan, **SATUSEHAT** sendiri adalah platform pertukaran data rekam
>   medis, bukan CDSS — tetapi ia menjadi fondasi data bagi CDSS di Indonesia.
>
> **Sistem yang sedang Anda nilai** termasuk kategori CDSS: ia memprediksi kadar
> glukosa pasien pada beberapa waktu ke depan, lalu menelusuri pedoman klinis yang
> relevan dengan kondisi terprediksi itu dan menyajikannya beserta rujukan halaman.
> Seluruh keluarannya harus ditinjau dokter.

### Usul tambahan pada bagian pembuka formulir

Tempatkan sebelum pertanyaan pertama, sejalan dengan Butir 2:

> **Status sistem yang dinilai.** Sistem ini adalah purwarupa penelitian
> (*proof-of-concept*) dan **belum pernah diuji pada pasien nyata di layanan
> klinis**. Data pasien yang tampil berasal dari dataset penelitian publik OhioT1DM
> (12 orang dewasa penyandang DM tipe 1 di Amerika Serikat, seluruhnya pengguna
> pompa insulin dan sensor CGM). Yang Anda nilai adalah kelayakan rancangan dan mutu
> keluarannya, bukan hasil penerapan klinis.

### Istilah lain pada formulir yang sebaiknya ikut dijelaskan

Alasan yang sama berlaku bagi istilah berikut, yang sudah dipakai di formulir tanpa
penjelasan:

| Istilah | Penjelasan singkat yang disarankan |
|---|---|
| CGM | *Continuous glucose monitoring*: sensor yang mengukur glukosa otomatis tiap beberapa menit. |
| Finger-stick / SMBG | Pengukuran glukosa mandiri dengan glukometer dan tusuk jari, tidak kontinu. |
| *Grounded* / berbasis rujukan | Setiap pernyataan sistem ditelusuri ke potongan dokumen pedoman beserta nomor halamannya. |
| *Doctor-mediated* | Keluaran sistem tidak sampai ke pasien secara langsung; dokter yang meninjau dan memutuskan. |
| Horizon prediksi | Seberapa jauh ke depan prediksi dibuat (30 dan 60 menit untuk CGM, sekitar 4 jam untuk finger-stick). |

---

## Berkas yang berubah

| Berkas | Perubahan |
|---|---|
| `backend/routes/patients.py` | Baru. `GET` daftar pasien dan `POST` pendaftaran pasien. |
| `backend/main.py` | Mendaftarkan router pasien. |
| `backend/services/supabase_data_service.py` | `list_patients()`, `create_patient()`, dan penyertaan badan galat PostgREST pada pesan kesalahan. |
| `frontend/src/lib/patients.ts` | Baru. Pemanggilan endpoint pendaftaran pasien. |
| `frontend/src/lib/patientState.ts` | Baru. Pasien aktif bersama untuk Dashboard, Logbook, dan Navbar. |
| `frontend/src/components/PatientSwitcher.tsx` | Baru. Panel pasien lama / pasien baru. |
| `frontend/src/components/CatatanSumberData.tsx` | Baru. Keterangan asal data dan status pengujian. |
| `frontend/src/components/Navbar.tsx` | Lencana pasien mengikuti pasien aktif, bukan `P001` tetap. |
| `frontend/src/app/page.tsx` | Memasang kedua komponen; pemilih pasien lama dicabut dari bagian sumber glukosa. |
| `frontend/src/app/logbook/page.tsx` | Memasang kedua komponen; sebelumnya tanpa pemilih pasien. |
| `frontend/src/app/dashboard.css` | Gaya kartu pasien dan catatan sumber data. |
