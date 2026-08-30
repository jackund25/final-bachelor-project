# Menjalankan sistem untuk sesi evaluasi dokter

Ditulis 24 Agustus 2026, menggantikan rencana penerapan awan.

**Susunannya:** frontend tetap di Vercel, **hanya backend** yang berjalan di laptop
dan diekspos lewat terowongan ngrok berdomain statis.

---

## Mengapa begini

**Mengapa backend tidak di awan.** Instans Render gratis dibatasi 512 MB,
sedangkan jejak memori terukur backend adalah **342 MB** (python 21 → fastapi 47 →
route 96 → PredictionService 157 → RAGPipeline() 266 → build() 342). Sisa 170 MB
itu habis oleh lingkungan Linux Render, sehingga pemanasan ter-OOM berulang sebelum
indeks BM25 atas 2.233 potongan selesai dibangun. Menaikkan ke 2 GB berbiaya
$25/bulan, dan Docker Space di Hugging Face kini menuntut langganan PRO.

**Mengapa frontend TETAP di Vercel.** Menjalankan keduanya di laptop sempat
dipertimbangkan dan ditinggalkan: setiap aset halaman akan melewati terowongan
gratis, dan dokter akan merasakan halaman yang berat. "Assessment lama" sudah
menjadi salah satu dari tiga keluhan pada evaluasi pertama — menambah kelambatan
pada lapisan yang tidak perlu adalah pertukaran yang salah arah. Vercel melayani
halaman dari CDN; hanya panggilan `/api/*` yang melewati laptop, dan itu memang
bagian yang harus lewat sana.

Keuntungan tambahan: dokter melihat URL aplikasi yang wajar, bukan domain
terowongan yang terlihat seperti tautan sementara.

**Konsekuensi yang tidak bisa dihindari.** Backend hanya hidup selama laptop
menyala. Evaluasi karena itu harus berupa **sesi terjadwal** — Anda mendampingi,
bukan mengirim tautan lalu dokter mengisi kapan saja.

Untuk TA ini pertukarannya menguntungkan: Anda dapat mengamati di mana dokter
bingung dan menanyakan lanjutan — data kualitatif yang tidak akan pernah keluar
dari Google Form.

---

## Arsitektur sesi

```
Peramban dokter
   |
   |-- halaman  --> Vercel (CDN)
   |
   `-- /api/*   --> https://rumble-drainable-gatherer.ngrok-free.dev
                          |
                          v
                    uvicorn :8000  (laptop, 342 MB)
                          |
                          v
                 ChromaDB + BM25 + Gemini + Supabase
```

---

## Persiapan sekali saja

### 1. Arahkan Vercel ke terowongan

Vercel → Settings → Environment Variables:

```
NEXT_PUBLIC_API_URL = https://rumble-drainable-gatherer.ngrok-free.dev
```

Lalu **redeploy frontend**. Variabel `NEXT_PUBLIC_*` dipanggang saat build, jadi
mengubah nilainya saja tidak berpengaruh sampai ada build ulang.

Karena domain ngrok-nya statis, langkah ini cukup **sekali** untuk seluruh masa
evaluasi.

### 2. Tidak ada yang perlu diubah di Supabase

Kunci `sb_publishable_*` Anda **tidak membatasi origin**. Terverifikasi dengan
preflight dari tiga origin, termasuk domain karangan yang tidak ada hubungannya
dengan proyek ini:

| Origin | `access-control-allow-origin` |
|---|---|
| domain Vercel | `*` |
| domain ngrok | `*` |
| `https://contoh-asing.example.com` | `*` |

> **Koreksi.** Versi pertama dokumen ini menyatakan Supabase memblokir origin asing
> dan menyuruh mendaftarkan domain ngrok. Itu **keliru**. Dugaan tersebut berangkat
> dari galat 403 yang terlihat di console dan disimpulkan tanpa memeriksa URL-nya.
> Setelah diperiksa, seluruh 403 itu menimpa `/_next/static/chunks/*.js` — berkas
> aplikasi sendiri — dan **tidak satu pun permintaan pernah sampai ke
> `supabase.co`**. Sebab sesungguhnya ada di bagian berikutnya.

### 2b. MENGAPA FRONTEND TIDAK BOLEH IKUT DITEMBUS

Ini pelajaran yang dibayar mahal, jadi jangan diulang.

Menembus frontend (`ngrok http ... 3000`) tampak masuk akal dan **gagal total**.
Terukur langsung pada domain ngrok Anda:

```
/_next/static/chunks/node_modules_@supabase_0kwankj._.js  -> 403
/_next/static/chunks/node_modules_@supabase_0kwankj._.js  -> 200
/_next/static/chunks/node_modules_@supabase_0kwankj._.js  -> 403
/_next/static/chunks/node_modules_@supabase_0kwankj._.js  -> 200
```

Berselang-seling tanpa henti. Dua sebab bertumpuk:

1. **Halaman peringatan ngrok** (`ERR_NGROK_6024`) menahan permintaan navigasi
   peramban. Header `ngrok-skip-browser-warning` tidak menolong di sini — header
   tidak dapat disisipkan pada navigasi, hanya pada `fetch`.
2. **Batas laju ngrok gratis.** `next dev` memancarkan puluhan chunk kecil plus
   polling HMR pada setiap muat halaman. Terowongan gratis menolak sebagian,
   aplikasi mencoba ulang, ditolak lagi.

Akibatnya klien Supabase tidak pernah selesai dimuat, sehingga dasbor menampilkan
`NO_DATA` dengan dropdown pasien kosong — gejala yang menyerupai kegagalan basis
data, padahal tidak ada satu pun kueri yang pernah dikirim.

Susunan pada dokumen ini menghindari keduanya: peramban menavigasi ke **Vercel**
(tidak ada halaman peringatan, aset dari CDN), dan hanya panggilan `/api/*` yang
melewati terowongan — satu permintaan per assessment, bukan puluhan per muat
halaman. Panggilan itu berupa `fetch`, sehingga `ngrok-skip-browser-warning` yang
dipasang `headerApi()` benar-benar berlaku.

### 3. CORS sudah siap

`allow_origins` di [backend/main.py](../backend/main.py) sudah memuat origin Vercel
Anda. Preflight terverifikasi, termasuk untuk header khusus ngrok di bawah.

### 4. Header pelewat peringatan ngrok

Sudah terpasang di [frontend/src/lib/api.ts](../frontend/src/lib/api.ts) lewat
`headerApi()`, dipakai `runClinicalAssessment` dan `saveLogbookEntry`.

Tanpa header `ngrok-skip-browser-warning`, ngrok gratis menyisipkan halaman
peringatan HTML pada permintaan yang terlihat berasal dari peramban — termasuk
`fetch()`. Akibatnya `response.json()` gagal mengurai HTML, dan dokter menerima
galat yang sama sekali tidak menjelaskan sebabnya.

---

## Menjalankan sesi

Dua terminal, biarkan keduanya terbuka.

**Terminal 1 — backend**

```bash
JEDA_PEMANASAN_DETIK=0 python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`JEDA_PEMANASAN_DETIK=0` mematikan jeda pemanasan yang hanya diperlukan di Render,
tempat instans lama dan baru berimpit selama deploy. Di sini pemanasan boleh mulai
seketika; ia tuntas dalam ~15 detik.

**Terminal 2 — terowongan**

```bash
ngrok http --domain=rumble-drainable-gatherer.ngrok-free.dev 8000
```

Perhatikan **8000**, bukan 3000. Yang ditembus adalah backend.

---

## Daftar periksa sebelum dokter masuk

**1. Backend siap — periksa dari mesin sendiri**

```bash
curl -s http://127.0.0.1:8000/health
```

| Medan | Nilai yang benar | Bila salah |
|---|---|---|
| `pemanasan.selesai` | `true` | masih membangun, tunggu ~15 detik |
| `retrieval.n_potongan` | **2233** | `36` berarti Chroma tidak terbaca |
| `retrieval.terdegradasi` | `false` | `true` berarti mundur ke potongan cadangan |
| `llm.siap` | `true` | `false` berarti advisory dijawab templat |

Kegagalan `n_potongan: 36` bersifat **senyap**: jawaban tetap keluar dan sitasinya
tetap membawa nomor halaman sungguhan, hanya saja penelusuran cuma menjangkau 1,6%
korpus. Tidak ada yang tampak salah. Karena itu angka ini wajib dilihat, bukan
diasumsikan.

**2. Terowongan tembus**

```bash
curl -s -H "ngrok-skip-browser-warning: 1" https://rumble-drainable-gatherer.ngrok-free.dev/health
```

Harus mengembalikan JSON yang sama. Kalau yang keluar HTML, header-nya tidak
terkirim atau terowongannya belum jalan.

**3. Buka aplikasi Vercel dari HP, bukan dari laptop**

Ini menguji jalur yang sesungguhnya dipakai dokter. Yang dipastikan: daftar pasien
P001–P008 muncul, angka glukosa terisi, dan satu assessment berjalan sampai
tuntas.

Yang harus terlihat pada hasilnya: **lima kartu SBAR terpisah**, tanpa tanda `**`
mentah, dan chip `[S1]` yang bisa diklik untuk menggulir ke rujukannya.

---

## Selama sesi

- **Matikan sleep laptop.** Layar boleh mati, mesin jangan.
- **Jangan jalankan eksperimen Tahap 8 di hari yang sama.** Crossfold menembak
  ratusan permintaan Gemini beruntun dan memicu 429; kalau itu terjadi saat dokter
  memakai sistem, mereka menerima jawaban templat dan menilai yang salah.
- **Panaskan jalurnya lebih dulu.** Jalankan satu assessment sendiri sebelum dokter
  masuk.
- ngrok gratis membatasi jumlah koneksi per menit. Untuk sesi satu-dua dokter ini
  tidak akan tersentuh.

---

## Sesudah evaluasi

Kembalikan `NEXT_PUBLIC_API_URL` di Vercel ke backend awan bila kelak dipakai
lagi, lalu redeploy. Selama variabelnya menunjuk domain ngrok, aplikasi Vercel
hanya berfungsi ketika laptop Anda menyala dan terowongannya hidup.

---

## Untuk naskah TA

Tulis terus terang bahwa evaluasi dijalankan pada **penerapan hibrida**: antarmuka
di awan, mesin inferensi pada perangkat peneliti yang diekspos lewat terowongan.
Sertakan alasannya — jejak memori 342 MB melampaui kapasitas efektif instans
gratis 512 MB.

Ini bukan kelemahan yang perlu disembunyikan. DECIDE-AI (Vasey dkk., Nat Med
2022;28:924-933) memang menempatkan evaluasi tahap awal pada konteks terbatas.
Yang menjadi masalah hanya bila naskah mengklaim sistem sudah berjalan di
produksi padahal tidak.

---

## Berkas terkait yang kini tidak terpakai

`Dockerfile` dan `.dockerignore` pernah disusun untuk Hugging Face Spaces, sebelum
diketahui bahwa Docker Space di sana menuntut langganan PRO. Keduanya DIHAPUS pada
30 Agustus 2026 karena tidak pernah dipakai jalur penerapan mana pun: tidak ada CI
yang membangun citra, naskah laporan tidak menyebutnya, dan penerapan yang benar-benar
dijalankan adalah frontend di Vercel dengan backend lokal lewat terowongan.

Ketentuan yang perlu diingat bila kelak penerapan berbasis kontainer dibutuhkan —
Google Cloud Run yang paling dekat — hanya dua, dan keduanya tercatat di tempat lain:
host memerlukan memori DI ATAS 512 MB (1 GB sudah lapang), sesuai pengukuran pada
bagian "Mengapa begini" di atas; dan `scikit-learn` wajib tetap 1.3.0 karena bundel
inferensi `.pkl` dibuat dengan versi itu, sebagaimana dicatat pada
`requirements.txt` dan `backend/requirements.txt`.

Proksi `rewrites()` di [frontend/next.config.ts](../frontend/next.config.ts) juga
tidak terpakai pada susunan ini. Ia dipertahankan karena menjadi jalur cadangan
bila kelak frontend perlu ikut dijalankan lokal: dengan
`NEXT_PUBLIC_API_URL=""` (string kosong), permintaan menjadi relatif dan
diteruskan proksi, sehingga cukup satu terowongan dan CORS tidak terlibat.
