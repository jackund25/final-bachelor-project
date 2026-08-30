# Audit Repositori

Ditulis 30 Agustus 2026. Mencakup dua hal yang diminta: kode/berkas yang sudah
tidak relevan, dan berkas berisi informasi khusus internal yang perlu dipindahkan
ke arsip.

Seluruh temuan diverifikasi terhadap isi repositori, bukan diperkirakan dari nama
berkas. Tidak ada berkas yang dihapus dalam audit ini.

**Catatan pelaksanaan:** operasi `mv` dan `rmdir` diblokir sistem dengan
`Permission denied`, sehingga pemindahannya belum dijalankan. Perintahnya
tersedia pada Bagian 6 untuk dijalankan manual.

---

## 1. Ringkasan temuan

| # | Temuan | Jumlah | Sifat |
|---|---|---|---|
| A | Berkas/direktori berisi informasi internal yang masih di pohon aktif | 3 | pindahkan |
| B | Dua berkas yang perlu keputusan Anda, bukan keputusan saya | 2 | konfirmasi |
| C | Modul `src/` tanpa pemanggil | 5 (+1 yang WAJIB dipertahankan) | pindahkan/tinjau |
| D | Direktori hasil yang sudah tersalip | 12 | pindahkan |
| E | Rujukan menggantung ke dokumen yang sudah diarsipkan | 20+ | perbaiki |
| F | Istilah yang sudah dicabut masih tertulis | 3 berkas | perbaiki |

---

## 2. Temuan A — informasi internal yang masih berada di pohon aktif

Kriteria yang saya pakai diturunkan dari isi `arsip/` dan `docs/arsip/` yang sudah
ada: dokumen kerja, catatan bimbingan, bahan presentasi, protokol percobaan, dan
keluaran yang dibangkitkan ulang.

### A.1 `TA_SOURCE_AUDIT/` → `arsip/`

Bundel audit kode sumber berukuran 15 MB berisi `MANIFEST.csv`, `MANIFEST.md`,
`SUMMARY.json`, dan `SOURCE_CODE_BUNDLE.zip`. Seluruhnya dibangkitkan ulang oleh
`collect_ta_source.py`, jadi tidak ada informasi yang hilang bila dipindahkan.

Sudah saya tambahkan ke `.gitignore` pada sesi sebelumnya, sehingga ia tidak akan
masuk commit — tetapi secara fisik masih menempati akar repositori.

### A.2 `docs/proposal_ta/` → `docs/arsip/`

Sudah diabaikan git dengan alasan yang tertulis pada `.gitignore` sendiri: memuat
figur dari publikasi berhak cipta dan lembar pengesahan bertanda tangan, serta
sudah tidak selaras dengan TA final. Alasan itu tepat, tetapi tempatnya belum
mengikuti.

### A.3 `scripts/notepad scripts/` → hapus

Direktori kosong. Tidak terlacak git, tidak berisi apa pun.

---

## 3. Temuan B — dua berkas yang perlu keputusan Anda

Keduanya memenuhi kriteria "internal", tetapi memindahkannya bisa merugikan dan
saya tidak punya dasar untuk memutuskannya sendiri.

### B.1 `docs/pape_iciss/` — apakah masih aktif?

`.gitignore` Anda sendiri menggolongkannya sebagai "draft internal". Isinya draf
paper ICISS2026 (`.tex`, `.md`, `.pdf`), template IEEE, dan satu berkas bernama
`contoh paper daffari untuk wording.pdf` yang tampaknya makalah pihak lain yang
dipakai sebagai rujukan gaya penulisan.

**Kalau paper itu masih dalam proses penyerahan, jangan dipindahkan** — draf yang
aktif bukan berkas tidak relevan. Kalau sudah selesai atau ditinggalkan, ia layak
masuk `docs/arsip/`.

Terlepas dari keputusan itu, `contoh paper daffari untuk wording.pdf` sebaiknya
dipindahkan ke `docs/arsip/` sekarang: ia karya pihak lain dan tidak punya peran
dalam artefak maupun laporan.

### B.2 `docs/DEPLOY_LOKAL_NGROK.md` — saya sarankan DIPERTAHANKAN, dengan satu suntingan

Berkas ini saya commit pada sesi sebelumnya. Setelah dibaca ulang, isinya sebagian
besar justru bahan Bab V yang berharga: alasan backend tidak ditempatkan di awan,
disertai jejak memori terukur per tahap (python 21 → fastapi 47 → route 96 →
PredictionService 157 → RAGPipeline 266 → build 342 MB) terhadap batas 512 MB
instans gratis. Argumen itu tidak ada di tempat lain.

Yang benar-benar internal hanyalah nama host terowongan pribadi, muncul tiga kali
(baris 47, 65, 190). Sarannya: ganti menjadi `https://<host-terowongan-anda>` dan
pertahankan berkasnya, alih-alih mengarsipkan seluruh dokumen.

---

## 4. Temuan C, D — kode dan hasil yang sudah tidak relevan

### C. Modul `src/` yang tidak pernah diimpor

Kelima modul berikut tidak diimpor dari mana pun (`src/`, `backend/`, `scripts/`,
`tests/`). Semuanya alat CLI sekali pakai dari masa percobaan, dan satu-satunya
yang menyebut namanya adalah `TA_SOURCE_AUDIT/MANIFEST.md` yang bersifat
bangkitan:

    src/data/error_analysis.py
    src/data/smbg_interval_audit.py
    src/data/smbg_training_sample_audit.py
    src/data/target_audit.py
    src/data/temporal_audit.py

Tujuan yang konsisten dengan kebiasaan repo ini: `docs/arsip/skrip/`.

**PENGECUALIAN YANG PENTING — jangan ikut dipindahkan:**

    src/data/ohio_parser.py

Ia juga tidak diimpor, tetapi ia penghasil dataset dan dirujuk oleh `README.md`
serta `docs/METHODOLOGY.md`. Memindahkannya akan memutus jalur reproduksi dari
berkas mentah OhioT1DM ke `data/raw/`. Ketiadaan impor di sini menandakan ia
dijalankan sebagai perintah, bukan menandakan ia mati.

### D. Direktori hasil yang sudah tersalip

Rantai penamaannya memperlihatkan urutan percobaan:

    retrieval_realcases
      -> retrieval_realcases_kb12
      -> retrieval_realcases_kb12_final
      -> retrieval_realcases_kb12_gbm
      -> retrieval_realcases_kb12_gbm_bm25
      -> retrieval_realcases_kb12_gbm_bm25_v2
      -> retrieval_realcases_kb12_gbm_bm25_final

Yang benar-benar dirujuk naskah laporan atau skrip pembangun angka Bab VI hanya
tiga:

    results/retrieval_realcases_kb12            (naskah)
    results/retrieval_realcases_kb12_final      (buat_ringkasan_bab6.py)
    results/retrieval_realcases_kb12_gbm_bm25_v2 (naskah)

Tujuh sisanya tidak dirujuk siapa pun:

    results/retrieval_realcases
    results/retrieval_realcases_kb12_sym
    results/retrieval_realcases_kb12_gbm
    results/retrieval_realcases_kb12_gbm_bm25
    results/retrieval_realcases_kb12_gbm_bm25_final
    results/retrieval_realcases_kb12_gbm_hibrida
    results/retrieval_realcases_kb12_gbm_kalimat
    results/retrieval_realcases_kb12_gbm_ml

Lima direktori `baseline_ablation*` juga tidak dirujuk naskah, tetapi berbeda
sifatnya: ia keluaran tujuh skrip yang masih ada (`ablation_rag_fullkb.py`,
`eval_chunk_size.py`, `eval_embedding_alternative.py`, `eval_kombinasi_b4b5.py`,
`eval_mmr_lambda.py`, `eval_reranking.py`, `eval_reranking_bentuk_kueri.py`),
sehingga dapat dibangun ulang. Aman diarsipkan, tetapi bukan yatim.

**Satu peringatan sebelum memindahkan.** `results/eval_rag/T13_hibrida_TEMUAN.md`
mengambil angkanya dari `results/retrieval_realcases_kb12_gbm_kalimat/crossfold.json`
— salah satu yang masuk daftar di atas. Bila direktori itu dipindahkan, perbarui
rujukan di dalam berkas temuan tersebut agar tidak ikut menggantung.

---

## 5. Temuan E, F — yang perlu diperbaiki, bukan dipindahkan

### E. Rujukan menggantung ke dokumen yang sudah diarsipkan

Lebih dari 20 rujukan pada berkas aktif menunjuk `docs/*.md` yang kini berada di
`docs/arsip/dokumen-kerja/`. Contoh yang paling terlihat: `T13_hibrida_TEMUAN.md`
menyebut `docs/PRAPENDAFTARAN_T13_HIBRIDA.md`. Berkas berikut termasuk yang
ditunjuk namun sudah tidak ada di lokasi lamanya:

    ALUR_SISTEM_FINAL.md          KEPUTUSAN_DIAMBIL.md
    ANALISIS_CHUNKING_T7.md       MIGRATION.md
    ARGUMEN_GBM.md                NASKAH_PENUTUP_BAB_VI.md
    AUDIT_KEBUTUHAN.md            PANDUAN_PENILAI_v2.md
    DAFTAR_KETERBATASAN.md        PRAPENDAFTARAN_T13_HIBRIDA.md
    HANDOFF.md                    PRAPENDAFTARAN_T14_RAGAS_LENGAN.md
    ...                           (dan seterusnya)

Ini efek samping arsip putaran sebelumnya. Perbaikannya cukup mengganti awalan
`docs/` menjadi `docs/arsip/dokumen-kerja/` pada berkas yang merujuk.

### F. Istilah yang sudah dicabut masih tertulis

    README.md            2 kemunculan "digital twin"
    docs/METHODOLOGY.md  2 kemunculan
    config.yaml          1 kemunculan

Yang paling perlu diperhatikan `README.md`, karena ia halaman pertama repositori
dan masih memposisikan sistem sebagai *simplified data-driven digital twin*.
Posisi itu sudah ditinggalkan; penelitian ini kini diposisikan sebagai RAG
terkondisi-prediksi untuk DM tipe 1. Nama direktori repositori sendiri
(`diabetes-digital-twin`) membawa istilah yang sama — mengganti nama direktori
bersifat opsional dan berdampak ke jalur kerja, jadi tidak saya sarankan sekarang.

---

## 6. Perintah untuk dijalankan manual

Dijalankan dari akar repositori. Tidak ada yang dihapus kecuali satu direktori
kosong; sisanya pemindahan yang dapat dikembalikan.

### 6.1 Temuan A — aman, tidak menyentuh berkas terlacak git

    mkdir -p docs/arsip/skrip
    mv TA_SOURCE_AUDIT arsip/TA_SOURCE_AUDIT
    mv docs/proposal_ta docs/arsip/proposal_ta
    rmdir "scripts/notepad scripts"

### 6.2 Temuan C — modul CLI sekali pakai

    git mv src/data/error_analysis.py              docs/arsip/skrip/
    git mv src/data/smbg_interval_audit.py         docs/arsip/skrip/
    git mv src/data/smbg_training_sample_audit.py  docs/arsip/skrip/
    git mv src/data/target_audit.py                docs/arsip/skrip/
    git mv src/data/temporal_audit.py              docs/arsip/skrip/

Sesudahnya jalankan `pytest` untuk memastikan tidak ada tes yang bergantung
padanya. Menurut penelusuran saya tidak ada, tetapi verifikasi lebih murah
daripada temuan belakangan.

### 6.3 Temuan D — hasil percobaan yang tersalip

    mkdir -p docs/arsip/hasil
    git mv results/retrieval_realcases                       docs/arsip/hasil/
    git mv results/retrieval_realcases_kb12_sym              docs/arsip/hasil/
    git mv results/retrieval_realcases_kb12_gbm              docs/arsip/hasil/
    git mv results/retrieval_realcases_kb12_gbm_bm25         docs/arsip/hasil/
    git mv results/retrieval_realcases_kb12_gbm_bm25_final   docs/arsip/hasil/
    git mv results/retrieval_realcases_kb12_gbm_hibrida      docs/arsip/hasil/
    git mv results/retrieval_realcases_kb12_gbm_ml           docs/arsip/hasil/

`retrieval_realcases_kb12_gbm_kalimat` sengaja tidak dimasukkan: perbarui dulu
rujukan di `results/eval_rag/T13_hibrida_TEMUAN.md`, baru pindahkan.

### 6.4 Yang JANGAN dijalankan

    src/data/ohio_parser.py        -> pertahankan (penghasil dataset, dirujuk README)
    docs/DEPLOY_LOKAL_NGROK.md     -> pertahankan, cukup samarkan host ngrok
    docs/pape_iciss/               -> tunggu keputusan Anda soal status paper

---

## 7. Yang sudah saya periksa dan ternyata bersih

Dicatat supaya tidak diperiksa ulang di kemudian hari.

- **Tidak ada kredensial yang bocor.** `.env` dan `.env*` sudah diabaikan git.
  Penelusuran token, kata sandi, dan kunci API pada berkas terlacak tidak
  menemukan apa pun. Satu-satunya identitas jaringan yang muncul adalah nama host
  ngrok pada Bagian B.2, dan itu bukan kredensial.
- **Tidak ada data pribadi penilai di dalam repositori.** `results/evaluasi_ahli/ringkasan.json`
  memakai kode P1–P4 tanpa nama maupun surel. Berkas tanggapan formulir yang
  memang memuat nama dan surel sudah berada di `docs/arsip/` dan sudah diabaikan
  git.
- **Direktori `evaluation/`** memuat instrumen penelitian yang sah (kunci jawaban,
  catatan, penilaian dua penilai) dan tetap relevan.
- **35 dari 41 modul `src/`** punya pemanggil yang jelas.
