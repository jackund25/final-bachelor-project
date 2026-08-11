# Prapendaftaran T1.4 — sensitivitas konstanta waktu peluruhan IOB dan COB

> **Ditulis 11 Agustus 2026 pada commit `9141f3a`.**
> **T1.4 BELUM dijalankan.** `results/eval_prediksi/sensitivitas_tau_h6.json` belum ada dan
> tidak satu pun angka RMSE untuk konfigurasi tau mana pun sudah dilihat. Skrip
> `scripts/eval_sensitivitas_tau.py` sudah ada tetapi belum pernah dieksekusi; perubahan
> yang akan dilakukan padanya dicatat di bawah dan ditulis **setelah** berkas ini disimpan.

## Yang dijanjikan laporan, dan mengapa ini harus dijawab

Subbab II.3.5 menyebut `insulin_tau_min = 240` (kerja insulin ~4 jam) dan
`carbs_tau_min = 180` (penyerapan karbohidrat ~3 jam) diambil dari literatur, lalu
**menjanjikan sensitivitasnya diuji pada Bab VI**. Sampai audit, tidak ada satu pun skrip
yang memvariasikan kedua nilai itu. T1.4 menutup janji tersebut.

Yang dibutuhkan Bab II bukan "nilai tau terbaik", melainkan **pernyataan bahwa konstanta
waktu itu parameter rancangan, disertai bukti seberapa peka hasil terhadap pemilihannya.**
Bila hasilnya tidak peka, itu justru temuan yang menguatkan penyederhanaan.

## Aturan keputusan, ditetapkan sebelum melihat hasil

**Nilai tau produksi TIDAK diubah pada sesi ini, apa pun hasilnya.** Mengubah tau mengubah
**rekayasa fitur**, dan karena itu membatalkan sekaligus: seluruh model terlatih, seluruh
angka prediksi (T1.1, T1.1b, T1.3, T2.1, T4.1), seluruh kueri terkondisi, dan seluruh angka
penelusuran (crossfold, realcases, T3.1, T3.2, T3.3). Cakupan pembatalannya lebih luas
daripada mengganti model, karena tau ada di hulu segalanya.

Bila ada nilai yang lebih baik, ia masuk **saran pengembangan dengan besaran terukur**.

## DEFINISI "SD" YANG DIPAKAI — ditetapkan di muka

Ini bagian yang tidak ada pada prapendaftaran T3.1, T3.3, dan T4.1, dan sengaja ditambahkan
di sini. T4.1 menemukan aturan pelaporan *"selisih yang lebih kecil daripada simpangan baku
antar-fold tidak boleh dinarasikan"* punya **tiga** definisi yang memberi verdik berbeda,
dan HANDOFF tidak menyebut yang mana. Menetapkannya di muka mencegah cacat itu terulang.

**Pertama, sebuah kejujuran teknis: T1.4 TIDAK punya struktur fold.** Ia memakai pembagian
pasien tunggal — latih 8 pasien, penyetelan `ohio_584`/`ohio_588`, pelaporan
`ohio_591`/`ohio_596` — bukan validasi silang 6 fold seperti T1.1, T2.1, dan T4.1. Karena
itu **"SD antar-fold" TIDAK TERDEFINISI di T1.4**, dalam ketiga variannya sekalipun. Ini
dinyatakan supaya tidak ada yang kemudian mengutip T1.4 seolah ia punya variabilitas
tingkat fold.

**Yang dipakai sebagai skala derau, ditetapkan sekarang:**

| | |
|---|---|
| ukuran derau | **SD bootstrap RMSE pada set penyetelan**, 1.000 resample, benih 42 |
| mengapa ini | ia mengukur ketidakpastian penaksiran RMSE itu sendiri pada himpunan yang sama, yaitu derau yang benar-benar relevan ketika yang dibandingkan adalah RMSE dari himpunan identik |
| kriteria "peka" | **rentang RMSE seluruh sapuan harus melampaui 2× SD bootstrap** |
| pelengkap deskriptif | SD atas 9 nilai RMSE konfigurasi, dilaporkan tetapi **bukan** dasar verdik |
| uji beda per konfigurasi | Wilcoxon berpasangan atas \|galat\| per sampel terhadap konfigurasi produksi, alpha 0,05 — sudah ada di skrip dan tidak diubah |

Alasan memakai 2× dan bukan 1×: dengan 9 konfigurasi, rentang adalah statistik ekstrem
(maks − min) yang secara alami melebar seiring banyaknya konfigurasi, bahkan bila seluruhnya
menaksir besaran yang sama. Ambang 1× akan terlalu mudah dilewati.

## Bukti yang sudah ada, dan batas atas yang diturunkan darinya

J7 mengukur permutation importance pada set hold-out (n = 26.445, 5 ulangan):

| fitur | kenaikan RMSE bila **dimusnahkan** | porsi |
|---|---:|---:|
| `cob` | **0,6743 ± 0,0278** | 9,66% |
| `iob` | **0,4369 ± 0,0311** | 6,26% |
| keduanya | — | 15,92% |

**Inilah batas atas yang berprinsip.** Mengacak `iob` sepenuhnya — memusnahkan seluruh
informasinya — hanya menaikkan RMSE **0,4369 mg/dL**. Mengubah `insulin_tau_min` dari 120 ke
360 **tidak** memusnahkan fiturnya: ia tetap jumlah tertimbang peluruhan kejadian insulin,
hanya dengan konstanta waktu berbeda. Isi informasinya sebagian besar bertahan.

Maka pengaruh menggeser tau **wajib lebih kecil** daripada pengaruh memusnahkan fiturnya.

## Yang diperkirakan, dinyatakan sebelum diukur

### D1 — sapuan tau nyaris datar, jauh di bawah biaya memusnahkan fitur

**Perkiraan: rentang RMSE atas seluruh 9 konfigurasi berada di 0,05–0,30 mg/dL, dan pasti
di bawah 0,4369 mg/dL** (biaya permutasi `iob`).

Batas 0,4369 adalah bagian yang paling dapat dibantah dari prapendaftaran ini: bila rentang
sapuan **melampaui** biaya memusnahkan `iob`, penalaran "menggeser tau lebih ringan daripada
memusnahkan fitur" runtuh, dan yang harus dicurigai lebih dulu adalah apakah
`engineer_features` benar-benar dihitung ulang dari data mentah untuk tiap konfigurasi —
menumpuk peluruhan dua kali akan menghasilkan fitur yang rusak, bukan fitur ber-tau lain.

Konsekuensi bila D1 terpenuhi: **hasilnya tidak peka**, dan itulah yang dibutuhkan Bab II.
Pemilihan tau dari literatur menjadi **pilihan rancangan yang tidak menentukan hasil**, bukan
sumber ketidakpastian yang tersembunyi.

### D2 — sumbu karbohidrat lebih berpengaruh daripada sumbu insulin

`cob` berbobot permutasi 0,6743 lawan `iob` 0,4369, yaitu **1,54×**.

**Perkiraan: rentang RMSE pada sumbu carbs lebih besar daripada rentang pada sumbu insulin.**
Perkiraan nisbahnya searah dengan 1,54× tetapi tidak dipatok angkanya, karena hubungan antara
bobot permutasi dan kepekaan terhadap parameter tidak harus linear.

### D3 — titik produksi bukan titik minimum, dan itu tidak penting

Bila permukaannya datar (D1), letak argmin ditentukan derau, bukan struktur.

**Perkiraan: konfigurasi produksi (240/180) BUKAN yang RMSE-nya terendah, dan konfigurasi
terbaik mengalahkannya kurang dari 0,20 mg/dL.**

Perkiraan ini sengaja dibuat supaya "produksi bukan optimum" **tidak dapat dijual sebagai
temuan** ketika ia terjadi. Pada permukaan datar, itu hasil yang diharapkan.

### D4 — keunggulan di set penyetelan tidak bertahan di set pelaporan

Pola ini sudah muncul dua kali: T1.1b (p = 0,9546 pada set penyetelan menjadi p ≈ 0 pada set
pelaporan) dan T3.2 (pemenang `pred_h12` menjadi peringkat terakhir).

**Perkiraan: konfigurasi pembanding terbaik dari set penyetelan TIDAK mempertahankan
keunggulannya di set pelaporan — entah selisihnya tidak signifikan, entah tandanya
berbalik.**

### D5 — jumlah konfigurasi yang berbeda signifikan tinggi, dan itu artefak n

Uji Wilcoxon berpasangan atas |galat| per sampel berjalan pada n ≈ 29.000.

**Perkiraan: lebih dari separuh dari 8 konfigurasi non-produksi akan "berbeda signifikan"
(p < 0,05) meskipun rentang RMSE-nya di bawah 0,30 mg/dL.**

Bila terjadi, itu **bukan** bukti tau berpengaruh. Ia bukti bahwa uji signifikansi pada n
puluhan ribu menolak H₀ untuk perbedaan yang tidak berarti — alasan mengapa kriteria besaran
efek ditetapkan di muka, bukan kriteria p.

## Rancangan sapuan: SATU-PER-SATU, dan konsekuensinya bagi penafsiran

Disapu **terpisah**, bukan bersama:

| sumbu | nilai | yang ditahan |
|---|---|---|
| insulin | 120, 180, **240**, 300, 360 | `carbs_tau_min` = 180 |
| carbs | 60, 120, **180**, 240, 300 | `insulin_tau_min` = 240 |

Titik produksi (240/180) muncul **tepat sekali** dan dibagi kedua sumbu → **9 konfigurasi**,
bukan 10. Rentang dipilih mengelilingi nilai literatur dengan lebar yang masuk akal
fisiologis (kerja insulin dilaporkan 2–6 jam, penyerapan karbohidrat 1–5 jam), ditetapkan
sebelum satu pun angka dilihat.

Grid penuh 5×5 = 25 konfigurasi × ~9 menit > 3,5 jam sengaja **tidak** dipakai; sebagian
besar selnya tidak menjawab pertanyaan Subbab II.3.5, yang menanyakan kepekaan terhadap
**masing-masing** konstanta.

**Konsekuensi bagi penafsiran, dinyatakan terus terang:** rancangan OAT **tidak dapat
menangkap interaksi** antara kedua konstanta. Yang sah disimpulkan hanya kepekaan di
sekitar titik produksi, sepanjang satu sumbu pada satu waktu. Aturan penafsirannya:

* bila **kedua sumbu datar** (D1 terpenuhi), interaksinya hampir pasti juga datar — dua
  parameter yang masing-masing tidak menggerakkan hasil sulit menggerakkannya bersama;
* bila **salah satu sumbu berpengaruh besar**, interaksinya layak diperiksa dan itu
  dinyatakan sebagai pekerjaan lanjutan, bukan diselundupkan sebagai kesimpulan T1.4.

## Perubahan pada skrip, dicatat sekarang

1. **Checkpoint per konfigurasi tau** ke `.cache/sensitivitas_tau_h{N}/`, bukan `results/`,
   menyimpan `yte` dan `yp` per konfigurasi sehingga |galat| per sampel tersedia dan
   Wilcoxon dapat dilanjutkan tanpa melatih ulang. Sidik jari memuat horizon,
   `max_gap_steps`, pembagian pasien, benih, konfigurasi RF, dan daftar sapuan; checkpoint
   tak sepadan **ditolak dengan pesan jelas**. Ini sapuan ~100 menit tanpa checkpoint sama
   sekali — persis keadaan yang membuat T1.1b kehilangan sepuluh konfigurasi.
2. **SD bootstrap RMSE** pada set penyetelan, 1.000 resample benih 42, untuk konfigurasi
   produksi — menjadi skala derau yang dideklarasikan di atas.
3. **Keempat penjaga diverifikasi dengan data tiruan** sebelum jalan panjang, seperti T4.1:
   penolakan checkpoint tak sepadan, pemulihan checkpoint sepadan, penjaga titik produksi
   wajib ada dalam sapuan, dan jalur serialisasi JSON.

Tidak satu pun mengubah cara model dilatih atau dinilai; angka yang dihasilkan tetap sepadan
dengan T1.1b yang memakai pembagian pasien identik.

## Kaitan

- `docs/PRAPENDAFTARAN_T4.1.md` — sumber temuan tiga definisi SD yang memaksa bagian definisi di atas
- `results/eval_prediksi/feature_importance.json` — sumber batas atas 0,4369 dan 0,6743
- `results/eval_prediksi/rf_ukuran_h6.json` — pembagian pasien identik
- Subbab II.3.5 laporan — janji yang ditutup percobaan ini
