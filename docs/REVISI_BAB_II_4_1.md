# Revisi Bab II subbab II.4.1 — pembenaran pemilihan Random Forest

> **Disiapkan 11 Agustus 2026 · keputusan #5, pilihan D · commit `db4184f`**
>
> **Naskah bab TIDAK disunting langsung.** Berkas ini memuat naskah pengganti untuk
> ditempelkan sendiri oleh penulis ke
> `docs/laporan_TA/TA-STI-template-1.0/Bab II - Studi.tex`.

---

## Mengapa subbab ini harus direvisi, bukan ditambah

Naskah lama membenarkan pemilihan Random Forest atas **dua alasan**, dan percobaan proyek
ini melemahkan **keduanya**. Keduanya **dibuang**, bukan diperhalus atau ditambal.

### Alasan lama 1 — keterjelasan kontribusi fitur

Yang sebenarnya dimiliki RF bukan "dapat dijelaskan sedangkan model lain tidak" —
permutation importance berlaku untuk model apa pun, termasuk LSTM. Yang RF miliki adalah
kontribusi fitur yang tersedia **bawaan dan murah** lewat `feature_importances_`.

**J7 menggugurkan tepat keunggulan itu.** `feature_importances_` adalah MDI, dan MDI **tidak
boleh dilaporkan** karena bias kardinalitas: `iob` punya 165.950 nilai unik lawan `glucose`
361, dan MDI melebihkan kontribusi insulin/karbohidrat menjadi 21,67% dari angka sebenarnya
**15,92%**, serta melebihkan fitur diurnal sampai 24×. Angka yang dilaporkan adalah
**permutation importance**, yang harus dihitung terpisah **pada model apa pun, termasuk
RF**.

Satu-satunya keterjelasan yang RF miliki bawaan adalah keterjelasan yang **ternyata tidak
dapat dipakai**.

### Alasan lama 2 — kebutuhan komputasi rendah

**Terbalik secara faktual**, dan sudah terbalik sejak T1.1:

| | RF | GBM | LSTM |
|---|---:|---:|---:|
| waktu latih (+30 mnt) | **708 dtk** | **3,4 dtk** | 477 dtk |
| ukuran model | **322 MB** | **0,485 MB** | **0,124 MB** |

RF adalah yang **paling mahal** dari ketiganya pada kedua besaran.

---

## NASKAH PENGGANTI

> Salin bagian di bawah ini. Sesuaikan perintah `\autocite` dan `\ref` dengan kunci yang
> berlaku pada naskah Anda.

Random Forest dipilih sebagai prediktor dasar pada awal penelitian ini. Pemilihan tersebut
bersifat **awal dan pragmatis**: ansambel pohon bekerja baik pada data tabular berukuran
sedang, tidak menuntut penyetelan ekstensif untuk mencapai kinerja yang wajar, dan
menghasilkan model yang deterministik sehingga seluruh percobaan hilir dapat direproduksi
tepat.

Pemilihan itu **tidak** diklaim sebagai pemilihan model terbaik. Perbandingan sistematis
terhadap dua keluarga model lain — jaringan rekuren LSTM dan *gradient boosting* — dilakukan
dan dilaporkan pada Bab~\ref{chap:evaluasi}, mencakup galat regresi, zona Clarke,
sensitivitas hipoglikemia, ukuran model, serta waktu latih dan inferensi. Hasil perbandingan
tersebut menunjukkan bahwa Random Forest **bukan** yang terbaik pada seluruh besaran yang
diukur, dan hal itu dinyatakan apa adanya pada bab tersebut.

Prediktor **tidak diganti pada akhir penelitian**, dan alasannya bersifat metodologis, bukan
karena Random Forest terbukti unggul. Kontribusi penelitian ini adalah **query
transformation terkondisi-prediksi** (*Prediction-Conditioned RAG*); prediktor merupakan
**komponen** yang memasok nilai terkondisi, bukan kontribusi yang sedang diuji. Mengganti
prediktor membatalkan seluruh rantai evaluasi yang bergantung padanya — validasi silang
*retrieval* lintas-pasien, evaluasi kasus nyata, ketiga percobaan susunan kueri, kalibrasi
konformal, serta evaluasi generasi yang dibatasi kuota model bahasa. Selisih akurasi yang
diperoleh dari penggantian tersebut, sebagaimana ditunjukkan Bab~\ref{chap:evaluasi}, berada
**dua orde besaran di bawah** toleransi per-pengukuran alat yang menghasilkan datanya,
sehingga tidak dapat ditafsirkan secara fisik. Menukar keabsahan seluruh rantai evaluasi
demi selisih sebesar itu tidak dapat dipertanggungjawabkan.

Keunggulan *gradient boosting* yang **tidak** bergantung pada ambang kebermaknaan mana pun
adalah efisiensinya: berkas modelnya sekitar 665 kali lebih kecil dan waktu latihnya sekitar
208 kali lebih singkat daripada Random Forest pada konfigurasi yang dibandingkan. Temuan itu
dicatat sebagai **saran pengembangan** pada Bab~\ref{chap:penutup}, bukan sebagai perubahan
yang dilakukan dalam penelitian ini.

---

## Catatan penempatan

1. Subbab ini kini **merujuk ke depan** ke Bab VI. Pastikan label `\label{chap:evaluasi}`
   dan `\label{chap:penutup}` sesuai dengan yang dipakai naskah Anda.
2. Kalimat *"Random Forest bukan yang terbaik pada seluruh besaran yang diukur"* harus
   **benar-benar didukung** Bab VI. Tabel tiga model di
   `docs/RINGKASAN_KEPUTUSAN_PEMBIMBING.md` bagian #5 sudah memuat angkanya.
3. Bila Bab VI belum memuat angka GBM, jangan tempelkan paragraf keempat lebih dulu —
   rujukan ke depan yang tidak dapat ditemukan pembaca lebih buruk daripada tidak ada.
4. Angka 665× dan 208× berasal dari `results/eval_prediksi/gradient_boosting_h6.json`
   (322,53 MB / 0,485 MB dan 708,1 dtk / 3,4 dtk). Keduanya dibulatkan ke bawah.
