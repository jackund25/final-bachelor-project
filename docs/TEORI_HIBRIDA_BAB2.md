# Bahan teori penelusuran hibrida untuk Bab II

Disusun 17 Agustus 2026, sebagai bahan subbab baru pada Bab II setelah penelusuran
hibrida diadopsi ke produksi (commit `ca5c919`).

**Status bacaan tiap rujukan dinyatakan terus terang di Bagian 5.** Aturan proyek
melarang mengutip yang belum dibaca, dan aturan itu berlaku pula di sini.

---

## 1. Dua keluarga penelusuran, dan mengapa keduanya tidak saling menggantikan

Penelusuran dokumen terbagi dua keluarga yang bekerja pada dasar yang berbeda.

**Penelusuran padat (*dense*)** memetakan kueri dan dokumen ke vektor pada satu ruang
makna, lalu mengukur kedekatannya. Inilah yang dipakai sistem ini melalui
`all-MiniLM-L6-v2` dan MMR \autocite{carbonell1998}. Kekuatannya menangkap kesamaan
makna meski katanya berbeda; kelemahannya bergantung sepenuhnya pada mutu ruang makna
itu untuk bahasa dan domain yang dihadapi.

**Penelusuran jarang (*sparse*)** menimbang kecocokan istilah secara harfiah. Skema
yang paling mapan adalah Okapi BM25, yang diuraikan Manning dkk. \autocite{manning2008}
pada Bab 11 sebagai pengembangan model relevansi probabilistik yang memperhitungkan
frekuensi istilah dan panjang dokumen. Kekuatannya menemukan istilah persis;
kelemahannya buta terhadap padanan kata yang tidak sama bentuknya.

Lee dkk. \autocite{lee2024} menyatakan pertukaran itu dengan tepat pada konteks
pedoman diabetes: penelusur padat unggul menangkap hubungan makna tetapi **dapat
melewatkan kecocokan kata kunci yang justru penting bagi peristilahan medis**,
sedangkan penelusur jarang menemukan istilah spesifik tetapi kerap melewatkan
informasi yang berkaitan secara konseptual. Contoh yang mereka pakai berasal dari
domain yang sama dengan penelitian ini: penelusur padat memahami konsep *pengelolaan
glukosa* namun meleset pada nama obat tertentu, sementara penelusur jarang menemukan
nama obatnya namun meleset pada kontraindikasi yang menyertainya.

---

## 2. Penggabungan peringkat: Reciprocal Rank Fusion

Kedua keluarga menghasilkan **skor yang tidak sebanding**. Skor kosinus berkisar pada
selang tertentu, sedangkan skor BM25 tidak berbatas dan bergantung pada statistik
korpus. Menjumlahkannya langsung tidak sah; menormalkannya lebih dulu menambah satu
parameter bebas yang harus disetel, dan setiap parameter yang disetel menuntut
protokol penyetelannya sendiri.

*Reciprocal Rank Fusion* menghindari persoalan itu dengan bekerja pada **peringkat**,
bukan skor:

$$\mathrm{skor}(d) = \sum_{r \in R} \frac{1}{k + \mathrm{peringkat}_r(d)}$$

dengan $R$ himpunan daftar peringkat yang digabung dan $k$ tetapan peredam. Metode ini
berasal dari Cormack dkk. (2009), dan dipakai Xiong dkk. \autocite{xiong2024} untuk
menggabungkan hasil beberapa penelusur pada tolok ukur RAG medis.

Nilai $k = 60$ dipertahankan pada nilai bakunya dan **tidak disetel** pada penelitian
ini, justru agar tidak menambah parameter bebas yang menuntut protokol tersendiri.

---

## 3. Bukti dari domain medis

Dua rujukan berikut menempatkan penelusuran hibrida sebagai **praktik baku pada RAG
medis**, bukan penyetelan khusus penelitian ini.

**Xiong dkk. \autocite{xiong2024}**, tolok ukur MedRAG, membandingkan satu penelusur
leksikal (BM25) dengan tiga penelusur padat (Contriever, SPECTER, MedCPT) pada korpus
medis. Dua temuannya relevan langsung:

1. BM25 dinilai **penelusur yang kuat** pada domain medis, sejalan dengan evaluasi lain.
2. Konfigurasi yang mereka anjurkan adalah **RRF-2**, yakni fusi BM25 dengan MedCPT,
   karena keduanya merupakan penelusur leksikal dan padat terbaik pada percobaan mereka.

**Lee dkk. \autocite{lee2024}** membangun sistem RAG atas pedoman diabetes — domain
yang sama dengan penelitian ini — dengan penelusuran ganda yang memadukan pencarian
makna dan pencarian kata kunci. Mereka juga menunjukkan bahwa **pilihan pemenggal kata
berpengaruh**, dengan memakai pemenggal khusus bahasa Korea bagi teks Korea.

---

## 4. Mengapa hal ini menjawab persoalan spesifik penelitian ini

Kesimpulan di atas berlaku umum. Yang membuatnya **wajib** di sini adalah keadaan yang
khas pada penelitian ini:

Model penelusuran padat yang dipakai, `all-MiniLM-L6-v2`, adalah model **berbahasa
Inggris**, sedangkan korpus pedomannya **berbahasa Indonesia**. Jalan yang paling
langsung — mengganti ke model multibahasa — sudah ditempuh dan **ditolak berdasarkan
pengukuran**: model multibahasa merusak kemampuan sistem membedakan kelas kondisi,
dan kelas `normal` hampir hilang. Pencocokan leksikal menawarkan jalan lain, sebab ia
**tidak bergantung pada ruang makna sama sekali** sehingga bekerja tepat pada titik
model bahasa Inggris paling lemah.

Perlu dicatat pula sebagai keterbatasan yang diketahui: pemenggalan kata yang dipakai
belum memenggal imbuhan bahasa Indonesia, sehingga *pemberian*, *diberikan*, dan
*berikan* dihitung sebagai tiga istilah berbeda. Lee dkk. menunjukkan pemenggal khusus
bahasa berpengaruh pada pedoman berbahasa Korea; padanan Indonesianya **belum diuji**
pada penelitian ini.

---

## 5. Status bacaan tiap rujukan — wajib dipatuhi saat menulis

| Rujukan | Status | Boleh dipakai untuk |
|---|---|---|
| Gao dkk. \autocite{gao2023} | **dibaca**, Bagian V.A.1 hal. 8 | strategi pemecahan dokumen |
| Xiong dkk. \autocite{xiong2024} | **dibaca**, bagian penelusur dan RRF | BM25 kuat di medis; anjuran RRF-2 |
| Lee dkk. \autocite{lee2024} | **dibaca**, bagian pendahuluan dan metode | pertukaran padat lawan jarang; pemenggal khusus bahasa |
| Manning dkk. \autocite{manning2008} | **dibaca sebagian** — hal. 217 dan pengantar Bab 11 | BM25 sebagai skema pembobotan mapan |
| Carbonell & Goldstein \autocite{carbonell1998} | sudah dipakai naskah | MMR |
| **Cormack dkk. (2009)** | **BELUM DIBACA** | hanya boleh disebut sebagai **asal** RRF, lewat Xiong dkk. Jangan mengutip isinya. |
| Robertson & Zaragoza | **BELUM DIBACA**, tidak ada di koleksi | jangan dipakai; BM25 cukup lewat Manning dkk. |

**Cara menyebut Cormack yang jujur**, bila diperlukan:
> "Reciprocal Rank Fusion \autocite[sebagaimana dirujuk][]{xiong2024} yang berasal dari
> Cormack dkk. (2009)"

Atau sederhananya: sebut RRF, sitasi `xiong2024` sebagai pemakaiannya pada domain medis.

---

## 6. Entri bib yang perlu ditambahkan

Hanya satu, dan **hanya bila** Cormack hendak disitasi langsung. Sitasinya sudah
diverifikasi dari daftar pustaka Xiong dkk.:

```bibtex
@inproceedings{cormack2009,
  author    = {Cormack, Gordon V. and Clarke, Charles L. A. and Buettcher, Stefan},
  title     = {Reciprocal Rank Fusion Outperforms Condorcet and Individual
               Rank Learning Methods},
  booktitle = {Proceedings of the 32nd International ACM SIGIR Conference on
               Research and Development in Information Retrieval},
  pages     = {758--759},
  year      = {2009},
}
```

**Saran: jangan tambahkan dulu.** Menyitasi paper yang belum dibaca melanggar aturan
proyek sendiri, dan `xiong2024` sudah memadai untuk membenarkan pemakaian RRF.
Tambahkan hanya setelah papernya benar-benar dibaca.

---

## 7. Tempat yang tepat di dalam naskah

- **Bab II**: subbab baru setelah pembahasan RAG, sejajar dengan subbab MMR yang sudah
  ada. Isinya Bagian 1–3 dokumen ini.
- **Bab IV**: satu alinea pada perancangan penelusuran, merujuk Bab II, memuat Bagian 4.
- **Bab VI**: hasil T13 dan T14 sebagai studi ablasi penelusuran.
- **Bab keterbatasan**: alinea terakhir Bagian 4 tentang pemenggalan imbuhan.
