# results/__Hasil\_Akhir__

Seluruh hasil akhir yang dipakai pada laporan tugas akhir, terpisah
dari jejak eksperimen di `results/`. Isi direktori ini **salinan**;
sumber kebenarannya tetap berkas asal pada kolom Sumber.

Direktori ini dibangkitkan skrip. Jangan menyunting isinya dengan
tangan, karena perubahan akan tertimpa. Untuk memperbarui:

```bash
conda run -n diabetes-ta python scripts/kumpulkan_hasil_akhir.py
```

Memeriksa apakah salinan masih sama dengan sumbernya:

```bash
conda run -n diabetes-ta python scripts/kumpulkan_hasil_akhir.py --periksa
```

## Isi

| Berkas | Menopang | Sumber | SHA-256 |
| --- | --- | --- | --- |
| `T1_prediksi/gbm_produksi_h30m.json` | Kinerja model produksi GBM, horizon 30 menit, beserta pembanding RF dan LSTM | `results/eval_prediksi/gradient_boosting_h6.json` | `b8cfe5e9c427e083` |
| `T1_prediksi/gbm_produksi_h60m.json` | Kinerja model produksi GBM, horizon 60 menit, beserta pembanding RF dan LSTM | `results/eval_prediksi/gradient_boosting_h12.json` | `75326e557db4e7aa` |
| `T1_prediksi/rf_vs_lstm_h30m.json` | Perbandingan RF terhadap LSTM, horizon 30 menit | `results/eval_prediksi/crossval_rf_vs_lstm_h6.json` | `86caa55feffb4eaa` |
| `T1_prediksi/rf_vs_lstm_h60m.json` | Perbandingan RF terhadap LSTM, horizon 60 menit | `results/eval_prediksi/crossval_rf_vs_lstm_h12.json` | `7e374b5b0e921f9a` |
| `T1_prediksi/crossfold_fingerstick.json` | Kinerja kanal finger-stick | `results/eval_prediksi/crossval_smbg.json` | `971f679f7e1d3c7f` |
| `T1_prediksi/konformal_h30m.json` | Kalibrasi ketidakpastian, horizon 30 menit | `results/eval_prediksi/conformal_h6.json` | `19bb6d289cb3fea1` |
| `T1_prediksi/konformal_h60m.json` | Kalibrasi ketidakpastian, horizon 60 menit | `results/eval_prediksi/conformal_h12.json` | `6bb1c7ad720eaca2` |
| `T1_prediksi/cakupan_konformal.json` | Gambar VI.5 dan VI.6 | `results/eval_prediksi/cakupan_conformal.json` | `556978fd7e95c59e` |
| `T1_prediksi/hipoglikemia_h30m.json` | Deteksi hipoglikemia, horizon 30 menit | `results/eval_prediksi/hipoglikemia_h6.json` | `93cc19de2c77e842` |
| `T1_prediksi/hipoglikemia_h60m.json` | Deteksi hipoglikemia, horizon 60 menit | `results/eval_prediksi/hipoglikemia_h12.json` | `a54482634a737657` |
| `T1_prediksi/pengklasifikasi_kondisi.json` | Pengklasifikasi kondisi terprediksi | `results/eval_prediksi/condition_classifier.json` | `3079f0a686a66763` |
| `T1_prediksi/clarke_grid_h30m.png` | Gambar VI.1, panel horizon 30 menit | `results/eval_prediksi/h30m/clarke_grid_gbm.png` | `0b1b08efd1764467` |
| `T1_prediksi/clarke_grid_h60m.png` | Gambar VI.2, panel horizon 60 menit | `results/eval_prediksi/h60m/clarke_grid_gbm.png` | `22f79d33cbe73421` |
| `T2_retrieval/crossfold.json` | Tabel VI.10, Lampiran B.1, dan angka MRR pada abstrak | `results/retrieval_realcases_kb12/crossfold.json` | `20b0102d9b74426e` |
| `T2_retrieval/summary.json` | Ringkasan penelusuran | `results/retrieval_realcases_kb12/summary.json` | `4f41378d91c6e309` |
| `T2_retrieval/per_kasus_divergen.csv` | Rincian per kasus divergen | `results/retrieval_realcases_kb12/per_case_divergen.csv` | `f4ca55ecd8dfaa9d` |
| `T2_retrieval/per_kasus_natural.csv` | Rincian per kasus natural | `results/retrieval_realcases_kb12/per_case_natural.csv` | `3ff37b2ec42615e7` |
| `T3_generation/generation_safety.json` | Gambar VI.9 | `results/eval_prediksi/generation_safety.json` | `c9fdf28a17c53f85` |
| `T3_generation/kestabilan_ragas.json` | Gambar VI.10 | `results/ragas/kestabilan.json` | `3ee8ddc7f57a468f` |
| `operasional/latensi_ujung_ke_ujung.json` | Gambar VI.11, waktu tanggap pipeline | `results/benchmark/latency_endtoend_gemini.json` | `06f410f743204b69` |
| `operasional/keterterapan.json` | Angka komputasi lokal | `results/benchmark/deployability.json` | `ecc830e6c72d939c` |
| `evaluasi_ahli/statistik_n12.json` | Tabel VI.16 sampai VI.19 | `results/hasil_form/terbaru/statistik.json` | `7d1dcd99ae07ed94` |
| `evaluasi_ahli/statistik_n12_skema_lama.json` | Gambar VI.14 | `results/hasil_form/terbaru/statistik_kompat.json` | `e8712734f092b6c4` |
| `korpus/manifest.csv` | Tabel D.1, korpus pedoman klinis | `data/knowledge_base/manifest.csv` | `82f706c4c4db3a84` |
| `ringkasan_untuk_bab6.json` | Ringkasan lintas-tujuan yang dipakai beberapa gambar | `results/ringkasan_untuk_bab6.json` | `f3d237bbe338b827` |

## Yang sengaja tidak ada di sini

- Varian `retrieval_realcases_*` selain `kb12`. Angka laporan berasal
  dari `kb12` lengan BM25; varian lain adalah eksperimen.
- Seluruh `baseline_ablation_*`, `tuning_log.json`, dan berkas
  bertanda `_arsip`, `_PARSIAL`, `_TIDAK_LAYAK`, `_kuota_habis`.
- Berkas mentah kuesioner. CSV dan PDF sumbernya memuat nama lengkap
  dan alamat surel responden, jadi tidak pernah dilacak git. Hanya
  statistik agregatnya yang disertakan.
