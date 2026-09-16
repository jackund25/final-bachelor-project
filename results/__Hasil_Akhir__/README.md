# results/__Hasil\_Akhir__

Seluruh hasil akhir yang dipakai pada laporan tugas akhir, terpisah
dari jejak eksperimen di `results/`. Isi direktori ini **salinan**;
sumber kebenarannya tetap berkas asal pada kolom Sumber.

Kriteria masuk: angka utama berkas muncul pada laporan yang
dikumpulkan. Berkas hasil yang benar tetapi tidak dikutip laporan
sengaja tidak disertakan, agar direktori ini tidak memuat angka yang
bertentangan dengan laporan.

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
| `T1_prediksi/holdout_semua_horizon.csv` | Tabel VI.3, kinerja hold-out ketiga model pada kedua horizon | `results/eval_prediksi/summary_all_horizons.csv` | `a54c7d17b39e73cf` |
| `T1_prediksi/crossfold_h30m.json` | Tabel VI.4, validasi silang lintas-pasien horizon +30 menit | `results/eval_prediksi/gradient_boosting_h6.json` | `b8cfe5e9c427e083` |
| `T1_prediksi/holdout_fingerstick.json` | Tabel VI.5, baris hold-out skenario finger-stick | `results/eval_prediksi/smbg_sweep.json` | `753a2879ab55fc5e` |
| `T1_prediksi/crossfold_fingerstick.json` | Tabel VI.5, baris cross-fold skenario finger-stick | `results/eval_prediksi/crossval_smbg.json` | `971f679f7e1d3c7f` |
| `T1_prediksi/pengklasifikasi_kondisi.json` | Tabel VI.8, regresi-lalu-ambang vs pengklasifikasi kondisi | `results/eval_prediksi/condition_classifier.json` | `3079f0a686a66763` |
| `T1_prediksi/cakupan_konformal.json` | Gambar VI.5 dan Tabel VI.6 kolom cakupan: 12 putaran dengan pasien pengukur terpisah | `results/eval_prediksi/cakupan_conformal.json` | `556978fd7e95c59e` |
| `T1_prediksi/konformal_h30m.json` | Tabel VI.6 dan Gambar VI.6: faktor q dan lebar interval, horizon +30 | `results/eval_prediksi/conformal_h6.json` | `19bb6d289cb3fea1` |
| `T1_prediksi/konformal_h60m.json` | Tabel VI.6 dan Gambar VI.6: faktor q dan lebar interval, horizon +60 | `results/eval_prediksi/conformal_h12.json` | `6bb1c7ad720eaca2` |
| `T1_prediksi/sapuan_ambang_hipoglikemia.json` | Sapuan ambang peringatan 70, 85, dan 90 mg/dL pada Bab VI | `results/eval_prediksi/hypo_improve.json` | `2a1f3b39ccfb0c32` |
| `T1_prediksi/clarke_grid_h30m.png` | Gambar VI.1a, Clarke Error Grid horizon +30 menit | `results/eval_prediksi/h30m/clarke_grid_gbm.png` | `0b1b08efd1764467` |
| `T1_prediksi/clarke_grid_h60m.png` | Gambar VI.1b, Clarke Error Grid horizon +60 menit | `results/eval_prediksi/h60m/clarke_grid_gbm.png` | `22f79d33cbe73421` |
| `T2_retrieval/crossfold.json` | Tabel VI.10, Lampiran B.1, abstrak, dan ablasi horizon conditioning | `results/retrieval_realcases_kb12/crossfold.json` | `20b0102d9b74426e` |
| `T2_retrieval/summary.json` | Kasus nyata dua pasien hold-out (120 kasus) dan kontrol lengan acak | `results/retrieval_realcases_kb12/summary.json` | `4f41378d91c6e309` |
| `T2_retrieval/per_kasus_divergen.csv` | Rincian per kasus divergen di balik Tabel VI.10 | `results/retrieval_realcases_kb12/per_case_divergen.csv` | `f4ca55ecd8dfaa9d` |
| `T2_retrieval/per_kasus_natural.csv` | Rincian per kasus natural di balik Tabel VI.10 | `results/retrieval_realcases_kb12/per_case_natural.csv` | `3ff37b2ec42615e7` |
| `T2_retrieval/ablasi_reranker.json` | Ablasi reranker pada Bab VI | `results/reranker_ablation.json` | `28ddd9cd60682ce9` |
| `T3_generation/generation_novelty.json` | sim_ref dan action coverage pada enam kasus, RAG standar vs terkondisi | `results/eval_prediksi/generation_novelty.json` | `b3520242bfa621b3` |
| `T3_generation/kestabilan_ragas.json` | Kestabilan penilai RAGAS antar-run pada Bab VI | `results/ragas/kestabilan.json` | `3ee8ddc7f57a468f` |
| `operasional/latensi_ujung_ke_ujung.json` | Waktu tanggap ujung-ke-ujung dan komputasi lokal pada Bab VI | `results/benchmark/latency_endtoend_gemini.json` | `06f410f743204b69` |
| `evaluasi_ahli/statistik_n12.json` | Tabel VI.16 sampai VI.19 | `results/hasil_form/terbaru/statistik.json` | `7d1dcd99ae07ed94` |
| `evaluasi_ahli/statistik_n12_skema_lama.json` | Gambar VI.14 | `results/hasil_form/terbaru/statistik_kompat.json` | `e8712734f092b6c4` |
| `korpus/manifest.csv` | Tabel D.1, korpus pedoman klinis | `data/knowledge_base/manifest.csv` | `82f706c4c4db3a84` |

## Yang sengaja tidak ada di sini

- Jejak eksperimen yang tidak dipakai laporan telah dipindahkan ke
  `arsip/code_eksperimen/results/` (di luar git): varian
  `retrieval_realcases_*` selain `kb12`, `baseline_ablation_*`, berkas
  bertanda `_arsip`/`_PARSIAL`/`_kuota_habis`, `deployability.json`, dan
  `ringkasan_untuk_bab6.json` (snapshot era Random Forest yang pernah
  bocor ke teks laporan).
- Hasil yang benar tetapi tidak dikutip laporan: validasi silang
  horizon +60 menit, `tuning_log.json`, pemeriksaan keterlacakan angka
  (`generation_safety.json`), dan `embedding_gemini_TIDAK_LAYAK.json`.
- Berkas mentah kuesioner. CSV dan PDF sumbernya memuat nama lengkap
  dan alamat surel responden, jadi tidak pernah dilacak git. Hanya
  statistik agregatnya yang disertakan.
