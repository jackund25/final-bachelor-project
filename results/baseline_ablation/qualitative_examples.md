# Contoh Kualitatif — Ablation Standard vs Prediction-Conditioned RAG


## Kasus D2 (Olahraga tanpa penyesuaian insulin) — mode: standard
- Current: 98.0 mg/dL | Predicted: 64.0 mg/dL | Ground truth: **Hipoglikemia**
- Query: _Kadar glukosa darah 98 mg/dL. Gula darah dalam rentang normal/target. Target kontrol glikemik dan pemantauan rutin diabetes._
- Top-3 dokumen ter-retrieve: Target Kontrol Glikemik (0.75), Hipoglikemia (0.68), Hiperglikemia (0.67)
- Hasil: ✓ menemukan dokumen antisipatif.

## Kasus D2 (Olahraga tanpa penyesuaian insulin) — mode: prediction_conditioned
- Current: 98.0 mg/dL | Predicted: 64.0 mg/dL | Ground truth: **Hipoglikemia**
- Query: _Kadar glukosa darah 64 mg/dL (prediksi 60 menit ke depan). Hipoglikemia, gula darah rendah di bawah 70 mg/dL. Penyebab, gejala, dan penanganan segera (aturan 15-15)._
- Top-3 dokumen ter-retrieve: Target Kontrol Glikemik (0.69), Hiperglikemia (0.68), Hipoglikemia (0.67)
- Hasil: ✓ menemukan dokumen antisipatif.

## Kasus D4 (Makan tinggi karbohidrat) — mode: standard
- Current: 150.0 mg/dL | Predicted: 214.0 mg/dL | Ground truth: **Hiperglikemia**
- Query: _Kadar glukosa darah 150 mg/dL. Gula darah dalam rentang normal/target. Target kontrol glikemik dan pemantauan rutin diabetes._
- Top-3 dokumen ter-retrieve: Target Kontrol Glikemik (0.78), Hipoglikemia (0.71), Hiperglikemia (0.70)
- Hasil: ✓ menemukan dokumen antisipatif.

## Kasus D4 (Makan tinggi karbohidrat) — mode: prediction_conditioned
- Current: 150.0 mg/dL | Predicted: 214.0 mg/dL | Ground truth: **Hiperglikemia**
- Query: _Kadar glukosa darah 214 mg/dL (prediksi 60 menit ke depan). Hiperglikemia, gula darah tinggi di atas 180 mg/dL. Penyebab, gejala, dan penanganan._
- Top-3 dokumen ter-retrieve: Hiperglikemia (0.70), Target Kontrol Glikemik (0.66), Hipoglikemia (0.62)
- Hasil: ✓ menemukan dokumen antisipatif.