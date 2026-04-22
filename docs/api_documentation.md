# API Documentation

Dokumen ini menjelaskan interface minimal dari komponen utama yang sudah diimplementasikan: Digital Twin state manager dan RAG pipeline.

## 1. Digital Twin State Manager

Class: `src.digital_twin.DigitalTwinStateManager`

Tujuan: menyimpan state pasien secara sederhana, persistable, dan dapat dipulihkan kembali setelah aplikasi restart.

### State Utama

State dasar yang dikelola:

- `current_glucose`
- `insulin_on_board`
- `carbs_on_board`
- `last_meal_time`
- `last_insulin_time`
- `activity_level`
- `stress_level`
- `timestamp`

### Method Utama

#### `create_state(patient_id, initial_state=None)`

Membuat state awal untuk pasien.

Contoh:

```python
from src.digital_twin import DigitalTwinStateManager

manager = DigitalTwinStateManager()
state = manager.create_state("P001", {"current_glucose": 130.0, "stress_level": 6})
```

#### `get_state(patient_id)`

Mengambil state pasien saat ini.

#### `update_state(patient_id, updates)`

Memperbarui field yang valid pada state pasien.

Contoh:

```python
manager.update_state("P001", {"carbs_on_board": 20.0, "activity_level": 15})
```

#### `append_event(patient_id, event_type, payload)`

Menambahkan event log untuk audit sederhana.

#### `save()`

Menyimpan seluruh state dan event ke file JSON.

#### `load()`

Memuat kembali state dari file JSON.

### Format Penyimpanan

File default: `data/processed/patient_states.json`

Struktur umum:

```json
{
  "P001": {
    "patient_id": "P001",
    "state": {
      "current_glucose": 130.0,
      "insulin_on_board": 0.0,
      "carbs_on_board": 20.0,
      "activity_level": 15,
      "stress_level": 6,
      "timestamp": "2026-04-22T10:00:00"
    },
    "events": []
  }
}
```

## 2. RAG Pipeline

Class: `src.rag.RAGPipeline`

Tujuan: menghasilkan explanation klinis sederhana dari prediksi glukosa dengan retrieval dari knowledge base medis.

### Method Utama

#### `build()`

Menyiapkan knowledge base, retriever, dan generator.

#### `answer(patient_state, prediction, query=None, top_k=None)`

Menjalankan pipeline lengkap dan mengembalikan hasil penjelasan.

Input minimum:

- `patient_state`: dictionary state pasien
- `prediction`: nilai prediksi glukosa

Output:

- `query`
- `risk_level`
- `prediction`
- `retrieved_docs`
- `explanation`

Contoh:

```python
from src.rag import RAGPipeline

pipeline = RAGPipeline(kb_dir="data/knowledge_base", llm_provider="template")
result = pipeline.answer(
		patient_state={
				"current_glucose": 205.0,
				"stress_level": 8,
				"activity_level": 10,
				"carbs_on_board": 25.0,
				"insulin_on_board": 0.0,
		},
		prediction=210.0,
)
```

### Fallback Mode

Pipeline dibuat tetap aman bila dependency eksternal tidak tersedia:

- retriever semantic dapat diganti ke keyword fallback,
- generator menggunakan provider `template` bila LLM eksternal tidak tersedia,
- tujuan utama saat ini adalah memastikan sistem tetap berjalan stabil di laptop biasa.

## 3. Integrasi di UI

Halaman yang sudah memakai komponen ini:

- [app/pages/2_Prediction.py](../app/pages/2_Prediction.py)
- [app/pages/3_WhatIf_Simulator.py](../app/pages/3_WhatIf_Simulator.py)

Alur sederhana:

1. Ambil data pasien.
2. Jalankan prediksi atau simulasi.
3. Kirim state pasien + hasil prediksi ke `RAGPipeline.answer()`.
4. Tampilkan explanation dan konteks medis yang di-retrieve.

## 4. Catatan Implementasi

Implementasi saat ini sengaja dibuat sederhana dulu:

- fokus pada fungsi utama,
- menghindari dependency yang berat,
- menjaga aplikasi tetap bisa dijalankan end-to-end,
- fancy layer seperti semantic retriever penuh atau LLM cloud dapat ditambahkan nanti jika diperlukan.
