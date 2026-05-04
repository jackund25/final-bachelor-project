# 🩺 Diabetes Digital Twin with RAG Framework

> Final Year Project - Institut Teknologi Bandung
>
> **Author:** Daffari Adiyatma (18222003)
>
> **Supervisors:**
>
> - Prof. Dr. Ir. Suhono Harso Supangkat, M.Eng.
> - Ir. Devi Willieam Anggara S.T., M.Phil., Ph.D

## 📋 Overview

System prediksi kadar glukosa darah berbasis Digital Twin dan Retrieval-Augmented Generation (RAG) untuk pasien Diabetes Tipe 2.

### Key Features

- ✅ **Data Generation**: Synthetic patient data via Simglucose
- ✅ **Digital Twin**: Virtual representation of patient metabolic state
- ✅ **RF Baseline Prediction**: Time-series window forecasting (active)
- ✅ **RAG Explanation**: Clinical reasoning with medical guidelines
- ✅ **What-If Analysis**: Scenario simulation for decision support

## 🚀 Quick Start

### 1. Installation

\`\`\`bash

# Clone repository

git clone <your-repo-url>
cd diabetes-digital-twin

# Create virtual environment

python -m venv ta

# On Windows PowerShell

.\ta\Scripts\Activate.ps1

# On Windows CMD

ta\Scripts\activate.bat

# Install dependencies

pip install -r requirements.txt
\`\`\`

### 2. Generate Data (Step A)

\`\`\`bash
python -m src.data.generator
\`\`\`

### 3. Initialize Digital Twin (Step C)

\`\`\`python
from src.digital_twin import PatientDigitalTwin

twin = PatientDigitalTwin(patient_id="P001")
twin.update_state({'glucose': 120, 'stress': 5})
\`\`\`

### 4. Setup RAG (Step D)

\`\`\`bash

# Download PERKENI PDF to data/knowledge_base/

python -m src.rag.knowledge_base --extract
\`\`\`

### 5. Train Model (Step B)

\`\`\`bash
python -m src.models.rf_model --config config.yaml --data_source auto
\`\`\`

### 6. Run Web App

\`\`\`bash
streamlit run app/streamlit_app.py
\`\`\`

## 📊 Project Timeline

- [x] **Week 1-2 (Jan):** Data generation
- [x] **Week 3-4 (Jan):** Digital Twin implementation
- [x] **Week 5-6 (Feb):** RAG system
- [x] **Week 7-8 (Feb):** RF baseline training
- [x] **Week 9-12 (Mar):** Integration & testing
- [x] **Week 13-16 (Apr):** Initial evaluation & docs
- [ ] **Week 17-18 (May):** Final report & defense

## 📚 Documentation

See `docs/` folder for detailed documentation.

## 🧪 Testing

\`\`\`bash
pytest tests/ -v --cov=src
\`\`\`

## 📝 License

Academic use only - ITB Final Year Project

## 🙏 Acknowledgments

- UVA/Padova T1DM Simulator team
- PERKENI for diabetes guidelines
- Anthropic Claude for development assistance
