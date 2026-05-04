# Architecture Overview

## System Goal

This project provides a lightweight Diabetes Digital Twin platform for:

1. Daily structured input from manual logbook or prepared datasets.
2. Short-term glucose prediction using a baseline machine learning model.
3. What-if scenario simulation through a patient twin state model.
4. Clinical explanation with Retrieval-Augmented Generation (RAG) fallback logic.

## High-Level Components

The system is organized into five logical layers.

### 1. Data Layer

- Raw data sources:
  - `data/raw/training_data_complete.csv`
  - `data/raw/ohio_t1dm_merged.csv`
  - `data/raw/manual_logbook.csv`
- Processed state storage:
  - `data/processed/patient_states.json`
- Knowledge base storage:
  - `data/knowledge_base/*.txt`
  - `data/knowledge_base/manual_kb.json` (fallback)

### 2. Core Domain Layer

- Data pipeline:
  - `src/data/loader.py`
  - `src/data/generator.py`
  - `src/data/preprocessor.py`
  - `src/data/ohio_parser.py`
- Prediction model:
  - `src/models/rf_model.py` (active baseline)
- Digital twin:
  - `src/digital_twin/patien_twin.py`
  - `src/digital_twin/state_manager.py`
  - `src/digital_twin/simulator.py`
- RAG:
  - `src/rag/knowledge_base.py`
  - `src/rag/pipeline.py`
  - `src/rag/generator.py`
  - `src/rag/retriever.py` (optional semantic retrieval)

### 3. Application Layer (Streamlit)

- Landing page:
  - `app/streamlit_app.py`
- Main pages:
  - `app/pages/1_Input_Logbook.py`
  - `app/pages/2_Prediction.py`
  - `app/pages/3_WhatIf_Simulator.py`

### 4. Utilities Layer

- Metrics and evaluation:
  - `src/utils/metrics.py`
- Logging:
  - `src/utils/logger.py`
- Plotting helpers:
  - `src/utils/visualization.py`

### 5. Test Layer

- Unit and integration tests under `tests/`.

## Runtime Flow

## Flow A: Prediction

1. User selects source and patient in the Prediction page.
2. `DiabetesDataLoader` resolves preferred source with fallback policy.
3. A sequence window is constructed and optionally overridden at the latest row.
4. Random Forest inference bundle performs one-step glucose prediction.
5. Risk level is computed (`<70`, `70-180`, `>180`).
6. `RAGPipeline.answer()` generates explanation using retrieval and generator fallback.

## Flow B: What-If Simulation

1. Latest patient row is transformed into `PatientDigitalTwin` state.
2. User defines scenario deltas (carbs, insulin, activity, stress, horizon).
3. `simulate_scenario()` computes predicted glucose impact.
4. Optional quick simulations run via `WhatIfSimulator` helper methods.
5. RAG explanation is generated from the simulated prediction output.

## Flow C: State Persistence

1. `DigitalTwinStateManager` creates or updates patient state records.
2. Events are appended for traceability.
3. State and events are serialized to JSON for recovery.

## Fallback and Reliability Strategy

- Data fallback: primary source to fallback source in loader policy.
- Retrieval fallback: semantic retriever to keyword overlap retriever.
- Generation fallback: external provider to template-based response.
- Visualization compatibility: Matplotlib is used in UI pages to avoid chart adapter incompatibility.

## Current Known Scope

- Active prediction engine is Random Forest baseline only.
- Clinical validation is currently software-level and dataset-level, not live patient deployment.

## Design Principles

1. Keep the platform runnable on standard laptop hardware.
2. Prefer graceful fallback over hard failure in optional modules.
3. Prioritize explainable outputs for decision support.
4. Separate domain logic (`src/`) from interface (`app/`) for maintainability.
