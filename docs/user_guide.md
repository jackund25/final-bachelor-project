# User Guide

## 1. Prerequisites

- Windows, macOS, or Linux
- Python 3.11
- Installed dependencies from `requirements.txt`

## 2. Environment Setup

From project root:

```powershell
python -m venv ta
.\ta\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation scripts:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\ta\Scripts\Activate.ps1
```

## 3. Run the App

```powershell
streamlit run app/streamlit_app.py
```

Open the local URL shown by Streamlit in browser.

## 4. Main Pages and Usage

## 4.1 Input Logbook

Purpose:

- Record structured daily entries manually.

How to use:

1. Fill patient ID, glucose, carbs, insulin, activity, stress, and optional notes.
2. Click `Save Logbook Entry`.
3. Confirm status panel and recent entries table are updated.

Stored file:

- `data/raw/manual_logbook.csv`

## 4.2 Prediction

Purpose:

- Predict next glucose value from selected patient window.

How to use:

1. Choose data source mode.
2. Select patient ID.
3. Optionally override latest row values.
4. Click `Predict Next Glucose`.
5. Review predicted value, risk level, trajectory plot, and explanation.

Notes:

- If manual logbook has limited rows, system uses fallback source policy.
- If only model file exists without bundle, a warning is shown.

## 4.3 What-If Simulator

Purpose:

- Simulate intervention scenarios before real action.

How to use:

1. Select patient ID.
2. Set scenario deltas (carbs, insulin, activity, stress, horizon).
3. Click `Run What-If Simulation`.
4. Review current vs predicted glucose and risk assessment.
5. Check RAG-based explanation and decision support recommendations.

## 5. Training Baseline Model

Run baseline training:

```powershell
python -m src.models.rf_model --config config.yaml --data_source auto
```

Output artifacts:

- `models/rf_baseline.pkl`
- `models/rf_inference_bundle.pkl`
- `models/rf_baseline_metrics.json`

## 6. Testing

Run full test suite:

```powershell
pytest tests -v
```

Run only core Step 1 tests:

```powershell
pytest tests/test_kb.py tests/test_logger.py tests/test_metrics.py -v
```

## 7. Troubleshooting

`Model belum tersedia` in Prediction page:

1. Train the model using the command in section 5.

PowerShell activation blocked:

1. Use process-scoped execution policy bypass shown in section 2.

Manual logbook not loaded in Prediction mode:

1. Ensure at least one entry is saved from Input Logbook page.

Dependency mismatch issues:

1. Use Python 3.11 and reinstall dependencies in a clean venv.
