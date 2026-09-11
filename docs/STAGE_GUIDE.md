# Stage-by-stage build and test guide

Every metric in this project is computed from a held-out test set after you run training. Do not type fake accuracy into a PPT.

Dataset banner to show in every demo slide:

> **Synthetic Dataset for Prototype and Model Development**  
> These records are **not** real government cases.

---

## Stage 1 — Dataset generation

**File:** `src/generate_dataset.py`

```powershell
python -m src.generate_dataset --n 5500
```

**Expected output**

- File `data/raw/synthetic_land_acquisition_cases.csv`
- More than 5,000 rows
- Printed delay class share and mean/median delay days
- Disclaimer printed in the console

**How to test**

```powershell
python -c "import pandas as pd; df=pd.read_csv('data/raw/synthetic_land_acquisition_cases.csv'); print(len(df), df.columns.tolist()[:8], df['delayed'].mean())"
```

**Common errors**

- Folder missing: the script creates `data/raw/` automatically.
- Delay labels look random: they are generated from disputes, pending survey/compensation/approval, objections, documents, etc., plus small noise.

---

## Stage 2 — Cleaning

**File:** `src/data_cleaning.py`

```powershell
python -m src.data_cleaning
```

**Expected output**

- `data/processed/clean_land_acquisition_cases.csv`
- Duplicate count printed (about 12 injected duplicates)
- Remaining missing cells = 0

**How to test:** open the clean CSV; `document_completeness_pct` should have no blanks.

---

## Stage 3 — EDA

**File:** `src/eda.py`

```powershell
python -m src.eda
```

**Expected output:** plots in `data/processed/eda/`

- `delay_days_histogram.png`
- `delayed_count.png`
- `delay_rate_by_project_type.png`
- `correlation_heatmap.png`
- `summary.csv`

**How to test:** delay rate should be higher when `legal_case=1` or `compensation_paid=0` (printed in console).

---

## Stage 4 — Feature engineering

**File:** `src/feature_engineering.py`

```powershell
python -m src.feature_engineering
```

**Expected output:** `data/processed/featured_land_acquisition_cases.csv`  
Console must say leakage is **None**.

Engineered fields: `pending_process_count`, `dispute_and_legal`, `docs_incomplete_flag`, `high_objection_flag`, `objections_per_landowner`, `complexity_index`.

**Constraint:** `delayed` and `expected_delay_days` are **never** model inputs.

---

## Stage 5 — Training

**File:** `src/train_model.py`

```powershell
python -m src.train_model
```

**Expected output**

- `models/best_classifier.joblib`
- `models/best_regressor.joblib`
- `models/preprocessor.joblib`
- `models/metrics.json`

Classifier chosen by highest **ROC-AUC**. Regressor chosen by lowest **RMSE**.

---

## Stage 6 — Prediction (library)

**File:** `src/predict.py`

Used by the API. After training, prediction is tested when you call `POST /predict`.

---

## Stage 7 — Explainability

**File:** `src/explain.py`

SHAP TreeExplainer on the trained classifier. Shown on case detail and Predict page as:

1. Ownership dispute — High impact  
2. …

---

## Stage 8 — Recommendations

**File:** `src/recommendations.py`

Rule engine + priority score + alert rules.

---

## Stage 9 — FastAPI

**Files:** `backend/main.py`, `database.py`, `models.py`, `schemas.py`

```powershell
python -m uvicorn backend.main:app --reload
```

Test: open `/docs`, authorize, call `GET /dashboard/stats`.

---

## Stage 10–12 — Dashboard, GIS, database, alerts

**File:** `dashboard/app.py`  
Database file: `land_acquisition.db` (created on API startup)

```powershell
streamlit run dashboard/app.py
```

GIS tiles: OpenStreetMap via Folium. Coordinates are **synthetic**.
