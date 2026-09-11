# System architecture

```
User / Officer
      |
      v
Streamlit dashboard (login, KPIs, cases, predict, GIS, alerts, assistant)
      |
      v
FastAPI backend (JWT, validation, case CRUD)
      |
      +--> SQLite (users, cases, predictions, actions, remarks, alerts)
      |
      v
ML prediction service
      |
      +--> Classification model (delay probability, risk level)
      +--> Regression model (expected delay days)
      |
      v
SHAP explainability (top delay reasons)
      |
      v
Rule-based recommendation engine + early-warning priority score
      |
      v
Dashboard / GIS / alert feed
```

## Design notes

- The UI never trains models. Training is an offline pipeline (`src/run_pipeline.py`).
- Bulk seeding uses model scores + heuristic main-reason for speed. Opening a case runs SHAP.
- Map coordinates are jittered around district centres for the prototype only.
- Outputs are advisory. Officers still follow statutory land-acquisition process.
