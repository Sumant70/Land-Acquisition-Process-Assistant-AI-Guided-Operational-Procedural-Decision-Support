# Process flowchart

```
Start
  |
  v
Officer logs in (JWT)
  |
  v
Dashboard loads KPIs + risk mix + alerts
  |
  v
Officer filters / searches cases
  |
  v
Open case
  |
  +--> Load attributes from SQLite
  +--> Classification: P(delay), risk band
  +--> Regression: expected delay days
  +--> SHAP: top contributing factors
  +--> Rules: recommended actions
  +--> Priority score: Immediate / Soon / Monitor / Normal
  |
  v
Officer may update status + remarks
  |
  v
New prediction snapshot stored (risk history)
  |
  v
GIS: synthetic map markers by risk colour
  |
  v
New case form -> /cases or /predict -> dashboard totals update
  |
  v
End (advisory output only)
```
