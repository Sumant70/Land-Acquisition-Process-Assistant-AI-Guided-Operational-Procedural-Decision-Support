# SIH presentation content (8–10 slides)

Use this text in PPT. Keep the synthetic-data disclaimer on every data slide.

## Slide 1 — Title

**AI-Powered Land Acquisition Delay Prediction & Decision Support System**  
Smart India Hackathon  
Decision-support tool for administrative officers  
*Not a system that passes legal or acquisition orders*

## Slide 2 — Problem

Land acquisition often slips because of ownership disputes, incomplete documents, pending survey/demarcation, objections, compensation, and approvals. Officers see many files at once and need help **prioritising** high-risk cases early.

## Slide 3 — Solution

A web dashboard that, for each case:

- Predicts delay risk and expected days
- Explains top reasons (SHAP)
- Recommends process actions
- Maps cases (prototype GIS)
- Stores history and alerts

## Slide 4 — Data (important)

**Synthetic Dataset for Prototype and Model Development** (~5,500 cases).  
Labels are generated from realistic risk factors, not independent random coins.  
This is **not** real government case data.

## Slide 5 — ML design

- **Model A:** Will the case be delayed? → probability + LOW/MEDIUM/HIGH/CRITICAL  
- **Model B:** Expected delay (days)  
- Algorithms compared: Random Forest, Gradient Boosting, XGBoost  
- Winner chosen from **held-out test metrics** in `models/metrics.json`  
- Inputs never include `delayed` or `expected_delay_days`

Paste the real Accuracy, Precision, Recall, F1, ROC-AUC, MAE, RMSE **after training**.

## Slide 6 — Explainability and recommendations

SHAP lists why risk is high (example: dispute, compensation pending, survey incomplete).  
Rules map those factors to officer actions (verify ownership, complete survey, process compensation, hear objections).  
Disclaimer on every recommendation.

## Slide 7 — System

Streamlit → FastAPI → SQLite  
Modules: predict, explain, recommend, GIS, early warning, case management, login/roles.

## Slide 8 — Demo snapshot

Dashboard KPIs → critical case → map → new prediction.

## Slide 9 — Impact and limits

**Impact:** faster attention to likely delays; transparent reasons.  
**Limits:** synthetic data; prototype security; no SMS/email; GIS is not cadastral.  
**Next:** departmental historical cases, official GIS, SSO, audit logs.

## Slide 10 — Thank you

Q&A: “Does the AI reject a file?” → No. Officers decide. The model only ranks and explains.
