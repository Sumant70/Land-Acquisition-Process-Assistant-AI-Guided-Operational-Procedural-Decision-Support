# 3–5 minute SIH demo script

**Opening line (10 seconds)**  
“This is a decision-support system for officers monitoring land-acquisition delays. All data on screen is a synthetic prototype dataset, not live government cases. The AI does not pass legal orders.”

## Minute 0:00–0:30 — Login and dashboard

1. Open Streamlit.
2. Log in as `officer` / `officer123`.
3. Point to KPIs: total cases, high-risk, critical, average predicted delay.
4. Show risk bar chart and the alert strip.

**Say:** “The system already ranked thousands of synthetic cases so the officer starts with exceptions, not a spreadsheet.”

## Minute 0:30–1:40 — Critical case

1. Open **Cases**, filter Risk = CRITICAL.
2. Select the top case.
3. Show risk badge, probability, estimated days.
4. Read 3–4 SHAP reasons (dispute, compensation, survey, documents).
5. Read two recommended actions.

**Say:** “The officer sees why the model is worried and what to check next. This is explainable decision support.”

## Minute 1:40–2:20 — GIS

1. Open **GIS map**.
2. Filter CRITICAL.
3. Click a red marker: case id, district, risk, delay, main reason.

**Say:** “Markers are synthetic coordinates near district centres for this prototype, coloured by risk.”

## Minute 2:20–3:20 — New case prediction

1. Open **Predict**.
2. Enter a difficult case: ownership dispute = Yes, legal case = Yes, compensation = No, survey = No, documents ~ 55%, objections high.
3. Click **Predict Delay Risk**.
4. Show CRITICAL/HIGH, probability, days, reasons, actions.
5. Confirm it saved; return to Overview if time allows.

## Minute 3:20–4:00 — Early warning + close

1. Open **Early warning**: Immediate Attention vs Action Required Soon.
2. Optional: Assistant → “Which cases need immediate attention?”
3. If judges ask metrics, log in as `admin` / `admin123` → Analytics → real `metrics.json` numbers.

**Closing line**  
“The prototype covers ranking, explanation, recommendations, GIS, and case history. Next steps would be real departmental data, role-based SSO, and official GIS parcels — still as support, not automated sanction.”
