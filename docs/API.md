# API documentation

Base URL: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

All case/prediction routes except `/` require header:

`Authorization: Bearer <token>`

## Auth

### POST `/auth/login`

OAuth2 form (`username`, `password`). Used by Swagger “Authorize”.

### POST `/auth/login-json`

JSON body:

```json
{ "username": "officer", "password": "officer123" }
```

Response: `access_token`, `role`, `full_name`.

### GET `/me`

Current user.

## Prediction

### POST `/predict`

Body: case attributes (no `delayed` / `expected_delay_days`).

Response includes `delay_probability`, `no_delay_probability`, `risk_level`, `expected_delay_days`, `reasons`, `recommendations`, `priority_label`, `alerts`.

## Cases

### GET `/cases`

Query: `state`, `district`, `project_type`, `risk_level`, `status`, `q`, `limit`, `offset`.

### GET `/cases/{case_id}`

Full detail, live SHAP explanation, prediction history, actions, remarks.

### POST `/cases`

Create + score + store.

### PUT `/cases/{case_id}`

Update status / process flags / remarks fields; stores a new prediction snapshot.

### POST `/cases/{case_id}/remarks`

Add an officer remark.

## Monitoring

### GET `/dashboard/stats`

Totals, high/critical counts, average predicted delay, delayed-like percentage, risk mix.

### GET `/high-risk-cases`

HIGH and CRITICAL, sorted by priority score.

### GET `/alerts`

Notification feed (in-app only; no SMS/email in this prototype).

### GET `/gis/points`

Synthetic lat/lon for Folium.

### GET `/analytics`

Admin only. State / district / project-type risk, common reasons, historical vs predicted sample.

### GET `/metrics`

Contents of `models/metrics.json` (real validation numbers).

### POST `/assistant/ask`

```json
{ "question": "Which cases need immediate attention?", "case_id": null }
```
