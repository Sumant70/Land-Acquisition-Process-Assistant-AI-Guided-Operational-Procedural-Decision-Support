# ER diagram (prototype SQLite)

```
users
-----
id PK
username UK
full_name
hashed_password
role                 -- officer | admin
is_active

cases
-----
id PK
case_id UK
state, district, project_type
land_area_hectares
number_of_landowners
ownership_dispute, legal_case
document_completeness_pct
survey_completed, demarcation_completed
compensation_paid, approval_completed
number_of_objections
historical_delay_rate_pct
distance_to_project_km
status
days_open
latitude, longitude
coordinate_source
remarks
created_at, updated_at

predictions
-----------
id PK
case_pk FK -> cases.id
predicted_at
delay_probability
no_delay_probability
risk_level
expected_delay_days
main_reason
reasons_json
priority_score
priority_label
model_version

actions
-------
id PK
case_pk FK -> cases.id
code
action
priority
status                 -- recommended | completed
created_at

case_remarks
------------
id PK
case_pk FK -> cases.id
officer_username
remark
created_at

alerts
------
id PK
case_id
alert_type
message
severity
is_read
created_at
```

Relationships: one case has many predictions (history), many actions, many remarks. Alerts reference `case_id` for a simple notification feed.
