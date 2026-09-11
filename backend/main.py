from __future__ import annotations

from datetime import datetime
import json
import uuid
from typing import Any, Optional

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth import (
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    require_analyst_or_admin,
    require_officer_or_admin,
    verify_password,
)
from backend.config import settings
from backend.database import Base, SessionLocal, engine, get_db
from backend.ml_service import reload_ml_artifacts, score_case
from backend.models import Action, Alert, AuditLog, Case, CaseRemark, CaseTimeline, Prediction, User
from backend.schemas import (
    ALLOWED_STATUSES,
    CaseCreate,
    CaseUpdate,
    LoginRequest,
    PredictRequest,
    RemarkCreate,
    TokenResponse,
    WhatIfRequest,
    WhatIfResponse,
)
from src.config import (
    DATASET_DISCLAIMER,
    FEATURED_DATASET_PATH,
    LIFECYCLE_STAGES,
    METRICS_PATH,
    PROJECT_ROOT,
    PROJECT_TYPES,
)
from src.location_master import (
    get_all_states,
    get_district_coordinates,
    get_districts_for_state,
    validate_location,
)
from src.predict import compute_risk_score, load_artifacts, predict_frame
from src.reason_fallback import heuristic_main_reason
from src.recommendations import build_recommendations, generate_alerts, priority_score

app = FastAPI(
    title="AI-Powered Land Acquisition Delay Prediction & Decision Support System",
    description=(
        "Executive Decision Support Platform for Smart India Hackathon (SIH). "
        "Predicts acquisition delay probabilities, expected delay timelines, calibrated 0-100 risk scores, "
        "and produces Explainable AI (SHAP) attributions with 5-attribute actionable recommendations. "
        "Trained on a **Synthetic Prototype Dataset** for demonstration."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _ensure_users(db: Session) -> None:
    if db.query(User).count() > 0:
        return
    db.add_all(
        [
            User(
                username=settings.default_admin_username,
                full_name="System Administrator (HQ)",
                hashed_password=hash_password(settings.default_admin_password),
                role="admin",
            ),
            User(
                username=settings.default_officer_username,
                full_name="Chief Land Acquisition Officer (CALA)",
                hashed_password=hash_password(settings.default_officer_password),
                role="officer",
            ),
            User(
                username="analyst",
                full_name="Senior GIS & Risk Analyst",
                hashed_password=hash_password("analyst123"),
                role="analyst",
            ),
            User(
                username="viewer",
                full_name="Public Works Nodal Observer",
                hashed_password=hash_password("viewer123"),
                role="viewer",
            ),
        ]
    )
    db.commit()


def _case_to_payload(row: dict) -> dict:
    keys = [
        "project_name",
        "state",
        "district",
        "project_type",
        "land_area_hectares",
        "affected_families",
        "number_of_landowners",
        "ownership_dispute",
        "legal_dispute",
        "legal_case",
        "number_of_legal_cases",
        "document_completeness_pct",
        "notification_status",
        "survey_status",
        "survey_completed",
        "demarcation_status",
        "demarcation_completed",
        "approval_status",
        "approval_completed",
        "approval_delay_days",
        "compensation_status",
        "compensation_paid",
        "compensation_delay_days",
        "possession_status",
        "rehabilitation_status",
        "resettlement_status",
        "stakeholder_responsiveness",
        "administrative_bottleneck",
        "inter_department_coordination",
        "number_of_objections",
        "historical_delay_rate_pct",
        "distance_to_project_km",
        "current_stage",
        "status",
        "days_open",
    ]
    return {k: row[k] for k in keys if k in row}


def _serialize_case(case: Case, prediction: Prediction | None = None) -> dict:
    pred = prediction or (case.predictions[-1] if case.predictions else None)
    return {
        "case_id": case.case_id,
        "project_name": case.project_name or f"{case.project_type} - {case.district}",
        "state": case.state,
        "district": case.district,
        "project_type": case.project_type,
        "land_area_hectares": case.land_area_hectares,
        "affected_families": case.affected_families,
        "number_of_landowners": case.number_of_landowners,
        "ownership_dispute": case.ownership_dispute,
        "legal_dispute": case.legal_dispute,
        "legal_case": case.legal_case,
        "number_of_legal_cases": case.number_of_legal_cases,
        "document_completeness_pct": case.document_completeness_pct,
        "notification_status": case.notification_status,
        "survey_status": case.survey_status,
        "survey_completed": case.survey_completed,
        "demarcation_status": case.demarcation_status,
        "demarcation_completed": case.demarcation_completed,
        "approval_status": case.approval_status,
        "approval_completed": case.approval_completed,
        "approval_delay_days": case.approval_delay_days,
        "compensation_status": case.compensation_status,
        "compensation_paid": case.compensation_paid,
        "compensation_delay_days": case.compensation_delay_days,
        "possession_status": case.possession_status,
        "rehabilitation_status": case.rehabilitation_status,
        "resettlement_status": case.resettlement_status,
        "stakeholder_responsiveness": case.stakeholder_responsiveness,
        "administrative_bottleneck": case.administrative_bottleneck,
        "inter_department_coordination": case.inter_department_coordination,
        "number_of_objections": case.number_of_objections,
        "historical_delay_rate_pct": case.historical_delay_rate_pct,
        "distance_to_project_km": case.distance_to_project_km,
        "current_stage": case.current_stage,
        "status": case.status,
        "days_open": case.days_open,
        "latitude": case.latitude,
        "longitude": case.longitude,
        "coordinate_source": case.coordinate_source,
        "remarks": case.remarks,
        "risk_score": pred.risk_score if pred else 0,
        "risk_level": pred.risk_level if pred else "LOW",
        "risk_category": pred.risk_level if pred else "LOW",
        "delay_probability": pred.delay_probability if pred else 0.0,
        "expected_delay_days": pred.expected_delay_days if pred else 0,
        "main_reason": pred.main_reason if pred else "Routine monitoring",
        "priority_score": pred.priority_score if pred else 0,
        "priority_label": pred.priority_label if pred else "Priority 4 — Routine Tracking",
        "predicted_at": pred.predicted_at.isoformat() if pred else None,
        "dataset_note": DATASET_DISCLAIMER,
    }


def seed_database_if_empty() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        _ensure_users(db)
        if db.query(Case).count() > 0:
            return
        csv_path = FEATURED_DATASET_PATH
        if not csv_path.exists():
            csv_path = PROJECT_ROOT / "data" / "processed" / "clean_land_acquisition_cases.csv"
        if not csv_path.exists():
            print("No dataset found to seed SQLite database. Run python -m src.run_pipeline")
            return

        df = pd.read_csv(csv_path)
        artifacts = load_artifacts()
        scored = predict_frame(df, artifacts=artifacts)
        now = datetime.utcnow()

        cases = []
        for _, row in scored.iterrows():
            st_name = str(row["state"])
            dist_name = str(row["district"])
            lat_val = float(row.get("latitude", 0.0))
            lon_val = float(row.get("longitude", 0.0))
            if lat_val == 0.0 or lon_val == 0.0:
                lat_val, lon_val = get_district_coordinates(st_name, dist_name)

            cases.append(
                Case(
                    case_id=str(row["case_id"]),
                    project_name=str(row.get("project_name", f"{row.get('project_type')} - {dist_name}")),
                    state=st_name,
                    district=dist_name,
                    project_type=str(row["project_type"]),
                    land_area_hectares=float(row["land_area_hectares"]),
                    affected_families=int(row.get("affected_families", 25)),
                    number_of_landowners=int(row["number_of_landowners"]),
                    ownership_dispute=int(row["ownership_dispute"]),
                    legal_dispute=int(row.get("legal_dispute", row.get("legal_case", 0))),
                    legal_case=int(row.get("legal_dispute", row.get("legal_case", 0))),
                    number_of_legal_cases=int(row.get("number_of_legal_cases", 0)),
                    document_completeness_pct=float(row["document_completeness_pct"]),
                    notification_status=str(row.get("notification_status", "Preliminary Section 11 Issued")),
                    survey_status=str(row.get("survey_status", "Completed")),
                    survey_completed=int(row["survey_completed"]),
                    demarcation_status=str(row.get("demarcation_status", "Completed")),
                    demarcation_completed=int(row["demarcation_completed"]),
                    approval_status=str(row.get("approval_status", "Approved")),
                    approval_completed=int(row["approval_completed"]),
                    approval_delay_days=int(row.get("approval_delay_days", 0)),
                    compensation_status=str(row.get("compensation_status", "Disbursed")),
                    compensation_paid=int(row["compensation_paid"]),
                    compensation_delay_days=int(row.get("compensation_delay_days", 0)),
                    possession_status=str(row.get("possession_status", "Pending")),
                    rehabilitation_status=str(row.get("rehabilitation_status", "Completed")),
                    resettlement_status=str(row.get("resettlement_status", "Completed")),
                    stakeholder_responsiveness=str(row.get("stakeholder_responsiveness", "High")),
                    administrative_bottleneck=str(row.get("administrative_bottleneck", "Low")),
                    inter_department_coordination=str(row.get("inter_department_coordination", "Seamless")),
                    number_of_objections=int(row["number_of_objections"]),
                    historical_delay_rate_pct=float(row["historical_delay_rate_pct"]),
                    distance_to_project_km=float(row["distance_to_project_km"]),
                    current_stage=str(row.get("current_stage", "Compensation")),
                    status=str(row.get("status", "Initiated")),
                    days_open=int(row.get("days_open", 30)),
                    latitude=lat_val,
                    longitude=lon_val,
                    coordinate_source=str(row.get("coordinate_source", "Synthetic prototype coordinates")),
                    remarks="",
                    created_at=now,
                    updated_at=now,
                )
            )

        db.add_all(cases)
        db.flush()

        id_map = {c.case_id: c.id for c in cases}
        predictions = []
        alerts = []
        actions = []
        timelines = []
        audit_logs = []

        for _, row in scored.iterrows():
            c_id = str(row["case_id"])
            c_pk = id_map[c_id]
            payload = _case_to_payload(row.to_dict())

            delay_p = float(row["delay_probability"])
            days = int(row["predicted_delay_days"])
            score_val = int(row.get("risk_score", int(delay_p * 100)))
            risk_cat = str(row["risk_level"])

            main_r = heuristic_main_reason(payload)
            pscore, plabel = priority_score(risk_cat, payload, delay_p, days)

            predictions.append(
                Prediction(
                    case_pk=c_pk,
                    predicted_at=now,
                    delay_probability=delay_p,
                    no_delay_probability=float(row["no_delay_probability"]),
                    risk_score=score_val,
                    risk_level=risk_cat,
                    expected_delay_days=days,
                    main_reason=main_r,
                    reasons_json=json.dumps([{"factor": main_r, "impact": "High impact", "shap_value": 0.45}]),
                    factors_reducing_json=json.dumps([{"factor": "Cadastral survey verified", "impact": "Medium impact", "shap_value": -0.22}]),
                    human_explanation=f"Project is categorized as {risk_cat} risk with {round(delay_p*100)}% delay probability and {days} days predicted delay.",
                    priority_score=pscore,
                    priority_label=plabel,
                )
            )

            # Alerts
            for alert in generate_alerts(payload, risk_cat, days):
                alerts.append(
                    Alert(
                        case_id=c_id,
                        alert_type=alert["type"],
                        message=alert["message"],
                        severity=alert["severity"],
                        is_read=False,
                        created_at=now,
                    )
                )

            # Recommendations for elevated risk
            if risk_cat in ["MEDIUM", "HIGH", "CRITICAL"]:
                for rec in build_recommendations(payload):
                    if rec["code"] == "DISCLAIMER":
                        continue
                    actions.append(
                        Action(
                            case_pk=c_pk,
                            code=rec["code"],
                            issue=rec.get("issue", ""),
                            action=rec["action"],
                            priority=rec["priority"],
                            department=rec.get("department", "Competent Authority"),
                            timeline=rec.get("timeline", "Within 15 days"),
                            status="recommended",
                            created_at=now,
                        )
                    )

            # Lifecycle Timeline
            cur_stage = str(row.get("current_stage", "Compensation"))
            passed = True
            for stg in LIFECYCLE_STAGES:
                if stg == cur_stage:
                    stg_stat = "In Progress"
                    is_bn = (risk_cat in ["HIGH", "CRITICAL"])
                    passed = False
                elif passed:
                    stg_stat = "Completed"
                    is_bn = False
                else:
                    stg_stat = "Pending"
                    is_bn = False

                timelines.append(
                    CaseTimeline(
                        case_pk=c_pk,
                        stage_name=stg,
                        status=stg_stat,
                        bottleneck_flag=is_bn,
                        notes="Milestone tracked in system",
                        updated_at=now,
                    )
                )

        audit_logs.append(
            AuditLog(
                user="system",
                action="INITIAL_SEED",
                case_id="ALL",
                previous_value="None",
                new_value=f"Seeded {len(cases)} cases across 36 States/UTs",
                timestamp=now,
            )
        )

        db.add_all(predictions)
        db.add_all(alerts)
        db.add_all(actions)
        db.add_all(timelines)
        db.add_all(audit_logs)
        db.commit()
        print(f"Successfully seeded {len(cases)} cases into SQLite database.")
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    seed_database_if_empty()


# --------------------------------------------------------------------------
# Root & Auth Endpoints
# --------------------------------------------------------------------------


@app.get("/")
def root():
    return {
        "platform": "AI-Powered Land Acquisition Delay Prediction & Decision Support System",
        "tagline": "From Reactive Monitoring to Predictive Governance",
        "version": "2.0.0",
        "hackathon": "Smart India Hackathon (SIH)",
        "disclaimer": DATASET_DISCLAIMER,
        "docs_url": "/docs",
        "status": "Operational",
    }


@app.get("/health")
@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "FastAPI Land Acquisition AI", "version": "2.0.0"}



@app.post("/auth/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.username, user.role)
    return TokenResponse(access_token=token, role=user.role, full_name=user.full_name)


@app.post("/auth/login-json", response_model=TokenResponse)
def login_json(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(user.username, user.role)
    return TokenResponse(access_token=token, role=user.role, full_name=user.full_name)


@app.get("/me")
def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "full_name": user.full_name, "role": user.role}


# --------------------------------------------------------------------------
# Location Master APIs (All 36 States & UTs, 786 Districts)
# --------------------------------------------------------------------------


@app.get("/states")
def list_states():
    """Return authoritative list of all 28 Indian States and 8 Union Territories."""
    return get_all_states()


@app.get("/states/{state}/districts")
def list_districts_for_state(state: str):
    """Return only officially valid districts belonging to the requested State/UT."""
    districts = get_districts_for_state(state)
    if not districts:
        raise HTTPException(status_code=404, detail=f"State/UT '{state}' not found in master database.")
    return districts


@app.get("/location/validate")
def check_location(state: str, district: str):
    is_valid = validate_location(state, district)
    return {
        "state": state,
        "district": district,
        "is_valid": is_valid,
        "message": "Valid State-District pair" if is_valid else f"District '{district}' does NOT belong to State/UT '{state}'.",
    }


# --------------------------------------------------------------------------
# Case Management & Predictions
# --------------------------------------------------------------------------


@app.post("/predict")
def predict(body: PredictRequest, user: User = Depends(get_current_user)):
    """Run model inference (Classification + Regression + SHAP + Risk Score 0-100)."""
    if not validate_location(body.state, body.district):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid location: District '{body.district}' is not part of '{body.state}'.",
        )
    payload = body.model_dump()
    result = score_case(payload)
    return {"ok": True, "requested_by": user.username, **result}


@app.post("/what-if", response_model=WhatIfResponse)
def what_if_simulator(body: WhatIfRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    What-If Scenario Simulation:
    Recalculates risk score, delay probability, and expected delay days under hypothetical milestone changes.
    """
    base_dict = {}
    case_id_label = body.case_id or "Simulated Custom Project"

    if body.case_id:
        case = db.query(Case).filter(Case.case_id == body.case_id).first()
        if not case:
            raise HTTPException(status_code=404, detail=f"Case {body.case_id} not found.")
        base_dict = _case_to_payload(case.__dict__)
    elif body.base_case:
        base_dict = body.base_case.copy()
    else:
        # Default baseline
        base_dict = {
            "state": "Uttarakhand",
            "district": "Dehradun",
            "project_type": "National Highway",
            "land_area_hectares": 20.0,
            "number_of_landowners": 45,
            "affected_families": 40,
            "ownership_dispute": 1,
            "legal_dispute": 1,
            "legal_case": 1,
            "document_completeness_pct": 65.0,
            "survey_completed": 1,
            "demarcation_completed": 0,
            "approval_completed": 0,
            "approval_delay_days": 45,
            "compensation_paid": 0,
            "compensation_delay_days": 60,
            "number_of_objections": 6,
            "historical_delay_rate_pct": 35.0,
            "distance_to_project_km": 10.0,
            "current_stage": "Approval",
            "status": "Initiated",
            "days_open": 90,
        }

    # Original evaluation
    orig_res = score_case(base_dict)

    # Apply scenario changes
    mod_dict = base_dict.copy()
    fields_to_check = [
        "compensation_paid",
        "compensation_status",
        "compensation_delay_days",
        "document_completeness_pct",
        "legal_dispute",
        "ownership_dispute",
        "approval_completed",
        "approval_status",
        "approval_delay_days",
        "number_of_objections",
        "survey_completed",
        "rehabilitation_status",
        "stakeholder_responsiveness",
        "inter_department_coordination",
        "administrative_bottleneck",
    ]
    for field in fields_to_check:
        val = getattr(body, field, None)
        if val is not None:
            mod_dict[field] = val

    sim_res = score_case(mod_dict)

    risk_delta = sim_res["risk_score"] - orig_res["risk_score"]
    delay_delta = sim_res["expected_delay_days"] - orig_res["expected_delay_days"]
    prob_delta = round(sim_res["delay_probability"] - orig_res["delay_probability"], 4)

    if risk_delta < 0:
        summary = (
            f"Intervention reduces Risk Score by {abs(risk_delta)} points "
            f"(from {orig_res['risk_score']} to {sim_res['risk_score']}) and avoids "
            f"{abs(delay_delta)} days of projected delay."
        )
    elif risk_delta == 0:
        summary = "No significant risk change observed with the selected parameter adjustments."
    else:
        summary = f"Adjustments increased the risk score by {risk_delta} points."

    return WhatIfResponse(
        case_id=case_id_label,
        original={
            "risk_score": orig_res["risk_score"],
            "risk_level": orig_res["risk_level"],
            "delay_probability": orig_res["delay_probability"],
            "expected_delay_days": orig_res["expected_delay_days"],
            "main_reason": orig_res["main_reason"],
        },
        simulated={
            "risk_score": sim_res["risk_score"],
            "risk_level": sim_res["risk_level"],
            "delay_probability": sim_res["delay_probability"],
            "expected_delay_days": sim_res["expected_delay_days"],
            "main_reason": sim_res["main_reason"],
            "recommendations": sim_res["recommendations"],
        },
        risk_score_delta=risk_delta,
        delay_days_delta=delay_delta,
        delay_probability_delta=prob_delta,
        summary=summary,
    )


@app.get("/cases")
def list_cases(
    state: Optional[str] = None,
    district: Optional[str] = None,
    project_type: Optional[str] = None,
    risk_level: Optional[str] = None,
    current_stage: Optional[str] = None,
    status: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    query = (
        db.query(Case, Prediction)
        .outerjoin(latest, Case.id == latest.c.case_pk)
        .outerjoin(Prediction, Prediction.id == latest.c.max_id)
    )

    if state:
        query = query.filter(Case.state == state)
    if district:
        query = query.filter(Case.district == district)
    if project_type:
        query = query.filter(Case.project_type == project_type)
    if status:
        query = query.filter(Case.status == status)
    if current_stage:
        query = query.filter(Case.current_stage == current_stage)
    if risk_level:
        query = query.filter(Prediction.risk_level == risk_level)
    if q:
        like = f"%{q}%"
        query = query.filter(
            Case.case_id.ilike(like)
            | Case.project_name.ilike(like)
            | Case.district.ilike(like)
            | Case.state.ilike(like)
        )

    total = query.count()
    rows = query.order_by(Prediction.priority_score.desc().nullslast(), Case.case_id).offset(offset).limit(limit).all()
    return {"total": total, "items": [_serialize_case(c, p) for c, p in rows]}


@app.get("/cases/{case_id}")
def get_case(case_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    history = [
        {
            "predicted_at": p.predicted_at.isoformat(),
            "risk_score": p.risk_score,
            "risk_level": p.risk_level,
            "delay_probability": p.delay_probability,
            "expected_delay_days": p.expected_delay_days,
            "main_reason": p.main_reason,
            "priority_label": p.priority_label,
        }
        for p in sorted(case.predictions, key=lambda x: x.predicted_at)
    ]

    actions = [
        {
            "code": a.code,
            "issue": a.issue,
            "action": a.action,
            "priority": a.priority,
            "department": a.department,
            "timeline": a.timeline,
            "status": a.status,
            "created_at": a.created_at.isoformat(),
        }
        for a in case.actions
    ]

    timeline_events = [
        {
            "stage_name": t.stage_name,
            "status": t.status,
            "bottleneck_flag": t.bottleneck_flag,
            "notes": t.notes,
        }
        for t in sorted(case.timeline, key=lambda x: x.id)
    ]

    remarks = [
        {"officer": r.officer_username, "remark": r.remark, "created_at": r.created_at.isoformat()}
        for r in sorted(case.remarks_log, key=lambda x: x.created_at, reverse=True)
    ]

    detail = _serialize_case(case)
    # Generate live bidirectional explanation
    live = score_case(_case_to_payload(detail))
    detail["live_explanation"] = live
    detail["prediction_history"] = history
    detail["actions"] = actions
    detail["timeline"] = timeline_events
    detail["remarks_log"] = remarks
    return detail


@app.get("/cases/{case_id}/explanation")
def get_case_explanation(case_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    payload = _case_to_payload(case.__dict__)
    return score_case(payload)


@app.get("/cases/{case_id}/recommendations")
def get_case_recommendations(case_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")
    payload = _case_to_payload(case.__dict__)
    recs = build_recommendations(payload)
    return {"case_id": case_id, "recommendations": recs}


@app.post("/cases")
def create_case(body: CaseCreate, user: User = Depends(require_officer_or_admin), db: Session = Depends(get_db)):
    if not validate_location(body.state, body.district):
        raise HTTPException(status_code=400, detail=f"District '{body.district}' is not valid for state '{body.state}'.")

    case_id = body.case_id or f"LA-2026-{uuid.uuid4().hex[:6].upper()}"
    if db.query(Case).filter(Case.case_id == case_id).first():
        raise HTTPException(status_code=400, detail=f"Case ID '{case_id}' already exists.")

    lat, lon = body.latitude, body.longitude
    if lat is None or lon is None:
        lat, lon = get_district_coordinates(body.state, body.district)

    payload = body.model_dump()
    scored = score_case(payload)
    now = datetime.utcnow()

    case = Case(
        case_id=case_id,
        project_name=body.project_name or f"{body.project_type} - {body.district}",
        state=body.state,
        district=body.district,
        project_type=body.project_type,
        land_area_hectares=body.land_area_hectares,
        affected_families=body.affected_families,
        number_of_landowners=body.number_of_landowners,
        ownership_dispute=body.ownership_dispute,
        legal_dispute=body.legal_dispute,
        legal_case=body.legal_dispute,
        number_of_legal_cases=body.number_of_legal_cases,
        document_completeness_pct=body.document_completeness_pct,
        notification_status=body.notification_status,
        survey_status=body.survey_status,
        survey_completed=body.survey_completed,
        demarcation_status=body.demarcation_status,
        demarcation_completed=body.demarcation_completed,
        approval_status=body.approval_status,
        approval_completed=body.approval_completed,
        approval_delay_days=body.approval_delay_days,
        compensation_status=body.compensation_status,
        compensation_paid=body.compensation_paid,
        compensation_delay_days=body.compensation_delay_days,
        possession_status=body.possession_status,
        rehabilitation_status=body.rehabilitation_status,
        resettlement_status=body.resettlement_status,
        stakeholder_responsiveness=body.stakeholder_responsiveness,
        administrative_bottleneck=body.administrative_bottleneck,
        inter_department_coordination=body.inter_department_coordination,
        number_of_objections=body.number_of_objections,
        historical_delay_rate_pct=body.historical_delay_rate_pct,
        distance_to_project_km=body.distance_to_project_km,
        current_stage=body.current_stage,
        status=body.status if body.status in ALLOWED_STATUSES else "Initiated",
        days_open=body.days_open,
        latitude=lat,
        longitude=lon,
        coordinate_source="Authoritative District Centroid (LGD)",
        remarks=body.remarks or "",
        created_at=now,
        updated_at=now,
    )
    db.add(case)
    db.flush()

    db.add(
        Prediction(
            case_pk=case.id,
            predicted_at=now,
            delay_probability=scored["delay_probability"],
            no_delay_probability=scored["no_delay_probability"],
            risk_score=scored["risk_score"],
            risk_level=scored["risk_level"],
            expected_delay_days=scored["expected_delay_days"],
            main_reason=scored["main_reason"],
            reasons_json=json.dumps(scored["reasons"]),
            factors_reducing_json=json.dumps(scored["factors_reducing_risk"]),
            human_explanation=scored["human_explanation"],
            priority_score=scored["priority_score"],
            priority_label=scored["priority_label"],
        )
    )

    # Actions
    for rec in scored["recommendations"]:
        if rec["code"] == "DISCLAIMER":
            continue
        db.add(
            Action(
                case_pk=case.id,
                code=rec["code"],
                issue=rec.get("issue", ""),
                action=rec["action"],
                priority=rec["priority"],
                department=rec.get("department", "Competent Authority"),
                timeline=rec.get("timeline", "Within 15 days"),
                status="recommended",
            )
        )

    # Alerts
    for alert in scored["alerts"]:
        db.add(
            Alert(
                case_id=case_id,
                alert_type=alert["type"],
                message=alert["message"],
                severity=alert["severity"],
            )
        )

    # Audit Trail
    db.add(
        AuditLog(
            user=user.username,
            action="CREATE_CASE",
            case_id=case_id,
            previous_value="None",
            new_value=json.dumps({"risk_score": scored["risk_score"], "stage": body.current_stage}),
            timestamp=now,
        )
    )

    db.commit()
    return {"ok": True, "case": _serialize_case(case), "prediction": scored}


@app.put("/cases/{case_id}")
def update_case(case_id: str, body: CaseUpdate, user: User = Depends(require_officer_or_admin), db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    data = body.model_dump(exclude_unset=True)
    if "status" in data and data["status"] not in ALLOWED_STATUSES:
        raise HTTPException(status_code=400, detail=f"Invalid status: {data['status']}")

    prev_state = {"status": case.status, "current_stage": case.current_stage, "comp": case.compensation_paid}

    for key, value in data.items():
        if hasattr(case, key) and value is not None:
            setattr(case, key, value)

    case.updated_at = datetime.utcnow()
    payload = _case_to_payload(case.__dict__)
    scored = score_case(payload)

    db.add(
        Prediction(
            case_pk=case.id,
            predicted_at=case.updated_at,
            delay_probability=scored["delay_probability"],
            no_delay_probability=scored["no_delay_probability"],
            risk_score=scored["risk_score"],
            risk_level=scored["risk_level"],
            expected_delay_days=scored["expected_delay_days"],
            main_reason=scored["main_reason"],
            reasons_json=json.dumps(scored["reasons"]),
            factors_reducing_json=json.dumps(scored["factors_reducing_risk"]),
            human_explanation=scored["human_explanation"],
            priority_score=scored["priority_score"],
            priority_label=scored["priority_label"],
        )
    )

    db.add(
        AuditLog(
            user=user.username,
            action="UPDATE_CASE",
            case_id=case_id,
            previous_value=json.dumps(prev_state),
            new_value=json.dumps({"new_risk": scored["risk_score"], "status": case.status}),
            timestamp=case.updated_at,
        )
    )

    db.commit()
    return {"ok": True, "case": _serialize_case(case), "prediction": scored}


@app.post("/cases/{case_id}/remarks")
def add_remark(case_id: str, body: RemarkCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    case = db.query(Case).filter(Case.case_id == case_id).first()
    if not case:
        raise HTTPException(status_code=404, detail="Case not found.")

    db.add(CaseRemark(case_pk=case.id, officer_username=user.username, remark=body.remark))
    existing = case.remarks or ""
    case.remarks = (existing + "\n" if existing else "") + f"[{user.username}] {body.remark}"
    db.commit()
    return {"ok": True}


# --------------------------------------------------------------------------
# Dashboard, Analytics & GIS Endpoints
# --------------------------------------------------------------------------


@app.get("/dashboard/stats")
def dashboard_stats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    rows = (
        db.query(Case, Prediction)
        .join(latest, Case.id == latest.c.case_pk)
        .join(Prediction, Prediction.id == latest.c.max_id)
        .all()
    )

    total = len(rows)
    if total == 0:
        return {"total_cases": 0, "dataset_note": DATASET_DISCLAIMER}

    risks = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    delays = []
    scores = []
    delayed_count = 0

    for _, p in rows:
        risks[p.risk_level] = risks.get(p.risk_level, 0) + 1
        delays.append(p.expected_delay_days)
        scores.append(p.risk_score)
        if p.delay_probability >= 0.5:
            delayed_count += 1

    return {
        "total_cases": total,
        "high_risk_cases": risks.get("HIGH", 0),
        "critical_cases": risks.get("CRITICAL", 0),
        "projects_at_risk": risks.get("HIGH", 0) + risks.get("CRITICAL", 0),
        "average_predicted_delay": round(sum(delays) / len(delays), 1),
        "average_risk_score": round(sum(scores) / len(scores), 1),
        "delayed_cases_percentage": round(100.0 * delayed_count / total, 1),
        "risk_distribution": risks,
        "dataset_note": DATASET_DISCLAIMER,
    }


@app.get("/high-risk-cases")
def high_risk_cases(limit: int = 50, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    rows = (
        db.query(Case, Prediction)
        .join(latest, Case.id == latest.c.case_pk)
        .join(Prediction, Prediction.id == latest.c.max_id)
        .filter(Prediction.risk_level.in_(["HIGH", "CRITICAL"]))
        .order_by(Prediction.priority_score.desc(), Prediction.risk_score.desc())
        .limit(limit)
        .all()
    )
    return {"items": [_serialize_case(c, p) for c, p in rows]}


@app.get("/analytics/state")
def analytics_state(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """State-wise comparative intelligence metrics."""
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    rows = (
        db.query(Case, Prediction)
        .join(latest, Case.id == latest.c.case_pk)
        .join(Prediction, Prediction.id == latest.c.max_id)
        .all()
    )

    state_map: dict[str, list[Prediction]] = {}
    for c, p in rows:
        state_map.setdefault(c.state, []).append(p)

    out = []
    for st_name, preds in state_map.items():
        total_p = len(preds)
        avg_d = round(sum(x.expected_delay_days for x in preds) / total_p, 1)
        avg_s = round(sum(x.risk_score for x in preds) / total_p, 1)
        crit = sum(1 for x in preds if x.risk_level == "CRITICAL")
        high = sum(1 for x in preds if x.risk_level == "HIGH")
        delayed_prob_avg = round(sum(x.delay_probability for x in preds) / total_p, 3)

        out.append(
            {
                "state": st_name,
                "total_projects": total_p,
                "average_delay_days": avg_d,
                "average_risk_score": avg_s,
                "critical_projects": crit,
                "high_risk_projects": high,
                "total_at_risk": crit + high,
                "average_delay_probability": delayed_prob_avg,
            }
        )

    out.sort(key=lambda x: x["total_at_risk"], reverse=True)
    return {"states": out}


@app.get("/analytics/district")
def analytics_district(state: Optional[str] = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """District-wise intelligence metrics, optionally scoped to a selected State/UT."""
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    q = (
        db.query(Case, Prediction)
        .join(latest, Case.id == latest.c.case_pk)
        .join(Prediction, Prediction.id == latest.c.max_id)
    )
    if state:
        q = q.filter(Case.state == state)

    rows = q.all()
    dist_map: dict[str, list[tuple[Case, Prediction]]] = {}
    for c, p in rows:
        key = f"{c.district} ({c.state})" if not state else c.district
        dist_map.setdefault(key, []).append((c, p))

    out = []
    for dist_key, items in dist_map.items():
        cnt = len(items)
        avg_d = round(sum(p.expected_delay_days for _, p in items) / cnt, 1)
        avg_s = round(sum(p.risk_score for _, p in items) / cnt, 1)
        crit = sum(1 for _, p in items if p.risk_level == "CRITICAL")
        high = sum(1 for _, p in items if p.risk_level == "HIGH")

        out.append(
            {
                "district": dist_key,
                "total_projects": cnt,
                "average_delay_days": avg_d,
                "average_risk_score": avg_s,
                "critical_projects": crit,
                "high_risk_projects": high,
                "total_at_risk": crit + high,
            }
        )

    out.sort(key=lambda x: x["total_at_risk"], reverse=True)
    return {"state": state, "districts": out}


@app.get("/gis/points")
def gis_points(
    state: Optional[str] = None,
    district: Optional[str] = None,
    risk_level: Optional[str] = None,
    project_type: Optional[str] = None,
    limit: int = Query(1000, ge=50, le=4000),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    q = (
        db.query(Case, Prediction)
        .join(latest, Case.id == latest.c.case_pk)
        .join(Prediction, Prediction.id == latest.c.max_id)
    )

    if state:
        q = q.filter(Case.state == state)
    if district:
        q = q.filter(Case.district == district)
    if risk_level:
        q = q.filter(Prediction.risk_level == risk_level)
    if project_type:
        q = q.filter(Case.project_type == project_type)

    rows = q.order_by(Prediction.priority_score.desc()).limit(limit).all()

    return {
        "disclaimer": "Synthetic project locations for prototype demonstration. Not real surveyed cadastral boundaries.",
        "count": len(rows),
        "items": [
            {
                "case_id": c.case_id,
                "project_name": c.project_name,
                "state": c.state,
                "district": c.district,
                "project_type": c.project_type,
                "risk_score": p.risk_score,
                "risk_level": p.risk_level,
                "delay_probability": p.delay_probability,
                "expected_delay_days": p.expected_delay_days,
                "main_reason": p.main_reason,
                "latitude": c.latitude,
                "longitude": c.longitude,
            }
            for c, p in rows
        ],
    }


@app.get("/alerts")
def list_alerts(
    unread_only: bool = False,
    severity: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    q = db.query(Alert).order_by(Alert.created_at.desc())
    if unread_only:
        q = q.filter(Alert.is_read.is_(False))
    if severity:
        q = q.filter(Alert.severity == severity)

    items = q.limit(limit).all()
    return {
        "items": [
            {
                "id": a.id,
                "case_id": a.case_id,
                "type": a.alert_type,
                "message": a.message,
                "severity": a.severity,
                "is_read": a.is_read,
                "created_at": a.created_at.isoformat(),
            }
            for a in items
        ]
    }


@app.post("/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    alert = db.query(Alert).filter(Alert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found.")
    alert.is_read = True
    db.commit()
    return {"ok": True, "alert_id": alert_id}


@app.post("/alerts/read-all")
def mark_all_alerts_read(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(Alert).update({"is_read": True})
    db.commit()
    return {"ok": True, "message": "All alerts marked as read."}


@app.get("/audit-logs")
def get_audit_logs(limit: int = 50, user: User = Depends(require_admin), db: Session = Depends(get_db)):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return {
        "items": [
            {
                "id": l.id,
                "user": l.user,
                "action": l.action,
                "case_id": l.case_id,
                "previous_value": l.previous_value,
                "new_value": l.new_value,
                "timestamp": l.timestamp.isoformat(),
            }
            for l in logs
        ]
    }


@app.get("/reports/summary")
def get_reports_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Generate risk summary digest for export and executive review."""
    latest = (
        db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
        .group_by(Prediction.case_pk)
        .subquery()
    )
    rows = (
        db.query(Case, Prediction)
        .join(latest, Case.id == latest.c.case_pk)
        .join(Prediction, Prediction.id == latest.c.max_id)
        .filter(Prediction.risk_level.in_(["CRITICAL", "HIGH"]))
        .order_by(Prediction.risk_score.desc())
        .limit(200)
        .all()
    )

    items = []
    for c, p in rows:
        items.append(
            {
                "case_id": c.case_id,
                "project_name": c.project_name,
                "state": c.state,
                "district": c.district,
                "project_type": c.project_type,
                "current_stage": c.current_stage,
                "risk_score": p.risk_score,
                "risk_level": p.risk_level,
                "delay_probability_pct": round(p.delay_probability * 100, 1),
                "expected_delay_days": p.expected_delay_days,
                "main_reason": p.main_reason,
                "status": c.status,
            }
        )
    return {"generated_at": datetime.utcnow().isoformat(), "count": len(items), "cases": items}


@app.get("/metrics")
def metrics(user: User = Depends(get_current_user)):
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found. Train models first.")
    return json.loads(METRICS_PATH.read_text())


# --------------------------------------------------------------------------
# Decision-Support Process Assistant
# --------------------------------------------------------------------------


@app.post("/assistant/ask")
def assistant_ask(body: dict, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    question = str(body.get("question", "")).strip().lower()
    case_id = body.get("case_id")
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    def _disclaimer():
        return " (Statutory Advisory: Decision support only; does not replace judicial orders under RFCTLARR Act, 2013)."

    if case_id:
        case = db.query(Case).filter(Case.case_id == case_id).first()
        if not case:
            return {"answer": f"Case '{case_id}' was not found in the register."}
        live = score_case(_case_to_payload(case.__dict__))

        if any(w in question for w in ["why", "reason", "factor", "driver", "cause"]):
            inc = "\n".join(f"- {f['factor']} ({f['impact']})" for f in live["factors_increasing_risk"])
            return {"answer": f"Case {case_id} is flagged as **{live['risk_level']} Risk** (Score: {live['risk_score']}/100):\n{inc}\n\nSummary: {live['human_explanation']}" + _disclaimer()}

        if any(w in question for w in ["action", "recommend", "should", "fix", "mitigate"]):
            recs = "\n".join(f"- **[{r['priority']}]** {r['action']} ({r['department']} — {r['timeline']})" for r in live["recommendations"] if r["code"] != "DISCLAIMER")
            return {"answer": f"Actionable recommendations for {case_id}:\n{recs}" + _disclaimer()}

        return {
            "answer": (
                f"{case_id} ({case.project_name}): Risk Score {live['risk_score']}/100 ({live['risk_level']}), "
                f"Delay Probability {round(live['delay_probability']*100)}%, Expected Delay {live['expected_delay_days']} days. "
                f"Main bottleneck: {live['main_reason']}."
            )
            + _disclaimer()
        }

    if any(w in question for w in ["critical", "immediate", "urgent", "attention", "high risk"]):
        latest = (
            db.query(Prediction.case_pk, func.max(Prediction.id).label("max_id"))
            .group_by(Prediction.case_pk)
            .subquery()
        )
        rows = (
            db.query(Case, Prediction)
            .join(latest, Case.id == latest.c.case_pk)
            .join(Prediction, Prediction.id == latest.c.max_id)
            .filter(Prediction.risk_level == "CRITICAL")
            .order_by(Prediction.risk_score.desc())
            .limit(5)
            .all()
        )
        if not rows:
            return {"answer": "No CRITICAL risk cases currently recorded in the active register." + _disclaimer()}
        lines = "\n".join(f"- **{c.case_id}** ({c.district}, {c.state}): Risk Score {p.risk_score}/100 · Delay: {p.expected_delay_days} days · Driver: {p.main_reason}" for c, p in rows)
        return {"answer": "Top 5 projects requiring **Immediate Critical Intervention**:\n" + lines + _disclaimer()}

    return {
        "answer": (
            "You can query specific case IDs (e.g. 'Why is LA-2026-00012 high risk?', 'What are the recommended actions?'), "
            "or ask executive queries like 'Which projects need immediate critical attention?'"
        )
        + _disclaimer()
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.api_host, port=settings.api_port, reload=False)
