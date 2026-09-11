from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    full_name: str


class CaseBase(BaseModel):
    project_name: Optional[str] = "Infrastructure Development Package"
    state: str
    district: str
    project_type: str
    land_area_hectares: float = Field(gt=0, le=500, default=10.0)
    affected_families: int = Field(ge=1, le=1000, default=25)
    number_of_landowners: int = Field(ge=1, le=1000, default=30)
    ownership_dispute: int = Field(ge=0, le=1, default=0)
    legal_dispute: int = Field(ge=0, le=1, default=0)
    legal_case: Optional[int] = Field(ge=0, le=1, default=0)
    number_of_legal_cases: int = Field(ge=0, le=20, default=0)
    document_completeness_pct: float = Field(ge=0, le=100, default=85.0)
    notification_status: str = "Declaration Section 19 Published"
    survey_status: str = "Completed"
    survey_completed: int = Field(ge=0, le=1, default=1)
    demarcation_status: str = "Completed"
    demarcation_completed: int = Field(ge=0, le=1, default=1)
    approval_status: str = "Approved"
    approval_completed: int = Field(ge=0, le=1, default=1)
    approval_delay_days: int = Field(ge=0, le=500, default=0)
    compensation_status: str = "Disbursed"
    compensation_paid: int = Field(ge=0, le=1, default=1)
    compensation_delay_days: int = Field(ge=0, le=500, default=0)
    possession_status: str = "Pending"
    rehabilitation_status: str = "Completed"
    resettlement_status: str = "Completed"
    stakeholder_responsiveness: str = "High"
    administrative_bottleneck: str = "Low"
    inter_department_coordination: str = "Seamless"
    number_of_objections: int = Field(ge=0, le=200, default=2)
    historical_delay_rate_pct: float = Field(ge=0, le=100, default=25.0)
    distance_to_project_km: float = Field(ge=0, le=200, default=5.0)
    current_stage: str = "Compensation"
    status: str = "Initiated"
    days_open: int = 30
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class CaseCreate(CaseBase):
    case_id: Optional[str] = None
    remarks: str = ""


class CaseUpdate(BaseModel):
    project_name: Optional[str] = None
    status: Optional[str] = None
    current_stage: Optional[str] = None
    remarks: Optional[str] = None
    survey_completed: Optional[int] = Field(default=None, ge=0, le=1)
    survey_status: Optional[str] = None
    demarcation_completed: Optional[int] = Field(default=None, ge=0, le=1)
    demarcation_status: Optional[str] = None
    compensation_paid: Optional[int] = Field(default=None, ge=0, le=1)
    compensation_status: Optional[str] = None
    compensation_delay_days: Optional[int] = Field(default=None, ge=0)
    approval_completed: Optional[int] = Field(default=None, ge=0, le=1)
    approval_status: Optional[str] = None
    approval_delay_days: Optional[int] = Field(default=None, ge=0)
    ownership_dispute: Optional[int] = Field(default=None, ge=0, le=1)
    legal_dispute: Optional[int] = Field(default=None, ge=0, le=1)
    legal_case: Optional[int] = Field(default=None, ge=0, le=1)
    number_of_legal_cases: Optional[int] = Field(default=None, ge=0)
    document_completeness_pct: Optional[float] = Field(default=None, ge=0, le=100)
    number_of_objections: Optional[int] = Field(default=None, ge=0, le=200)
    rehabilitation_status: Optional[str] = None
    resettlement_status: Optional[str] = None
    possession_status: Optional[str] = None
    stakeholder_responsiveness: Optional[str] = None
    administrative_bottleneck: Optional[str] = None
    inter_department_coordination: Optional[str] = None


class RemarkCreate(BaseModel):
    remark: str = Field(min_length=2, max_length=2000)


class PredictRequest(CaseBase):
    case_id: Optional[str] = None


class WhatIfRequest(BaseModel):
    case_id: Optional[str] = None
    base_case: Optional[dict[str, Any]] = None
    # Simulated modifications
    compensation_paid: Optional[int] = None
    compensation_status: Optional[str] = None
    compensation_delay_days: Optional[int] = None
    document_completeness_pct: Optional[float] = None
    legal_dispute: Optional[int] = None
    ownership_dispute: Optional[int] = None
    approval_completed: Optional[int] = None
    approval_status: Optional[str] = None
    approval_delay_days: Optional[int] = None
    number_of_objections: Optional[int] = None
    survey_completed: Optional[int] = None
    rehabilitation_status: Optional[str] = None
    stakeholder_responsiveness: Optional[str] = None
    inter_department_coordination: Optional[str] = None
    administrative_bottleneck: Optional[str] = None


class WhatIfResponse(BaseModel):
    case_id: str
    original: dict[str, Any]
    simulated: dict[str, Any]
    risk_score_delta: int
    delay_days_delta: int
    delay_probability_delta: float
    summary: str


class TimelineEventOut(BaseModel):
    stage_name: str
    status: str
    bottleneck_flag: bool
    notes: str


ALLOWED_STATUSES = {
    "Initiated",
    "Survey Pending",
    "Survey Completed",
    "Objection Pending",
    "Approval Pending",
    "Compensation Pending",
    "Possession Pending",
    "Completed",
}
