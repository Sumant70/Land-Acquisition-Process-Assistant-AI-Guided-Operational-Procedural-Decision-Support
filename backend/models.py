from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(30), default="officer")  # admin | officer | analyst | viewer
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    project_name: Mapped[str] = mapped_column(String(200), default="", index=True)
    state: Mapped[str] = mapped_column(String(80), index=True)
    district: Mapped[str] = mapped_column(String(80), index=True)
    project_type: Mapped[str] = mapped_column(String(80), index=True)
    land_area_hectares: Mapped[float] = mapped_column(Float)
    affected_families: Mapped[int] = mapped_column(Integer, default=1)
    number_of_landowners: Mapped[int] = mapped_column(Integer)
    ownership_dispute: Mapped[int] = mapped_column(Integer, default=0)
    legal_dispute: Mapped[int] = mapped_column(Integer, default=0)
    legal_case: Mapped[int] = mapped_column(Integer, default=0)
    number_of_legal_cases: Mapped[int] = mapped_column(Integer, default=0)
    document_completeness_pct: Mapped[float] = mapped_column(Float)
    notification_status: Mapped[str] = mapped_column(String(80), default="Preliminary Section 11 Issued")
    survey_status: Mapped[str] = mapped_column(String(80), default="Completed")
    survey_completed: Mapped[int] = mapped_column(Integer, default=0)
    demarcation_status: Mapped[str] = mapped_column(String(80), default="Completed")
    demarcation_completed: Mapped[int] = mapped_column(Integer, default=0)
    approval_status: Mapped[str] = mapped_column(String(80), default="Approved")
    approval_completed: Mapped[int] = mapped_column(Integer, default=0)
    approval_delay_days: Mapped[int] = mapped_column(Integer, default=0)
    compensation_status: Mapped[str] = mapped_column(String(80), default="Disbursed")
    compensation_paid: Mapped[int] = mapped_column(Integer, default=0)
    compensation_delay_days: Mapped[int] = mapped_column(Integer, default=0)
    possession_status: Mapped[str] = mapped_column(String(80), default="Pending")
    rehabilitation_status: Mapped[str] = mapped_column(String(80), default="Completed")
    resettlement_status: Mapped[str] = mapped_column(String(80), default="Completed")
    stakeholder_responsiveness: Mapped[str] = mapped_column(String(60), default="High")
    administrative_bottleneck: Mapped[str] = mapped_column(String(60), default="Low")
    inter_department_coordination: Mapped[str] = mapped_column(String(60), default="Seamless")
    number_of_objections: Mapped[int] = mapped_column(Integer, default=0)
    historical_delay_rate_pct: Mapped[float] = mapped_column(Float, default=0)
    distance_to_project_km: Mapped[float] = mapped_column(Float, default=0)
    current_stage: Mapped[str] = mapped_column(String(80), default="Survey", index=True)
    status: Mapped[str] = mapped_column(String(40), default="Initiated", index=True)
    days_open: Mapped[int] = mapped_column(Integer, default=0)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    coordinate_source: Mapped[str] = mapped_column(String(200), default="Synthetic prototype coordinates")
    remarks: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="case", cascade="all, delete-orphan")
    actions = relationship("Action", back_populates="case", cascade="all, delete-orphan")
    remarks_log = relationship("CaseRemark", back_populates="case", cascade="all, delete-orphan")
    timeline = relationship("CaseTimeline", back_populates="case", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    delay_probability: Mapped[float] = mapped_column(Float)
    no_delay_probability: Mapped[float] = mapped_column(Float)
    risk_score: Mapped[int] = mapped_column(Integer, default=0, index=True)
    risk_level: Mapped[str] = mapped_column(String(20), index=True)
    expected_delay_days: Mapped[int] = mapped_column(Integer)
    main_reason: Mapped[str] = mapped_column(String(200), default="")
    reasons_json: Mapped[str] = mapped_column(Text, default="[]")
    factors_reducing_json: Mapped[str] = mapped_column(Text, default="[]")
    human_explanation: Mapped[str] = mapped_column(Text, default="")
    priority_score: Mapped[int] = mapped_column(Integer, default=0)
    priority_label: Mapped[str] = mapped_column(String(80), default="")
    model_version: Mapped[str] = mapped_column(String(40), default="sih-v2")

    case = relationship("Case", back_populates="predictions")


class Action(Base):
    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    code: Mapped[str] = mapped_column(String(60))
    issue: Mapped[str] = mapped_column(String(255), default="")
    action: Mapped[str] = mapped_column(Text)
    priority: Mapped[str] = mapped_column(String(20), default="Medium")
    department: Mapped[str] = mapped_column(String(120), default="Competent Authority")
    timeline: Mapped[str] = mapped_column(String(80), default="Within 15 days")
    status: Mapped[str] = mapped_column(String(20), default="recommended")  # recommended | completed
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="actions")


class CaseTimeline(Base):
    __tablename__ = "case_timeline"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    stage_name: Mapped[str] = mapped_column(String(80), index=True)
    status: Mapped[str] = mapped_column(String(30), default="Pending")  # Completed | In Progress | Pending | Delayed
    bottleneck_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="timeline")


class CaseRemark(Base):
    __tablename__ = "case_remarks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_pk: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    officer_username: Mapped[str] = mapped_column(String(80))
    remark: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    case = relationship("Case", back_populates="remarks_log")


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[str] = mapped_column(String(40), index=True)
    alert_type: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    severity: Mapped[str] = mapped_column(String(20), default="HIGH")
    is_read: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user: Mapped[str] = mapped_column(String(80), index=True)
    action: Mapped[str] = mapped_column(String(100))
    case_id: Mapped[str] = mapped_column(String(40), index=True)
    previous_value: Mapped[str] = mapped_column(Text, default="")
    new_value: Mapped[str] = mapped_column(Text, default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
