from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd

try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from src.config import FEATURE_COLUMNS_PATH

HUMAN_LABELS = {
    "ownership_dispute": "Active ownership dispute",
    "legal_dispute": "Pending court litigation / legal dispute",
    "legal_case": "Pending court litigation",
    "number_of_legal_cases": "Multiple pending court cases",
    "litigation_intensity": "High litigation intensity",
    "compensation_paid": "Compensation pending disbursement",
    "compensation_status": "Compensation award pending",
    "compensation_delay_days": "Excessive compensation delay",
    "survey_completed": "Land survey incomplete",
    "survey_status": "Survey pending / contested",
    "demarcation_completed": "Boundary demarcation incomplete",
    "demarcation_status": "Demarcation contested",
    "approval_completed": "Statutory / administrative approval pending",
    "approval_status": "Pending department clearances",
    "approval_delay_days": "Prolonged approval review period",
    "document_completeness_pct": "Incomplete land records & documentation",
    "docs_incomplete_flag": "Severe document deficit (<75%)",
    "document_gap_score": "Missing title & mutation documents",
    "number_of_objections": "High volume of landowner objections",
    "high_objection_flag": "Critical objection load",
    "objections_per_landowner": "High objection concentration per owner",
    "objections_per_family": "High objection rate per affected family",
    "number_of_landowners": "Extensive number of affected titleholders",
    "affected_families": "Large number of affected project families",
    "land_area_hectares": "Large land parcel acreage",
    "historical_delay_rate_pct": "High historical delay track record in district",
    "distance_to_project_km": "Remoteness from project alignment",
    "pending_process_count": "Multiple simultaneous milestone process bottlenecks",
    "process_completion_rate": "Low overall milestone completion",
    "dispute_and_legal": "Compounded ownership disputes and legal proceedings",
    "complexity_index": "High overall project operational complexity",
    "delay_pipeline_days": "Cumulative administrative review backlog",
    "severe_bottleneck_flag": "Severe revenue administration bottleneck",
    "coordination_friction_flag": "Inter-department coordination friction",
    "stakeholder_resistance_flag": "Local community / stakeholder resistance",
    "administrative_bottleneck": "District administrative bottleneck",
    "inter_department_coordination": "Inter-agency coordination delay",
    "stakeholder_responsiveness": "Low community / stakeholder responsiveness",
    "rehabilitation_status": "Incomplete Rehabilitation & Resettlement (R&R)",
    "resettlement_status": "Pending resettlement infrastructure",
    "possession_status": "Physical possession obstructed",
    "notification_status": "Preliminary notification stage",
    "current_stage": "Early / complex acquisition lifecycle stage",
    "project_type": "Project sector baseline delay propensity",
    "state": "State-specific regulatory cycle",
    "district": "District administrative throughput",
}

POSITIVE_LABELS = {
    "compensation_paid": "Compensation disbursed to titleholders",
    "survey_completed": "Cadastral survey completed and verified",
    "demarcation_completed": "Field demarcation completed",
    "approval_completed": "Statutory approvals secured",
    "document_completeness_pct": "High document completeness & clear titles",
    "process_completion_rate": "Advanced milestone completion rate",
    "stakeholder_responsiveness": "High community alignment & cooperation",
    "inter_department_coordination": "Seamless inter-agency coordination",
    "administrative_bottleneck": "Minimal administrative friction",
    "number_of_objections": "Low objection count from landowners",
    "ownership_dispute": "Clear ownership with zero disputes",
    "legal_dispute": "Zero litigation or court stays",
    "pending_process_count": "Few pending milestone requirements",
}


def _base_feature(transformed_name: str) -> str:
    # ColumnTransformer prefixes: cat__state_Maharashtra or num__land_area_hectares
    name = transformed_name.split("__", 1)[-1]
    for prefix in (
        "state_",
        "district_",
        "project_type_",
        "notification_status_",
        "survey_status_",
        "demarcation_status_",
        "approval_status_",
        "compensation_status_",
        "possession_status_",
        "rehabilitation_status_",
        "resettlement_status_",
        "stakeholder_responsiveness_",
        "administrative_bottleneck_",
        "inter_department_coordination_",
        "current_stage_",
    ):
        if name.startswith(prefix):
            return prefix[:-1]
    return name


def _impact_label(value: float, max_abs: float) -> str:
    if max_abs <= 0:
        return "Low impact"
    ratio = abs(value) / max_abs
    if ratio >= 0.65:
        return "High impact"
    if ratio >= 0.30:
        return "Medium impact"
    return "Low impact"


def explain_prediction(transformed_row: np.ndarray, classifier) -> dict[str, Any]:
    """
    Compute TreeExplainer SHAP values.
    Returns:
    - factors_increasing_risk: top delay-increasing drivers
    - factors_reducing_risk: top delay-reducing drivers
    - human_explanation: executive plain-English summary
    - technical_explanation: exact SHAP metrics
    """
    values = None
    if _HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed_row)
            if isinstance(shap_values, list):
                values = np.array(shap_values[1][0])
            else:
                arr = np.array(shap_values)
                values = arr[0] if arr.ndim > 1 else arr
        except Exception:
            values = None

    if values is None:
        # Resilient tree-based feature attribution fallback
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        else:
            importances = np.ones(transformed_row.shape[1]) / float(transformed_row.shape[1])
        row_vec = transformed_row[0] if transformed_row.ndim > 1 else transformed_row
        values = row_vec * importances

    if not FEATURE_COLUMNS_PATH.exists():
        return {
            "factors_increasing_risk": [],
            "factors_reducing_risk": [],
            "reasons": [],
            "human_explanation": "Explanation models not yet trained.",
            "technical_explanation": {},
        }

    feature_info = json.loads(FEATURE_COLUMNS_PATH.read_text())
    names = feature_info["transformed_names"]

    contrib = pd.DataFrame({"feature": names, "shap": values})
    contrib["base"] = contrib["feature"].map(_base_feature)

    grouped = contrib.groupby("base", as_index=False)["shap"].sum()
    max_abs = float(grouped["shap"].abs().max()) if not grouped.empty else 1.0

    # 1. Factors increasing risk (shap > 0)
    inc_df = grouped[grouped["shap"] > 0].sort_values("shap", ascending=False)
    increasing = []
    for _, row in inc_df.head(5).iterrows():
        base = str(row["base"])
        val = float(row["shap"])
        increasing.append(
            {
                "factor": HUMAN_LABELS.get(base, base.replace("_", " ").title()),
                "feature": base,
                "impact": _impact_label(val, max_abs),
                "shap_value": round(val, 4),
            }
        )

    # 2. Factors reducing risk (shap < 0)
    red_df = grouped[grouped["shap"] < 0].sort_values("shap", ascending=True)
    reducing = []
    for _, row in red_df.head(5).iterrows():
        base = str(row["base"])
        val = float(row["shap"])
        reducing.append(
            {
                "factor": POSITIVE_LABELS.get(base, f"Favorable {base.replace('_', ' ')}"),
                "feature": base,
                "impact": _impact_label(abs(val), max_abs),
                "shap_value": round(val, 4),
            }
        )

    # Backward compatibility: reasons = increasing
    reasons = increasing if increasing else [{"factor": "No major delay risk driver detected", "impact": "Low impact", "shap_value": 0.0}]

    # 3. Human-readable narrative explanation
    if increasing:
        top_risk_names = [f"**{r['factor']}**" for r in increasing[:3]]
        risk_text = f"Primary delay risk is propelled by {', '.join(top_risk_names)}."
    else:
        risk_text = "Project exhibits standard procedural indicators with no major roadblock."

    if reducing:
        top_safe_names = [f"**{r['factor']}**" for r in reducing[:2]]
        safe_text = f" However, progress is stabilized by {', '.join(top_safe_names)}."
    else:
        safe_text = ""

    human_explanation = f"{risk_text}{safe_text}"

    # 4. Technical explanation
    technical = {
        "net_shap_log_odds_shift": round(float(grouped["shap"].sum()), 4),
        "total_risk_drivers_count": int(len(inc_df)),
        "total_protective_factors_count": int(len(red_df)),
        "dominant_feature": increasing[0]["feature"] if increasing else None,
        "max_driver_shap": increasing[0]["shap_value"] if increasing else 0.0,
    }

    return {
        "factors_increasing_risk": increasing,
        "factors_reducing_risk": reducing,
        "reasons": reasons,
        "human_explanation": human_explanation,
        "technical_explanation": technical,
    }


def format_reason_lines(reasons: list[dict[str, Any]]) -> list[str]:
    lines = []
    for i, item in enumerate(reasons, start=1):
        lines.append(f"{i}. {item['factor']} ({item['impact']})")
    return lines
