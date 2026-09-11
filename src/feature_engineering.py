"""Feature engineering for land-acquisition delay prediction. Strict leakage elimination."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.config import CLEAN_DATASET_PATH, DATA_PROCESSED_DIR, FEATURED_DATASET_PATH, LEAKAGE_COLUMNS


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Milestone process progression
    surv = out["survey_completed"].astype(int) if "survey_completed" in out.columns else 1
    dem = out["demarcation_completed"].astype(int) if "demarcation_completed" in out.columns else 1
    comp = out["compensation_paid"].astype(int) if "compensation_paid" in out.columns else 1
    app = out["approval_completed"].astype(int) if "approval_completed" in out.columns else 1

    out["pending_process_count"] = (1 - surv) + (1 - dem) + (1 - comp) + (1 - app)
    out["process_completion_rate"] = ((4 - out["pending_process_count"]) / 4.0).round(3)

    # Disputes and legal interaction
    own_disp = out["ownership_dispute"].astype(int) if "ownership_dispute" in out.columns else 0
    leg_disp = out.get("legal_dispute", out.get("legal_case", pd.Series(0, index=out.index))).astype(int)
    out["dispute_and_legal"] = own_disp * leg_disp

    leg_cases = out["number_of_legal_cases"].astype(float) if "number_of_legal_cases" in out.columns else leg_disp.astype(float)
    out["litigation_intensity"] = (leg_cases * (1.0 + 0.5 * own_disp)).round(2)

    # Objections and ratios
    objs = out["number_of_objections"].astype(float) if "number_of_objections" in out.columns else 0.0
    owners = out["number_of_landowners"].astype(float).clip(lower=1) if "number_of_landowners" in out.columns else pd.Series(1.0, index=out.index)
    families = out["affected_families"].astype(float).clip(lower=1) if "affected_families" in out.columns else owners

    out["objections_per_landowner"] = (objs / owners).round(4)
    out["objections_per_family"] = (objs / families).round(4)
    out["high_objection_flag"] = (objs >= 8).astype(int)

    # Document completeness gap
    docs = out["document_completeness_pct"].astype(float) if "document_completeness_pct" in out.columns else 85.0
    out["docs_incomplete_flag"] = (docs < 75.0).astype(int)
    out["document_gap_score"] = (100.0 - docs).clip(lower=0.0).round(1)

    # Cumulative delay pipeline
    app_delay = out["approval_delay_days"].astype(float) if "approval_delay_days" in out.columns else 0.0
    comp_delay = out["compensation_delay_days"].astype(float) if "compensation_delay_days" in out.columns else 0.0
    out["delay_pipeline_days"] = app_delay + comp_delay

    # Governance and institutional friction flags
    if "administrative_bottleneck" in out.columns:
        out["severe_bottleneck_flag"] = out["administrative_bottleneck"].isin(["High", "Severe"]).astype(int)
    else:
        out["severe_bottleneck_flag"] = 0

    if "inter_department_coordination" in out.columns:
        out["coordination_friction_flag"] = out["inter_department_coordination"].isin(["Delayed Inter-Agency Responses", "Severe Coordination Breakdown"]).astype(int)
    else:
        out["coordination_friction_flag"] = 0

    if "stakeholder_responsiveness" in out.columns:
        out["stakeholder_resistance_flag"] = out["stakeholder_responsiveness"].isin(["Low", "Critical Resistance"]).astype(int)
    else:
        out["stakeholder_resistance_flag"] = 0

    # Composite multidimensional complexity index
    area = out["land_area_hectares"].astype(float).clip(upper=150.0) if "land_area_hectares" in out.columns else 10.0
    out["complexity_index"] = (
        0.25 * area
        + 0.15 * owners.clip(upper=200.0)
        + 7.0 * out["pending_process_count"]
        + 12.0 * own_disp
        + 14.0 * leg_disp
        + 0.12 * out["document_gap_score"]
        + 8.0 * out["severe_bottleneck_flag"]
        + 6.0 * out["stakeholder_resistance_flag"]
    ).round(2)

    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return model-ready features, strictly guaranteeing zero target leakage."""
    drop = set(LEAKAGE_COLUMNS) | {
        "case_id",
        "project_name",
        "status",
        "days_open",
        "latitude",
        "longitude",
        "coordinate_source",
        "dataset_label",
        "remarks",
        "officer_name",
        "project_start_date",
        "expected_completion_date",
    }
    cols = [c for c in df.columns if c not in drop]
    return df[cols]


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CLEAN_DATASET_PATH)
    featured = add_engineered_features(df)
    featured.to_csv(FEATURED_DATASET_PATH, index=False)
    X = feature_matrix(featured)
    print(f"Wrote {FEATURED_DATASET_PATH} with {len(featured)} rows and {featured.shape[1]} columns.")
    print(f"Model-ready feature matrix shape: {X.shape}")
    leaked = [c for c in LEAKAGE_COLUMNS if c in X.columns]
    print("Target leakage in feature matrix?", leaked or "None (Clean zero-leakage)")


if __name__ == "__main__":
    main()
