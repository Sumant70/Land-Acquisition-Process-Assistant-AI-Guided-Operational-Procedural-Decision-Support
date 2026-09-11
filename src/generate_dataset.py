"""
Generate an authoritative synthetic land-acquisition dataset covering all 28 States and 8 Union Territories.

Label: Synthetic Dataset for Prototype and Model Development (Smart India Hackathon).
These records are realistic simulations and are NOT real government cases.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DATA_RAW_DIR,
    DATASET_DISCLAIMER,
    LIFECYCLE_STAGES,
    PROJECT_TYPES,
    RANDOM_SEED,
    RAW_DATASET_PATH,
)
from src.location_master import get_all_locations, load_master_data

PROJECT_TYPE_BASE_DELAY = {
    "National Highway": 32,
    "Railway": 36,
    "Metro Rail": 45,
    "Irrigation & Canal": 28,
    "Power Transmission & Renewable": 24,
    "Industrial Corridor & SEZ": 48,
    "Airport": 42,
    "Defense & Strategic Road": 30,
    "Smart City Infrastructure": 34,
}

PROJECT_NAME_TEMPLATES = {
    "National Highway": ["NH-{num} Expressway Alignment", "NH-{num} 4-Laning Expansion", "NH-{num} Ring Road & Bypass", "NH-{num} Economic Corridor Link"],
    "Railway": ["DFC Railway Double Line Section {num}", "High Speed Rail Corridor Package {num}", "Gauge Conversion & Line Quadrupling {num}"],
    "Metro Rail": ["Metro Corridor Phase-{num} Section", "Airport Metro Express Line Extension {num}", "Metro Elevated Viaduct Package {num}"],
    "Irrigation & Canal": ["Major River Lift Irrigation Canal {num}", "Inter-Basin Water Transfer Canal {num}", "Barrage Command Area Network {num}"],
    "Power Transmission & Renewable": ["765kV Green Energy Corridor {num}", "Ultra-Mega Solar Park Substation Link {num}", "Inter-State Power Transmission Grid {num}"],
    "Industrial Corridor & SEZ": ["National Industrial Corridor Node {num}", "Multi-Modal Logistics Park Access {num}", "Special Economic Zone Phase-{num}"],
    "Airport": ["Greenfield International Airport Runway {num}", "Regional Airport Cargo Hub Expansion {num}", "Airport Terminal Connector Road {num}"],
    "Defense & Strategic Road": ["Border Strategic Highway Package {num}", "Defense Logistics Corridor Link {num}", "Strategic Tunnel Approach Road {num}"],
    "Smart City Infrastructure": ["Smart City Outer Arterial Ring {num}", "Urban Transit & Flyover Package {num}", "Riverfront Development Corridor {num}"],
}


def generate_dataset(n_rows: int = 5500, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = get_all_locations()
    if not locations:
        raise RuntimeError("Master location dataset returned 0 locations.")

    # Uniform or weighted choice across valid (state, district, lat, lon)
    loc_indices = rng.integers(0, len(locations), size=n_rows)
    selected_locs = [locations[i] for i in loc_indices]

    states = [loc[0] for loc in selected_locs]
    districts = [loc[1] for loc in selected_locs]
    base_lats = [loc[2] for loc in selected_locs]
    base_lons = [loc[3] for loc in selected_locs]

    # Add small centroid jitter (up to ~4-8 km) so points within the same district are slightly dispersed
    lats = [round(lat + float(rng.normal(0, 0.05)), 6) for lat in base_lats]
    lons = [round(lon + float(rng.normal(0, 0.05)), 6) for lon in base_lons]

    project_types = rng.choice(PROJECT_TYPES, size=n_rows)

    # Project names
    project_names = []
    for i in range(n_rows):
        ptype = project_types[i]
        templates = PROJECT_NAME_TEMPLATES.get(ptype, ["Infrastructure Project {num}"])
        tmpl = rng.choice(templates)
        num = rng.integers(101, 999)
        project_names.append(f"{tmpl.format(num=num)} - {districts[i]}")

    # Scale: land area and landowners
    land_area = np.round(rng.lognormal(mean=2.1, sigma=0.85, size=n_rows).clip(0.5, 350.0), 2)
    landowners = rng.integers(3, 200, size=n_rows)
    landowners = np.clip((landowners + (land_area * 1.1).astype(int)), 2, 450)
    affected_families = np.clip((landowners * rng.uniform(0.6, 1.4, size=n_rows)).astype(int), 1, 550)

    # Legal & ownership disputes
    ownership_dispute = rng.binomial(1, 0.20, size=n_rows)
    legal_dispute = rng.binomial(1, 0.14 + 0.38 * ownership_dispute, size=n_rows)
    legal_cases_count = np.where(legal_dispute == 1, rng.integers(1, 6, size=n_rows), 0)

    # Objections
    base_objections = rng.poisson(lam=3.2, size=n_rows)
    objections = np.clip(base_objections + (ownership_dispute * rng.integers(3, 15, size=n_rows)) + (legal_dispute * rng.integers(2, 8, size=n_rows)), 0, 75)

    # Documentation completeness (0 to 100)
    document_completeness = np.clip(rng.normal(84.0, 14.0, size=n_rows), 25.0, 100.0).round(1)

    # Historical delay rate in district/type
    historical_delay = np.clip(rng.normal(29.0, 11.0, size=n_rows), 4.0, 85.0).round(1)

    # Distance to project alignment (km)
    distance_km = np.clip(rng.gamma(2.4, 4.0, size=n_rows), 0.5, 60.0).round(2)

    # Milestone statuses
    survey_completed = rng.binomial(1, 0.76, size=n_rows)
    survey_status = np.where(survey_completed == 1, "Completed", rng.choice(["Pending", "In Progress", "Disputed"], p=[0.55, 0.35, 0.10], size=n_rows))

    demarcation_completed = np.where(survey_completed == 1, rng.binomial(1, 0.80, size=n_rows), rng.binomial(1, 0.15, size=n_rows))
    demarcation_status = np.where(demarcation_completed == 1, "Completed", rng.choice(["Pending", "Joint Verification Ongoing", "Boundary Objection"], p=[0.60, 0.25, 0.15], size=n_rows))

    approval_completed = rng.binomial(1, 0.68, size=n_rows)
    approval_status = np.where(approval_completed == 1, "Approved", rng.choice(["Pending Revenue Board", "Pending Forest/Env Clearance", "Under Scrutiny"], p=[0.50, 0.30, 0.20], size=n_rows))
    approval_delay_days = np.where(approval_completed == 1, 0, rng.integers(10, 180, size=n_rows))

    compensation_paid = np.where(approval_completed == 1, rng.binomial(1, 0.74, size=n_rows), rng.binomial(1, 0.18, size=n_rows))
    compensation_status = np.where(compensation_paid == 1, "Disbursed", rng.choice(["Pending Award Inquiry", "Funds Allocation Pending", "Disputed Accounts", "Disbursement In Progress"], p=[0.40, 0.25, 0.20, 0.15], size=n_rows))
    compensation_delay_days = np.where(compensation_paid == 1, 0, rng.integers(15, 240, size=n_rows))

    notification_status = rng.choice(["Preliminary Section 11 Issued", "Declaration Section 19 Published", "Final Award Section 23", "Notification Lapsed/Re-issued"], p=[0.35, 0.40, 0.20, 0.05], size=n_rows)

    rehabilitation_status = np.where(compensation_paid == 1, rng.choice(["Completed", "Resettlement Centers Ready", "Pending Land Allocation", "Not Applicable"], p=[0.45, 0.30, 0.20, 0.05], size=n_rows), rng.choice(["Pending", "Survey Ongoing", "Grievance Pending"], p=[0.60, 0.30, 0.10], size=n_rows))

    resettlement_status = np.where(rehabilitation_status == "Completed", "Completed", rng.choice(["In Progress", "Pending", "Objections Registered"], p=[0.45, 0.40, 0.15], size=n_rows))

    possession_status = np.where(
        (compensation_paid == 1) & (demarcation_completed == 1) & np.isin(rehabilitation_status, ["Completed", "Resettlement Centers Ready", "Not Applicable"]),
        rng.choice(["Possession Taken", "Partial Possession", "Notice Issued"], p=[0.65, 0.25, 0.10], size=n_rows),
        rng.choice(["Pending", "Resistance Encountered", "Encroachment Clearance Needed"], p=[0.60, 0.25, 0.15], size=n_rows),
    )

    # Governance & stakeholder ratings
    stakeholder_responsiveness = rng.choice(["High", "Medium", "Low", "Critical Resistance"], p=[0.38, 0.42, 0.15, 0.05], size=n_rows)
    administrative_bottleneck = rng.choice(["Low", "Moderate", "High", "Severe"], p=[0.45, 0.35, 0.15, 0.05], size=n_rows)
    inter_department_coordination = rng.choice(["Seamless", "Moderate Coordination", "Delayed Inter-Agency Responses", "Severe Coordination Breakdown"], p=[0.40, 0.38, 0.16, 0.06], size=n_rows)

    # Current lifecycle stage
    def assign_lifecycle_stage(comp_paid, app_comp, dem_comp, surv_comp, poss_stat):
        if poss_stat == "Possession Taken":
            return "Completion"
        if poss_stat == "Partial Possession":
            return "Possession"
        if comp_paid == 1:
            return "Rehabilitation & Resettlement"
        if app_comp == 1:
            return "Compensation"
        if dem_comp == 1:
            return "Approval"
        if surv_comp == 1:
            return "Objection Handling"
        return "Survey"

    current_stages = [
        assign_lifecycle_stage(compensation_paid[i], approval_completed[i], demarcation_completed[i], survey_completed[i], possession_status[i])
        for i in range(n_rows)
    ]

    # Start dates and timeline
    base_date = datetime(2023, 1, 1)
    start_dates = []
    comp_dates = []
    for i in range(n_rows):
        start_offset = int(rng.integers(0, 900))
        duration = int(rng.integers(180, 720))
        s_date = base_date + timedelta(days=start_offset)
        c_date = s_date + timedelta(days=duration)
        start_dates.append(s_date.strftime("%Y-%m-%d"))
        comp_dates.append(c_date.strftime("%Y-%m-%d"))

    # Causal Grounded Delay Logic (Zero Target Leakage)
    # Calibrated so that on-time projects (well-managed) have low delay (<75 days),
    # while complex disputed projects experience substantial delays (90 to 300+ days).
    doc_gap = (100.0 - document_completeness) / 100.0
    comp_gap = np.where(compensation_paid == 0, 1.0, 0.0)
    app_gap = np.where(approval_completed == 0, 1.0, 0.0)
    surv_gap = np.where(survey_completed == 0, 1.0, 0.0)
    dem_gap = np.where(demarcation_completed == 0, 1.0, 0.0)
    resp_factor = {"High": -15.0, "Medium": 5.0, "Low": 25.0, "Critical Resistance": 50.0}
    coord_factor = {"Seamless": -15.0, "Moderate Coordination": 5.0, "Delayed Inter-Agency Responses": 25.0, "Severe Coordination Breakdown": 45.0}
    bottle_factor = {"Low": -10.0, "Moderate": 5.0, "High": 20.0, "Severe": 38.0}

    delay_days = np.array([PROJECT_TYPE_BASE_DELAY[p] for p in project_types], dtype=float) - 25.0
    delay_days += 35.0 * ownership_dispute
    delay_days += 45.0 * legal_dispute + 5.0 * legal_cases_count
    delay_days += 32.0 * comp_gap + 0.12 * compensation_delay_days
    delay_days += 22.0 * app_gap + 0.10 * approval_delay_days
    delay_days += 20.0 * surv_gap + 15.0 * dem_gap
    delay_days += 30.0 * doc_gap
    delay_days += 1.0 * objections
    delay_days += 0.20 * np.log1p(landowners) * 10
    delay_days += 0.10 * land_area
    delay_days += 0.25 * (historical_delay - 25.0)
    delay_days += np.array([resp_factor.get(r, 0.0) for r in stakeholder_responsiveness])
    delay_days += np.array([coord_factor.get(c, 0.0) for c in inter_department_coordination])
    delay_days += np.array([bottle_factor.get(b, 0.0) for b in administrative_bottleneck])
    # Natural variance
    delay_days += rng.normal(0, 10.0, size=n_rows)
    delay_days = np.clip(delay_days, 0, 365).round(0).astype(int)

    # Delay classification threshold: 75 days delay considered delayed
    delayed = (delay_days >= 75).astype(int)
    # Flip minor fraction (3%) for real-world unpredictability
    flip = rng.random(n_rows) < 0.03
    delayed = np.where(flip, 1 - delayed, delayed)

    # Overall operational status
    def get_status(delayed_flag, poss_stat, comp_paid, surv_comp):
        if poss_stat == "Possession Taken":
            return "Completed"
        if comp_paid == 0 and delayed_flag == 1:
            return "Compensation Pending"
        if surv_comp == 0:
            return "Survey Pending"
        if delayed_flag == 1:
            return "Objection Pending"
        return "Initiated"

    statuses = [get_status(delayed[i], possession_status[i], compensation_paid[i], survey_completed[i]) for i in range(n_rows)]
    days_open = np.clip(delay_days + rng.integers(-15, 60, size=n_rows), 10, 600)

    # Small realistic missingness in raw data (1-2%) to test data pipeline robustness
    doc_missing = rng.random(n_rows) < 0.025
    document_completeness[doc_missing] = np.nan
    hist_missing = rng.random(n_rows) < 0.015
    historical_delay[hist_missing] = np.nan

    df = pd.DataFrame(
        {
            "case_id": [f"LA-2026-{i+1:05d}" for i in range(n_rows)],
            "project_name": project_names,
            "state": states,
            "district": districts,
            "project_type": project_types,
            "land_area_hectares": land_area,
            "affected_families": affected_families,
            "number_of_landowners": landowners,
            "ownership_dispute": ownership_dispute,
            "legal_dispute": legal_dispute,
            "legal_case": legal_dispute,  # Backward-compatible alias
            "number_of_legal_cases": legal_cases_count,
            "number_of_objections": objections,
            "document_completeness_pct": document_completeness,
            "notification_status": notification_status,
            "survey_status": survey_status,
            "survey_completed": survey_completed,
            "demarcation_status": demarcation_status,
            "demarcation_completed": demarcation_completed,
            "approval_status": approval_status,
            "approval_completed": approval_completed,
            "approval_delay_days": approval_delay_days,
            "compensation_status": compensation_status,
            "compensation_paid": compensation_paid,
            "compensation_delay_days": compensation_delay_days,
            "possession_status": possession_status,
            "rehabilitation_status": rehabilitation_status,
            "resettlement_status": resettlement_status,
            "stakeholder_responsiveness": stakeholder_responsiveness,
            "administrative_bottleneck": administrative_bottleneck,
            "inter_department_coordination": inter_department_coordination,
            "historical_delay_rate_pct": historical_delay,
            "distance_to_project_km": distance_km,
            "project_start_date": start_dates,
            "expected_completion_date": comp_dates,
            "current_stage": current_stages,
            "status": statuses,
            "days_open": days_open,
            "latitude": lats,
            "longitude": lons,
            "expected_delay_days": delay_days,
            "delayed": delayed,
            "coordinate_source": "Synthetic prototype coordinates (district centroid + jitter)",
            "dataset_label": DATASET_DISCLAIMER,
        }
    )

    # Inject a tiny set of duplicate records to test cleaning stage
    dup_count = 15
    dup_idx = rng.integers(0, n_rows, size=dup_count)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic land-acquisition cases across all 36 States/UTs.")
    parser.add_argument("--n", type=int, default=5500, help="Number of unique cases.")
    parser.add_argument("--out", type=Path, default=RAW_DATASET_PATH)
    args = parser.parse_args()

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_rows=args.n)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} rows across {df['state'].nunique()} States/UTs and {df['district'].nunique()} districts.")
    print(f"Saved to {args.out}")
    print("Delayed distribution:\n", df["delayed"].value_counts(normalize=True).round(3))
    print("Expected delay days mean:", round(df["expected_delay_days"].mean(), 1), "median:", df["expected_delay_days"].median())


if __name__ == "__main__":
    main()
