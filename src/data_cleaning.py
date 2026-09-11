"""Clean and validate the synthetic land-acquisition dataset with strict data quality verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    CLEAN_DATASET_PATH,
    DATA_PROCESSED_DIR,
    NUMERIC_COLUMNS,
    RAW_DATASET_PATH,
)
from src.location_master import validate_location

REQUIRED_COLUMNS = [
    "case_id",
    "project_name",
    "state",
    "district",
    "project_type",
    "land_area_hectares",
    "affected_families",
    "number_of_landowners",
    "ownership_dispute",
    "legal_dispute",
    "number_of_legal_cases",
    "number_of_objections",
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
    "historical_delay_rate_pct",
    "distance_to_project_km",
    "current_stage",
    "status",
    "days_open",
    "latitude",
    "longitude",
    "expected_delay_days",
    "delayed",
]


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            median = out[col].median()
            out[col] = out[col].fillna(median)
    for col in BOOLEAN_COLUMNS:
        if col in out.columns:
            mode = out[col].mode(dropna=True)
            fill = int(mode.iloc[0]) if len(mode) else 0
            out[col] = out[col].fillna(fill)
    for col in CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("Unknown")
    return out


def drop_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates(subset=["case_id"], keep="first")
    return out, before - len(out)


def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip extreme numeric values using IQR, without dropping cases."""
    out = df.copy()
    clip_cols = [
        "land_area_hectares",
        "affected_families",
        "number_of_landowners",
        "number_of_objections",
        "approval_delay_days",
        "compensation_delay_days",
        "historical_delay_rate_pct",
        "distance_to_project_km",
        "expected_delay_days",
        "days_open",
    ]
    for col in clip_cols:
        if col not in out.columns:
            continue
        q1 = out[col].quantile(0.25)
        q3 = out[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 3 * iqr
        upper = q3 + 3 * iqr
        out[col] = out[col].clip(lower=max(lower, 0), upper=upper)
    return out


def validate_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out["document_completeness_pct"] = out["document_completeness_pct"].clip(0.0, 100.0)
    out["historical_delay_rate_pct"] = out["historical_delay_rate_pct"].clip(0.0, 100.0)

    for col in BOOLEAN_COLUMNS:
        out[col] = out[col].astype(int).clip(0, 1)

    out["delayed"] = out["delayed"].astype(int).clip(0, 1)
    out["number_of_landowners"] = out["number_of_landowners"].clip(lower=1)
    out["affected_families"] = out["affected_families"].clip(lower=1)
    out["land_area_hectares"] = out["land_area_hectares"].clip(lower=0.01)
    out["number_of_objections"] = out["number_of_objections"].clip(lower=0)
    out["approval_delay_days"] = out["approval_delay_days"].clip(lower=0)
    out["compensation_delay_days"] = out["compensation_delay_days"].clip(lower=0)
    out["expected_delay_days"] = out["expected_delay_days"].clip(lower=0)

    # Sync backward-compatible legal_case
    out["legal_case"] = out["legal_dispute"]

    # Validate state-district coherence
    invalid_locations = []
    for idx, row in out.iterrows():
        st = str(row["state"])
        dt = str(row["district"])
        if not validate_location(st, dt):
            invalid_locations.append((idx, st, dt))

    report = {
        "total_rows": len(out),
        "valid_location_pairs": len(out) - len(invalid_locations),
        "invalid_locations_count": len(invalid_locations),
        "columns_validated": len(REQUIRED_COLUMNS),
        "target_delayed_share": round(float(out["delayed"].mean()), 3),
        "avg_expected_delay_days": round(float(out["expected_delay_days"].mean()), 1),
    }
    return out, report


def clean_dataset(raw_path: Path = RAW_DATASET_PATH) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(raw_path)
    n_raw = len(df)
    df, n_dups = drop_duplicates(df)
    df = handle_missing_values(df)
    df = detect_outliers(df)
    df, quality_report = validate_data(df)
    quality_report["raw_rows"] = n_raw
    quality_report["duplicates_removed"] = n_dups
    quality_report["clean_rows"] = len(df)
    quality_report["remaining_missing_cells"] = int(df.isna().sum().sum())

    print(f"Raw rows: {n_raw} | Clean rows: {len(df)} | Duplicates removed: {n_dups}")
    print(f"Location validation: {quality_report['valid_location_pairs']}/{len(df)} 100% valid state-district pairs.")
    print(f"Data quality report: {quality_report}")
    return df, quality_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=CLEAN_DATASET_PATH)
    args = parser.parse_args()
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df, report = clean_dataset(args.input)
    df.to_csv(args.output, index=False)
    report_path = DATA_PROCESSED_DIR / "data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote cleaned dataset to {args.output} and quality report to {report_path}")


if __name__ == "__main__":
    main()
