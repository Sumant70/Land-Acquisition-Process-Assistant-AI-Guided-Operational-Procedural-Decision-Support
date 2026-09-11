"""Load trained models and score a single case or a dataframe with calibrated risk scoring."""

from __future__ import annotations

from typing import Any
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BOOLEAN_COLUMNS,
    CATEGORICAL_COLUMNS,
    CLASSIFIER_PATH,
    NUMERIC_COLUMNS,
    PREPROCESSOR_PATH,
    REGRESSOR_PATH,
)
from src.feature_engineering import add_engineered_features, feature_matrix


def compute_risk_score(delay_probability: float, expected_delay_days: float) -> tuple[int, str]:
    """
    Compute a calibrated 0-100 Risk Score combining delay probability and expected delay days.
    0-25: LOW
    26-50: MEDIUM
    51-75: HIGH
    76-100: CRITICAL
    """
    p_component = float(delay_probability) * 100.0
    # 180 days is considered severe delay in major infrastructure milestones
    d_component = min(100.0, (float(max(0.0, expected_delay_days)) / 180.0) * 100.0)
    raw_score = 0.60 * p_component + 0.40 * d_component
    score = int(round(np.clip(raw_score, 0.0, 100.0)))

    if score >= 76:
        category = "CRITICAL"
    elif score >= 51:
        category = "HIGH"
    elif score >= 26:
        category = "MEDIUM"
    else:
        category = "LOW"
    return score, category


def risk_level_from_probability(p: float) -> str:
    if p >= 0.75:
        return "CRITICAL"
    if p >= 0.50:
        return "HIGH"
    if p >= 0.25:
        return "MEDIUM"
    return "LOW"


def load_artifacts():
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classifier = joblib.load(CLASSIFIER_PATH)
    regressor = joblib.load(REGRESSOR_PATH)
    return preprocessor, classifier, regressor


def prepare_frame(payload: dict[str, Any] | pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame([payload]) if isinstance(payload, dict) else payload.copy()

    # Ensure all expected columns exist with defaults if scoring an ad-hoc single case
    for col in BOOLEAN_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(int)
        else:
            df[col] = 0

    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown"

    featured = add_engineered_features(df)
    return feature_matrix(featured)


def predict_case(payload: dict[str, Any], artifacts=None) -> dict[str, Any]:
    preprocessor, classifier, regressor = artifacts or load_artifacts()
    frame = prepare_frame(payload)
    X = preprocessor.transform(frame)

    delay_probability = float(classifier.predict_proba(X)[0, 1])
    no_delay_probability = 1.0 - delay_probability
    expected_delay_days = float(max(0.0, regressor.predict(X)[0]))

    risk_score, risk_category = compute_risk_score(delay_probability, expected_delay_days)

    return {
        "delay_probability": round(delay_probability, 4),
        "no_delay_probability": round(no_delay_probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_category,
        "risk_category": risk_category,
        "expected_delay_days": int(round(expected_delay_days)),
        "prepared_row": frame.iloc[0].to_dict(),
        "transformed": X,
    }


def predict_frame(df: pd.DataFrame, artifacts=None) -> pd.DataFrame:
    preprocessor, classifier, regressor = artifacts or load_artifacts()
    frame = prepare_frame(df)
    X = preprocessor.transform(frame)

    probs = classifier.predict_proba(X)[:, 1]
    delays = regressor.predict(X)

    out = df.copy()
    out["delay_probability"] = probs.round(4)
    out["no_delay_probability"] = (1.0 - probs).round(4)

    scores = []
    categories = []
    pred_delays = []
    for p, d in zip(probs, delays):
        d_val = int(round(max(0.0, d)))
        score, cat = compute_risk_score(p, d_val)
        scores.append(score)
        categories.append(cat)
        pred_delays.append(d_val)

    out["risk_score"] = scores
    out["risk_level"] = categories
    out["risk_category"] = categories
    out["predicted_delay_days"] = pred_delays
    return out
