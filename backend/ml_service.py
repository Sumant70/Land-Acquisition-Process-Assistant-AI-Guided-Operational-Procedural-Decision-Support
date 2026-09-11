from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import explain_prediction, format_reason_lines
from src.predict import compute_risk_score, load_artifacts, predict_case
from src.recommendations import build_recommendations, generate_alerts, priority_score

_ARTIFACTS = None


def get_ml_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts()
    return _ARTIFACTS


def reload_ml_artifacts():
    global _ARTIFACTS
    _ARTIFACTS = load_artifacts()
    return _ARTIFACTS


def score_case(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = get_ml_artifacts()
    result = predict_case(payload, artifacts=artifacts)
    explanation = explain_prediction(result["transformed"], artifacts[1])

    reasons = explanation.get("reasons", [])
    factors_increasing = explanation.get("factors_increasing_risk", [])
    factors_reducing = explanation.get("factors_reducing_risk", [])
    human_explanation = explanation.get("human_explanation", "")
    technical_explanation = explanation.get("technical_explanation", {})

    recs = build_recommendations(payload, reasons)
    score, label = priority_score(
        result["risk_level"],
        payload,
        result["delay_probability"],
        result["expected_delay_days"],
    )
    alerts = generate_alerts(payload, result["risk_level"], result["expected_delay_days"])
    main_reason = factors_increasing[0]["factor"] if factors_increasing else (reasons[0]["factor"] if reasons else "Routine operational monitoring")

    return {
        "delay_probability": result["delay_probability"],
        "no_delay_probability": result["no_delay_probability"],
        "risk_score": result.get("risk_score", score),
        "risk_level": result["risk_level"],
        "risk_category": result["risk_level"],
        "expected_delay_days": result["expected_delay_days"],
        "reasons": reasons,
        "factors_increasing_risk": factors_increasing,
        "factors_reducing_risk": factors_reducing,
        "reason_lines": format_reason_lines(reasons),
        "human_explanation": human_explanation,
        "technical_explanation": technical_explanation,
        "main_reason": main_reason,
        "recommendations": recs,
        "priority_score": score,
        "priority_label": label,
        "alerts": alerts,
        "disclaimer": "Decision-support output only. Not a legal or administrative order. Trained on a synthetic prototype dataset.",
    }
