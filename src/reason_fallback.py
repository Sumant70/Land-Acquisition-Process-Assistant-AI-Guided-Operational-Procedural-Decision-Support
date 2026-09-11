from __future__ import annotations

from typing import Any


def heuristic_main_reason(case: dict[str, Any]) -> str:
    """Fast reason for bulk scoring. Live /predict uses SHAP instead."""
    scored: list[tuple[int, str]] = []
    if int(case.get("ownership_dispute") or 0):
        scored.append((90, "Ownership dispute"))
    if int(case.get("legal_case") or 0):
        scored.append((88, "Pending legal case"))
    if int(case.get("compensation_paid") or 0) == 0:
        scored.append((80, "Compensation not paid / pending"))
    if int(case.get("survey_completed") or 0) == 0:
        scored.append((75, "Land survey incomplete"))
    if int(case.get("approval_completed") or 0) == 0:
        scored.append((72, "Approval pending"))
    if float(case.get("document_completeness_pct") or 100) < 70:
        scored.append((78, "Incomplete documents"))
    if int(case.get("number_of_objections") or 0) >= 8:
        scored.append((70, "High number of objections"))
    if int(case.get("demarcation_completed") or 0) == 0:
        scored.append((60, "Demarcation incomplete"))
    if int(case.get("number_of_landowners") or 0) >= 80:
        scored.append((55, "Large number of landowners"))
    if not scored:
        return "Routine process monitoring"
    scored.sort(reverse=True)
    return scored[0][1]
