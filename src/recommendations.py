"""
Structured Recommendation & Early Warning Engine for Land Acquisition Decision Support.
Produces 5-attribute actionable interventions according to statutory milestones.
"""

from __future__ import annotations

from typing import Any


def build_recommendations(case: dict[str, Any], reasons: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []

    # 1. Legal Dispute & Litigation
    has_legal = int(case.get("legal_dispute", case.get("legal_case", 0)) or 0) == 1
    if has_legal:
        recs.append(
            {
                "code": "LEGAL_DISPUTE",
                "issue": "Pending litigation or court stay petition active",
                "action": "Convene emergency legal cell coordination with Government Pleader to file counter-affidavit and vacate interim stays before the next milestone.",
                "priority": "High",
                "department": "Legal Cell & District Collectorate",
                "timeline": "Immediate (within 7 days)",
            }
        )

    # 2. Ownership Dispute
    if int(case.get("ownership_dispute") or 0) == 1:
        recs.append(
            {
                "code": "OWNERSHIP_DISPUTE",
                "issue": "Contested title or inheritance mutation dispute among co-sharers",
                "action": "Refer disputed apportionment to Special Land Acquisition Officer (SLAO) summary enquiry; deposit disputed shares in escrow under Section 77(2) to prevent work stalling.",
                "priority": "High",
                "department": "Revenue & Land Records Cell",
                "timeline": "Within 10–14 days",
            }
        )

    # 3. Compensation Status
    comp_paid = int(case.get("compensation_paid", 1) or 0)
    comp_stat = str(case.get("compensation_status", "Disbursed"))
    comp_delay = int(case.get("compensation_delay_days", 0) or 0)
    if comp_paid == 0 or comp_stat not in ["Disbursed", "Completed"]:
        prio = "High" if comp_delay >= 45 else "Medium"
        recs.append(
            {
                "code": "COMPENSATION_PENDING",
                "issue": f"Compensation disbursement backlog ({comp_stat} - {comp_delay} days delayed)",
                "action": "Organize dedicated camp-mode award disbursement; verify bank account mutations and execute direct benefit transfer (DBT) to eligible titleholders.",
                "priority": prio,
                "department": "Competent Authority for Land Acquisition (CALA) / Treasury",
                "timeline": "Within 15 days",
            }
        )

    # 4. Document Completeness
    completeness = float(case.get("document_completeness_pct") or 85.0)
    if completeness < 75.0:
        recs.append(
            {
                "code": "DOCUMENTS_INCOMPLETE",
                "issue": f"Severe land record documentation deficit (Current: {completeness:.1f}%)",
                "action": "Deploy rapid revenue village task-force with digitized RoR (Record of Rights) and cadastral maps to clear missing khatiyan/patta records.",
                "priority": "High",
                "department": "Tehsil / Taluk Revenue Office",
                "timeline": "Within 15 days",
            }
        )
    elif completeness < 85.0:
        recs.append(
            {
                "code": "DOCUMENTS_PARTIAL",
                "issue": f"Partial documentation gaps identified ({completeness:.1f}%)",
                "action": "Issue targeted checklists to remaining landholders for submission of Aadhaar/PAN and non-encumbrance certificates.",
                "priority": "Medium",
                "department": "Revenue Sub-Divisional Office (SDO)",
                "timeline": "Within 21 days",
            }
        )

    # 5. Survey & Demarcation
    surv_comp = int(case.get("survey_completed", 1) or 0)
    if surv_comp == 0:
        recs.append(
            {
                "code": "SURVEY_PENDING",
                "issue": "Cadastral field survey and boundary identification incomplete",
                "action": "Mobilize DGPS/Drone survey teams under joint supervision of State Survey Directorate and project executing agency.",
                "priority": "High",
                "department": "Directorate of Survey & Land Records",
                "timeline": "Within 10 days",
            }
        )

    dem_comp = int(case.get("demarcation_completed", 1) or 0)
    if dem_comp == 0 and surv_comp == 1:
        recs.append(
            {
                "code": "DEMARCATION_PENDING",
                "issue": "Joint physical peg-marking and boundary pillar erection pending",
                "action": "Schedule joint field demarcation with Gram Panchayat pradhans and project engineers.",
                "priority": "Medium",
                "department": "Project Executing Agency & Circle Office",
                "timeline": "Within 15 days",
            }
        )

    # 6. Approvals
    app_comp = int(case.get("approval_completed", 1) or 0)
    if app_comp == 0:
        recs.append(
            {
                "code": "APPROVAL_PENDING",
                "issue": "Statutory forest/environmental clearances or administrative sanction pending",
                "action": "Escalate via Project Monitoring Group (PMG) portal for fast-track inter-departmental nodal officer clearance.",
                "priority": "High",
                "department": "Environment/Forest Nodal Cell & Secretariat",
                "timeline": "Within 14 days",
            }
        )

    # 7. Objections Load
    objs = int(case.get("number_of_objections", 0) or 0)
    if objs >= 8:
        recs.append(
            {
                "code": "HIGH_OBJECTIONS",
                "issue": f"Elevated volume of Section 15 objections received ({objs} active objections)",
                "action": "Schedule structured hearings in batches; publish reasoned hearing orders under Section 15(2) with clear notice to claimants.",
                "priority": "High",
                "department": "Hearing Officer / Additional District Magistrate (LA)",
                "timeline": "Within 10–15 days",
            }
        )

    # 8. Rehabilitation & Resettlement (R&R)
    rehab_stat = str(case.get("rehabilitation_status", "Completed"))
    if rehab_stat in ["Pending", "Survey Ongoing", "Grievance Pending", "Pending Land Allocation"]:
        recs.append(
            {
                "code": "REHABILITATION_PENDING",
                "issue": f"R&R scheme execution incomplete ({rehab_stat})",
                "action": "Convene R&R Committee meeting; verify provision of developed plots, transit allowances, and livelihood assistance awards.",
                "priority": "High",
                "department": "Rehabilitation & Resettlement Authority",
                "timeline": "Within 20 days",
            }
        )

    # 9. Institutional Friction & Stakeholder Engagement
    friction = str(case.get("inter_department_coordination", "Seamless"))
    if friction in ["Delayed Inter-Agency Responses", "Severe Coordination Breakdown"]:
        recs.append(
            {
                "code": "COORDINATION_FRICTION",
                "issue": f"Inter-agency coordination bottleneck ({friction})",
                "action": "Convene weekly District-level coordination committee chaired by District Magistrate/Collector with all stakeholder departments.",
                "priority": "Medium",
                "department": "District Magistrate Office",
                "timeline": "Weekly review",
            }
        )

    if not recs:
        recs.append(
            {
                "code": "ROUTINE_MONITORING",
                "issue": "All process milestones currently on schedule",
                "action": "Maintain routine milestone tracking and ensure timely physical possession handover upon milestone completion.",
                "priority": "Low",
                "department": "Field Inspection Unit",
                "timeline": "Monthly review",
            }
        )

    recs.append(
        {
            "code": "DISCLAIMER",
            "issue": "Statutory Governance Disclaimer",
            "action": "Decision support recommendation only. Does not replace statutory proceedings prescribed under RFCTLARR Act, 2013 or National Highway Act, 1956.",
            "priority": "Info",
            "department": "System Advisory",
            "timeline": "Informational",
        }
    )
    return recs


def priority_score(risk_level: str, case: dict[str, Any], delay_probability: float, expected_delay_days: int) -> tuple[int, str]:
    """
    Compute unified Priority Queue rating:
    Priority 1 -> CRITICAL
    Priority 2 -> HIGH
    Priority 3 -> MEDIUM
    Priority 4 -> LOW
    """
    base_score = {"CRITICAL": 85, "HIGH": 65, "MEDIUM": 40, "LOW": 15}.get(risk_level, 20)

    # Additive urgency points based on critical pending items
    urgency = 0
    if int(case.get("legal_dispute", case.get("legal_case", 0)) or 0) == 1:
        urgency += 14
    if int(case.get("ownership_dispute", 0) or 0) == 1:
        urgency += 10
    if int(case.get("compensation_paid", 1) or 0) == 0:
        urgency += 12
    if int(case.get("approval_completed", 1) or 0) == 0:
        urgency += 8
    if int(case.get("number_of_objections", 0) or 0) >= 8:
        urgency += 8

    delay_boost = min(15, expected_delay_days // 15)
    score = int(min(100, base_score + 0.3 * urgency + delay_boost))

    if risk_level == "CRITICAL" or score >= 80:
        label = "Priority 1 — Immediate Intervention"
    elif risk_level == "HIGH" or score >= 60:
        label = "Priority 2 — High Urgency Action"
    elif risk_level == "MEDIUM" or score >= 35:
        label = "Priority 3 — Proactive Monitoring"
    else:
        label = "Priority 4 — Routine Tracking"
    return score, label


def generate_alerts(case: dict[str, Any], risk_level: str, expected_delay_days: int) -> list[dict[str, str]]:
    alerts = []
    case_id = str(case.get("case_id", "Project"))

    if risk_level == "CRITICAL":
        alerts.append({
            "type": "CRITICAL_DELAY_RISK",
            "message": f"CRITICAL: Project {case_id} has {round(float(case.get('delay_probability', 0.85))*100)}% probability of severe acquisition delay.",
            "severity": "CRITICAL",
        })

    if expected_delay_days >= 120:
        alerts.append({
            "type": "PROLONGED_TIMELINE",
            "message": f"HIGH: Expected delay exceeds 120 days ({expected_delay_days} days predicted).",
            "severity": "HIGH",
        })

    comp_delay = int(case.get("compensation_delay_days", 0) or 0)
    if int(case.get("compensation_paid", 1) or 0) == 0 or comp_delay >= 30:
        alerts.append({
            "type": "COMPENSATION_BOTTLENECK",
            "message": f"HIGH: Compensation award pending for {comp_delay} days. Award disbursement camp required.",
            "severity": "HIGH",
        })

    if int(case.get("legal_dispute", case.get("legal_case", 0)) or 0) == 1:
        alerts.append({
            "type": "LITIGATION_ALERT",
            "message": f"HIGH: Active court litigation / legal injunction risk flagged for {case_id}.",
            "severity": "HIGH",
        })

    docs = float(case.get("document_completeness_pct", 85.0) or 85.0)
    if docs < 75.0:
        alerts.append({
            "type": "DOCUMENTATION_DEFICIT",
            "message": f"MEDIUM: Document completeness dropped below statutory threshold ({docs:.1f}%).",
            "severity": "MEDIUM",
        })

    objs = int(case.get("number_of_objections", 0) or 0)
    if objs >= 8:
        alerts.append({
            "type": "HIGH_OBJECTION_LOAD",
            "message": f"MEDIUM: High landowner objection load ({objs} objections pending inquiry).",
            "severity": "MEDIUM",
        })

    return alerts
