"""
AI-Powered Land Acquisition Delay Prediction & Decision Support System
Executive Intelligence Dashboard for Smart India Hackathon (SIH).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LIFECYCLE_STAGES, PROJECT_TYPES
from src.location_master import get_all_states, get_districts_for_state, validate_location

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DISCLAIMER = (
    "Synthetic Prototype Dataset for Demonstration & Evaluation (Smart India Hackathon). "
    "Records, simulated parameters, and map coordinates are generated for prototype evaluation. "
    "They are NOT real government or judicial land acquisition records."
)

RISK_COLORS = {
    "LOW": "#059669",      # Emerald Green
    "MEDIUM": "#d97706",   # Amber
    "HIGH": "#ea580c",     # Deep Orange
    "CRITICAL": "#dc2626", # Crimson Red
}

RISK_BG_COLORS = {
    "LOW": "#ecfdf5",
    "MEDIUM": "#fffbeb",
    "HIGH": "#fff7ed",
    "CRITICAL": "#fef2f2",
}

STAGE_ICONS = {
    "Preliminary Planning": "📋",
    "Notification": "📢",
    "Survey": "📐",
    "Land Identification": "🗺️",
    "Ownership Verification": "🔍",
    "Objection Handling": "⚖️",
    "Approval": "🏛️",
    "Compensation": "💰",
    "Rehabilitation & Resettlement": "🏡",
    "Possession": "🚩",
    "Completion": "✅",
}


def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #1e293b;
        }
        
        .main {
            background-color: #f8fafc;
        }
        
        /* Top Navigation & Brand Header */
        .platform-header {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            padding: 1.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
            border-left: 6px solid #2563eb;
        }
        .platform-header h1 {
            color: #ffffff !important;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
        }
        .platform-header p {
            color: #94a3b8 !important;
            font-size: 0.95rem;
            margin: 0.35rem 0 0 0;
        }
        
        /* KPI Cards */
        .kpi-container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .kpi-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        }
        .kpi-card .label {
            font-size: 0.78rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }
        .kpi-card .value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.1;
        }
        .kpi-card .subtext {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 0.3rem;
        }
        
        /* Risk Badges */
        .risk-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .risk-badge-CRITICAL { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
        .risk-badge-HIGH { background: #fff7ed; color: #c2410c; border: 1px solid #ffedd5; }
        .risk-badge-MEDIUM { background: #fffbeb; color: #b45309; border: 1px solid #fef3c7; }
        .risk-badge-LOW { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
        
        /* Disclaimer Card */
        .disclaimer-card {
            background: #f1f5f9;
            border: 1px solid #cbd5e1;
            border-left: 4px solid #64748b;
            padding: 0.75rem 1rem;
            border-radius: 6px;
            font-size: 0.8rem;
            color: #475569;
            margin-bottom: 1.25rem;
        }
        
        /* Lifecycle Progress Tracker */
        .lifecycle-bar {
            display: flex;
            overflow-x: auto;
            gap: 0.5rem;
            padding: 0.75rem 0;
            margin-bottom: 1.25rem;
        }
        .lifecycle-node {
            flex: 1;
            min-width: 110px;
            padding: 0.7rem 0.6rem;
            border-radius: 8px;
            text-align: center;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid #e2e8f0;
            background: white;
        }
        .node-completed { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
        .node-in-progress { background: #eff6ff; color: #1d4ed8; border-color: #bfdbfe; font-weight: 700; }
        .node-bottleneck { background: #fef2f2; color: #991b1b; border-color: #fecaca; font-weight: 700; animation: pulse 2s infinite; }
        .node-pending { background: #f8fafc; color: #94a3b8; }
        
        /* Recommendation Cards */
        .rec-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left: 4px solid #2563eb;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 0.75rem;
        }
        .rec-card.high { border-left-color: #ea580c; }
        .rec-card.critical { border-left-color: #dc2626; }
        .rec-card .rec-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }
        .rec-card .rec-meta {
            font-size: 0.8rem;
            color: #64748b;
            margin-top: 0.4rem;
        }
        
        /* Dataframes & Tables */
        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            background: white;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            color: #e2e8f0;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            color: #ffffff;
        }
        section[data-testid="stSidebar"] .stRadio > label {
            color: #94a3b8;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label {
            color: #cbd5e1;
            padding: 0.4rem 0.6rem;
            border-radius: 6px;
            transition: background 0.15s;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
            background: #1e293b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def api(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", {})
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=60, **kwargs)
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach Backend API at {API_URL}. Please ensure uvicorn is running: python -m uvicorn backend.main:app --port 8000")
        st.stop()

    if resp.status_code == 401:
        st.session_state.clear()
        st.warning("Session expired. Please log in again.")
        st.rerun()
    return resp


def login_page():
    st.markdown(
        """
        <div class="platform-header">
            <h1>🏛️ Land Acquisition Intelligence Platform</h1>
            <p>AI-Powered Predictive Decision Support System · Smart India Hackathon (SIH)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='disclaimer-card'><b>Notice:</b> {DISCLAIMER}</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c2:
        st.subheader("Secure Officer Login")
        st.caption("Access the decision-support system with authorized departmental credentials.")
        with st.form("login_form"):
            username = st.text_input("Officer / User ID", value="officer")
            password = st.text_input("Password", type="password", value="officer123")
            submitted = st.form_submit_button("Sign In to Decision Support", use_container_width=True)

        if submitted:
            resp = api("POST", "/auth/login-json", json={"username": username, "password": password})
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.token = data["access_token"]
                st.session_state.role = data["role"]
                st.session_state.full_name = data["full_name"]
                st.session_state.username = username
                st.rerun()
            else:
                st.error(resp.json().get("detail", "Authentication failed. Check credentials."))

        st.markdown(
            """
            ---
            **Demo Credentials (Role-Based Access):**
            - **Admin (HQ):** `admin` / `admin123` (Full administrative & MLOps access)
            - **Officer (CALA):** `officer` / `officer123` (Cases, predictions, actions, alerts)
            - **Analyst:** `analyst` / `analyst123` (GIS, state & district analytics)
            - **Viewer:** `viewer` / `viewer123` (Read-only observation)
            """
        )


def render_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="platform-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Page: Executive Overview (Dashboard)
# --------------------------------------------------------------------------


def page_overview():
    render_header(
        "Land Acquisition Intelligence Platform",
        "Predictive Governance & Decision Support Dashboard · Real-Time Delay Risk Monitoring",
    )
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    stats = api("GET", "/dashboard/stats").json()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(
            f"<div class='kpi-card'><div class='label'>Total Cases</div><div class='value'>{stats.get('total_cases', 0):,}</div><div class='subtext'>36 States & UTs</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='kpi-card'><div class='label'>Critical Risk</div><div class='value' style='color:#dc2626;'>{stats.get('critical_cases', 0):,}</div><div class='subtext'>Score 76–100</div></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='kpi-card'><div class='label'>High Risk</div><div class='value' style='color:#ea580c;'>{stats.get('high_risk_cases', 0):,}</div><div class='subtext'>Score 51–75</div></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='kpi-card'><div class='label'>Projects at Risk</div><div class='value' style='color:#0f172a;'>{stats.get('projects_at_risk', 0):,}</div><div class='subtext'>High + Critical</div></div>",
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            f"<div class='kpi-card'><div class='label'>Avg Delay</div><div class='value'>{stats.get('average_predicted_delay', 0)} d</div><div class='subtext'>Expected days</div></div>",
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            f"<div class='kpi-card'><div class='label'>Avg Risk Score</div><div class='value'>{stats.get('average_risk_score', 0)}/100</div><div class='subtext'>Calibrated index</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    ch1, ch2 = st.columns([1, 1.2])
    with ch1:
        st.markdown("#### Portfolio Risk Distribution")
        dist = stats.get("risk_distribution", {})
        if dist:
            fig_pie = px.pie(
                names=list(dist.keys()),
                values=list(dist.values()),
                color=list(dist.keys()),
                color_discrete_map=RISK_COLORS,
                hole=0.45,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=280)
            st.plotly_chart(fig_pie, use_container_width=True)

    with ch2:
        st.markdown("#### State-Wise Risk Hotspots")
        st_data = api("GET", "/analytics/state").json().get("states", [])[:8]
        if st_data:
            df_st = pd.DataFrame(st_data)
            fig_bar = px.bar(
                df_st,
                x="state",
                y=["critical_projects", "high_risk_projects"],
                title="",
                labels={"value": "Projects Count", "state": "State / UT", "variable": "Risk Severity"},
                color_discrete_map={"critical_projects": "#dc2626", "high_risk_projects": "#ea580c"},
                barmode="stack",
            )
            fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_bar, use_container_width=True)

    # High Priority Projects Table
    st.markdown("---")
    st.markdown("### 🚨 High-Priority Projects Requiring Executive Intervention")
    high_cases = api("GET", "/high-risk-cases", params={"limit": 10}).json().get("items", [])
    if high_cases:
        df_high = pd.DataFrame(high_cases)[
            ["case_id", "project_name", "state", "district", "current_stage", "risk_score", "risk_level", "delay_probability", "expected_delay_days", "main_reason"]
        ]
        df_high["delay_probability"] = (df_high["delay_probability"] * 100).round(1).astype(str) + "%"
        df_high.rename(
            columns={
                "case_id": "Case ID",
                "project_name": "Project Name",
                "state": "State",
                "district": "District",
                "current_stage": "Stage",
                "risk_score": "Risk Score",
                "risk_level": "Category",
                "delay_probability": "Delay Prob",
                "expected_delay_days": "Expected Delay (Days)",
                "main_reason": "Primary Roadblock",
            },
            inplace=True,
        )
        st.dataframe(df_high, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Page: Case Register (Projects)
# --------------------------------------------------------------------------


def page_cases():
    render_header("Project Case Register", "Authoritative Land Acquisition Records across 28 States & 8 Union Territories")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    # Filter Bar
    all_states = ["All States"] + get_all_states()
    f1, f2, f3, f4, f5 = st.columns(5)
    sel_state = f1.selectbox("Filter State/UT", all_states)

    if sel_state != "All States":
        valid_districts = ["All Districts"] + get_districts_for_state(sel_state)
    else:
        valid_districts = ["All Districts"]
    sel_district = f2.selectbox("Filter District", valid_districts)

    sel_risk = f3.selectbox("Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    sel_stage = f4.selectbox("Current Stage", ["All Stages"] + LIFECYCLE_STAGES)
    search_q = f5.text_input("Search Case ID / Project", placeholder="e.g. LA-2026-00012")

    params = {"limit": 150}
    if sel_state != "All States":
        params["state"] = sel_state
    if sel_district != "All Districts":
        params["district"] = sel_district
    if sel_risk != "All":
        params["risk_level"] = sel_risk
    if sel_stage != "All Stages":
        params["current_stage"] = sel_stage
    if search_q.strip():
        params["q"] = search_q.strip()

    data = api("GET", "/cases", params=params).json()
    items = data.get("items", [])
    st.caption(f"Showing **{len(items)}** of **{data.get('total', 0)}** matching records.")

    if not items:
        st.info("No projects match the selected criteria.")
        return

    df_view = pd.DataFrame(items)[
        ["case_id", "project_name", "state", "district", "project_type", "current_stage", "risk_score", "risk_level", "delay_probability", "expected_delay_days", "status"]
    ].copy()
    df_view["delay_probability"] = (df_view["delay_probability"] * 100).round(1).astype(str) + "%"
    df_view.rename(
        columns={
            "case_id": "Case ID",
            "project_name": "Project Name",
            "state": "State",
            "district": "District",
            "project_type": "Sector",
            "current_stage": "Current Stage",
            "risk_score": "Risk Score",
            "risk_level": "Risk Tier",
            "delay_probability": "Delay Prob",
            "expected_delay_days": "Expected Delay",
            "status": "Status",
        },
        inplace=True,
    )
    st.dataframe(df_view, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔎 Drill-Down into Project Details")
    selected_id = st.selectbox("Select Project to Inspect", [""] + [c["case_id"] for c in items])
    if selected_id:
        render_project_details(selected_id)


# --------------------------------------------------------------------------
# Page: Project Details & Lifecycle
# --------------------------------------------------------------------------


def render_project_details(case_id: str):
    detail = api("GET", f"/cases/{case_id}").json()
    live = detail.get("live_explanation", {})
    risk_cat = detail.get("risk_level", "LOW")
    risk_score = detail.get("risk_score", 0)

    st.markdown(
        f"""
        <div style="background:white; border:1px solid #e2e8f0; border-radius:10px; padding:1.25rem; margin-top:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.05);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <h2 style="margin:0; font-size:1.4rem; color:#0f172a;">{detail.get('project_name')}</h2>
                    <p style="margin:0.25rem 0 0 0; color:#64748b; font-size:0.9rem;">
                        <b>Case ID:</b> {case_id} &nbsp;|&nbsp; <b>Location:</b> {detail.get('district')}, {detail.get('state')} &nbsp;|&nbsp; <b>Sector:</b> {detail.get('project_type')}
                    </p>
                </div>
                <div>
                    <span class="risk-badge risk-badge-{risk_cat}">{risk_cat} RISK (Score: {risk_score}/100)</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Delay Probability", f"{round(detail.get('delay_probability', 0)*100, 1)}%")
    m2.metric("Expected Delay", f"{detail.get('expected_delay_days', 0)} days")
    m3.metric("Land Area", f"{detail.get('land_area_hectares', 0)} ha")
    m4.metric("Affected Families", f"{detail.get('affected_families', 0):,}")
    m5.metric("Document Completeness", f"{detail.get('document_completeness_pct', 0)}%")

    # 11-Stage Lifecycle Visualizer
    st.markdown("#### 🔄 Statutory Land Acquisition Lifecycle Progression")
    timeline_events = detail.get("timeline", [])
    cur_stage = detail.get("current_stage", "Compensation")

    cols = st.columns(len(LIFECYCLE_STAGES))
    for i, stg in enumerate(LIFECYCLE_STAGES):
        is_cur = (stg == cur_stage)
        # Find matching stage status if available
        matched = next((t for t in timeline_events if t.get("stage_name") == stg), None)
        stat = matched.get("status", "Completed" if i < LIFECYCLE_STAGES.index(cur_stage) else "Pending") if cur_stage in LIFECYCLE_STAGES else "Pending"
        
        icon = STAGE_ICONS.get(stg, "•")
        if is_cur:
            node_class = "node-bottleneck" if risk_cat in ["HIGH", "CRITICAL"] else "node-in-progress"
            stat_label = "BOTTLENECK" if risk_cat in ["HIGH", "CRITICAL"] else "CURRENT"
        elif stat == "Completed":
            node_class = "node-completed"
            stat_label = "COMPLETED"
        else:
            node_class = "node-pending"
            stat_label = "PENDING"

        cols[i].markdown(
            f"""
            <div class="lifecycle-node {node_class}">
                <div style="font-size:1.1rem; margin-bottom:0.2rem;">{icon}</div>
                <div style="line-height:1.1; margin-bottom:0.25rem;">{stg}</div>
                <span style="font-size:0.65rem; text-transform:uppercase;">{stat_label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Explainable AI (SHAP) & Actions
    x1, x2 = st.columns([1.1, 1])
    with x1:
        st.markdown("#### 🧠 Explainable AI (SHAP) — Decision Attribution")
        st.caption("Human-understandable attribution explaining why the AI predicted this risk level.")
        
        inc = live.get("factors_increasing_risk", [])
        if inc:
            st.markdown("**Top Delay Risk Escalators:**")
            for f in inc:
                st.markdown(f"- 🔴 **{f['factor']}** &nbsp;·&nbsp; <span style='color:#dc2626; font-size:0.85rem;'>{f['impact']} (+{f['shap_value']:.3f} SHAP)</span>", unsafe_allow_html=True)

        red = live.get("factors_reducing_risk", [])
        if red:
            st.markdown("<br>**Protective / Mitigating Factors:**", unsafe_allow_html=True)
            for f in red:
                st.markdown(f"- 🟢 **{f['factor']}** &nbsp;·&nbsp; <span style='color:#059669; font-size:0.85rem;'>{f['impact']} ({f['shap_value']:.3f} SHAP)</span>", unsafe_allow_html=True)

        st.info(f"**Executive Summary:** {live.get('human_explanation', 'Evaluation complete.')}")

    with x2:
        st.markdown("#### ⚡ Recommended Corrective Interventions")
        st.caption("Actionable guidance tailored to the identified milestone bottlenecks.")
        recs = live.get("recommendations", [])
        for r in recs:
            if r.get("code") == "DISCLAIMER":
                continue
            prio = r.get("priority", "Medium").lower()
            st.markdown(
                f"""
                <div class="rec-card {prio}">
                    <div class="rec-title"><b>[{r.get('priority')}]</b> {r.get('issue', 'Intervention Required')}</div>
                    <div style="font-size:0.88rem; color:#334155;">{r.get('action')}</div>
                    <div class="rec-meta">🏛️ <b>Dept:</b> {r.get('department')} &nbsp;|&nbsp; ⏱️ <b>Timeline:</b> {r.get('timeline')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Action Update Form
    st.markdown("---")
    st.markdown("#### 📝 Record Administrative Action or Milestone Update")
    with st.form(f"update_form_{case_id}"):
        u1, u2, u3 = st.columns(3)
        new_stage = u1.selectbox("Update Lifecycle Stage", LIFECYCLE_STAGES, index=LIFECYCLE_STAGES.index(cur_stage) if cur_stage in LIFECYCLE_STAGES else 0)
        new_comp_paid = u2.selectbox("Compensation Paid?", [1, 0], format_func=lambda x: "Yes (Disbursed)" if x else "No (Pending)", index=0 if detail.get("compensation_paid") else 1)
        new_status = u3.selectbox("Operational Status", list(ALLOWED_STATUSES), index=0)
        officer_remark = st.text_area("Official Note / Action Taken", placeholder="e.g. Compensation camp convened; dispute referred to SLAO summary court.")
        submit_btn = st.form_submit_button("Record Action & Recalculate Risk")

    if submit_btn:
        update_payload = {
            "current_stage": new_stage,
            "compensation_paid": new_comp_paid,
            "status": new_status,
        }
        api("PUT", f"/cases/{case_id}", json=update_payload)
        if officer_remark.strip():
            api("POST", f"/cases/{case_id}/remarks", json={"remark": officer_remark.strip()})
        st.success(f"Case {case_id} updated. Risk scores and audit logs refreshed.")
        st.rerun()


# --------------------------------------------------------------------------
# Page: GIS Risk Map
# --------------------------------------------------------------------------


def page_gis_map():
    render_header("Geographic Risk Intelligence Map", "Interactive GIS Visualization of Infrastructure Projects across India")
    st.markdown(
        "<div class='disclaimer-card'><b>GIS Disclosure:</b> Map pins display synthetic project centroids near official district headquarters for prototype demonstration. Not surveyed boundary polygons.</div>",
        unsafe_allow_html=True,
    )

    g1, g2, g3, g4 = st.columns(4)
    all_states = ["All States"] + get_all_states()
    sel_st = g1.selectbox("GIS Filter: State/UT", all_states, key="gis_st")

    if sel_st != "All States":
        valid_dist = ["All Districts"] + get_districts_for_state(sel_st)
    else:
        valid_dist = ["All Districts"]
    sel_dt = g2.selectbox("GIS Filter: District", valid_dist, key="gis_dt")

    sel_rk = g3.selectbox("Risk Filter", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"], key="gis_rk")
    sel_pt = g4.selectbox("Sector Filter", ["All Sectors"] + PROJECT_TYPES, key="gis_pt")

    params = {"limit": 1200}
    if sel_st != "All States":
        params["state"] = sel_st
    if sel_dt != "All Districts":
        params["district"] = sel_dt
    if sel_rk != "All":
        params["risk_level"] = sel_rk
    if sel_pt != "All Sectors":
        params["project_type"] = sel_pt

    gis_res = api("GET", "/gis/points", params=params).json()
    points = gis_res.get("items", [])
    st.caption(f"Visualizing **{len(points)}** infrastructure projects on OpenStreetMap.")

    # Center map based on selection
    if sel_st != "All States" and points:
        center_lat = points[0]["latitude"]
        center_lon = points[0]["longitude"]
        zoom_lvl = 7
    else:
        center_lat, center_lon = 22.8, 79.2
        zoom_lvl = 5

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_lvl, tiles="OpenStreetMap")
    folium_colors = {"LOW": "green", "MEDIUM": "orange", "HIGH": "darkorange", "CRITICAL": "red"}

    for p in points:
        popup_html = f"""
        <div style="font-family:Inter,sans-serif; width:260px; font-size:12px; line-height:1.4;">
            <b style="font-size:13px; color:#0f172a;">{p['project_name']}</b><br>
            <span style="color:#64748b;">ID: {p['case_id']} · {p['district']}, {p['state']}</span><br>
            <hr style="margin:4px 0; border:0; border-top:1px solid #e2e8f0;">
            <b>Sector:</b> {p['project_type']}<br>
            <b>Risk Score:</b> <span style="font-weight:700; color:{RISK_COLORS.get(p['risk_level'], '#000')}">{p.get('risk_score', 0)}/100 ({p['risk_level']})</span><br>
            <b>Expected Delay:</b> {p['expected_delay_days']} days<br>
            <b>Primary Driver:</b> {p['main_reason']}<br>
            <i style="color:#94a3b8; font-size:10px;">Synthetic prototype coordinates</i>
        </div>
        """
        folium.CircleMarker(
            location=[p["latitude"], p["longitude"]],
            radius=6,
            color=folium_colors.get(p["risk_level"], "blue"),
            fill=True,
            fill_color=folium_colors.get(p["risk_level"], "blue"),
            fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(fmap)

    st_folium(fmap, width="100%", height=600)


# --------------------------------------------------------------------------
# Page: Early Warning Center
# --------------------------------------------------------------------------


def page_early_warning():
    render_header("Early Warning & Proactive Intervention Center", "Prioritized Escalation Queue for Multi-Agency Roadblocks")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    # Escalation Queues (Priority 1 to 4)
    q1, q2 = st.columns([1.3, 1])
    with q1:
        st.markdown("### 🚨 Priority Escalation Queue")
        st.caption("Projects prioritized by composite urgency, litigation, and compensation bottlenecks.")

        high_list = api("GET", "/high-risk-cases", params={"limit": 30}).json().get("items", [])
        if high_list:
            df_pq = pd.DataFrame(high_list)[
                ["case_id", "project_name", "state", "district", "risk_level", "risk_score", "expected_delay_days", "priority_label", "main_reason"]
            ]
            df_pq.rename(
                columns={
                    "case_id": "Case ID",
                    "project_name": "Project",
                    "state": "State",
                    "district": "District",
                    "risk_level": "Tier",
                    "risk_score": "Score",
                    "expected_delay_days": "Delay",
                    "priority_label": "Queue Urgency",
                    "main_reason": "Escalation Cause",
                },
                inplace=True,
            )
            st.dataframe(df_pq, use_container_width=True, hide_index=True)
        else:
            st.info("No cases currently pending in high-priority escalation queue.")

    with q2:
        st.markdown("### 🔔 Real-Time Early Warning Alerts")
        alerts_res = api("GET", "/alerts", params={"limit": 25}).json().get("items", [])
        unread_count = sum(1 for a in alerts_res if not a.get("is_read"))
        st.caption(f"{unread_count} unread automated alerts.")

        if st.button("Mark All Alerts as Read"):
            api("POST", "/alerts/read-all")
            st.rerun()

        for a in alerts_res[:15]:
            sev = a.get("severity", "MEDIUM")
            border_col = RISK_COLORS.get(sev, "#2563eb")
            is_read = a.get("is_read")
            st.markdown(
                f"""
                <div style="background:{'#f8fafc' if is_read else '#ffffff'}; border:1px solid #e2e8f0; border-left:4px solid {border_col}; padding:0.65rem 0.85rem; border-radius:6px; margin-bottom:0.5rem;">
                    <div style="display:flex; justify-content:space-between;">
                        <b style="font-size:0.82rem; color:{border_col};">{a.get('type')}</b>
                        <span style="font-size:0.75rem; color:#94a3b8;">{a.get('created_at', '')[:10]}</span>
                    </div>
                    <div style="font-size:0.85rem; color:#334155; margin-top:0.2rem;">{a.get('message')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------
# Page: State Analytics
# --------------------------------------------------------------------------


def page_state_analytics():
    render_header("State & UT Analytics", "Comparative Land Acquisition Performance Across Indian States & UTs")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    st_data = api("GET", "/analytics/state").json().get("states", [])
    if not st_data:
        st.info("State analytics data is loading.")
        return

    df_st = pd.DataFrame(st_data)

    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("#### State Selection & Deep-Dive")
        selected_state = st.selectbox("Select State/UT to Inspect", df_st["state"].tolist())
        st_row = df_st[df_st["state"] == selected_state].iloc[0]

        s1, s2 = st.columns(2)
        s1.metric("Total Infrastructure Projects", f"{st_row['total_projects']}")
        s2.metric("Projects at Elevated Risk", f"{st_row['total_at_risk']}")
        s3, s4 = st.columns(2)
        s3.metric("Average Delay Timeline", f"{st_row['average_delay_days']} days")
        s4.metric("State Avg Risk Score", f"{st_row['average_risk_score']}/100")

    with c2:
        st.markdown("#### State vs Average Delay Timeline (Top 12)")
        fig_del = px.bar(
            df_st.sort_values("average_delay_days", ascending=False).head(12),
            x="state",
            y="average_delay_days",
            labels={"state": "State / UT", "average_delay_days": "Avg Delay (Days)"},
            color="average_delay_days",
            color_continuous_scale="Reds",
        )
        fig_del.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
        st.plotly_chart(fig_del, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Comprehensive State-Wise Acquisition Performance Register")
    df_st_display = df_st.copy()
    df_st_display["average_delay_probability"] = (df_st_display["average_delay_probability"] * 100).round(1).astype(str) + "%"
    df_st_display.rename(
        columns={
            "state": "State / UT",
            "total_projects": "Total Projects",
            "average_delay_days": "Avg Delay (Days)",
            "average_risk_score": "Avg Risk Score",
            "critical_projects": "Critical Risk",
            "high_risk_projects": "High Risk",
            "total_at_risk": "Total at Risk",
            "average_delay_probability": "Avg Delay Prob",
        },
        inplace=True,
    )
    st.dataframe(df_st_display, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Page: District Analytics (Cascading State -> District)
# --------------------------------------------------------------------------


def page_district_analytics():
    render_header("District-Level Analytics", "Strict State-to-District Hierarchical Governance & Throughput Tracking")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    all_states = get_all_states()
    c1, c2 = st.columns(2)
    sel_state = c1.selectbox("Step 1: Select State / Union Territory", all_states, index=all_states.index("Uttarakhand") if "Uttarakhand" in all_states else 0)

    # Cascading valid districts only
    valid_districts = get_districts_for_state(sel_state)
    sel_district = c2.selectbox("Step 2: Select District (Filtered under State)", valid_districts)

    st.caption(f"Analyzing official district records for **{sel_district}**, **{sel_state}**.")

    dist_res = api("GET", "/analytics/district", params={"state": sel_state}).json()
    dist_list = dist_res.get("districts", [])
    df_dist = pd.DataFrame(dist_list)

    # District KPIs
    target_dist = df_dist[df_dist["district"] == sel_district]
    if not target_dist.empty:
        d_row = target_dist.iloc[0]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("District Total Projects", f"{d_row['total_projects']}")
        k2.metric("Critical Projects", f"{d_row['critical_projects']}")
        k3.metric("High Risk Projects", f"{d_row['high_risk_projects']}")
        k4.metric("Avg Projected Delay", f"{d_row['average_delay_days']} days")

    st.markdown("---")
    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"#### District Delay Comparisons in {sel_state}")
        if not df_dist.empty:
            fig_dt = px.bar(
                df_dist.sort_values("average_delay_days", ascending=False).head(15),
                x="district",
                y="average_delay_days",
                labels={"district": "District", "average_delay_days": "Avg Delay (Days)"},
                color="average_risk_score",
                color_continuous_scale="Blues",
            )
            fig_dt.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_dt, use_container_width=True)

    with d2:
        st.markdown(f"#### Risk Severity Distribution in {sel_state}")
        if not df_dist.empty:
            fig_stk = px.bar(
                df_dist.sort_values("total_at_risk", ascending=False).head(15),
                x="district",
                y=["critical_projects", "high_risk_projects"],
                labels={"value": "Projects", "district": "District", "variable": "Risk Level"},
                color_discrete_map={"critical_projects": "#dc2626", "high_risk_projects": "#ea580c"},
                barmode="stack",
            )
            fig_stk.update_layout(height=320, margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_stk, use_container_width=True)


# --------------------------------------------------------------------------
# Page: What-If Scenario Simulator
# --------------------------------------------------------------------------


def page_what_if():
    render_header("What-If Scenario Simulator", "Evaluate Policy & Operational Interventions with Live Model Prediction")
    st.markdown(
        "<div class='disclaimer-card'>Test hypothetical milestone resolutions (e.g. completing compensation, vacating legal injunctions, increasing document completeness) and observe live model predictions.</div>",
        unsafe_allow_html=True,
    )

    # Project selection or manual payload
    sample_cases = api("GET", "/high-risk-cases", params={"limit": 15}).json().get("items", [])
    case_opts = ["Custom Simulated Scenario"] + [f"{c['case_id']} - {c['project_name']}" for c in sample_cases]

    sel_case_str = st.selectbox("Select Baseline Project for Scenario Modeling", case_opts)

    selected_case_id = None
    if sel_case_str != "Custom Simulated Scenario":
        selected_case_id = sel_case_str.split(" - ")[0]

    st.markdown("#### Scenario Modification Levers")
    w1, w2, w3 = st.columns(3)
    sim_comp = w1.selectbox("Simulated Compensation Status", ["No Change", "Disbursed / Completed", "Pending Award"], index=1)
    sim_legal = w2.selectbox("Simulated Legal / Litigation Status", ["No Change", "Litigation Resolved / Stay Vacated", "Active Litigation"], index=1)
    sim_docs = w3.slider("Simulated Document Completeness (%)", 40, 100, 95)

    w4, w5, w6 = st.columns(3)
    sim_approval = w4.selectbox("Statutory Approvals", ["No Change", "Clearances Approved", "Clearances Pending"], index=1)
    sim_objs = w5.slider("Landowner Objections (Simulated)", 0, 30, 2)
    sim_stakeholder = w6.selectbox("Community Responsiveness", ["No Change", "High Cooperation", "Medium", "Critical Resistance"], index=1)

    sim_payload: dict[str, Any] = {"case_id": selected_case_id}
    if sim_comp == "Disbursed / Completed":
        sim_payload["compensation_paid"] = 1
        sim_payload["compensation_status"] = "Disbursed"
        sim_payload["compensation_delay_days"] = 0
    elif sim_comp == "Pending Award":
        sim_payload["compensation_paid"] = 0
        sim_payload["compensation_status"] = "Pending Award Inquiry"

    if sim_legal == "Litigation Resolved / Stay Vacated":
        sim_payload["legal_dispute"] = 0
    elif sim_legal == "Active Litigation":
        sim_payload["legal_dispute"] = 1

    sim_payload["document_completeness_pct"] = float(sim_docs)

    if sim_approval == "Clearances Approved":
        sim_payload["approval_completed"] = 1
        sim_payload["approval_status"] = "Approved"
        sim_payload["approval_delay_days"] = 0
    elif sim_approval == "Clearances Pending":
        sim_payload["approval_completed"] = 0

    sim_payload["number_of_objections"] = sim_objs

    if sim_stakeholder == "High Cooperation":
        sim_payload["stakeholder_responsiveness"] = "High"
    elif sim_stakeholder == "Critical Resistance":
        sim_payload["stakeholder_responsiveness"] = "Critical Resistance"

    if st.button("Run What-If Model Inference", type="primary"):
        sim_res = api("POST", "/what-if", json=sim_payload).json()
        orig = sim_res["original"]
        sim = sim_res["simulated"]

        st.markdown("### 📊 Scenario Simulation Results")
        st.success(f"**Impact Summary:** {sim_res['summary']}")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Score", f"{sim['risk_score']}/100", delta=f"{sim_res['risk_score_delta']} pts", delta_color="inverse")
        r2.metric("Delay Probability", f"{round(sim['delay_probability']*100)}%", delta=f"{round(sim_res['delay_probability_delta']*100, 1)}%", delta_color="inverse")
        r3.metric("Expected Delay", f"{sim['expected_delay_days']} days", delta=f"{sim_res['delay_days_delta']} days", delta_color="inverse")
        r4.metric("Risk Category", f"{sim['risk_level']}", delta=f"Prev: {orig['risk_level']}")

        # Comparison Bar Chart
        comp_df = pd.DataFrame(
            [
                {"Metric": "Risk Score (0-100)", "Baseline (Before)": orig["risk_score"], "Simulated (After)": sim["risk_score"]},
                {"Metric": "Expected Delay (Days)", "Baseline (Before)": orig["expected_delay_days"], "Simulated (After)": sim["expected_delay_days"]},
                {"Metric": "Delay Probability (%)", "Baseline (Before)": round(orig["delay_probability"] * 100), "Simulated (After)": round(sim["delay_probability"] * 100)},
            ]
        )
        fig_cmp = go.Figure(
            data=[
                go.Bar(name="Baseline (Before)", x=comp_df["Metric"], y=comp_df["Baseline (Before)"], marker_color="#dc2626"),
                go.Bar(name="Simulated (After Intervention)", x=comp_df["Metric"], y=comp_df["Simulated (After)"], marker_color="#059669"),
            ]
        )
        fig_cmp.update_layout(barmode="group", height=320, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cmp, use_container_width=True)


# --------------------------------------------------------------------------
# Page: SIH Demo Mode
# --------------------------------------------------------------------------


def page_sih_demo_mode():
    render_header("Smart India Hackathon (SIH) Presentation Flow", "Live Demonstration Script for Hackathon Jury Evaluation")
    st.markdown(
        """
        <div class="disclaimer-card">
            <b>SIH Demo Instructions:</b> This section guides the presentation team through the required 11-step evaluation pitch flow.
            Each step links directly to live working features with pre-configured synthetic cases representing <b>LOW</b>, <b>MEDIUM</b>, <b>HIGH</b>, and <b>CRITICAL</b> risk tiers.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🏆 11-Step SIH Pitch Walkthrough")
    demo_steps = [
        ("Step 1: Open Executive Dashboard", "Show 5,000+ cases, 36 States & UTs, Critical Risk counter, and Average Delay KPI cards."),
        ("Step 2: Prioritize High-Risk Projects", "Navigate to High-Priority queue; filter by CRITICAL risk tier."),
        ("Step 3: Select Focal Project", "Drill-down into a focal project displaying 91% delay probability, 87/100 risk score, 96 expected delay days."),
        ("Step 4: Explain Primary Roadblocks", "Demonstrate WHY: Active court litigation, pending compensation disbursement, low document completeness."),
        ("Step 5: Explainable AI (SHAP) Attribution", "Show bidirectional SHAP waterfall/factors: factors increasing risk vs. factors reducing risk."),
        ("Step 6: Actionable Recommendations", "Display 5-attribute structured recommendations (Issue, Action, Priority, Department, Timeline)."),
        ("Step 7: Geographic GIS Risk Map", "Inspect project coordinates, district centroid clustering, and spatial risk distribution on OpenStreetMap."),
        ("Step 8: What-If Scenario Simulator", "Simulate completing compensation and vacating legal stays; observe model-predicted 28-day delay reduction."),
        ("Step 9: State-Wise Analytics", "Show comparative state delay timelines and elevated-risk project concentrations."),
        ("Step 10: District-Wise Analytics", "Demonstrate strict State -> District cascading dropdown (e.g. Uttarakhand -> Dehradun / Almora)."),
        ("Step 11: Early Warning & Audit Trail", "Show automated alert generation and administrative audit log tracking state transitions."),
    ]

    for title, desc in demo_steps:
        with st.expander(title, expanded=False):
            st.write(desc)

    st.markdown("---")
    st.subheader("🎯 1-Click Representative Archetype Test Cases")
    arch1, arch2, arch3, arch4 = st.columns(4)

    with arch1:
        st.markdown(
            """
            <div style="background:#ecfdf5; border:1px solid #a7f3d0; padding:0.75rem; border-radius:8px;">
                <b style="color:#065f46;">🟢 LOW RISK ARCHETYPE</b><br>
                <small>Clear titles, survey complete, compensation paid, zero litigation.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect Low Risk Case", key="arch_low"):
            low_case = api("GET", "/cases", params={"risk_level": "LOW", "limit": 1}).json().get("items", [])
            if low_case:
                st.session_state.demo_case_id = low_case[0]["case_id"]

    with arch2:
        st.markdown(
            """
            <div style="background:#fffbeb; border:1px solid #fef3c7; padding:0.75rem; border-radius:8px;">
                <b style="color:#b45309;">🟡 MEDIUM RISK ARCHETYPE</b><br>
                <small>Partial document gaps, moderate objections, survey complete.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect Medium Risk Case", key="arch_med"):
            med_case = api("GET", "/cases", params={"risk_level": "MEDIUM", "limit": 1}).json().get("items", [])
            if med_case:
                st.session_state.demo_case_id = med_case[0]["case_id"]

    with arch3:
        st.markdown(
            """
            <div style="background:#fff7ed; border:1px solid #ffedd5; padding:0.75rem; border-radius:8px;">
                <b style="color:#c2410c;">🟠 HIGH RISK ARCHETYPE</b><br>
                <small>Unresolved compensation backlog, pending clearances, objections >8.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect High Risk Case", key="arch_high"):
            high_case = api("GET", "/cases", params={"risk_level": "HIGH", "limit": 1}).json().get("items", [])
            if high_case:
                st.session_state.demo_case_id = high_case[0]["case_id"]

    with arch4:
        st.markdown(
            """
            <div style="background:#fef2f2; border:1px solid #fecaca; padding:0.75rem; border-radius:8px;">
                <b style="color:#991b1b;">🔴 CRITICAL RISK ARCHETYPE</b><br>
                <small>Active court litigation, ownership dispute, severe administrative delay.</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Inspect Critical Risk Case", key="arch_crit"):
            crit_case = api("GET", "/cases", params={"risk_level": "CRITICAL", "limit": 1}).json().get("items", [])
            if crit_case:
                st.session_state.demo_case_id = crit_case[0]["case_id"]

    if "demo_case_id" in st.session_state:
        st.markdown("---")
        render_project_details(st.session_state.demo_case_id)


# --------------------------------------------------------------------------
# Page: MLOps & Continuous Learning Architecture
# --------------------------------------------------------------------------


def page_mlops_governance():
    render_header("MLOps, Model Validation & Governance", "Statutory Compliance, Model Metrics & Governed Retraining Lifecycle")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    metrics = api("GET", "/metrics").json()

    st.markdown("### 📈 Model Evaluation Metrics (Held-Out Test Set)")
    st.caption("Real metrics computed on held-out test data. Zero target leakage.")

    cls_res = metrics.get("classification", {})
    reg_res = metrics.get("regression", {})
    best_cls = metrics.get("best_classifier", "xgboost")
    best_reg = metrics.get("best_regressor", "xgboost")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### Classification Performance (Best: `{best_cls}`)")
        cls_table = []
        for model_name, m_dict in cls_res.items():
            cls_table.append(
                {
                    "Algorithm": model_name.replace("_", " ").title(),
                    "Accuracy": f"{m_dict.get('accuracy', 0)*100:.1f}%",
                    "Precision": f"{m_dict.get('precision', 0)*100:.1f}%",
                    "Recall": f"{m_dict.get('recall', 0)*100:.1f}%",
                    "F1 Score": f"{m_dict.get('f1', 0):.3f}",
                    "ROC-AUC": f"{m_dict.get('roc_auc', 0):.3f}",
                }
            )
        st.dataframe(pd.DataFrame(cls_table), use_container_width=True, hide_index=True)

    with c2:
        st.markdown(f"#### Regression Performance (Best: `{best_reg}`)")
        reg_table = []
        for model_name, m_dict in reg_res.items():
            reg_table.append(
                {
                    "Algorithm": model_name.replace("_", " ").title(),
                    "MAE (Days)": f"{m_dict.get('mae', 0):.1f}",
                    "RMSE (Days)": f"{m_dict.get('rmse', 0):.1f}",
                    "R² Score": f"{m_dict.get('r2', 0):.3f}",
                }
            )
        st.dataframe(pd.DataFrame(reg_table), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔒 Continuous Learning Architecture")
    st.markdown(
        """
        The platform implements a **Safe Governed MLOps Lifecycle** to prevent uncontrolled autonomous drift:
        
        ```
        New Authorized Case Data
                ↓
        Schema & Spatial Coherence Validation (validate_location)
                ↓
        Pre-Processing & Feature Pipeline (Zero Target Leakage)
                ↓
        Candidate Model Training & Validation on Held-Out Test Set
                ↓
        Automated Champion vs. Challenger Metric Comparison
                ↓
        Administrative Oversight & Model Governance Sign-off
                ↓
        Atomic Production Model Hot-Reload
        ```
        
        > **Honest Disclosure:** The prototype does not execute automatic retraining in the wild without human-in-the-loop review. Models are retrained through verified administrative pipelines.
        """
    )

    # Audit Trail Listing
    if st.session_state.get("role") == "admin":
        st.markdown("---")
        st.markdown("### 📋 System Audit Trail (Admin View)")
        audit_res = api("GET", "/audit-logs", params={"limit": 25}).json().get("items", [])
        if audit_res:
            df_audit = pd.DataFrame(audit_res)[["timestamp", "user", "action", "case_id", "previous_value", "new_value"]]
            st.dataframe(df_audit, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------
# Page: Reports & Export
# --------------------------------------------------------------------------


def page_reports():
    render_header("Executive Reports & Export", "Export Formal Project Risk Summaries for District & Ministry Review")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    summary_data = api("GET", "/reports/summary").json().get("cases", [])
    st.caption(f"Showing **{len(summary_data)}** high and critical priority infrastructure projects requiring intervention.")

    if summary_data:
        df_rep = pd.DataFrame(summary_data)
        st.dataframe(df_rep, use_container_width=True, hide_index=True)

        csv_bytes = df_rep.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Executive Risk Digest (CSV)",
            data=csv_bytes,
            file_name=f"Land_Acquisition_Risk_Summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------
# Page: Predict New Case
# --------------------------------------------------------------------------


def page_predict_new():
    render_header("New Project Delay Risk Appraisal", "Evaluate Early-Stage Land Acquisition Risk with Predictive ML")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    all_states = get_all_states()
    with st.form("new_case_form"):
        st.subheader("1. Location & Project Metadata")
        p1, p2, p3 = st.columns(3)
        proj_name = p1.text_input("Project Name", value="Regional Corridor Expansion Phase-1")
        sel_st = p2.selectbox("State / UT", all_states, index=all_states.index("Uttarakhand") if "Uttarakhand" in all_states else 0)
        valid_dist = get_districts_for_state(sel_st)
        sel_dt = p3.selectbox("District", valid_dist)

        p4, p5, p6 = st.columns(3)
        proj_type = p4.selectbox("Sector / Project Type", PROJECT_TYPES)
        land_area = p5.number_input("Land Area (Hectares)", 0.1, 500.0, 15.0)
        landowners = p6.number_input("Number of Landowners", 1, 1000, 35)

        st.subheader("2. Statutory & Field Process Status")
        q1, q2, q3, q4 = st.columns(4)
        surv_comp = q1.selectbox("Survey Completed?", [1, 0], format_func=lambda x: "Completed" if x else "Pending")
        dem_comp = q2.selectbox("Demarcation Completed?", [1, 0], format_func=lambda x: "Completed" if x else "Pending")
        app_comp = q3.selectbox("Clearance Approvals?", [1, 0], format_func=lambda x: "Approved" if x else "Pending")
        comp_paid = q4.selectbox("Compensation Paid?", [0, 1], format_func=lambda x: "Disbursed" if x else "Pending")

        st.subheader("3. Legal & Dispute Roadblocks")
        d1, d2, d3, d4 = st.columns(4)
        own_disp = d1.selectbox("Ownership Dispute?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        leg_disp = d2.selectbox("Court Litigation / Stay?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        num_objs = d3.number_input("Number of Objections", 0, 100, 4)
        doc_pct = d4.slider("Document Completeness (%)", 20, 100, 75)

        save_case = st.checkbox("Save this appraised project directly into the official Case Register", value=True)
        submit_pred = st.form_submit_button("Appraise Acquisition Delay Risk", type="primary")

    if submit_pred:
        payload = {
            "project_name": proj_name,
            "state": sel_st,
            "district": sel_dt,
            "project_type": proj_type,
            "land_area_hectares": land_area,
            "number_of_landowners": landowners,
            "affected_families": max(1, int(landowners * 0.9)),
            "survey_completed": surv_comp,
            "demarcation_completed": dem_comp,
            "approval_completed": app_comp,
            "compensation_paid": comp_paid,
            "ownership_dispute": own_disp,
            "legal_dispute": leg_disp,
            "legal_case": leg_disp,
            "number_of_objections": num_objs,
            "document_completeness_pct": doc_pct,
            "status": "Initiated",
            "current_stage": "Survey" if surv_comp == 0 else "Compensation",
        }

        if save_case:
            resp = api("POST", "/cases", json=payload)
        else:
            resp = api("POST", "/predict", json=payload)

        if resp.status_code == 200:
            data = resp.json()
            pred = data.get("prediction", data)
            st.markdown("---")
            st.markdown(f"### Appraisal Outcome: **{pred['risk_level']} RISK** (Score: {pred['risk_score']}/100)")

            k1, k2, k3 = st.columns(3)
            k1.metric("Delay Probability", f"{round(pred['delay_probability']*100)}%")
            k2.metric("Projected Timeline Impact", f"{pred['expected_delay_days']} days")
            k3.metric("Priority Queue", pred.get("priority_label", "Priority 3"))

            st.info(f"**AI Reasoning:** {pred.get('human_explanation')}")
        else:
            st.error(resp.text)


# --------------------------------------------------------------------------
# Page: Process Assistant
# --------------------------------------------------------------------------


def page_assistant():
    render_header("Land Acquisition Process Assistant", "AI-Guided Operational & Procedural Decision Support")
    st.markdown(f"<div class='disclaimer-card'>{DISCLAIMER}</div>", unsafe_allow_html=True)

    case_id_q = st.text_input("Enter Case ID to query (Optional):", placeholder="e.g. LA-2026-00012")
    user_q = st.text_area("Ask a procedural query:", placeholder="e.g. Why is this case high risk and what are the recommended steps?")

    if st.button("Query Process Assistant"):
        if not user_q.strip():
            st.warning("Please type a question.")
            return
        resp = api("POST", "/assistant/ask", json={"question": user_q, "case_id": case_id_q or None})
        if resp.status_code == 200:
            st.markdown("### Assistant Advisory")
            st.write(resp.json().get("answer"))
        else:
            st.error(resp.text)


# --------------------------------------------------------------------------
# Main App Router
# --------------------------------------------------------------------------


def main():
    st.set_page_config(
        page_title="Land Acquisition Intelligence Platform | SIH",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    if "token" not in st.session_state:
        login_page()
        return

    with st.sidebar:
        st.markdown("### 🏛️ LandAcq AI")
        st.markdown(f"**User:** {st.session_state.get('full_name')}")
        role_label = st.session_state.get('role', 'officer').upper()
        st.markdown(f"<span class='risk-badge risk-badge-LOW' style='font-size:0.7rem;'>ROLE: {role_label}</span>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        pages = [
            "Executive Overview",
            "Case Register",
            "GIS Risk Map",
            "Early Warning Center",
            "State Analytics",
            "District Analytics",
            "What-If Simulator",
            "Appraise New Case",
            "SIH Demo Walkthrough",
            "MLOps & Governance",
            "Reports & Export",
            "Process Assistant",
        ]
        page = st.radio("Navigation", pages, index=0)

        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.clear()
            st.rerun()

        st.caption(f"v2.0.0 · Smart India Hackathon\n{DISCLAIMER[:85]}...")

    if page == "Executive Overview":
        page_overview()
    elif page == "Case Register":
        page_cases()
    elif page == "GIS Risk Map":
        page_gis_map()
    elif page == "Early Warning Center":
        page_early_warning()
    elif page == "State Analytics":
        page_state_analytics()
    elif page == "District Analytics":
        page_district_analytics()
    elif page == "What-If Simulator":
        page_what_if()
    elif page == "Appraise New Case":
        page_predict_new()
    elif page == "SIH Demo Walkthrough":
        page_sih_demo_mode()
    elif page == "MLOps & Governance":
        page_mlops_governance()
    elif page == "Reports & Export":
        page_reports()
    elif page == "Process Assistant":
        page_assistant()


if __name__ == "__main__":
    main()
