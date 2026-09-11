"""Automated end-to-end verification of FastAPI backend and ML services."""

import json
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_root():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert "platform" in data
    assert data["version"] == "2.0.0"
    print("[PASS] GET / passed")


def test_auth():
    # Login as admin
    res = client.post("/auth/login-json", json={"username": "admin", "password": "admin123"})
    assert res.status_code == 200
    token = res.json()["access_token"]
    assert token

    # Test /me
    me_res = client.get("/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    assert me_res.json()["role"] == "admin"

    # Login as officer
    res_off = client.post("/auth/login-json", json={"username": "officer", "password": "officer123"})
    assert res_off.status_code == 200
    assert res_off.json()["role"] == "officer"
    print("[PASS] Auth endpoints passed (admin & officer)")


def test_locations():
    # Test all 36 States/UTs
    res = client.get("/states")
    assert res.status_code == 200
    states = res.json()
    assert len(states) == 36
    assert "Uttarakhand" in states
    assert "Maharashtra" in states

    # Test districts of Uttarakhand
    uk_res = client.get("/states/Uttarakhand/districts")
    assert uk_res.status_code == 200
    uk_districts = uk_res.json()
    assert len(uk_districts) == 13
    assert "Dehradun" in uk_districts
    assert "Almora" in uk_districts

    # Test validation endpoint
    val_true = client.get("/location/validate?state=Uttarakhand&district=Dehradun")
    assert val_true.json()["is_valid"] is True

    val_false = client.get("/location/validate?state=Uttarakhand&district=Pune")
    assert val_false.json()["is_valid"] is False
    print("[PASS] Location master endpoints passed (36 States/UTs, 13 UK districts, strict validation)")


def test_dashboard_stats():
    res_login = client.post("/auth/login-json", json={"username": "officer", "password": "officer123"})
    token = res_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    res = client.get("/dashboard/stats", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total_cases"] >= 5000
    assert "risk_distribution" in data
    assert "CRITICAL" in data["risk_distribution"]
    print(f"[PASS] Dashboard stats passed ({data['total_cases']} total cases, avg risk score: {data['average_risk_score']})")


def test_cases_and_details():
    res_login = client.post("/auth/login-json", json={"username": "officer", "password": "officer123"})
    token = res_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # List cases
    res = client.get("/cases?limit=10", headers=headers)
    assert res.status_code == 200
    items = res.json()["items"]
    assert len(items) == 10

    case_id = items[0]["case_id"]
    detail_res = client.get(f"/cases/{case_id}", headers=headers)
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["case_id"] == case_id
    assert "live_explanation" in detail
    assert "timeline" in detail
    assert len(detail["timeline"]) == 11
    print(f"[PASS] Case listing and details passed (11-stage timeline verified for {case_id})")


def test_prediction_and_what_if():
    res_login = client.post("/auth/login-json", json={"username": "officer", "password": "officer123"})
    token = res_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Predict
    payload = {
        "project_name": "NH-48 Expressway Feeder",
        "state": "Uttarakhand",
        "district": "Dehradun",
        "project_type": "National Highway",
        "land_area_hectares": 20.0,
        "affected_families": 40,
        "number_of_landowners": 45,
        "ownership_dispute": 1,
        "legal_dispute": 1,
        "legal_case": 1,
        "document_completeness_pct": 60.0,
        "survey_completed": 1,
        "demarcation_completed": 0,
        "approval_completed": 0,
        "compensation_paid": 0,
        "number_of_objections": 6,
        "historical_delay_rate_pct": 30.0,
        "distance_to_project_km": 5.0,
        "current_stage": "Approval",
        "status": "Initiated",
    }
    pred_res = client.post("/predict", json=payload, headers=headers)
    assert pred_res.status_code == 200
    pdata = pred_res.json()
    assert 0 <= pdata["risk_score"] <= 100
    assert pdata["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert len(pdata["factors_increasing_risk"]) > 0
    assert len(pdata["factors_reducing_risk"]) > 0

    # What-If Simulator
    what_if_res = client.post(
        "/what-if",
        json={
            "base_case": payload,
            "compensation_paid": 1,
            "compensation_status": "Disbursed",
            "legal_dispute": 0,
            "document_completeness_pct": 95.0,
        },
        headers=headers,
    )
    assert what_if_res.status_code == 200
    wdata = what_if_res.json()
    assert wdata["risk_score_delta"] < 0
    assert "summary" in wdata
    print(f"[PASS] Predict and What-If simulator passed (Risk delta: {wdata['risk_score_delta']} pts, Delay delta: {wdata['delay_days_delta']} days)")


def test_analytics_and_gis():
    res_login = client.post("/auth/login-json", json={"username": "officer", "password": "officer123"})
    token = res_login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # State analytics
    st_res = client.get("/analytics/state", headers=headers)
    assert st_res.status_code == 200
    assert len(st_res.json()["states"]) > 0

    # District analytics
    dt_res = client.get("/analytics/district?state=Uttarakhand", headers=headers)
    assert dt_res.status_code == 200
    assert len(dt_res.json()["districts"]) > 0

    # GIS points
    gis_res = client.get("/gis/points?state=Uttarakhand&limit=50", headers=headers)
    assert gis_res.status_code == 200
    assert len(gis_res.json()["items"]) > 0

    # Reports
    rep_res = client.get("/reports/summary", headers=headers)
    assert rep_res.status_code == 200
    assert "cases" in rep_res.json()
    print("[PASS] State analytics, District analytics, GIS points, and Reports passed")


if __name__ == "__main__":
    test_root()
    test_auth()
    test_locations()
    test_dashboard_stats()
    test_cases_and_details()
    test_prediction_and_what_if()
    test_analytics_and_gis()
    print("\nALL BACKEND ENDPOINTS AND LOGIC VERIFIED SUCCESSFULLY!")
