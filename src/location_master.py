"""
Master location registry for India States, Union Territories, and Districts.
Provides authoritative LGD-compatible lists, coordinates, and validation functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MASTER_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "india_states_districts.json"

_LOCATION_CACHE: dict[str, Any] | None = None


def load_master_data() -> dict[str, Any]:
    global _LOCATION_CACHE
    if _LOCATION_CACHE is None:
        if not MASTER_DATA_PATH.exists():
            raise FileNotFoundError(f"Master location file not found at {MASTER_DATA_PATH}")
        with open(MASTER_DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        _LOCATION_CACHE = {
            "states": data,
            "state_dict": {s["state"]: s for s in data},
            "valid_pairs": {
                (s["state"].strip().lower(), d["name"].strip().lower() if isinstance(d, dict) else d.strip().lower())
                for s in data
                for d in s["districts"]
            },
        }
    return _LOCATION_CACHE


def get_all_states() -> list[str]:
    data = load_master_data()
    return sorted(s["state"] for s in data["states"])


def get_districts_for_state(state: str) -> list[str]:
    data = load_master_data()
    state_obj = data["state_dict"].get(state)
    if not state_obj:
        return []
    districts = state_obj.get("districts", [])
    names = [d["name"] if isinstance(d, dict) else d for d in districts]
    return sorted(names)


def validate_location(state: str, district: str) -> bool:
    """Return True only if district belongs to the selected state/UT."""
    if not state or not district:
        return False
    data = load_master_data()
    return (state.strip().lower(), district.strip().lower()) in data["valid_pairs"]


def get_district_coordinates(state: str, district: str) -> tuple[float, float]:
    """Return representative (latitude, longitude) for district centroid."""
    data = load_master_data()
    state_obj = data["state_dict"].get(state)
    if state_obj:
        for d in state_obj.get("districts", []):
            if isinstance(d, dict) and d["name"].lower() == district.lower():
                return float(d.get("latitude", state_obj.get("latitude", 22.5))), float(d.get("longitude", state_obj.get("longitude", 79.0)))
        return float(state_obj.get("latitude", 22.5)), float(state_obj.get("longitude", 79.0))
    return 22.9734, 78.6569


def get_state_details(state: str) -> dict[str, Any] | None:
    data = load_master_data()
    return data["state_dict"].get(state)


def get_all_locations() -> list[tuple[str, str, float, float]]:
    """Return all valid (state, district, latitude, longitude) tuples from the master dataset."""
    data = load_master_data()
    results = []
    for s in data["states"]:
        state_name = s["state"]
        state_lat = float(s.get("latitude", 22.5))
        state_lon = float(s.get("longitude", 79.0))
        for d in s.get("districts", []):
            if isinstance(d, dict):
                d_name = d["name"]
                d_lat = float(d.get("latitude", state_lat))
                d_lon = float(d.get("longitude", state_lon))
            else:
                d_name = str(d)
                d_lat, d_lon = state_lat, state_lon
            results.append((state_name, d_name, d_lat, d_lon))
    return results
