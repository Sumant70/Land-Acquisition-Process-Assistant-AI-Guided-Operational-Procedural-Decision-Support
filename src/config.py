"""Shared paths, schemas, and constants for the Land Acquisition AI DSS platform."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_DIR = DATA_PROCESSED_DIR / "eda"

MASTER_LOCATION_PATH = PROJECT_ROOT / "data" / "india_states_districts.json"
RAW_DATASET_PATH = DATA_RAW_DIR / "synthetic_land_acquisition_cases.csv"
CLEAN_DATASET_PATH = DATA_PROCESSED_DIR / "clean_land_acquisition_cases.csv"
FEATURED_DATASET_PATH = DATA_PROCESSED_DIR / "featured_land_acquisition_cases.csv"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_COLUMNS_PATH = MODELS_DIR / "feature_columns.json"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
CLASSIFIER_PATH = MODELS_DIR / "best_classifier.joblib"
REGRESSOR_PATH = MODELS_DIR / "best_regressor.joblib"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.joblib"

DATASET_DISCLAIMER = (
    "Synthetic Prototype Dataset for Demonstration & Model Development. "
    "Records, simulated parameters, and map coordinates are generated for Smart India Hackathon (SIH) prototype evaluation. "
    "They are NOT real government or judicial land acquisition records."
)

# Target columns - STRICTLY EXCLUDED FROM MODEL INPUTS (Zero Target Leakage)
TARGET_CLASS = "delayed"
TARGET_REGRESSION = "expected_delay_days"
LEAKAGE_COLUMNS = {
    TARGET_CLASS,
    TARGET_REGRESSION,
    "risk_score",
    "delay_probability",
    "no_delay_probability",
    "risk_level",
    "predicted_delay_days",
}

PROJECT_TYPES = [
    "National Highway",
    "Railway",
    "Metro Rail",
    "Irrigation & Canal",
    "Power Transmission & Renewable",
    "Industrial Corridor & SEZ",
    "Airport",
    "Defense & Strategic Road",
    "Smart City Infrastructure",
]

LIFECYCLE_STAGES = [
    "Preliminary Planning",
    "Notification",
    "Survey",
    "Land Identification",
    "Ownership Verification",
    "Objection Handling",
    "Approval",
    "Compensation",
    "Rehabilitation & Resettlement",
    "Possession",
    "Completion",
]

CATEGORICAL_COLUMNS = [
    "state",
    "district",
    "project_type",
    "notification_status",
    "survey_status",
    "demarcation_status",
    "approval_status",
    "compensation_status",
    "possession_status",
    "rehabilitation_status",
    "resettlement_status",
    "stakeholder_responsiveness",
    "administrative_bottleneck",
    "inter_department_coordination",
    "current_stage",
]

BOOLEAN_COLUMNS = [
    "ownership_dispute",
    "legal_dispute",
    "survey_completed",
    "demarcation_completed",
    "compensation_paid",
    "approval_completed",
]

NUMERIC_COLUMNS = [
    "land_area_hectares",
    "affected_families",
    "number_of_landowners",
    "number_of_legal_cases",
    "number_of_objections",
    "document_completeness_pct",
    "approval_delay_days",
    "compensation_delay_days",
    "historical_delay_rate_pct",
    "distance_to_project_km",
]

RANDOM_SEED = 42
TEST_SIZE = 0.2
