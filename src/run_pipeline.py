"""Run dataset generation → cleaning → EDA → features → training in order."""

from src.config import CLEAN_DATASET_PATH, DATA_PROCESSED_DIR, DATA_RAW_DIR, FEATURED_DATASET_PATH, RAW_DATASET_PATH
from src.data_cleaning import clean_dataset
from src.eda import run_eda
from src.feature_engineering import add_engineered_features
from src.generate_dataset import generate_dataset, DATASET_DISCLAIMER
from src.train_model import train


def main() -> None:
    print("=== 1. Generate synthetic dataset ===")
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(n_rows=5500)
    df.to_csv(RAW_DATASET_PATH, index=False)
    print(DATASET_DISCLAIMER)
    print(f"Wrote {len(df)} rows to {RAW_DATASET_PATH}")

    print("\n=== 2. Clean and validate ===")
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean, report = clean_dataset(RAW_DATASET_PATH)
    clean.to_csv(CLEAN_DATASET_PATH, index=False)
    print(f"Wrote {CLEAN_DATASET_PATH}")

    print("\n=== 3. EDA ===")
    run_eda()

    print("\n=== 4. Feature engineering ===")
    featured = add_engineered_features(clean)
    featured.to_csv(FEATURED_DATASET_PATH, index=False)
    print(f"Wrote {FEATURED_DATASET_PATH}")

    print("\n=== 5. Train models (this can take a few minutes) ===")
    train()
    print("\nPipeline complete. Models are in the models/ folder.")


if __name__ == "__main__":
    main()
