"""Exploratory data analysis. Saves plots headlessly; does not train models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import CLEAN_DATASET_PATH, DATASET_DISCLAIMER, EDA_DIR

sns.set_theme(style="whitegrid")


def _save(fig: plt.Figure, name: str) -> None:
    EDA_DIR.mkdir(parents=True, exist_ok=True)
    path = EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Saved {path}")


def run_eda(path: Path = CLEAN_DATASET_PATH) -> None:
    df = pd.read_csv(path)
    print(DATASET_DISCLAIMER)
    print("\nShape:", df.shape)
    print("\nDelay rate:\n", df["delayed"].value_counts(normalize=True))
    print("\nExpected delay days describe:\n", df["expected_delay_days"].describe())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.histplot(df["expected_delay_days"], bins=35, ax=ax, color="#1e3a8a", kde=True)
    ax.set_title("Expected Delay Days Distribution — Prototype")
    _save(fig, "delay_days_histogram.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x="delayed", data=df, ax=ax, palette=["#059669", "#dc2626"])
    ax.set_title("Delayed vs On-Time Distribution — Prototype")
    _save(fig, "delayed_count.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    order = df.groupby("project_type")["delayed"].mean().sort_values(ascending=False).index
    sns.barplot(x="project_type", y="delayed", data=df, order=order, ax=ax, palette="Blues_r")
    ax.set_ylabel("Delay Rate")
    ax.set_title("Delay Rate by Project Type — Prototype")
    ax.tick_params(axis="x", rotation=30)
    _save(fig, "delay_rate_by_project_type.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    corr_cols = [c for c in [
        "land_area_hectares",
        "affected_families",
        "number_of_landowners",
        "ownership_dispute",
        "legal_dispute",
        "number_of_objections",
        "document_completeness_pct",
        "survey_completed",
        "demarcation_completed",
        "compensation_paid",
        "approval_completed",
        "approval_delay_days",
        "compensation_delay_days",
        "historical_delay_rate_pct",
        "expected_delay_days",
        "delayed",
    ] if c in df.columns]

    sns.heatmap(df[corr_cols].corr(numeric_only=True), cmap="RdYlBu_r", center=0, ax=ax, annot=False)
    ax.set_title("Feature Correlation Heatmap — Prototype")
    _save(fig, "correlation_heatmap.png")

    summary_path = EDA_DIR / "summary.csv"
    df.describe(include="all").to_csv(summary_path)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    run_eda()
