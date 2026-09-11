"""Train classification and regression models. Metrics are computed on a held-out test set."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.config import (
    CATEGORICAL_COLUMNS,
    FEATURED_DATASET_PATH,
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODELS_DIR,
    PREPROCESSOR_PATH,
    RANDOM_SEED,
    SHAP_BACKGROUND_PATH,
    TARGET_CLASS,
    TARGET_REGRESSION,
    TEST_SIZE,
    CLASSIFIER_PATH,
    REGRESSOR_PATH,
)
from src.feature_engineering import feature_matrix


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler(), num),
        ]
    )


def classification_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def train() -> dict:
    df = pd.read_csv(FEATURED_DATASET_PATH)
    X = feature_matrix(df)
    y_cls = df[TARGET_CLASS].astype(int)
    y_reg = df[TARGET_REGRESSION].astype(float)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X,
        y_cls,
        y_reg,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_cls,
    )

    preprocessor = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    X_train_p = preprocessor.transform(X_train)
    X_test_p = preprocessor.transform(X_test)

    classifiers = {
        "random_forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            max_depth=4,
            max_features="sqrt",
            learning_rate=0.08,
            random_state=RANDOM_SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric="logloss",
            scale_pos_weight=float((y_cls_train == 0).sum() / max((y_cls_train == 1).sum(), 1)),
        ),
    }

    cls_results = {}
    best_cls_name = None
    best_cls_auc = -1.0
    best_cls_model = None
    for name, model in classifiers.items():
        print(f"Training classifier: {name}...", flush=True)
        model.fit(X_train_p, y_cls_train)
        prob = model.predict_proba(X_test_p)[:, 1]
        pred = (prob >= 0.5).astype(int)
        metrics = classification_metrics(y_cls_test, pred, prob)
        cls_results[name] = metrics
        print(f"[Classification] {name}: {metrics}", flush=True)
        if metrics["roc_auc"] > best_cls_auc:
            best_cls_auc = metrics["roc_auc"]
            best_cls_name = name
            best_cls_model = model

    regressors = {
        "random_forest": RandomForestRegressor(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=4,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=120,
            max_depth=4,
            max_features="sqrt",
            learning_rate=0.08,
            random_state=RANDOM_SEED,
        ),
        "xgboost": XGBRegressor(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    reg_results = {}
    best_reg_name = None
    best_reg_rmse = float("inf")
    best_reg_model = None
    for name, model in regressors.items():
        print(f"Training regressor: {name}...", flush=True)
        model.fit(X_train_p, y_reg_train)
        pred = model.predict(X_test_p)
        metrics = regression_metrics(y_reg_test, pred)
        reg_results[name] = metrics
        print(f"[Regression] {name}: {metrics}", flush=True)
        if metrics["rmse"] < best_reg_rmse:
            best_reg_rmse = metrics["rmse"]
            best_reg_name = name
            best_reg_model = model

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(best_cls_model, CLASSIFIER_PATH)
    joblib.dump(best_reg_model, REGRESSOR_PATH)

    # Small background sample for SHAP (TreeExplainer is fast on trees).
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(X_train_p.shape[0], size=min(200, X_train_p.shape[0]), replace=False)
    joblib.dump(X_train_p[idx], SHAP_BACKGROUND_PATH)

    feature_names = preprocessor.get_feature_names_out().tolist()
    FEATURE_COLUMNS_PATH.write_text(json.dumps({"input_columns": list(X.columns), "transformed_names": feature_names}, indent=2))

    report = {
        "dataset": "Synthetic Dataset for Prototype and Model Development",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classification": cls_results,
        "best_classifier": best_cls_name,
        "regression": reg_results,
        "best_regressor": best_reg_name,
        "selection_rule": {
            "classifier": "highest ROC-AUC on held-out test set",
            "regressor": "lowest RMSE on held-out test set",
        },
        "note": "Metrics are computed on held-out test data. Do not treat them as production government-system accuracy.",
    }
    METRICS_PATH.write_text(json.dumps(report, indent=2))
    print("\nBest classifier:", best_cls_name, cls_results[best_cls_name])
    print("Best regressor:", best_reg_name, reg_results[best_reg_name])
    print(f"Wrote metrics to {METRICS_PATH}")
    return report


if __name__ == "__main__":
    train()
