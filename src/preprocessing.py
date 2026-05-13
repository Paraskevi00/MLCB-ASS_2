"""Preprocessing and model registry for Assignment #2.

Important design choice: this module does NOT one-hot encode, impute, or scale
before cross-validation. It only defines raw column groups and functions that
build sklearn Pipeline/ColumnTransformer objects. The transformers are fitted
inside each CV fold by sklearn, preventing data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def mutual_info_score_func(X, y):
    """Pickle-safe mutual information score function for SelectKBest."""
    return mutual_info_classif(X, y, random_state=RANDOM_STATE)

def make_onehot_encoder():
    """Version-compatible dense OneHotEncoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import FunctionTransformer

RANDOM_STATE = 42
TARGET_COL = "num"

RAW_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
ORDINAL_FEATURES = ["ca"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]


def load_heart_disease(data_path: str | Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the raw Cleveland heart-disease CSV.

    Returns raw 13-column features and binary target. Missing values coded as
    '?' or blank are left as NaN so the Pipeline can impute them inside CV.
    """
    data_path = Path(data_path)
    df = pd.read_csv(data_path, na_values=["?", "", "NA", "nan"])
    df.columns = [str(c).strip() for c in df.columns]

    if TARGET_COL not in df.columns:
        df = pd.read_csv(
            data_path,
            header=None,
            names=RAW_FEATURES + [TARGET_COL],
            na_values=["?", "", "NA", "nan"],
        )

    missing_cols = [c for c in RAW_FEATURES + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    df = df[RAW_FEATURES + [TARGET_COL]].copy()
    for c in RAW_FEATURES + [TARGET_COL]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    y = df[TARGET_COL].astype(int)
    # Safety: if the original UCI multi-class target appears, binarize it.
    y = (y > 0).astype(int)
    X = df[RAW_FEATURES].copy()
    return X, y


def make_preprocessor() -> ColumnTransformer:
    """Create a raw-data preprocessor fitted only when the Pipeline is fitted.

    Continuous and ordinal columns: median imputation + standardization.
    Binary columns: most-frequent imputation, kept numeric.
    Nominal categoricals: most-frequent imputation + one-hot encoding.
    """
    continuous_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, CONTINUOUS_FEATURES + ORDINAL_FEATURES),
            ("binary", binary_pipe, BINARY_FEATURES),
            ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_feature_selector(k: int | str = "all", random_state: int = RANDOM_STATE):
    """Model-agnostic feature selector.

    Mutual information is fitted inside the Pipeline on training folds only.
    k='all' keeps all features and is compatible with feature_select__k.
    """
    return SelectKBest(score_func=mutual_info_score_func, k=k)


def build_pipeline(estimator, use_feature_selection: bool = False,
                   k: int | str = "all", random_state: int = RANDOM_STATE) -> Pipeline:
    """Build the complete inference pipeline: preprocess -> optional FS -> model."""
    steps = [("preprocess", make_preprocessor())]
    if use_feature_selection:
        steps.append(("feature_select", make_feature_selector(k=k, random_state=random_state)))
    steps.append(("classifier", estimator))
    return Pipeline(steps)


def get_feature_names_from_pipeline(fitted_pipeline: Pipeline) -> np.ndarray:
    """Return names after preprocessing and optional feature selection."""
    names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    if "feature_select" in fitted_pipeline.named_steps:
        selector = fitted_pipeline.named_steps["feature_select"]
        if hasattr(selector, "get_support"):
            names = names[selector.get_support()]
    return np.asarray(names)


def get_estimators_and_param_spaces(random_state: int = RANDOM_STATE):
    """Return estimators and randomized-search spaces using pipeline param names."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier

    estimators: List[Tuple[str, Any]] = [
        ("LR_ElasticNet", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            class_weight="balanced", max_iter=5000, random_state=random_state,
        )),
        ("GNB", GaussianNB()),
        ("LDA", LinearDiscriminantAnalysis()),
        ("RF", RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state,
            n_jobs=-1,
        )),
    ]

    spaces: Dict[str, Dict[str, Any]] = {
        "LR_ElasticNet": {
            "classifier__C": loguniform(1e-3, 30),
            "classifier__l1_ratio": uniform(0, 1),
        },
        "GNB": {
            "classifier__var_smoothing": loguniform(1e-11, 1e-1),
        },
        "LDA": [
            {"classifier__solver": ["svd"]},
            {"classifier__solver": ["lsqr", "eigen"], "classifier__shrinkage": uniform(0, 1)},
        ],
        "RF": {
            "classifier__n_estimators": randint(100, 700),
            "classifier__max_depth": [None, 2, 3, 4, 5, 8, 12],
            "classifier__min_samples_split": randint(2, 25),
            "classifier__min_samples_leaf": randint(1, 12),
            "classifier__max_features": ["sqrt", "log2", 0.5, None],
        },
    }

    try:
        from lightgbm import LGBMClassifier
        estimators.append(("LightGBM", LGBMClassifier(
            objective="binary", class_weight="balanced", random_state=random_state,
            verbosity=-1,
        )))
        spaces["LightGBM"] = {
            "classifier__n_estimators": randint(80, 600),
            "classifier__num_leaves": randint(4, 64),
            "classifier__max_depth": [-1, 2, 3, 4, 5, 8],
            "classifier__learning_rate": loguniform(1e-3, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
            "classifier__reg_alpha": loguniform(1e-4, 10),
            "classifier__reg_lambda": loguniform(1e-4, 10),
            "classifier__min_child_samples": randint(5, 50),
        }
    except Exception:
        pass

    try:
        from xgboost import XGBClassifier
        estimators.append(("XGBoost", XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )))
        spaces["XGBoost"] = {
            "classifier__n_estimators": randint(80, 600),
            "classifier__max_depth": randint(2, 8),
            "classifier__learning_rate": loguniform(1e-3, 0.3),
            "classifier__subsample": uniform(0.6, 0.4),
            "classifier__colsample_bytree": uniform(0.6, 0.4),
            "classifier__reg_alpha": loguniform(1e-4, 10),
            "classifier__reg_lambda": loguniform(1e-4, 10),
            "classifier__min_child_weight": randint(1, 10),
        }
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier
        estimators.append(("CatBoost", CatBoostClassifier(
            loss_function="Logloss", auto_class_weights="Balanced",
            random_seed=random_state, verbose=False,
        )))
        spaces["CatBoost"] = {
            "classifier__iterations": randint(80, 600),
            "classifier__depth": randint(2, 8),
            "classifier__learning_rate": loguniform(1e-3, 0.3),
            "classifier__l2_leaf_reg": loguniform(1e-3, 20),
        }
    except Exception:
        pass

    return estimators, spaces
