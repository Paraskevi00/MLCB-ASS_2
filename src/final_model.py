"""Final model training, saving, SHAP, and error-analysis helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .preprocessing import build_pipeline, get_feature_names_from_pipeline, RANDOM_STATE
from .rncv import _positive_scores, _safe_scores

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(exist_ok=True, parents=True)


def select_best_hyperparams_cv5(
    estimator,
    param_space: Dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    use_feature_selection: bool = False,
    feature_k_grid=None,
    n_iter: int = 60,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
):
    """Final 5-fold CV search on all available data.

    The estimator is a complete raw-data Pipeline, so all preprocessing and
    feature selection are refit inside each CV split.
    """
    pipe = build_pipeline(clone(estimator), use_feature_selection=use_feature_selection, k="all", random_state=random_state)
    space = dict(param_space) if not isinstance(param_space, list) else param_space
    if use_feature_selection:
        feature_k_grid = feature_k_grid or [3, 5, 8, 10, "all"]
        if isinstance(space, list):
            space = [dict(s, **{"feature_select__k": feature_k_grid}) for s in space]
        else:
            space["feature_select__k"] = feature_k_grid
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    search = RandomizedSearchCV(
        pipe, space, n_iter=n_iter, cv=cv, scoring=scoring, refit=True,
        random_state=random_state, n_jobs=n_jobs, error_score=np.nan,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X, y)
    return search


def train_and_save_final_pipeline(search: RandomizedSearchCV, model_filename: str = "best_model.pkl"):
    """Save the already refitted best estimator as a complete raw-data Pipeline."""
    final_pipeline = search.best_estimator_
    out_path = MODELS_DIR / model_filename
    with open(out_path, "wb") as f:
        pickle.dump(final_pipeline, f)
    return final_pipeline, out_path


def load_final_pipeline(model_filename: str = "best_model.pkl"):
    with open(MODELS_DIR / model_filename, "rb") as f:
        return pickle.load(f)


def transformed_feature_frame(fitted_pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Return the numeric matrix seen by the classifier with correct feature names."""
    preprocess = fitted_pipeline.named_steps["preprocess"]
    Xt = preprocess.transform(X)
    names = preprocess.get_feature_names_out()
    if "feature_select" in fitted_pipeline.named_steps:
        selector = fitted_pipeline.named_steps["feature_select"]
        if hasattr(selector, "get_support"):
            Xt = selector.transform(Xt)
            names = names[selector.get_support()]
    return pd.DataFrame(Xt, columns=names, index=X.index)


def compute_shap_values(final_pipeline, X):
    import shap
    import pandas as pd

    classifier = final_pipeline.named_steps["classifier"]

    # Apply every fitted step before the classifier:
    # preprocessing + feature selection
    X_model = final_pipeline[:-1].transform(X)

    # Get feature names after preprocessing
    feature_names = final_pipeline.named_steps["preprocess"].get_feature_names_out()

    # If feature selection exists, keep only selected names
    if "feature_select" in final_pipeline.named_steps:
        selector = final_pipeline.named_steps["feature_select"]
        if hasattr(selector, "get_support"):
            feature_names = feature_names[selector.get_support()]

    X_model = pd.DataFrame(X_model, columns=feature_names)

    clf_name = classifier.__class__.__name__.lower()

    if "forest" in clf_name or "xgb" in clf_name or "lgbm" in clf_name or "catboost" in clf_name:
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_model)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    elif "logistic" in clf_name or "linear" in clf_name or "discriminant" in clf_name:
        explainer = shap.LinearExplainer(classifier, X_model)
        shap_values = explainer.shap_values(X_model)
    else:
        background = shap.sample(X_model, min(50, len(X_model)), random_state=42)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(X_model)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    return shap_values, X_model, explainer


def validation_error_analysis(final_estimator, X: pd.DataFrame, y: pd.Series, test_size: float = 0.25,
                              random_state: int = RANDOM_STATE):
    """Bonus helper: train/validation split, predictions, FP/FN grouping."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = clone(final_estimator)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_score = _positive_scores(model, X_val)
    metrics = _safe_scores(y_val, y_pred, y_score)
    groups = pd.DataFrame(X_val).copy()
    groups["y_true"] = y_val.values
    groups["y_pred"] = y_pred
    groups["error_group"] = np.select(
        [
            (groups.y_true == 1) & (groups.y_pred == 1),
            (groups.y_true == 0) & (groups.y_pred == 0),
            (groups.y_true == 0) & (groups.y_pred == 1),
            (groups.y_true == 1) & (groups.y_pred == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="unknown",
    )
    return model, metrics, groups, classification_report(y_val, y_pred), confusion_matrix(y_val, y_pred)
