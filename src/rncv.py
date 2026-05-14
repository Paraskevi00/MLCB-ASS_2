"""Repeated nested cross-validation.

The key invariant is simple: outer and inner CV receive raw X only. Every
preprocessing/feature-selection operation is contained inside an sklearn
Pipeline, so sklearn refits these steps on the training portion of each split.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score, f1_score,
    recall_score, precision_score, average_precision_score, confusion_matrix,
    make_scorer,
)

from .preprocessing import build_pipeline, get_feature_names_from_pipeline, RANDOM_STATE


def _safe_scores(y_true, y_pred, y_score) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    out = {
        "MCC": matthews_corrcoef(y_true, y_pred),
        "BA": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
    }
    try:
        out["AUC"] = roc_auc_score(y_true, y_score)
    except Exception:
        out["AUC"] = np.nan
    try:
        out["PRAUC"] = average_precision_score(y_true, y_score)
    except Exception:
        out["PRAUC"] = np.nan
    return out


def _positive_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return estimator.predict(X)


def bootstrap_ci(values, statistic=np.median, n_boot=2000, ci=0.95, random_state=RANDOM_STATE):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    stats = [statistic(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return float(np.percentile(stats, 100 * alpha)), float(np.percentile(stats, 100 * (1 - alpha)))


@dataclass
class RNCVConfig:
    n_rounds: int = 10
    n_outer: int = 5
    n_inner: int = 3
    n_iter: int = 40
    random_state: int = RANDOM_STATE
    scoring: str = "roc_auc"
    n_jobs: int = -1


class RepeatedNestedCV:
    """Repeated nested CV runner for binary classification."""

    def __init__(
        self,
        estimators: List[Tuple[str, Any]],
        param_spaces: Dict[str, Any],
        config: Optional[RNCVConfig] = None,
        use_feature_selection: bool = False,
        feature_k_grid: Optional[List[int]] = None,
    ):
        self.estimators = estimators
        self.param_spaces = param_spaces
        self.config = config or RNCVConfig()
        self.use_feature_selection = use_feature_selection
        self.feature_k_grid = feature_k_grid or [3, 5, 8, 10, "all"]
        self.results_: Dict[str, pd.DataFrame] = {}
        self.best_params_: Dict[str, List[Dict[str, Any]]] = {}
        self.selected_features_: Dict[str, List[List[str]]] = {}

    def _space_for(self, name: str):
        base = self.param_spaces.get(name, {})
        if not self.use_feature_selection:
            return base
        # Add selector__k without mutating the original object.
        if isinstance(base, list):
            return [dict(d, **{"feature_select__k": self.feature_k_grid}) for d in base]
        return dict(base, **{"feature_select__k": self.feature_k_grid})

    def run(self, X: pd.DataFrame, y: pd.Series, tune: bool = True) -> Dict[str, pd.DataFrame]:
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.results_, self.best_params_, self.selected_features_ = {}, {}, {}

        for est_name, estimator in self.estimators:
            print(f"\n=== {est_name} | tune={tune} | FS={self.use_feature_selection} ===")
            records: List[Dict[str, Any]] = []
            params_seen: List[Dict[str, Any]] = []
            selected_seen: List[List[str]] = []

            for r in range(self.config.n_rounds):
                outer_seed = self.config.random_state + 1000 * r
                outer_cv = StratifiedKFold(
                    n_splits=self.config.n_outer,
                    shuffle=True,
                    random_state=outer_seed,
                )
                for fold, (tr, te) in enumerate(outer_cv.split(X, y), start=1):
                    X_train, X_test = X.iloc[tr], X.iloc[te]
                    y_train, y_test = y.iloc[tr], y.iloc[te]
                    fold_seed = outer_seed + 17 * fold

                    base_pipe = build_pipeline(
                        clone(estimator),
                        use_feature_selection=self.use_feature_selection,
                        k="all",
                        random_state=fold_seed,
                    )

                    if tune:
                        inner_cv = StratifiedKFold(
                            n_splits=self.config.n_inner,
                            shuffle=True,
                            random_state=fold_seed,
                        )
                        search = RandomizedSearchCV(
                            estimator=base_pipe,
                            param_distributions=self._space_for(est_name),
                            n_iter=self.config.n_iter,
                            scoring=self.config.scoring,
                            cv=inner_cv,
                            refit=True,
                            random_state=fold_seed,
                            n_jobs=self.config.n_jobs,
                            error_score=np.nan,
                        )
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            search.fit(X_train, y_train)
                        fitted = search.best_estimator_
                        best_params = search.best_params_
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fitted = base_pipe.fit(X_train, y_train)
                        best_params = {}

                    y_pred = fitted.predict(X_test)
                    y_score = _positive_scores(fitted, X_test)
                    metrics = _safe_scores(y_test, y_pred, y_score)
                    selected_k = best_params.get("feature_select__k", "all")
                    metrics.update({
                         "estimator": est_name,
                         "round": r + 1,
                         "fold": fold,
                         "selected_k": selected_k,
                         })
                    records.append(metrics)
                    params_seen.append(best_params)

                    if self.use_feature_selection:
                        try:
                            selected_seen.append(list(get_feature_names_from_pipeline(fitted)))
                        except Exception:
                            selected_seen.append([])

                print(f"  round {r+1}/{self.config.n_rounds} complete", end="\r")

            self.results_[est_name] = pd.DataFrame(records)
            self.best_params_[est_name] = params_seen
            self.selected_features_[est_name] = selected_seen
        print("\nDone.")
        return self.results_

    def summarise(self, n_boot: int = 2000) -> pd.DataFrame:
        metrics = ["MCC", "AUC", "BA", "F1", "Recall", "Specificity", "Precision", "PRAUC"]
        rows = []
        for est, df in self.results_.items():
            row = {"estimator": est}
            for m in metrics:
                vals = df[m].astype(float).values
                med = float(np.nanmedian(vals))
                lo, hi = bootstrap_ci(vals, statistic=np.nanmedian, n_boot=n_boot, random_state=self.config.random_state)
                row[f"{m}_median"] = med
                row[f"{m}_CI_lo"] = lo
                row[f"{m}_CI_hi"] = hi
            rows.append(row)
        return pd.DataFrame(rows).set_index("estimator").sort_values("MCC_median", ascending=False)

    def formatted_summary(self, stage: str) -> pd.DataFrame:
        summary = self.summarise()
        metrics = ["MCC", "AUC", "BA", "F1", "Recall", "Specificity", "Precision", "PRAUC"]
        rows = []
        for est in summary.index:
            row = {"Model": est, "Stage": stage}
            for m in metrics:
                row[f"{m} (95% CI)"] = (
                    f"{summary.loc[est, f'{m}_median']:.3f} "
                    f"[{summary.loc[est, f'{m}_CI_lo']:.3f}, {summary.loc[est, f'{m}_CI_hi']:.3f}]"
                )
            rows.append(row)
        return pd.DataFrame(rows)

    def feature_selection_report(self) -> pd.DataFrame:
        rows = []
        total = self.config.n_rounds * self.config.n_outer
        for est, lists in self.selected_features_.items():
            counts: Dict[str, int] = {}
            for feats in lists:
                for f in feats:
                    counts[f] = counts.get(f, 0) + 1
            for feat, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                rows.append({
                    "estimator": est,
                    "feature": feat,
                    "selection_count": count,
                    "selection_frequency_%": 100 * count / total,
                })
        return pd.DataFrame(rows)


def run_default_repeated_cv(estimators, X, y, n_rounds=10, n_outer=5, random_state=RANDOM_STATE):
    """Baseline comparison with repeated stratified 5-fold CV and no tuning."""
    cfg = RNCVConfig(n_rounds=n_rounds, n_outer=n_outer, n_inner=3, n_iter=1, random_state=random_state)
    runner = RepeatedNestedCV(estimators=estimators, param_spaces={}, config=cfg, use_feature_selection=False)
    return runner.run(X, y, tune=False), runner
