"""
Behavioral Risk Classifier — Phase 2.

Pipeline: StandardScaler + {LogisticRegression | XGBoost | RandomForest}
Training: SMOTE on train split only, stratified 70/15/15, 5-fold GridSearchCV.
Output:   support_recommendation_score 0–100 (calibrated probability × 100).
          Never a clinical label. Every prediction includes top-3 SHAP features.

Score bands (see CLAUDE.md §Output Design):
    0–40   baseline
   41–65   check-in suggested
   66–85   outreach recommended
   86–100  priority follow-up
"""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

FEATURE_COLS: list[str] = [
    "engagement_variance",
    "sleep_schedule_drift",
    "social_activity_decline",
    "academic_trend",
    "missed_class_streak",
    "financial_stress_flag",
    "help_seeking_flag",
    "self_report_score",
]

DEMOGRAPHIC_COLS: list[str] = [
    "gender",
    "race_ethnicity",
    "first_gen",
    "international_student",
    "financial_aid_status",
]

TARGET_COL = "support_recommended"

SCORE_BANDS: dict[str, tuple[int, int]] = {
    "baseline": (0, 40),
    "check_in_suggested": (41, 65),
    "outreach_recommended": (66, 85),
    "priority_follow_up": (86, 100),
}


def score_to_band(score: float) -> str:
    s = int(score)
    for band, (lo, hi) in SCORE_BANDS.items():
        if lo <= s <= hi:
            return band
    return "priority_follow_up"


@dataclass
class PredictionResult:
    score: float
    score_band: str
    calibrated_probability: float
    top_features: list[dict]  # [{"feature": str, "direction": str, "shap_value": float}]
    display_text: str


@dataclass
class ModelBundle:
    model_name: Literal["xgboost", "logistic_regression", "random_forest"]
    calibrated_pipeline: object       # CalibratedClassifierCV wrapping Pipeline(scaler, clf)
    raw_pipeline: object              # Pipeline(scaler, clf) — needed for SHAP TreeExplainer
    feature_names: list[str]
    train_metrics: dict = field(default_factory=dict)
    val_metrics: dict = field(default_factory=dict)
    cv_f1_mean: float = 0.0
    cv_f1_std: float = 0.0
    threshold: float = 0.5            # decision threshold (may be adjusted post bias-audit)
    trained_at: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _make_base_pipeline(
    model_name: str,
    random_state: int = 42,
) -> Pipeline:
    if model_name == "logistic_regression":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            verbosity=0,
        )
    elif model_name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _hyperparameter_grid(model_name: str) -> dict:
    if model_name == "logistic_regression":
        return {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__penalty": ["l2"]}
    elif model_name == "xgboost":
        return {
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__n_estimators": [100, 200],
        }
    elif model_name == "random_forest":
        return {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 5],
        }
    return {}


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "positive_rate": float(y_pred.mean()),
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def train(
    df: pd.DataFrame,
    model_name: Literal["xgboost", "logistic_regression", "random_forest"] = "xgboost",
    feature_cols: list[str] = FEATURE_COLS,
    target_col: str = TARGET_COL,
    random_state: int = 42,
    use_smote: bool = True,
    cv_folds: int = 5,
    tune_hyperparams: bool = True,
    verbose: bool = True,
) -> ModelBundle:
    """
    Train a behavioral risk classifier.

    SMOTE applied to training split only — never leaks into val/test.
    Calibrated via Platt scaling (CalibratedClassifierCV).
    """
    from imblearn.over_sampling import SMOTE

    X = df[feature_cols].values
    y = df[target_col].values

    # Stratified 70/15/15 split
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
    train_val_idx, test_idx = next(sss_test.split(X, y))

    X_trainval, y_trainval = X[train_val_idx], y[train_val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.85,  # 15% of full data out of the 85% remaining
        random_state=random_state,
    )
    train_idx, val_idx = next(sss_val.split(X_trainval, y_trainval))
    X_train, y_train = X_trainval[train_idx], y_trainval[train_idx]
    X_val, y_val = X_trainval[val_idx], y_trainval[val_idx]

    if verbose:
        print(f"Split — train: {len(X_train):,} | val: {len(X_val):,} | test: {len(X_test):,}")
        print(f"Train positive rate: {y_train.mean():.3f}")

    # SMOTE on training data only (before pipeline fit)
    if use_smote:
        smote = SMOTE(random_state=random_state)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        if verbose:
            print(f"After SMOTE — train: {len(X_train_res):,} | positive rate: {y_train_res.mean():.3f}")
    else:
        X_train_res, y_train_res = X_train, y_train

    pipeline = _make_base_pipeline(model_name, random_state)

    # GridSearchCV hyperparameter tuning
    if tune_hyperparams:
        param_grid = _hyperparameter_grid(model_name)
        if param_grid:
            gs = GridSearchCV(
                pipeline,
                param_grid,
                cv=cv_folds,
                scoring="f1",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train_res, y_train_res)
            pipeline = gs.best_estimator_
            if verbose:
                print(f"Best params: {gs.best_params_}")
                print(f"CV best F1: {gs.best_score_:.4f}")
        else:
            pipeline.fit(X_train_res, y_train_res)
    else:
        pipeline.fit(X_train_res, y_train_res)

    # 5-fold CV on final pipeline (post-tuning, on resampled train)
    cv_scores = cross_val_score(pipeline, X_train_res, y_train_res, cv=cv_folds, scoring="f1")
    cv_f1_mean, cv_f1_std = float(cv_scores.mean()), float(cv_scores.std())
    if verbose:
        print(f"5-fold CV F1: {cv_f1_mean:.4f} ± {cv_f1_std:.4f}")

    # Platt scaling calibration (fit on val set, not train)
    calibrated = CalibratedClassifierCV(pipeline, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)

    # Metrics
    val_prob = calibrated.predict_proba(X_val)[:, 1]
    val_pred = (val_prob >= 0.5).astype(int)
    val_metrics = _compute_metrics(y_val, val_pred, val_prob)

    test_prob = calibrated.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    train_metrics = _compute_metrics(y_test, test_pred, test_prob)

    if verbose:
        print(f"\nVal  — F1: {val_metrics['f1']:.4f} | AUC: {val_metrics['roc_auc']:.4f} | Brier: {val_metrics['brier_score']:.4f}")
        print(f"Test — F1: {train_metrics['f1']:.4f} | AUC: {train_metrics['roc_auc']:.4f} | Brier: {train_metrics['brier_score']:.4f}")

    return ModelBundle(
        model_name=model_name,
        calibrated_pipeline=calibrated,
        raw_pipeline=pipeline,
        feature_names=list(feature_cols),
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        cv_f1_mean=cv_f1_mean,
        cv_f1_std=cv_f1_std,
        trained_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _get_shap_values(bundle: ModelBundle, X: np.ndarray) -> np.ndarray:
    """Compute SHAP values for X using the raw (uncalibrated) pipeline."""
    import shap

    pipeline = bundle.raw_pipeline
    scaler = pipeline.named_steps["scaler"]
    clf = pipeline.named_steps["clf"]
    X_scaled = scaler.transform(X)

    if bundle.model_name in ("xgboost", "random_forest"):
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_scaled)
        # TreeExplainer for binary returns list[array] for RF, array for XGB
        if isinstance(sv, list):
            sv = sv[1]
        return sv
    else:
        explainer = shap.LinearExplainer(clf, X_scaled, feature_dependence="independent")
        return explainer.shap_values(X_scaled)


def predict_score(
    bundle: ModelBundle,
    X: pd.DataFrame | np.ndarray,
    compute_shap: bool = True,
) -> list[PredictionResult]:
    """
    Return a PredictionResult for each row in X.

    Each result contains:
    - score (0–100, calibrated probability × 100)
    - score_band
    - top_features (top-3 SHAP contributors with direction)
    - display_text (counselor-facing summary, no clinical labels)
    """
    if isinstance(X, pd.DataFrame):
        X_arr = X[bundle.feature_names].values
        feature_names = bundle.feature_names
    else:
        X_arr = X
        feature_names = bundle.feature_names

    probs = bundle.calibrated_pipeline.predict_proba(X_arr)[:, 1]
    scores = (probs * 100).clip(0, 100)

    shap_values = _get_shap_values(bundle, X_arr) if compute_shap else None

    results = []
    for i, (score, prob) in enumerate(zip(scores, probs)):
        band = score_to_band(score)

        top_features = []
        if shap_values is not None:
            sv_row = shap_values[i]
            abs_sv = np.abs(sv_row)
            top3_idx = np.argsort(abs_sv)[::-1][:3]
            for idx in top3_idx:
                top_features.append(
                    {
                        "feature": feature_names[idx],
                        "shap_value": float(sv_row[idx]),
                        "direction": "increases_score" if sv_row[idx] > 0 else "decreases_score",
                        "feature_value": float(X_arr[i, idx]),
                    }
                )

        display_text = _format_display_text(score, band, top_features)

        results.append(
            PredictionResult(
                score=float(score),
                score_band=band,
                calibrated_probability=float(prob),
                top_features=top_features,
                display_text=display_text,
            )
        )
    return results


def _format_display_text(score: float, band: str, top_features: list[dict]) -> str:
    """Counselor-facing summary — no clinical labels, no diagnosis language."""
    band_labels = {
        "baseline": "Baseline — no specific action indicated",
        "check_in_suggested": "Check-in suggested",
        "outreach_recommended": "Outreach recommended",
        "priority_follow_up": "Priority follow-up",
    }
    band_label = band_labels.get(band, band)
    parts = [f"Support recommendation score: {score:.0f}/100 - {band_label}"]

    if top_features:
        feature_labels = {
            "engagement_variance": "LMS engagement variability",
            "sleep_schedule_drift": "Sleep schedule consistency",
            "social_activity_decline": "Social activity trend",
            "academic_trend": "Academic performance trend",
            "missed_class_streak": "Consecutive absences",
            "financial_stress_flag": "Financial stress indicator",
            "help_seeking_flag": "Support services engagement",
            "self_report_score": "Wellbeing self-report",
        }
        signal_parts = []
        for f in top_features:
            label = feature_labels.get(f["feature"], f["feature"])
            direction = "(+)" if f["direction"] == "increases_score" else "(-)"
            signal_parts.append(f"{label} {direction}")
        parts.append("Top contributing signals: " + ", ".join(signal_parts))

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    bundle: ModelBundle,
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
) -> dict:
    """Return full evaluation metrics on a held-out set."""
    if isinstance(X, pd.DataFrame):
        X_arr = X[bundle.feature_names].values
    else:
        X_arr = X

    probs = bundle.calibrated_pipeline.predict_proba(X_arr)[:, 1]
    preds = (probs >= bundle.threshold).astype(int)
    return _compute_metrics(y, preds, probs)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = Path(__file__).parent.parent / "models"


def save_model(bundle: ModelBundle, path: str | Path | None = None) -> Path:
    if path is None:
        path = _DEFAULT_MODEL_DIR / "risk_classifier.pkl"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[risk_classifier] saved to {path}")
    return path


def load_model(path: str | Path | None = None) -> ModelBundle:
    if path is None:
        path = _DEFAULT_MODEL_DIR / "risk_classifier.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)
