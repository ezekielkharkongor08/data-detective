"""Bias / fairness detection check.

Uses cross-validated ROC AUC instead of raw accuracy because accuracy is
misleading on imbalanced targets: a majority-class-only predictor scores
~76% on Adult and ~73% on Telco with zero real signal. ROC AUC is invariant
to class balance.

A sensitive attribute (e.g. gender, race) that alone can predict the target
indicates potential bias: the protected column carries signal about the
outcome, which can lead to unfair model behavior.

Supports both classification (ROC AUC) and regression (R^2).

Returns a list of Finding objects (empty means no bias detected).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.schemas.finding import Finding, Severity


# Cross-validation folds
_CV_FOLDS = 5

# Minimum rows required to run the bias test.
_MIN_ROWS = 10

# Auto-detection: a column is a candidate sensitive column when it is
# string/category and its unique-value ratio is at or below this threshold.
_SENSITIVE_UNIQUE_RATIO = 0.1

# Auto-detection: maximum unique values for a candidate sensitive column.
_SENSITIVE_MAX_UNIQUE = 10

# ROC AUC thresholds for single-column bias detection (classification).
# Lower than leakage thresholds because any predictive power from a
# protected attribute is a fairness concern.
_CRITICAL_AUC = 0.90
_HIGH_AUC = 0.80
_MEDIUM_AUC = 0.70

# Random-guess AUC baseline.
_RANDOM_AUC = 0.5

# R^2 thresholds for single-column bias detection (regression).
_CRITICAL_R2 = 0.70
_HIGH_R2 = 0.50
_MEDIUM_R2 = 0.30


def _is_continuous_target(y: pd.Series) -> bool:
    """Decide whether the target column is continuous (regression) or
    discrete (classification).

    A target is continuous when it is numeric, has more than 2 unique
    values, AND many distinct values relative to the number of rows
    (e.g. salary, price, age). A binary target is always classification.
    """
    if not pd.api.types.is_numeric_dtype(y):
        return False
    n_rows = len(y)
    if n_rows == 0:
        return False
    nunique = y.nunique(dropna=True)
    # Binary targets are always classification.
    if nunique <= 2:
        return False
    unique_ratio = nunique / n_rows
    return unique_ratio > 0.1


def _build_column_pipeline(series: pd.Series, regression: bool = False) -> Pipeline:
    """Build a preprocessing + model pipeline for a single column.

    All preprocessing happens inside the pipeline and is re-fit on each
    CV training fold, so no information from a validation fold leaks into
    the training data.
    """
    model: object
    if regression:
        model = LinearRegression()
    else:
        model = LogisticRegression(max_iter=1000)

    if pd.api.types.is_numeric_dtype(series):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", model),
        ])

    # Categorical / string column: impute then one-hot encode.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ("clf", model),
    ])


def _majority_baseline_pct(y: pd.Series) -> float:
    """Return the majority-class share as a percentage (classification only)."""
    if len(y) == 0:
        return 0.0
    counts = y.value_counts(normalize=True)
    return float(counts.iloc[0] * 100)


def _auto_detect_sensitive_columns(
    df: pd.DataFrame, target_column: Optional[str]
) -> list[str]:
    """Auto-detect candidate sensitive columns.

    Returns string/category columns with low cardinality (few unique values
    relative to row count) that are not the target column. These are typical
    sensitive attributes such as gender, race, or marital status.
    """
    candidates: list[str] = []
    n_rows = len(df)
    if n_rows == 0:
        return candidates

    for col in df.columns:
        if col == target_column:
            continue
        series = df[col]
        # Only consider string/category columns as potential sensitive attrs.
        if not (
            pd.api.types.is_string_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
        ):
            continue
        nunique = series.nunique(dropna=True)
        if nunique < 2:
            continue
        unique_ratio = nunique / n_rows
        # A column is a candidate sensitive column if it has few unique
        # values relative to row count, or a small absolute number of
        # unique values (covers small datasets like unit tests).
        if (
            unique_ratio <= _SENSITIVE_UNIQUE_RATIO
            or nunique <= _SENSITIVE_MAX_UNIQUE
        ):
            candidates.append(col)

    return candidates


def bias_check(
    csv_path: str,
    target_column: Optional[str] = None,
    sensitive_columns: Optional[list[str]] = None,
) -> list[Finding]:
    """Detect potential bias: a sensitive attribute that alone predicts the target.

    Args:
        csv_path: Path to the CSV file.
        target_column: Name of the target column (required).
        sensitive_columns: Columns to test for bias. If None, candidate
            sensitive columns are auto-detected (low-cardinality string
            columns that are not the target).

    Returns:
        A list of Finding objects. Empty list means no bias detected.
        For classification, findings are based on cross-validated ROC AUC.
        For regression, findings are based on cross-validated R^2.
    """
    if not target_column:
        return [
            Finding(
                check_name="bias",
                severity=Severity.MEDIUM,
                evidence="No target_column configured; cannot run bias check.",
                affected_columns=[],
                suggested_fix="Provide the target column name.",
                confidence=1.0,
            )
        ]

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return [
            Finding(
                check_name="bias",
                severity=Severity.HIGH,
                evidence=f"Failed to read CSV: {exc}",
                affected_columns=[],
                suggested_fix="Verify the file path and format.",
                confidence=1.0,
            )
        ]

    if target_column not in df.columns:
        return [
            Finding(
                check_name="bias",
                severity=Severity.MEDIUM,
                evidence=f"Target column '{target_column}' not found in CSV.",
                affected_columns=[],
                suggested_fix="Provide the correct target column name.",
                confidence=1.0,
            )
        ]

    n_rows = len(df)
    if n_rows == 0:
        return []

    y = df[target_column]
    is_continuous = _is_continuous_target(y)

    # Determine which sensitive columns to test.
    if sensitive_columns is not None:
        candidate_cols = [c for c in sensitive_columns if c in df.columns]
    else:
        candidate_cols = _auto_detect_sensitive_columns(df, target_column)

    findings: list[Finding] = []

    if is_continuous:
        # --- Regression path: cross-validated R^2 ---
        if not pd.api.types.is_numeric_dtype(y):
            return []
        y_values = y.values.astype(float)
        cv = KFold(n_splits=_CV_FOLDS, shuffle=True, random_state=42)
        scorer_name = "r2"

        for col in candidate_cols:
            series = df[col]

            # Constant column: can't train.
            if series.nunique(dropna=True) < 2:
                continue

            try:
                pipeline = _build_column_pipeline(series, regression=True)
                X = np.asarray(series).reshape(-1, 1)
                scores = cross_val_score(
                    pipeline, X, y_values, cv=cv, scoring=scorer_name
                )
                # NaN-aware aggregation (undefined R^2 on tiny folds).
                scores = scores[~np.isnan(scores)]
                if len(scores) == 0:
                    continue
                r2_mean = float(np.mean(scores))
                r2_std = float(np.std(scores))
            except Exception:
                continue

            if r2_mean >= _CRITICAL_R2:
                severity = Severity.CRITICAL
            elif r2_mean >= _HIGH_R2:
                severity = Severity.HIGH
            elif r2_mean >= _MEDIUM_R2:
                severity = Severity.MEDIUM
            else:
                continue  # No bias flag

            findings.append(
                Finding(
                    check_name="bias",
                    severity=severity,
                    evidence=(
                        f"Sensitive column '{col}' alone predicts target "
                        f"'{target_column}' with R^2 {r2_mean:.4f} ± {r2_std:.4f} "
                        f"({_CV_FOLDS}-fold CV). A sensitive attribute should "
                        "not carry this much signal about the outcome."
                    ),
                    affected_columns=[col],
                    suggested_fix=(
                        f"Review '{col}' as a feature. Consider removing it, "
                        "applying fairness mitigation, or collecting more "
                        "balanced data."
                    ),
                    confidence=0.85,
                )
            )

        return findings

    # --- Classification path: cross-validated ROC AUC ---
    # Encode target once for the model.
    if not pd.api.types.is_numeric_dtype(y):
        y_values = LabelEncoder().fit_transform(y.astype(str))
    else:
        y_values = y.values

    n_classes = len(set(y_values.tolist()))
    if n_classes < 2:
        return []

    cv = StratifiedKFold(n_splits=_CV_FOLDS, shuffle=True, random_state=42)
    scorer_name = "roc_auc" if n_classes <= 2 else "roc_auc_ovr"

    # Stratified CV can't have more folds than the smallest class size.
    class_counts = pd.Series(y_values).value_counts()
    min_class_size = int(class_counts.min())
    actual_n_splits = min(_CV_FOLDS, min_class_size)
    if actual_n_splits < 2:
        return []
    cv = StratifiedKFold(n_splits=actual_n_splits, shuffle=True, random_state=42)

    for col in candidate_cols:
        series = df[col]

        # Constant column: can't train.
        if series.nunique(dropna=True) < 2:
            continue

        try:
            pipeline = _build_column_pipeline(series, regression=False)
            X = np.asarray(series).reshape(-1, 1)
            scores = cross_val_score(
                pipeline, X, y_values, cv=cv, scoring=scorer_name
            )
            # NaN-aware aggregation.
            scores = scores[~np.isnan(scores)]
            if len(scores) == 0:
                continue
            auc_mean = float(np.mean(scores))
            auc_std = float(np.std(scores))
        except Exception:
            continue

        baseline_pct = _majority_baseline_pct(df[target_column])
        lift = auc_mean - _RANDOM_AUC

        if auc_mean >= _CRITICAL_AUC:
            severity = Severity.CRITICAL
        elif auc_mean >= _HIGH_AUC:
            severity = Severity.HIGH
        elif auc_mean >= _MEDIUM_AUC:
            severity = Severity.MEDIUM
        else:
            continue  # No bias flag

        findings.append(
            Finding(
                check_name="bias",
                severity=severity,
                evidence=(
                    f"Sensitive column '{col}' alone predicts target "
                    f"'{target_column}' with ROC AUC {auc_mean:.4f} ± {auc_std:.4f} "
                    f"({actual_n_splits}-fold stratified CV). Majority-class baseline: "
                    f"{baseline_pct:.1f}%. Lift over random (0.5000): {lift:+.4f}. "
                    "A sensitive attribute should not carry this much signal "
                    "about the outcome — it may introduce unfair bias."
                ),
                affected_columns=[col],
                suggested_fix=(
                    f"Review '{col}' as a feature. Consider removing it, "
                    "applying fairness mitigation, or collecting more "
                    "balanced data."
                ),
                confidence=0.85,
            )
        )

    return findings


__all__ = ["bias_check"]
