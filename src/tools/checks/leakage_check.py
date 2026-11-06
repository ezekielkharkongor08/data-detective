"""Target leakage detection check (M3).

Trains a quick baseline model using only one suspicious column to predict the
target. If a single column alone can predict the target too well, that column
likely leaks the answer.

Classification uses cross-validated ROC AUC instead of raw accuracy because
accuracy is misleading on imbalanced targets: a majority-class-only predictor
scores ~76% on Adult and ~73% on Telco with zero real signal. ROC AUC is
invariant to class balance -- a useless predictor scores 0.5 regardless.

Regression uses cross-validated R^2. A column that alone explains almost all
variance of the target (R^2 near 1.0) is suspicious.

Supports binary, multiclass (one-vs-rest macro AUC), and continuous targets.

Design:
- Cross-validation (k=5) preserves class balance in each fold (StratifiedKFold
  for classification; plain KFold for regression).
- Imputers and encoders live inside a scikit-learn Pipeline and are fitted
  per fold, so no information from a validation fold leaks into the training
  data.
- High-cardinality columns (>= 90% unique values) are skipped; the ID-like
  check covers them.

Returns a list of Finding objects (empty means no leakage detected).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.check_config import check_config
from src.schemas.finding import Finding, Severity
from src.tools.checks.registry import register_check


# Check thresholds live in the shared config (check_config.leakage).


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


def _compute_baseline_score(X: pd.Series, y: pd.Series, is_continuous: bool) -> float:
    """Compute baseline score using a scikit-learn Dummy model.

    Drops missing values from both X and y before fitting. The Dummy model
    ignores X, so any feature column can be passed; only y drives the score.
    """
    valid = X.notna() & y.notna()
    X_clean = X[valid]
    y_clean = y[valid]

    if len(y_clean) == 0:
        return 0.0

    # Convert to numpy to avoid ArrowStringArray reshape issues (pandas 3.x).
    X_np = X_clean.to_numpy().reshape(-1, 1)

    if is_continuous:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_np, y_clean.to_numpy().astype(float))
        return float(dummy.score(X_np, y_clean.to_numpy().astype(float)))
    else:
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(X_np, y_clean.to_numpy())
        return float(dummy.score(X_np, y_clean.to_numpy()))


@register_check(
    name="target_leakage",
    required=("csv_path", "target_column"),
    optional=("suspicious_columns",),
    severity_key="target_leakage",
    description="Deep leakage check: does any single column alone predict the target?",
    target_hint=True,
    # Runs in the router-triggered deep_checks step, not the parallel fan-out.
    parallel=False,
    tool_name="run_leakage_check",
    param_types=(("suspicious_columns", list[str]),),
)
def leakage_check(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    suspicious_columns: Optional[list[str]] = None,
) -> list[Finding]:
    """Detect target leakage: a feature that alone predicts the target too well.

    Args:
        df: The pre-loaded DataFrame.
        target_column: Name of the target column (required).
        suspicious_columns: Columns to test for leakage. If None, all
            non-target columns are considered. When provided (for example
            from the router's suspect columns), the CV loop runs only on
            these columns, avoiding an expensive scan of every column.

    Returns:
        A list of Finding objects. Empty list means no leakage detected.
        For classification, findings are based on cross-validated ROC AUC.
        For regression, findings are based on cross-validated R^2.
    """
    if not target_column:
        return [
            Finding(
                check_name="target_leakage",
                severity=Severity.MEDIUM,
                evidence="No target_column configured; cannot run leakage check.",
                affected_columns=[],
                confidence=1.0,
            )
        ]

    if target_column not in df.columns:
        return [
            Finding(
                check_name="target_leakage",
                severity=Severity.MEDIUM,
                evidence=f"Target column '{target_column}' not found in CSV.",
                affected_columns=[],
                confidence=1.0,
            )
        ]

    n_rows = len(df)
    if n_rows == 0:
        return []

    # Sample large datasets before CV so the check stays fast. The sample is
    # noted in the evidence so the user knows results are approximate.
    sampled = False
    if n_rows > check_config.leakage.max_rows_for_cv:
        df = df.sample(
            n=check_config.leakage.max_rows_for_cv,
            random_state=check_config.leakage.cv_random_state,
        )
        sampled = True
        n_rows = len(df)

    y = df[target_column]
    is_continuous = _is_continuous_target(y)

    # Determine which columns to test.
    if suspicious_columns is None:
        candidate_cols = [c for c in df.columns if c != target_column]
    else:
        candidate_cols = [c for c in suspicious_columns if c in df.columns]

    # Compute baseline score once using a Dummy model (same for all columns
    # since Dummy ignores X). Use first candidate column as X.
    baseline_score = 0.0
    if candidate_cols:
        baseline_score = _compute_baseline_score(
            df[candidate_cols[0]], y, is_continuous
        )

    findings: list[Finding] = []

    if is_continuous:
        # --- Regression path: cross-validated R^2 ---
        if not pd.api.types.is_numeric_dtype(y):
            # Should not happen (_is_continuous_target checks this), but guard.
            return []
        y_values = y.values.astype(float)
        cv = KFold(
            n_splits=check_config.leakage.cv_folds,
            shuffle=True,
            random_state=check_config.leakage.cv_random_state,
        )
        scorer_name = "r2"

        for col in candidate_cols:
            series = df[col]

            # Constant column: can't train.
            if series.nunique(dropna=True) < 2:
                continue

            # High-cardinality column: skip on larger datasets (ID-like check
            # covers them). On small datasets the ratio is not meaningful.
            unique_ratio = series.nunique(dropna=True) / n_rows
            if n_rows >= 50 and unique_ratio >= check_config.leakage.high_cardinality_ratio:
                continue

            try:
                pipeline = _build_column_pipeline(series, regression=True)
                X = np.asarray(series).reshape(-1, 1)
                scores = cross_val_score(
                    pipeline, X, y_values, cv=cv, scoring=scorer_name
                )
                # Some folds may be NaN on tiny datasets (undefined R^2 with
                # <2 test samples). Use nan-aware aggregation.
                scores = scores[~np.isnan(scores)]
                if len(scores) == 0:
                    continue
                r2_mean = float(np.mean(scores))
                r2_std = float(np.std(scores))
            except Exception:
                continue

            if r2_mean >= check_config.leakage.critical_r2:
                severity = Severity.CRITICAL
            elif r2_mean >= check_config.leakage.high_r2:
                severity = Severity.HIGH
            elif r2_mean >= check_config.leakage.medium_r2:
                severity = Severity.MEDIUM
            else:
                continue  # No leakage flag

            # Sampling note appended to evidence so users know results are
            # approximate when the dataset was downsampled for CV.
            sample_note = (
                f"Dataset was sampled to {check_config.leakage.max_rows_for_cv} "
                "rows before CV, so the score is approximate. "
                if sampled
                else ""
            )
            findings.append(
                Finding(
                    check_name="target_leakage",
                    severity=severity,
                    evidence=(
                        f"Column '{col}' alone predicts target '{target_column}' "
                        f"with R^2 {r2_mean:.4f} ± {r2_std:.4f} "
                        f"({check_config.leakage.cv_folds}-fold CV, random_state "
                        f"{check_config.leakage.cv_random_state}). "
                        f"{sample_note}"
                        f"Baseline R^2 (predict mean): "
                        f"{baseline_score:.4f}. A single feature should not explain this "
                        "much of the target -- it likely leaks the answer."
                    ),
                    affected_columns=[col],
                    confidence=0.9,
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

    cv = StratifiedKFold(
        n_splits=check_config.leakage.cv_folds,
        shuffle=True,
        random_state=check_config.leakage.cv_random_state,
    )
    scorer_name = "roc_auc" if n_classes <= 2 else "roc_auc_ovr"

    # Stratified CV can't have more folds than the smallest class size.
    class_counts = pd.Series(y_values).value_counts()
    min_class_size = int(class_counts.min())
    actual_n_splits = min(check_config.leakage.cv_folds, min_class_size)
    if actual_n_splits < 2:
        return []
    cv = StratifiedKFold(
        n_splits=actual_n_splits, shuffle=True, random_state=check_config.leakage.cv_random_state
    )

    for col in candidate_cols:
        series = df[col]

        # Constant column: can't train.
        if series.nunique(dropna=True) < 2:
            continue

        # High-cardinality column: skip on larger datasets (ID-like check
        # covers them). On small datasets the ratio is not meaningful.
        unique_ratio = series.nunique(dropna=True) / n_rows
        if n_rows >= 50 and unique_ratio >= check_config.leakage.high_cardinality_ratio:
            continue

        try:
            pipeline = _build_column_pipeline(series, regression=False)
            X = np.asarray(series).reshape(-1, 1)
            scores = cross_val_score(
                pipeline, X, y_values, cv=cv, scoring=scorer_name
            )
            # Some folds may be NaN on tiny datasets. Use nan-aware
            # aggregation.
            scores = scores[~np.isnan(scores)]
            if len(scores) == 0:
                continue
            auc_mean = float(np.mean(scores))
            auc_std = float(np.std(scores))
        except Exception:
            # If the model can't train or score on this column, skip it.
            continue

        baseline_score_cls = baseline_score
        lift = auc_mean - check_config.leakage.random_auc

        if auc_mean >= check_config.leakage.critical_auc:
            severity = Severity.CRITICAL
        elif auc_mean >= check_config.leakage.high_auc:
            severity = Severity.HIGH
        elif auc_mean >= check_config.leakage.medium_auc:
            severity = Severity.MEDIUM
        else:
            continue  # No leakage flag

        # Sampling note appended to evidence so users know results are
        # approximate when the dataset was downsampled for CV.
        sample_note = (
            f"Dataset was sampled to {check_config.leakage.max_rows_for_cv} "
            "rows before CV, so the score is approximate. "
            if sampled
            else ""
        )
        findings.append(
            Finding(
                check_name="target_leakage",
                severity=severity,
                evidence=(
                    f"Column '{col}' alone predicts target '{target_column}' "
                    f"with ROC AUC {auc_mean:.4f} ± {auc_std:.4f} "
                    f"({actual_n_splits}-fold stratified CV, random_state "
                    f"{check_config.leakage.cv_random_state}). "
                    f"{sample_note}"
                    f"Baseline accuracy (prior): {baseline_score_cls:.4f}. "
                    f"Lift over random (0.5000): {lift:+.4f}. A single feature should not "
                    "predict the target this well -- it likely leaks the "
                    "answer."
                ),
                affected_columns=[col],
                confidence=0.9,
            )
        )

    return findings


__all__ = ["leakage_check"]
