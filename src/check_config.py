"""Shared check thresholds and algorithmic constants.

Centralizing thresholds here means changing a detection criterion requires
editing one file. The registry's ``severity_key`` field maps each check to
its config section (for example ``class_imbalance`` -> imbalance thresholds).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BiasConfig:
    """Fairness (bias) check thresholds."""

    min_group_size: int = 30
    p_value_limit: float = 0.01
    di_critical: float = 0.4
    di_high: float = 0.6
    di_medium: float = 0.8


@dataclass(frozen=True)
class LeakageConfig:
    """Target leakage check thresholds."""

    cv_folds: int = 5
    min_rows: int = 10
    high_cardinality_ratio: float = 0.9
    critical_auc: float = 0.98
    high_auc: float = 0.92
    medium_auc: float = 0.85
    random_auc: float = 0.5
    critical_r2: float = 0.98
    high_r2: float = 0.90
    medium_r2: float = 0.70
    max_rows_for_cv: int = 100000   # sample above this before CV
    cv_random_state: int = 42       # fixed seed so CV results are deterministic


@dataclass(frozen=True)
class ImbalanceConfig:
    """Class imbalance check thresholds."""

    high_threshold: float = 0.10    # minority share < 10% -> HIGH
    medium_threshold: float = 0.30  # minority share < 30% -> MEDIUM


@dataclass(frozen=True)
class SchemaConfig:
    """Schema check thresholds."""

    numeric_pct_threshold: float = 0.8  # > 80% numeric-like text -> MEDIUM


@dataclass(frozen=True)
class MissingConfig:
    """Missing value check thresholds."""

    critical_pct: float = 50.0  # > 50% missing -> CRITICAL
    high_pct: float = 20.0      # > 20% missing -> HIGH


@dataclass(frozen=True)
class DuplicatesConfig:
    """Duplicate row check thresholds."""

    high_pct: float = 10.0   # > 10% duplicate rows -> HIGH


@dataclass(frozen=True)
class TargetTypeConfig:
    """Target type detection thresholds."""

    continuous_unique_ratio: float = 0.1


@dataclass(frozen=True)
class SkewnessConfig:
    """Skewness check thresholds."""

    skew_threshold: float = 1.0        # |skew| > 1.0 -> MEDIUM
    high_skew_threshold: float = 2.0   # |skew| > 2.0 -> HIGH


@dataclass(frozen=True)
class OutlierConfig:
    """Outlier check thresholds (IQR method)."""

    outlier_pct_threshold: float = 5.0   # > 5% outliers -> MEDIUM
    high_outlier_pct_threshold: float = 10.0   # > 10% outliers -> HIGH


@dataclass(frozen=True)
class SemanticConfig:
    """Thresholds for column semantic classification.

    Every ratio uses n_eff (non-missing value count), never the raw row
    count, so a column with many missing values is not misclassified.
    Ratios are scale-free, so the same rule works on files of any size.
    """

    # Uniqueness rules need this many non-missing values; below it, the
    # column is classified only by dtype / exact-two-values and marked ambiguous.
    MIN_ROWS: int = 50

    # Deterministic sample for expensive parse/pairwise tests.
    SAMPLE_SIZE: int = 2000

    # Parse ratio to call a text column a date.
    DATE_PARSE_RATIO: float = 0.8

    # Parse ratio to take the numeric path.
    NUMERIC_PARSE_RATIO: float = 0.9

    # Uniqueness share (nunique / n_eff) above which a column is identifier-like.
    IDENTIFIER_UNIQUE_RATIO: float = 0.99

    # One-to-one mapping ratio for treating an int column as an encoded
    # duplicate of a text column.
    ENCODED_DUP_RATIO: float = 0.95

    # Max candidate columns per side in the encoded-duplicate pairwise scan.
    PAIRWISE_CAP: int = 30

    # Cardinality ratio at or below which a string column is categorical.
    CARDINALITY_RATIO: float = 0.05

    # Int columns with at most this many distinct values may be discrete.
    DISCRETE_MAX_UNIQUE: int = 20

    # Absolute distinct-value cap for treating low-cardinality text as categorical.
    DISCRETE_ABS_CAP: int = 20

    # Parse ratio below which confidence is reduced.
    PARSE_HIGH_CONF: float = 0.99


@dataclass(frozen=True)
class CheckConfig:
    """All check thresholds grouped by check name."""

    bias: BiasConfig = BiasConfig()
    leakage: LeakageConfig = LeakageConfig()
    imbalance: ImbalanceConfig = ImbalanceConfig()
    schema: SchemaConfig = SchemaConfig()
    missing: MissingConfig = MissingConfig()
    duplicates: DuplicatesConfig = DuplicatesConfig()
    target_type: TargetTypeConfig = TargetTypeConfig()
    skewness: SkewnessConfig = SkewnessConfig()
    outliers: OutlierConfig = OutlierConfig()
    semantic: SemanticConfig = SemanticConfig()


# Single shared instance used by all checks.
check_config = CheckConfig()


__all__ = [
    "CheckConfig",
    "BiasConfig",
    "LeakageConfig",
    "ImbalanceConfig",
    "SchemaConfig",
    "MissingConfig",
    "DuplicatesConfig",
    "TargetTypeConfig",
    "SkewnessConfig",
    "OutlierConfig",
    "SemanticConfig",
    "check_config",
]
