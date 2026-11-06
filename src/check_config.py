"""Shared check thresholds and algorithmic constants.

Centralizing thresholds in one place means changing a detection criterion
requires editing one file, not every check module. Each check imports the
values it needs from here instead of defining its own module-level
constants.

The registry's ``severity_key`` field names the config section that belongs
to each check (for example ``class_imbalance`` maps to the imbalance
thresholds below).
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
    max_rows_for_cv: int = 100000   # sample above this before CV (1 lakh rows)
    cv_random_state: int = 42      # fixed seed for deterministic CV


@dataclass(frozen=True)
class ImbalanceConfig:
    """Class imbalance check thresholds."""

    high_threshold: float = 0.10   # minority share < 10%  -> HIGH
    medium_threshold: float = 0.30  # minority share < 30% -> MEDIUM


@dataclass(frozen=True)
class SchemaConfig:
    """Schema check thresholds."""

    numeric_pct_threshold: float = 0.8  # > 80% numeric-like text -> MEDIUM


@dataclass(frozen=True)
class MissingConfig:
    """Missing value check thresholds."""

    critical_pct: float = 50.0   # > 50% missing -> CRITICAL
    high_pct: float = 20.0       # > 20% missing -> HIGH
    # > 0% missing -> LOW (no constant needed)


@dataclass(frozen=True)
class DuplicatesConfig:
    """Duplicate row check thresholds."""

    high_pct: float = 10.0   # > 10% duplicate rows -> HIGH


@dataclass(frozen=True)
class TargetTypeConfig:
    """Target type detection thresholds."""

    continuous_unique_ratio: float = 0.1


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


# Module-level instance used by all checks.
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
    "check_config",
]
