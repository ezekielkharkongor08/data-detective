"""Tests for the shared check config (src/check_config.py)."""

from src.check_config import (
    BiasConfig,
    CheckConfig,
    DuplicatesConfig,
    ImbalanceConfig,
    LeakageConfig,
    MissingConfig,
    SchemaConfig,
    TargetTypeConfig,
    check_config,
)


def test_check_config_has_all_sections():
    """Every registered check has a matching config section."""
    assert isinstance(check_config.bias, BiasConfig)
    assert isinstance(check_config.leakage, LeakageConfig)
    assert isinstance(check_config.imbalance, ImbalanceConfig)
    assert isinstance(check_config.schema, SchemaConfig)
    assert isinstance(check_config.missing, MissingConfig)
    assert isinstance(check_config.duplicates, DuplicatesConfig)
    assert isinstance(check_config.target_type, TargetTypeConfig)


def test_check_config_default_values():
    """Default threshold values match the check implementations."""
    # Bias / fairness
    assert check_config.bias.min_group_size == 30
    assert check_config.bias.p_value_limit == 0.01
    assert check_config.bias.di_critical == 0.4
    assert check_config.bias.di_high == 0.6
    assert check_config.bias.di_medium == 0.8

    # Leakage
    assert check_config.leakage.cv_folds == 5
    assert check_config.leakage.min_rows == 10
    assert check_config.leakage.high_cardinality_ratio == 0.9
    assert check_config.leakage.critical_auc == 0.98
    assert check_config.leakage.high_auc == 0.92
    assert check_config.leakage.medium_auc == 0.85
    assert check_config.leakage.random_auc == 0.5
    assert check_config.leakage.critical_r2 == 0.98
    assert check_config.leakage.high_r2 == 0.90
    assert check_config.leakage.medium_r2 == 0.70

    # Imbalance
    assert check_config.imbalance.high_threshold == 0.10
    assert check_config.imbalance.medium_threshold == 0.30

    # Schema
    assert check_config.schema.numeric_pct_threshold == 0.8

    # Missing
    assert check_config.missing.critical_pct == 50.0
    assert check_config.missing.high_pct == 20.0

    # Duplicates
    assert check_config.duplicates.high_pct == 10.0

    # Target type
    assert check_config.target_type.continuous_unique_ratio == 0.1


def test_check_config_is_frozen():
    """Config instances are immutable (dataclass frozen=True)."""
    import dataclasses

    # Mutating any field on a frozen config instance should raise.
    try:
        check_config.bias = check_config.bias  # type: ignore[misc]
        assert False, "frozen dataclass should not allow field assignment"
    except (dataclasses.FrozenInstanceError, AttributeError):
        pass
    try:
        check_config.bias.min_group_size = 999  # type: ignore[misc]
        assert False, "frozen dataclass should not allow attribute assignment"
    except (dataclasses.FrozenInstanceError, AttributeError):
        pass


def test_check_config_is_singleton():
    """The module-level instance is reused across imports."""
    from src.check_config import check_config as cc2

    assert check_config is cc2
