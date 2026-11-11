import numpy as np
import pandas as pd
import pytest

from house_prices.features import (
    FILL_WITH_MODE, FILL_WITH_NONE, FILL_WITH_ZERO, ID, ORDINAL_MAPS, QUALITY_COLUMNS, TARGET,
    InvalidDataError, build_features,
)


def make_frame(n, with_target, start_id=1):
    rng = np.random.default_rng(0)
    data = {ID: range(start_id, start_id + n)}
    for column in FILL_WITH_NONE:
        data[column] = [None] * n
    for column in FILL_WITH_ZERO:
        data[column] = [np.nan] + list(rng.integers(1, 100, n - 1))
    for column in FILL_WITH_MODE:
        data[column] = [None] + ["A"] * (n - 1)
    for column in QUALITY_COLUMNS:
        data[column] = [None] + ["Gd"] * (n - 1)
    data["LotFrontage"] = [np.nan] + list(rng.integers(40, 100, n - 1))
    data["Neighborhood"] = ["North", "South"] * (n // 2)
    data["MSSubClass"] = [20, 60] * (n // 2)
    data["LotArea"] = rng.integers(5000, 15000, n)
    if with_target:
        data[TARGET] = rng.integers(100000, 300000, n)
    return pd.DataFrame(data)


def test_no_missing_values_after_processing():
    X, X_test, _ = build_features(make_frame(6, True), make_frame(4, False, start_id=100))
    assert not X.isnull().any().any()
    assert not X_test.isnull().any().any()


def test_train_and_test_have_identical_columns():
    X, X_test, _ = build_features(make_frame(6, True), make_frame(4, False, start_id=100))
    assert list(X.columns) == list(X_test.columns)


def test_target_is_log_transformed():
    train = make_frame(6, True)
    _, _, y_log = build_features(train, make_frame(4, False, start_id=100))
    assert np.allclose(y_log, np.log1p(train[TARGET]))


def test_id_is_not_used_as_a_feature():
    X, _, _ = build_features(make_frame(6, True), make_frame(4, False, start_id=100))
    assert ID not in X.columns


def test_quality_ratings_become_numbers():
    X, _, _ = build_features(make_frame(6, True), make_frame(4, False, start_id=100))
    assert X["KitchenQual"].dtype.kind in "if"
    assert set(X["ExterQual"].unique()) <= {0, 4}
    assert set(ORDINAL_MAPS) <= set(X.columns)


def test_mssubclass_is_treated_as_category():
    X, _, _ = build_features(make_frame(6, True), make_frame(4, False, start_id=100))
    assert "MSSubClass" not in X.columns
    assert any(c.startswith("MSSubClass_") for c in X.columns)


def test_missing_required_column_is_rejected():
    train = make_frame(6, True).drop(columns=["Neighborhood"])
    with pytest.raises(InvalidDataError):
        build_features(train, make_frame(4, False, start_id=100))


def test_train_without_target_is_rejected():
    with pytest.raises(InvalidDataError):
        build_features(make_frame(6, False), make_frame(4, False, start_id=100))
