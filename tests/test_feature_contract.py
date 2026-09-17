import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from feature_contract import (
    assert_no_leakage, ALLOWED_FEATURES, FORBIDDEN_FEATURES, DEAD_FEATURES,
)


def test_allowed_features_pass():
    assert_no_leakage(ALLOWED_FEATURES)


def test_forbidden_feature_is_caught():
    columns = ALLOWED_FEATURES[:3] + ["second_pricer_buy_price"]
    with pytest.raises(ValueError, match="утечка"):
        assert_no_leakage(columns)


def test_unknown_column_is_caught_not_silently_allowed():
    columns = ALLOWED_FEATURES[:3] + ["some_new_column_from_next_export"]
    with pytest.raises(ValueError, match="не описаны в контракте"):
        assert_no_leakage(columns)


def test_dead_feature_is_caught():
    columns = ALLOWED_FEATURES[:3] + ["is_turbo"]
    with pytest.raises(ValueError, match="100% NaN"):
        assert_no_leakage(columns)


def test_allowed_forbidden_dead_lists_do_not_overlap():
    assert set(ALLOWED_FEATURES).isdisjoint(set(FORBIDDEN_FEATURES))
    assert set(ALLOWED_FEATURES).isdisjoint(set(DEAD_FEATURES))
    assert set(FORBIDDEN_FEATURES).isdisjoint(set(DEAD_FEATURES))
