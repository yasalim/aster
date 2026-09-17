import os
import sys

import pytest
from fastapi.testclient import TestClient


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from api import app, get_model
from feature_contract import FORBIDDEN_FEATURES


client = TestClient(app)
VALID = {"mark": "Toyota", "model": "Camry", "car_year": 2020, "mileage": 85000}


def test_price_returns_ordered_positive_band():
    response = client.post("/price", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["price_from"] <= body["price_to"]
    assert body["currency"] == "KZT"


def test_empty_input_is_rejected():
    response = client.post("/price", json={})
    assert response.status_code == 422


@pytest.mark.parametrize("leaked", ["second_pricer_buy_price", "confirmed_rgv_buy_price"])
def test_leakage_field_is_rejected_at_api_boundary(leaked):
    response = client.post("/price", json={**VALID, leaked: 9_999_999})
    assert response.status_code == 422
    assert "extra_forbidden" in response.text


def test_persisted_model_contains_no_forbidden_features():
    bundle = get_model()
    assert set(bundle.raw_feature_cols).isdisjoint(FORBIDDEN_FEATURES)


def test_negative_mileage_is_rejected():
    response = client.post("/price", json={**VALID, "mileage": -1})
    assert response.status_code == 422
