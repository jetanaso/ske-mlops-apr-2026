"""
tests/test_serving.py
Unit tests for the FastAPI serving endpoint.
Used in GitHub Actions CI pipeline.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

mock_model = MagicMock()
mock_model.predict.return_value = np.array([3_500_000.0, 4_200_000.0])

with patch("mlflow.pyfunc.load_model", return_value=mock_model), \
     patch("mlflow.set_tracking_uri"):
    from serving.main import app, model_cache
    model_cache["model"] = mock_model

from fastapi.testclient import TestClient

client = TestClient(app)

SAMPLE = {
    "area": 120.5, "bedrooms": 3, "bathrooms": 2,
    "floor": 5, "age": 10, "distance_bts": 0.8,
    "distance_center": 5.2, "parking": 1,
    "quality": "good", "direction": "north",
}


def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"
    assert res.json()["model_loaded"] is True


def test_predict_single():
    res = client.post("/predict", json={"features": [SAMPLE]})
    assert res.status_code == 200
    body = res.json()
    assert len(body["predictions"]) == 1
    assert isinstance(body["predictions"][0], float)
    assert body["model_stage"] == "Production"


def test_predict_batch():
    res = client.post("/predict", json={"features": [SAMPLE, SAMPLE]})
    assert res.status_code == 200
    assert len(res.json()["predictions"]) == 2


def test_503_when_model_not_loaded():
    model_cache.clear()
    res = client.post("/predict", json={"features": [SAMPLE]})
    assert res.status_code == 503
    model_cache["model"] = mock_model


def test_root():
    res = client.get("/")
    assert res.status_code == 200
