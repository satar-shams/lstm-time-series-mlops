import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

@pytest.mark.integration
def test_health():
    response = client.get("/api/v1/health")

    assert response.status_code == 200

@pytest.mark.integration
def test_predict_success():
    payload = {
        "data": [1.0] * 30,
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert isinstance(data["prediction"], float)

@pytest.mark.integration
def test_predict_invalid_input_length():
    payload = {
        "data": [1.0] * 29,
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422

@pytest.mark.integration
def test_predict_invalid_input_length_too_long():
    payload = {
        "data": [1.0] * 31,
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
    )

    assert response.status_code == 422