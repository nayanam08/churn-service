from fastapi.testclient import TestClient

from app import app

GOOD = {
    "tenure": 12, "MonthlyCharges": 70.5, "TotalCharges": 846.0,
    "SeniorCitizen": 0, "Contract": "Month-to-month",
    "InternetService": "Fiber optic", "PaymentMethod": "Electronic check",
}


def test_health():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["model_loaded"] is True


def test_predict():
    with TestClient(app) as c:
        r = c.post("/predict", json=GOOD)
        assert r.status_code == 200
        assert 0 <= r.json()["churn_probability"] <= 1


def test_rejects_bad_input():
    with TestClient(app) as c:
        r = c.post("/predict", json={**GOOD, "Contract": "Forever"})
        assert r.status_code == 422
