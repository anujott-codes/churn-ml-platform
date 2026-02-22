def test_predict_success(client):
    payload = {
        "age": 30,
        "gender": "Male",
        "tenure": 12,
        "usage_frequency": 5,
        "support_calls": 2,
        "payment_delay": 0,
        "subscription_type": "Basic",
        "contract_length": "Annual",
        "total_spend": 500,
        "last_interaction": 10
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1
    assert data["probability"] == 0.8
    assert data["churn"] is True

def test_predict_missing_field(client):
    payload = {
        "age": 30  
    }

    response = client.post("/api/v1/predict", json=payload)

    assert response.status_code == 422  