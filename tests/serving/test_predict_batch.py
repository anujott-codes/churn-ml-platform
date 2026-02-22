import io

def test_batch_success(client):
    csv_content = """age,gender,tenure,usage_frequency,support_calls,payment_delay,subscription_type,contract_length,total_spend,last_interaction
30,Male,12,5,2,0,Basic,Annual,500,10
"""

    file = {
        "file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    }

    response = client.post("/api/v1/predict-batch", files=file)

    assert response.status_code == 200
    assert "predictions" in response.json()

def test_batch_empty(client):
    csv_content = ""

    file = {
        "file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")
    }

    response = client.post("/api/v1/predict-batch", files=file)

    assert response.status_code == 400