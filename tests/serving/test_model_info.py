def test_model_info(client):
    response = client.get("/api/v1/model-info")

    assert response.status_code == 200
    data = response.json()

    assert data["model_name"] == "CustomerChurnModel"
    assert data["alias"] == "champion"