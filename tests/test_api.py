# tests/test_api.py
def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in ["healthy", "degraded"]

def test_ingest_document(client, mock_payload):
    response = client.post("/ingest", json=mock_payload)
    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert data["status"] == "success"

def test_query_endpoint(client):
    payload = {
        "query": "test document",
        "top_k": 1,
        "include_sources": True
    }
    response = client.post("/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data