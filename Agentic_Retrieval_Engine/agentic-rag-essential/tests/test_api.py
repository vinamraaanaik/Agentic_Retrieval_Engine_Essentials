from fastapi.testclient import TestClient
from agentic_rag.api import app

client = TestClient(app)


def test_health():
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_query():
    r = client.post("/api/v1/query", json={"query": "Which customers have high churn and unresolved tickets?"})
    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"]
    assert "answer" in data
