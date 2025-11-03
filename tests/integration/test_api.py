from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200

def test_run_dry_ok():
    r = client.post("/run", json={"prompt": "schedule lunch tomorrow", "dry_run": True})
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "logs" in data
