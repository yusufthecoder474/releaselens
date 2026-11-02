"""API tests with FastAPI TestClient. Uses the temp DB/Chroma from conftest."""
from fastapi.testclient import TestClient

from app.config import samples_dir
from app.main import app

client = TestClient(app)


def _sample_bytes(name: str) -> bytes:
    return (samples_dir() / f"{name}.xml").read_bytes()


def test_index_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "ReleaseLens" in resp.text
    assert "Try with a sample" in resp.text


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_report_page_with_upload():
    resp = client.post("/report", files={"file": ("real_bug.xml", _sample_bytes("real_bug"), "application/xml")})
    assert resp.status_code == 200
    assert "RED" in resp.text
    assert "Tests to re-check first" in resp.text


def test_report_page_with_sample_button():
    resp = client.post("/report", data={"sample": "healthy"})
    assert resp.status_code == 200
    assert "GREEN" in resp.text


def test_api_analyze_json():
    resp = client.post("/api/analyze", files={"file": ("flaky_heavy.xml", _sample_bytes("flaky_heavy"), "application/xml")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "YELLOW"
    assert body["total"] == 30
    assert body["groups"]


def test_api_runs_and_history():
    resp = client.get("/api/runs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)
    assert len(resp.json()) >= 1
    hist = client.get("/api/tests/test_dashboard_load/history")
    assert hist.status_code == 200
    assert "history" in hist.json()
