from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import onboarding_store

client = TestClient(app)


def test_new_user_is_not_acknowledged(tmp_path, monkeypatch):
    monkeypatch.setattr(
        onboarding_store, "_store", onboarding_store.JSONFileOnboardingStore(Path(tmp_path) / "ob.json")
    )
    response = client.get("/api/v1/onboarding/status", params={"user_id": "brand_new_user"})
    assert response.status_code == 200
    body = response.json()
    assert body["acknowledged"] is False
    assert body["acknowledged_at"] is None


def test_acknowledge_then_status_reflects_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        onboarding_store, "_store", onboarding_store.JSONFileOnboardingStore(Path(tmp_path) / "ob.json")
    )
    ack_response = client.post("/api/v1/onboarding/acknowledge", params={"user_id": "u1"})
    assert ack_response.status_code == 200
    ack_body = ack_response.json()
    assert ack_body["acknowledged"] is True
    assert ack_body["acknowledged_at"] is not None

    status_response = client.get("/api/v1/onboarding/status", params={"user_id": "u1"})
    status_body = status_response.json()
    assert status_body["acknowledged"] is True
    assert status_body["acknowledged_at"] == ack_body["acknowledged_at"]


def test_acknowledgment_is_per_user(tmp_path, monkeypatch):
    monkeypatch.setattr(
        onboarding_store, "_store", onboarding_store.JSONFileOnboardingStore(Path(tmp_path) / "ob.json")
    )
    client.post("/api/v1/onboarding/acknowledge", params={"user_id": "acked_user"})

    other_response = client.get("/api/v1/onboarding/status", params={"user_id": "other_user"})
    assert other_response.json()["acknowledged"] is False
