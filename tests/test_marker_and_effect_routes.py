from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import indicator_store, marker_store

client = TestClient(app)


def test_register_marker_invalid_date_returns_422():
    response = client.post(
        "/api/v1/marker/register",
        params={"user_id": "u1", "marker_date": "2024/01/01", "note": "메모"},
    )
    assert response.status_code == 422


def test_register_and_list_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(marker_store, "_store", marker_store.MarkerStore(Path(tmp_path) / "markers.json"))

    register_response = client.post(
        "/api/v1/marker/register",
        params={"user_id": "u1", "marker_date": "2024-01-01", "note": "스킨케어 시작"},
    )
    assert register_response.status_code == 200
    body = register_response.json()
    assert body["marker_date"] == "2024-01-01"
    assert body["note"] == "스킨케어 시작"
    marker_id = body["marker_id"]

    list_response = client.get("/api/v1/marker/list", params={"user_id": "u1"})
    assert list_response.status_code == 200
    markers = list_response.json()["markers"]
    assert len(markers) == 1
    assert markers[0]["marker_id"] == marker_id


def test_effect_judge_unknown_indicator_returns_400(tmp_path, monkeypatch):
    monkeypatch.setattr(marker_store, "_store", marker_store.MarkerStore(Path(tmp_path) / "markers.json"))
    marker_store.get_marker_store().add_marker("u1", "2024-01-01", "메모")
    marker_id = marker_store.get_marker_store().get_markers("u1")[0]["marker_id"]

    response = client.get(
        "/api/v1/effect/judge",
        params={"user_id": "u1", "indicator": "not_real", "marker_id": marker_id},
    )
    assert response.status_code == 400


def test_effect_judge_unknown_marker_returns_404(tmp_path, monkeypatch):
    monkeypatch.setattr(marker_store, "_store", marker_store.MarkerStore(Path(tmp_path) / "markers.json"))

    response = client.get(
        "/api/v1/effect/judge",
        params={"user_id": "u1", "indicator": "face_width_ratio", "marker_id": "no_such_marker"},
    )
    assert response.status_code == 404


def test_effect_judge_pending_when_insufficient_data(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indicator_store, "_store", indicator_store.IndicatorStore(Path(tmp_path) / "ind.json")
    )
    monkeypatch.setattr(marker_store, "_store", marker_store.MarkerStore(Path(tmp_path) / "markers.json"))

    marker_store.get_marker_store().add_marker("u1", "2024-01-10", "메모")
    marker_id = marker_store.get_marker_store().get_markers("u1")[0]["marker_id"]

    response = client.get(
        "/api/v1/effect/judge",
        params={"user_id": "u1", "indicator": "face_width_ratio", "marker_id": marker_id},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["verdict"] == "pending"
    assert len(body["reasons"]) > 0
