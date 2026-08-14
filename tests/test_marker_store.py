from pathlib import Path

from app.services.marker_store import MarkerStore


def test_add_and_get_marker(tmp_path):
    store = MarkerStore(Path(tmp_path) / "markers.json")
    record = store.add_marker("u1", "2024-01-01", "레이저 시술 받음")

    assert record["marker_date"] == "2024-01-01"
    assert record["note"] == "레이저 시술 받음"
    assert record["marker_id"]
    assert record["created_at"]

    fetched = store.get_marker("u1", record["marker_id"])
    assert fetched == record


def test_get_marker_returns_none_when_not_found(tmp_path):
    store = MarkerStore(Path(tmp_path) / "markers.json")
    assert store.get_marker("u1", "no_such_id") is None


def test_get_markers_lists_all_for_user(tmp_path):
    store = MarkerStore(Path(tmp_path) / "markers.json")
    store.add_marker("u1", "2024-01-01", "첫번째")
    store.add_marker("u1", "2024-02-01", "두번째")
    store.add_marker("u2", "2024-01-01", "다른 사용자")

    markers = store.get_markers("u1")
    assert len(markers) == 2
    assert {m["note"] for m in markers} == {"첫번째", "두번째"}
    assert store.get_markers("u2")[0]["note"] == "다른 사용자"
    assert store.get_markers("brand_new_user") == []
