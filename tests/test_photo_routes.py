from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import vector_store

client = TestClient(app)

GARBAGE_BYTES = b"this is not a valid image file at all"


def test_evaluate_batch_invalid_images_are_skipped():
    response = client.post(
        "/api/v1/photo/evaluate-batch",
        files=[
            ("files", ("a.jpg", GARBAGE_BYTES, "image/jpeg")),
            ("files", ("b.jpg", GARBAGE_BYTES, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 2
    assert body["skipped_count"] == 2
    assert body["succeeded_count"] == 0
    assert body["failed_count"] == 0
    assert {r["filename"] for r in body["results"]} == {"a.jpg", "b.jpg"}
    assert all(r["status"] == "skipped" for r in body["results"])


def test_evaluate_batch_photo_key_uses_filename():
    response = client.post(
        "/api/v1/photo/evaluate-batch",
        files=[("files", ("my_photo.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["photo_key"] == "my_photo.jpg"


def test_evaluate_batch_unregistered_user_returns_404(tmp_path, monkeypatch):
    # /evaluate와 동일하게, user_id를 넘겼는데 기준 벡터가 없으면 배치 전체가 404여야 한다.
    monkeypatch.setattr(vector_store, "_store", vector_store.JSONFileVectorStore(Path(tmp_path) / "vec.json"))
    response = client.post(
        "/api/v1/photo/evaluate-batch",
        params={"user_id": "never_registered"},
        files=[("files", ("a.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 404


def test_evaluate_batch_missing_files_field_returns_422():
    # files는 필수 필드라, 아예 안 보내면 422여야 한다.
    response = client.post("/api/v1/photo/evaluate-batch")
    assert response.status_code == 422
