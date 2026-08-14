from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import indicator_store

client = TestClient(app)

GARBAGE_BYTES = b"this is not a valid image file at all"


def test_curve_unknown_indicator_returns_400():
    response = client.get("/api/v1/indicator/curve", params={"user_id": "u1", "indicator": "not_real"})
    assert response.status_code == 400


def test_curve_no_records_is_not_eligible(tmp_path, monkeypatch):
    monkeypatch.setattr(
        indicator_store, "_store", indicator_store.IndicatorStore(Path(tmp_path) / "ind.json")
    )
    response = client.get(
        "/api/v1/indicator/curve", params={"user_id": "brand_new_user", "indicator": "face_width_ratio"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["eligible"] is False
    assert body["points"] == []
    assert len(body["reasons"]) > 0


def test_extract_invalid_date_format_returns_422():
    response = client.post(
        "/api/v1/indicator/extract",
        params={"user_id": "u1", "photo_key": "k1", "captured_at": "2024/01/01"},
        files={"file": ("bad.jpg", GARBAGE_BYTES, "image/jpeg")},
    )
    assert response.status_code == 422


def test_extract_invalid_image_returns_422():
    response = client.post(
        "/api/v1/indicator/extract",
        params={"user_id": "u1", "photo_key": "k1", "captured_at": "2024-01-01"},
        files={"file": ("bad.webp", GARBAGE_BYTES, "image/webp")},
    )
    assert response.status_code == 422


def test_extract_batch_invalid_fallback_date_returns_422():
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1", "fallback_captured_at": "2024/01/01"},
        files=[("files", ("a.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 422


def test_extract_batch_skips_files_without_exif_and_no_fallback():
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1"},
        files=[("files", ("no_exif.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert body["skipped_count"] == 1
    assert body["succeeded_count"] == 0
    assert body["results"][0]["status"] == "skipped"
    assert "EXIF" in body["results"][0]["reason"]


def test_extract_batch_uses_fallback_date_then_fails_on_invalid_image():
    # 촬영일은 fallback으로 확보되지만, 이미지 자체가 얼굴 인식이 불가능한
    # 쓰레기 바이트라서 결과는 "failed"가 되어야 한다 (날짜 처리와 이미지
    # 처리가 분리되어 있는지 확인).
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1", "fallback_captured_at": "2024-06-01"},
        files=[("files", ("no_exif.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["status"] == "failed"
    assert body["failed_count"] == 1


def test_extract_batch_multiple_files_are_counted_independently():
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1"},
        files=[
            ("files", ("a.jpg", GARBAGE_BYTES, "image/jpeg")),
            ("files", ("b.jpg", GARBAGE_BYTES, "image/jpeg")),
        ],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 2
    assert len(body["results"]) == 2
    assert {r["filename"] for r in body["results"]} == {"a.jpg", "b.jpg"}
