from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.services import indicator_store, vector_store

client = TestClient(app)

GARBAGE_BYTES = b"this is not a valid image file at all"


def _register_reference_vector(tmp_path, monkeypatch, user_id: str = "u1") -> None:
    """지표 추출 엔드포인트는 이제 본인 기준 벡터 등록을 선행조건으로 요구하므로,
    등록 이후 흐름만 확인하려는 테스트는 먼저 더미 벡터를 등록해둔다."""
    monkeypatch.setattr(vector_store, "_store", vector_store.JSONFileVectorStore(Path(tmp_path) / "vec.json"))
    vector_store.get_vector_store().save(user_id, [0.1, 0.2, 0.3])


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


def test_extract_invalid_image_returns_422(tmp_path, monkeypatch):
    _register_reference_vector(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/indicator/extract",
        params={"user_id": "u1", "photo_key": "k1", "captured_at": "2024-01-01"},
        files={"file": ("bad.webp", GARBAGE_BYTES, "image/webp")},
    )
    assert response.status_code == 422


def test_extract_unregistered_user_returns_404(tmp_path, monkeypatch):
    # 기준 벡터를 등록하지 않은 사용자는 지표 추출 자체가 선행조건 미충족으로 막혀야 한다
    # (기능명세서 2.2 선행조건: "본인 기준 벡터가 등록되어 있어야 한다").
    monkeypatch.setattr(vector_store, "_store", vector_store.JSONFileVectorStore(Path(tmp_path) / "vec.json"))
    response = client.post(
        "/api/v1/indicator/extract",
        params={"user_id": "never_registered", "photo_key": "k1", "captured_at": "2024-01-01"},
        files={"file": ("bad.webp", GARBAGE_BYTES, "image/webp")},
    )
    assert response.status_code == 404


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


def test_extract_batch_uses_fallback_date_then_skips_invalid_image(tmp_path, monkeypatch):
    # 촬영일은 fallback으로 확보되지만, 이미지 자체가 손상된/지원하지 않는
    # 바이트라서 "skipped"로 기록되어야 한다 (기능명세서 2.1: "지원하지 않는
    # 형식이거나 손상된 파일은 건너뛰고 사유를 기록한다"). 날짜 처리와 이미지
    # 처리가 분리되어 있는지도 함께 확인한다.
    _register_reference_vector(tmp_path, monkeypatch)
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1", "fallback_captured_at": "2024-06-01"},
        files=[("files", ("no_exif.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["status"] == "skipped"
    assert body["skipped_count"] == 1
    assert body["failed_count"] == 0


def test_extract_batch_unregistered_user_marks_dated_files_failed():
    # 기준 벡터가 등록되지 않은 사용자는, 촬영일이 확보된 파일에 한해
    # "failed"로 기록된다 (날짜조차 없는 파일은 애초에 판정 단계에 도달하지
    # 않으므로 등록 여부와 무관하게 "skipped"로 남는다 — 아래에서 함께 확인).
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "never_registered_batch_user", "fallback_captured_at": "2024-06-01"},
        files=[("files", ("no_exif.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["status"] == "failed"
    assert "기준 얼굴 벡터" in body["results"][0]["reason"]


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


def test_extract_batch_year_below_minimum_reported_as_notice_not_blocked():
    # 연도별 최소 권장 장수(BATCH_MIN_PHOTOS_PER_YEAR, 기본 5장) 미만이어도
    # 업로드 자체를 막지는 않고 year_notices로만 안내한다 (기능명세서 2.1
    # 비즈니스 규칙 — 상한과 달리 하한은 차단이 아니라 안내).
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1", "fallback_captured_at": "2024-06-01"},
        files=[("files", ("only_one.jpg", GARBAGE_BYTES, "image/jpeg"))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 1
    assert any("2024" in notice for notice in body["year_notices"])


def test_extract_batch_year_over_maximum_is_skipped():
    # 같은 연도에 BATCH_MAX_PHOTOS_PER_YEAR(기본 30장)를 넘는 파일을 올리면
    # 초과분은 처리하지 않고 "skipped"로 기록해야 한다.
    files = [
        ("files", (f"photo_{i}.jpg", GARBAGE_BYTES, "image/jpeg")) for i in range(32)
    ]
    response = client.post(
        "/api/v1/indicator/extract-batch",
        params={"user_id": "u1", "fallback_captured_at": "2024-06-01"},
        files=files,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["total_count"] == 32
    over_limit_results = [r for r in body["results"] if r["status"] == "skipped" and "최대" in (r["reason"] or "")]
    assert len(over_limit_results) == 2  # 32장 중 30장 초과분 2장
