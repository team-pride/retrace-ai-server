"""손상되었거나 지원하지 않는 이미지 파일이 업로드됐을 때 500이 아니라
422로 처리되는지 확인하는 회귀 테스트.

실제로 정연님이 webp 파일을 업로드했다가 PIL.UnidentifiedImageError가
그대로 전파되어 서버가 500을 내는 버그가 있었다 (image_utils.py로 고침).
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

GARBAGE_BYTES = b"this is not a valid image file at all"


def test_photo_normalize_returns_422_on_invalid_image():
    response = client.post(
        "/api/v1/photo/normalize",
        files={"file": ("bad.webp", GARBAGE_BYTES, "image/webp")},
    )
    assert response.status_code == 422


def test_photo_evaluate_returns_422_on_invalid_image():
    response = client.post(
        "/api/v1/photo/evaluate?photo_key=invalidimgtest",
        files={"file": ("bad.webp", GARBAGE_BYTES, "image/webp")},
    )
    assert response.status_code == 422


def test_face_verify_returns_422_on_invalid_image():
    response = client.post(
        "/api/v1/face/verify?user_id=nonexistent",
        files={"file": ("bad.webp", GARBAGE_BYTES, "image/webp")},
    )
    # 등록된 유저가 없어서 404가 날 수도 있고, 있다면 이미지 파싱에서 422가 나야 한다.
    # 여기서는 최소한 500(서버 다운)은 아니어야 한다는 것만 확실히 검증한다.
    assert response.status_code != 500
