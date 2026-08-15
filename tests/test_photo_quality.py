import numpy as np
import pytest

from app.services.photo_quality_service import (
    PhotoMetrics,
    _crop_with_margin,
    _estimate_pitch_deg,
    _estimate_yaw_deg,
    grade_metrics,
)


def _landmarks(overrides: dict[int, tuple[float, float]]) -> np.ndarray:
    arr = np.zeros((68, 2), dtype=np.float64)
    for idx, (x, y) in overrides.items():
        arr[idx] = (x, y)
    return arr


def test_estimate_yaw_deg_centered_nose_is_zero():
    landmarks = _landmarks({0: (-50.0, 0.0), 16: (50.0, 0.0), 30: (0.0, -20.0)})
    assert _estimate_yaw_deg(landmarks) == pytest.approx(0.0)


def test_estimate_yaw_deg_off_center_nose_is_larger():
    landmarks = _landmarks({0: (-50.0, 0.0), 16: (50.0, 0.0), 30: (25.0, -20.0)})
    assert _estimate_yaw_deg(landmarks) == pytest.approx(45.0)


def test_estimate_pitch_deg_expected_position_is_zero():
    eye_overrides = {i: (0.0, 0.0) for i in range(36, 48)}
    landmarks = _landmarks({**eye_overrides, 8: (0.0, 100.0), 30: (0.0, 45.0)})
    assert _estimate_pitch_deg(landmarks) == pytest.approx(0.0)


def test_estimate_pitch_deg_looking_down_is_larger():
    eye_overrides = {i: (0.0, 0.0) for i in range(36, 48)}
    landmarks = _landmarks({**eye_overrides, 8: (0.0, 100.0), 30: (0.0, 70.0)})
    assert _estimate_pitch_deg(landmarks) == pytest.approx(22.5)


def test_grade_pass():
    metrics = PhotoMetrics(yaw_deg=2.0, pitch_deg=2.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "pass"
    assert result.reasons == []


def test_grade_conditional_yaw():
    metrics = PhotoMetrics(yaw_deg=10.0, pitch_deg=2.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"
    assert any("좌우" in r for r in result.reasons)


def test_grade_exclude_yaw():
    metrics = PhotoMetrics(yaw_deg=20.0, pitch_deg=2.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"


def test_grade_conditional_pitch():
    metrics = PhotoMetrics(yaw_deg=2.0, pitch_deg=7.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"
    assert any("상하" in r for r in result.reasons)


def test_grade_exclude_pitch():
    metrics = PhotoMetrics(yaw_deg=2.0, pitch_deg=15.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"


def test_grade_conditional_blur():
    metrics = PhotoMetrics(yaw_deg=2.0, pitch_deg=2.0, blur_variance=60.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"


def test_grade_exclude_blur():
    metrics = PhotoMetrics(yaw_deg=2.0, pitch_deg=2.0, blur_variance=10.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"


def test_grade_conditional_low_confidence():
    metrics = PhotoMetrics(yaw_deg=2.0, pitch_deg=2.0, blur_variance=200.0, detector_confidence=0.3)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"


def test_exclude_wins_over_conditional():
    # yaw는 conditional 수준이지만 블러가 exclude 수준이면 최종은 exclude
    metrics = PhotoMetrics(yaw_deg=10.0, pitch_deg=2.0, blur_variance=10.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"


def test_crop_with_margin_extracts_region_around_face():
    # 얼굴이 여러 개 검출됐을 때 선택된 얼굴 영역만 잘라내는 데 쓰인다
    # (기능명세서 2.2 다중 얼굴 처리).
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[50:100, 50:100] = 255  # 얼굴 영역만 흰색으로 표시

    crop = _crop_with_margin(img, {"x": 50, "y": 50, "w": 50, "h": 50}, margin_ratio=0.0)

    assert crop.shape[:2] == (50, 50)
    assert (crop == 255).all()


def test_crop_with_margin_clamps_to_image_bounds():
    # 여유(margin)를 더해도 이미지 경계를 벗어나지 않아야 한다.
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = _crop_with_margin(img, {"x": 0, "y": 0, "w": 100, "h": 100}, margin_ratio=0.5)

    assert crop.shape[:2] == (100, 100)


def test_crop_with_margin_returns_contiguous_array():
    # numpy 슬라이싱은 원본의 뷰(비연속 메모리)를 반환하는데, dlib은 ctypes로
    # 버퍼를 직접 읽어서 비연속 배열을 넘기면 예외 없이 얼굴 검출에 조용히
    # 실패한다(실측 확인). 크롭 결과가 항상 연속 배열이어야 한다.
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    crop = _crop_with_margin(img, {"x": 50, "y": 50, "w": 50, "h": 50}, margin_ratio=0.3)

    assert crop.flags["C_CONTIGUOUS"] is True
