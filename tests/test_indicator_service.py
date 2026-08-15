import numpy as np
import pytest

from app.services import indicator_service


def _landmarks(overrides: dict[int, tuple[float, float]]) -> np.ndarray:
    arr = np.zeros((68, 2), dtype=np.float64)
    for idx, (x, y) in overrides.items():
        arr[idx] = (x, y)
    return arr


def _eye_pair_landmarks(right_center=(0.0, 0.0), left_center=(100.0, 0.0), gap=10.0) -> dict:
    rx, ry = right_center
    lx, ly = left_center
    return {
        36: (rx - 10, ry), 37: (rx - 5, ry - gap / 2), 38: (rx + 5, ry - gap / 2),
        39: (rx + 10, ry), 40: (rx + 5, ry + gap / 2), 41: (rx - 5, ry + gap / 2),
        42: (lx - 10, ly), 43: (lx - 5, ly - gap / 2), 44: (lx + 5, ly - gap / 2),
        45: (lx + 10, ly), 46: (lx + 5, ly + gap / 2), 47: (lx - 5, ly + gap / 2),
    }


def test_fold_angle_within_range_unchanged():
    assert indicator_service._fold_angle(45.0) == 45.0
    assert indicator_service._fold_angle(-45.0) == -45.0


def test_fold_angle_folds_beyond_90():
    assert indicator_service._fold_angle(100.0) == pytest.approx(-80.0)
    assert indicator_service._fold_angle(-100.0) == pytest.approx(80.0)


def test_interpupillary_distance():
    landmarks = _landmarks(_eye_pair_landmarks())
    ipd = indicator_service._interpupillary_distance(landmarks)
    assert ipd == pytest.approx(100.0)


def test_face_width_ratio():
    overrides = _eye_pair_landmarks()
    overrides[0] = (0.0, 100.0)
    overrides[16] = (200.0, 100.0)
    landmarks = _landmarks(overrides)
    ipd = indicator_service._interpupillary_distance(landmarks)
    ratio = indicator_service._face_width_ratio(landmarks, ipd)
    assert ratio == pytest.approx(2.0)  # 200px 폭 / IPD 100px


def test_jaw_angle_deg_vertical_line_is_zero():
    # 귀밑 기준점이 턱 끝 바로 위(수직선상)에 있으면 기울기 0도
    landmarks = _landmarks({8: (0.0, 0.0), 2: (0.0, -20.0), 14: (0.0, -20.0)})
    angle = indicator_service._jaw_angle_deg(landmarks)
    assert angle == pytest.approx(0.0)


def test_jaw_angle_deg_wider_offset_is_larger():
    # 귀밑 기준점이 턱 끝에서 옆으로 더 벌어져 있을수록(더 처진 형태) 기울기가 커진다
    steep = _landmarks({8: (0.0, 0.0), 2: (-5.0, -20.0), 14: (5.0, -20.0)})
    saggy = _landmarks({8: (0.0, 0.0), 2: (-15.0, -20.0), 14: (15.0, -20.0)})
    steep_angle = indicator_service._jaw_angle_deg(steep)
    saggy_angle = indicator_service._jaw_angle_deg(saggy)
    assert saggy_angle > steep_angle


def test_eyelid_height_ratio():
    landmarks = _landmarks(_eye_pair_landmarks(gap=10.0))
    ipd = indicator_service._interpupillary_distance(landmarks)
    ratio = indicator_service._eyelid_height_ratio(landmarks, ipd)
    assert ratio == pytest.approx(0.05)  # 위 눈꺼풀-눈 중심 수직 거리(5px) / IPD(100px)


def test_mouth_corner_angle_deg_level():
    landmarks = _landmarks({48: (0.0, 0.0), 54: (100.0, 0.0)})
    angle = indicator_service._mouth_corner_angle_deg(landmarks)
    assert angle == pytest.approx(0.0)


def test_mouth_corner_angle_deg_tilted():
    landmarks = _landmarks({48: (0.0, 0.0), 54: (100.0, 10.0)})
    angle = indicator_service._mouth_corner_angle_deg(landmarks)
    assert angle == pytest.approx(5.71, abs=0.01)


def test_extract_indicators_zero_ipd_raises():
    # 왼쪽/오른쪽 눈 중심이 같은 위치 -> IPD 0
    landmarks_overrides = _eye_pair_landmarks(right_center=(0.0, 0.0), left_center=(0.0, 0.0))
    landmarks = _landmarks(landmarks_overrides)
    ipd = indicator_service._interpupillary_distance(landmarks)
    assert ipd == 0.0
