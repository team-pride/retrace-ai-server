import numpy as np

from app.services.normalization_service import (
    CropOutOfBoundsError,
    _assert_crop_within_bounds,
    _eye_distance,
    _signed_tilt_angle,
    _white_balance_via_eye_patches,
    build_alignment_matrix,
)


def test_signed_tilt_angle_zero_when_level():
    angle = _signed_tilt_angle((80, 100), (120, 100))
    assert abs(angle) < 0.01


def test_signed_tilt_angle_matches_known_tilt():
    # dx=100, dy=30 -> atan2(30,100) ~= 16.7도
    angle = _signed_tilt_angle((80, 100), (180, 130))
    assert 15 < angle < 18


def test_eye_distance():
    assert _eye_distance((0, 0), (3, 4)) == 5.0


def test_alignment_matrix_levels_eyes():
    left_eye = (80, 100)
    right_eye = (180, 130)  # 기울어진 눈
    M, angle, scale, dist = build_alignment_matrix(left_eye, right_eye)

    pts = np.array(
        [[left_eye[0], left_eye[1], 1], [right_eye[0], right_eye[1], 1]], dtype=np.float64
    ).T
    transformed = M @ pts

    # 변환 후 두 눈의 y좌표가 거의 같아야 함 (수평 정렬 확인)
    assert abs(transformed[1, 0] - transformed[1, 1]) < 0.5
    # x좌표 간격은 스케일이 적용된 거리와 비슷해야 함
    assert abs(transformed[0, 1] - transformed[0, 0]) > 0


def test_crop_out_of_bounds_raises_for_tiny_source():
    # 아주 작은 원본 + 눈이 가장자리 -> 정렬 후 크롭이 원본 밖으로 나가야 함
    left_eye = (10, 10)
    right_eye = (15, 10)
    M, *_ = build_alignment_matrix(left_eye, right_eye)
    try:
        _assert_crop_within_bounds(M, orig_width=20, orig_height=20)
        assert False, "CropOutOfBoundsError가 발생해야 합니다"
    except CropOutOfBoundsError:
        pass


def test_crop_within_bounds_for_large_source():
    left_eye = (700, 500)
    right_eye = (900, 500)
    M, *_ = build_alignment_matrix(left_eye, right_eye)
    # 예외 없이 통과해야 함
    _assert_crop_within_bounds(M, orig_width=1600, orig_height=1200)


def test_white_balance_reduces_color_cast():
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    img[:, :, 0] = 200  # R 채널만 높여서 붉은 색조를 만듦
    corrected, skipped = _white_balance_via_eye_patches(img, (50, 50), (50, 50), patch_radius=20)

    assert skipped is False
    r, g = int(corrected[:, :, 0].mean()), int(corrected[:, :, 1].mean())
    assert abs(r - g) < abs(200 - 150)


def test_white_balance_skipped_when_sample_too_small():
    # 흰자 근사 패치가 이미지 경계(모서리)에 걸려 아주 작게 잘리면 표본이
    # NORM_MIN_WHITE_SAMPLE_PIXELS(기본 30)에 못 미쳐 보정을 건너뛰어야 한다.
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    img[:, :, 0] = 200
    corrected, skipped = _white_balance_via_eye_patches(img, (0, 0), (0, 0), patch_radius=1)

    assert skipped is True
    assert np.array_equal(corrected, img)  # 보정 없이 원본 그대로


def test_white_balance_skipped_when_no_samples_at_all():
    # 눈 위치가 이미지 밖이라 패치 자체가 비면(size=0) 보정을 건너뛰어야 한다.
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    corrected, skipped = _white_balance_via_eye_patches(img, (-500, -500), (-500, -500), patch_radius=5)

    assert skipped is True
    assert np.array_equal(corrected, img)
