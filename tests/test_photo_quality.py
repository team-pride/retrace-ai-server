from app.services.photo_quality_service import PhotoMetrics, _compute_roll_angle, grade_metrics


def test_roll_angle_upright_anatomical_order():
    # DeepFace 기준 left_eye(해부학적 왼쪽)는 이미지 픽셀상 오른쪽(더 큰 x)에 찍힌다.
    # 이 경우에도 눈높이가 같으면 각도는 0에 가까워야 한다 (180도로 튀면 버그).
    left_eye = (120, 50)  # 사람 기준 왼쪽 눈 -> 픽셀 x가 더 큼
    right_eye = (80, 50)  # 사람 기준 오른쪽 눈 -> 픽셀 x가 더 작음
    angle = _compute_roll_angle(left_eye, right_eye)
    assert angle < 1.0


def test_roll_angle_tilted_head():
    left_eye = (120, 60)
    right_eye = (80, 50)
    angle = _compute_roll_angle(left_eye, right_eye)
    assert 10 < angle < 20


def test_grade_pass():
    metrics = PhotoMetrics(angle_deg=2.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "pass"
    assert result.reasons == []


def test_grade_conditional_angle():
    metrics = PhotoMetrics(angle_deg=10.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"
    assert any("각도" in r for r in result.reasons)


def test_grade_exclude_angle():
    metrics = PhotoMetrics(angle_deg=20.0, blur_variance=200.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"


def test_grade_conditional_blur():
    metrics = PhotoMetrics(angle_deg=2.0, blur_variance=60.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"


def test_grade_exclude_blur():
    metrics = PhotoMetrics(angle_deg=2.0, blur_variance=10.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"


def test_grade_conditional_low_confidence():
    metrics = PhotoMetrics(angle_deg=2.0, blur_variance=200.0, detector_confidence=0.3)
    result = grade_metrics(metrics)
    assert result.grade == "conditional"


def test_exclude_wins_over_conditional():
    # 각도는 conditional 수준이지만 블러가 exclude 수준이면 최종은 exclude
    metrics = PhotoMetrics(angle_deg=10.0, blur_variance=10.0, detector_confidence=0.95)
    result = grade_metrics(metrics)
    assert result.grade == "exclude"
