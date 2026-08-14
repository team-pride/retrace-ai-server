from datetime import date

from app.services import curve_service
from app.services.curve_service import ChangePoint, CurvePoint


def _record(photo_key: str, captured_at: str, value: float) -> dict:
    return {
        "photo_key": photo_key,
        "captured_at": captured_at,
        "face_width_ratio": value,
        "jaw_angle_deg": value,
        "eyelid_height_ratio": value,
        "mouth_corner_angle_deg": value,
        "ipd_px": 60.0,
    }


def test_check_eligibility_fails_when_total_too_low():
    records = [_record(f"p{i}", "2024-01-01", 1.0) for i in range(5)]
    eligible, reasons = curve_service._check_eligibility(records)
    assert eligible is False
    assert any("20장" in r for r in reasons)


def test_check_eligibility_fails_when_a_year_is_sparse():
    records = [_record(f"p{i}", "2023-01-01", 1.0) for i in range(18)]
    records += [_record("p18", "2024-01-01", 1.0)]  # 2024년 1장뿐
    records += [_record("p19", "2024-01-02", 1.0)]  # 2024년 2장 (기준 3장 미달)
    eligible, reasons = curve_service._check_eligibility(records)
    assert eligible is False
    assert any("2024" in r and "3장" in r for r in reasons)


def test_check_eligibility_passes():
    records = []
    for year in (2022, 2023, 2024):
        for i in range(7):
            records.append(_record(f"{year}_{i}", f"{year}-0{(i % 9) + 1}-01", 1.0))
    eligible, reasons = curve_service._check_eligibility(records)
    assert eligible is True
    assert reasons == []


def test_group_by_date_uses_median_for_duplicate_dates():
    records = [
        _record("a", "2024-03-01", 10.0),
        _record("b", "2024-03-01", 20.0),
        _record("c", "2024-03-01", 30.0),
        _record("d", "2024-04-01", 5.0),
    ]
    grouped = curve_service._group_by_date(records, "face_width_ratio")
    assert grouped == [
        (date(2024, 3, 1), 20.0, 3),
        (date(2024, 4, 1), 5.0, 1),
    ]


def test_detect_change_points_finds_direction_reversal():
    points = [
        CurvePoint(date(2024, 1, 1), 10.0, 1, "high"),
        CurvePoint(date(2024, 2, 1), 12.0, 1, "high"),  # 상승 -> 여기가 정점(변화점)
        CurvePoint(date(2024, 3, 1), 8.0, 1, "high"),  # 하락 시작
        CurvePoint(date(2024, 4, 1), 4.0, 1, "high"),  # 계속 하락
    ]
    change_points = curve_service._detect_change_points(points)
    assert len(change_points) == 1
    cp = change_points[0]
    assert cp.captured_at == date(2024, 2, 1)
    assert cp.direction == "down"


def test_detect_change_points_no_reversal_when_monotonic():
    points = [
        CurvePoint(date(2024, 1, 1), 10.0, 1, "high"),
        CurvePoint(date(2024, 2, 1), 12.0, 1, "high"),
        CurvePoint(date(2024, 3, 1), 15.0, 1, "high"),
    ]
    assert curve_service._detect_change_points(points) == []


def test_detect_change_points_requires_at_least_three_points():
    points = [
        CurvePoint(date(2024, 1, 1), 10.0, 1, "high"),
        CurvePoint(date(2024, 2, 1), 5.0, 1, "high"),
    ]
    assert curve_service._detect_change_points(points) == []


def test_build_curve_unknown_indicator_raises():
    import pytest

    with pytest.raises(curve_service.UnknownIndicatorError):
        curve_service.build_curve("someone", "not_a_real_indicator")


def test_build_curve_not_eligible_when_no_records(tmp_path, monkeypatch):
    from pathlib import Path

    from app.services import indicator_store

    monkeypatch.setattr(indicator_store, "_store", indicator_store.IndicatorStore(Path(tmp_path) / "ind.json"))

    result = curve_service.build_curve("brand_new_user", "face_width_ratio")
    assert result.eligible is False
    assert result.points == []
    assert result.total_count == 0


def test_build_curve_eligible_with_enough_records(tmp_path, monkeypatch):
    from pathlib import Path

    from app.services import indicator_store

    store = indicator_store.IndicatorStore(Path(tmp_path) / "ind.json")
    monkeypatch.setattr(indicator_store, "_store", store)

    for year in (2022, 2023, 2024):
        for i in range(7):
            store.add_record(
                "user_x",
                _record(f"{year}_{i}", f"{year}-0{(i % 9) + 1}-0{(i % 9) + 1}", float(i)),
            )

    result = curve_service.build_curve("user_x", "face_width_ratio")
    assert result.eligible is True
    assert result.total_count == 21
    assert len(result.points) > 0
