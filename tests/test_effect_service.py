from pathlib import Path

import pytest

from app.services import curve_service, effect_service, indicator_store, marker_store


def _record(photo_key: str, captured_at: str, value: float) -> dict:
    return {
        "photo_key": photo_key,
        "captured_at": captured_at,
        "face_width_ratio": value,
        "jaw_angle_deg": value,
        "eyelid_height_ratio": value,
        "mouth_corner_angle_deg": value,
        "ipd_px": 100.0,
    }


@pytest.fixture()
def isolated_stores(tmp_path, monkeypatch):
    ind_store = indicator_store.IndicatorStore(Path(tmp_path) / "ind.json")
    mk_store = marker_store.MarkerStore(Path(tmp_path) / "markers.json")
    monkeypatch.setattr(indicator_store, "_store", ind_store)
    monkeypatch.setattr(marker_store, "_store", mk_store)
    return ind_store, mk_store


def test_judge_effect_unknown_indicator_raises(isolated_stores):
    ind_store, mk_store = isolated_stores
    marker = mk_store.add_marker("u1", "2024-01-05", "메모")

    with pytest.raises(curve_service.UnknownIndicatorError):
        effect_service.judge_effect("u1", "not_a_real_indicator", marker["marker_id"])


def test_judge_effect_unknown_marker_raises(isolated_stores):
    with pytest.raises(effect_service.MarkerNotFoundError):
        effect_service.judge_effect("u1", "face_width_ratio", "no_such_marker")


def test_judge_effect_pending_when_not_enough_before_points(isolated_stores):
    ind_store, mk_store = isolated_stores
    marker = mk_store.add_marker("u1", "2024-01-10", "메모")
    # 마커 이전 기록이 1개뿐 (기준 3개 미달)
    ind_store.add_record("u1", _record("p1", "2024-01-01", 10.0))
    ind_store.add_record("u1", _record("p2", "2024-01-11", 20.0))
    ind_store.add_record("u1", _record("p3", "2024-01-12", 21.0))

    result = effect_service.judge_effect("u1", "face_width_ratio", marker["marker_id"])

    assert result.verdict == effect_service.VERDICT_PENDING
    assert any("이전" in r for r in result.reasons)


def test_judge_effect_pending_when_not_enough_after_points(isolated_stores):
    ind_store, mk_store = isolated_stores
    marker = mk_store.add_marker("u1", "2024-01-10", "메모")
    ind_store.add_record("u1", _record("p1", "2024-01-01", 10.0))
    ind_store.add_record("u1", _record("p2", "2024-01-02", 11.0))
    ind_store.add_record("u1", _record("p3", "2024-01-03", 12.0))
    # 마커 이후 기록이 1개뿐 (기준 2개 미달)
    ind_store.add_record("u1", _record("p4", "2024-01-11", 20.0))

    result = effect_service.judge_effect("u1", "face_width_ratio", marker["marker_id"])

    assert result.verdict == effect_service.VERDICT_PENDING
    assert any("이후" in r for r in result.reasons)


def test_judge_effect_not_observed_when_trend_continues_unchanged(isolated_stores):
    ind_store, mk_store = isolated_stores
    marker = mk_store.add_marker("u1", "2024-01-10", "메모")
    # 마커 이전: 하루 +1씩 증가하는 완벽한 직선 (2024-01-01 -> 10, 02 -> 11, 03 -> 12)
    ind_store.add_record("u1", _record("p1", "2024-01-01", 10.0))
    ind_store.add_record("u1", _record("p2", "2024-01-02", 11.0))
    ind_store.add_record("u1", _record("p3", "2024-01-03", 12.0))
    # 마커 이후: 같은 추세를 그대로 연장한 값 (2024-01-10은 epoch+9일 -> 19, 01-11 -> 20)
    ind_store.add_record("u1", _record("p4", "2024-01-10", 19.0))
    ind_store.add_record("u1", _record("p5", "2024-01-11", 20.0))

    result = effect_service.judge_effect("u1", "face_width_ratio", marker["marker_id"])

    assert result.verdict == effect_service.VERDICT_NOT_OBSERVED
    assert result.before_count == 3
    assert result.after_count == 2
    assert result.mean_deviation == pytest.approx(0.0, abs=1e-9)


def test_judge_effect_observed_when_actual_diverges_from_prediction(isolated_stores):
    ind_store, mk_store = isolated_stores
    marker = mk_store.add_marker("u1", "2024-01-10", "메모")
    ind_store.add_record("u1", _record("p1", "2024-01-01", 10.0))
    ind_store.add_record("u1", _record("p2", "2024-01-02", 11.0))
    ind_store.add_record("u1", _record("p3", "2024-01-03", 12.0))
    # 예측선(추세 연장)은 19, 20이어야 하는데 실제로는 훨씬 큰 값 -> 예측과 크게 어긋남
    ind_store.add_record("u1", _record("p4", "2024-01-10", 40.0))
    ind_store.add_record("u1", _record("p5", "2024-01-11", 42.0))

    result = effect_service.judge_effect("u1", "face_width_ratio", marker["marker_id"])

    assert result.verdict == effect_service.VERDICT_OBSERVED
    assert result.mean_deviation is not None
    assert result.mean_deviation > 0
    assert len(result.prediction_line) == 2
    assert len(result.actual_after) == 2


def test_judge_effect_ignores_other_indicators_and_users(isolated_stores):
    ind_store, mk_store = isolated_stores
    marker = mk_store.add_marker("u1", "2024-01-10", "메모")
    ind_store.add_record("u1", _record("p1", "2024-01-01", 10.0))
    ind_store.add_record("u1", _record("p2", "2024-01-02", 11.0))
    ind_store.add_record("u1", _record("p3", "2024-01-03", 12.0))
    ind_store.add_record("u1", _record("p4", "2024-01-10", 19.0))
    ind_store.add_record("u1", _record("p5", "2024-01-11", 20.0))

    result = effect_service.judge_effect("u1", "jaw_angle_deg", marker["marker_id"])
    assert result.indicator == "jaw_angle_deg"
