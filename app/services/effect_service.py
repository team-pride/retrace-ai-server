"""관리 효과 판정 (우선순위 5).

기능명세서 5.1 "예측선 기반 변화 관찰 판정"을 그대로 구현한다.
- 마커 이전 구간의 추세(기울기)를 그대로 이어갔을 경우의 예측선을 만든다
  (마커 이전 구간에 대한 단순 선형회귀를 마커 이후 날짜로 연장).
- 실제 값과 예측선의 차이(평균 절대편차)를 계산한다.
- "데이터 흔들림 범위"는 마커 이전 구간에서 실측치가 그 구간 자체의
  추세선을 벗어나는 정도(잔차의 표준편차)로 정의한다. 마커 이후 편차가
  이 흔들림 범위의 EFFECT_NOISE_THRESHOLD_FACTOR배를 넘으면 "관찰됨",
  아니면 "관찰되지 않음"으로 판정한다.
- 마커 이전/이후 표본이 부족하면(EFFECT_MIN_POINTS_* 미달) "판단 보류".
- 의료적 효과(좋아짐/나빠짐) 단정은 하지 않는다 — "그대로 갔을 경우"와
  달라졌는지 여부만 본다 (기능명세서 5 비즈니스 규칙).
- 기능명세서 5.1 데이터 규칙("판정 결과는 마커 식별자, 지표 종류, 결과 상태,
  차이 값, 신뢰도로 저장한다")에 맞춰 판정에 사용된 표본 수(마커 전+후) 기준
  구간 신뢰도(low/medium/high)도 함께 계산한다.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date

from app.core.config import settings
from app.services.curve_service import INDICATOR_FIELDS, UnknownIndicatorError, _group_by_date
from app.services.indicator_store import get_indicator_store
from app.services.marker_store import get_marker_store

VERDICT_OBSERVED = "observed"
VERDICT_NOT_OBSERVED = "not_observed"
VERDICT_PENDING = "pending"


class MarkerNotFoundError(Exception):
    """등록되지 않은 마커 id를 요청한 경우."""


@dataclass
class PredictionPoint:
    captured_at: date
    predicted_value: float


@dataclass
class ObservedPoint:
    captured_at: date
    value: float


@dataclass
class EffectResult:
    indicator: str
    marker_id: str
    marker_date: date
    verdict: str  # "observed" | "not_observed" | "pending"
    reasons: list[str] = field(default_factory=list)
    before_count: int = 0
    after_count: int = 0
    noise_baseline: float | None = None
    mean_deviation: float | None = None
    confidence: str | None = None  # "high" | "medium" | "low"
    prediction_line: list[PredictionPoint] = field(default_factory=list)
    actual_after: list[ObservedPoint] = field(default_factory=list)


def _confidence_level(total_count: int) -> str:
    if total_count >= settings.EFFECT_CONFIDENCE_HIGH_MIN_TOTAL:
        return "high"
    if total_count >= settings.EFFECT_CONFIDENCE_MEDIUM_MIN_TOTAL:
        return "medium"
    return "low"


def _linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """최소제곱 1차 회귀. (기울기, 절편)을 반환한다. 점이 모두 같은 x면 기울기 0."""
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return 0.0, mean_y
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return slope, intercept


def judge_effect(user_id: str, indicator: str, marker_id: str) -> EffectResult:
    if indicator not in INDICATOR_FIELDS:
        raise UnknownIndicatorError(
            f"알 수 없는 지표입니다: {indicator}. 사용 가능한 값: {', '.join(INDICATOR_FIELDS)}"
        )

    marker = get_marker_store().get_marker(user_id, marker_id)
    if marker is None:
        raise MarkerNotFoundError(f"마커를 찾을 수 없습니다: {marker_id}")

    marker_date = date.fromisoformat(marker["marker_date"])

    records = get_indicator_store().get_records(user_id)
    grouped = _group_by_date(records, indicator)  # [(날짜, 중앙값, 표본수), ...] 날짜순

    before = [(d, v) for d, v, _ in grouped if d < marker_date]
    after = [(d, v) for d, v, _ in grouped if d >= marker_date]

    reasons: list[str] = []
    if len(before) < settings.EFFECT_MIN_POINTS_BEFORE_MARKER:
        reasons.append(
            f"마커 이전 지표 기록이 {settings.EFFECT_MIN_POINTS_BEFORE_MARKER}개 이상 필요해요 "
            f"(현재 {len(before)}개)."
        )
    if len(after) < settings.EFFECT_MIN_POINTS_AFTER_MARKER:
        reasons.append(
            f"마커 이후 지표 기록이 {settings.EFFECT_MIN_POINTS_AFTER_MARKER}개 이상 필요해요 "
            f"(현재 {len(after)}개)."
        )

    if reasons:
        return EffectResult(
            indicator=indicator,
            marker_id=marker_id,
            marker_date=marker_date,
            verdict=VERDICT_PENDING,
            reasons=reasons,
            before_count=len(before),
            after_count=len(after),
        )

    epoch = before[0][0]  # 날짜를 정수(경과일)로 바꾸기 위한 기준점
    before_xs = [float((d - epoch).days) for d, _ in before]
    before_ys = [v for _, v in before]
    slope, intercept = _linear_fit(before_xs, before_ys)

    # "데이터 흔들림 범위" = 마커 이전 구간에서 실측치가 그 구간 자체 추세선을 벗어나는 정도
    residuals_before = [y - (slope * x + intercept) for x, y in zip(before_xs, before_ys)]
    noise_baseline = statistics.pstdev(residuals_before) if len(residuals_before) > 1 else 0.0
    if noise_baseline == 0.0:
        # 마커 이전 표본이 완벽한 직선을 이루면(표본이 적을 때 흔함) 흔들림 범위가
        # 0이 되어 사소한 편차도 전부 "관찰됨"으로 잡히므로, 최소한의 여유를 둔다.
        noise_baseline = abs(statistics.mean(before_ys)) * 0.01

    prediction_line = [
        PredictionPoint(captured_at=d, predicted_value=slope * (d - epoch).days + intercept) for d, _ in after
    ]
    deviations = [actual - pred.predicted_value for (_, actual), pred in zip(after, prediction_line)]
    mean_deviation = statistics.mean(abs(dv) for dv in deviations)

    threshold = noise_baseline * settings.EFFECT_NOISE_THRESHOLD_FACTOR
    verdict = VERDICT_OBSERVED if mean_deviation > threshold else VERDICT_NOT_OBSERVED

    return EffectResult(
        indicator=indicator,
        marker_id=marker_id,
        marker_date=marker_date,
        verdict=verdict,
        reasons=[],
        before_count=len(before),
        after_count=len(after),
        noise_baseline=noise_baseline,
        mean_deviation=mean_deviation,
        confidence=_confidence_level(len(before) + len(after)),
        prediction_line=prediction_line,
        actual_after=[ObservedPoint(captured_at=d, value=v) for d, v in after],
    )
