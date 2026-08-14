"""지표 시계열 곡선 생성 + 변화점 자동 탐지 (우선순위 4).

indicator_store에 쌓인 사용자의 지표 기록을 시계열로 재구성한다.

기능명세서 3.1 규칙을 그대로 따른다.
- 같은 날짜에 여러 장이 있으면 중앙값으로 묶고, 그 외에는 사진이 있는
  시점만으로 그래프를 구성한다 (일 단위로 보간하지 않는다).
- 곡선 생성 기준: 연도별 최소 CURVE_MIN_PHOTOS_PER_YEAR장 이상 통과 +
  전체 CURVE_MIN_TOTAL_PHOTOS장 이상. 미달이면 곡선을 만들지 않고 부족한
  조건을 안내한다.
- 표본이 적은 연도(< CURVE_MIN_PHOTOS_PER_YEAR)에 속한 구간은 신뢰도를
  낮게 표시한다 (곡선을 점선으로 그리는 판단은 프론트에서 하고, 여기서는
  confidence 값만 내려준다).
- 곡선 방향이 바뀌는 지점(변화점)을 탐지해 날짜/방향/변화폭을 기록한다.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from app.core.config import settings
from app.services.indicator_store import IndicatorRecord, get_indicator_store, parse_iso_date

INDICATOR_FIELDS = (
    "face_width_ratio",
    "jaw_angle_deg",
    "eyelid_height_ratio",
    "mouth_corner_angle_deg",
)


class UnknownIndicatorError(Exception):
    """지원하지 않는 지표 이름을 요청한 경우."""


@dataclass
class CurvePoint:
    captured_at: date
    value: float
    sample_count: int
    confidence: str  # "high" | "low"


@dataclass
class ChangePoint:
    captured_at: date
    direction: str  # "up" | "down" (그 지점부터 바뀐 방향)
    magnitude: float  # 전환 전후 기울기 차이 (값 단위 / 일)


@dataclass
class CurveResult:
    indicator: str
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    total_count: int = 0
    per_year_counts: dict[int, int] = field(default_factory=dict)
    points: list[CurvePoint] = field(default_factory=list)
    change_points: list[ChangePoint] = field(default_factory=list)


def _per_year_counts(records: list[IndicatorRecord]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for r in records:
        counts[parse_iso_date(r["captured_at"]).year] += 1
    return dict(sorted(counts.items()))


def _check_eligibility(records: list[IndicatorRecord]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    total = len(records)
    if total < settings.CURVE_MIN_TOTAL_PHOTOS:
        reasons.append(
            f"전체 통과 사진이 {settings.CURVE_MIN_TOTAL_PHOTOS}장 이상 필요해요 (현재 {total}장)."
        )

    per_year = _per_year_counts(records)
    if not per_year:
        reasons.append("판정을 통과한 사진이 아직 없어요.")
    else:
        for year, count in per_year.items():
            if count < settings.CURVE_MIN_PHOTOS_PER_YEAR:
                reasons.append(
                    f"{year}년 사진이 {settings.CURVE_MIN_PHOTOS_PER_YEAR}장 이상 필요해요 (현재 {count}장)."
                )

    return (len(reasons) == 0), reasons


def _group_by_date(records: list[IndicatorRecord], indicator: str) -> list[tuple[date, float, int]]:
    """같은 날짜의 기록은 중앙값으로 묶는다. (촬영일, 중앙값, 표본 수) 리스트를 날짜순으로 반환."""
    by_date: dict[date, list[float]] = defaultdict(list)
    for r in records:
        d = parse_iso_date(r["captured_at"])
        by_date[d].append(r[indicator])

    grouped = [(d, statistics.median(values), len(values)) for d, values in by_date.items()]
    grouped.sort(key=lambda item: item[0])
    return grouped


def _detect_change_points(points: list[CurvePoint]) -> list[ChangePoint]:
    """연속 구간 기울기의 부호가 바뀌는 지점(국소 방향 전환)을 찾는다."""
    if len(points) < 3:
        return []

    slopes: list[float] = []
    for i in range(len(points) - 1):
        days = (points[i + 1].captured_at - points[i].captured_at).days
        if days <= 0:
            slopes.append(0.0)
            continue
        slopes.append((points[i + 1].value - points[i].value) / days)

    change_points: list[ChangePoint] = []
    for i in range(1, len(slopes)):
        prev_slope, curr_slope = slopes[i - 1], slopes[i]
        if prev_slope == 0 or curr_slope == 0:
            continue
        if (prev_slope > 0) != (curr_slope > 0):
            change_points.append(
                ChangePoint(
                    captured_at=points[i].captured_at,
                    direction="up" if curr_slope > 0 else "down",
                    magnitude=abs(curr_slope - prev_slope),
                )
            )
    return change_points


def build_curve(user_id: str, indicator: str) -> CurveResult:
    if indicator not in INDICATOR_FIELDS:
        raise UnknownIndicatorError(
            f"알 수 없는 지표입니다: {indicator}. 사용 가능한 값: {', '.join(INDICATOR_FIELDS)}"
        )

    records = get_indicator_store().get_records(user_id)
    eligible, reasons = _check_eligibility(records)
    per_year = _per_year_counts(records)

    if not eligible:
        return CurveResult(
            indicator=indicator,
            eligible=False,
            reasons=reasons,
            total_count=len(records),
            per_year_counts=per_year,
        )

    grouped = _group_by_date(records, indicator)
    points = [
        CurvePoint(
            captured_at=d,
            value=value,
            sample_count=count,
            confidence="high" if per_year.get(d.year, 0) >= settings.CURVE_MIN_PHOTOS_PER_YEAR else "low",
        )
        for d, value, count in grouped
    ]
    change_points = _detect_change_points(points)

    return CurveResult(
        indicator=indicator,
        eligible=True,
        reasons=[],
        total_count=len(records),
        per_year_counts=per_year,
        points=points,
        change_points=change_points,
    )
