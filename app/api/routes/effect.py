from fastapi import APIRouter, HTTPException, status

from app.schemas.effect import EffectJudgeResponse, ObservedPointResponse, PredictionPointResponse
from app.services import curve_service, effect_service

router = APIRouter(prefix="/effect", tags=["effect"])


@router.get("/judge", response_model=EffectJudgeResponse)
async def judge_effect(user_id: str, indicator: str, marker_id: str):
    """관리 마커를 기준으로 예측선(그대로 갔을 경우) vs 실제 곡선을 비교해
    변화 관찰 여부를 판정한다 (우선순위 5, 기능명세서 5.1).

    판정 결과는 세 가지 중 하나다:
    - observed: 마커 이후 실제 값이 예측선(마커 이전 추세를 이어간 값)에서
      데이터 흔들림 범위를 벗어나게 달라졌다.
    - not_observed: 마커 이후 실제 값이 예측선과 흔들림 범위 안에서 비슷하다.
    - pending: 마커 이전/이후 지표 표본이 부족해 판단할 수 없다.

    의료적 효과(좋아짐/나빠짐)를 단정하지 않는다 — "그대로 갔을 경우"와
    달라졌는지 여부만 판정한다.
    """
    try:
        result = effect_service.judge_effect(user_id, indicator, marker_id)
    except curve_service.UnknownIndicatorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except effect_service.MarkerNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))

    return EffectJudgeResponse(
        user_id=user_id,
        indicator=result.indicator,
        marker_id=result.marker_id,
        marker_date=result.marker_date.isoformat(),
        verdict=result.verdict,
        reasons=result.reasons,
        before_count=result.before_count,
        after_count=result.after_count,
        noise_baseline=round(result.noise_baseline, 6) if result.noise_baseline is not None else None,
        mean_deviation=round(result.mean_deviation, 6) if result.mean_deviation is not None else None,
        prediction_line=[
            PredictionPointResponse(
                captured_at=p.captured_at.isoformat(), predicted_value=round(p.predicted_value, 4)
            )
            for p in result.prediction_line
        ],
        actual_after=[
            ObservedPointResponse(captured_at=p.captured_at.isoformat(), value=round(p.value, 4))
            for p in result.actual_after
        ],
    )
