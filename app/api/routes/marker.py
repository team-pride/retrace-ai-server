from datetime import date

from fastapi import APIRouter, HTTPException, status

from app.schemas.marker import MarkerListResponse, MarkerResponse
from app.services.marker_store import get_marker_store

router = APIRouter(prefix="/marker", tags=["marker"])


@router.post("/register", response_model=MarkerResponse)
async def register_marker(user_id: str, marker_date: str, note: str):
    """관리 마커를 등록한다 (우선순위 5).

    - marker_date: 이 시점을 기준으로 지표 시계열이 "이전/이후"로 나뉘어
      관리 효과 판정(/api/v1/effect/judge)에 쓰인다. YYYY-MM-DD 형식.
    - note: 사용자가 입력한 원문 문장. 종류를 별도로 분류하지 않고 자유
      입력 그대로 저장한다 (기능명세서 4.1 비즈니스 규칙).
    """
    try:
        date.fromisoformat(marker_date)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="marker_date는 YYYY-MM-DD 형식이어야 합니다.",
        )

    record = get_marker_store().add_marker(user_id, marker_date, note)
    return MarkerResponse(user_id=user_id, **record)


@router.get("/list", response_model=MarkerListResponse)
async def list_markers(user_id: str):
    """사용자가 등록한 관리 마커 목록을 반환한다."""
    records = get_marker_store().get_markers(user_id)
    return MarkerListResponse(
        user_id=user_id,
        markers=[MarkerResponse(user_id=user_id, **r) for r in records],
    )
