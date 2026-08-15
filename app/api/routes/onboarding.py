from fastapi import APIRouter

from app.schemas.onboarding import OnboardingStatusResponse
from app.services.onboarding_store import get_onboarding_store

router = APIRouter(prefix="/onboarding", tags=["onboarding"])


@router.post("/acknowledge", response_model=OnboardingStatusResponse)
async def acknowledge_onboarding(user_id: str):
    """측정 범위 고지 온보딩 화면 확인 처리 (기능명세서 1.1 "측정 범위 고지 온보딩").

    사용자가 온보딩 고지(측정 가능 항목/측정 불가 항목) 화면에서 확인 버튼을
    누르면 프론트가 호출한다. 확인 여부와 확인 시각을 사용자 단위로 저장해서,
    이후 세션에서는 이 화면을 다시 거치지 않고 바로 분석 흐름으로 진입할 수
    있게 한다. 설정 화면에서 "측정 범위 다시 보기"를 선택해 재노출하는 것은
    이 확인 상태를 바꾸지 않는다(별도 읽기 전용 조회이므로).
    """
    acknowledged_at = get_onboarding_store().acknowledge(user_id)
    return OnboardingStatusResponse(user_id=user_id, acknowledged=True, acknowledged_at=acknowledged_at)


@router.get("/status", response_model=OnboardingStatusResponse)
async def get_onboarding_status(user_id: str):
    """온보딩 고지 확인 여부를 조회한다.

    프론트가 앱 시작 시 호출해서, 이미 확인된 사용자면 온보딩 고지 화면을
    건너뛰고 바로 분석 흐름(기준 벡터 등록 또는 사진 업로드)으로 보낼 수 있다.
    """
    acknowledged_at = get_onboarding_store().get_acknowledged_at(user_id)
    return OnboardingStatusResponse(
        user_id=user_id,
        acknowledged=acknowledged_at is not None,
        acknowledged_at=acknowledged_at,
    )
