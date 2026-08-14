from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """AWS/Docker 헬스체크, 스프링 서버 연동 테스트용."""
    return {"status": "ok"}
