from pydantic import BaseModel, Field


class FaceRegisterResponse(BaseModel):
    user_id: str
    registered_images: int
    message: str = "기준 얼굴 벡터가 등록되었습니다."


class FaceVerifyResponse(BaseModel):
    user_id: str
    is_match: bool
    distance: float = Field(..., description="코사인 거리 (0에 가까울수록 유사)")
    threshold: float
