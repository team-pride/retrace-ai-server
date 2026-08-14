from pydantic import BaseModel


class PhotoEvaluateResponse(BaseModel):
    photo_key: str
    grade: str  # "pass" | "conditional" | "exclude"
    reasons: list[str]
    angle_deg: float
    blur_variance: float
    detector_confidence: float
    attempt_count: int
    retries_remaining: int
    final_excluded: bool
