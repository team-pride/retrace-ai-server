from pydantic import BaseModel


class PhotoNormalizeResponse(BaseModel):
    grade: str  # "ok" | "exclude"
    reasons: list[str]
    rotation_deg: float | None = None
    scale_factor: float | None = None
    eye_distance_px: float | None = None
    image_base64: str | None = None  # PNG, data URI 접두어 없이 순수 base64


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
