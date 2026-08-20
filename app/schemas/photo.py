from pydantic import BaseModel


class PhotoNormalizeResponse(BaseModel):
    grade: str  # "ok" | "conditional" | "exclude"
    reasons: list[str]
    rotation_deg: float | None = None
    scale_factor: float | None = None
    eye_distance_px: float | None = None
    image_base64: str | None = None  # PNG, data URI 접두어 없이 순수 base64


class PhotoEvaluateResponse(BaseModel):
    photo_key: str
    grade: str  # "pass" | "conditional" | "exclude"
    reasons: list[str]
    yaw_deg: float
    pitch_deg: float
    blur_variance: float
    detector_confidence: float
    attempt_count: int
    retries_remaining: int
    final_excluded: bool


class PhotoEvaluateBatchItemResponse(BaseModel):
    filename: str
    photo_key: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str | None = None
    grade: str | None = None
    reasons: list[str] | None = None
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    blur_variance: float | None = None
    detector_confidence: float | None = None
    attempt_count: int | None = None
    retries_remaining: int | None = None
    final_excluded: bool | None = None


class PhotoEvaluateBatchResponse(BaseModel):
    total_count: int
    succeeded_count: int
    skipped_count: int
    failed_count: int
    results: list[PhotoEvaluateBatchItemResponse]
