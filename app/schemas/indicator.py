from pydantic import BaseModel


class IndicatorExtractResponse(BaseModel):
    user_id: str
    photo_key: str
    captured_at: str
    face_width_ratio: float
    jaw_angle_deg: float
    eyelid_height_ratio: float
    mouth_corner_angle_deg: float
    ipd_px: float


class CurvePointResponse(BaseModel):
    captured_at: str
    value: float
    sample_count: int
    confidence: str  # "high" | "low"


class ChangePointResponse(BaseModel):
    captured_at: str
    direction: str  # "up" | "down"
    magnitude: float


class CurveResponse(BaseModel):
    user_id: str
    indicator: str
    eligible: bool
    reasons: list[str]
    total_count: int
    per_year_counts: dict[str, int]
    points: list[CurvePointResponse]
    change_points: list[ChangePointResponse]


class BatchExtractItemResponse(BaseModel):
    filename: str
    photo_key: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str | None = None
    captured_at: str | None = None
    face_width_ratio: float | None = None
    jaw_angle_deg: float | None = None
    eyelid_height_ratio: float | None = None
    mouth_corner_angle_deg: float | None = None
    ipd_px: float | None = None


class BatchExtractResponse(BaseModel):
    user_id: str
    total_count: int
    succeeded_count: int
    skipped_count: int
    failed_count: int
    results: list[BatchExtractItemResponse]
