from pydantic import BaseModel


class PredictionPointResponse(BaseModel):
    captured_at: str
    predicted_value: float


class ObservedPointResponse(BaseModel):
    captured_at: str
    value: float


class EffectJudgeResponse(BaseModel):
    user_id: str
    indicator: str
    marker_id: str
    marker_date: str
    verdict: str  # "observed" | "not_observed" | "pending"
    reasons: list[str]
    before_count: int
    after_count: int
    noise_baseline: float | None
    mean_deviation: float | None
    confidence: str | None  # "high" | "medium" | "low"
    prediction_line: list[PredictionPointResponse]
    actual_after: list[ObservedPointResponse]
