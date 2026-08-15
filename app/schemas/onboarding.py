from pydantic import BaseModel


class OnboardingStatusResponse(BaseModel):
    user_id: str
    acknowledged: bool
    acknowledged_at: str | None = None
