from pydantic import BaseModel


class MarkerResponse(BaseModel):
    marker_id: str
    user_id: str
    marker_date: str
    note: str
    created_at: str


class MarkerListResponse(BaseModel):
    user_id: str
    markers: list[MarkerResponse]
