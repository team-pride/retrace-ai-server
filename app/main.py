from fastapi import FastAPI

from app.api.routes import face, health, indicator, photo
from app.core.config import settings

app = FastAPI(title=settings.APP_NAME)

app.include_router(health.router)
app.include_router(face.router, prefix="/api/v1")
app.include_router(photo.router, prefix="/api/v1")
app.include_router(indicator.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"service": settings.APP_NAME, "status": "running"}
