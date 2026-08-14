from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """서버 전역 설정.

    환경변수 접두사는 AI_ 이다. 예) AI_FACE_MATCH_THRESHOLD=0.4
    .env 파일 또는 실제 환경변수(Docker의 environment)로 덮어쓸 수 있다.
    """

    APP_NAME: str = "retrace-ai-server"
    DATA_DIR: str = "./data"

    # --- 얼굴 인식 설정 (우선순위 1) ---
    FACE_MODEL_NAME: str = "Facenet512"
    FACE_DETECTOR_BACKEND: str = "retinaface"
    # 코사인 거리 임계값: 이 값 이하면 본인으로 판정. 초기값이며 추후 실측 데이터로 튜닝 필요.
    FACE_MATCH_THRESHOLD: float = 0.40

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AI_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
