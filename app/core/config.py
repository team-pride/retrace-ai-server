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

    # --- 사진 판정 설정 (우선순위 2) ---
    # 각도(도): PASS 이하는 통과, PASS~EXCLUDE는 조건부, EXCLUDE 초과는 제외
    PHOTO_ANGLE_PASS_DEG: float = 8.0
    PHOTO_ANGLE_EXCLUDE_DEG: float = 15.0
    # 블러(라플라시안 분산): 낮을수록 흐림. THRESHOLD 이상이면 통과.
    PHOTO_BLUR_VARIANCE_THRESHOLD: float = 80.0
    # 같은 photo_key로 판정 실패(exclude) 시 허용 재시도 횟수
    PHOTO_MAX_RETRY: int = 3

    # --- 정규화 설정 (우선순위 3) ---
    NORM_OUTPUT_WIDTH: int = 400
    NORM_OUTPUT_HEIGHT: int = 500
    # 출력 이미지에서 왼쪽 눈의 상대 x 위치 (오른쪽 눈은 1 - 이 값에 위치)
    NORM_LEFT_EYE_X: float = 0.35
    # 출력 이미지에서 두 눈의 상대 y 위치
    NORM_EYE_Y: float = 0.35
    # 정렬 후 크롭 범위 판정 여유(px). 0이면 원본을 조금이라도 벗어나면 바로 제외.
    NORM_CROP_MARGIN_PX: float = 0.0

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AI_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
