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
    # 촬영 각도(도): PASS 이하는 통과, PASS~EXCLUDE는 조건부, EXCLUDE 초과는 제외.
    # 원본 문서 정의("정면 기준 좌우 15° 이내, 상하 10° 이내") 중 EXCLUDE
    # 경계를 그대로 쓰고, PASS는 기존 비율(EXCLUDE의 절반 정도)을 유지했다.
    PHOTO_YAW_PASS_DEG: float = 8.0  # 좌우 회전
    PHOTO_YAW_EXCLUDE_DEG: float = 15.0
    PHOTO_PITCH_PASS_DEG: float = 5.0  # 상하 기울임
    PHOTO_PITCH_EXCLUDE_DEG: float = 10.0
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
    # 흰자(sclera) 근사 색보정에 필요한 최소 샘플 픽셀 수. 이보다 적으면
    # "흰자 영역이 충분히 확보되지 않음"으로 보고 색보정을 건너뛰고 조건부 등급으로 낮춘다.
    NORM_MIN_WHITE_SAMPLE_PIXELS: int = 30

    # --- 지표/곡선 설정 (우선순위 4) ---
    # 곡선을 생성하려면 연도별 최소 이 장수 이상 통과해야 한다.
    CURVE_MIN_PHOTOS_PER_YEAR: int = 3
    # 곡선을 생성하려면 전체 통과 장수가 이 값 이상이어야 한다.
    CURVE_MIN_TOTAL_PHOTOS: int = 20
    # 일괄 업로드(연도별) 최소/최대 장수 제한 (기능명세서 2.1 비즈니스 규칙).
    # 최대치는 초과분을 건너뛰어 서버 부하를 막고, 최소치는 부족하다는 안내만
    # 하고 업로드 자체를 막지는 않는다(더 채워서 다시 올릴 수 있으므로).
    BATCH_MIN_PHOTOS_PER_YEAR: int = 5
    BATCH_MAX_PHOTOS_PER_YEAR: int = 30

    # --- 관리 효과 판정 설정 (우선순위 5) ---
    # 예측선(추세선)을 추정하려면 마커 이전 구간에 최소 이만큼의 지표 기록이 필요하다.
    EFFECT_MIN_POINTS_BEFORE_MARKER: int = 3
    # 실제 곡선과 예측선을 비교하려면 마커 이후 구간에 최소 이만큼의 지표 기록이 필요하다.
    EFFECT_MIN_POINTS_AFTER_MARKER: int = 2
    # 마커 이후 실측치가 예측선에서 벗어난 정도가 "데이터 흔들림 범위"(마커 이전
    # 구간의 잔차 표준편차)의 이 배수를 넘으면 "관찰됨"으로 판정한다.
    EFFECT_NOISE_THRESHOLD_FACTOR: float = 1.5
    # 판정 결과의 "구간 신뢰도" 표시 기준. 마커 이전+이후 지표 기록 수 합계 기준.
    EFFECT_CONFIDENCE_MEDIUM_MIN_TOTAL: int = 8
    EFFECT_CONFIDENCE_HIGH_MIN_TOTAL: int = 15

    model_config = SettingsConfigDict(env_file=".env", env_prefix="AI_", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
