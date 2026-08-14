import base64

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.photo import PhotoEvaluateResponse, PhotoNormalizeResponse
from app.services import face_service, normalization_service, photo_quality_service
from app.services.retry_tracker import get_retry_tracker

router = APIRouter(prefix="/photo", tags=["photo"])


@router.post("/normalize", response_model=PhotoNormalizeResponse)
async def normalize_photo(file: UploadFile = File(...)):
    """사진 정규화 (우선순위 3).

    눈 중심선 수평 정렬 + 눈동자 간 거리 기준 크기 정렬 + 눈 흰자 기준 색조 보정을
    적용한다. 정렬 후 목표 크롭 영역이 원본 이미지 범위를 벗어나면 grade="exclude"로
    제외 처리한다 (예외를 던지지 않고 응답으로 알려줘서 클라이언트가 재촬영 안내를
    바로 보여줄 수 있게 한다).
    """
    content = await file.read()
    try:
        result = normalization_service.normalize_face(content)
    except face_service.NoFaceDetectedError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="얼굴을 찾지 못했습니다.")
    except face_service.MultipleFacesDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except face_service.InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except normalization_service.CropOutOfBoundsError as exc:
        return PhotoNormalizeResponse(grade="exclude", reasons=[str(exc)])

    png_bytes = normalization_service.image_to_png_bytes(result.image)
    image_b64 = base64.b64encode(png_bytes).decode("ascii")

    return PhotoNormalizeResponse(
        grade="ok",
        reasons=[],
        rotation_deg=round(result.rotation_deg, 2),
        scale_factor=round(result.scale_factor, 3),
        eye_distance_px=round(result.eye_distance_px, 1),
        image_base64=image_b64,
    )


@router.post("/evaluate", response_model=PhotoEvaluateResponse)
async def evaluate_photo(photo_key: str, file: UploadFile = File(...)):
    """사진 판정 (우선순위 2).

    각도(±8도)/블러/얼굴 검출 신뢰도를 기준으로 pass / conditional / exclude
    등급을 매긴다.

    - photo_key: 재시도 횟수를 추적하는 단위 키. 예) "user123_2024-05_front"
      같은 photo_key로 판정에 PHOTO_MAX_RETRY(기본 3)회 실패(exclude)하면
      해당 사진은 최종 제외 처리된다.
    """
    tracker = get_retry_tracker()

    if tracker.get_count(photo_key) >= settings.PHOTO_MAX_RETRY:
        return PhotoEvaluateResponse(
            photo_key=photo_key,
            grade="exclude",
            reasons=[f"재시도 {settings.PHOTO_MAX_RETRY}회 초과로 이미 최종 제외된 사진입니다."],
            angle_deg=0.0,
            blur_variance=0.0,
            detector_confidence=0.0,
            attempt_count=tracker.get_count(photo_key),
            retries_remaining=0,
            final_excluded=True,
        )

    content = await file.read()
    try:
        metrics = photo_quality_service.detect_and_measure(content)
    except face_service.NoFaceDetectedError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="얼굴을 찾지 못했습니다.")
    except face_service.MultipleFacesDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except face_service.InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    result = photo_quality_service.grade_metrics(metrics)

    attempt_count = tracker.get_count(photo_key)
    final_excluded = False

    if result.grade == "exclude":
        attempt_count = tracker.record_failure(photo_key)
        if attempt_count >= settings.PHOTO_MAX_RETRY:
            final_excluded = True
            result.reasons.append(f"재시도 {settings.PHOTO_MAX_RETRY}회 초과로 최종 제외되었습니다.")
    else:
        # pass/conditional이면 재도전 없이 통과된 것이므로 실패 카운트 초기화
        tracker.reset(photo_key)
        attempt_count = 0

    return PhotoEvaluateResponse(
        photo_key=photo_key,
        grade=result.grade,
        reasons=result.reasons,
        angle_deg=round(metrics.angle_deg, 2),
        blur_variance=round(metrics.blur_variance, 2),
        detector_confidence=round(metrics.detector_confidence, 3),
        attempt_count=attempt_count,
        retries_remaining=max(0, settings.PHOTO_MAX_RETRY - attempt_count),
        final_excluded=final_excluded,
    )
