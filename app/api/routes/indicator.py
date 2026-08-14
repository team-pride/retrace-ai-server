from datetime import date

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.indicator import (
    BatchExtractItemResponse,
    BatchExtractResponse,
    ChangePointResponse,
    CurvePointResponse,
    CurveResponse,
    IndicatorExtractResponse,
)
from app.services import curve_service, face_service, image_utils, indicator_service, normalization_service
from app.services.indicator_store import IndicatorRecord, get_indicator_store

router = APIRouter(prefix="/indicator", tags=["indicator"])


@router.post("/extract", response_model=IndicatorExtractResponse)
async def extract_indicator(
    user_id: str,
    photo_key: str,
    captured_at: str,
    file: UploadFile = File(...),
):
    """사진 한 장에서 얼굴 기하 지표 4종을 추출해 저장한다 (우선순위 4).

    - captured_at: 사진 촬영일, "YYYY-MM-DD" 형식. 시계열 곡선의 x축으로 쓰인다.
    - photo_key: 같은 키로 다시 호출하면 이전 기록을 덮어쓴다 (재추출 시 중복 방지).

    내부적으로 정규화(우선순위 3)를 먼저 거친 뒤 정규화된 표준 이미지에서
    dlib 68점 랜드마크로 지표를 계산한다.
    """
    try:
        date.fromisoformat(captured_at)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="captured_at은 YYYY-MM-DD 형식이어야 합니다.",
        )

    content = await file.read()
    try:
        norm_result = normalization_service.normalize_face(content)
    except face_service.NoFaceDetectedError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="얼굴을 찾지 못했습니다.")
    except face_service.MultipleFacesDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except face_service.InvalidImageError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except normalization_service.CropOutOfBoundsError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        indicators = indicator_service.extract_indicators(norm_result.image)
    except indicator_service.LandmarkDetectionError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    record: IndicatorRecord = {
        "photo_key": photo_key,
        "captured_at": captured_at,
        "face_width_ratio": indicators.face_width_ratio,
        "jaw_angle_deg": indicators.jaw_angle_deg,
        "eyelid_height_ratio": indicators.eyelid_height_ratio,
        "mouth_corner_angle_deg": indicators.mouth_corner_angle_deg,
        "ipd_px": indicators.ipd_px,
    }
    get_indicator_store().add_record(user_id, record)

    return IndicatorExtractResponse(user_id=user_id, **record)


@router.get("/curve", response_model=CurveResponse)
async def get_curve(user_id: str, indicator: str):
    """저장된 지표 기록으로 시계열 곡선 + 변화점을 계산해 반환한다 (우선순위 4).

    indicator: face_width_ratio | jaw_angle_deg | eyelid_height_ratio | mouth_corner_angle_deg

    표본이 기준(연도별 3장, 전체 20장)에 못 미치면 eligible=false와 함께
    무엇이 부족한지 reasons로 알려주고 곡선은 만들지 않는다.
    """
    try:
        result = curve_service.build_curve(user_id, indicator)
    except curve_service.UnknownIndicatorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    return CurveResponse(
        user_id=user_id,
        indicator=result.indicator,
        eligible=result.eligible,
        reasons=result.reasons,
        total_count=result.total_count,
        per_year_counts={str(year): count for year, count in result.per_year_counts.items()},
        points=[
            CurvePointResponse(
                captured_at=p.captured_at.isoformat(),
                value=round(p.value, 4),
                sample_count=p.sample_count,
                confidence=p.confidence,
            )
            for p in result.points
        ],
        change_points=[
            ChangePointResponse(
                captured_at=c.captured_at.isoformat(),
                direction=c.direction,
                magnitude=round(c.magnitude, 6),
            )
            for c in result.change_points
        ],
    )


@router.post("/extract-batch", response_model=BatchExtractResponse)
async def extract_indicator_batch(
    user_id: str,
    fallback_captured_at: str | None = None,
    files: list[UploadFile] = File(...),
):
    """사진 여러 장을 한 번에 업로드해 지표를 일괄 추출/저장한다 (우선순위 4).

    사진마다 촬영일을 일일이 넣지 않아도 되도록, 각 파일의 EXIF
    DateTimeOriginal을 자동으로 읽어 촬영일로 사용한다 (기능명세서 2.1:
    "선택된 파일 각각에서 촬영일 정보를 추출한다").

    - fallback_captured_at: EXIF 촬영일이 없는 사진(카톡 전송, 스크린샷 등)에
      적용할 기본 촬영일 (YYYY-MM-DD). 생략하면 EXIF가 없는 사진은 건너뛴다.
      테스트용 이미지처럼 EXIF가 애초에 없는 파일로 곡선 기능을 시험해볼 때 유용하다.
    - photo_key는 업로드 파일명을 그대로 사용한다. 같은 파일명으로 다시
      올리면 이전 기록을 덮어쓴다.

    한 장이 실패해도 전체 요청을 실패시키지 않고, 파일별 결과(ok/skipped/failed)를
    모아서 반환한다.
    """
    if fallback_captured_at is not None:
        try:
            date.fromisoformat(fallback_captured_at)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="fallback_captured_at은 YYYY-MM-DD 형식이어야 합니다.",
            )

    store = get_indicator_store()
    results: list[BatchExtractItemResponse] = []

    for upload in files:
        filename = upload.filename or "unnamed"
        content = await upload.read()

        captured_at = image_utils.extract_captured_date(content)
        captured_at_str = captured_at.isoformat() if captured_at else None
        if captured_at_str is None:
            if fallback_captured_at is not None:
                captured_at_str = fallback_captured_at
            else:
                results.append(
                    BatchExtractItemResponse(
                        filename=filename,
                        photo_key=filename,
                        status="skipped",
                        reason="사진에서 촬영일(EXIF)을 찾을 수 없습니다. fallback_captured_at을 지정하거나 "
                        "원본 사진(캡처/편집되지 않은)을 올려주세요.",
                    )
                )
                continue

        try:
            norm_result = normalization_service.normalize_face(content)
        except face_service.NoFaceDetectedError:
            results.append(
                BatchExtractItemResponse(
                    filename=filename, photo_key=filename, status="failed", reason="얼굴을 찾지 못했습니다."
                )
            )
            continue
        except face_service.MultipleFacesDetectedError as exc:
            results.append(
                BatchExtractItemResponse(filename=filename, photo_key=filename, status="failed", reason=str(exc))
            )
            continue
        except face_service.InvalidImageError as exc:
            results.append(
                BatchExtractItemResponse(filename=filename, photo_key=filename, status="failed", reason=str(exc))
            )
            continue
        except normalization_service.CropOutOfBoundsError as exc:
            results.append(
                BatchExtractItemResponse(filename=filename, photo_key=filename, status="failed", reason=str(exc))
            )
            continue

        try:
            indicators = indicator_service.extract_indicators(norm_result.image)
        except indicator_service.LandmarkDetectionError as exc:
            results.append(
                BatchExtractItemResponse(filename=filename, photo_key=filename, status="failed", reason=str(exc))
            )
            continue

        record: IndicatorRecord = {
            "photo_key": filename,
            "captured_at": captured_at_str,
            "face_width_ratio": indicators.face_width_ratio,
            "jaw_angle_deg": indicators.jaw_angle_deg,
            "eyelid_height_ratio": indicators.eyelid_height_ratio,
            "mouth_corner_angle_deg": indicators.mouth_corner_angle_deg,
            "ipd_px": indicators.ipd_px,
        }
        store.add_record(user_id, record)

        results.append(
            BatchExtractItemResponse(
                filename=filename,
                photo_key=filename,
                status="ok",
                captured_at=captured_at_str,
                face_width_ratio=indicators.face_width_ratio,
                jaw_angle_deg=indicators.jaw_angle_deg,
                eyelid_height_ratio=indicators.eyelid_height_ratio,
                mouth_corner_angle_deg=indicators.mouth_corner_angle_deg,
                ipd_px=indicators.ipd_px,
            )
        )

    succeeded_count = sum(1 for r in results if r.status == "ok")
    skipped_count = sum(1 for r in results if r.status == "skipped")
    failed_count = sum(1 for r in results if r.status == "failed")

    return BatchExtractResponse(
        user_id=user_id,
        total_count=len(results),
        succeeded_count=succeeded_count,
        skipped_count=skipped_count,
        failed_count=failed_count,
        results=results,
    )
