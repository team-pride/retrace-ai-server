from datetime import date

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
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
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/indicator", tags=["indicator"])


def _require_reference_vector(user_id: str) -> list[float]:
    """지표 추출 파이프라인의 선행조건: 본인 기준 벡터가 등록되어 있어야 한다
    (기능명세서 2.2 선행조건). 등록 안 된 사용자는 404로 안내한다.
    """
    reference_vector = get_vector_store().get(user_id)
    if reference_vector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="등록된 기준 얼굴 벡터가 없습니다. 먼저 /api/v1/face/register를 호출하세요.",
        )
    return reference_vector


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
    dlib 68점 랜드마크로 지표를 계산한다. 정규화 단계에서 등록된 본인 기준
    벡터와 비교해 얼굴이 여러 개 검출되면 가장 가까운 얼굴을 고르고, 그
    얼굴마저 본인이 아니라고 판단되면 422로 거부한다 (기능명세서 2.2).
    """
    try:
        date.fromisoformat(captured_at)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="captured_at은 YYYY-MM-DD 형식이어야 합니다.",
        )

    reference_vector = _require_reference_vector(user_id)

    content = await file.read()
    try:
        norm_result = normalization_service.normalize_face(content, reference_vector=reference_vector)
    except face_service.NoFaceDetectedError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="얼굴을 찾지 못했습니다.")
    except face_service.MultipleFacesDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except face_service.PersonMismatchError as exc:
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
    모아서 반환한다. 지원하지 않는 형식이거나 손상된 파일은 "skipped"로,
    얼굴 검출/본인 판정/랜드마크 등 판정 자체가 실패한 경우는 "failed"로 구분한다
    (기능명세서 2.1 "지원하지 않는 형식이거나 손상된 파일은 건너뛰고 사유를 기록한다").

    연도별 업로드 장수가 BATCH_MAX_PHOTOS_PER_YEAR(기본 30장)를 넘으면 초과분은
    건너뛰고, BATCH_MIN_PHOTOS_PER_YEAR(기본 5장) 미만인 연도는 차단하지 않되
    응답의 `year_notices`로 안내한다 (기능명세서 2.1 비즈니스 규칙: "연도별로
    업로드 최소 5장 ~ 최대 30장 제한").
    """
    if fallback_captured_at is not None:
        try:
            date.fromisoformat(fallback_captured_at)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="fallback_captured_at은 YYYY-MM-DD 형식이어야 합니다.",
            )

    # 본인 기준 벡터는 실제로 판정 단계까지 도달하는 파일이 있을 때만 필요하다.
    # (EXIF/fallback 둘 다 없어 날짜 자체를 못 정한 파일만 있는 경우까지 등록을
    # 강제하면 배치 업로드 자체가 막혀버리므로, 여기서는 404로 막지 않고
    # 아래 루프에서 실제로 필요한 시점에만 "failed" 사유로 안내한다.)
    reference_vector = get_vector_store().get(user_id)

    store = get_indicator_store()

    # 1단계: 각 파일을 읽고 촬영일(연도)을 먼저 확정한다 (연도별 상한 계산에 필요).
    items: list[dict] = []
    for upload in files:
        filename = upload.filename or "unnamed"
        content = await upload.read()
        captured_at = image_utils.extract_captured_date(content)
        captured_at_str = captured_at.isoformat() if captured_at else None
        if captured_at_str is None and fallback_captured_at is not None:
            captured_at_str = fallback_captured_at
        items.append({"filename": filename, "content": content, "captured_at_str": captured_at_str})

    # 2단계: 연도별 상한(BATCH_MAX_PHOTOS_PER_YEAR) 초과분을 미리 표시해둔다.
    year_counts: dict[int, int] = {}
    over_limit_indices: set[int] = set()
    for idx, item in enumerate(items):
        if item["captured_at_str"] is None:
            continue
        year = date.fromisoformat(item["captured_at_str"]).year
        year_counts[year] = year_counts.get(year, 0) + 1
        if year_counts[year] > settings.BATCH_MAX_PHOTOS_PER_YEAR:
            over_limit_indices.add(idx)

    year_notices = [
        f"{year}년 업로드 {count}장 (권장 최소 {settings.BATCH_MIN_PHOTOS_PER_YEAR}장에 미달)"
        for year, count in sorted(year_counts.items())
        if count < settings.BATCH_MIN_PHOTOS_PER_YEAR
    ]

    # 3단계: 실제 판정/지표 추출.
    results: list[BatchExtractItemResponse] = []

    for idx, item in enumerate(items):
        filename = item["filename"]
        content = item["content"]
        captured_at_str = item["captured_at_str"]

        if captured_at_str is None:
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

        if idx in over_limit_indices:
            year = date.fromisoformat(captured_at_str).year
            results.append(
                BatchExtractItemResponse(
                    filename=filename,
                    photo_key=filename,
                    status="skipped",
                    reason=f"{year}년 업로드 장수가 최대 {settings.BATCH_MAX_PHOTOS_PER_YEAR}장을 초과해 건너뛰었습니다.",
                )
            )
            continue

        if reference_vector is None:
            results.append(
                BatchExtractItemResponse(
                    filename=filename,
                    photo_key=filename,
                    status="failed",
                    reason="등록된 기준 얼굴 벡터가 없습니다. 먼저 /api/v1/face/register를 호출하세요.",
                )
            )
            continue

        try:
            norm_result = normalization_service.normalize_face(content, reference_vector=reference_vector)
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
        except face_service.PersonMismatchError as exc:
            results.append(
                BatchExtractItemResponse(filename=filename, photo_key=filename, status="failed", reason=str(exc))
            )
            continue
        except face_service.InvalidImageError as exc:
            # 지원하지 않는 형식/손상된 파일: 판정 실패가 아니라 건너뛴 것으로 기록한다.
            results.append(
                BatchExtractItemResponse(filename=filename, photo_key=filename, status="skipped", reason=str(exc))
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
        year_notices=year_notices,
    )
