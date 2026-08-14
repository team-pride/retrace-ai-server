from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.schemas.face import FaceRegisterResponse, FaceVerifyResponse
from app.services import face_service
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/face", tags=["face"])


@router.post("/register", response_model=FaceRegisterResponse)
async def register_face(user_id: str, files: List[UploadFile] = File(...)):
    """본인 확인용 기준 얼굴 벡터 등록 (셀피 여러 장 -> 평균 벡터).

    원본 셀피 이미지는 저장하지 않고, 추출된 벡터만 저장한다.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="이미지가 필요합니다.")

    embeddings = []
    for f in files:
        content = await f.read()
        try:
            embedding = face_service.extract_embedding(content)
        except face_service.NoFaceDetectedError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"{f.filename}에서 얼굴을 찾지 못했습니다.",
            )
        except face_service.MultipleFacesDetectedError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
        embeddings.append(embedding)

    reference_vector = face_service.average_embeddings(embeddings)
    get_vector_store().save(user_id, reference_vector)

    return FaceRegisterResponse(user_id=user_id, registered_images=len(files))


@router.post("/verify", response_model=FaceVerifyResponse)
async def verify_face(user_id: str, file: UploadFile = File(...)):
    """대조 사진과 기준 벡터를 비교해 본인 여부를 판정한다."""
    reference_vector = get_vector_store().get(user_id)
    if reference_vector is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="등록된 기준 얼굴 벡터가 없습니다. 먼저 /api/v1/face/register를 호출하세요.",
        )

    content = await file.read()
    try:
        embedding = face_service.extract_embedding(content)
    except face_service.NoFaceDetectedError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="얼굴을 찾지 못했습니다.")
    except face_service.MultipleFacesDetectedError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    is_match, distance = face_service.is_same_person(embedding, reference_vector)

    return FaceVerifyResponse(
        user_id=user_id,
        is_match=is_match,
        distance=round(distance, 4),
        threshold=settings.FACE_MATCH_THRESHOLD,
    )
