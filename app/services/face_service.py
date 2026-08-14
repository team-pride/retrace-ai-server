"""얼굴 인식 핵심 로직 (우선순위 1).

- 셀피 이미지 여러 장으로부터 임베딩을 추출해 평균 기준 벡터를 계산한다.
- 대조 사진의 임베딩과 기준 벡터를 비교해 본인 여부를 판정한다.

DeepFace는 최초 호출 시점에 지연 임포트한다. 모델 가중치 다운로드 때문에
임포트 자체가 느리고, 헬스체크 등 얼굴 인식과 무관한 요청까지 지연시키지
않기 위함이다.
"""
from __future__ import annotations

import io
from typing import Sequence

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.model_loader import load_deepface


class NoFaceDetectedError(Exception):
    """이미지에서 얼굴을 찾지 못한 경우"""


class MultipleFacesDetectedError(Exception):
    """이미지에서 얼굴이 2개 이상 검출된 경우 (등록 시에는 1인만 허용)"""


def _bytes_to_ndarray(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(image)


def extract_embedding(image_bytes: bytes, *, allow_multiple: bool = False) -> list[float]:
    """이미지 바이트에서 얼굴 임베딩 벡터를 추출한다.

    원본 이미지는 디스크에 저장하지 않고 메모리에서만 처리한다 (원본 셀피 미저장 원칙).
    """
    deepface = load_deepface()
    img_array = _bytes_to_ndarray(image_bytes)

    try:
        results = deepface.represent(
            img_path=img_array,
            model_name=settings.FACE_MODEL_NAME,
            detector_backend=settings.FACE_DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError as exc:
        # DeepFace는 얼굴을 못 찾으면 ValueError를 던진다
        raise NoFaceDetectedError(str(exc)) from exc

    if not results:
        raise NoFaceDetectedError("얼굴을 검출하지 못했습니다.")
    if len(results) > 1 and not allow_multiple:
        raise MultipleFacesDetectedError(
            f"얼굴이 {len(results)}개 검출되었습니다. 한 명만 나오도록 촬영해주세요."
        )

    return results[0]["embedding"]


def average_embeddings(embeddings: Sequence[Sequence[float]]) -> list[float]:
    if not embeddings:
        raise ValueError("평균을 계산할 임베딩이 없습니다.")
    matrix = np.array(embeddings, dtype=np.float64)
    return matrix.mean(axis=0).tolist()


def cosine_distance(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    a = np.array(vec_a, dtype=np.float64)
    b = np.array(vec_b, dtype=np.float64)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    similarity = float(np.dot(a, b) / denom)
    return 1.0 - similarity


def is_same_person(vec_a: Sequence[float], vec_b: Sequence[float]) -> tuple[bool, float]:
    distance = cosine_distance(vec_a, vec_b)
    return distance <= settings.FACE_MATCH_THRESHOLD, distance
