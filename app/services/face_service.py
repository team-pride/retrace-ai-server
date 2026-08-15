"""얼굴 인식 핵심 로직 (우선순위 1).

- 셀피 이미지 여러 장으로부터 임베딩을 추출해 평균 기준 벡터를 계산한다.
- 대조 사진의 임베딩과 기준 벡터를 비교해 본인 여부를 판정한다.

DeepFace는 최초 호출 시점에 지연 임포트한다. 모델 가중치 다운로드 때문에
임포트 자체가 느리고, 헬스체크 등 얼굴 인식과 무관한 요청까지 지연시키지
않기 위함이다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.core.config import settings
from app.services.image_utils import InvalidImageError, bytes_to_ndarray
from app.services.model_loader import load_deepface

__all__ = [
    "InvalidImageError",
    "NoFaceDetectedError",
    "MultipleFacesDetectedError",
    "DifferentPersonError",
    "PersonMismatchError",
    "DetectedFace",
    "extract_embedding",
    "detect_all_faces",
    "select_matching_face",
    "average_embeddings",
    "cosine_distance",
    "is_same_person",
    "check_same_person_embeddings",
]


class NoFaceDetectedError(Exception):
    """이미지에서 얼굴을 찾지 못한 경우"""


class MultipleFacesDetectedError(Exception):
    """이미지에서 얼굴이 2개 이상 검출된 경우 (등록 시에는 1인만 허용)"""


class DifferentPersonError(Exception):
    """기준 벡터 등록용으로 올린 셀피 여러 장이 서로 다른 인물로 판정된 경우"""


class PersonMismatchError(Exception):
    """검출된 얼굴(들) 중 기준 벡터와 가장 가까운 것마저 임계값을 넘는 경우
    (= 등록된 본인이 사진에 없는 것으로 판정됨)"""


@dataclass
class DetectedFace:
    embedding: list[float]
    facial_area: dict
    confidence: float


def extract_embedding(image_bytes: bytes, *, allow_multiple: bool = False) -> list[float]:
    """이미지 바이트에서 얼굴 임베딩 벡터를 추출한다.

    원본 이미지는 디스크에 저장하지 않고 메모리에서만 처리한다 (원본 셀피 미저장 원칙).
    """
    deepface = load_deepface()
    img_array = bytes_to_ndarray(image_bytes)  # InvalidImageError는 호출부에서 처리

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


def detect_all_faces(image_bytes: bytes) -> list[DetectedFace]:
    """이미지에서 검출되는 모든 얼굴의 임베딩·위치·검출 신뢰도를 반환한다.

    extract_embedding과 달리 얼굴이 여러 개여도 예외를 던지지 않고 전부
    반환한다. 기준 벡터와 비교해 "본인" 얼굴 하나를 골라내는 용도로 쓰인다
    (select_matching_face 참고).
    """
    deepface = load_deepface()
    img_array = bytes_to_ndarray(image_bytes)  # InvalidImageError는 호출부에서 처리

    try:
        results = deepface.represent(
            img_path=img_array,
            model_name=settings.FACE_MODEL_NAME,
            detector_backend=settings.FACE_DETECTOR_BACKEND,
            enforce_detection=True,
        )
    except ValueError as exc:
        raise NoFaceDetectedError(str(exc)) from exc

    if not results:
        raise NoFaceDetectedError("얼굴을 검출하지 못했습니다.")

    return [
        DetectedFace(
            embedding=r["embedding"],
            facial_area=r["facial_area"],
            confidence=float(r.get("face_confidence", 0.0)),
        )
        for r in results
    ]


def select_matching_face(faces: Sequence[DetectedFace], reference_vector: Sequence[float]) -> DetectedFace:
    """검출된 얼굴들 중 기준 벡터와 가장 가까운 얼굴 하나를 고른다.

    기능명세서 2.2 "얼굴이 여러 개 검출되면 기준 벡터와 가장 가까운 얼굴 하나만
    사용한다" + "기준 벡터와의 거리로 본인 여부를 판정한다"를 함께 구현한다.
    가장 가까운 얼굴마저 FACE_MATCH_THRESHOLD를 넘으면(=사진에 본인이 없으면)
    PersonMismatchError를 던진다.
    """
    if not faces:
        raise NoFaceDetectedError("얼굴을 검출하지 못했습니다.")

    distance, best = min(
        ((cosine_distance(f.embedding, reference_vector), f) for f in faces),
        key=lambda pair: pair[0],
    )

    if distance > settings.FACE_MATCH_THRESHOLD:
        raise PersonMismatchError(
            f"검출된 얼굴이 등록된 본인 기준 벡터와 일치하지 않습니다 (거리 {distance:.3f}). "
            "본인이 나온 사진인지 확인해주세요."
        )

    return best


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


def check_same_person_embeddings(embeddings: Sequence[Sequence[float]]) -> None:
    """기준 벡터 등록용 셀피 여러 장의 임베딩이 모두 동일 인물인지 확인한다.

    등록 셀피 중 하나라도 다른 사람 사진이 섞이면 평균 기준 벡터가 오염돼
    이후 모든 본인 판정이 잘못된다. 그래서 평균을 내기 전에 모든 쌍의
    코사인 거리를 확인하고, 임계값(FACE_MATCH_THRESHOLD와 동일 기준)을
    넘는 쌍이 하나라도 있으면 DifferentPersonError를 던진다
    (유저플로우 문서 "동일 인물 여부?" 분기 / 기능명세서 1.2 예외:
    "서로 다른 인물이 섞여 임베딩 간 거리가 큰 경우 등록을 중단하고 같은
    사람의 사진을 요청한다").
    """
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            distance = cosine_distance(embeddings[i], embeddings[j])
            if distance > settings.FACE_MATCH_THRESHOLD:
                raise DifferentPersonError(
                    f"업로드한 사진들에서 서로 다른 얼굴이 감지되었습니다 (유사도 거리 {distance:.3f}). "
                    "같은 사람의 사진으로 다시 선택해주세요."
                )
