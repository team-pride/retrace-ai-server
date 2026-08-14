"""사진 판정 로직 (우선순위 2).

각도(±8도)/블러/얼굴 검출 신뢰도를 측정해서 pass / conditional / exclude
등급을 매긴다. 표정·품질 판정은 1차로는 검출 신뢰도로 근사하고,
추후 DeepFace.analyze(emotion 등)로 확장할 수 있게 분리해뒀다.

재시도 로직(3회 실패 시 제외)은 이 모듈이 아니라 retry_tracker.py +
api/routes/photo.py 에서 처리한다. 이 모듈은 순수 판정 로직만 담당해서
테스트하기 쉽게 유지한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from app.core.config import settings
from app.services.face_service import MultipleFacesDetectedError, NoFaceDetectedError
from app.services.image_utils import bytes_to_ndarray
from app.services.model_loader import load_deepface


@dataclass
class PhotoMetrics:
    angle_deg: float
    blur_variance: float
    detector_confidence: float


@dataclass
class PhotoGradeResult:
    grade: str  # "pass" | "conditional" | "exclude"
    reasons: list[str] = field(default_factory=list)
    metrics: PhotoMetrics | None = None


def _compute_roll_angle(left_eye, right_eye) -> float:
    """두 눈 좌표로 얼굴 기울기(roll) 각도를 계산한다. 0도에 가까울수록 수평.

    DeepFace의 left_eye/right_eye는 해부학적(사람 기준) 좌/우라서 이미지
    픽셀 좌표상으로는 left_eye.x가 더 클 때가 많다 (거울 반전이 아닌 경우).
    그래서 raw atan2 값은 0도가 아니라 180도 근처로 나올 수 있으므로,
    -90~90도 범위로 접어서 "수평선 대비 기울기" 크기만 남긴다.
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return abs(angle)


def _compute_blur_variance(face_rgb: np.ndarray) -> float:
    """라플라시안 분산으로 블러 정도를 측정한다. 낮을수록 흐릿함."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_and_measure(image_bytes: bytes) -> PhotoMetrics:
    """얼굴을 검출하고 각도/블러/검출 신뢰도를 측정한다.

    얼굴을 못 찾으면 NoFaceDetectedError, 여러 명이 잡히면
    MultipleFacesDetectedError를 던진다 (face_service와 동일한 예외 재사용).
    """
    deepface = load_deepface()
    img_array = bytes_to_ndarray(image_bytes)  # InvalidImageError는 호출부에서 처리

    try:
        faces = deepface.extract_faces(
            img_path=img_array,
            detector_backend=settings.FACE_DETECTOR_BACKEND,
            enforce_detection=True,
            align=False,
        )
    except ValueError as exc:
        raise NoFaceDetectedError(str(exc)) from exc

    if not faces:
        raise NoFaceDetectedError("얼굴을 검출하지 못했습니다.")
    if len(faces) > 1:
        raise MultipleFacesDetectedError(
            f"얼굴이 {len(faces)}개 검출되었습니다. 한 명만 나오도록 촬영해주세요."
        )

    face = faces[0]
    area = face["facial_area"]
    confidence = float(face.get("confidence", 0.0))

    left_eye = area.get("left_eye")
    right_eye = area.get("right_eye")
    angle = _compute_roll_angle(left_eye, right_eye) if left_eye and right_eye else 0.0

    face_arr = face["face"]
    # DeepFace는 0~1 범위 float 이미지를 반환하므로 0~255로 변환 후 블러 측정
    face_img = (face_arr * 255).astype(np.uint8) if face_arr.max() <= 1.0 else face_arr.astype(np.uint8)
    blur_variance = _compute_blur_variance(face_img)

    return PhotoMetrics(angle_deg=angle, blur_variance=blur_variance, detector_confidence=confidence)


def grade_metrics(metrics: PhotoMetrics) -> PhotoGradeResult:
    """측정값을 기준으로 pass / conditional / exclude 등급을 매긴다.

    순수 함수라 DeepFace 없이도 단위 테스트 가능하다.
    """
    reasons: list[str] = []
    exclude = False
    conditional = False

    if metrics.angle_deg > settings.PHOTO_ANGLE_EXCLUDE_DEG:
        exclude = True
        reasons.append(
            f"각도 {metrics.angle_deg:.1f}도로 허용 범위({settings.PHOTO_ANGLE_EXCLUDE_DEG}도) 초과"
        )
    elif metrics.angle_deg > settings.PHOTO_ANGLE_PASS_DEG:
        conditional = True
        reasons.append(
            f"각도 {metrics.angle_deg:.1f}도로 기준({settings.PHOTO_ANGLE_PASS_DEG}도)보다 큼"
        )

    if metrics.blur_variance < settings.PHOTO_BLUR_VARIANCE_THRESHOLD / 2:
        exclude = True
        reasons.append(f"심한 블러 (선명도 {metrics.blur_variance:.1f})")
    elif metrics.blur_variance < settings.PHOTO_BLUR_VARIANCE_THRESHOLD:
        conditional = True
        reasons.append(f"블러 의심 (선명도 {metrics.blur_variance:.1f})")

    if metrics.detector_confidence < 0.5:
        conditional = True
        reasons.append(f"얼굴 검출 신뢰도 낮음 ({metrics.detector_confidence:.2f})")

    if exclude:
        grade = "exclude"
    elif conditional:
        grade = "conditional"
    else:
        grade = "pass"

    return PhotoGradeResult(grade=grade, reasons=reasons, metrics=metrics)
