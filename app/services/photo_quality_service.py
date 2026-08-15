"""사진 판정 로직 (우선순위 2).

촬영 각도(좌우 yaw / 상하 pitch)·블러·얼굴 검출 신뢰도를 측정해서
pass / conditional / exclude 등급을 매긴다. 표정·품질 판정은 1차로는
검출 신뢰도로 근사하고, 추후 DeepFace.analyze(emotion 등)로 확장할 수
있게 분리해뒀다.

각도는 플로우 문서 정의("정면 기준 좌우 15° 이내, 상하 10° 이내")를 따라
좌우 회전(yaw)과 상하 기울임(pitch) 두 축으로 판정한다. 처음 이 모듈을
만들 때는 DeepFace/retinaface가 눈 2개 좌표만 줘서 얼굴이 옆으로
기울었는지(roll)만 잴 수 있었는데, 그건 실제로 필요한 "정면에서 얼마나
돌아갔는지"(yaw/pitch)와는 다른 축이었다. 이후 우선순위 4에서 dlib 68점
랜드마크를 추가했으므로, 여기서도 그 랜드마크로 yaw/pitch를 근사한다.
(눈 기울기(roll)는 정규화 단계(우선순위 3)에서 어차피 수평으로 보정되므로
여기서 별도로 판정하지 않는다.)

재시도 로직(3회 실패 시 제외)은 이 모듈이 아니라 retry_tracker.py +
api/routes/photo.py 에서 처리한다. 이 모듈은 순수 판정 로직만 담당해서
테스트하기 쉽게 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from app.core.config import settings
from app.services import face_service, indicator_service
from app.services.face_service import MultipleFacesDetectedError, NoFaceDetectedError
from app.services.image_utils import bytes_to_ndarray
from app.services.model_loader import load_deepface


@dataclass
class PhotoMetrics:
    yaw_deg: float
    pitch_deg: float
    blur_variance: float
    detector_confidence: float


@dataclass
class PhotoGradeResult:
    grade: str  # "pass" | "conditional" | "exclude"
    reasons: list[str] = field(default_factory=list)
    metrics: PhotoMetrics | None = None


def _estimate_yaw_deg(landmarks: np.ndarray) -> float:
    """dlib 68점 랜드마크로 좌우 회전(yaw)을 근사한다.

    3D 머리 자세 추정(solvePnP + 카메라 내부 파라미터) 없이, 코 끝(30)이
    얼굴 폭 기준점(귀 쪽 0/16번)의 중심에서 좌우로 얼마나 치우쳐 있는지로
    근사한다. 완전 측면에 가까울수록 큰 값이 나오는 단조 근사치일 뿐 정확한
    각도 값은 아니다 — 실측 사진으로 스케일 보정이 필요하다.
    """
    left_x, right_x = landmarks[0][0], landmarks[16][0]
    nose_x = landmarks[30][0]
    face_width = right_x - left_x
    if face_width == 0:
        return 0.0
    center_x = (left_x + right_x) / 2.0
    offset_ratio = (nose_x - center_x) / (face_width / 2.0)
    return abs(offset_ratio) * 90.0


def _estimate_pitch_deg(landmarks: np.ndarray) -> float:
    """dlib 68점 랜드마크로 상하 기울임(pitch)을 근사한다.

    코 끝(30)이 눈 중심-턱 끝 구간에서 정면 기준 예상 위치(경험적으로
    약 45% 지점) 대비 얼마나 벗어나 있는지로 근사한다. yaw와 마찬가지로
    정확한 3D 각도가 아닌 단조 근사치이며 실측 보정이 필요하다.
    """
    eye_center_y = (landmarks[36:42].mean(axis=0)[1] + landmarks[42:48].mean(axis=0)[1]) / 2.0
    chin_y = landmarks[8][1]
    nose_y = landmarks[30][1]
    face_height = chin_y - eye_center_y
    if face_height == 0:
        return 0.0
    expected_ratio = 0.45
    actual_ratio = (nose_y - eye_center_y) / face_height
    return abs(actual_ratio - expected_ratio) * 90.0


def _compute_blur_variance(face_rgb: np.ndarray) -> float:
    """라플라시안 분산으로 블러 정도를 측정한다. 낮을수록 흐릿함."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _crop_with_margin(img_array: np.ndarray, facial_area: dict, margin_ratio: float = 0.3) -> np.ndarray:
    """facial_area(x, y, w, h) 주변에 여유를 두고 원본 이미지에서 얼굴 영역을 잘라낸다.

    numpy 슬라이싱 결과는 원본 배열의 뷰(view)라 메모리가 비연속(non-contiguous)일
    수 있는데, dlib은 ctypes로 이미지 버퍼를 직접 읽어서 비연속 배열을 넘기면
    예외 없이 조용히 얼굴을 못 찾거나 잘못된 결과를 낸다(실측으로 확인함 —
    동일한 크롭 영역인데 연속 배열로 복사하면 랜드마크 검출이 정상 동작).
    그래서 반환 직전에 np.ascontiguousarray로 항상 복사본을 만든다.
    """
    h, w = img_array.shape[:2]
    x, y, fw, fh = facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"]
    mx, my = int(fw * margin_ratio), int(fh * margin_ratio)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(w, x + fw + mx), min(h, y + fh + my)
    crop = img_array[y0:y1, x0:x1]
    if not crop.size:
        crop = img_array
    return np.ascontiguousarray(crop)


def detect_and_measure(image_bytes: bytes, reference_vector: Sequence[float] | None = None) -> PhotoMetrics:
    """얼굴을 검출하고 각도(yaw/pitch)/블러/검출 신뢰도를 측정한다.

    reference_vector가 주어지면(=본인 기준 벡터가 등록된 사용자 컨텍스트) 얼굴이
    여러 개 검출돼도 예외를 던지지 않고 기준 벡터와 가장 가까운 얼굴 하나만
    골라서 판정한다(기능명세서 2.2 "얼굴이 여러 개 검출되면 기준 벡터와 가장
    가까운 얼굴 하나만 사용한다"). 그 얼굴마저 임계값을 넘으면(본인이 아니면)
    PersonMismatchError를 던진다.

    reference_vector가 없으면(비교 대상이 없는 경우) 기존처럼 얼굴을 못 찾으면
    NoFaceDetectedError, 여러 명이 잡히면 MultipleFacesDetectedError를 던진다.
    """
    img_array = bytes_to_ndarray(image_bytes)  # InvalidImageError는 호출부에서 처리

    if reference_vector is not None:
        faces = face_service.detect_all_faces(image_bytes)
        best = face_service.select_matching_face(faces, reference_vector)
        confidence = best.confidence
        measure_source = _crop_with_margin(img_array, best.facial_area)
    else:
        deepface = load_deepface()
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
        confidence = float(face.get("confidence", 0.0))
        measure_source = img_array

    try:
        landmarks = indicator_service._detect_landmarks(measure_source)
        yaw = _estimate_yaw_deg(landmarks)
        pitch = _estimate_pitch_deg(landmarks)
    except indicator_service.LandmarkDetectionError:
        # dlib이 랜드마크를 못 찾으면(측면에 가까운 사진 등) 각도가 매우 크다고
        # 보수적으로 처리해 아래 판정 단계에서 자연스럽게 exclude 되게 한다.
        yaw = pitch = 999.0

    if reference_vector is not None:
        blur_variance = _compute_blur_variance(measure_source)
    else:
        face_arr = face["face"]
        # DeepFace는 0~1 범위 float 이미지를 반환하므로 0~255로 변환 후 블러 측정
        face_img = (face_arr * 255).astype(np.uint8) if face_arr.max() <= 1.0 else face_arr.astype(np.uint8)
        blur_variance = _compute_blur_variance(face_img)

    return PhotoMetrics(yaw_deg=yaw, pitch_deg=pitch, blur_variance=blur_variance, detector_confidence=confidence)


def grade_metrics(metrics: PhotoMetrics) -> PhotoGradeResult:
    """측정값을 기준으로 pass / conditional / exclude 등급을 매긴다.

    순수 함수라 DeepFace 없이도 단위 테스트 가능하다.
    """
    reasons: list[str] = []
    exclude = False
    conditional = False

    if metrics.yaw_deg > settings.PHOTO_YAW_EXCLUDE_DEG:
        exclude = True
        reasons.append(
            f"좌우 각도 {metrics.yaw_deg:.1f}도로 허용 범위({settings.PHOTO_YAW_EXCLUDE_DEG}도) 초과"
        )
    elif metrics.yaw_deg > settings.PHOTO_YAW_PASS_DEG:
        conditional = True
        reasons.append(
            f"좌우 각도 {metrics.yaw_deg:.1f}도로 기준({settings.PHOTO_YAW_PASS_DEG}도)보다 큼"
        )

    if metrics.pitch_deg > settings.PHOTO_PITCH_EXCLUDE_DEG:
        exclude = True
        reasons.append(
            f"상하 각도 {metrics.pitch_deg:.1f}도로 허용 범위({settings.PHOTO_PITCH_EXCLUDE_DEG}도) 초과"
        )
    elif metrics.pitch_deg > settings.PHOTO_PITCH_PASS_DEG:
        conditional = True
        reasons.append(
            f"상하 각도 {metrics.pitch_deg:.1f}도로 기준({settings.PHOTO_PITCH_PASS_DEG}도)보다 큼"
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
