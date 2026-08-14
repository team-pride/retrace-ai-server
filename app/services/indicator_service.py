"""얼굴 기하 지표 추출 (우선순위 4).

정규화된 표준 이미지(normalization_service.normalize_face의 출력)에서
dlib 68점 랜드마크를 얻어 순수 기하 지표 4가지를 계산한다.

DeepFace + retinaface는 눈 2개 위치만 제공해서 턱선·눈꺼풀·입가 랜드마크를
얻을 수 없다 (실측 확인함: facial_area에 left_eye/right_eye만 있고 코·입은 없음).
그래서 dlib의 68점 랜드마크 예측기(shape_predictor_68_face_landmarks.dat)를
추가로 사용해 얼굴 폭·턱선 각도·눈꺼풀 높이·입가 각도를 실제 해부학적
기준점으로 계산한다. 모델 파일은 리포에 직접 커밋하지 않고(90MB대라 너무
큼) pip 패키지 face_recognition_models(=dlib 공식 배포 모델을 그대로
번들링한 패키지)를 의존성으로 추가해서 설치 시점에 받아온다.

지표는 모두 눈동자 간 거리(IPD)를 기준자로 삼아 비율/각도로 환산한다
(기능명세서 3.1: "모든 지표를 눈동자 간 거리 대비 비율로 환산").

dlib 68점 인덱스 (iBUG 300-W 기준):
- 턱선: 0-16 (0=오른쪽 귀 쪽, 8=턱 끝, 16=왼쪽 귀 쪽)
- 오른쪽 눈썹: 17-21, 왼쪽 눈썹: 22-26
- 코: 27-35
- 오른쪽 눈: 36-41, 왼쪽 눈: 42-47
- 입 바깥쪽: 48-59, 입 안쪽: 60-67
"""
from __future__ import annotations

import importlib.util
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


class LandmarkDetectionError(Exception):
    """정규화된 이미지에서 68점 랜드마크를 찾지 못한 경우."""


@dataclass
class FaceIndicators:
    face_width_ratio: float
    jaw_angle_deg: float
    eyelid_height_ratio: float
    mouth_corner_angle_deg: float
    ipd_px: float


def _dlib_model_path() -> str:
    """face_recognition_models 패키지 안의 68점 랜드마크 모델(.dat) 경로를 찾는다.

    face_recognition_models는 `import face_recognition_models`만 해도
    __init__.py 최상단에서 `from pkg_resources import resource_filename`을
    실행한다. pkg_resources는 setuptools에 들어있던 구식 API인데, 최근
    setuptools(v82+)에서 아예 빠져버려서 "ModuleNotFoundError: No module
    named 'pkg_resources'"로 서버가 죽는 환경이 있다 (pip로 setuptools를
    새로 깔아도 최신 버전이면 여전히 없음).
    그래서 패키지를 임포트/실행하지 않고, importlib으로 설치 위치만 찾아서
    파일 경로를 직접 조립한다 (패키지 내부 구조는 고정되어 있음: models/*.dat).
    """
    spec = importlib.util.find_spec("face_recognition_models")
    if spec is None or spec.origin is None:
        raise LandmarkDetectionError(
            "face_recognition_models 패키지를 찾을 수 없습니다. `pip install -r requirements.txt`를 확인해주세요."
        )
    package_dir = os.path.dirname(spec.origin)
    model_path = os.path.join(package_dir, "models", "shape_predictor_68_face_landmarks.dat")
    if not os.path.exists(model_path):
        raise LandmarkDetectionError(f"랜드마크 모델 파일을 찾을 수 없습니다: {model_path}")
    return model_path


@lru_cache(maxsize=1)
def _load_dlib_predictor():
    """dlib은 무거운 임포트라 지연 로딩 + 캐싱한다 (model_loader.load_deepface와 동일한 패턴)."""
    import dlib

    model_path = _dlib_model_path()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor


def _landmarks_to_array(shape) -> np.ndarray:
    return np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float64)


def _detect_landmarks(image: np.ndarray) -> np.ndarray:
    """정규화된 RGB 이미지에서 68점 랜드마크 좌표 배열 (68, 2)을 반환한다."""
    detector, predictor = _load_dlib_predictor()
    dets = detector(image, 1)
    if not dets:
        raise LandmarkDetectionError("정규화된 이미지에서 얼굴 랜드마크를 찾지 못했습니다.")
    # 정규화 단계에서 이미 얼굴 하나만 통과시키지만, 혹시 여러 개 검출되면 가장 큰 것만 사용
    rect = max(dets, key=lambda d: d.width() * d.height())
    shape = predictor(image, rect)
    return _landmarks_to_array(shape)


def _fold_angle(angle_deg: float) -> float:
    """부호 있는 각도를 -90~90도로 접는다 (수평 기준 작은 쪽 각도).

    photo_quality_service._compute_roll_angle / normalization_service._signed_tilt_angle와
    동일한 패턴. 두 점의 좌우 순서가 뒤바뀌어도 항상 같은 값이 나오게 한다.
    """
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    return angle_deg


def _interpupillary_distance(landmarks: np.ndarray) -> float:
    right_eye_center = landmarks[36:42].mean(axis=0)
    left_eye_center = landmarks[42:48].mean(axis=0)
    return float(np.linalg.norm(left_eye_center - right_eye_center))


def _face_width_ratio(landmarks: np.ndarray, ipd: float) -> float:
    """턱선 최외곽 두 점(0, 16) 사이 거리 / IPD. 광대·턱 폭을 함께 반영하는 근사치."""
    width_px = float(np.linalg.norm(landmarks[0] - landmarks[16]))
    return width_px / ipd


def _jaw_angle_deg(landmarks: np.ndarray) -> float:
    """턱 끝(8)을 꼭짓점으로 턱선 위쪽 두 점(4, 12)까지의 두 벡터가 이루는 각도(0~180도).

    값이 작을수록 V라인(뾰족한 턱), 클수록 각지거나 둥근 턱에 가깝다.
    """
    chin = landmarks[8]
    left = landmarks[4]
    right = landmarks[12]
    v1 = left - chin
    v2 = right - chin
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        raise LandmarkDetectionError("턱선 각도를 계산할 수 없습니다.")
    cos_angle = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def _eyelid_height_ratio(landmarks: np.ndarray, ipd: float) -> float:
    """양쪽 눈의 (윗꺼풀-아랫꺼풀) 수직 개폐 거리를 평균 내 IPD로 나눈 값."""
    right_upper = landmarks[[37, 38]].mean(axis=0)
    right_lower = landmarks[[40, 41]].mean(axis=0)
    left_upper = landmarks[[43, 44]].mean(axis=0)
    left_lower = landmarks[[46, 47]].mean(axis=0)
    right_gap = np.linalg.norm(right_upper - right_lower)
    left_gap = np.linalg.norm(left_upper - left_lower)
    return float((right_gap + left_gap) / 2.0 / ipd)


def _mouth_corner_angle_deg(landmarks: np.ndarray) -> float:
    """양쪽 입꼬리(48, 54)를 잇는 선이 수평에서 얼마나 기울었는지 (부호 있음).

    정규화 단계에서 눈 중심선을 이미 수평으로 맞춰뒀기 때문에, 이 각도는
    머리 기울기가 아니라 입꼬리 자체의 비대칭(한쪽만 처짐/올라감)을 반영한다.
    """
    left_corner = landmarks[48]
    right_corner = landmarks[54]
    dx = right_corner[0] - left_corner[0]
    dy = right_corner[1] - left_corner[1]
    angle = math.degrees(math.atan2(dy, dx))
    return _fold_angle(angle)


def extract_indicators(normalized_image: np.ndarray) -> FaceIndicators:
    """normalization_service.normalize_face()의 출력 이미지에서 지표 4종을 계산한다."""
    landmarks = _detect_landmarks(normalized_image)
    ipd = _interpupillary_distance(landmarks)
    if ipd <= 0:
        raise LandmarkDetectionError("눈동자 간 거리를 계산할 수 없습니다.")

    return FaceIndicators(
        face_width_ratio=_face_width_ratio(landmarks, ipd),
        jaw_angle_deg=_jaw_angle_deg(landmarks),
        eyelid_height_ratio=_eyelid_height_ratio(landmarks, ipd),
        mouth_corner_angle_deg=_mouth_corner_angle_deg(landmarks),
        ipd_px=ipd,
    )
