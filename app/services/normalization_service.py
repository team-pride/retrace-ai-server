"""사진 정규화 로직 (우선순위 3).

얼굴 랜드마크(양쪽 눈)를 기준으로 아래 순서를 한 번의 아핀 변환으로 처리한다.

- 눈 중심선 수평 정렬 (회전)
- 눈동자 간 거리 기준 크기 정렬 (스케일)
- 정렬 후 목표 크롭 영역이 원본 이미지 밖으로 나가면 CropOutOfBoundsError로 제외 처리

그 다음 눈 흰자(sclera) 영역을 참조 삼아 색조(화이트 밸런스)를 보정한다.

회전/스케일을 합친 아핀 변환은 dlib 계열에서 흔히 쓰는 face-align 방식
(눈 위치를 출력 이미지의 고정된 상대 좌표에 맞추는 방식)을 따른다.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from app.core.config import settings
from app.services import face_service
from app.services.face_service import MultipleFacesDetectedError, NoFaceDetectedError
from app.services.image_utils import bytes_to_ndarray
from app.services.model_loader import load_deepface


class CropOutOfBoundsError(Exception):
    """정렬 후 목표 크롭 영역이 원본 이미지 범위를 벗어난 경우"""


@dataclass
class NormalizationResult:
    image: np.ndarray  # RGB, uint8, (NORM_OUTPUT_HEIGHT, NORM_OUTPUT_WIDTH, 3)
    scale_factor: float
    rotation_deg: float
    eye_distance_px: float
    grade: str = "ok"  # "ok" | "conditional" (흰자 부족으로 색보정 생략)
    white_balance_skipped: bool = False


def image_to_png_bytes(image: np.ndarray) -> bytes:
    pil_img = Image.fromarray(image)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def _signed_tilt_angle(left_eye, right_eye) -> float:
    """눈 중심선이 수평에서 얼마나 기울어져 있는지 (부호 포함, -90~90도).

    DeepFace의 left_eye/right_eye는 해부학적 좌/우라 픽셀 좌표상 순서가
    반대일 수 있어서(photo_quality_service의 각도 계산과 동일한 이슈),
    -90~90도로 접어서 항상 "수평선 대비 작은 쪽" 각도를 쓴다.
    """
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = math.degrees(math.atan2(dy, dx))
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def _eye_distance(left_eye, right_eye) -> float:
    return float(math.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1]))


def build_alignment_matrix(left_eye, right_eye):
    """회전(눈 수평 정렬) + 스케일(눈동자 간 거리 정렬)을 합친 아핀 변환 행렬을 만든다.

    출력 이미지에서 두 눈은 (NORM_LEFT_EYE_X, NORM_EYE_Y) / (1-NORM_LEFT_EYE_X, NORM_EYE_Y)
    상대 위치에 오도록 배치한다.

    반환: (M, angle_deg, scale, current_eye_distance_px)
    """
    angle = _signed_tilt_angle(left_eye, right_eye)
    current_dist = _eye_distance(left_eye, right_eye)
    if current_dist <= 0:
        raise CropOutOfBoundsError("눈 사이 거리를 계산할 수 없습니다.")

    desired_dist = (1.0 - 2 * settings.NORM_LEFT_EYE_X) * settings.NORM_OUTPUT_WIDTH
    scale = desired_dist / current_dist

    eyes_center = (
        (left_eye[0] + right_eye[0]) / 2.0,
        (left_eye[1] + right_eye[1]) / 2.0,
    )

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    tx = settings.NORM_OUTPUT_WIDTH * 0.5
    ty = settings.NORM_OUTPUT_HEIGHT * settings.NORM_EYE_Y
    M[0, 2] += tx - eyes_center[0]
    M[1, 2] += ty - eyes_center[1]

    return M, angle, scale, current_dist


def _assert_crop_within_bounds(M, orig_width: int, orig_height: int) -> None:
    """출력 캔버스 네 모서리를 역변환해서 원본 이미지 범위 안에 들어오는지 확인한다.

    하나라도 범위를 벗어나면 그 부분은 원본에 없는 픽셀(검은 여백)로 채워진다는
    뜻이라 CropOutOfBoundsError를 던져서 이 사진을 제외 처리하게 한다.
    """
    full_M = np.vstack([M, [0.0, 0.0, 1.0]])
    inv_M = np.linalg.inv(full_M)
    corners = np.array(
        [
            [0, 0, 1],
            [settings.NORM_OUTPUT_WIDTH, 0, 1],
            [0, settings.NORM_OUTPUT_HEIGHT, 1],
            [settings.NORM_OUTPUT_WIDTH, settings.NORM_OUTPUT_HEIGHT, 1],
        ],
        dtype=np.float64,
    ).T
    src_corners = inv_M @ corners
    xs = src_corners[0, :]
    ys = src_corners[1, :]
    margin = settings.NORM_CROP_MARGIN_PX

    if xs.min() < -margin or xs.max() > orig_width + margin or ys.min() < -margin or ys.max() > orig_height + margin:
        raise CropOutOfBoundsError(
            "정렬 후 크롭 영역이 원본 이미지 범위를 벗어났습니다. "
            "얼굴이 더 크게, 정중앙에 나오도록 다시 촬영해주세요."
        )


def _white_balance_via_eye_patches(
    image: np.ndarray, left_eye_out, right_eye_out, patch_radius: int = 12
) -> tuple[np.ndarray, bool]:
    """눈 흰자(sclera)를 근사하는 밝은 픽셀을 눈 주변에서 샘플링해 화이트 밸런스를 보정한다.

    두 눈 주변 패치에서 밝기 상위 10% 픽셀을 sclera 근사치로 보고, 그 평균이
    무채색(회색)이 되도록 채널별 게인을 계산해 이미지 전체에 적용한다.
    극단적인 보정을 막기 위해 게인은 0.7~1.4로 clip한다.

    흰자로 볼만한 밝은 픽셀 표본이 충분히 확보되지 않으면(눈이 감겼거나,
    안경 반사, 얼굴이 이미지 경계에 걸쳐 패치가 잘리는 경우 등) 색보정을
    건너뛰고 원본을 그대로 반환한다 (기능명세서 2.3: "흰자 영역이 충분히
    확보되지 않으면 색온도 보정을 건너뛰고 조건부 등급으로 낮춘다").

    반환값: (이미지, 보정을_건너뛰었는지)
    """
    h, w = image.shape[:2]
    samples = []
    for ex, ey in (left_eye_out, right_eye_out):
        x0, x1 = max(0, int(ex - patch_radius)), min(w, int(ex + patch_radius))
        y0, y1 = max(0, int(ey - patch_radius)), min(h, int(ey + patch_radius))
        patch = image[y0:y1, x0:x1]
        if patch.size:
            samples.append(patch.reshape(-1, 3))

    if not samples:
        return image, True

    all_pixels = np.concatenate(samples, axis=0).astype(np.float64)
    if all_pixels.shape[0] < settings.NORM_MIN_WHITE_SAMPLE_PIXELS:
        return image, True

    brightness = all_pixels.mean(axis=1)
    threshold = np.percentile(brightness, 90)
    bright_pixels = all_pixels[brightness >= threshold]
    if bright_pixels.size == 0:
        return image, True

    reference = bright_pixels.mean(axis=0)  # [R, G, B]
    target = reference.mean()
    if target <= 1:
        return image, True

    gains = np.clip(target / np.clip(reference, 1, None), 0.7, 1.4)
    corrected = image.astype(np.float64) * gains
    return np.clip(corrected, 0, 255).astype(np.uint8), False


def _resolve_face_area(image_bytes: bytes, img_array: np.ndarray, reference_vector: Sequence[float] | None) -> dict:
    """검출 대상 얼굴의 facial_area(위치 정보)를 결정한다.

    reference_vector가 주어지면(=본인 기준 벡터가 등록된 사용자 컨텍스트)
    얼굴이 여러 개 검출돼도 예외를 던지지 않고 기준 벡터와 가장 가까운
    얼굴 하나를 골라 사용한다 (기준 벡터와 거리가 너무 멀면 본인이 아닌
    것으로 보고 PersonMismatchError). reference_vector가 없으면(온보딩
    이전 등, 비교 대상이 없는 경우) 기존처럼 얼굴이 정확히 하나여야 한다.
    """
    if reference_vector is not None:
        faces = face_service.detect_all_faces(image_bytes)
        best = face_service.select_matching_face(faces, reference_vector)
        return best.facial_area

    deepface = load_deepface()
    faces = deepface.extract_faces(
        img_path=img_array,
        detector_backend=settings.FACE_DETECTOR_BACKEND,
        enforce_detection=True,
        align=False,
    )
    if not faces:
        raise NoFaceDetectedError("얼굴을 검출하지 못했습니다.")
    if len(faces) > 1:
        raise MultipleFacesDetectedError(
            f"얼굴이 {len(faces)}개 검출되었습니다. 한 명만 나오도록 촬영해주세요."
        )
    return faces[0]["facial_area"]


def normalize_face(image_bytes: bytes, reference_vector: Sequence[float] | None = None) -> NormalizationResult:
    """얼굴을 검출해서 정렬(회전+스케일) → 크롭 범위 검증 → 화이트 밸런스 보정까지 수행한다.

    reference_vector를 넘기면 다중 얼굴 중 본인 기준 벡터와 가장 가까운
    얼굴을 선택하고, 그 얼굴마저 임계값을 넘으면 PersonMismatchError를 던진다.
    """
    img_array = bytes_to_ndarray(image_bytes)  # InvalidImageError는 호출부에서 처리

    try:
        area = _resolve_face_area(image_bytes, img_array, reference_vector)
    except ValueError as exc:
        raise NoFaceDetectedError(str(exc)) from exc

    left_eye = area.get("left_eye")
    right_eye = area.get("right_eye")
    if not left_eye or not right_eye:
        raise NoFaceDetectedError("눈 위치를 찾지 못해 정규화할 수 없습니다.")

    M, angle, scale, eye_distance = build_alignment_matrix(left_eye, right_eye)
    _assert_crop_within_bounds(M, img_array.shape[1], img_array.shape[0])

    output_size = (settings.NORM_OUTPUT_WIDTH, settings.NORM_OUTPUT_HEIGHT)
    aligned = cv2.warpAffine(img_array, M, output_size, flags=cv2.INTER_CUBIC)

    left_eye_out = (
        settings.NORM_OUTPUT_WIDTH * settings.NORM_LEFT_EYE_X,
        settings.NORM_OUTPUT_HEIGHT * settings.NORM_EYE_Y,
    )
    right_eye_out = (
        settings.NORM_OUTPUT_WIDTH * (1 - settings.NORM_LEFT_EYE_X),
        settings.NORM_OUTPUT_HEIGHT * settings.NORM_EYE_Y,
    )
    balanced, white_balance_skipped = _white_balance_via_eye_patches(aligned, left_eye_out, right_eye_out)

    return NormalizationResult(
        image=balanced,
        scale_factor=scale,
        rotation_deg=angle,
        eye_distance_px=eye_distance,
        grade="conditional" if white_balance_skipped else "ok",
        white_balance_skipped=white_balance_skipped,
    )
