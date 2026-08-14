"""이미지 바이트 디코딩 공용 유틸.

face_service / photo_quality_service / normalization_service가 각자
동일한 디코딩 코드를 복붙해서 쓰고 있었는데, 손상되었거나 지원하지 않는
이미지 형식이 들어왔을 때 처리가 하나도 없어서 500 에러로 서버가 죽는
문제가 있었다 (PIL.UnidentifiedImageError가 그대로 위로 전파됨).
여기서 한 번에 잡아서 InvalidImageError로 변환한다.

또한 아이폰 등에서 촬영한 세로 사진은 픽셀 자체는 가로로 저장하고 EXIF
Orientation 태그로만 "이렇게 회전해서 보여줘"라고 표시하는 경우가 많다.
이 태그를 무시하고 원본 픽셀을 그대로 읽으면 얼굴이 90도 옆으로 누운
채로 얼굴 인식에 들어가서 검출에 실패한다. ImageOps.exif_transpose로
회전을 실제 픽셀에 반영한 뒤 처리한다.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImageError(Exception):
    """업로드된 파일을 이미지로 열 수 없는 경우 (손상되었거나 지원하지 않는 형식)."""


def bytes_to_ndarray(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # EXIF Orientation 태그를 실제 픽셀 회전으로 반영 (아이폰 세로 사진 대응)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise InvalidImageError(
            "이미지 파일을 읽을 수 없습니다. 파일이 손상되었거나 지원하지 않는 형식일 수 있습니다. "
            "JPG 또는 PNG 파일로 다시 시도해주세요."
        ) from exc
    except OSError as exc:
        # 잘려나간(truncated) 파일 등 PIL이 다른 예외로 던지는 경우도 있음
        raise InvalidImageError(
            "이미지 파일을 처리하는 중 오류가 발생했습니다. 파일이 손상되었을 수 있습니다."
        ) from exc
    return np.array(image)
