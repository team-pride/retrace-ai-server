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
from datetime import date

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# EXIF 태그 번호 (PIL의 ExifTags 상수와 동일한 값)
_EXIF_SUBIFD_TAG = 0x8769  # Exif IFD Pointer — DateTimeOriginal은 이 서브 IFD 안에 있음
_EXIF_DATETIME_ORIGINAL = 0x9003
_TIFF_DATETIME = 0x0132  # IFD0의 DateTime (촬영일이 아니라 마지막 수정일에 가까움, 최후 fallback)


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


def extract_captured_date(image_bytes: bytes) -> date | None:
    """이미지의 EXIF 촬영일(DateTimeOriginal)을 읽어 date로 반환한다.

    카톡/에어드랍 등으로 전달돼 EXIF가 제거된 사진이나 스크린샷처럼 애초에
    EXIF가 없는 경우 None을 반환한다 (기능명세서 리스크 항목: "촬영일 소실").
    호출부에서 None이면 판정 대상에서 제외하고 사유를 기록해야 한다.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif = image.getexif()
    except (UnidentifiedImageError, OSError):
        return None

    date_str = None
    try:
        exif_ifd = exif.get_ifd(_EXIF_SUBIFD_TAG)
        date_str = exif_ifd.get(_EXIF_DATETIME_ORIGINAL)
    except (KeyError, AttributeError, ValueError):
        date_str = None

    if not date_str:
        # 실제 카메라 사진은 DateTimeOriginal이 Exif 서브 IFD 안에 중첩되어
        # 있지만, PIL로 exif를 직접 만들어 저장하면(테스트용 이미지 등)
        # 서브 IFD 없이 최상위에 그대로 저장되는 경우가 있어 여기서도 확인한다.
        date_str = exif.get(_EXIF_DATETIME_ORIGINAL)

    if not date_str:
        date_str = exif.get(_TIFF_DATETIME)

    if not date_str:
        return None

    try:
        # EXIF 날짜 형식: "YYYY:MM:DD HH:MM:SS"
        date_part = str(date_str).split(" ")[0]
        return date.fromisoformat(date_part.replace(":", "-"))
    except ValueError:
        return None
