import io

import pytest
from PIL import Image

from app.services.image_utils import InvalidImageError, bytes_to_ndarray


def test_bytes_to_ndarray_valid_image():
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), color=(10, 20, 30)).save(buf, format="PNG")
    arr = bytes_to_ndarray(buf.getvalue())
    assert arr.shape == (3, 4, 3)


def test_bytes_to_ndarray_raises_on_garbage():
    with pytest.raises(InvalidImageError):
        bytes_to_ndarray(b"this is definitely not an image file, just text bytes")


def test_bytes_to_ndarray_raises_on_empty():
    with pytest.raises(InvalidImageError):
        bytes_to_ndarray(b"")


def test_bytes_to_ndarray_applies_exif_orientation():
    # 아이폰처럼 EXIF Orientation 태그로 회전 정보를 저장하는 경우를 재현한다.
    # 원본 픽셀은 4(width) x 2(height)지만, Orientation=6(90도 회전 필요)이 붙으면
    # 실제로 보여줘야 할 이미지는 2(width) x 4(height)로 가로/세로가 뒤바뀐다.
    img = Image.new("RGB", (4, 2), color=(10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation 태그
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    arr = bytes_to_ndarray(buf.getvalue())

    # EXIF 회전이 반영됐다면 배열의 height/width가 원본과 뒤바뀌어야 한다.
    assert arr.shape[0] == 4
    assert arr.shape[1] == 2
