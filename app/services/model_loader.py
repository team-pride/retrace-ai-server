"""DeepFace 지연 로딩 공용 유틸.

DeepFace는 최초 사용 시점에 지연 임포트한다. 모델 가중치 다운로드 때문에
임포트 자체가 느리고, 헬스체크 등 얼굴 인식과 무관한 요청까지 지연시키지
않기 위함이다. face_service / photo_quality_service가 공통으로 사용한다.
"""
from __future__ import annotations


def load_deepface():
    try:
        from deepface import DeepFace
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "deepface가 설치되어 있지 않습니다. requirements.txt를 설치해주세요."
        ) from exc
    return DeepFace
