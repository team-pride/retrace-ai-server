"""온보딩(측정 범위 고지) 확인 여부 저장소 (기능명세서 1.1).

"온보딩 고지 확인 여부가 사용자 상태로 저장된다... 이후 세션에서는 이
화면을 다시 거치지 않고 바로 분석 흐름으로 진입한다"를 구현한다.
vector_store.py와 동일한 JSON 파일 기반 패턴. 나중에 DB로 교체 예정.
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


class OnboardingStore(ABC):
    @abstractmethod
    def acknowledge(self, user_id: str) -> str:
        """확인 처리하고 확인 시각(ISO 8601 문자열)을 반환한다."""

    @abstractmethod
    def get_acknowledged_at(self, user_id: str) -> str | None:
        """확인한 적이 없으면 None을 반환한다."""


class JSONFileOnboardingStore(OnboardingStore):
    """개발/테스트용 로컬 파일 저장소. 운영 환경에서는 DB로 교체 예정."""

    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._lock = threading.Lock()
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self._file_path.exists():
            self._file_path.write_text("{}", encoding="utf-8")

    def _read_all(self) -> dict:
        with self._file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write_all(self, data: dict) -> None:
        with self._file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    def acknowledge(self, user_id: str) -> str:
        with self._lock:
            data = self._read_all()
            acknowledged_at = datetime.now(timezone.utc).isoformat()
            data[user_id] = acknowledged_at
            self._write_all(data)
            return acknowledged_at

    def get_acknowledged_at(self, user_id: str) -> str | None:
        with self._lock:
            data = self._read_all()
            return data.get(user_id)


_store: OnboardingStore | None = None


def get_onboarding_store() -> OnboardingStore:
    global _store
    if _store is None:
        _store = JSONFileOnboardingStore(Path(settings.DATA_DIR) / "onboarding_ack.json")
    return _store
