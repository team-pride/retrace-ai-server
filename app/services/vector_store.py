"""얼굴 기준 벡터 저장소.

지금은 로컬 JSON 파일 기반 구현체만 제공한다.
나중에 실제 DB/Spring 서버와 연동할 때는 VectorStore 인터페이스를 구현하는
다른 클래스로 교체하면 되고, get_vector_store()의 반환값만 바꿔주면 된다.
"""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import settings


class VectorStore(ABC):
    @abstractmethod
    def save(self, user_id: str, vector: list[float]) -> None: ...

    @abstractmethod
    def get(self, user_id: str) -> list[float] | None: ...

    @abstractmethod
    def delete(self, user_id: str) -> None: ...


class JSONFileVectorStore(VectorStore):
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

    def save(self, user_id: str, vector: list[float]) -> None:
        with self._lock:
            data = self._read_all()
            data[user_id] = vector
            self._write_all(data)

    def get(self, user_id: str) -> list[float] | None:
        with self._lock:
            data = self._read_all()
            return data.get(user_id)

    def delete(self, user_id: str) -> None:
        with self._lock:
            data = self._read_all()
            data.pop(user_id, None)
            self._write_all(data)


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        _store = JSONFileVectorStore(Path(settings.DATA_DIR) / "face_vectors.json")
    return _store
