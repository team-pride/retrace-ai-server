"""사진 판정 재시도 카운터 (우선순위 2).

동일 photo_key(예: "{user_id}_{month}_{shot_type}")가 판정에 실패(exclude)한
횟수를 누적해서, PHOTO_MAX_RETRY 회를 넘기면 해당 사진을 최종 제외 처리한다.

지금은 로컬 JSON 파일 기반이다. vector_store.py와 마찬가지로 추후 DB/Redis로
교체할 수 있게 별도 모듈로 분리해뒀다.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from app.core.config import settings


class RetryTracker:
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

    def record_failure(self, photo_key: str) -> int:
        """실패를 기록하고 누적 실패 횟수를 반환한다."""
        with self._lock:
            data = self._read_all()
            count = data.get(photo_key, 0) + 1
            data[photo_key] = count
            self._write_all(data)
            return count

    def get_count(self, photo_key: str) -> int:
        with self._lock:
            data = self._read_all()
            return data.get(photo_key, 0)

    def reset(self, photo_key: str) -> None:
        with self._lock:
            data = self._read_all()
            if photo_key in data:
                del data[photo_key]
                self._write_all(data)


_tracker: RetryTracker | None = None


def get_retry_tracker() -> RetryTracker:
    global _tracker
    if _tracker is None:
        _tracker = RetryTracker(Path(settings.DATA_DIR) / "photo_retry_counts.json")
    return _tracker
