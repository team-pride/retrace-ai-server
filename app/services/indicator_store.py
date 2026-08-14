"""사용자별 지표 기록 저장소 (우선순위 4).

vector_store.py / retry_tracker.py와 같은 패턴으로, 지금은 로컬 JSON 파일에
저장한다. 곡선(curve_service.py)은 이 저장소에서 사용자의 전체 지표 기록을
읽어와 시계열로 재구성한다.
"""
from __future__ import annotations

import json
import threading
from datetime import date
from pathlib import Path
from typing import TypedDict

from app.core.config import settings


class IndicatorRecord(TypedDict):
    photo_key: str
    captured_at: str  # ISO 날짜 문자열 (YYYY-MM-DD)
    face_width_ratio: float
    jaw_angle_deg: float
    eyelid_height_ratio: float
    mouth_corner_angle_deg: float
    ipd_px: float


class IndicatorStore:
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

    def add_record(self, user_id: str, record: IndicatorRecord) -> None:
        """같은 photo_key로 이미 저장된 기록이 있으면 덮어쓴다 (재추출 시 중복 방지)."""
        with self._lock:
            data = self._read_all()
            records = data.setdefault(user_id, [])
            records[:] = [r for r in records if r["photo_key"] != record["photo_key"]]
            records.append(dict(record))
            self._write_all(data)

    def get_records(self, user_id: str) -> list[IndicatorRecord]:
        with self._lock:
            data = self._read_all()
            return list(data.get(user_id, []))


_store: IndicatorStore | None = None


def get_indicator_store() -> IndicatorStore:
    global _store
    if _store is None:
        _store = IndicatorStore(Path(settings.DATA_DIR) / "face_indicators.json")
    return _store


def parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)
