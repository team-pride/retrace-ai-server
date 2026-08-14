"""관리 마커 저장소 (우선순위 5).

기능명세서 4.1 "변화 시점 질문과 관리 마커 등록"의 데이터 규칙을 따른다.
- "데이터: 마커는 종류, 날짜, 원문 문장, 등록 경로로 저장한다."
- "비즈니스 규칙: 관리 마커는 자유 입력만 받고 종류로 분류하지 않는다."
  -> 그래서 여기서는 '종류'를 별도 enum으로 나누지 않고, 사용자가 적은
     원문 문장(note)과 날짜만 받는다. 문장에서 종류/날짜를 자동으로 뽑아내는
     자연어 처리(4.1의 "자유 문장을 분석해 관리 종류와 날짜를 추출")는 이
     AI 서버의 범위 밖이라 판단해 생략했다 — 프론트/백엔드에서 이미 날짜를
     확정해 넘겨준다고 가정한다.

우선순위 5(관리 효과 판정, effect_service.py)는 이 저장소의 marker_date를
기준으로 지표 시계열을 이전/이후로 나눠 예측선-실제 곡선을 비교한다.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from app.core.config import settings


class MarkerRecord(TypedDict):
    marker_id: str
    marker_date: str  # ISO 날짜 (YYYY-MM-DD). 이 날짜를 기준으로 이전/이후 구간을 나눈다.
    note: str  # 사용자가 입력한 원문 문장 (예: "레이저 시술 받음", "스킨케어 시작")
    created_at: str  # ISO datetime (등록 시각)


class MarkerStore:
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

    def add_marker(self, user_id: str, marker_date: str, note: str) -> MarkerRecord:
        record: MarkerRecord = {
            "marker_id": uuid.uuid4().hex[:12],
            "marker_date": marker_date,
            "note": note,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            data = self._read_all()
            markers = data.setdefault(user_id, [])
            markers.append(dict(record))
            self._write_all(data)
        return record

    def get_markers(self, user_id: str) -> list[MarkerRecord]:
        with self._lock:
            data = self._read_all()
            return list(data.get(user_id, []))

    def get_marker(self, user_id: str, marker_id: str) -> MarkerRecord | None:
        for m in self.get_markers(user_id):
            if m["marker_id"] == marker_id:
                return m
        return None


_store: MarkerStore | None = None


def get_marker_store() -> MarkerStore:
    global _store
    if _store is None:
        _store = MarkerStore(Path(settings.DATA_DIR) / "care_markers.json")
    return _store
