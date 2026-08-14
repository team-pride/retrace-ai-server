from pathlib import Path

from app.services.retry_tracker import RetryTracker


def test_record_failure_increments(tmp_path: Path):
    tracker = RetryTracker(tmp_path / "retry.json")
    assert tracker.get_count("k1") == 0
    assert tracker.record_failure("k1") == 1
    assert tracker.record_failure("k1") == 2
    assert tracker.get_count("k1") == 2


def test_reset_clears_count(tmp_path: Path):
    tracker = RetryTracker(tmp_path / "retry.json")
    tracker.record_failure("k1")
    tracker.record_failure("k1")
    tracker.reset("k1")
    assert tracker.get_count("k1") == 0


def test_keys_are_independent(tmp_path: Path):
    tracker = RetryTracker(tmp_path / "retry.json")
    tracker.record_failure("k1")
    tracker.record_failure("k2")
    tracker.record_failure("k2")
    assert tracker.get_count("k1") == 1
    assert tracker.get_count("k2") == 2
