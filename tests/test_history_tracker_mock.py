"""
Smoke tests for history_tracker.py. The most important behavior to verify
is idempotency: running the tracker twice on the same day should UPDATE
that day's snapshot, not create duplicate rows.
Run with: python3 tests/test_history_tracker_mock.py
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.history_tracker import record_shortlist_snapshot, _upsert_daily_snapshot


def _make_scored_df():
    return pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "rank": 1, "sector_rank": 1, "composite_score": 0.8, "price": 100.0, "trailing_pe": 15.0},
        {"ticker": "B.NS", "sector": "y", "rank": 2, "sector_rank": 1, "composite_score": 0.6, "price": 50.0, "trailing_pe": 20.0},
    ])


def test_first_snapshot_creates_history_file():
    with tempfile.TemporaryDirectory() as tmp:
        scored_path = f"{tmp}/scored.csv"
        history_path = f"{tmp}/history.csv"
        _make_scored_df().to_csv(scored_path, index=False)

        with patch("src.history_tracker.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-04"
            record_shortlist_snapshot(scored_path=scored_path, history_path=history_path, top_n=10)

        assert Path(history_path).exists()
        history = pd.read_csv(history_path)
        assert len(history) == 2
        assert set(history["ticker"]) == {"A.NS", "B.NS"}
        assert (history["snapshot_date"] == "2026-07-04").all()
        print("PASS: first snapshot creates the history file with correct rows")


def test_rerunning_same_day_updates_not_duplicates():
    """The core idempotency guarantee: running the tracker twice on the
    same day should leave exactly one snapshot for that day, with the
    latest data — not two duplicated sets of rows."""
    with tempfile.TemporaryDirectory() as tmp:
        scored_path = f"{tmp}/scored.csv"
        history_path = f"{tmp}/history.csv"

        df1 = _make_scored_df()
        df1.to_csv(scored_path, index=False)
        with patch("src.history_tracker.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-04"
            record_shortlist_snapshot(scored_path=scored_path, history_path=history_path, top_n=10)

        df2 = _make_scored_df()
        df2.loc[df2.ticker == "A.NS", "composite_score"] = 0.95
        df2.to_csv(scored_path, index=False)
        with patch("src.history_tracker.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-04"
            record_shortlist_snapshot(scored_path=scored_path, history_path=history_path, top_n=10)

        history = pd.read_csv(history_path)
        assert len(history) == 2, f"expected exactly 2 rows (not duplicated), got {len(history)}"
        a_row = history[history.ticker == "A.NS"].iloc[0]
        assert abs(a_row["composite_score"] - 0.95) < 1e-9, "should reflect the UPDATED score, not the stale first-run value"
        print("PASS: re-running on the same day updates that day's snapshot, doesn't duplicate rows")


def test_multiple_days_accumulate_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        scored_path = f"{tmp}/scored.csv"
        history_path = f"{tmp}/history.csv"
        _make_scored_df().to_csv(scored_path, index=False)

        with patch("src.history_tracker.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-01"
            record_shortlist_snapshot(scored_path=scored_path, history_path=history_path, top_n=10)

        with patch("src.history_tracker.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-02"
            record_shortlist_snapshot(scored_path=scored_path, history_path=history_path, top_n=10)

        history = pd.read_csv(history_path)
        assert history["snapshot_date"].nunique() == 2
        assert len(history) == 4
        print("PASS: snapshots from different days both accumulate correctly (2 days x 2 companies = 4 rows)")


def test_top_n_limits_snapshot_size():
    with tempfile.TemporaryDirectory() as tmp:
        scored_path = f"{tmp}/scored.csv"
        history_path = f"{tmp}/history.csv"
        df = pd.DataFrame([
            {"ticker": f"T{i}.NS", "sector": "x", "rank": i + 1, "sector_rank": 1, "composite_score": 1.0 - i * 0.1, "price": 100.0, "trailing_pe": 15.0}
            for i in range(5)
        ])
        df.to_csv(scored_path, index=False)

        with patch("src.history_tracker.datetime") as mock_dt:
            mock_dt.now.return_value.strftime.return_value = "2026-07-04"
            snapshot = record_shortlist_snapshot(scored_path=scored_path, history_path=history_path, top_n=3)

        assert len(snapshot) == 3
        assert list(snapshot["ticker"]) == ["T0.NS", "T1.NS", "T2.NS"], "should keep the top 3 by rank"
        print("PASS: top_n correctly limits the snapshot to the requested number of companies")


if __name__ == "__main__":
    test_first_snapshot_creates_history_file()
    test_rerunning_same_day_updates_not_duplicates()
    test_multiple_days_accumulate_correctly()
    test_top_n_limits_snapshot_size()
    print("\nAll history tracker mock tests passed.")
