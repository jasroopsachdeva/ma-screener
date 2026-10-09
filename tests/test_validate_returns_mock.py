"""
Mock tests for src/validate_returns.py: snapshots are read in run-number
order (as text, run-100 sorts before run-99), and the independent-window
summary only counts windows that start a full horizon apart.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import validate_returns as vr


def test_load_snapshots_keeps_earliest_run_of_a_day():
    # Same-day re-run: run-99 then run-100. The dedup keeps the first one read.
    with tempfile.TemporaryDirectory() as tmp:
        snap_dir = os.path.join(tmp, "snapshots")
        for run, rank in (("99", 1), ("100", 2)):
            os.makedirs(os.path.join(snap_dir, f"run-{run}"))
            with open(os.path.join(snap_dir, f"run-{run}", "scored_universe.csv"), "w") as f:
                f.write(f"fetched_at,rank,ticker\n2026-10-12T05:00:00Z,{rank},ABC.NS\n")
        with patch.object(vr, "SNAPSHOT_DIR", snap_dir), \
                patch.object(vr, "SNAPSHOT_GLOB", os.path.join(snap_dir, "**", "scored_universe.csv")):
            assert vr.load_snapshots()["rank"].tolist() == [1]
    print("PASS: backtest dedup keeps run-99 over run-100 on the same day")


def test_non_overlapping_windows_start_a_full_horizon_apart():
    # Daily as-of dates Jul 9 - Sep 10 with a 30-day horizon: 64 windows, but
    # only those starting 30+ days apart are independent observations.
    dates = list(pd.date_range("2026-07-09", "2026-09-10", freq="D"))
    keep = vr.non_overlapping(dates, 30)
    assert [str(dates[i].date()) for i in keep] == ["2026-07-09", "2026-08-08", "2026-09-07"]
    assert vr.non_overlapping([], 30) == []
    print(f"PASS: {len(dates)} daily windows reduce to {len(keep)} non-overlapping ones")


if __name__ == "__main__":
    test_load_snapshots_keeps_earliest_run_of_a_day()
    test_non_overlapping_windows_start_a_full_horizon_apart()
    print("\nAll validate_returns mock tests passed.")
