"""
Data-location rules that only break on Render: snapshots are ordered by run
number (as text, run-100 sorts before run-99), nothing may read
data/processed relative to the CWD (it is gitignored, so absent there), and
freshness is judged by the data's own fetched_at, never by file mtimes.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import api
from src import validate_returns as vr


def _snap(root, run, body="ticker\n"):
    d = Path(root) / "snapshots" / f"run-{run}" / f"screener-results-{run}" / "data" / "processed"
    d.mkdir(parents=True)
    (d / "scored_universe.csv").write_text(body)
    return d


def test_api_serves_highest_run_number():
    with tempfile.TemporaryDirectory() as tmp:
        for run in ("09", "99"):
            _snap(tmp, run)
        newest = _snap(tmp, "100")
        with patch.object(api, "ROOT", Path(tmp)):
            assert api._resolve_data_dir() == newest
    print("PASS: API serves run-100 over run-99 (numeric, not string order)")


def test_api_without_any_data_falls_back_to_processed():
    with tempfile.TemporaryDirectory() as tmp:
        with patch.object(api, "ROOT", Path(tmp)):
            assert api._resolve_data_dir() == Path(tmp) / "data" / "processed"
    print("PASS: no data at all -> data/processed, so endpoints 503 instead of the import crashing")


def test_api_serves_newer_of_processed_and_snapshot():
    # A stale local data/processed (Aug 5) used to mask a fresher snapshot (Oct 9).
    def scored(day):
        return f"ticker,fetched_at\nX.NS,2026-{day}T05:00:00+00:00\n"

    with tempfile.TemporaryDirectory() as tmp:
        snap = _snap(tmp, "96", scored("10-09"))
        processed = Path(tmp) / "data" / "processed"
        processed.mkdir(parents=True)
        with patch.object(api, "ROOT", Path(tmp)):
            (processed / "scored_universe.csv").write_text(scored("08-05"))
            assert api._resolve_data_dir() == snap, "stale local data must not mask a newer snapshot"
            (processed / "scored_universe.csv").write_text(scored("10-10"))
            assert api._resolve_data_dir() == processed, "a fresh local pipeline run should win"
        with patch.object(api, "DATA_DIR", str(processed)):
            assert api.get_hero_stats()["updated"] == "Oct 10, 05:00 UTC", "updated must come from fetched_at"
    print("PASS: newer of data/processed and the latest snapshot is served; 'updated' comes from fetched_at")


def test_backtest_keeps_earliest_run_of_a_day():
    # Same-day re-run: run-99 then run-100. The dedup keeps the first one read.
    with tempfile.TemporaryDirectory() as tmp:
        for run, rank in (("99", 1), ("100", 2)):
            _snap(tmp, run, f"fetched_at,rank,ticker\n2026-10-12T05:00:00Z,{rank},ABC.NS\n")
        snap_dir = os.path.join(tmp, "snapshots")
        with patch.object(vr, "SNAPSHOT_DIR", snap_dir), \
                patch.object(vr, "SNAPSHOT_GLOB", os.path.join(snap_dir, "**", "scored_universe.csv")):
            assert vr.load_snapshots()["rank"].tolist() == [1]
    print("PASS: backtest dedup keeps run-99 over run-100 on the same day")


def test_summary_pdf_reads_data_dir_not_cwd():
    ticker = api._load_scored()["ticker"].iloc[0]
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # like Render: no data/processed under the CWD
        try:
            assert api.get_summary_pdf(ticker).media_type == "application/pdf"
        finally:
            os.chdir(cwd)
    print("PASS: summary PDF reads DATA_DIR, so it works with no data/processed under the CWD")


if __name__ == "__main__":
    test_api_serves_highest_run_number()
    test_api_without_any_data_falls_back_to_processed()
    test_api_serves_newer_of_processed_and_snapshot()
    test_backtest_keeps_earliest_run_of_a_day()
    test_summary_pdf_reads_data_dir_not_cwd()
    print("\nAll snapshot-order / data-path tests passed.")
