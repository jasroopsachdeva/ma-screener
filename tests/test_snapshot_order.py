"""Newest/oldest snapshot must be decided by run number, not string order:
as text, run-100 sorts before run-99."""

from src import api
from src import validate_returns as vr


def _snap(root, run, body="ticker\n"):
    d = root / "snapshots" / f"run-{run}" / f"screener-results-{run}" / "data" / "processed"
    d.mkdir(parents=True)
    (d / "scored_universe.csv").write_text(body)
    return d


def test_api_serves_highest_run_number(tmp_path, monkeypatch):
    for run in ("09", "99"):
        _snap(tmp_path, run)
    newest = _snap(tmp_path, "100")
    monkeypatch.setattr(api, "ROOT", tmp_path)
    assert api._resolve_data_dir() == newest


def test_api_without_any_data_falls_back_to_processed(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    assert api._resolve_data_dir() == tmp_path / "data" / "processed"


def test_backtest_keeps_earliest_run_of_a_day(tmp_path, monkeypatch):
    # Same-day re-run: run-99 then run-100. The dedup keeps the first one read.
    for run, rank in (("99", 1), ("100", 2)):
        _snap(tmp_path, run, f"fetched_at,rank,ticker\n2026-10-12T05:00:00Z,{rank},ABC.NS\n")
    monkeypatch.setattr(vr, "SNAPSHOT_DIR", str(tmp_path / "snapshots"))
    monkeypatch.setattr(vr, "SNAPSHOT_GLOB", str(tmp_path / "snapshots" / "**" / "scored_universe.csv"))
    assert vr.load_snapshots()["rank"].tolist() == [1]
