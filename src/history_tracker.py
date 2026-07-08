"""
history_tracker.py — Records daily snapshots of the screener's output.

This is the FOUNDATION for a genuine, unbiased backtest. A backtest needs
to know what the model ranked highly at some point in the PAST, then check
REAL performance from that point forward. Since this project is only
starting to track history now, there's no past history to check yet — but
every day this runs, it builds one more data point. In a few weeks,
backtest.py's validate_forward_performance() will have real, unbiased
data to check picks against.

Idempotent: running this multiple times on the same day updates that
day's snapshot rather than creating duplicate rows.

Usage:
    python -m src.history_tracker
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _upsert_daily_snapshot(new_rows: pd.DataFrame, history_path: str, date_str: str) -> pd.DataFrame:
    """Append new_rows to the history file at history_path, replacing any
    existing rows for date_str (so re-running on the same day updates
    that day's snapshot rather than duplicating it)."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if history_path.exists():
        existing = pd.read_csv(history_path)
        existing = existing[existing["snapshot_date"] != date_str]
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined.to_csv(history_path, index=False)
    return combined


def record_shortlist_snapshot(
    scored_path: str = "data/processed/scored_universe.csv",
    history_path: str = "data/history/shortlist_history.csv",
    top_n: int = 10,
) -> pd.DataFrame:
    df = pd.read_csv(scored_path)
    if df.empty:
        raise ValueError(f"{scored_path} is empty — run the scoring step first.")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cols = [c for c in ["ticker", "sector", "rank", "sector_rank", "composite_score", "price", "trailing_pe"] if c in df.columns]
    snapshot = df.sort_values("rank").head(top_n)[cols].copy()
    snapshot.insert(0, "snapshot_date", date_str)

    combined = _upsert_daily_snapshot(snapshot, history_path, date_str)
    logger.info(f"Recorded shortlist snapshot for {date_str} ({len(snapshot)} companies) -> {history_path} "
                f"({combined['snapshot_date'].nunique()} distinct days tracked so far)")
    return snapshot


def record_acquisition_likelihood_snapshot(
    likelihood_path: str = "data/processed/acquisition_likelihood.csv",
    history_path: str = "data/history/acquisition_likelihood_history.csv",
    top_n: int = 10,
) -> pd.DataFrame:
    df = pd.read_csv(likelihood_path)
    if df.empty:
        raise ValueError(f"{likelihood_path} is empty — run the acquisition_likelihood step first.")

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cols = [c for c in [
        "ticker", "sector", "acquisition_likelihood_rank", "acquisition_likelihood_score",
        "composite_score", "rank",
    ] if c in df.columns]
    snapshot = df.sort_values("acquisition_likelihood_rank").head(top_n)[cols].copy()
    snapshot.insert(0, "snapshot_date", date_str)

    combined = _upsert_daily_snapshot(snapshot, history_path, date_str)
    logger.info(f"Recorded acquisition-likelihood snapshot for {date_str} ({len(snapshot)} companies) -> {history_path} "
                f"({combined['snapshot_date'].nunique()} distinct days tracked so far)")
    return snapshot


def load_history(history_path: str = "data/history/shortlist_history.csv") -> pd.DataFrame:
    if not Path(history_path).exists():
        return pd.DataFrame()
    return pd.read_csv(history_path)


if __name__ == "__main__":
    shortlist_snap = record_shortlist_snapshot()
    likelihood_snap = None
    try:
        likelihood_snap = record_acquisition_likelihood_snapshot()
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Skipped acquisition-likelihood snapshot: {e}")

    print(f"\nToday's shortlist snapshot ({len(shortlist_snap)} companies):")
    print(shortlist_snap.to_string(index=False))
    if likelihood_snap is not None:
        print(f"\nToday's acquisition-likelihood snapshot ({len(likelihood_snap)} companies):")
        print(likelihood_snap.to_string(index=False))
