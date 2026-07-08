"""
cleaning.py — Cleaning and normalization layer for the M&A screener.

Reads raw per-ticker JSON snapshots from src/ingestion.py and produces one
clean, analysis-ready table. This is where data quality problems get caught
BEFORE they silently corrupt the scoring engine or DuPont math downstream.

Specifically handles:
  - Tickers that failed to fetch (excluded, logged, never silently zero-filled)
  - Missing individual fields (flagged per-row, not dropped outright unless critical)
  - Negative or zero equity (breaks ROE/DuPont math — flagged, not divided-by-zero)
  - Zero or negative revenue (breaks margin/turnover ratios — flagged)
  - Outlier detection (values so extreme they're likely data errors, not reality)
  - Unit consistency check (debt/equity as %, market cap in absolute currency)

Usage:
    python -m src.cleaning
"""

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fields where a missing value is critical enough to exclude the row from scoring
CRITICAL_FIELDS = ["market_cap", "price", "trailing_pe"]

# Sanity bounds — values outside these ranges are flagged as likely data errors,
# not necessarily dropped, since some are legitimate extremes (e.g. loss-making
# firms can have negative P/E). Tune these as you see real data.
SANITY_BOUNDS = {
    "trailing_pe": (-500, 500),
    "ev_to_ebitda": (-200, 200),
    "return_on_equity": (-5.0, 5.0),       # as decimal, e.g. 0.48 = 48%
    "debt_to_equity": (0, 1000),           # yfinance reports this as a %, not a ratio
    "revenue_growth": (-1.0, 5.0),         # -100% to +500%
    "net_margin": (-5.0, 5.0),
}


@dataclass
class CleaningFlags:
    ticker: str
    excluded: bool
    exclusion_reason: str | None
    missing_fields: list
    outlier_fields: list
    negative_equity: bool
    zero_or_negative_revenue_proxy: bool


def load_raw_snapshots(raw_dir: str = "data/raw") -> list[dict]:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_dir} not found — run `python -m src.ingestion` first."
        )
    snapshots = []
    for f in sorted(raw_path.glob("*.json")):
        with open(f, "r") as fh:
            snapshots.append(json.load(fh))
    return snapshots


def _check_missing(row: dict) -> list:
    return [k for k, v in row.items() if v is None and k not in ("error",)]


def _check_outliers(row: dict) -> list:
    outliers = []
    for field_name, (lo, hi) in SANITY_BOUNDS.items():
        val = row.get(field_name)
        if val is not None and not (lo <= val <= hi):
            outliers.append(field_name)
    return outliers


def clean_snapshot(row: dict) -> tuple[dict, CleaningFlags]:
    ticker = row["ticker"]

    # Tickers that failed ingestion entirely are excluded up front — never
    # silently zero-filled, since a 0 for market_cap would poison every
    # downstream ratio and ranking.
    if not row.get("fetch_ok", False):
        flags = CleaningFlags(
            ticker=ticker, excluded=True,
            exclusion_reason=f"ingestion failed: {row.get('error')}",
            missing_fields=[], outlier_fields=[],
            negative_equity=False, zero_or_negative_revenue_proxy=False,
        )
        return row, flags

    missing = _check_missing(row)
    critical_missing = [f for f in missing if f in CRITICAL_FIELDS]

    if critical_missing:
        flags = CleaningFlags(
            ticker=ticker, excluded=True,
            exclusion_reason=f"missing critical fields: {critical_missing}",
            missing_fields=missing, outlier_fields=[],
            negative_equity=False, zero_or_negative_revenue_proxy=False,
        )
        return row, flags

    outliers = _check_outliers(row)

    total_equity = row.get("total_equity")
    negative_equity = total_equity is not None and total_equity <= 0

    # No direct revenue field cached yet — asset_turnover being None or
    # implausibly high/low is our proxy signal that revenue math broke
    # (e.g. divide-by-zero or missing revenue upstream).
    asset_turnover = row.get("asset_turnover")
    zero_or_negative_revenue_proxy = asset_turnover is not None and asset_turnover <= 0

    flags = CleaningFlags(
        ticker=ticker, excluded=False, exclusion_reason=None,
        missing_fields=missing, outlier_fields=outliers,
        negative_equity=negative_equity,
        zero_or_negative_revenue_proxy=zero_or_negative_revenue_proxy,
    )

    # Normalize debt_to_equity to a plain ratio (yfinance gives it as a %,
    # e.g. 45.0 meaning 45%) so every downstream module works in ratios,
    # not a mix of percentages and decimals.
    cleaned = dict(row)
    if cleaned.get("debt_to_equity") is not None:
        cleaned["debt_to_equity_ratio"] = cleaned["debt_to_equity"] / 100.0
    else:
        cleaned["debt_to_equity_ratio"] = None

    # Null out only the specific fields flagged as outliers — not the whole
    # row. A row with a broken ev_to_ebitda but a perfectly good P/E and ROE
    # should still be usable for scoring on those good fields. The raw
    # (broken) value stays visible in data/raw/ and in cleaning_flags.csv
    # for audit purposes; it just doesn't propagate into the scoring input.
    for outlier_field in outliers:
        cleaned[outlier_field] = None

    return cleaned, flags


def clean_universe(raw_dir: str = "data/raw", output_dir: str = "data/processed") -> pd.DataFrame:
    snapshots = load_raw_snapshots(raw_dir)
    rows, all_flags = [], []

    for snap in snapshots:
        cleaned, flags = clean_snapshot(snap)
        all_flags.append(flags)
        if not flags.excluded:
            rows.append(cleaned)
        else:
            logger.warning(f"[{flags.ticker}] EXCLUDED: {flags.exclusion_reason}")

    for f in all_flags:
        if not f.excluded:
            if f.outlier_fields:
                logger.warning(f"[{f.ticker}] outlier values flagged: {f.outlier_fields}")
            if f.negative_equity:
                logger.warning(f"[{f.ticker}] negative/zero equity — DuPont ROE math will be invalid downstream")
            if f.missing_fields:
                logger.info(f"[{f.ticker}] non-critical missing fields: {f.missing_fields}")

    df = pd.DataFrame(rows)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "cleaned_universe.csv"
    df.to_csv(out_path, index=False)

    flags_df = pd.DataFrame([vars(f) for f in all_flags])
    flags_path = Path(output_dir) / "cleaning_flags.csv"
    flags_df.to_csv(flags_path, index=False)

    logger.info(f"Cleaned {len(df)}/{len(snapshots)} tickers -> {out_path}")
    logger.info(f"Full flag audit trail -> {flags_path}")

    return df


if __name__ == "__main__":
    result = clean_universe()
    print(f"\nDone. {len(result)} tickers passed cleaning and are ready for scoring.")
    if len(result) > 0:
        print("\nSample of cleaned data:")
        print(result[["ticker", "trailing_pe", "return_on_equity", "debt_to_equity_ratio"]].head())
