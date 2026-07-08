"""
Smoke test for cleaning.py, covering the edge cases that matter most:
negative equity, missing critical fields, outlier values, and failed-fetch
exclusion. Uses your real ACC.NS data as the "known good" baseline.
Run with: python3 tests/test_cleaning_mock.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cleaning import clean_snapshot

REAL_ACC_DATA = {
    "ticker": "ACC.NS", "fetched_at": "2026-07-01T13:14:49.457877",
    "market_cap": 250170179584, "price": 1332.2, "trailing_pe": 11.739512,
    "ev_to_ebitda": 9.087, "return_on_equity": 0.109280005, "debt_to_equity": 2.085,
    "revenue_growth": 0.323, "net_margin": 0.082959995,
    "asset_turnover": 1.2187217464845206, "total_debt": 4286099968,
    "total_equity": 205505400000.0, "fetch_ok": True, "error": None,
}


def test_known_good_data_passes_clean():
    cleaned, flags = clean_snapshot(REAL_ACC_DATA)
    assert flags.excluded is False
    assert flags.negative_equity is False
    assert flags.outlier_fields == []
    assert cleaned["debt_to_equity_ratio"] == 0.02085  # 2.085% -> 0.02085 ratio
    print("PASS: real ACC data cleans through with no flags, D/E normalized correctly")


def test_failed_fetch_excluded():
    row = {"ticker": "BAD.NS", "fetch_ok": False, "error": "network timeout"}
    cleaned, flags = clean_snapshot(row)
    assert flags.excluded is True
    assert "network timeout" in flags.exclusion_reason
    print("PASS: failed-fetch ticker is excluded with reason logged, not zero-filled")


def test_negative_equity_flagged_not_crashed():
    row = dict(REAL_ACC_DATA)
    row["ticker"] = "NEGEQ.NS"
    row["total_equity"] = -50000000.0  # e.g. a company that's wiped out its equity base
    cleaned, flags = clean_snapshot(row)
    assert flags.excluded is False, "negative equity should be flagged, not silently excluded"
    assert flags.negative_equity is True
    print("PASS: negative equity is flagged (would break DuPont ROE), doesn't crash the pipeline")


def test_missing_critical_field_excludes_row():
    row = dict(REAL_ACC_DATA)
    row["ticker"] = "NOPRICE.NS"
    row["market_cap"] = None
    cleaned, flags = clean_snapshot(row)
    assert flags.excluded is True
    assert "market_cap" in flags.exclusion_reason
    print("PASS: missing critical field (market_cap) correctly excludes the row")


def test_outlier_pe_flagged():
    row = dict(REAL_ACC_DATA)
    row["ticker"] = "WEIRDPE.NS"
    row["trailing_pe"] = 8000.0  # clearly a data error, not a real P/E
    cleaned, flags = clean_snapshot(row)
    assert flags.excluded is False
    assert "trailing_pe" in flags.outlier_fields
    print("PASS: absurd P/E value is flagged as outlier, not silently accepted")


def test_outlier_field_nulled_but_row_kept():
    """Regression test for the real HCLTECH.NS / INFY.NS bug: yfinance
    returned ev_to_ebitda values around 900-1000 (clearly broken) while
    trailing_pe and return_on_equity for the same tickers were normal.
    The broken field should be nulled so it can't poison scoring, but
    the row must survive since the other fields are perfectly usable."""
    row = dict(REAL_ACC_DATA)
    row["ticker"] = "HCLTECH.NS"
    row["ev_to_ebitda"] = 999.789  # actual broken value pulled from yfinance
    row["trailing_pe"] = 16.868372  # actual good value, should survive untouched
    row["return_on_equity"] = 0.23357  # actual good value, should survive untouched

    cleaned, flags = clean_snapshot(row)

    assert flags.excluded is False, "row should not be excluded over one bad field"
    assert "ev_to_ebitda" in flags.outlier_fields, "broken value should be flagged"
    assert cleaned["ev_to_ebitda"] is None, "broken value must be nulled, not passed through"
    assert cleaned["trailing_pe"] == 16.868372, "good fields must survive untouched"
    assert cleaned["return_on_equity"] == 0.23357, "good fields must survive untouched"
    print("PASS: broken ev_to_ebitda is nulled from scoring input; good fields on same row survive")


def test_zero_revenue_proxy_flagged():
    row = dict(REAL_ACC_DATA)
    row["ticker"] = "ZEROREV.NS"
    row["asset_turnover"] = 0.0
    cleaned, flags = clean_snapshot(row)
    assert flags.zero_or_negative_revenue_proxy is True
    print("PASS: zero asset turnover (revenue proxy issue) is flagged")


if __name__ == "__main__":
    test_known_good_data_passes_clean()
    test_failed_fetch_excluded()
    test_negative_equity_flagged_not_crashed()
    test_missing_critical_field_excludes_row()
    test_outlier_pe_flagged()
    test_outlier_field_nulled_but_row_kept()
    test_zero_revenue_proxy_flagged()
    print("\nAll cleaning mock tests passed.")
