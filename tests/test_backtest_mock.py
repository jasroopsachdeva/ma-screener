"""
Smoke tests for backtest.py. Key things verified:
1. fetch_price_return's math is hand-checkable
2. compute_trailing_performance's group averages are hand-checkable, and
   it always surfaces its own bias caveat
3. validate_forward_performance CORRECTLY refuses to fabricate a result
   when there isn't enough real history yet
Run with: python3 tests/test_backtest_mock.py
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import fetch_price_return, compute_trailing_performance, validate_forward_performance


def test_fetch_price_return_matches_hand_calc():
    mock_hist = pd.DataFrame({"Close": [100.0, 110.0, 120.0]})
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_hist

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = fetch_price_return("TEST.NS", lookback_days=180)

    assert abs(result - 0.20) < 1e-9, f"expected 20% return ((120-100)/100), got {result}"
    print(f"PASS: fetch_price_return matches hand calculation (20.0% return)")


def test_fetch_price_return_handles_empty_history_gracefully():
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = fetch_price_return("DELISTED.NS", lookback_days=180)

    assert result is None, "empty price history should return None, not raise or return a bogus number"
    print("PASS: empty price history is handled gracefully (returns None, doesn't crash)")


def test_fetch_price_return_handles_exception_gracefully():
    with patch("yfinance.Ticker", side_effect=Exception("network error")):
        result = fetch_price_return("BROKEN.NS", lookback_days=180)
    assert result is None, "a fetch exception should return None, not propagate and crash the caller"
    print("PASS: a fetch exception is caught and returns None rather than crashing")


def test_compute_trailing_performance_group_averages_match_hand_calc():
    """3 companies, controlled fake returns via patched fetch_price_return:
    A.NS=+10%, B.NS=+20%, C.NS=-5%. Composite top-2 = [A,B] avg=+15%.
    Naive top-2 (by lowest P/E) = [C,A] avg=+2.5%. Universe avg of all 3
    = (10+20-5)/3 = +8.333%."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "rank": 1, "trailing_pe": 20.0},
        {"ticker": "B.NS", "rank": 2, "trailing_pe": 30.0},
        {"ticker": "C.NS", "rank": 3, "trailing_pe": 10.0},
    ])
    fake_returns = {"A.NS": 0.10, "B.NS": 0.20, "C.NS": -0.05}

    with patch("src.backtest.fetch_price_return", side_effect=lambda t, **kw: fake_returns[t]):
        result = compute_trailing_performance(df, lookback_days=180, top_n=2)

    assert abs(result["composite_top_n_avg_return"] - 0.15) < 1e-9, (
        f"expected composite top-2 avg of 15% ((10+20)/2), got {result['composite_top_n_avg_return']}"
    )
    assert abs(result["naive_top_n_avg_return"] - 0.025) < 1e-9, (
        f"expected naive top-2 avg of 2.5% ((-5+10)/2), got {result['naive_top_n_avg_return']}"
    )
    assert abs(result["universe_avg_return"] - (0.25 / 3)) < 1e-9
    assert "NOT A BACKTEST" in result["caveat"], "the bias caveat must always be present in the output"
    print(f"PASS: trailing performance group averages match hand calculation, caveat is present")


def test_validate_forward_performance_reports_insufficient_history_when_no_file():
    with tempfile.TemporaryDirectory() as tmp:
        result = validate_forward_performance(history_path=f"{tmp}/nonexistent.csv", min_days_elapsed=30)
    assert result["status"] == "insufficient_history"
    print("PASS: correctly reports insufficient history when no history file exists")


def test_validate_forward_performance_reports_insufficient_history_when_too_recent():
    with tempfile.TemporaryDirectory() as tmp:
        history_path = f"{tmp}/history.csv"
        recent_date = (pd.Timestamp.now() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
        pd.DataFrame([{"snapshot_date": recent_date, "ticker": "A.NS", "rank": 1}]).to_csv(history_path, index=False)

        result = validate_forward_performance(history_path=history_path, min_days_elapsed=30)

    assert result["status"] == "insufficient_history", (
        "a 5-day-old snapshot must NOT be treated as sufficient for a 30-day forward check"
    )
    print("PASS: correctly reports insufficient history when the oldest snapshot isn't old enough yet")


def test_validate_forward_performance_computes_real_forward_return_when_history_is_old_enough():
    with tempfile.TemporaryDirectory() as tmp:
        history_path = f"{tmp}/history.csv"
        old_date = (pd.Timestamp.now() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        pd.DataFrame([
            {"snapshot_date": old_date, "ticker": "A.NS", "rank": 1},
            {"snapshot_date": old_date, "ticker": "B.NS", "rank": 2},
        ]).to_csv(history_path, index=False)

        with patch("src.backtest.fetch_price_return", side_effect=lambda t, **kw: {"A.NS": 0.30, "B.NS": 0.10}[t]):
            result = validate_forward_performance(history_path=history_path, min_days_elapsed=30)

    assert result["status"] == "ok"
    assert result["as_of_date"] == old_date
    assert abs(result["avg_forward_return"] - 0.20) < 1e-9, f"expected avg of 20% ((30+10)/2), got {result['avg_forward_return']}"
    assert "no look-ahead bias" in result["note"]
    print(f"PASS: forward performance correctly computed once history is old enough "
          f"(as_of={result['as_of_date']}, avg_return={result['avg_forward_return']:.1%})")


if __name__ == "__main__":
    test_fetch_price_return_matches_hand_calc()
    test_fetch_price_return_handles_empty_history_gracefully()
    test_fetch_price_return_handles_exception_gracefully()
    test_compute_trailing_performance_group_averages_match_hand_calc()
    test_validate_forward_performance_reports_insufficient_history_when_no_file()
    test_validate_forward_performance_reports_insufficient_history_when_too_recent()
    test_validate_forward_performance_computes_real_forward_return_when_history_is_old_enough()
    print("\nAll backtest mock tests passed.")
