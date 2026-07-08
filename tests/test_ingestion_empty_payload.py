"""
Regression test for the real LTIM.NS bug: yfinance's t.info call returned
successfully (no exception) but with every meaningful field null. The old
code treated this as fetch_ok=True with 12 null fields — a false positive.
Fixed code should treat an empty payload as a failure and retry, then
correctly mark fetch_ok=False if it never gets real data.
Run with: python3 tests/test_ingestion_empty_payload.py
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import fetch_ticker


def test_empty_info_payload_is_not_a_false_success():
    """Reproduces the exact LTIM.NS scenario: no exception thrown, but
    info dict has no marketCap and no price."""
    mock_ticker = MagicMock()
    mock_ticker.info = {}  # what yfinance actually returned for LTIM.NS

    with patch("src.ingestion.yf.Ticker", return_value=mock_ticker):
        snap = fetch_ticker("LTIM.NS", retries=1, backoff_seconds=0)

    assert snap.fetch_ok is False, (
        "an empty info payload must NOT be recorded as a successful fetch "
        "(this was the real LTIM.NS bug — fetch_ok=True with all fields null)"
    )
    assert snap.market_cap is None
    assert "empty info payload" in snap.error
    print("PASS: empty yfinance payload is now correctly treated as a failed fetch, not a false success")


def test_partial_data_still_succeeds():
    """Sanity check the fix isn't too aggressive: if price exists but some
    other fields are missing, that's still a real success, not a failure."""
    mock_ticker = MagicMock()
    mock_ticker.info = {"currentPrice": 1500.0}  # price present, rest missing
    mock_ticker.balance_sheet = MagicMock(empty=True)
    mock_ticker.financials = MagicMock(empty=True)

    with patch("src.ingestion.yf.Ticker", return_value=mock_ticker):
        snap = fetch_ticker("PARTIAL.NS", retries=1, backoff_seconds=0)

    assert snap.fetch_ok is True, "a real price value should still count as a successful fetch"
    assert snap.price == 1500.0
    assert snap.market_cap is None, "missing individual fields should stay null, not block the whole fetch"
    print("PASS: partial-but-real data (price present) is still correctly treated as a success")


if __name__ == "__main__":
    test_empty_info_payload_is_not_a_false_success()
    test_partial_data_still_succeeds()
    print("\nAll ingestion empty-payload tests passed.")
