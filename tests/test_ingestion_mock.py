"""
Quick smoke test for ingestion.py logic, using mocked yfinance calls.
Validates: successful fetch parsing, retry-then-fail isolation, and caching.
Run with: python3 tests/test_ingestion_mock.py
"""

import sys
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import fetch_ticker, fetch_universe, load_config


def test_successful_fetch():
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "marketCap": 500000000000,
        "currentPrice": 2450.5,
        "trailingPE": 28.4,
        "enterpriseToEbitda": 15.2,
        "returnOnEquity": 0.18,
        "debtToEquity": 45.0,
        "revenueGrowth": 0.12,
        "profitMargins": 0.15,
        "totalDebt": 12000000000,
    }
    mock_ticker.balance_sheet = MagicMock(empty=True)
    mock_ticker.financials = MagicMock(empty=True)

    with patch("src.ingestion.yf.Ticker", return_value=mock_ticker):
        snap = fetch_ticker("TESTCO.NS")

    assert snap.fetch_ok is True, "expected successful fetch"
    assert snap.market_cap == 500000000000
    assert snap.trailing_pe == 28.4
    print("PASS: successful fetch parses correctly")


def test_config_validation_catches_duplicate_tickers():
    import tempfile, yaml
    from src.ingestion import load_config
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/bad_config.yaml"
        config = {
            "universe": [{"ticker": "ACC.NS", "sector": "cement"}, {"ticker": "ACC.NS", "sector": "cement"}],
            "data": {"cache_dir": "x", "cache_expiry_days": 1},
        }
        with open(path, "w") as f:
            yaml.dump(config, f)
        try:
            load_config(path)
            assert False, "expected a ValueError for duplicate tickers"
        except ValueError as e:
            assert "ACC.NS" in str(e) and "duplicate" in str(e).lower()
            print("PASS: config validation catches a duplicate ticker (real risk with 200 manually-curated entries)")


def test_config_validation_catches_missing_ns_suffix():
    import tempfile, yaml
    from src.ingestion import load_config
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/bad_config.yaml"
        config = {
            "universe": [{"ticker": "ACC", "sector": "cement"}],  # missing .NS
            "data": {"cache_dir": "x", "cache_expiry_days": 1},
        }
        with open(path, "w") as f:
            yaml.dump(config, f)
        try:
            load_config(path)
            assert False, "expected a ValueError for a missing .NS suffix"
        except ValueError as e:
            assert "ACC" in str(e) and ".NS" in str(e)
            print("PASS: config validation catches a ticker missing the .NS suffix")


def test_config_validation_catches_bad_weights():
    import tempfile, yaml
    from src.ingestion import load_config
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/bad_config.yaml"
        config = {
            "universe": [{"ticker": "ACC.NS", "sector": "cement"}],
            "data": {"cache_dir": "x", "cache_expiry_days": 1},
            "scoring": {"weights": {"valuation": 0.30, "quality": 0.30, "leverage": 0.20, "growth": 0.30}},  # sums to 1.10
        }
        with open(path, "w") as f:
            yaml.dump(config, f)
        try:
            load_config(path)
            assert False, "expected a ValueError for weights not summing to 1.0"
        except ValueError as e:
            assert "1.1" in str(e) or "weights" in str(e).lower()
            print("PASS: config validation catches scoring weights that don't sum to 1.0")


def test_config_validation_passes_on_the_real_expanded_config():
    """Sanity check against the actual 200-ticker config used in
    production — should load clean with no validation errors."""
    from src.ingestion import load_config
    config = load_config("config/universe.yaml")
    assert len(config["universe"]) >= 190, f"expected ~200 tickers, got {len(config['universe'])}"
    print(f"PASS: the real config/universe.yaml ({len(config['universe'])} tickers) passes validation cleanly")


def test_failed_fetch_isolated():
    with patch("src.ingestion.yf.Ticker", side_effect=Exception("network timeout")):
        snap = fetch_ticker("BADCO.NS", retries=1, backoff_seconds=0)

    assert snap.fetch_ok is False, "expected failed fetch to be marked fetch_ok=False"
    assert snap.error == "network timeout"
    assert snap.market_cap is None, "failed fetch should not silently zero-fill numeric fields"
    print("PASS: failed fetch is isolated, not silently zero-filled")


def test_universe_mixed_success_and_failure():
    test_dir = Path("data/raw_test")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    config = {
        "universe": ["GOOD.NS", "BAD.NS"],
        "data": {"cache_dir": str(test_dir), "cache_expiry_days": 1},
    }

    def side_effect(ticker):
        if ticker == "GOOD.NS":
            m = MagicMock()
            m.info = {"marketCap": 1000, "currentPrice": 10}
            m.balance_sheet = MagicMock(empty=True)
            m.financials = MagicMock(empty=True)
            return m
        raise Exception("simulated failure")

    with patch("src.ingestion.yf.Ticker", side_effect=side_effect):
        results = fetch_universe(config)

    assert len(results) == 2
    good = next(r for r in results if r.ticker == "GOOD.NS")
    bad = next(r for r in results if r.ticker == "BAD.NS")
    assert good.fetch_ok is True
    assert bad.fetch_ok is False, "one bad ticker should not crash the whole universe run"

    shutil.rmtree(test_dir)
    print("PASS: mixed universe run isolates the bad ticker, keeps the good one")


if __name__ == "__main__":
    test_successful_fetch()
    test_failed_fetch_isolated()
    test_universe_mixed_success_and_failure()
    test_config_validation_catches_duplicate_tickers()
    test_config_validation_catches_missing_ns_suffix()
    test_config_validation_catches_bad_weights()
    test_config_validation_passes_on_the_real_expanded_config()
    print("\nAll mock tests passed.")
