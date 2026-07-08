"""
Smoke tests for comps_dupont.py. Uses ACC's real data (hand-verified
earlier in this project) to confirm the DuPont math actually reconciles,
plus edge cases: negative equity and sector valuation labeling.
Run with: python3 tests/test_comps_dupont_mock.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.comps_dupont import compute_sector_comps, compute_dupont

# Real ACC.NS data, hand-verified earlier: total_debt/total_equity = 2.09%
# matched the reported debt_to_equity almost exactly.
REAL_ACC_ROW = {
    "ticker": "ACC.NS", "sector": "cement", "trailing_pe": 11.739512,
    "return_on_equity": 0.109280005, "net_margin": 0.082959995,
    "asset_turnover": 1.2187217464845206, "total_debt": 4286099968,
    "total_equity": 205505400000.0,
}


def test_dupont_reconciles_with_real_acc_data():
    df = pd.DataFrame([REAL_ACC_ROW])
    result = compute_dupont(df)
    row = result.iloc[0]

    expected_equity_multiplier = (4286099968 + 205505400000.0) / 205505400000.0
    assert abs(row["equity_multiplier"] - expected_equity_multiplier) < 1e-6

    implied_roe = row["net_margin"] * row["asset_turnover"] * row["equity_multiplier"]
    reported_roe = REAL_ACC_ROW["return_on_equity"]
    gap = abs(implied_roe - reported_roe)

    # Should be reasonably close — exact match isn't expected since
    # asset_turnover uses an equity-based proxy rather than true total
    # assets, but a huge gap would signal something is actually broken.
    assert gap < 0.05, f"DuPont-implied ROE ({implied_roe:.4f}) diverges too far from reported ROE ({reported_roe:.4f})"
    print(f"PASS: DuPont decomposition on real ACC data reconciles (implied={implied_roe:.4f} vs reported={reported_roe:.4f}, gap={gap:.4f})")


def test_negative_equity_does_not_crash_and_is_flagged():
    df = pd.DataFrame([{
        "ticker": "BROKEN.NS", "sector": "cement", "trailing_pe": 15.0,
        "return_on_equity": -0.5, "net_margin": -0.10, "asset_turnover": 0.8,
        "total_debt": 5000000000, "total_equity": -1000000000.0,  # negative equity
    }])
    result = compute_dupont(df)
    row = result.iloc[0]

    assert pd.isna(row["equity_multiplier"]), "equity multiplier must not be computed on negative equity"
    assert row["dupont_flag"] == "negative or missing equity — equity multiplier not computed"
    print("PASS: negative equity is flagged and excluded from the equity multiplier calc, doesn't crash")


def test_valuation_label_cheap_vs_rich():
    df = pd.DataFrame([
        {"ticker": "CHEAP.NS", "sector": "it", "trailing_pe": 10.0},
        {"ticker": "MID.NS", "sector": "it", "trailing_pe": 15.0},
        {"ticker": "RICH.NS", "sector": "it", "trailing_pe": 25.0},
    ])
    result = compute_sector_comps(df)

    cheap_label = result.loc[result.ticker == "CHEAP.NS", "valuation_label"].iloc[0]
    rich_label = result.loc[result.ticker == "RICH.NS", "valuation_label"].iloc[0]

    assert cheap_label == "cheap vs sector", f"expected 'cheap vs sector', got '{cheap_label}'"
    assert rich_label == "rich vs sector", f"expected 'rich vs sector', got '{rich_label}'"
    print(f"PASS: valuation labels correctly identify cheap ({cheap_label}) vs rich ({rich_label}) vs sector median")


def test_missing_required_columns_fails_clearly():
    """Regression test for the real bug: an older scored_universe.csv
    missing asset_turnover/total_debt/total_equity used to crash with a
    raw pandas KeyError traceback. Should now raise a clear, actionable
    ValueError instead."""
    df = pd.DataFrame([{
        "ticker": "ACC.NS", "sector": "cement", "trailing_pe": 11.7,
        "return_on_equity": 0.109, "net_margin": 0.083,
        # asset_turnover, total_debt, total_equity deliberately omitted
    }])
    try:
        compute_dupont(df)
        assert False, "expected a ValueError for missing required columns"
    except ValueError as e:
        assert "asset_turnover" in str(e) or "total_debt" in str(e), "error message should name the missing columns"
        assert "re-run" in str(e).lower(), "error message should tell the user how to fix it"
        print("PASS: missing required columns raises a clear, actionable error instead of a raw KeyError")


def test_gap_column_abs_does_not_crash_with_missing_roe():
    """Regression test for the real bug: a row with missing return_on_equity
    (e.g. SHREECEM.NS, which had total_debt/total_equity but no reported
    ROE from yfinance) left dupont_vs_reported_roe_gap as Python None on
    that row, and calling .abs() on a column mixing None with floats threw
    TypeError: bad operand type for abs(): 'NoneType'. Should now compute
    cleanly as NaN, which .abs() and comparisons handle natively."""
    df = pd.DataFrame([
        {  # normal row, ROE present — should get a real gap value
            "ticker": "ACC.NS", "sector": "cement", "trailing_pe": 11.7,
            "return_on_equity": 0.109, "net_margin": 0.083, "asset_turnover": 1.22,
            "total_debt": 4286099968, "total_equity": 205505400000.0,
        },
        {  # SHREECEM-like row: ROE missing, everything else present
            "ticker": "SHREECEM.NS", "sector": "cement", "trailing_pe": 54.4,
            "return_on_equity": None, "net_margin": 0.1453, "asset_turnover": 0.38149,
            "total_debt": 33939900416, "total_equity": 232675300000.0,
        },
    ])
    result = compute_dupont(df)

    # This is the exact line that crashed in build_report() — must not raise
    try:
        gaps = result["dupont_vs_reported_roe_gap"].abs()
    except TypeError as e:
        assert False, f".abs() still crashes on this column: {e}"

    shreecem_gap = result.loc[result.ticker == "SHREECEM.NS", "dupont_vs_reported_roe_gap"].iloc[0]
    assert pd.isna(shreecem_gap), "gap should be NaN (not computable without reported ROE), not crash"

    acc_gap = result.loc[result.ticker == "ACC.NS", "dupont_vs_reported_roe_gap"].iloc[0]
    assert pd.notna(acc_gap), "ACC's gap should still compute normally since it has all required fields"
    print("PASS: .abs() on the gap column no longer crashes when a row (e.g. SHREECEM) has missing ROE")


def test_implied_roe_recovers_even_when_reported_roe_missing():
    """Regression test for the exact SHREECEM.NS bug: dupont_implied_roe
    was incorrectly gated on return_on_equity ALSO being present, which
    defeated the entire point of the recovery feature — a company with
    missing reported ROE but present margin/turnover/leverage data should
    still get a computed implied ROE."""
    df = pd.DataFrame([{
        "ticker": "SHREECEM.NS", "sector": "cement", "trailing_pe": 54.4,
        "return_on_equity": None,  # this is genuinely missing from yfinance
        "net_margin": 0.1453, "asset_turnover": 0.900116,
        "total_debt": 33939900416, "total_equity": 232675300000.0,
    }])
    result = compute_dupont(df)
    row = result.iloc[0]

    assert pd.notna(row["dupont_implied_roe"]), (
        "dupont_implied_roe must compute from margin/turnover/multiplier "
        "alone, even when reported ROE is missing — this is the whole "
        "point of the recovery feature"
    )
    assert pd.isna(row["dupont_vs_reported_roe_gap"]), (
        "the gap check correctly stays NaN since there's no reported ROE "
        "to compare against — but that must not block the implied ROE itself"
    )
    print(f"PASS: SHREECEM-like row recovers a real dupont_implied_roe ({row['dupont_implied_roe']:.4f}) "
          f"despite missing reported ROE")


if __name__ == "__main__":
    test_dupont_reconciles_with_real_acc_data()
    test_negative_equity_does_not_crash_and_is_flagged()
    test_valuation_label_cheap_vs_rich()
    test_missing_required_columns_fails_clearly()
    test_gap_column_abs_does_not_crash_with_missing_roe()
    test_implied_roe_recovers_even_when_reported_roe_missing()
    print("\nAll comps + DuPont mock tests passed.")
