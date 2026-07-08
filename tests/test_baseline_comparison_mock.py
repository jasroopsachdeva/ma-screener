"""
Smoke tests for baseline_comparison.py. Uses a constructed scenario where
composite and naive P/E rankings deliberately diverge, so the "caught by
composite only" / "caught by naive only" logic can be verified by hand.
Run with: python3 tests/test_baseline_comparison_mock.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_comparison import compute_naive_rank, compare_rankings


def _make_diverging_universe():
    # 4 companies, same sector. Composite rank (already computed, as if
    # from scoring.py) intentionally disagrees with what a naive P/E-only
    # sort would produce:
    #   - CHEAP.NS: lowest P/E (naive rank 1) but composite rank 4 (bad on
    #     quality/leverage/growth) -> should be "caught by naive only"
    #   - QUALITY.NS: highest P/E (naive rank 4) but composite rank 1
    #     (excellent on everything else) -> should be "caught by composite only"
    return pd.DataFrame([
        {"ticker": "CHEAP.NS", "sector": "cement", "rank": 4, "trailing_pe": 5.0},
        {"ticker": "MID1.NS", "sector": "cement", "rank": 2, "trailing_pe": 15.0},
        {"ticker": "MID2.NS", "sector": "cement", "rank": 3, "trailing_pe": 20.0},
        {"ticker": "QUALITY.NS", "sector": "cement", "rank": 1, "trailing_pe": 40.0},
    ])


def test_naive_rank_matches_hand_calc():
    df = _make_diverging_universe()
    ranked = compute_naive_rank(df)
    cheap_naive_rank = ranked.loc[ranked.ticker == "CHEAP.NS", "naive_rank"].iloc[0]
    quality_naive_rank = ranked.loc[ranked.ticker == "QUALITY.NS", "naive_rank"].iloc[0]
    assert cheap_naive_rank == 1.0, f"CHEAP.NS (lowest P/E) should be naive rank 1, got {cheap_naive_rank}"
    assert quality_naive_rank == 4.0, f"QUALITY.NS (highest P/E) should be naive rank 4, got {quality_naive_rank}"
    print("PASS: naive P/E-only rank matches hand calculation")


def test_divergent_names_identified_correctly():
    df = _make_diverging_universe()
    result = compare_rankings(df, top_n=1)

    assert "QUALITY.NS" in result["caught_by_composite_only"], (
        "QUALITY.NS (composite rank 1, naive rank 4) should be flagged as "
        "caught by composite but missed by a naive P/E-only screen"
    )
    assert "CHEAP.NS" in result["caught_by_naive_only"], (
        "CHEAP.NS (naive rank 1, composite rank 4) should be flagged as "
        "cheap on P/E alone but not composite-favored"
    )
    print(f"PASS: divergent names correctly identified — "
          f"composite-only: {result['caught_by_composite_only']}, "
          f"naive-only: {result['caught_by_naive_only']}")


def test_identical_rankings_produce_full_overlap():
    """Sanity check the other direction: if composite rank and naive P/E
    rank are identical, top-N overlap should be 100% and there should be
    zero divergent names — proves the comparison isn't just always
    flagging differences regardless of the actual data."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "it", "rank": 1, "trailing_pe": 10.0},
        {"ticker": "B.NS", "sector": "it", "rank": 2, "trailing_pe": 20.0},
        {"ticker": "C.NS", "sector": "it", "rank": 3, "trailing_pe": 30.0},
    ])
    result = compare_rankings(df, top_n=2)
    assert result["overlap_count"] == 2
    assert result["overlap_pct"] == 1.0
    assert result["caught_by_composite_only"] == []
    assert result["caught_by_naive_only"] == []
    assert abs(result["rank_correlation"] - 1.0) < 1e-9, "identical rankings should have perfect correlation"
    print("PASS: identical composite/naive rankings correctly show 100% overlap and zero divergence")


def test_no_internal_contradiction_between_overlap_and_divergence():
    """Regression test for the real bug: with sector sizes of 7-8, a
    sector-relative naive rank never exceeded 8, so 'naive_rank <= 10' was
    trivially true for nearly the whole universe. This silently turned
    'naive top 10' into 'everyone,' producing a self-contradictory report:
    100% overlap AND several 'caught by naive only' names — mathematically
    impossible if 'top 10' means what it says. This test reproduces the
    exact universe shape that broke and checks the reported numbers are
    internally consistent, not just that the code runs."""
    df = pd.DataFrame(
        [{"ticker": f"C{i}.NS", "sector": "cement", "rank": i + 1, "trailing_pe": 10 + i} for i in range(8)]
        + [{"ticker": f"I{i}.NS", "sector": "it", "rank": i + 9, "trailing_pe": 10 + i} for i in range(7)]
    )
    result = compare_rankings(df, top_n=10)

    # If overlap is 100% of top_n, "caught by naive only" MUST be empty —
    # anything else is a logical contradiction (this is exactly what broke).
    if result["overlap_count"] == result["top_n"]:
        assert result["caught_by_naive_only"] == [], (
            f"CONTRADICTION: reported {result['overlap_count']}/{result['top_n']} overlap "
            f"(100%) but also listed 'naive only' names: {result['caught_by_naive_only']} — "
            f"these cannot both be true"
        )

    # Independently verify the overlap count is mathematically consistent
    # with the two sets reported, regardless of what the numbers turn out to be.
    composite_only = set(result["caught_by_composite_only"])
    naive_only = set(result["caught_by_naive_only"])
    overlap_set = set(result["overlap_tickers"])
    assert composite_only.isdisjoint(overlap_set), "a ticker can't be both 'composite only' and in the overlap"
    assert naive_only.isdisjoint(overlap_set), "a ticker can't be both 'naive only' and in the overlap"
    assert composite_only.isdisjoint(naive_only), "a ticker can't be both 'composite only' and 'naive only'"
    print(f"PASS: no internal contradiction — overlap={result['overlap_count']}/{result['top_n']}, "
          f"composite-only={len(composite_only)}, naive-only={len(naive_only)}, all mutually consistent")


if __name__ == "__main__":
    test_naive_rank_matches_hand_calc()
    test_divergent_names_identified_correctly()
    test_identical_rankings_produce_full_overlap()
    test_no_internal_contradiction_between_overlap_and_divergence()
    print("\nAll baseline comparison mock tests passed.")
