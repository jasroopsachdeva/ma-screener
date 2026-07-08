"""
Smoke tests for explainability.py. Uses a hand-checkable company profile
where the strongest and weakest buckets are unambiguous by construction.
Run with: python3 tests/test_explainability_mock.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.explainability import explain_company, _bucket_contribution

WEIGHTS = {"valuation": 0.30, "quality": 0.30, "leverage": 0.20, "growth": 0.20}


def _make_clear_row():
    # Constructed so quality is UNAMBIGUOUSLY the strongest bucket (score
    # 1.0) and leverage UNAMBIGUOUSLY the weakest (score 0.0) — every other
    # bucket sits in between, so there's no ambiguity about which should
    # be picked as strongest/weakest.
    return pd.Series({
        "ticker": "TEST.NS", "sector": "cement", "rank": 1, "sector_rank": 1,
        "valuation_bucket_score": 0.5,
        "quality_bucket_score": 1.0,
        "leverage_bucket_score": 0.0,
        "growth_bucket_score": 0.5,
        "trailing_pe_score": 0.4, "ev_to_ebitda_score": 0.6,      # valuation metrics
        "return_on_equity_score": 1.0, "net_margin_score": 1.0,    # quality metrics (both perfect)
        "debt_to_equity_ratio_score": 0.0,                        # leverage metric (worst possible)
        "revenue_growth_score": 0.5,                              # growth metric
    })


def test_strongest_and_weakest_bucket_identified_correctly():
    row = _make_clear_row()
    result = explain_company(row, WEIGHTS)
    assert result["strongest_bucket"] == "quality", f"expected quality as strongest, got {result['strongest_bucket']}"
    assert result["weakest_bucket"] == "leverage", f"expected leverage as weakest, got {result['weakest_bucket']}"
    print("PASS: strongest bucket (quality) and weakest bucket (leverage) correctly identified")


def test_strongest_metric_within_bucket_identified_correctly():
    row = _make_clear_row()
    result = explain_company(row, WEIGHTS)
    # Both quality metrics are tied at 1.0 — either is a valid answer, just
    # confirm it picked one of the two quality metrics, not something else.
    assert result["strongest_metric"] in ("return_on_equity", "net_margin")
    assert result["weakest_metric"] == "debt_to_equity_ratio"
    print("PASS: strongest/weakest metric correctly scoped to the right bucket")


def test_bucket_contributions_sum_to_composite_when_all_buckets_present():
    row = _make_clear_row()
    contributions = _bucket_contribution(row, WEIGHTS)
    total = sum(contributions.values())
    expected_composite = (
        WEIGHTS["valuation"] * 0.5 + WEIGHTS["quality"] * 1.0
        + WEIGHTS["leverage"] * 0.0 + WEIGHTS["growth"] * 0.5
    )
    assert abs(total - expected_composite) < 1e-9, f"contributions sum to {total}, expected {expected_composite}"
    print(f"PASS: bucket contributions sum exactly to the composite score ({total:.4f})")


def test_missing_bucket_reweights_contributions_correctly():
    """If a bucket is missing (NaN), contributions should reweight across
    the remaining buckets — same discipline as the composite score itself
    — not silently treat the missing bucket as a zero contribution."""
    row = _make_clear_row()
    row["growth_bucket_score"] = None
    contributions = _bucket_contribution(row, WEIGHTS)

    assert "growth" not in contributions, "missing bucket should not appear in contributions at all"
    total_weight_used = WEIGHTS["valuation"] + WEIGHTS["quality"] + WEIGHTS["leverage"]
    expected_quality_contribution = (WEIGHTS["quality"] / total_weight_used) * 1.0
    assert abs(contributions["quality"] - expected_quality_contribution) < 1e-9
    print("PASS: missing bucket is excluded and remaining contributions are correctly reweighted")


def test_summary_string_is_readable_and_mentions_key_facts():
    row = _make_clear_row()
    result = explain_company(row, WEIGHTS)
    summary = result["summary"]
    assert "TEST.NS" in summary
    assert "quality" in summary
    assert "leverage" in summary
    assert "#1" in summary
    print(f"PASS: summary string is readable and includes ticker, rank, strongest/weakest factors:\n      \"{summary}\"")


if __name__ == "__main__":
    test_strongest_and_weakest_bucket_identified_correctly()
    test_strongest_metric_within_bucket_identified_correctly()
    test_bucket_contributions_sum_to_composite_when_all_buckets_present()
    test_missing_bucket_reweights_contributions_correctly()
    test_summary_string_is_readable_and_mentions_key_facts()
    print("\nAll explainability mock tests passed.")
