"""
Smoke tests for acquisition_likelihood.py. The most important test here
proves this module is a genuinely DIFFERENT lens from the composite
score — not just a relabeled copy — by constructing a company that's
cheap + low-leverage + weak-quality-vs-sector and confirming it scores
HIGH on acquisition likelihood while scoring LOW on the composite score.
Run with: python3 tests/test_acquisition_likelihood_mock.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.acquisition_likelihood import score_acquisition_likelihood, DEFAULT_WEIGHTS


def _make_df():
    return pd.DataFrame([
        {
            # "VALUE_TARGET": cheap, low leverage, weak quality vs sector —
            # the classic turnaround/value-buyout thesis. Composite score
            # should rank this LOW (weak quality drags it down); acquisition
            # likelihood should rank this HIGH (that weakness IS the thesis).
            "ticker": "VALUE_TARGET.NS", "sector": "x", "total_equity": 100_000.0,
            "valuation_bucket_score": 0.90,  # very cheap vs sector
            "quality_bucket_score": 0.10,     # weak margins/ROE vs sector
            "leverage_bucket_score": 0.85,    # low leverage vs sector
            "composite_score": 0.45,           # for reference — dragged down by weak quality
        },
        {
            # "QUALITY_COMPOUNDER": expensive, low leverage, strong quality.
            # Composite score should rank this reasonably well; acquisition
            # likelihood should rank this LOW — it's not a value target,
            # it's fairly priced for its quality (nothing to "fix").
            "ticker": "QUALITY_COMPOUNDER.NS", "sector": "x", "total_equity": 200_000.0,
            "valuation_bucket_score": 0.15,  # expensive vs sector
            "quality_bucket_score": 0.90,     # strong margins/ROE vs sector
            "leverage_bucket_score": 0.80,    # low leverage vs sector
            "composite_score": 0.70,
        },
        {
            # "DISTRESSED": negative equity — must be excluded entirely,
            # not scored as a high-turnaround-potential name.
            "ticker": "DISTRESSED.NS", "sector": "x", "total_equity": -50_000.0,
            "valuation_bucket_score": 0.95,
            "quality_bucket_score": 0.05,
            "leverage_bucket_score": 0.20,
            "composite_score": 0.20,
        },
    ])


def test_value_target_scores_higher_on_acquisition_likelihood_than_quality_compounder():
    """The core proof this is a genuinely different lens: VALUE_TARGET
    should OUTRANK QUALITY_COMPOUNDER on acquisition likelihood, despite
    QUALITY_COMPOUNDER having a much higher composite_score. If this
    module just reproduced the composite ranking, it would add nothing."""
    df = _make_df()
    result = score_acquisition_likelihood(df)

    value_row = result[result.ticker == "VALUE_TARGET.NS"].iloc[0]
    quality_row = result[result.ticker == "QUALITY_COMPOUNDER.NS"].iloc[0]

    assert value_row["acquisition_likelihood_score"] > quality_row["acquisition_likelihood_score"], (
        "VALUE_TARGET (cheap+weak-quality+low-leverage) should score HIGHER on "
        "acquisition likelihood than QUALITY_COMPOUNDER (expensive+strong-quality)"
    )
    assert value_row["composite_score"] < quality_row["composite_score"], (
        "sanity check on the test setup: VALUE_TARGET's composite_score should "
        "be lower than QUALITY_COMPOUNDER's (this is what makes the inversion meaningful)"
    )
    print(
        f"PASS: VALUE_TARGET scores higher on acquisition likelihood "
        f"({value_row['acquisition_likelihood_score']:.3f}) than QUALITY_COMPOUNDER "
        f"({quality_row['acquisition_likelihood_score']:.3f}), while ranking LOWER on "
        f"composite_score ({value_row['composite_score']} vs {quality_row['composite_score']}) — "
        f"confirms this is a genuinely different lens, not a relabeled composite score"
    )


def test_acquisition_likelihood_score_matches_hand_calculation():
    """Hand-verify the exact weighted formula for VALUE_TARGET.NS:
    valuation=0.90 (weight 0.40), turnaround=1-0.10=0.90 (weight 0.35),
    leverage=0.85 (weight 0.25).
    score = 0.40*0.90 + 0.35*0.90 + 0.25*0.85 = 0.36 + 0.315 + 0.2125 = 0.8875
    """
    df = _make_df()
    result = score_acquisition_likelihood(df)
    value_row = result[result.ticker == "VALUE_TARGET.NS"].iloc[0]

    expected = 0.40 * 0.90 + 0.35 * (1 - 0.10) + 0.25 * 0.85
    assert abs(value_row["acquisition_likelihood_score"] - expected) < 1e-9, (
        f"expected {expected:.4f}, got {value_row['acquisition_likelihood_score']:.4f}"
    )
    assert abs(value_row["turnaround_potential_score"] - 0.90) < 1e-9, "turnaround should be 1 - quality_bucket_score"
    print(f"PASS: acquisition likelihood score matches hand calculation ({expected:.4f})")


def test_negative_equity_company_is_excluded_not_scored():
    """A negative-equity company must be excluded entirely — its weak
    quality score reflects real distress, not a turnaround opportunity,
    and scoring it would conflate the two very different situations."""
    df = _make_df()
    result = score_acquisition_likelihood(df)
    assert "DISTRESSED.NS" not in result["ticker"].values, (
        "negative-equity company must be excluded from acquisition-likelihood "
        "scoring entirely, not scored as a high-turnaround-potential candidate"
    )
    print("PASS: negative-equity company is excluded from scoring, not ranked")


def test_missing_component_is_reweighted_not_zeroed():
    """If one component (e.g. leverage_bucket_score) is missing for a
    company, the score should be computed from the remaining two,
    reweighted — same discipline as scoring.py's composite score."""
    df = _make_df()
    df.loc[df.ticker == "VALUE_TARGET.NS", "leverage_bucket_score"] = None

    result = score_acquisition_likelihood(df)
    value_row = result[result.ticker == "VALUE_TARGET.NS"].iloc[0]

    assert value_row["likelihood_components_used"] == 2, "should report 2 components used, not 3"
    expected_weight_sum = DEFAULT_WEIGHTS["valuation"] + DEFAULT_WEIGHTS["turnaround_potential"]
    expected = (DEFAULT_WEIGHTS["valuation"] * 0.90 + DEFAULT_WEIGHTS["turnaround_potential"] * 0.90) / expected_weight_sum
    assert abs(value_row["acquisition_likelihood_score"] - expected) < 1e-9
    print("PASS: missing component (leverage) is correctly reweighted across the remaining 2 components")


def test_sector_rank_computed_correctly():
    """Both companies are in sector 'x' — VALUE_TARGET should be sector
    rank 1 (highest acquisition likelihood in its sector)."""
    df = _make_df()
    result = score_acquisition_likelihood(df)
    value_row = result[result.ticker == "VALUE_TARGET.NS"].iloc[0]
    assert value_row["acquisition_likelihood_sector_rank"] == 1
    print("PASS: sector rank correctly computed (VALUE_TARGET is #1 in its sector)")


def test_missing_required_columns_raises_clear_error():
    df = pd.DataFrame([{"ticker": "A.NS", "valuation_bucket_score": 0.5}])  # missing several required cols
    try:
        score_acquisition_likelihood(df)
        assert False, "expected a ValueError for missing required columns"
    except ValueError as e:
        assert "quality_bucket_score" in str(e) or "leverage_bucket_score" in str(e)
        print("PASS: missing required columns raises a clear, actionable error")


if __name__ == "__main__":
    test_value_target_scores_higher_on_acquisition_likelihood_than_quality_compounder()
    test_acquisition_likelihood_score_matches_hand_calculation()
    test_negative_equity_company_is_excluded_not_scored()
    test_missing_component_is_reweighted_not_zeroed()
    test_sector_rank_computed_correctly()
    test_missing_required_columns_raises_clear_error()
    print("\nAll acquisition likelihood mock tests passed.")
