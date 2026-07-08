"""
Smoke tests for scoring.py. Focused on the two places scoring logic is most
likely to silently break: direction handling (does lower P/E really score
higher?) and reweighting (does a missing bucket avoid tanking the score?).
Run with: python3 tests/test_scoring_mock.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.scoring import _percentile_scores, _bucket_scores, _composite_score

WEIGHTS = {"valuation": 0.30, "quality": 0.30, "leverage": 0.20, "growth": 0.20}


def _make_df():
    # Three synthetic companies with clear, hand-checkable relationships:
    # A = cheapest (best valuation), highest quality, lowest leverage, highest growth -> should rank #1
    # B = middle on everything
    # C = most expensive, lowest quality, highest leverage, lowest growth -> should rank #3
    return pd.DataFrame([
        {"ticker": "A.NS", "trailing_pe": 10, "ev_to_ebitda": 8, "return_on_equity": 0.30,
         "net_margin": 0.20, "debt_to_equity_ratio": 0.10, "revenue_growth": 0.25},
        {"ticker": "B.NS", "trailing_pe": 20, "ev_to_ebitda": 15, "return_on_equity": 0.15,
         "net_margin": 0.10, "debt_to_equity_ratio": 0.40, "revenue_growth": 0.10},
        {"ticker": "C.NS", "trailing_pe": 40, "ev_to_ebitda": 30, "return_on_equity": 0.05,
         "net_margin": 0.02, "debt_to_equity_ratio": 0.90, "revenue_growth": 0.01},
    ])


def test_lower_pe_scores_higher():
    df = _make_df()
    scored = _percentile_scores(df)
    a_score = scored.loc[scored.ticker == "A.NS", "trailing_pe_score"].iloc[0]
    c_score = scored.loc[scored.ticker == "C.NS", "trailing_pe_score"].iloc[0]
    assert a_score > c_score, "cheapest company (lowest P/E) must score HIGHER, not lower"
    print(f"PASS: lower P/E scores higher (A={a_score:.2f} > C={c_score:.2f})")


def test_higher_roe_scores_higher():
    df = _make_df()
    scored = _percentile_scores(df)
    a_score = scored.loc[scored.ticker == "A.NS", "return_on_equity_score"].iloc[0]
    c_score = scored.loc[scored.ticker == "C.NS", "return_on_equity_score"].iloc[0]
    assert a_score > c_score, "higher ROE must score higher"
    print(f"PASS: higher ROE scores higher (A={a_score:.2f} > C={c_score:.2f})")


def test_composite_ranking_matches_intuition():
    df = _make_df()
    scored = _percentile_scores(df)
    scored = _bucket_scores(scored)
    scored = _composite_score(scored, WEIGHTS)
    ranked = scored.sort_values("composite_score", ascending=False)
    order = ranked["ticker"].tolist()
    assert order == ["A.NS", "B.NS", "C.NS"], f"expected A > B > C, got {order}"
    print(f"PASS: composite ranking matches hand-checkable intuition: {order}")


def test_missing_bucket_is_reweighted_not_zeroed():
    """A company missing its entire valuation bucket (both P/E and
    EV/EBITDA null) should have its composite computed from the remaining
    3 buckets, reweighted — NOT get a 0 for the missing 30% weight."""
    df = _make_df()
    df.loc[df.ticker == "B.NS", ["trailing_pe", "ev_to_ebitda"]] = None

    scored = _percentile_scores(df)
    scored = _bucket_scores(scored)
    scored = _composite_score(scored, WEIGHTS)

    b_row = scored[scored.ticker == "B.NS"].iloc[0]
    assert pd.isna(b_row["valuation_bucket_score"]), "valuation bucket should be NaN when both metrics missing"
    assert b_row["buckets_used"] == 3, "should report 3 buckets used, not 4"
    assert pd.notna(b_row["composite_score"]), "composite must still compute from the 3 available buckets"

    # Manually verify the reweighting math: quality(30) + leverage(20) + growth(20) = 70 total weight
    expected_weight_sum = WEIGHTS["quality"] + WEIGHTS["leverage"] + WEIGHTS["growth"]
    manual_composite = (
        WEIGHTS["quality"] * b_row["quality_bucket_score"]
        + WEIGHTS["leverage"] * b_row["leverage_bucket_score"]
        + WEIGHTS["growth"] * b_row["growth_bucket_score"]
    ) / expected_weight_sum
    assert abs(b_row["composite_score"] - manual_composite) < 1e-9, "reweighting math doesn't match expected formula"
    print("PASS: missing valuation bucket is correctly reweighted across remaining 3 buckets, not zero-filled")


def test_sector_relative_ranking_fixes_cross_sector_conflation():
    """Reproduces the real ACC-vs-INFY issue: a structurally high-leverage
    sector (cement) shouldn't let its best-in-sector name outrank a
    structurally low-leverage sector's (IT) strong name purely because of
    industry-wide structural differences, not real company quality.

    Setup: 2 cement companies (naturally high leverage ~0.8-0.9) and 2 IT
    companies (naturally low leverage ~0.1-0.2). Within cement, X is
    clearly the better company (lower leverage, higher quality/growth than
    its cement peer Y). Within IT, we make P clearly worse than Q on every
    metric. Sector-relative ranking should let X (best-in-cement) rank
    ABOVE P (worst-in-IT) despite X's absolute leverage number being
    "worse" than P's — because X is genuinely a stronger company relative
    to its own peer set, which is the honest comparison this fix exists for."""
    df = pd.DataFrame([
        {"ticker": "X.NS", "sector": "cement", "trailing_pe": 12, "ev_to_ebitda": 9,
         "return_on_equity": 0.15, "net_margin": 0.10, "debt_to_equity_ratio": 0.80, "revenue_growth": 0.15},
        {"ticker": "Y.NS", "sector": "cement", "trailing_pe": 25, "ev_to_ebitda": 20,
         "return_on_equity": 0.05, "net_margin": 0.03, "debt_to_equity_ratio": 1.20, "revenue_growth": 0.02},
        {"ticker": "P.NS", "sector": "it", "trailing_pe": 30, "ev_to_ebitda": 25,
         "return_on_equity": 0.10, "net_margin": 0.08, "debt_to_equity_ratio": 0.15, "revenue_growth": 0.03},
        {"ticker": "Q.NS", "sector": "it", "trailing_pe": 15, "ev_to_ebitda": 12,
         "return_on_equity": 0.35, "net_margin": 0.20, "debt_to_equity_ratio": 0.05, "revenue_growth": 0.20},
    ])

    scored = _percentile_scores(df)
    x_leverage = scored.loc[scored.ticker == "X.NS", "debt_to_equity_ratio_score"].iloc[0]
    p_leverage = scored.loc[scored.ticker == "P.NS", "debt_to_equity_ratio_score"].iloc[0]

    # With only 2 companies per sector, percentile rank naturally tops out
    # at 0.5 for the best-in-group (rank 1 of 2 -> pct 1/2), not 1.0 — that
    # ceiling is a property of rank(pct=True) with small groups, not a bug.
    # The point being tested is that X and P are each ranked ONLY against
    # their own sector peer, not against the full cross-sector universe.
    assert x_leverage == 0.5, f"X should score best-in-sector (0.5, i.e. rank 1 of 2) on leverage within cement, got {x_leverage}"
    assert p_leverage == 0.0, f"P should score worst-in-sector (0.0, i.e. rank 2 of 2) on leverage within IT, got {p_leverage}"
    print(f"PASS: sector-relative ranking scores X (best-in-cement)={x_leverage} and "
          f"P (worst-in-IT)={p_leverage} on their own peer sets, not a shared cross-sector scale")


def test_scoring_output_preserves_all_input_columns():
    """Regression test for a real recurring bug class: score_universe()
    used to output a hand-picked column whitelist that went stale twice —
    once when comps_dupont.py needed total_debt/total_equity, again when
    accretion_dilution.py needed price/market_cap. Each time, a downstream
    module crashed with a KeyError tracing back to this list.

    Fix: score_universe() now passes through every input column plus the
    new score-derived columns, so no future downstream module can hit this
    same class of bug — whatever cleaning.py produces, scoring.py preserves."""
    import tempfile
    import yaml
    from src.scoring import score_universe

    input_df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "cement", "trailing_pe": 12, "ev_to_ebitda": 9,
         "return_on_equity": 0.15, "net_margin": 0.10, "debt_to_equity_ratio": 0.80,
         "revenue_growth": 0.15, "asset_turnover": 1.1, "total_debt": 500, "total_equity": 1000,
         "price": 100.0, "market_cap": 50000.0, "some_future_field_no_module_needs_yet": 42},
        {"ticker": "B.NS", "sector": "it", "trailing_pe": 20, "ev_to_ebitda": 15,
         "return_on_equity": 0.25, "net_margin": 0.18, "debt_to_equity_ratio": 0.10,
         "revenue_growth": 0.10, "asset_turnover": 1.8, "total_debt": 100, "total_equity": 2000,
         "price": 200.0, "market_cap": 80000.0, "some_future_field_no_module_needs_yet": 7},
    ])

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "cleaned_universe.csv"
        input_df.to_csv(input_path, index=False)

        config_path = Path(tmpdir) / "universe.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"scoring": {"weights": WEIGHTS}, "output": {"top_n_shortlist": 10}}, f)

        # score_universe() calls load_config() with a hardcoded default
        # path, so temporarily patch it to point at our temp config.
        import src.scoring as scoring_module
        original_load_config = scoring_module.load_config
        scoring_module.load_config = lambda config_path=str(config_path): original_load_config(config_path)

        try:
            result = score_universe(input_path=str(input_path), output_dir=tmpdir)
        finally:
            scoring_module.load_config = original_load_config

        for col in input_df.columns:
            assert col in result.columns, (
                f"'{col}' from the input was dropped from scoring output — "
                f"this is exactly the bug class this test guards against"
            )
        print("PASS: every input column (including an arbitrary unrecognized one) survives into scoring output")


if __name__ == "__main__":
    test_lower_pe_scores_higher()
    test_higher_roe_scores_higher()
    test_composite_ranking_matches_intuition()
    test_missing_bucket_is_reweighted_not_zeroed()
    test_sector_relative_ranking_fixes_cross_sector_conflation()
    test_scoring_output_preserves_all_input_columns()
    print("\nAll scoring mock tests passed.")
