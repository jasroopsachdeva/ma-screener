"""
Smoke tests for accretion_dilution.py. Uses a hand-calculated deal scenario
so the math can be verified independently, not just "code runs."

Hand calculation (verify against test assertions below):
  Acquirer A: price=100, market_cap=1,000,000 -> shares=10,000
              ROE=0.20, total_equity=500,000 -> net_income=100,000, EPS=10.0
  Target T:   price=50, market_cap=200,000 -> shares=4,000
              ROE=0.10, total_equity=300,000 -> net_income=30,000, EPS=7.5

  Deal: 20% premium, 50% cash / 50% stock
    purchase_price_per_share = 50 * 1.20 = 60
    deal_value = 60 * 4,000 = 240,000
    cash = 120,000 | stock = 120,000
    new_shares_issued = 120,000 / 100 = 1,200
    pro_forma_net_income = 100,000 + 30,000 = 130,000
    pro_forma_shares = 10,000 + 1,200 = 11,200
    pro_forma_eps = 130,000 / 11,200 = 11.607142857...
    accretion % = (11.6071428... - 10) / 10 = 16.0714...%  -> ACCRETIVE

Run with: python3 tests/test_accretion_dilution_mock.py
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.accretion_dilution import DealInputs, run_deal, run_sensitivity, _company_fundamentals, find_optimal_terms, find_best_targets, project_multi_year_accretion


def _make_universe():
    return pd.DataFrame([
        {"ticker": "A.NS", "price": 100.0, "market_cap": 1_000_000.0, "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "T.NS", "price": 50.0, "market_cap": 200_000.0, "return_on_equity": 0.10, "total_equity": 300_000.0},
    ])


def test_fundamentals_derivation_matches_hand_calc():
    df = _make_universe()
    a = _company_fundamentals(df, "A.NS")
    assert a["shares_outstanding"] == 10_000.0
    assert a["net_income"] == 100_000.0
    assert a["eps"] == 10.0
    print("PASS: derived fundamentals (shares, net income, EPS) match hand calculation")


def test_deal_math_matches_hand_calc():
    df = _make_universe()
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=0.50)
    result = run_deal(df, deal)

    assert result.deal_value == 240_000.0
    assert result.cash_consideration == 120_000.0
    assert result.stock_consideration == 120_000.0
    assert result.new_shares_issued == 1_200.0
    assert result.pro_forma_net_income == 130_000.0
    assert result.pro_forma_shares == 11_200.0

    expected_eps = 130_000.0 / 11_200.0
    assert abs(result.pro_forma_eps - expected_eps) < 1e-9

    expected_accretion = (expected_eps - 10.0) / 10.0
    assert abs(result.accretion_dilution_pct - expected_accretion) < 1e-9
    assert result.is_accretive is True
    print(f"PASS: full deal math matches hand calculation (accretion = {result.accretion_dilution_pct:.4%}, expected ~16.07%)")


def test_all_cash_deal_has_no_dilution_from_new_shares():
    """A 100% cash deal should issue zero new shares — pro-forma shares
    should equal the acquirer's standalone share count exactly."""
    df = _make_universe()
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0)
    result = run_deal(df, deal)

    assert result.new_shares_issued == 0.0
    assert result.pro_forma_shares == 10_000.0
    print("PASS: 100% cash deal issues zero new shares, pro-forma share count unchanged")


def test_debt_financed_cash_deducts_after_tax_interest():
    """Hand calculation: 100% cash deal, 100% of that cash is debt-funded
    at 10% interest, 20% tax rate.
      cash_consideration = 240,000 (from the original hand calc, at 20%
                            premium, 100% cash: deal_value = 60*4000=240,000)
      new_debt_raised = 240,000 * 1.0 = 240,000
      interest_expense = 240,000 * 0.10 = 24,000
      after_tax_interest_cost = 24,000 * (1 - 0.20) = 19,200
      pro_forma_net_income = 100,000 + 30,000 - 19,200 = 110,800
      pro_forma_shares = 10,000 (no new shares, 100% cash deal)
      pro_forma_eps = 110,800 / 10,000 = 11.08
      accretion = (11.08 - 10.0) / 10.0 = 10.8%  -> still accretive, but
      LESS accretive than the no-debt-cost version (which would show
      130,000/10,000=13.0 EPS, 30% accretion) — this is the exact gap
      the feature exists to close."""
    df = _make_universe()
    deal = DealInputs(
        "A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0,
        debt_funded_pct=1.0, interest_rate=0.10, tax_rate=0.20,
    )
    result = run_deal(df, deal)

    assert result.new_debt_raised == 240_000.0
    assert result.interest_expense == 24_000.0
    assert abs(result.after_tax_interest_cost - 19_200.0) < 1e-9
    assert abs(result.pro_forma_net_income - 110_800.0) < 1e-9
    assert abs(result.pro_forma_eps - 11.08) < 1e-9

    expected_accretion = (11.08 - 10.0) / 10.0
    assert abs(result.accretion_dilution_pct - expected_accretion) < 1e-9
    print(f"PASS: debt-financed cash correctly deducts after-tax interest cost "
          f"(accretion drops to {result.accretion_dilution_pct:.2%} from the debt-free 30.00%)")


def test_default_synergy_pct_is_zero_and_unchanged():
    """Backward compatibility: with no synergy args passed, synergy_income
    must be exactly 0 and results must match the original hand calc from
    test_deal_math_matches_hand_calc (16.0714% accretion)."""
    df = _make_universe()
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=0.50)
    result = run_deal(df, deal)
    assert result.synergy_income == 0.0
    assert abs(result.accretion_dilution_pct - 0.160714285714) < 1e-6
    print("PASS: default synergy_pct=0.0 leaves results unchanged from before this feature was added")


def test_revenue_synergy_uplifts_target_net_income():
    """Hand calc: 100% cash, 20% premium, 20% revenue synergy on target's NI.
      revenue_synergy_income = 30,000 * 0.20 = 6,000
      pro_forma_net_income = 100,000 + 30,000 + 6,000 = 136,000
      pro_forma_shares = 10,000 (100% cash, no new shares)
      pro_forma_eps = 13.6 -> accretion = 36.0%
    """
    df = _make_universe()
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0, revenue_synergy_pct=0.20)
    result = run_deal(df, deal)

    assert abs(result.synergy_income - 6_000.0) < 1e-9
    assert abs(result.pro_forma_net_income - 136_000.0) < 1e-9
    assert abs(result.accretion_dilution_pct - 0.36) < 1e-9
    print(f"PASS: revenue synergy correctly uplifts target NI (synergy_income={result.synergy_income:,.0f}, "
          f"accretion={result.accretion_dilution_pct:.1%})")


def test_cost_synergy_is_tax_effected_on_combined_income():
    """Hand calc: 100% cash, 20% premium, 10% cost synergy, 25% tax rate (default).
      combined_pre_synergy_ni = 130,000
      cost_synergy_income = 130,000 * 0.10 * (1 - 0.25) = 9,750
      pro_forma_net_income = 130,000 + 9,750 = 139,750
      pro_forma_eps = 13.975 -> accretion = 39.75%
    """
    df = _make_universe()
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0, cost_synergy_pct=0.10)
    result = run_deal(df, deal)

    assert abs(result.synergy_income - 9_750.0) < 1e-9
    assert abs(result.pro_forma_net_income - 139_750.0) < 1e-9
    assert abs(result.accretion_dilution_pct - 0.3975) < 1e-9
    print(f"PASS: cost synergy is correctly tax-effected on combined pre-synergy income "
          f"(synergy_income={result.synergy_income:,.0f}, accretion={result.accretion_dilution_pct:.2%})")


def test_full_stack_synergies_plus_debt_financing_combine_correctly():
    """Integration test: revenue synergy + cost synergy + debt-financed
    cash, all active together. Hand calc:
      combined_pre_synergy_ni = 130,000
      revenue_synergy_income = 30,000 * 0.20 = 6,000
      cost_synergy_income = 130,000 * 0.10 * (1 - 0.20) = 10,400
      synergy_income = 16,400
      new_debt_raised = 240,000 (100% cash, 100% debt-funded)
      interest_expense = 240,000 * 0.10 = 24,000
      after_tax_interest_cost = 24,000 * (1 - 0.20) = 19,200
      pro_forma_net_income = 130,000 + 16,400 - 19,200 = 127,200
      pro_forma_eps = 12.72 -> accretion = 27.2%
    """
    df = _make_universe()
    deal = DealInputs(
        "A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0,
        revenue_synergy_pct=0.20, cost_synergy_pct=0.10,
        debt_funded_pct=1.0, interest_rate=0.10, tax_rate=0.20,
    )
    result = run_deal(df, deal)

    assert abs(result.synergy_income - 16_400.0) < 1e-9
    assert abs(result.after_tax_interest_cost - 19_200.0) < 1e-9
    assert abs(result.pro_forma_net_income - 127_200.0) < 1e-9
    assert abs(result.accretion_dilution_pct - 0.272) < 1e-9
    print(f"PASS: synergies and debt financing combine correctly in the full pro-forma calc "
          f"(synergy_income={result.synergy_income:,.0f}, "
          f"after_tax_interest_cost={result.after_tax_interest_cost:,.0f}, "
          f"accretion={result.accretion_dilution_pct:.1%})")


def test_default_behavior_unchanged_when_debt_funded_pct_is_zero():
    """Backward compatibility check: with debt_funded_pct defaulting to
    0.0, results must be identical to the original (pre-interest-expense)
    behavior — the new feature must be strictly opt-in."""
    df = _make_universe()
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=0.50)  # no debt args passed at all
    result = run_deal(df, deal)

    assert result.new_debt_raised == 0.0
    assert result.interest_expense == 0.0
    assert result.after_tax_interest_cost == 0.0
    assert result.pro_forma_net_income == 130_000.0  # same as the original hand calc, unaffected
    print("PASS: default behavior (debt_funded_pct=0.0) is unchanged from before this feature was added")


def test_avg_price_basis_uses_fetched_average_not_spot():
    """When avg_price_days > 0, the deal price basis should come from the
    (mocked) averaged price, not the spot price in the scored universe —
    while shares_outstanding must stay tied to the ORIGINAL spot price,
    since that's what market_cap was actually computed against."""
    df = _make_universe()  # T.NS spot price = 50.0
    mock_avg_price = 55.0  # simulate a 30-day average that differs from spot

    with patch("src.accretion_dilution.fetch_avg_price", return_value=mock_avg_price):
        deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=0.50, avg_price_days=30)
        result = run_deal(df, deal)

    # purchase_price_per_share should be based on 55.0, not 50.0:
    # 55.0 * 1.20 = 66.0 -> deal_value = 66.0 * 4,000 shares = 264,000
    # (T.NS shares_outstanding = 200,000 / 50.0 = 4,000, using SPOT price)
    assert abs(result.deal_value - 264_000.0) < 1e-6, (
        f"expected deal_value based on averaged price (55.0), got {result.deal_value} "
        f"— suggests spot price was used instead of the averaged price"
    )
    print(f"PASS: avg_price_days correctly uses the fetched average price (55.0) for the deal "
          f"basis instead of spot price (50.0), deal_value={result.deal_value:,.0f}")


def test_avg_price_fetch_failure_raises_clear_error():
    with patch("src.accretion_dilution.fetch_avg_price", side_effect=ValueError("Could not fetch 30-day price history for 'T.NS'")):
        df = _make_universe()
        deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=0.50, avg_price_days=30)
        try:
            run_deal(df, deal)
            assert False, "expected a ValueError when the average price fetch fails"
        except ValueError as e:
            assert "T.NS" in str(e)
            print("PASS: a failed average-price fetch raises a clear error rather than silently using spot price")


def test_refinance_target_debt_adds_to_new_debt_and_interest():
    """Hand calculation: same base deal as the interest expense test
    (100% cash, 20% premium -> deal_value=240,000), but now ALSO refinance
    the target's existing debt of 50,000 at the same 10% rate, 20% tax.
      new_debt_from_cash = 240,000 (100% cash, 0% debt_funded_pct here —
                            isolate the refinancing effect on its own)
      target_debt_refinanced = 50,000
      new_debt_raised = 0 (debt_funded_pct=0, so cash itself isn't debt-funded)
                       + 50,000 (refinanced target debt) = 50,000
      interest_expense = 50,000 * 0.10 = 5,000
      after_tax_interest_cost = 5,000 * (1 - 0.20) = 4,000
      pro_forma_net_income = 100,000 + 30,000 - 4,000 = 126,000
    """
    df = _make_universe()
    df.loc[df.ticker == "T.NS", "total_debt"] = 50_000.0

    deal = DealInputs(
        "A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0,
        debt_funded_pct=0.0,  # isolate: cash itself is NOT debt-funded here
        interest_rate=0.10, tax_rate=0.20,
        refinance_target_debt=True,
    )
    result = run_deal(df, deal)

    assert result.target_debt_refinanced == 50_000.0
    assert result.new_debt_raised == 50_000.0
    assert abs(result.after_tax_interest_cost - 4_000.0) < 1e-9
    assert abs(result.pro_forma_net_income - 126_000.0) < 1e-9
    print(f"PASS: refinancing target's existing debt correctly adds to new_debt_raised and "
          f"interest cost (pro-forma NI = {result.pro_forma_net_income:,.0f}, expected 126,000)")


def test_refinance_target_debt_without_total_debt_raises_clear_error():
    df = _make_universe()  # no total_debt column at all
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0, refinance_target_debt=True)
    try:
        run_deal(df, deal)
        assert False, "expected a ValueError when refinance_target_debt=True but total_debt is missing"
    except ValueError as e:
        assert "T.NS" in str(e) and "total_debt" in str(e)
        print("PASS: refinance_target_debt=True with missing total_debt raises a clear, actionable error")


def test_missing_ticker_raises_clear_error():
    df = _make_universe()
    deal = DealInputs("NOTREAL.NS", "T.NS", premium_pct=0.20, cash_pct=0.5)
    try:
        run_deal(df, deal)
        assert False, "expected a ValueError for a ticker not in the universe"
    except ValueError as e:
        assert "NOTREAL.NS" in str(e)
        print("PASS: missing/unknown ticker raises a clear error naming the ticker")


def test_missing_required_field_raises_clear_error():
    df = _make_universe()
    df.loc[df.ticker == "A.NS", "return_on_equity"] = None
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=0.5)
    try:
        run_deal(df, deal)
        assert False, "expected a ValueError for a missing required field"
    except ValueError as e:
        assert "A.NS" in str(e) and "return_on_equity" in str(e)
        print("PASS: missing required field raises a clear error naming the ticker and field")


def test_sensitivity_sweep_runs_and_shows_expected_shape():
    df = _make_universe()
    table = run_sensitivity(df, "A.NS", "T.NS", premium_range=(0.10, 0.30), cash_pct_range=(0.0, 1.0))
    assert len(table) == 4  # 2 premiums x 2 cash mixes
    assert set(table.columns) >= {"premium_pct", "cash_pct", "pro_forma_eps", "accretion_dilution_pct", "is_accretive"}
    print("PASS: sensitivity sweep produces the expected grid shape and columns")


def test_sensitivity_sweep_with_debt_is_no_longer_premium_insensitive_at_full_cash():
    """Regression test for the exact real-world observation: with no debt
    cost modeled, a 100%-cash deal's EPS was IDENTICAL across every
    premium level (since higher premium just meant more cash, at zero
    cost). With debt financing enabled, a higher premium means more debt,
    more interest expense — so pro-forma EPS should now DECREASE as
    premium rises, even at cash_pct=1.0."""
    df = _make_universe()
    table = run_sensitivity(
        df, "A.NS", "T.NS",
        premium_range=(0.10, 0.20, 0.30, 0.40), cash_pct_range=(1.0,),
        debt_funded_pct=1.0, interest_rate=0.10, tax_rate=0.20,
    )
    eps_values = table.sort_values("premium_pct")["pro_forma_eps"].tolist()
    assert len(set(eps_values)) == len(eps_values), (
        "with debt financing enabled, EPS should differ across premium levels "
        "at 100% cash — a flat/identical EPS here would mean interest cost "
        "isn't actually being applied"
    )
    assert eps_values == sorted(eps_values, reverse=True), (
        "EPS should strictly DECREASE as premium rises at 100% cash with debt "
        "financing on, since a higher premium means more debt and more interest cost"
    )
    print(f"PASS: with debt financing enabled, 100%-cash EPS correctly decreases "
          f"as premium rises: {eps_values} (previously flat/identical without this feature)")


def test_find_optimal_terms_identifies_best_combo_correctly():
    """Hand-verifiable expectation, no debt cost modeled: pro_forma_net_income
    is constant (100,000 + 30,000 = 130,000) regardless of premium/cash mix,
    since there's no interest cost. new_shares_issued only depends on the
    STOCK portion, so higher cash_pct always means fewer new shares, hence
    higher (or equal) EPS — cash_pct=1.0 should always be part of the winner.
    At cash_pct=1.0 specifically, zero new shares are issued regardless of
    premium, so EPS is IDENTICAL (13.0, a 30% accretion) across every premium
    tested — pandas idxmax() picks the first such row, which is the lowest
    premium tested (0.10) paired with cash_pct=1.0."""
    df = _make_universe()
    result = find_optimal_terms(
        df, "A.NS", "T.NS",
        premium_range=(0.10, 0.20, 0.30), cash_pct_range=(0.0, 0.5, 1.0),
        debt_funded_pct=0.0,
    )
    best = result["best"]

    assert best["cash_pct"] == 1.0, (
        f"with no debt cost, 100% cash should always be part of the optimal "
        f"combination (it minimizes dilution) — got cash_pct={best['cash_pct']}"
    )
    assert best["premium_pct"] == 0.10, (
        f"among ties at cash_pct=1.0 (EPS is premium-invariant there with no "
        f"debt cost), idxmax should return the first (lowest premium) — "
        f"got premium_pct={best['premium_pct']}"
    )
    assert abs(best["accretion_dilution_pct"] - 30.0) < 0.01, (
        f"expected exactly 30.0% accretion (130,000/10,000=13.0 EPS vs "
        f"standalone 10.0 EPS), got {best['accretion_dilution_pct']}"
    )
    assert len(result["grid"]) == 9, "expected a 3x3 grid (9 combinations)"
    print(f"PASS: find_optimal_terms correctly identifies the best combination "
          f"(premium={best['premium_pct']:.0%}, cash={best['cash_pct']:.0%}, "
          f"accretion={best['accretion_dilution_pct']:.1f}%)")


def test_find_best_targets_ranks_correctly_by_hand_calc():
    """Hand-verifiable scenario: with cash_pct=1.0 (100% cash, no debt cost),
    ZERO new shares are issued regardless of which target is being evaluated,
    so pro_forma_shares is CONSTANT (10,000, the acquirer's own share count)
    across every candidate target. That makes accretion purely a function of
    each target's net income — the target contributing more net income
    always ranks higher, with no dilution channel to complicate the ordering.

    Setup (all at 20% premium, 100% cash):
      A.NS (acquirer): EPS=10.0, shares=10,000 (unchanged from earlier tests)
      T1.NS: NI=30,000  -> pro_forma_eps=(100,000+30,000)/10,000=13.0 -> +30% accretion
      T2.NS: NI=50,000  -> pro_forma_eps=(100,000+50,000)/10,000=15.0 -> +50% accretion (BEST)
      T3.NS: NI=2,000   -> pro_forma_eps=(100,000+2,000)/10,000=10.2 -> +2% accretion (WORST)
    Expected ranking: T2.NS > T1.NS > T3.NS
    """
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "T1.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0},
        {"ticker": "T2.NS", "sector": "y", "price": 20.0, "market_cap": 80_000.0,
         "return_on_equity": 0.25, "total_equity": 200_000.0},
        {"ticker": "T3.NS", "sector": "z", "price": 200.0, "market_cap": 400_000.0,
         "return_on_equity": 0.02, "total_equity": 100_000.0},
    ])

    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=1.0)

    assert list(result["target"]) == ["T2.NS", "T1.NS", "T3.NS"], (
        f"expected ranking T2 > T1 > T3 by net income (constant denominator "
        f"at 100% cash), got {list(result['target'])}"
    )
    assert list(result["rank"]) == [1, 2, 3]
    assert "A.NS" not in result["target"].values, "acquirer must not appear as its own candidate target"

    t2_acc = result.loc[result.target == "T2.NS", "accretion_dilution_pct"].iloc[0]
    t1_acc = result.loc[result.target == "T1.NS", "accretion_dilution_pct"].iloc[0]
    t3_acc = result.loc[result.target == "T3.NS", "accretion_dilution_pct"].iloc[0]
    assert abs(t2_acc - 50.0) < 0.01, f"expected T2 accretion of 50.0%, got {t2_acc}"
    assert abs(t1_acc - 30.0) < 0.01, f"expected T1 accretion of 30.0%, got {t1_acc}"
    assert abs(t3_acc - 2.0) < 0.01, f"expected T3 accretion of 2.0%, got {t3_acc}"

    print(f"PASS: find_best_targets correctly ranks candidates by hand-calculated "
          f"accretion (T2={t2_acc}% > T1={t1_acc}% > T3={t3_acc}%)")


def test_find_best_targets_skips_broken_candidates_gracefully():
    """A candidate missing required fields should be skipped and logged,
    not crash the whole scan — same error-isolation discipline used
    throughout the pipeline (ingestion, cleaning, etc.)."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "GOOD.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0},
        {"ticker": "BROKEN.NS", "sector": "y", "price": None, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0},  # missing price
    ])
    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=1.0)

    assert "GOOD.NS" in result["target"].values
    assert "BROKEN.NS" not in result["target"].values, "broken candidate should be skipped, not crash or appear with garbage data"
    assert len(result) == 1
    print("PASS: a candidate missing required fields is skipped gracefully, doesn't crash the scan")


def test_find_best_targets_top_n_slices_correctly():
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "T1.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0},
        {"ticker": "T2.NS", "sector": "y", "price": 20.0, "market_cap": 80_000.0,
         "return_on_equity": 0.25, "total_equity": 200_000.0},
        {"ticker": "T3.NS", "sector": "z", "price": 200.0, "market_cap": 400_000.0,
         "return_on_equity": 0.02, "total_equity": 100_000.0},
    ])
    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=1.0, top_n=2)
    assert len(result) == 2
    assert list(result["target"]) == ["T2.NS", "T1.NS"]
    print("PASS: top_n correctly slices to the requested number of best targets")


def test_negative_equity_target_raises_clear_error():
    """Regression test for a real gap found while testing: net_income =
    ROE x total_equity produces a sign-flipped, nonsensical value if
    total_equity is negative (e.g. a company with accumulated losses
    exceeding paid-in capital, like Vodafone Idea in the real universe).
    This must be caught explicitly, not silently produce a wrong-signed
    net income that could rank a broken company as a great target."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "NEGEQ.NS", "sector": "y", "price": 10.0, "market_cap": 50_000.0,
         "return_on_equity": -0.5, "total_equity": -20_000.0},  # negative equity
    ])
    try:
        _company_fundamentals(df, "NEGEQ.NS")
        assert False, "expected a ValueError for negative total_equity"
    except ValueError as e:
        assert "NEGEQ.NS" in str(e) and "equity" in str(e).lower()
        print("PASS: negative equity company raises a clear error instead of computing a sign-flipped net income")


def test_negative_equity_target_excluded_from_best_targets_scan():
    """Same real-world scenario, but verified end-to-end through
    find_best_targets: a negative-equity candidate should be skipped
    (logged), not crash the scan or appear in the ranked results."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "GOOD.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0},
        {"ticker": "NEGEQ.NS", "sector": "y", "price": 10.0, "market_cap": 50_000.0,
         "return_on_equity": -0.5, "total_equity": -20_000.0},
    ])
    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=0.5)
    assert "NEGEQ.NS" not in result["target"].values
    assert "GOOD.NS" in result["target"].values
    print("PASS: negative-equity candidate is excluded from best-targets results, doesn't crash the scan")


def test_oversized_target_excluded_by_default_size_filter():
    """Regression test for the real issue found: a mega-cap target (e.g.
    LIC relative to Coforge) can show absurd accretion (1000%+) purely from
    scale mismatch, which is structurally implausible as a real deal — a
    company overwhelmingly does not acquire something many times its own
    size. Default max_target_size_pct=1.0 should exclude any target whose
    market cap exceeds the acquirer's own."""
    df = pd.DataFrame([
        {"ticker": "SMALL.NS", "sector": "it", "price": 100.0, "market_cap": 100_000.0,
         "return_on_equity": 0.15, "total_equity": 50_000.0},
        {"ticker": "REASONABLE.NS", "sector": "it", "price": 50.0, "market_cap": 40_000.0,  # 40% of acquirer, plausible
         "return_on_equity": 0.10, "total_equity": 30_000.0},
        {"ticker": "MEGACAP.NS", "sector": "energy_power", "price": 500.0, "market_cap": 5_000_000.0,  # 50x the acquirer, implausible
         "return_on_equity": 0.20, "total_equity": 2_000_000.0},
    ])
    result = find_best_targets(df, "SMALL.NS", premium_pct=0.20, cash_pct=0.5)

    assert "MEGACAP.NS" not in result["target"].values, (
        "a target 50x the acquirer's market cap should be excluded as a "
        "structurally implausible deal by the default size filter"
    )
    assert "REASONABLE.NS" in result["target"].values, (
        "a target within a plausible size range (40% of acquirer) should NOT be excluded"
    )
    print("PASS: default size filter excludes a structurally implausible 50x-larger target, "
          "keeps a plausibly-sized one")


def test_relative_size_pct_column_present_for_transparency():
    """Even for a target that PASSES the size filter, relative_size_pct
    should be visible in the output — the filter shouldn't be an invisible
    cutoff; the person using this should be able to see and judge the
    size ratio for every kept target, not just trust a hidden threshold."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "T1.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0},
    ])
    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=0.5)
    assert "relative_size_pct" in result.columns
    t1_ratio = result.loc[result.target == "T1.NS", "relative_size_pct"].iloc[0]
    assert abs(t1_ratio - 20.0) < 0.01, f"expected 20.0% (200,000/1,000,000), got {t1_ratio}"
    print(f"PASS: relative_size_pct is visible for transparency even on a passing target ({t1_ratio}%)")


def test_max_target_size_pct_is_configurable():
    """The size filter should be adjustable, not a hardcoded cutoff —
    raising max_target_size_pct should let a previously-excluded larger
    target back in."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0},
        {"ticker": "BIG.NS", "sector": "y", "price": 100.0, "market_cap": 2_000_000.0,  # 200% of acquirer
         "return_on_equity": 0.10, "total_equity": 900_000.0},
    ])

    # With only these 2 companies and BIG.NS filtered out by the default
    # threshold, there are zero valid candidates left — correctly raises,
    # rather than silently returning an empty result.
    try:
        find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=0.5)
        assert False, "expected a ValueError since BIG.NS is the only candidate and gets filtered by default"
    except ValueError as e:
        assert "No valid targets" in str(e)

    # Raising the threshold should let BIG.NS back in.
    included_with_higher_threshold = find_best_targets(
        df, "A.NS", premium_pct=0.20, cash_pct=0.5, max_target_size_pct=3.0
    )
    assert "BIG.NS" in included_with_higher_threshold["target"].values
    print("PASS: max_target_size_pct is configurable — a 200%-of-acquirer target is correctly "
          "excluded (raising ValueError when it's the only candidate) at the default threshold, "
          "but included when the threshold is raised")


def test_acquisition_likelihood_merged_into_best_targets_when_columns_available():
    """When the scored universe has the bucket-score columns acquisition
    likelihood needs, find_best_targets should merge in each target's
    acquisition_likelihood_score/rank — tying the deal-mechanics ranking
    to the "is this genuinely a classic undervalued target" signal."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0,
         "valuation_bucket_score": 0.5, "quality_bucket_score": 0.9, "leverage_bucket_score": 0.8},
        {"ticker": "T1.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0,
         "valuation_bucket_score": 0.90, "quality_bucket_score": 0.10, "leverage_bucket_score": 0.85},
    ])
    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=0.5, include_acquisition_likelihood=True)

    assert "target_acquisition_likelihood_score" in result.columns
    t1_row = result[result.target == "T1.NS"].iloc[0]
    expected = 0.40 * 0.90 + 0.35 * (1 - 0.10) + 0.25 * 0.85  # same formula as acquisition_likelihood.py's DEFAULT_WEIGHTS
    assert abs(t1_row["target_acquisition_likelihood_score"] - expected) < 1e-9
    print(f"PASS: acquisition likelihood score correctly merged into best-targets output "
          f"(T1.NS score={t1_row['target_acquisition_likelihood_score']:.4f})")


def test_acquisition_likelihood_tie_in_can_be_disabled():
    """include_acquisition_likelihood=False should skip the merge entirely."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "sector": "x", "price": 100.0, "market_cap": 1_000_000.0,
         "return_on_equity": 0.20, "total_equity": 500_000.0,
         "valuation_bucket_score": 0.5, "quality_bucket_score": 0.9, "leverage_bucket_score": 0.8},
        {"ticker": "T1.NS", "sector": "y", "price": 50.0, "market_cap": 200_000.0,
         "return_on_equity": 0.10, "total_equity": 300_000.0,
         "valuation_bucket_score": 0.90, "quality_bucket_score": 0.10, "leverage_bucket_score": 0.85},
    ])
    result = find_best_targets(df, "A.NS", premium_pct=0.20, cash_pct=0.5, include_acquisition_likelihood=False)
    assert "target_acquisition_likelihood_score" not in result.columns
    print("PASS: include_acquisition_likelihood=False correctly skips the merge")


def test_multi_year_synergy_ramp_matches_hand_calc():
    """Hand calc (verified independently): 100% cash, 20% premium, 20%
    revenue synergy, no growth, ramp=(0.5, 1.0) over 2 years.
      Year 1: synergy = 30,000*0.20*0.5=3,000 -> pro_forma_ni=133,000 ->
              eps=13.3 -> accretion=(13.3-10)/10=33.0%
      Year 2: synergy = 30,000*0.20*1.0=6,000 -> pro_forma_ni=136,000 ->
              eps=13.6 -> accretion=36.0% (matches the single-period
              full-ramp revenue synergy test's 36.0% exactly, as expected
              since ramp=1.0 in year 2 IS the full run-rate case)
    """
    df = _make_universe()
    df["revenue_growth"] = 0.0  # isolate ramp effect from growth compounding
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0, revenue_synergy_pct=0.20)
    result = project_multi_year_accretion(df, deal, years=2, synergy_ramp=(0.5, 1.0))

    year1 = result[result.year == 1].iloc[0]
    year2 = result[result.year == 2].iloc[0]
    assert abs(year1["accretion_dilution_pct"] - 33.0) < 0.01, f"expected 33.0%, got {year1['accretion_dilution_pct']}"
    assert abs(year2["accretion_dilution_pct"] - 36.0) < 0.01, f"expected 36.0%, got {year2['accretion_dilution_pct']}"
    assert year1["synergy_ramp_pct"] == 50.0 and year2["synergy_ramp_pct"] == 100.0
    print(f"PASS: multi-year synergy ramp matches hand calc (Year 1={year1['accretion_dilution_pct']}%, "
          f"Year 2={year2['accretion_dilution_pct']}% — Year 2 at full ramp matches the single-period case)")


def test_multi_year_standalone_earnings_growth_matches_hand_calc():
    """Hand calc (verified independently): 100% cash, 20% premium, no
    synergy, acquirer grows 10%/yr, target grows 20%/yr.
      Year 1: acquirer_ni=110,000, target_ni=36,000 -> pro_forma_eps=14.6,
              acquirer_standalone_eps=11.0 -> accretion=32.73%
      Year 2: acquirer_ni=121,000, target_ni=43,200 -> pro_forma_eps=16.42,
              acquirer_standalone_eps=12.1 -> accretion=35.70%
    """
    df = _make_universe()
    df.loc[df.ticker == "A.NS", "revenue_growth"] = 0.10
    df.loc[df.ticker == "T.NS", "revenue_growth"] = 0.20
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0)
    result = project_multi_year_accretion(df, deal, years=2, synergy_ramp=(1.0, 1.0))

    year1 = result[result.year == 1].iloc[0]
    year2 = result[result.year == 2].iloc[0]
    assert abs(year1["accretion_dilution_pct"] - 32.73) < 0.01, f"expected 32.73%, got {year1['accretion_dilution_pct']}"
    assert abs(year2["accretion_dilution_pct"] - 35.70) < 0.01, f"expected 35.70%, got {year2['accretion_dilution_pct']}"
    print(f"PASS: standalone earnings growth compounds correctly (Year 1={year1['accretion_dilution_pct']}%, "
          f"Year 2={year2['accretion_dilution_pct']}%)")


def test_missing_revenue_growth_defaults_to_zero_not_error():
    """A company with no revenue_growth data shouldn't block the multi-year
    projection — it should default to flat (0%) growth rather than raising."""
    df = _make_universe()  # no revenue_growth column at all
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0)
    result = project_multi_year_accretion(df, deal, years=2)
    assert len(result) == 2
    print("PASS: missing revenue_growth data defaults to flat growth, doesn't block the projection")


def test_multi_year_ramp_extends_last_value_if_years_exceed_ramp_length():
    """If years > len(synergy_ramp), the last ramp value should be used
    for all remaining years (full run-rate), not raise an index error."""
    df = _make_universe()
    df["revenue_growth"] = 0.0
    deal = DealInputs("A.NS", "T.NS", premium_pct=0.20, cash_pct=1.0, revenue_synergy_pct=0.20)
    result = project_multi_year_accretion(df, deal, years=4, synergy_ramp=(0.5, 1.0))
    assert len(result) == 4
    assert result.iloc[2]["synergy_ramp_pct"] == 100.0  # year 3, beyond ramp schedule, holds at last value
    assert result.iloc[3]["synergy_ramp_pct"] == 100.0  # year 4, same
    print("PASS: years beyond the ramp schedule length correctly hold at the last (full run-rate) ramp value")


if __name__ == "__main__":
    test_fundamentals_derivation_matches_hand_calc()
    test_deal_math_matches_hand_calc()
    test_all_cash_deal_has_no_dilution_from_new_shares()
    test_debt_financed_cash_deducts_after_tax_interest()
    test_default_synergy_pct_is_zero_and_unchanged()
    test_revenue_synergy_uplifts_target_net_income()
    test_cost_synergy_is_tax_effected_on_combined_income()
    test_full_stack_synergies_plus_debt_financing_combine_correctly()
    test_default_behavior_unchanged_when_debt_funded_pct_is_zero()
    test_avg_price_basis_uses_fetched_average_not_spot()
    test_avg_price_fetch_failure_raises_clear_error()
    test_refinance_target_debt_adds_to_new_debt_and_interest()
    test_refinance_target_debt_without_total_debt_raises_clear_error()
    test_missing_ticker_raises_clear_error()
    test_missing_required_field_raises_clear_error()
    test_sensitivity_sweep_runs_and_shows_expected_shape()
    test_sensitivity_sweep_with_debt_is_no_longer_premium_insensitive_at_full_cash()
    test_find_optimal_terms_identifies_best_combo_correctly()
    test_find_best_targets_ranks_correctly_by_hand_calc()
    test_find_best_targets_skips_broken_candidates_gracefully()
    test_find_best_targets_top_n_slices_correctly()
    test_negative_equity_target_raises_clear_error()
    test_negative_equity_target_excluded_from_best_targets_scan()
    test_oversized_target_excluded_by_default_size_filter()
    test_relative_size_pct_column_present_for_transparency()
    test_max_target_size_pct_is_configurable()
    test_acquisition_likelihood_merged_into_best_targets_when_columns_available()
    test_acquisition_likelihood_tie_in_can_be_disabled()
    test_multi_year_synergy_ramp_matches_hand_calc()
    test_multi_year_standalone_earnings_growth_matches_hand_calc()
    test_missing_revenue_growth_defaults_to_zero_not_error()
    test_multi_year_ramp_extends_last_value_if_years_exceed_ramp_length()
    print("\nAll accretion/dilution mock tests passed.")
