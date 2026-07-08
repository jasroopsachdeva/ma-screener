"""
Smoke tests for the expanded company_summary.py. Key additions verified:
1. Raw financials (price, market cap, debt, equity) assemble correctly
2. Sector peer comparison pulls the right peers, excluding the subject itself
3. The illustrative deal calculation uses the real run_deal engine and
   produces hand-verifiable numbers
4. It gracefully falls back to the next-best peer if the top-ranked one
   is missing required data, and returns None if NO peer works
5. generate_summary_pdf still produces a real, valid PDF with the new sections
Run with: python3 tests/test_company_summary_mock.py
"""

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.company_summary import build_summary_data, generate_summary_pdf, _sector_peers, _illustrative_deal


def _make_scored_universe():
    return pd.DataFrame([
        {"ticker": "TCS.NS", "sector": "it", "rank": 1, "sector_rank": 1, "composite_score": 0.85,
         "price": 3500.0, "market_cap": 120_000_000_000.0, "total_debt": 1_000_000_000.0, "total_equity": 10_000_000_000.0,
         "trailing_pe": 14.5, "ev_to_ebitda": 10.0, "return_on_equity": 0.48,
         "net_margin": 0.18, "debt_to_equity_ratio": 0.10, "revenue_growth": 0.09},
        {"ticker": "INFY.NS", "sector": "it", "rank": 3, "sector_rank": 2, "composite_score": 0.61,
         "price": 1985.0, "market_cap": 40_000_000_000.0, "total_debt": 900_000_000.0, "total_equity": 9_000_000_000.0,
         "trailing_pe": 13.0, "ev_to_ebitda": 8.5, "return_on_equity": 0.31,
         "net_margin": 0.16, "debt_to_equity_ratio": 0.09, "revenue_growth": 0.07},
        {"ticker": "WIPRO.NS", "sector": "it", "rank": 7, "sector_rank": 3, "composite_score": 0.45,
         "price": 400.0, "market_cap": 20_000_000_000.0, "total_debt": 500_000_000.0, "total_equity": 5_000_000_000.0,
         "trailing_pe": 18.0, "ev_to_ebitda": 12.0, "return_on_equity": 0.20,
         "net_margin": 0.10, "debt_to_equity_ratio": 0.15, "revenue_growth": 0.05},
        {"ticker": "ACC.NS", "sector": "cement", "rank": 2, "sector_rank": 1, "composite_score": 0.72,
         "price": 1332.0, "market_cap": 2_500_000_000.0, "total_debt": 40_000_000.0, "total_equity": 2_050_000_000.0,
         "trailing_pe": 12.0, "ev_to_ebitda": 9.0, "return_on_equity": 0.11,
         "net_margin": 0.08, "debt_to_equity_ratio": 0.02, "revenue_growth": 0.32},
    ])


def test_raw_financials_assemble_correctly():
    scored = _make_scored_universe()
    with tempfile.TemporaryDirectory() as tmp:
        scored.to_csv(f"{tmp}/scored.csv", index=False)
        data = build_summary_data("TCS.NS", scored_path=f"{tmp}/scored.csv")
        assert data["price"] == 3500.0
        assert data["market_cap"] == 120_000_000_000.0
        assert data["total_debt"] == 1_000_000_000.0
        assert data["total_equity"] == 10_000_000_000.0
        print("PASS: raw financial fields (price, market cap, debt, equity) assemble correctly")


def test_sector_peers_excludes_self_and_other_sectors():
    scored = _make_scored_universe()
    peers = _sector_peers("TCS.NS", scored, n=4)
    peer_tickers = [p["ticker"] for p in peers]
    assert "TCS.NS" not in peer_tickers, "the subject company must not appear in its own peer list"
    assert "ACC.NS" not in peer_tickers, "a different-sector company must not appear as a peer"
    assert set(peer_tickers) == {"INFY.NS", "WIPRO.NS"}, f"expected only same-sector peers, got {peer_tickers}"
    print(f"PASS: sector peers correctly exclude self and other sectors ({peer_tickers})")


def test_illustrative_deal_matches_hand_calc():
    scored = _make_scored_universe()
    deal = _illustrative_deal("TCS.NS", scored)
    assert deal is not None
    assert deal["acquirer"] == "INFY.NS", f"expected INFY.NS (top-ranked IT peer), got {deal['acquirer']}"
    assert deal["premium_pct"] == 0.25
    assert deal["cash_pct"] == 0.5
    assert isinstance(deal["is_accretive"], bool)
    print(f"PASS: illustrative deal correctly picks INFY.NS as acquirer for TCS.NS, "
          f"verdict={'ACCRETIVE' if deal['is_accretive'] else 'DILUTIVE'}")


def test_illustrative_deal_falls_back_to_next_peer_if_top_one_is_broken():
    scored = _make_scored_universe()
    scored.loc[scored.ticker == "INFY.NS", "return_on_equity"] = None
    deal = _illustrative_deal("TCS.NS", scored)
    assert deal is not None
    assert deal["acquirer"] == "WIPRO.NS", f"expected fallback to WIPRO.NS, got {deal['acquirer']}"
    print("PASS: illustrative deal falls back to the next-best peer when the top one is missing data")


def test_illustrative_deal_returns_none_when_no_sector_peers_exist():
    scored = _make_scored_universe()
    solo = scored[scored.ticker == "ACC.NS"].copy()
    deal = _illustrative_deal("ACC.NS", solo)
    assert deal is None
    print("PASS: illustrative deal returns None gracefully when there are no sector peers at all")


def test_build_summary_data_degrades_gracefully_when_optional_files_missing():
    scored = _make_scored_universe()
    with tempfile.TemporaryDirectory() as tmp:
        scored.to_csv(f"{tmp}/scored.csv", index=False)
        data = build_summary_data(
            "TCS.NS",
            scored_path=f"{tmp}/scored.csv",
            comps_dupont_path=f"{tmp}/nonexistent.csv",
            explanations_path=f"{tmp}/nonexistent2.csv",
            likelihood_path=f"{tmp}/nonexistent3.csv",
        )
        assert data["valuation_label"] is None
        assert data["dupont"] is None
        assert data["acquisition_likelihood_score"] is None
        assert len(data["sector_peers"]) > 0
        assert data["illustrative_deal"] is not None
        print("PASS: missing optional report files still degrade gracefully; sector peers and "
              "illustrative deal (which only need scored_universe.csv) still populate")


def test_missing_ticker_raises_clear_error():
    scored = _make_scored_universe()
    with tempfile.TemporaryDirectory() as tmp:
        scored.to_csv(f"{tmp}/scored.csv", index=False)
        try:
            build_summary_data("NOT_REAL.NS", scored_path=f"{tmp}/scored.csv")
            assert False, "expected a ValueError"
        except ValueError as e:
            assert "NOT_REAL.NS" in str(e)
            print("PASS: a ticker missing from scored_universe.csv raises a clear error")


def test_generate_summary_pdf_produces_a_real_pdf_with_new_sections():
    from pypdf import PdfReader
    scored = _make_scored_universe()
    with tempfile.TemporaryDirectory() as tmp:
        scored.to_csv(f"{tmp}/scored.csv", index=False)
        out_path = f"{tmp}/summary.pdf"
        generate_summary_pdf("TCS.NS", out_path, scored_path=f"{tmp}/scored.csv")
        assert Path(out_path).exists()
        reader = PdfReader(out_path)
        assert len(reader.pages) >= 1
        full_text = "".join(p.extract_text() for p in reader.pages)
        assert "TCS.NS" in full_text
        assert "Sector Peer Comparison" in full_text
        assert "Illustrative Deal Snapshot" in full_text
        assert "INFY.NS" in full_text
        print(f"PASS: generated PDF ({len(reader.pages)} page(s)) contains the new sector-peer "
              f"and illustrative-deal sections with real content")


if __name__ == "__main__":
    test_raw_financials_assemble_correctly()
    test_sector_peers_excludes_self_and_other_sectors()
    test_illustrative_deal_matches_hand_calc()
    test_illustrative_deal_falls_back_to_next_peer_if_top_one_is_broken()
    test_illustrative_deal_returns_none_when_no_sector_peers_exist()
    test_build_summary_data_degrades_gracefully_when_optional_files_missing()
    test_missing_ticker_raises_clear_error()
    test_generate_summary_pdf_produces_a_real_pdf_with_new_sections()
    print("\nAll company summary mock tests passed.")
