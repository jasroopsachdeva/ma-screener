"""
test_integration_golden_snapshot.py — End-to-end pipeline test against a
committed, fixed dataset (tests/fixtures/golden_raw_data/).

Why this exists: every other test file mocks a single function in
isolation. This one runs cleaning -> scoring -> comps_dupont ->
explainability -> baseline_comparison as a REAL chain, against REAL
numbers (the actual ACC/TCS/INFY/HCLTECH/SHREECEM data hand-verified
throughout this project's development, plus LTIM's real empty-payload
failure) — without depending on yfinance being reachable or well-behaved
at test time.

This is what should run in CI on every push: deterministic, no network
flakiness, and it exercises the exact edge cases (broken ev_to_ebitda,
missing ROE, a fully-failed ticker) that real bugs were found in during
development — so a future regression in how any of these are handled
gets caught here, not three modules downstream in production.

Run with: python3 tests/test_integration_golden_snapshot.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import cleaning, scoring, comps_dupont, explainability, baseline_comparison

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "golden_raw_data"


def _make_temp_config(tmpdir: Path) -> Path:
    """Minimal config pointing at the golden fixture universe, written to
    a temp dir so this test never touches the real config/universe.yaml."""
    import yaml
    config = {
        "universe": [],
        "data": {"cache_dir": str(tmpdir / "raw"), "cache_expiry_days": 1},
        "scoring": {"weights": {"valuation": 0.30, "quality": 0.30, "leverage": 0.20, "growth": 0.20}},
        "output": {"top_n_shortlist": 5},
    }
    config_path = tmpdir / "universe.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_full_pipeline_runs_clean_against_golden_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        raw_dir = tmpdir / "raw"
        processed_dir = tmpdir / "processed"
        raw_dir.mkdir()

        for f in FIXTURES_DIR.glob("*.json"):
            shutil.copy(f, raw_dir / f.name)

        config_path = _make_temp_config(tmpdir)

        cleaned = cleaning.clean_universe(raw_dir=str(raw_dir), output_dir=str(processed_dir))
        assert not cleaned.empty, "cleaning produced zero rows from the golden snapshot"
        assert "LTIM.NS" not in cleaned["ticker"].values, "LTIM.NS's real empty-payload failure should still be excluded"
        assert "ACC.NS" in cleaned["ticker"].values
        assert "HCLTECH.NS" in cleaned["ticker"].values
        hcltech_row = cleaned[cleaned.ticker == "HCLTECH.NS"].iloc[0]
        assert pd.isna(hcltech_row["ev_to_ebitda"]), "HCLTECH's known-broken ev_to_ebitda (999.789) must still be nulled by cleaning"
        assert pd.notna(hcltech_row["trailing_pe"]), "HCLTECH's good fields must survive despite the one broken field"
        print(f"PASS: cleaning stage — {len(cleaned)}/7 golden tickers retained, "
              f"LTIM excluded, HCLTECH's broken field nulled but row kept")

        import src.scoring as scoring_module
        original_load_config = scoring_module.load_config
        scoring_module.load_config = lambda config_path=str(config_path): original_load_config(config_path)
        try:
            scored = scoring.score_universe(
                input_path=str(processed_dir / "cleaned_universe.csv"),
                output_dir=str(processed_dir),
            )
        finally:
            scoring_module.load_config = original_load_config

        assert not scored.empty
        assert scored["rank"].is_unique
        assert scored["composite_score"].between(0, 1).all()
        assert "SHREECEM.NS" in scored["ticker"].values, "SHREECEM (missing ROE but otherwise valid) should still be scored"
        print(f"PASS: scoring stage — {len(scored)} companies ranked, all composite scores in [0,1]")

        comps = comps_dupont.build_report(
            scored_path=str(processed_dir / "scored_universe.csv"),
            output_dir=str(processed_dir),
            top_n=5,
        )
        assert not comps.empty
        shreecem_dupont = comps[comps.ticker == "SHREECEM.NS"]
        if len(shreecem_dupont) > 0:
            assert pd.notna(shreecem_dupont.iloc[0]["dupont_implied_roe"]), (
                "SHREECEM's DuPont-implied ROE should recover even though reported ROE was missing — "
                "this is the exact real bug this project hit and fixed"
            )
            print("PASS: comps+DuPont stage — SHREECEM's ROE correctly recovers via DuPont despite missing reported ROE")
        else:
            print("PASS: comps+DuPont stage completed (SHREECEM not in top 5 of this small fixture set)")

        explanations = explainability.build_explanations(
            scored_path=str(processed_dir / "scored_universe.csv"),
            output_dir=str(processed_dir),
            top_n=5,
        )
        assert not explanations.empty
        assert all(isinstance(s, str) and len(s) > 0 for s in explanations["summary"])
        print(f"PASS: explainability stage — {len(explanations)} plain-English summaries generated")

        comparison = baseline_comparison.build_comparison_report(
            scored_path=str(processed_dir / "scored_universe.csv"),
            output_dir=str(processed_dir),
            top_n=5,
        )
        composite_only = set(comparison["caught_by_composite_only"])
        naive_only = set(comparison["caught_by_naive_only"])
        overlap = set(comparison["overlap_tickers"])
        assert composite_only.isdisjoint(naive_only), "no ticker should be in both divergence sets"
        assert composite_only.isdisjoint(overlap) and naive_only.isdisjoint(overlap)
        print(f"PASS: baseline comparison stage — internally consistent "
              f"(overlap={len(overlap)}, composite-only={len(composite_only)}, naive-only={len(naive_only)})")

        print("\nFull pipeline ran end-to-end against the golden snapshot with no live network dependency.")


if __name__ == "__main__":
    test_full_pipeline_runs_clean_against_golden_snapshot()
    print("\nAll golden snapshot integration tests passed.")
