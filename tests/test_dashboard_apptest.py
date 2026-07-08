"""
test_dashboard_apptest.py — Headless execution test for the Streamlit
dashboard, using Streamlit's own AppTest framework.

Why this exists: a pure `import src.dashboard` only proves the file
compiles — it doesn't execute any of the tab logic (everything inside
`with tab_x:` blocks, button clicks, slider-triggered code paths). This
test actually RUNS the app, including clicking the sidebar button and
moving sliders to trigger code paths that only fire at non-default
values (e.g. the multi-year synergy table only renders when the synergy
sliders are above 0%). This is exactly the kind of test that caught a
real deprecated-parameter warning and confirmed the full tab surface
during the dashboard's visual redesign — worth keeping as a permanent
regression guard, not a one-off check.

Run with: python3 tests/test_dashboard_apptest.py
"""

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _write_fixture_data(data_dir: Path):
    (data_dir / "processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"ticker": "TCS.NS", "sector": "it", "rank": 1, "sector_rank": 1, "composite_score": 0.85,
         "trailing_pe": 14.5, "ev_to_ebitda": 10.0, "return_on_equity": 0.48, "net_margin": 0.18,
         "debt_to_equity_ratio": 0.10, "revenue_growth": 0.09, "price": 3500.0, "market_cap": 12_000_000.0,
         "total_equity": 1_000_000.0, "total_debt": 100_000.0},
        {"ticker": "ACC.NS", "sector": "cement", "rank": 2, "sector_rank": 1, "composite_score": 0.72,
         "trailing_pe": 12.0, "ev_to_ebitda": 9.0, "return_on_equity": 0.11, "net_margin": 0.08,
         "debt_to_equity_ratio": 0.02, "revenue_growth": 0.32, "price": 1332.0, "market_cap": 250_000.0,
         "total_equity": 205_000.0, "total_debt": 4_000.0},
        {"ticker": "INFY.NS", "sector": "it", "rank": 3, "sector_rank": 2, "composite_score": 0.61,
         "trailing_pe": 13.0, "ev_to_ebitda": 8.5, "return_on_equity": 0.31, "net_margin": 0.16,
         "debt_to_equity_ratio": 0.09, "revenue_growth": 0.07, "price": 1985.0, "market_cap": 398_000.0,
         "total_equity": 9_786_000.0, "total_debt": 967_000.0},
    ]).to_csv(data_dir / "processed" / "scored_universe.csv", index=False)


def test_dashboard_loads_with_no_exceptions():
    from streamlit.testing.v1 import AppTest
    import tempfile

    original_cwd = os.getcwd()
    dashboard_path = str(Path(__file__).resolve().parent.parent / "src" / "dashboard.py")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            _write_fixture_data(Path(tmp) / "data")

            at = AppTest.from_file(dashboard_path)
            at.run(timeout=30)

            assert not at.exception, f"dashboard raised on initial load: {at.exception}"
            assert len(at.tabs) == 6, f"expected 6 tabs, found {len(at.tabs)}"
            print("PASS: dashboard loads with no exceptions across all 6 tabs")
        finally:
            os.chdir(original_cwd)


def test_generate_pdf_button_does_not_raise():
    from streamlit.testing.v1 import AppTest
    import tempfile

    original_cwd = os.getcwd()
    dashboard_path = str(Path(__file__).resolve().parent.parent / "src" / "dashboard.py")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            _write_fixture_data(Path(tmp) / "data")

            at = AppTest.from_file(dashboard_path)
            at.run(timeout=30)
            assert len(at.sidebar.button) > 0, "expected a button in the sidebar"

            at.sidebar.button[0].click().run(timeout=30)
            assert not at.exception, f"clicking Generate PDF raised: {at.exception}"
            print("PASS: Generate PDF button click does not raise")
        finally:
            os.chdir(original_cwd)


def test_multi_year_synergy_code_path_does_not_raise():
    """The multi-year projection table only renders when a synergy slider
    is above 0% — this is exactly the kind of conditionally-rendered code
    path that only running (not just importing) the app can verify."""
    from streamlit.testing.v1 import AppTest
    import tempfile

    original_cwd = os.getcwd()
    dashboard_path = str(Path(__file__).resolve().parent.parent / "src" / "dashboard.py")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            _write_fixture_data(Path(tmp) / "data")

            at = AppTest.from_file(dashboard_path)
            at.run(timeout=30)

            revenue_synergy_slider = next(
                (s for s in at.slider if s.label and "Revenue synergy" in s.label), None
            )
            assert revenue_synergy_slider is not None, "expected to find the revenue synergy slider"

            revenue_synergy_slider.set_value(20).run(timeout=30)
            assert not at.exception, f"multi-year synergy code path raised: {at.exception}"
            print("PASS: multi-year synergy table code path (only active at non-zero synergy) does not raise")
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    test_dashboard_loads_with_no_exceptions()
    test_generate_pdf_button_does_not_raise()
    test_multi_year_synergy_code_path_does_not_raise()
    print("\nAll dashboard AppTest checks passed.")
