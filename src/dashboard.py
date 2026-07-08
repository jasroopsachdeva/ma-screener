"""
dashboard.py — Streamlit dashboard for the M&A screener.

Visual direction: a deal-screening terminal, not a generic SaaS dashboard —
grounded in the subject (capital markets, deal desks, trading terminals).
Deep navy-charcoal base, a dual accent (teal for navigation/actions, gold
for key numbers), semantic emerald/coral reserved strictly for accretive/
dilutive. Fraunces (serif) for headers, Inter for body/data density, IBM
Plex Mono for tickers and scores — a genuine terminal cue, not decoration.
The signature element is the hero "ticker strip": real numbers pulled live
from the screener's own output, not a marketing banner.

Loads the outputs already produced by the pipeline (scored_universe.csv,
comps_dupont_report.csv, explanations.csv, baseline_comparison.csv) and
presents them, plus a live accretion/dilution calculator so the
sensitivity toggle is genuinely interactive rather than a static table.

Run with:
    streamlit run src/dashboard.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

# Streamlit adds the script's own directory (src/) to sys.path, not the
# project root — so `from src.accretion_dilution import ...` can't resolve
# on its own. Explicitly add the project root (one level up from this
# file) so the same `src.` import style works whether this module is run
# via `python -m src.dashboard` or `streamlit run src/dashboard.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.accretion_dilution import DealInputs, run_deal, find_optimal_terms, find_best_targets, project_multi_year_accretion

st.set_page_config(page_title="M&A Screener", layout="wide", page_icon="◆")

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# Palette (named, deliberate — not a generic "dark mode + one accent"):
#   bg-base       #0B1120  deep navy-charcoal, not pure black
#   bg-surface    #141B2D  card/surface background
#   border        #232D45  hairline borders on cards/dividers
#   text-primary  #E8ECF4  near-white, cool tone
#   text-muted    #8B96AE  secondary text, captions
#   accent-teal   #22D3B0  primary: navigation, active states, primary actions
#   accent-gold   #E8A33D  secondary: key numbers, scores, highlights
#   positive      #34D399  semantic only — accretive / cheap-vs-sector / good
#   negative      #F0546B  semantic only — dilutive / rich-vs-sector / bad
#
# Type: Fraunces (serif, display) for headers and the wordmark — Inter for
# body/UI density — IBM Plex Mono for tickers, ranks, and scores, the one
# genuinely "terminal" cue in the system.

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg-base: #0B1120;
    --bg-surface: #141B2D;
    --bg-surface-hover: #1A2338;
    --border-subtle: #232D45;
    --text-primary: #E8ECF4;
    --text-muted: #8B96AE;
    --accent-teal: #22D3B0;
    --accent-gold: #E8A33D;
    --positive: #34D399;
    --negative: #F0546B;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

[data-testid="stAppViewContainer"] {
    background-color: var(--bg-base);
}
[data-testid="stHeader"] {
    background-color: var(--bg-base);
    border-bottom: 1px solid var(--border-subtle);
}
.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

h1, h2, h3 {
    font-family: 'Fraunces', Georgia, serif !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.01em;
}
h1 { font-weight: 600 !important; }
h2, h3 { font-weight: 500 !important; }

p, span, div, label {
    color: var(--text-primary);
}
[data-testid="stCaptionContainer"], .stCaption, small {
    color: var(--text-muted) !important;
}

/* Wordmark */
.wordmark {
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-weight: 500;
    font-size: 2.4rem;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    margin-bottom: 0;
    line-height: 1.1;
}
.wordmark span { color: var(--accent-teal); font-style: normal; }
.subtitle {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin-top: 4px;
    margin-bottom: 1.4rem;
}

/* Hero ticker strip — the signature element, built from real screener data */
.hero-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0;
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.6rem;
}
.hero-stat {
    flex: 1;
    min-width: 140px;
    padding: 16px 20px;
    border-right: 1px solid var(--border-subtle);
}
.hero-stat:last-child { border-right: none; }
.hero-label {
    display: block;
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 4px;
}
.hero-value {
    display: block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: var(--accent-gold);
    line-height: 1.2;
}
.hero-value.teal { color: var(--accent-teal); }

/* Badges (verdicts, labels) */
.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 999px;
    border: 1px solid;
    letter-spacing: 0.03em;
}
.badge-positive { color: var(--positive); border-color: var(--positive); background-color: rgba(52,211,153,0.10); }
.badge-negative { color: var(--negative); border-color: var(--negative); background-color: rgba(240,84,107,0.10); }
.badge-neutral  { color: var(--accent-gold); border-color: var(--accent-gold); background-color: rgba(232,163,61,0.10); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] h3 { font-size: 1.1rem !important; }

/* Tabs — pill nav, teal active state. Inactive tabs need a REAL visible
   boundary at rest (subtle background + border), not transparent — a
   fully transparent inactive state reads as plain running text rather
   than distinct clickable buttons. */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    background-color: var(--bg-base);
    padding: 6px;
    border-radius: 12px;
    border: 1px solid var(--border-subtle);
}
.stTabs [data-baseweb="tab"] {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 40px;
    padding: 0 18px;
    margin: 0;
    border-radius: 8px;
    color: var(--text-muted);
    font-weight: 500;
    font-size: 0.88rem;
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    white-space: nowrap;
    transition: background-color 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: var(--bg-surface-hover);
    color: var(--text-primary);
    border-color: var(--accent-teal);
}
.stTabs [aria-selected="true"] {
    background-color: var(--accent-teal) !important;
    color: var(--bg-base) !important;
    border-color: var(--accent-teal) !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
.stTabs [data-baseweb="tab-border"] { display: none; }

/* Cards — bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 0.4rem 0.2rem;
}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    background-color: var(--accent-teal);
    color: var(--bg-base);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background-color: #1BBE9C;
    color: var(--bg-base);
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent-gold);
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted);
}

/* Sliders */
.stSlider [data-baseweb="slider"] > div > div { background-color: var(--accent-teal) !important; }

/* Expanders */
[data-testid="stExpander"] {
    background-color: var(--bg-surface-hover);
    border: 1px solid var(--border-subtle);
    border-radius: 10px;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border-subtle);
}

/* Alerts */
[data-testid="stSuccess"], [data-testid="stInfo"], [data-testid="stWarning"], [data-testid="stError"] {
    border-radius: 10px;
}
</style>
"""


def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def badge_html(text: str, kind: str = "neutral") -> str:
    cls = {"positive": "badge-positive", "negative": "badge-negative", "neutral": "badge-neutral"}.get(kind, "badge-neutral")
    return f'<span class="badge {cls}">{text}</span>'


def render_hero(scored: pd.DataFrame, scored_path: str = "data/processed/scored_universe.csv"):
    """The signature element: a ticker-strip hero built entirely from the
    screener's own live data, styled like a trading terminal's top strip —
    not a decorative banner."""
    total_companies = len(scored)
    sectors = scored["sector"].nunique() if "sector" in scored.columns else 0
    top_pick = scored.sort_values("rank").iloc[0] if "rank" in scored.columns and not scored.empty else None
    avg_score = scored["composite_score"].mean() if "composite_score" in scored.columns else None

    try:
        updated = datetime.fromtimestamp(os.path.getmtime(scored_path)).strftime("%b %d, %H:%M")
    except OSError:
        updated = "—"

    top_pick_str = f"{top_pick['ticker']}" if top_pick is not None else "—"
    avg_score_str = f"{avg_score:.3f}" if avg_score is not None and pd.notna(avg_score) else "—"

    html = f"""
    <div class="hero-strip">
        <div class="hero-stat"><span class="hero-label">Universe</span><span class="hero-value teal">{total_companies}</span></div>
        <div class="hero-stat"><span class="hero-label">Sectors</span><span class="hero-value teal">{sectors}</span></div>
        <div class="hero-stat"><span class="hero-label">Top Pick</span><span class="hero-value">{top_pick_str}</span></div>
        <div class="hero-stat"><span class="hero-label">Avg Composite</span><span class="hero-value">{avg_score_str}</span></div>
        <div class="hero-stat"><span class="hero-label">Updated</span><span class="hero-value teal">{updated}</span></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# --- Styler helpers: CSS-only cell coloring (st.dataframe cannot render
# arbitrary HTML in cells, only Styler-applied CSS properties) ---

def _lerp_hex(t: float, low: str, high: str) -> str:
    t = max(0.0, min(1.0, t))
    lo = tuple(int(low.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    hi = tuple(int(high.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    rgb = tuple(round(lo[i] + (hi[i] - lo[i]) * t) for i in range(3))
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _score_cell_style(val, vmin=0.0, vmax=1.0):
    if pd.isna(val):
        return ""
    t = 0.0 if vmax == vmin else (val - vmin) / (vmax - vmin)
    bg = _lerp_hex(t, "#1A2338", "#22D3B0")
    text_color = "#0B1120" if t > 0.55 else "#E8ECF4"
    return f"background-color: {bg}; color: {text_color}; font-weight: 600;"


def _bool_cell_style(val):
    if val is True:
        return "background-color: rgba(52,211,153,0.16); color: #34D399; font-weight: 700;"
    if val is False:
        return "background-color: rgba(240,84,107,0.16); color: #F0546B; font-weight: 700;"
    return ""


def _signed_pct_cell_style(val):
    if pd.isna(val):
        return ""
    return "color: #34D399; font-weight: 700;" if val > 0 else "color: #F0546B; font-weight: 700;"


def _apply_cellwise(styler, func, subset):
    """pandas 2.1+ renamed Styler.applymap to Styler.map; support both."""
    try:
        return styler.map(func, subset=subset)
    except AttributeError:
        return styler.applymap(func, subset=subset)


def styled_table(df: pd.DataFrame, format_map: dict = None, score_cols=(), bool_cols=(), signed_pct_cols=(), mono_cols=()):
    """Wraps the common pattern used throughout this dashboard: format
    numbers, then color-code score/verdict/accretion columns so the table
    reads at a glance rather than as a wall of plain numbers."""
    styler = df.style
    if format_map:
        styler = styler.format(format_map)
    for col in score_cols:
        if col in df.columns:
            vmin, vmax = df[col].min(), df[col].max()
            styler = _apply_cellwise(styler, lambda v, lo=vmin, hi=vmax: _score_cell_style(v, lo, hi), subset=[col])
    for col in bool_cols:
        if col in df.columns:
            styler = _apply_cellwise(styler, _bool_cell_style, subset=[col])
    for col in signed_pct_cols:
        if col in df.columns:
            styler = _apply_cellwise(styler, _signed_pct_cell_style, subset=[col])
    for col in mono_cols:
        if col in df.columns:
            styler = styler.set_properties(subset=[col], **{"font-family": "'IBM Plex Mono', monospace"})
    return styler


@st.cache_data(ttl=3600)
def load_data():
    scored = pd.read_csv("data/processed/scored_universe.csv")
    try:
        comps_dupont = pd.read_csv("data/processed/comps_dupont_report.csv")
    except FileNotFoundError:
        comps_dupont = None
    try:
        explanations = pd.read_csv("data/processed/explanations.csv")
    except FileNotFoundError:
        explanations = None
    try:
        baseline = pd.read_csv("data/processed/baseline_comparison.csv")
    except FileNotFoundError:
        baseline = None
    return scored, comps_dupont, explanations, baseline


@st.cache_data(show_spinner="Scanning candidate targets...")
def cached_find_best_targets(scored_df, acquirer, premium_pct, cash_pct, debt_funded_pct,
                              interest_rate, tax_rate, max_target_size_pct):
    """Cached wrapper around the ~200-company scan — without this,
    every single slider tweak (including ones unrelated to the scan,
    since Streamlit reruns the whole script on any interaction) would
    re-run the full scan from scratch. Cache key is the full set of
    scan parameters, so a genuinely different acquirer/terms combo still
    triggers a fresh computation."""
    return find_best_targets(
        scored_df, acquirer,
        premium_pct=premium_pct, cash_pct=cash_pct,
        debt_funded_pct=debt_funded_pct, interest_rate=interest_rate, tax_rate=tax_rate,
        max_target_size_pct=max_target_size_pct,
    )


@st.cache_data(show_spinner="Scoring acquisition likelihood...")
def cached_score_acquisition_likelihood(scored_df):
    """Cached wrapper — recomputing this on every sector-filter or
    top-N slider tweak in the Acquisition Likelihood tab is wasted work
    since the underlying scores don't depend on those display-only filters."""
    from src.acquisition_likelihood import score_acquisition_likelihood
    return score_acquisition_likelihood(scored_df)


def main():
    inject_theme()

    st.markdown('<div class="wordmark">M&amp;A <span>Screener</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Multi-factor NSE screener — valuation, quality, leverage, and growth, '
        'ranked sector-relative. Not investment advice; see README for methodology.</div>',
        unsafe_allow_html=True,
    )

    try:
        scored, comps_dupont, explanations, baseline = load_data()
    except FileNotFoundError:
        st.error(
            "No scored data found. Run the pipeline first: `python -m src.run_pipeline`"
        )
        return

    render_hero(scored)

    with st.sidebar:
        st.subheader("One-Page Company Summary")
        st.caption("Pulls rank, DuPont, valuation label, acquisition likelihood, and explanation into a single shareable PDF.")
        summary_ticker = st.selectbox("Ticker", sorted(scored["ticker"].tolist()), key="summary_ticker")
        if st.button("Generate PDF"):
            try:
                from src.company_summary import generate_summary_pdf
                import tempfile
                tmp_path = str(Path(tempfile.gettempdir()) / f"summary_{summary_ticker.replace('.', '_')}.pdf")
                generate_summary_pdf(summary_ticker, tmp_path)
                with open(tmp_path, "rb") as f:
                    st.download_button(
                        "Download PDF", data=f.read(),
                        file_name=f"summary_{summary_ticker.replace('.', '_')}.pdf",
                        mime="application/pdf",
                    )
            except ValueError as e:
                st.error(str(e))

    tab_shortlist, tab_dupont, tab_deal, tab_best_targets, tab_likelihood, tab_baseline = st.tabs(
        ["Shortlist", "Comps + DuPont", "Deal Simulator", "Best Targets Finder", "Acquisition Likelihood", "vs. Naive Baseline"]
    )

    with tab_shortlist:
        with st.container(border=True):
            st.subheader("Top-Ranked Companies")

            sectors_available_main = sorted(scored["sector"].dropna().unique().tolist()) if "sector" in scored.columns else []
            sector_filter_main = st.multiselect(
                "Filter sectors (leave empty for all)", options=sectors_available_main, default=[], key="main_sector_filter",
            )
            scored_filtered = scored[scored["sector"].isin(sector_filter_main)] if sector_filter_main else scored

            if scored_filtered.empty:
                st.warning("No companies match the selected sector filter.")
            else:
                top_n = st.slider("Show top N", min_value=5, max_value=max(5, len(scored_filtered)), value=min(10, len(scored_filtered)))
                display_cols = [c for c in [
                    "rank", "ticker", "sector", "sector_rank", "composite_score",
                    "trailing_pe", "return_on_equity", "debt_to_equity_ratio", "revenue_growth",
                ] if c in scored_filtered.columns]
                top_rows = scored_filtered.sort_values("rank").head(top_n)
                st.dataframe(
                    styled_table(
                        top_rows[display_cols],
                        format_map={
                            "composite_score": "{:.3f}",
                            "trailing_pe": "{:.1f}",
                            "return_on_equity": "{:.1%}",
                            "debt_to_equity_ratio": "{:.2f}",
                            "revenue_growth": "{:.1%}",
                        },
                        score_cols=["composite_score"],
                        mono_cols=["ticker"],
                    ),
                    width='stretch',
                    hide_index=True,
                )

                st.subheader("Composite Score by Sector")
                if "sector" in top_rows.columns:
                    chart_data = top_rows.set_index("ticker")[["composite_score"]]
                    st.bar_chart(chart_data, color="#22D3B0")

                if explanations is not None:
                    st.subheader("Why did each company rank this way?")
                    shown_tickers = top_rows["ticker"].tolist()
                    filtered_explanations = explanations[explanations["ticker"].isin(shown_tickers)]
                    if filtered_explanations.empty:
                        st.caption(
                            "No pre-generated explanations available for the current sector filter — "
                            "explanations are only generated for the overall top shortlist. "
                            "Re-run `python -m src.explainability` with a higher --top-n to cover more sectors."
                        )
                    else:
                        for _, row in filtered_explanations.iterrows():
                            with st.expander(row["ticker"]):
                                st.write(row["summary"])

    with tab_dupont:
        with st.container(border=True):
            st.subheader("Comps + DuPont Breakdown")
            if comps_dupont is not None:
                dupont_cols = [c for c in [
                    "rank", "ticker", "sector", "valuation_label",
                    "return_on_equity", "net_margin", "asset_turnover", "equity_multiplier",
                    "dupont_implied_roe", "dupont_vs_reported_roe_gap",
                ] if c in comps_dupont.columns]
                st.dataframe(
                    styled_table(comps_dupont[dupont_cols], mono_cols=["ticker"]),
                    width='stretch', hide_index=True,
                )
                st.caption(
                    "DuPont: ROE = Net Margin x Asset Turnover x Equity Multiplier. "
                    "Equity Multiplier is approximated as (Debt + Equity) / Equity, "
                    "not exact total assets — see module docstring for full scope notes."
                )
            else:
                st.info("Run `python -m src.comps_dupont` to generate this report.")

    with tab_deal:
        with st.container(border=True):
            st.subheader("Accretion / Dilution Simulator")
            st.caption(
                "Simplified model: no tax step-up adjustments beyond the interest tax shield. "
                "Enough to demonstrate deal mechanics, not a full model."
            )

            tickers = sorted(scored["ticker"].tolist())
            col1, col2 = st.columns(2)
            with col1:
                acquirer = st.selectbox("Acquirer", tickers, index=0)
            with col2:
                target = st.selectbox("Target", tickers, index=min(1, len(tickers) - 1))

            premium = st.slider("Premium over target's current price", 0, 100, 25, format="%d%%") / 100
            cash_pct = st.slider("Cash portion of consideration", 0, 100, 50, format="%d%%") / 100

            with st.expander("Advanced: debt financing on the cash portion"):
                debt_funded_pct = st.slider("Fraction of cash funded via new debt (vs. reserves)", 0, 100, 0, format="%d%%") / 100
                interest_rate = st.slider("Assumed interest rate on new debt", 0.0, 15.0, 8.0, step=0.5, format="%.1f%%") / 100
                tax_rate = st.slider("Assumed tax rate (for interest tax shield)", 0.0, 40.0, 25.0, step=1.0, format="%.0f%%") / 100

            with st.expander("Advanced: synergy assumptions"):
                st.caption(
                    "Simplified single-year static uplift, not a phased multi-year synergy schedule. "
                    "Revenue synergy uplifts the target's own net income; cost synergy is a tax-effected "
                    "uplift on the combined pre-synergy net income."
                )
                revenue_synergy_pct = st.slider("Revenue synergy (% uplift to target's net income)", 0, 50, 0, format="%d%%") / 100
                cost_synergy_pct = st.slider("Cost synergy (% of combined net income, tax-effected)", 0, 50, 0, format="%d%%") / 100

            if acquirer == target:
                st.warning("Acquirer and target must be different companies.")
            else:
                try:
                    deal = DealInputs(
                        acquirer, target, premium, cash_pct,
                        debt_funded_pct=debt_funded_pct, interest_rate=interest_rate, tax_rate=tax_rate,
                        revenue_synergy_pct=revenue_synergy_pct, cost_synergy_pct=cost_synergy_pct,
                    )
                    result = run_deal(scored, deal)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Acquirer Standalone EPS", f"{result.acquirer_standalone_eps:.2f}")
                    m2.metric(
                        "Pro-Forma EPS",
                        f"{result.pro_forma_eps:.2f}",
                        f"{result.accretion_dilution_pct:+.2%}",
                    )
                    with m3:
                        st.markdown('<div style="padding-top:6px;"><span style="color:#8B96AE;font-size:0.875rem;">Verdict</span><br>' +
                                    badge_html("ACCRETIVE" if result.is_accretive else "DILUTIVE",
                                               "positive" if result.is_accretive else "negative") +
                                    "</div>", unsafe_allow_html=True)

                    with st.expander("Deal mechanics detail"):
                        st.write(f"Deal value: {result.deal_value:,.0f}")
                        st.write(f"Cash consideration: {result.cash_consideration:,.0f}")
                        st.write(f"Stock consideration: {result.stock_consideration:,.0f}")
                        st.write(f"New shares issued: {result.new_shares_issued:,.0f}")
                        if debt_funded_pct > 0:
                            st.write(f"New debt raised: {result.new_debt_raised:,.0f}")
                            st.write(f"After-tax interest cost: {result.after_tax_interest_cost:,.0f}")
                        if revenue_synergy_pct > 0 or cost_synergy_pct > 0:
                            st.write(f"Total synergy income: {result.synergy_income:,.0f}")

                    if revenue_synergy_pct > 0 or cost_synergy_pct > 0:
                        st.subheader("Multi-Year View: Synergies Phased In Over Time")
                        st.caption(
                            "Real deals don't realize 100% of synergies on day one — this phases them in over "
                            "3 years (25% / 60% / 100%) while growing standalone earnings at each company's own "
                            "revenue growth rate. Deal structure (shares, debt, interest cost) stays fixed at "
                            "entry — no debt paydown is modeled over the projection (that's the LBO model's job)."
                        )
                        multi_year = project_multi_year_accretion(scored, deal, years=3)
                        st.dataframe(
                            styled_table(
                                multi_year,
                                format_map={
                                    "synergy_ramp_pct": "{:.0f}%",
                                    "acquirer_standalone_eps": "{:.2f}",
                                    "pro_forma_eps": "{:.2f}",
                                    "synergy_income": "{:,.0f}",
                                    "accretion_dilution_pct": "{:+.2f}%",
                                },
                                bool_cols=["is_accretive"],
                                signed_pct_cols=["accretion_dilution_pct"],
                            ),
                            width='stretch',
                            hide_index=True,
                        )

                    st.subheader("Accretion/Dilution Heatmap: Finding the Best Terms")
                    st.caption(
                        "Sweeps premium (0-50%) x cash/stock mix (0-100%) using the same debt-financing "
                        "assumptions set above, and highlights the single combination that maximizes "
                        "accretion (or minimizes dilution, if nothing in the grid is accretive)."
                    )
                    optimal = find_optimal_terms(
                        scored, acquirer, target,
                        debt_funded_pct=debt_funded_pct, interest_rate=interest_rate, tax_rate=tax_rate,
                        revenue_synergy_pct=revenue_synergy_pct, cost_synergy_pct=cost_synergy_pct,
                    )
                    best, grid = optimal["best"], optimal["grid"]

                    best_verdict_kind = "positive" if best["is_accretive"] else "negative"
                    best_verdict_text = "ACCRETIVE" if best["is_accretive"] else "DILUTIVE"
                    best_pct_abs = abs(best["accretion_dilution_pct"])
                    best_badge = badge_html(f"{best_verdict_text} {best_pct_abs:.2f}%", best_verdict_kind)
                    st.markdown(
                        f'<div style="background-color:rgba(34,211,176,0.08); border:1px solid var(--border-subtle); '
                        f'border-radius:10px; padding:14px 18px; margin-bottom:12px;">'
                        f'Best terms: <strong>{best["premium_pct"]:.0%} premium</strong>, '
                        f'<strong>{best["cash_pct"]:.0%} cash / {1 - best["cash_pct"]:.0%} stock</strong> &rarr; '
                        f'{best_badge}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    pivot = grid.pivot(index="premium_pct", columns="cash_pct", values="accretion_dilution_pct")
                    fig = go.Figure(data=go.Heatmap(
                        z=pivot.values,
                        x=[f"{c:.0%}" for c in pivot.columns],
                        y=[f"{p:.0%}" for p in pivot.index],
                        colorscale=[[0, "#F0546B"], [0.5, "#141B2D"], [1, "#22D3B0"]],
                        zmid=0,
                        colorbar=dict(title="Accretion/<br>Dilution %"),
                        hovertemplate="Premium: %{y}<br>Cash: %{x}<br>Accretion/Dilution: %{z:.2f}%<extra></extra>",
                    ))
                    best_x = f"{best['cash_pct']:.0%}"
                    best_y = f"{best['premium_pct']:.0%}"
                    fig.add_annotation(
                        x=best_x, y=best_y, text="★ Best", showarrow=True,
                        arrowhead=2, ax=30, ay=-30, font=dict(color="#0B1120", size=12),
                        bgcolor="#E8A33D", bordercolor="#0B1120",
                    )
                    fig.update_layout(
                        xaxis_title="Cash % of Consideration",
                        yaxis_title="Premium %",
                        height=450,
                        margin=dict(l=10, r=10, t=30, b=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#E8ECF4"),
                    )
                    st.plotly_chart(fig, width='stretch')
                except ValueError as e:
                    st.error(str(e))

    with tab_best_targets:
        with st.container(border=True):
            st.subheader("Best Acquisition Targets Finder")
            st.caption(
                "Pick an acquirer and a set of deal terms — scans every other company in the "
                "universe as a potential target under those SAME terms, ranked by resulting "
                "accretion. Answers 'which of these companies should this acquirer actually be "
                "looking at,' rather than checking one pair at a time."
            )

            tickers_bt = sorted(scored["ticker"].tolist())
            acquirer_bt = st.selectbox("Acquirer", tickers_bt, index=0, key="best_targets_acquirer")

            col1, col2 = st.columns(2)
            with col1:
                premium_bt = st.slider("Premium over target's price", 0, 100, 25, format="%d%%", key="bt_premium") / 100
            with col2:
                cash_pct_bt = st.slider("Cash portion of consideration", 0, 100, 50, format="%d%%", key="bt_cash") / 100

            with st.expander("Advanced: debt financing on the cash portion"):
                debt_funded_pct_bt = st.slider("Fraction of cash funded via new debt", 0, 100, 0, format="%d%%", key="bt_debt") / 100
                interest_rate_bt = st.slider("Assumed interest rate", 0.0, 15.0, 8.0, step=0.5, format="%.1f%%", key="bt_rate") / 100
                tax_rate_bt = st.slider("Assumed tax rate", 0.0, 40.0, 25.0, step=1.0, format="%.0f%%", key="bt_tax") / 100

            max_size_pct_bt = st.slider(
                "Max target size (% of acquirer's own market cap)", 10, 300, 100, step=10, format="%d%%", key="bt_max_size",
                help=(
                    "A mega-cap acquirer can make almost ANY target look wildly accretive purely from "
                    "scale mismatch — this isn't a real deal insight, just arithmetic. This filter excludes "
                    "candidates too large to be a structurally plausible target. Default 100% means the "
                    "target can't be larger than the acquirer itself."
                ),
            ) / 100

            sectors_available = sorted(scored["sector"].dropna().unique().tolist()) if "sector" in scored.columns else []
            sector_filter = st.multiselect(
                "Filter target sectors (leave empty for all sectors)",
                options=sectors_available, default=[], key="bt_sector_filter",
            )

            try:
                all_targets = cached_find_best_targets(
                    scored, acquirer_bt,
                    premium_bt, cash_pct_bt,
                    debt_funded_pct_bt, interest_rate_bt, tax_rate_bt,
                    max_size_pct_bt,
                )
                if sector_filter:
                    all_targets = all_targets[all_targets["sector"].isin(sector_filter)]

                n_accretive = int(all_targets["is_accretive"].sum())
                st.caption(f"{n_accretive}/{len(all_targets)} evaluated targets would be accretive under these terms "
                           f"(after excluding structurally implausible oversized candidates).")

                top_n_bt = st.slider("Show top N targets", 5, min(50, len(all_targets)), min(10, len(all_targets)), key="bt_top_n")
                display = all_targets.head(top_n_bt).copy()
                display["rank"] = range(1, len(display) + 1)

                display_cols = ["rank", "target", "sector", "relative_size_pct", "accretion_dilution_pct", "is_accretive", "deal_value"]
                if "target_composite_score" in display.columns:
                    display_cols.insert(4, "target_composite_score")
                if "target_sector_rank" in display.columns:
                    display_cols.insert(5, "target_sector_rank")
                if "target_acquisition_likelihood_score" in display.columns:
                    display_cols.insert(6, "target_acquisition_likelihood_score")
                display_cols = [c for c in display_cols if c in display.columns]

                format_map = {"accretion_dilution_pct": "{:+.2f}%", "deal_value": "{:,.0f}", "relative_size_pct": "{:.1f}%"}
                score_cols = []
                if "target_composite_score" in display.columns:
                    format_map["target_composite_score"] = "{:.3f}"
                    score_cols.append("target_composite_score")
                if "target_acquisition_likelihood_score" in display.columns:
                    format_map["target_acquisition_likelihood_score"] = "{:.3f}"
                    score_cols.append("target_acquisition_likelihood_score")

                st.dataframe(
                    styled_table(
                        display[display_cols], format_map=format_map,
                        score_cols=score_cols, bool_cols=["is_accretive"],
                        signed_pct_cols=["accretion_dilution_pct"], mono_cols=["target"],
                    ),
                    width='stretch',
                    hide_index=True,
                )
                st.caption(
                    "relative_size_pct: target's market cap as a % of the acquirer's own. "
                    "target_acquisition_likelihood_score: is this target ALSO a classic cheap/undervalued "
                    "candidate (separate from the main quality score) — high accretion + high likelihood "
                    "score together is a stronger signal than accretion alone."
                )

                fig_bt = go.Figure(go.Bar(
                    x=display["accretion_dilution_pct"],
                    y=display["target"],
                    orientation="h",
                    marker_color=["#34D399" if v > 0 else "#F0546B" for v in display["accretion_dilution_pct"]],
                ))
                fig_bt.update_layout(
                    xaxis_title="Accretion / Dilution %",
                    yaxis=dict(autorange="reversed"),
                    height=max(300, 30 * len(display)),
                    margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E8ECF4"),
                )
                st.plotly_chart(fig_bt, width='stretch')
            except ValueError as e:
                st.error(str(e))

    with tab_likelihood:
        with st.container(border=True):
            st.subheader("Acquisition Likelihood — Value / Turnaround Candidates")
            st.caption(
                "A DIFFERENT lens from the main shortlist. The composite score rewards quality — "
                "this instead flags companies that are cheap, carry low leverage (balance sheet "
                "capacity for a buyer), but underperform on quality/margins relative to their own "
                "sector — the classic private-equity 'buy it cheap, fix the operations' thesis. "
                "A company can score well here specifically BECAUSE it ranks weakly on the main shortlist."
            )
            try:
                likelihood_full = cached_score_acquisition_likelihood(scored)

                sectors_avail_lh = sorted(likelihood_full["sector"].dropna().unique().tolist()) if "sector" in likelihood_full.columns else []
                sector_filter_lh = st.multiselect(
                    "Filter sectors (leave empty for all)", options=sectors_avail_lh, default=[], key="lh_sector_filter",
                )
                if sector_filter_lh:
                    likelihood_full = likelihood_full[likelihood_full["sector"].isin(sector_filter_lh)]

                top_n_lh = st.slider("Show top N", 5, min(50, len(likelihood_full)), min(15, len(likelihood_full)), key="lh_top_n")
                display_lh = likelihood_full.head(top_n_lh)

                lh_cols = [c for c in [
                    "acquisition_likelihood_rank", "ticker", "sector", "acquisition_likelihood_sector_rank",
                    "acquisition_likelihood_score", "valuation_bucket_score", "turnaround_potential_score",
                    "leverage_bucket_score", "composite_score", "rank",
                ] if c in display_lh.columns]

                renamed = display_lh[lh_cols].rename(columns={"rank": "main_screener_rank", "composite_score": "main_composite_score"})
                st.dataframe(
                    styled_table(
                        renamed,
                        format_map={
                            "acquisition_likelihood_score": "{:.3f}",
                            "valuation_bucket_score": "{:.3f}",
                            "turnaround_potential_score": "{:.3f}",
                            "leverage_bucket_score": "{:.3f}",
                            "main_composite_score": "{:.3f}",
                        },
                        score_cols=["acquisition_likelihood_score"],
                        mono_cols=["ticker"],
                    ),
                    width='stretch',
                    hide_index=True,
                )
                st.caption(
                    "main_screener_rank / main_composite_score: shown for contrast against the main "
                    "shortlist tab — a low main rank alongside a high acquisition-likelihood score is "
                    "exactly the 'undervalued, needs fixing' pattern this tab is built to surface."
                )
            except ValueError as e:
                st.error(str(e))

    with tab_baseline:
        with st.container(border=True):
            st.subheader("Composite Score vs. Naive P/E-Only Baseline")
            st.caption(
                "Not a backtest — no forward-returns validation. This shows whether the "
                "multi-factor score ranks companies meaningfully differently from a simple "
                "'cheapest P/E' sort, and which specific names diverge."
            )
            if baseline is not None:
                st.dataframe(
                    styled_table(
                        baseline[["ticker", "sector", "rank", "naive_rank", "composite_score", "trailing_pe"]],
                        format_map={"composite_score": "{:.3f}"},
                        score_cols=["composite_score"], mono_cols=["ticker"],
                    ),
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.info("Run `python -m src.baseline_comparison` to generate this report.")


if __name__ == "__main__":
    main()