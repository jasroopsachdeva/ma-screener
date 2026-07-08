"""
company_summary.py — Shareable company summary for a single company.

Pulls together the screener's rank/composite score, raw financials, the
DuPont breakdown, sector peer comparison, the full acquisition-likelihood
breakdown, the plain-English explanation, AND a real illustrative
accretion/dilution calculation (using this company as a hypothetical
target against the top-ranked company in its own sector) into a single
shareable PDF — the kind of company profile a real analyst hands someone
before a conversation, not just a screenshot of a spreadsheet row.

Every secondary input is OPTIONAL and degrades gracefully: comps+DuPont
and acquisition-likelihood only cover the top-N shortlist, the
illustrative deal calc needs a viable sector peer with complete data, and
none of these being unavailable should block the summary — only the core
scored_universe.csv entry is required.

Usage:
    python -m src.company_summary --ticker TCS.NS
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _safe_read_csv(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


def _sector_peers(ticker: str, scored: pd.DataFrame, n: int = 4) -> list:
    """A handful of same-sector peers for context — seeing a company's
    numbers in isolation says much less than seeing them next to the
    companies it's actually being ranked against."""
    row = scored[scored["ticker"] == ticker]
    if row.empty or "sector" not in scored.columns:
        return []
    sector = row.iloc[0].get("sector")
    if pd.isna(sector):
        return []
    peers = scored[(scored["sector"] == sector) & (scored["ticker"] != ticker)].sort_values("rank").head(n)
    cols = [c for c in ["ticker", "rank", "composite_score", "trailing_pe"] if c in peers.columns]
    return peers[cols].to_dict("records")


def _illustrative_deal(ticker: str, scored: pd.DataFrame) -> dict | None:
    """A real accretion/dilution calculation (using the same tested
    run_deal engine as the interactive simulator) with this company as
    the TARGET and the top-ranked company in its own sector as an
    illustrative acquirer, at standard terms (25% premium, 50/50
    cash-stock, no debt financing or synergies). Purely illustrative —
    not a real deal recommendation, just a concrete "what would a
    plausible in-sector acquisition of this company look like" data
    point, using the acquirer/target that are actually available rather
    than an arbitrary pairing.

    Returns None gracefully if there's no viable peer or either company
    is missing required data — this is a bonus section, not core."""
    from src.accretion_dilution import DealInputs, run_deal

    row = scored[scored["ticker"] == ticker]
    if row.empty or "sector" not in scored.columns:
        return None
    sector = row.iloc[0].get("sector")
    if pd.isna(sector):
        return None

    peers = scored[(scored["sector"] == sector) & (scored["ticker"] != ticker)].sort_values("rank")
    for _, peer_row in peers.iterrows():
        acquirer_ticker = peer_row["ticker"]
        try:
            deal = DealInputs(acquirer_ticker, ticker, premium_pct=0.25, cash_pct=0.5)
            result = run_deal(scored, deal)
            return {
                "acquirer": acquirer_ticker,
                "premium_pct": 0.25,
                "cash_pct": 0.5,
                "pro_forma_eps": result.pro_forma_eps,
                "acquirer_standalone_eps": result.acquirer_standalone_eps,
                "accretion_dilution_pct": result.accretion_dilution_pct,
                "is_accretive": result.is_accretive,
            }
        except ValueError:
            continue
    return None


def build_summary_data(
    ticker: str,
    scored_path: str = "data/processed/scored_universe.csv",
    comps_dupont_path: str = "data/processed/comps_dupont_report.csv",
    explanations_path: str = "data/processed/explanations.csv",
    likelihood_path: str = "data/processed/acquisition_likelihood.csv",
) -> dict:
    scored = pd.read_csv(scored_path)
    row = scored[scored["ticker"] == ticker]
    if row.empty:
        raise ValueError(
            f"'{ticker}' not found in {scored_path} — check the ticker is in "
            f"config/universe.yaml and that scoring has run."
        )
    row = row.iloc[0]

    data = {
        "ticker": ticker,
        "sector": row.get("sector"),
        "rank": row.get("rank"),
        "sector_rank": row.get("sector_rank"),
        "composite_score": row.get("composite_score"),
        "price": row.get("price"),
        "market_cap": row.get("market_cap"),
        "total_debt": row.get("total_debt"),
        "total_equity": row.get("total_equity"),
        "trailing_pe": row.get("trailing_pe"),
        "ev_to_ebitda": row.get("ev_to_ebitda"),
        "return_on_equity": row.get("return_on_equity"),
        "net_margin": row.get("net_margin"),
        "debt_to_equity_ratio": row.get("debt_to_equity_ratio"),
        "revenue_growth": row.get("revenue_growth"),
        "valuation_label": None,
        "dupont": None,
        "acquisition_likelihood_score": None,
        "acquisition_likelihood_rank": None,
        "acquisition_likelihood_components": None,
        "explanation_summary": None,
        "sector_peers": _sector_peers(ticker, scored),
        "illustrative_deal": _illustrative_deal(ticker, scored),
    }

    comps_dupont = _safe_read_csv(comps_dupont_path)
    if comps_dupont is not None:
        cd_row = comps_dupont[comps_dupont["ticker"] == ticker]
        if not cd_row.empty:
            cd_row = cd_row.iloc[0]
            data["valuation_label"] = cd_row.get("valuation_label")
            data["dupont"] = {
                "net_margin": cd_row.get("net_margin"),
                "asset_turnover": cd_row.get("asset_turnover"),
                "equity_multiplier": cd_row.get("equity_multiplier"),
                "dupont_implied_roe": cd_row.get("dupont_implied_roe"),
                "gap": cd_row.get("dupont_vs_reported_roe_gap"),
            }

    likelihood = _safe_read_csv(likelihood_path)
    if likelihood is not None:
        l_row = likelihood[likelihood["ticker"] == ticker]
        if not l_row.empty:
            l_row = l_row.iloc[0]
            data["acquisition_likelihood_score"] = l_row.get("acquisition_likelihood_score")
            data["acquisition_likelihood_rank"] = l_row.get("acquisition_likelihood_rank")
            data["acquisition_likelihood_components"] = {
                "valuation": l_row.get("valuation_bucket_score"),
                "turnaround_potential": l_row.get("turnaround_potential_score"),
                "leverage": l_row.get("leverage_bucket_score"),
            }

    explanations = _safe_read_csv(explanations_path)
    if explanations is not None:
        e_row = explanations[explanations["ticker"] == ticker]
        if not e_row.empty:
            data["explanation_summary"] = e_row.iloc[0].get("summary")

    return data


def generate_summary_pdf(ticker: str, output_path: str, **data_kwargs) -> str:
    """Builds a shareable company profile PDF — 1-2 pages depending on how
    much data is available, organized so sections with unavailable data
    are simply omitted rather than left blank or causing an error."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT

    data = build_summary_data(ticker, **data_kwargs)

    ACCENT = colors.HexColor("#5B4FE8")
    ACCENT_DIM = colors.HexColor("#EEEDFC")
    TEXT_PRIMARY = colors.HexColor("#1C1D2E")
    TEXT_MUTED = colors.HexColor("#6B7085")
    POSITIVE = colors.HexColor("#16A34A")
    NEGATIVE = colors.HexColor("#DC2626")
    BORDER = colors.HexColor("#E4E6F0")

    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        topMargin=0.6 * inch, bottomMargin=0.55 * inch,
    )

    title_style = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=TEXT_PRIMARY, spaceAfter=2)
    subtitle_style = ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10.5, leading=14, spaceAfter=16, textColor=TEXT_MUTED)
    heading_style = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=12, leading=16, spaceAfter=7, spaceBefore=16, textColor=ACCENT)
    body_style = ParagraphStyle("Body", fontName="Helvetica", fontSize=9.8, leading=14, spaceAfter=6, textColor=TEXT_PRIMARY, alignment=TA_LEFT)
    muted_style = ParagraphStyle("Muted", fontName="Helvetica", fontSize=9, leading=13, spaceAfter=6, textColor=TEXT_MUTED)
    footer_style = ParagraphStyle("Footer", fontName="Helvetica-Oblique", fontSize=7.5, leading=11, textColor=TEXT_MUTED)

    def fmt_pct(v):
        return f"{v:.1%}" if v is not None and pd.notna(v) else "N/A"

    def fmt_num(v, decimals=2):
        return f"{v:.{decimals}f}" if v is not None and pd.notna(v) else "N/A"

    def fmt_cr(v):
        if v is None or pd.isna(v):
            return "N/A"
        return f"{v / 1e7:,.0f} Cr"

    def section_table(rows, col_widths):
        t = Table(rows, colWidths=col_widths)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, -1), TEXT_PRIMARY),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.5, BORDER),
        ]))
        return t

    story = []

    sector_str = data["sector"] if data["sector"] else "—"
    story.append(Paragraph(f"{data['ticker']}", title_style))
    story.append(Paragraph(f"Sector: {sector_str}", subtitle_style))

    story.append(Paragraph("Screening Summary", heading_style))
    rank_str = f"#{int(data['rank'])} overall" if pd.notna(data.get("rank")) else "N/A"
    sector_rank_str = f"#{int(data['sector_rank'])} in sector" if pd.notna(data.get("sector_rank")) else "N/A"
    composite_str = fmt_num(data.get("composite_score"), 3)
    story.append(Paragraph(f"Rank: <b>{rank_str}</b> &nbsp;|&nbsp; {sector_rank_str} &nbsp;|&nbsp; Composite score: <b>{composite_str}</b>", body_style))

    story.append(Paragraph("Company Overview", heading_style))
    overview_rows = [
        ["Price", fmt_num(data.get("price"), 2), "Market Cap", fmt_cr(data.get("market_cap"))],
        ["Total Debt", fmt_cr(data.get("total_debt")), "Total Equity", fmt_cr(data.get("total_equity"))],
    ]
    story.append(section_table(overview_rows, [80, 110, 90, 110]))

    story.append(Paragraph("Key Metrics", heading_style))
    metrics_rows = [
        ["P/E", fmt_num(data.get("trailing_pe"), 1), "EV/EBITDA", fmt_num(data.get("ev_to_ebitda"), 1)],
        ["ROE", fmt_pct(data.get("return_on_equity")), "Net Margin", fmt_pct(data.get("net_margin"))],
        ["Debt/Equity", fmt_num(data.get("debt_to_equity_ratio"), 2), "Revenue Growth", fmt_pct(data.get("revenue_growth"))],
    ]
    story.append(section_table(metrics_rows, [80, 110, 90, 110]))

    if data.get("valuation_label"):
        story.append(Paragraph("Valuation vs. Sector", heading_style))
        label_text = data["valuation_label"].replace("_", " ").title()
        badge_color = POSITIVE if "cheap" in data["valuation_label"] else (NEGATIVE if "rich" in data["valuation_label"] else TEXT_MUTED)
        story.append(Paragraph(f'<font color="{badge_color}"><b>{label_text}</b></font>', body_style))

    if data.get("sector_peers"):
        story.append(Paragraph("Sector Peer Comparison", heading_style))
        story.append(Paragraph(
            f"How {data['ticker']} compares to its closest-ranked peers in the same sector, for context.",
            muted_style,
        ))
        peer_rows = [["Ticker", "Rank", "Composite", "P/E"]]
        for peer in data["sector_peers"]:
            peer_rows.append([
                peer.get("ticker", "—"),
                f"#{int(peer['rank'])}" if pd.notna(peer.get("rank")) else "—",
                fmt_num(peer.get("composite_score"), 3),
                fmt_num(peer.get("trailing_pe"), 1),
            ])
        story.append(section_table(peer_rows, [110, 60, 90, 90]))

    if data.get("dupont"):
        d = data["dupont"]
        story.append(Paragraph("DuPont Breakdown (ROE = Net Margin &times; Asset Turnover &times; Equity Multiplier)", heading_style))
        story.append(Paragraph(
            f"Net Margin: {fmt_pct(d.get('net_margin'))} &nbsp;&times;&nbsp; "
            f"Asset Turnover: {fmt_num(d.get('asset_turnover'))} &nbsp;&times;&nbsp; "
            f"Equity Multiplier: {fmt_num(d.get('equity_multiplier'))} "
            f"= Implied ROE: <b>{fmt_pct(d.get('dupont_implied_roe'))}</b>",
            body_style,
        ))
        gap = d.get("gap")
        if gap is not None and pd.notna(gap) and abs(gap) > 0.05:
            story.append(Paragraph(
                f"Note: implied ROE diverges {fmt_pct(abs(gap))} from reported ROE — "
                f"worth a manual check (equity-multiplier is a proxy, see README).",
                muted_style,
            ))

    if data.get("acquisition_likelihood_score") is not None and pd.notna(data.get("acquisition_likelihood_score")):
        story.append(Paragraph("Acquisition Likelihood (Value / Turnaround Lens)", heading_style))
        rank_txt = f", rank #{int(data['acquisition_likelihood_rank'])}" if pd.notna(data.get("acquisition_likelihood_rank")) else ""
        story.append(Paragraph(f"Overall score: <b>{fmt_num(data['acquisition_likelihood_score'], 3)}</b>{rank_txt}", body_style))
        comp = data.get("acquisition_likelihood_components") or {}
        if any(v is not None and pd.notna(v) for v in comp.values()):
            comp_rows = [
                ["Valuation", fmt_num(comp.get("valuation"), 3),
                 "Turnaround Potential", fmt_num(comp.get("turnaround_potential"), 3)],
                ["Leverage", fmt_num(comp.get("leverage"), 3), "", ""],
            ]
            story.append(section_table(comp_rows, [110, 80, 130, 80]))
            story.append(Paragraph(
                "Turnaround Potential is the inverse of the quality score — high here means "
                "weak margins/ROE relative to sector, which is the actual thesis this lens looks for.",
                muted_style,
            ))

    if data.get("illustrative_deal"):
        deal = data["illustrative_deal"]
        story.append(Paragraph("Illustrative Deal Snapshot", heading_style))
        story.append(Paragraph(
            f"Every other section on this page is about {data['ticker']} itself. This one asks a "
            f"different question: <b>if {data['ticker']} were ACQUIRED, what would that do to the "
            f"acquirer?</b> — a standard way of gauging how attractive a company is as a target. "
            f"Using {data['ticker']}'s top-ranked in-sector peer, <b>{deal['acquirer']}</b>, as an "
            f"illustrative acquirer (25% premium, 50% cash / 50% stock, no synergies or debt financing), "
            f"the numbers below are <b>{deal['acquirer']}'s own EPS</b> before and after the hypothetical "
            f"deal — not {data['ticker']}'s. Purely illustrative, not a real deal recommendation.",
            muted_style,
        ))
        verdict = "ACCRETIVE" if deal["is_accretive"] else "DILUTIVE"
        verdict_color = POSITIVE if deal["is_accretive"] else NEGATIVE
        pct = abs(deal["accretion_dilution_pct"]) * 100
        story.append(Paragraph(
            f"If {deal['acquirer']} acquired {data['ticker']}: {deal['acquirer']}'s own standalone EPS "
            f"({fmt_num(deal['acquirer_standalone_eps'], 2)}) would move to a pro-forma "
            f"{fmt_num(deal['pro_forma_eps'], 2)} &nbsp;&mdash;&nbsp; "
            f'<font color="{verdict_color}"><b>{verdict} by {pct:.2f}%</b></font> for {deal["acquirer"]}.',
            body_style,
        ))

    if data.get("explanation_summary"):
        story.append(Paragraph("Why It Ranked This Way", heading_style))
        story.append(Paragraph(data["explanation_summary"], body_style))

    story.append(Spacer(1, 16))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    story.append(Paragraph(
        f"Generated by M&amp;A Screener on {generated_at}. Not investment advice — "
        f"see project README for full methodology and limitations.",
        footer_style,
    ))

    doc.build(story)
    return output_path


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Generate a company summary PDF")
    parser.add_argument("--ticker", required=True, help="Ticker to summarize, e.g. TCS.NS")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--scored-path", default="data/processed/scored_universe.csv")
    parser.add_argument("--comps-dupont-path", default="data/processed/comps_dupont_report.csv")
    parser.add_argument("--explanations-path", default="data/processed/explanations.csv")
    parser.add_argument("--likelihood-path", default="data/processed/acquisition_likelihood.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(args.output_dir) / f"summary_{args.ticker.replace('.', '_')}.pdf")
    result_path = generate_summary_pdf(
        args.ticker, out_path,
        scored_path=args.scored_path,
        comps_dupont_path=args.comps_dupont_path,
        explanations_path=args.explanations_path,
        likelihood_path=args.likelihood_path,
    )
    print(f"\nCompany summary for {args.ticker} saved -> {result_path}")
