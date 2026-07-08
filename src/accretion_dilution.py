"""
accretion_dilution.py — Simplified accretion/dilution simulator.

Given an acquirer and a target from the screened universe, plus assumed
deal terms (premium %, cash/stock mix), estimates the pro-forma EPS impact
on the acquirer. This is the feature that makes the tool genuinely
M&A-specific rather than a general equity screener — it answers "if I
actually bought this company, would my EPS go up or down."

SCOPE — deliberately simplified (documented, not hidden):
  - No synergy assumptions (revenue or cost synergies both ignored)
  - No tax step-up or amortization effects beyond the interest tax shield
  - Interest expense on the CASH portion IS modeled, but only when
    explicitly opted into via debt_funded_pct > 0 (default: 0, meaning
    cash is assumed funded from existing reserves at no cost). When
    enabled, uses a flat assumed interest rate and tax rate rather than
    the acquirer's actual cost of debt or marginal tax rate.
  - Net income derived as ROE x Total Equity (since raw net income isn't
    pulled directly from yfinance in this pipeline) — this is an estimate,
    not a reported figure, and inherits any imprecision from ROE/equity data
  - Shares outstanding derived as Market Cap / Price — a standard
    approximation, but not identical to the company's reported diluted
    share count

This is intentionally the "simplified version" scoped from the start:
enough to demonstrate real deal-mechanics understanding without pretending
to modeling precision the underlying data doesn't support.

Usage:
    python -m src.accretion_dilution --acquirer TCS.NS --target COFORGE.NS --premium 0.25 --cash-pct 0.5
    python -m src.accretion_dilution --acquirer TCS.NS --target COFORGE.NS --cash-pct 1.0 --debt-funded-pct 1.0 --interest-rate 0.09
    python -m src.accretion_dilution --acquirer TCS.NS --target COFORGE.NS --sensitivity
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DealInputs:
    acquirer_ticker: str
    target_ticker: str
    premium_pct: float   # e.g. 0.25 = 25% premium over target's current price
    cash_pct: float       # e.g. 0.5 = 50% of deal funded in cash, rest in acquirer stock

    # Debt financing for the cash portion — all optional, all default to
    # "no new debt" so existing behavior is unchanged unless explicitly
    # opted into. When debt_funded_pct > 0, the after-tax interest cost on
    # that new debt is deducted from pro-forma net income, which is the
    # piece a pure cash-vs-stock model is missing: cash doesn't dilute
    # shares, but debt-funded cash isn't free — it costs interest instead.
    debt_funded_pct: float = 0.0   # fraction of the CASH consideration funded via new debt (0.0 = fully reserves, 1.0 = fully new debt)
    interest_rate: float = 0.08    # assumed cost of new debt, annualized (8% is a reasonable placeholder for Indian corporate debt — override with a real rate if known)
    tax_rate: float = 0.25         # assumed effective tax rate, used for the interest tax shield (~25% is a reasonable placeholder for Indian corporates)

    # Price basis for the premium calc. 0 (default) = use the spot price
    # already in the scored universe (unchanged, original behavior). A
    # positive value (e.g. 30) fetches a trailing N-day average close via
    # yfinance and uses THAT instead — the conventional real-world basis
    # for pricing a deal premium, since spot price alone can be noisy.
    avg_price_days: int = 0

    # If True, the target's EXISTING debt is added to new_debt_raised (as
    # if the acquirer refinances it at the same assumed interest_rate),
    # on top of any new debt used to fund the cash consideration. Many
    # real deals trigger change-of-control refinancing clauses on a
    # target's existing debt — defaulting to False assumes the target's
    # debt is left in place untouched, which is the simpler but not
    # universally realistic assumption.
    refinance_target_debt: bool = False

    # Synergy assumptions — both optional, both default to 0.0 (no
    # synergies) so existing behavior is unchanged unless explicitly
    # opted into. This is deliberately a simplified, single static %
    # applied immediately, not a phased-in multi-year synergy schedule a
    # real banker's model would build — enough to demonstrate the concept
    # without pretending to a precision the model doesn't support.
    revenue_synergy_pct: float = 0.0  # % uplift to the TARGET's standalone net income (e.g. cross-selling through acquirer's channels)
    cost_synergy_pct: float = 0.0     # % of COMBINED pre-synergy net income recovered via cost/overhead cuts, tax-effected like interest expense


@dataclass
class DealResult:
    acquirer_ticker: str
    target_ticker: str
    premium_pct: float
    cash_pct: float
    acquirer_standalone_eps: float
    target_standalone_eps: float
    deal_value: float
    cash_consideration: float
    stock_consideration: float
    new_shares_issued: float
    new_debt_raised: float
    target_debt_refinanced: float
    interest_expense: float
    after_tax_interest_cost: float
    synergy_income: float
    pro_forma_net_income: float
    pro_forma_shares: float
    pro_forma_eps: float
    accretion_dilution_pct: float
    is_accretive: bool


def load_universe(path: str = "data/processed/scored_universe.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["ticker", "price", "market_cap", "return_on_equity", "total_equity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required columns {missing} — "
            f"re-run `python -m src.scoring` with the current version."
        )
    return df


def fetch_avg_price(ticker: str, days: int) -> float:
    """Trailing N-day average closing price via yfinance. Real deal
    premiums are conventionally priced off a 30/60/90-day average, not a
    single day's spot price, which can be noisy — using the spot price
    alone can make a deal look more or less attractive than it really is
    purely due to short-term price movement unrelated to the deal itself.

    Raises on failure rather than silently falling back to spot price —
    if the caller asked for an averaged basis, a silent fallback would
    misrepresent what basis was actually used."""
    import yfinance as yf
    hist = yf.Ticker(ticker).history(period=f"{days}d")
    if hist is None or hist.empty or "Close" not in hist.columns:
        raise ValueError(
            f"Could not fetch {days}-day price history for '{ticker}' from yfinance "
            f"— try a shorter window, or omit avg_price_days to use the spot price instead."
        )
    return float(hist["Close"].mean())


def _company_fundamentals(df: pd.DataFrame, ticker: str, avg_price_days: int = 0) -> dict:
    row = df[df["ticker"] == ticker]
    if row.empty:
        raise ValueError(f"'{ticker}' not found in the scored universe — check the ticker is in your config/universe.yaml")
    row = row.iloc[0]

    for field in ["price", "market_cap", "return_on_equity", "total_equity"]:
        if pd.isna(row[field]):
            raise ValueError(
                f"'{ticker}' is missing '{field}', which the accretion/dilution "
                f"model requires — this company can't be used as an acquirer or "
                f"target until that field is available."
            )

    if row["total_equity"] <= 0:
        raise ValueError(
            f"'{ticker}' has negative or zero total_equity ({row['total_equity']:,.0f}) — "
            f"net_income = ROE x equity would produce a sign-flipped, nonsensical value "
            f"for a company with a negative equity base (e.g. accumulated losses "
            f"exceeding paid-in capital). This company can't be used as an acquirer "
            f"or target in the accretion/dilution model until its equity position "
            f"is positive."
        )

    if row["market_cap"] <= 0 or row["price"] <= 0:
        raise ValueError(f"'{ticker}' has non-positive market_cap or price — cannot derive shares outstanding.")

    # shares_outstanding MUST stay tied to the spot price the market_cap
    # figure was actually computed at — averaging the price here would
    # desync it from market_cap and produce a share count that doesn't
    # correspond to any real snapshot of the company. The averaged price
    # (when requested) is used separately, only as the DEAL price basis.
    spot_price = row["price"]
    shares_outstanding = row["market_cap"] / spot_price
    net_income = row["return_on_equity"] * row["total_equity"]
    eps = net_income / shares_outstanding

    if avg_price_days > 0:
        deal_price = fetch_avg_price(ticker, avg_price_days)
    else:
        deal_price = spot_price

    total_debt = row["total_debt"] if "total_debt" in row.index and pd.notna(row.get("total_debt")) else None

    return {
        "ticker": ticker,
        "spot_price": spot_price,
        "deal_price": deal_price,
        "shares_outstanding": shares_outstanding,
        "net_income": net_income,
        "eps": eps,
        "total_debt": total_debt,
    }


def run_deal(df: pd.DataFrame, deal: DealInputs) -> DealResult:
    acquirer = _company_fundamentals(df, deal.acquirer_ticker, avg_price_days=deal.avg_price_days)
    target = _company_fundamentals(df, deal.target_ticker, avg_price_days=deal.avg_price_days)

    # Premium is priced off the DEAL price basis (spot, or averaged if
    # avg_price_days was set) — this is the conventional real-world
    # approach, since a single day's spot price can be noisy.
    purchase_price_per_share = target["deal_price"] * (1 + deal.premium_pct)
    deal_value = purchase_price_per_share * target["shares_outstanding"]

    cash_consideration = deal_value * deal.cash_pct
    stock_consideration = deal_value * (1 - deal.cash_pct)
    # New shares are issued at the acquirer's DEAL price basis too, for
    # the same reason — the issuance price in a real deal is typically
    # set relative to a recent average, not a single noisy spot tick.
    new_shares_issued = stock_consideration / acquirer["deal_price"]

    # Debt financing on the cash portion: cash doesn't dilute shares, but
    # if it's funded by new borrowing rather than existing reserves, it
    # isn't free — it costs interest, net of the tax shield. This is the
    # piece a pure cash-vs-stock model misses (see DealInputs docstring).
    new_debt_from_cash = cash_consideration * deal.debt_funded_pct

    # Optional: refinancing the target's EXISTING debt (many real deals
    # trigger change-of-control clauses requiring this). Defaults off —
    # see DealInputs.refinance_target_debt docstring.
    target_debt_refinanced = 0.0
    if deal.refinance_target_debt:
        if target["total_debt"] is None:
            raise ValueError(
                f"refinance_target_debt=True was set, but '{deal.target_ticker}' "
                f"is missing total_debt in the scored universe — can't model "
                f"refinancing without knowing the target's existing debt balance."
            )
        target_debt_refinanced = target["total_debt"]

    new_debt_raised = new_debt_from_cash + target_debt_refinanced
    interest_expense = new_debt_raised * deal.interest_rate
    after_tax_interest_cost = interest_expense * (1 - deal.tax_rate)

    # Synergies (opt-in, default 0 = no change from before). Simplified,
    # single-year static uplift rather than a phased multi-year schedule —
    # see DealInputs docstring for scope notes.
    #   - revenue_synergy_pct: uplifts the TARGET's own net income (e.g.
    #     cross-selling through the acquirer's channels)
    #   - cost_synergy_pct: a further uplift on the COMBINED pre-synergy
    #     net income base, representing overhead/cost cuts — tax-effected
    #     the same way interest expense is, since realized cost savings
    #     are pretax dollars that get taxed like any other income
    combined_pre_synergy_ni = acquirer["net_income"] + target["net_income"]
    revenue_synergy_income = target["net_income"] * deal.revenue_synergy_pct
    cost_synergy_income = combined_pre_synergy_ni * deal.cost_synergy_pct * (1 - deal.tax_rate)
    synergy_income = revenue_synergy_income + cost_synergy_income

    pro_forma_net_income = combined_pre_synergy_ni + synergy_income - after_tax_interest_cost
    pro_forma_shares = acquirer["shares_outstanding"] + new_shares_issued
    pro_forma_eps = pro_forma_net_income / pro_forma_shares

    accretion_dilution_pct = (pro_forma_eps - acquirer["eps"]) / acquirer["eps"]

    return DealResult(
        acquirer_ticker=deal.acquirer_ticker,
        target_ticker=deal.target_ticker,
        premium_pct=deal.premium_pct,
        cash_pct=deal.cash_pct,
        acquirer_standalone_eps=acquirer["eps"],
        target_standalone_eps=target["eps"],
        deal_value=deal_value,
        cash_consideration=cash_consideration,
        stock_consideration=stock_consideration,
        new_shares_issued=new_shares_issued,
        new_debt_raised=new_debt_raised,
        target_debt_refinanced=target_debt_refinanced,
        interest_expense=interest_expense,
        after_tax_interest_cost=after_tax_interest_cost,
        synergy_income=synergy_income,
        pro_forma_net_income=pro_forma_net_income,
        pro_forma_shares=pro_forma_shares,
        pro_forma_eps=pro_forma_eps,
        accretion_dilution_pct=accretion_dilution_pct,
        is_accretive=bool(accretion_dilution_pct > 0),
    )


def run_sensitivity(
    df: pd.DataFrame,
    acquirer_ticker: str,
    target_ticker: str,
    premium_range=(0.10, 0.20, 0.30, 0.40),
    cash_pct_range=(0.0, 0.25, 0.50, 0.75, 1.00),
    debt_funded_pct: float = 0.0,
    interest_rate: float = 0.08,
    tax_rate: float = 0.25,
    revenue_synergy_pct: float = 0.0,
    cost_synergy_pct: float = 0.0,
) -> pd.DataFrame:
    """Sweep premium % and cash/stock mix to show how sensitive the deal's
    accretion/dilution outcome is to the negotiated terms — this is the
    'toggle' version of the simulator, useful for a live demo.

    debt_funded_pct/interest_rate/tax_rate apply uniformly across the
    whole sweep (e.g. "assume any cash portion, at any cash_pct, is 100%
    debt-funded at 9% interest") — this is what makes the 100%-cash column
    of the sweep stop being flat/premium-insensitive, since financing a
    bigger cash payment with debt now costs more interest.

    revenue_synergy_pct/cost_synergy_pct likewise apply uniformly across
    the whole sweep (see DealInputs docstring for the synergy model)."""
    rows = []
    for premium in premium_range:
        for cash_pct in cash_pct_range:
            deal = DealInputs(
                acquirer_ticker, target_ticker, premium, cash_pct,
                debt_funded_pct=debt_funded_pct, interest_rate=interest_rate, tax_rate=tax_rate,
                revenue_synergy_pct=revenue_synergy_pct, cost_synergy_pct=cost_synergy_pct,
            )
            result = run_deal(df, deal)
            rows.append({
                "premium_pct": premium,
                "cash_pct": cash_pct,
                "pro_forma_eps": round(result.pro_forma_eps, 4),
                "accretion_dilution_pct": round(result.accretion_dilution_pct * 100, 2),
                "is_accretive": result.is_accretive,
            })
    return pd.DataFrame(rows)


def find_optimal_terms(
    df: pd.DataFrame,
    acquirer_ticker: str,
    target_ticker: str,
    premium_range=None,
    cash_pct_range=None,
    debt_funded_pct: float = 0.0,
    interest_rate: float = 0.08,
    tax_rate: float = 0.25,
    revenue_synergy_pct: float = 0.0,
    cost_synergy_pct: float = 0.0,
) -> dict:
    """Sweep a finer premium x cash-mix grid than run_sensitivity's default
    and identify the single combination that maximizes accretion (or
    minimizes dilution, if nothing in the grid is accretive). Returns both
    the winning combination and the full grid, so the grid can be plotted
    as a heatmap (e.g. in the dashboard) with the optimum highlighted.

    Default grid: premium 0-50% in 5% steps, cash 0-100% in 10% steps
    (121 combinations) — fine enough to be a useful heatmap, coarse enough
    to compute instantly since run_deal is pure arithmetic, no I/O."""
    if premium_range is None:
        premium_range = [round(0.05 * i, 2) for i in range(11)]  # 0.00 .. 0.50
    if cash_pct_range is None:
        cash_pct_range = [round(0.10 * i, 2) for i in range(11)]  # 0.00 .. 1.00

    grid = run_sensitivity(
        df, acquirer_ticker, target_ticker,
        premium_range=premium_range, cash_pct_range=cash_pct_range,
        debt_funded_pct=debt_funded_pct, interest_rate=interest_rate, tax_rate=tax_rate,
        revenue_synergy_pct=revenue_synergy_pct, cost_synergy_pct=cost_synergy_pct,
    )

    best_idx = grid["accretion_dilution_pct"].idxmax()
    best = grid.loc[best_idx].to_dict()

    return {"best": best, "grid": grid}


def project_multi_year_accretion(
    df: pd.DataFrame,
    deal: DealInputs,
    years: int = 3,
    synergy_ramp: tuple = (0.25, 0.60, 1.00),
) -> pd.DataFrame:
    """Multi-year accretion/dilution projection with synergies phased in
    over time, instead of the single-period "full run-rate, day one" view
    run_deal() gives. Real deals don't realize 100% of modeled synergies
    immediately — integration takes time — so a static one-period number
    overstates the near-term picture and understates the longer-term one.

    What changes year over year:
      - Acquirer's and target's own standalone net income grow at their
        own revenue_growth rate (from the scored universe) — used as an
        earnings-growth proxy, consistent with how revenue_growth is used
        as the fundamental growth signal elsewhere in this project.
      - Synergy income (both revenue and cost synergy, using the SAME
        revenue_synergy_pct/cost_synergy_pct as deal) is scaled by
        synergy_ramp[year-1] — e.g. the default (0.25, 0.60, 1.00) means
        25% of full run-rate synergies land in year 1, 60% by year 2,
        100% (full run-rate) by year 3.

    What stays STATIC across years (explicit scope limitation, not an
    oversight): the deal structure itself — shares issued, debt raised,
    interest cost — is fixed at entry. This model does NOT run a debt
    paydown schedule over the projection period; that is deliberately the
    LBO model's job, not a strategic accretion/dilution model's. A real
    banker's model for a strategic (non-LBO) acquirer often does the same
    simplification for a quick accretion/dilution view, though a fuller
    model would amortize acquisition debt over the projection too.

    Missing revenue_growth for a company defaults to 0% (flat) rather
    than raising an error — this is a refinement on top of the core deal
    math, not something that should block the calculation entirely.
    """
    # Entry deal structure computed ONCE, with synergies zeroed out here
    # since this function models them separately, year by year, rather
    # than through run_deal's own single-period synergy calc.
    entry_deal = DealInputs(
        deal.acquirer_ticker, deal.target_ticker, deal.premium_pct, deal.cash_pct,
        debt_funded_pct=deal.debt_funded_pct, interest_rate=deal.interest_rate, tax_rate=deal.tax_rate,
        avg_price_days=deal.avg_price_days, refinance_target_debt=deal.refinance_target_debt,
        revenue_synergy_pct=0.0, cost_synergy_pct=0.0,
    )
    entry = run_deal(df, entry_deal)

    acquirer_row = df.loc[df["ticker"] == deal.acquirer_ticker].iloc[0]
    target_row = df.loc[df["ticker"] == deal.target_ticker].iloc[0]
    acquirer_growth = acquirer_row.get("revenue_growth")
    target_growth = target_row.get("revenue_growth")
    acquirer_growth = 0.0 if pd.isna(acquirer_growth) else acquirer_growth
    target_growth = 0.0 if pd.isna(target_growth) else target_growth

    acquirer_fund = _company_fundamentals(df, deal.acquirer_ticker, avg_price_days=deal.avg_price_days)
    target_fund = _company_fundamentals(df, deal.target_ticker, avg_price_days=deal.avg_price_days)
    acquirer_ni_base = acquirer_fund["net_income"]
    target_ni_base = target_fund["net_income"]
    acquirer_shares_standalone = acquirer_fund["shares_outstanding"]

    rows = []
    for year in range(1, years + 1):
        ramp = synergy_ramp[year - 1] if year - 1 < len(synergy_ramp) else synergy_ramp[-1]

        acquirer_ni_year = acquirer_ni_base * (1 + acquirer_growth) ** year
        target_ni_year = target_ni_base * (1 + target_growth) ** year
        combined_pre_synergy_ni = acquirer_ni_year + target_ni_year

        revenue_synergy_income = target_ni_year * deal.revenue_synergy_pct * ramp
        cost_synergy_income = combined_pre_synergy_ni * deal.cost_synergy_pct * (1 - deal.tax_rate) * ramp
        synergy_income_year = revenue_synergy_income + cost_synergy_income

        pro_forma_ni_year = combined_pre_synergy_ni + synergy_income_year - entry.after_tax_interest_cost
        pro_forma_eps_year = pro_forma_ni_year / entry.pro_forma_shares

        acquirer_standalone_eps_year = acquirer_ni_year / acquirer_shares_standalone

        accretion_dilution_pct_year = (pro_forma_eps_year - acquirer_standalone_eps_year) / acquirer_standalone_eps_year

        rows.append({
            "year": year,
            "synergy_ramp_pct": round(ramp * 100, 1),
            "acquirer_standalone_eps": round(acquirer_standalone_eps_year, 4),
            "pro_forma_eps": round(pro_forma_eps_year, 4),
            "synergy_income": round(synergy_income_year, 0),
            "accretion_dilution_pct": round(accretion_dilution_pct_year * 100, 2),
            "is_accretive": bool(accretion_dilution_pct_year > 0),
        })

    return pd.DataFrame(rows)


def find_best_targets(
    df: pd.DataFrame,
    acquirer_ticker: str,
    premium_pct: float = 0.25,
    cash_pct: float = 0.5,
    debt_funded_pct: float = 0.0,
    interest_rate: float = 0.08,
    tax_rate: float = 0.25,
    max_target_size_pct: float = 1.0,
    top_n: int = None,
    include_acquisition_likelihood: bool = True,
) -> pd.DataFrame:
    """For a given acquirer, simulate the SAME deal terms (premium, cash/
    stock mix, financing assumptions) against every other company in the
    universe, and rank them by resulting accretion. This is what turns the
    simulator from "check one specific pair" into "which of these ~200
    companies should this acquirer actually be looking at."

    Fixed deal terms are used across every candidate (not individually
    optimized per target) so the ranking answers a consistent question —
    "which target is the best fit under these terms" — rather than "which
    target has the best best-case deal," which would be a different (and
    less useful) comparison since every target would get its own most
    favorable premium/mix.

    SIZE FEASIBILITY FILTER: a mega-cap acquirer can make almost ANY
    target look wildly accretive purely due to scale mismatch (adding a
    huge target's net income to a much larger acquirer barely dilutes the
    ratio) — this is a well-known limitation of accretion/dilution as a
    metric on its own, and without a check for it, a small-cap acquirer
    "acquiring" a company many times its own size (structurally
    implausible — real acquirers are overwhelmingly larger than their
    targets) would show up looking like a great deal. max_target_size_pct
    (default 1.0 = target's market cap can't exceed the acquirer's own)
    filters these out. Every candidate's relative_size_pct is still shown
    for transparency even when it passes the filter, so the person using
    this can judge for themselves rather than trust an invisible cutoff.

    ACQUISITION LIKELIHOOD TIE-IN: when include_acquisition_likelihood is
    True (default), each target's acquisition_likelihood_score is merged
    in — this answers a genuinely different question than accretion alone
    ("is this target ALSO a classic cheap/undervalued candidate, or is it
    accretive purely because of scale/financing mechanics"). Computed once
    for the whole universe before the per-target loop, not per-candidate,
    since it doesn't depend on the acquirer. If the required columns
    aren't available, this is skipped with a warning rather than failing
    the whole scan — the primary accretion ranking shouldn't break just
    because a secondary signal is unavailable.

    Candidates that fail (missing required fields for that specific
    company) are skipped and logged, not allowed to crash the whole scan —
    same error-isolation discipline as ingestion.py's per-ticker handling.
    """
    if acquirer_ticker not in df["ticker"].values:
        raise ValueError(f"'{acquirer_ticker}' not found in the scored universe")

    acquirer_row = df.loc[df["ticker"] == acquirer_ticker].iloc[0]
    if pd.isna(acquirer_row.get("market_cap")) or acquirer_row["market_cap"] <= 0:
        raise ValueError(f"Acquirer '{acquirer_ticker}' is missing a valid market_cap — can't assess relative deal size.")
    acquirer_market_cap = acquirer_row["market_cap"]

    likelihood_map = {}
    if include_acquisition_likelihood:
        try:
            from src.acquisition_likelihood import score_acquisition_likelihood
            likelihood_df = score_acquisition_likelihood(df)
            likelihood_map = likelihood_df.set_index("ticker")[
                ["acquisition_likelihood_score", "acquisition_likelihood_rank"]
            ].to_dict("index")
        except ValueError as e:
            logger.warning(f"Skipped acquisition-likelihood tie-in for best-targets output: {e}")

    candidate_tickers = df.loc[df["ticker"] != acquirer_ticker, "ticker"].tolist()
    rows, skipped, oversized = [], [], []

    for target_ticker in candidate_tickers:
        try:
            target_row = df.loc[df["ticker"] == target_ticker].iloc[0]
            target_market_cap = target_row.get("market_cap")
            relative_size_pct = (
                (target_market_cap / acquirer_market_cap) if pd.notna(target_market_cap) and target_market_cap > 0 else None
            )

            deal = DealInputs(
                acquirer_ticker, target_ticker, premium_pct, cash_pct,
                debt_funded_pct=debt_funded_pct, interest_rate=interest_rate, tax_rate=tax_rate,
            )
            result = run_deal(df, deal)

            if relative_size_pct is not None and relative_size_pct > max_target_size_pct:
                oversized.append(target_ticker)
                continue

            row_data = {
                "target": target_ticker,
                "sector": target_row.get("sector"),
                "relative_size_pct": round(relative_size_pct * 100, 1) if relative_size_pct is not None else None,
                "accretion_dilution_pct": round(result.accretion_dilution_pct * 100, 2),
                "pro_forma_eps": round(result.pro_forma_eps, 4),
                "is_accretive": result.is_accretive,
                "deal_value": round(result.deal_value, 0),
            }
            # Tie the target's own screening quality into the output —
            # a target that's accretive purely on scale mismatch but a
            # genuinely weak business (low composite_score) is a very
            # different finding than one that's both accretive AND
            # highly ranked by the multi-factor screen.
            if "composite_score" in target_row.index:
                row_data["target_composite_score"] = target_row.get("composite_score")
            if "sector_rank" in target_row.index:
                row_data["target_sector_rank"] = target_row.get("sector_rank")
            if target_ticker in likelihood_map:
                row_data["target_acquisition_likelihood_score"] = likelihood_map[target_ticker]["acquisition_likelihood_score"]
                row_data["target_acquisition_likelihood_rank"] = likelihood_map[target_ticker]["acquisition_likelihood_rank"]

            rows.append(row_data)
        except ValueError:
            skipped.append(target_ticker)
            continue

    if skipped:
        logger.warning(
            f"{len(skipped)}/{len(candidate_tickers)} candidate targets skipped "
            f"(missing required fields, e.g. price/market_cap/ROE/total_equity, "
            f"or negative equity): {skipped[:10]}{'...' if len(skipped) > 10 else ''}"
        )
    if oversized:
        logger.info(
            f"{len(oversized)}/{len(candidate_tickers)} candidates excluded as structurally "
            f"implausible targets (market cap exceeds {max_target_size_pct:.0%} of {acquirer_ticker}'s "
            f"own market cap): {oversized[:10]}{'...' if len(oversized) > 10 else ''}"
        )

    if not rows:
        raise ValueError(
            f"No valid targets could be evaluated for acquirer '{acquirer_ticker}' after "
            f"skipping missing-data and oversized candidates — try raising max_target_size_pct."
        )

    result_df = pd.DataFrame(rows).sort_values("accretion_dilution_pct", ascending=False).reset_index(drop=True)
    result_df["rank"] = result_df.index + 1

    return result_df.head(top_n) if top_n else result_df


def _parse_args():
    parser = argparse.ArgumentParser(description="Accretion/dilution simulator")
    parser.add_argument("--acquirer", required=True, help="Acquirer ticker, e.g. TCS.NS")
    parser.add_argument("--target", default=None, help="Target ticker, e.g. COFORGE.NS (not needed with --best-targets)")
    parser.add_argument("--premium", type=float, default=0.25, help="Premium over target price, e.g. 0.25 for 25%")
    parser.add_argument("--cash-pct", type=float, default=0.5, help="Fraction of deal paid in cash, e.g. 0.5")
    parser.add_argument("--debt-funded-pct", type=float, default=0.0, help="Fraction of the CASH portion funded via new debt rather than reserves, e.g. 1.0 = fully debt-funded")
    parser.add_argument("--interest-rate", type=float, default=0.08, help="Assumed interest rate on new debt, e.g. 0.08 for 8%")
    parser.add_argument("--tax-rate", type=float, default=0.25, help="Assumed effective tax rate for the interest tax shield, e.g. 0.25")
    parser.add_argument("--avg-price-days", type=int, default=0, help="If >0, price the deal off a trailing N-day average close (fetched live via yfinance) instead of spot price, e.g. 30")
    parser.add_argument("--refinance-target-debt", action="store_true", help="Also refinance the target's existing total_debt at the same assumed interest_rate/tax_rate")
    parser.add_argument("--revenue-synergy-pct", type=float, default=0.0, help="Assumed % uplift to the target's net income from revenue synergies, e.g. 0.10 for 10%")
    parser.add_argument("--cost-synergy-pct", type=float, default=0.0, help="Assumed % of combined pre-synergy net income recovered via cost synergies, e.g. 0.05 for 5%")
    parser.add_argument("--sensitivity", action="store_true", help="Run a premium x cash-mix sensitivity sweep instead of a single scenario")
    parser.add_argument("--optimize", action="store_true", help="Find the premium/cash-mix combination that maximizes accretion, over a finer grid than --sensitivity")
    parser.add_argument("--best-targets", action="store_true", help="Scan every other company in the universe as a potential target for --acquirer, using the given deal terms, ranked by accretion")
    parser.add_argument("--multi-year", action="store_true", help="Project accretion/dilution over multiple years with synergies phased in via --synergy-ramp, instead of a single-period view")
    parser.add_argument("--years", type=int, default=3, help="Number of years to project with --multi-year, e.g. 3")
    parser.add_argument("--synergy-ramp", type=str, default="0.25,0.60,1.00", help="Comma-separated synergy realization schedule for --multi-year, e.g. '0.25,0.60,1.00' for 25%%/60%%/100%% by year")
    parser.add_argument("--top-n", type=int, default=10, help="Number of top targets to show with --best-targets, e.g. 10")
    parser.add_argument("--max-target-size-pct", type=float, default=1.0, help="With --best-targets: exclude candidates whose market cap exceeds this fraction of the acquirer's own (default 1.0 = target can't be larger than acquirer)")
    parser.add_argument("--universe-path", default="data/processed/scored_universe.csv")
    parser.add_argument("--output-dir", default="data/processed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    df = load_universe(args.universe_path)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.best_targets and not args.target:
        raise SystemExit("--target is required unless --best-targets is set")

    if args.multi_year:
        deal = DealInputs(args.acquirer, args.target, args.premium, args.cash_pct,
                           debt_funded_pct=args.debt_funded_pct, interest_rate=args.interest_rate, tax_rate=args.tax_rate,
                           avg_price_days=args.avg_price_days, refinance_target_debt=args.refinance_target_debt,
                           revenue_synergy_pct=args.revenue_synergy_pct, cost_synergy_pct=args.cost_synergy_pct)
        ramp = tuple(float(x) for x in args.synergy_ramp.split(","))
        result = project_multi_year_accretion(df, deal, years=args.years, synergy_ramp=ramp)
        out_path = Path(args.output_dir) / f"multi_year_accretion_{args.acquirer.replace('.', '_')}_{args.target.replace('.', '_')}.csv"
        result.to_csv(out_path, index=False)
        print(f"\nMulti-year accretion/dilution: {args.acquirer} acquires {args.target}")
        print(f"Synergy ramp: {ramp} | Premium: {args.premium:.0%} | Cash/Stock: {args.cash_pct:.0%}/{1-args.cash_pct:.0%}")
        print(f"\n{result.to_string(index=False)}")
        print(f"\nSaved -> {out_path}")
        print("\nNote: deal structure (shares issued, debt, interest cost) is fixed at entry — "
              "no debt paydown schedule is modeled over the projection period (that's the LBO model's job).")
    elif args.best_targets:
        result = find_best_targets(
            df, args.acquirer,
            premium_pct=args.premium, cash_pct=args.cash_pct,
            debt_funded_pct=args.debt_funded_pct, interest_rate=args.interest_rate, tax_rate=args.tax_rate,
            max_target_size_pct=args.max_target_size_pct,
        )
        out_path = Path(args.output_dir) / f"best_targets_for_{args.acquirer.replace('.', '_')}.csv"
        result.to_csv(out_path, index=False)

        print(f"\nBest acquisition targets for {args.acquirer}")
        print(f"(deal terms: {args.premium:.0%} premium, {args.cash_pct:.0%} cash / {1 - args.cash_pct:.0%} stock"
              + (f", {args.debt_funded_pct:.0%} of cash debt-funded @ {args.interest_rate:.1%}" if args.debt_funded_pct > 0 else "")
              + f", targets capped at {args.max_target_size_pct:.0%} of acquirer's market cap)")
        print(f"\nTop {args.top_n} of {len(result)} evaluated targets:")
        print(result.head(args.top_n).to_string(index=False))
        print(f"\nFull ranked list ({len(result)} targets) saved -> {out_path}")
    elif args.optimize:
        result = find_optimal_terms(
            df, args.acquirer, args.target,
            debt_funded_pct=args.debt_funded_pct, interest_rate=args.interest_rate, tax_rate=args.tax_rate,
        )
        best, grid = result["best"], result["grid"]
        out_path = Path(args.output_dir) / f"accretion_optimal_grid_{args.acquirer.replace('.', '_')}_{args.target.replace('.', '_')}.csv"
        grid.to_csv(out_path, index=False)
        verdict = "ACCRETIVE" if best["is_accretive"] else "DILUTIVE"
        print(f"\nOptimal deal terms: {args.acquirer} acquiring {args.target}")
        if args.debt_funded_pct > 0:
            print(f"(cash portion assumed {args.debt_funded_pct:.0%} debt-funded @ {args.interest_rate:.1%} interest, {args.tax_rate:.0%} tax rate)")
        print(f"Best premium: {best['premium_pct']:.0%} | Best cash/stock mix: {best['cash_pct']:.0%} cash / {1 - best['cash_pct']:.0%} stock")
        print(f"Result: {verdict} by {abs(best['accretion_dilution_pct']):.2f}%")
        print(f"\nFull grid ({len(grid)} combinations) saved -> {out_path}")
    elif args.sensitivity:
        table = run_sensitivity(
            df, args.acquirer, args.target,
            debt_funded_pct=args.debt_funded_pct, interest_rate=args.interest_rate, tax_rate=args.tax_rate,
        )
        out_path = Path(args.output_dir) / f"accretion_sensitivity_{args.acquirer.replace('.', '_')}_{args.target.replace('.', '_')}.csv"
        table.to_csv(out_path, index=False)
        print(f"\nSensitivity sweep: {args.acquirer} acquiring {args.target}")
        if args.debt_funded_pct > 0:
            print(f"(cash portion assumed {args.debt_funded_pct:.0%} debt-funded @ {args.interest_rate:.1%} interest, {args.tax_rate:.0%} tax rate)")
        print(table.to_string(index=False))
        print(f"\nSaved -> {out_path}")
    else:
        deal = DealInputs(args.acquirer, args.target, args.premium, args.cash_pct,
                           debt_funded_pct=args.debt_funded_pct, interest_rate=args.interest_rate, tax_rate=args.tax_rate,
                           avg_price_days=args.avg_price_days, refinance_target_debt=args.refinance_target_debt,
                           revenue_synergy_pct=args.revenue_synergy_pct, cost_synergy_pct=args.cost_synergy_pct)
        result = run_deal(df, deal)
        verdict = "ACCRETIVE" if result.is_accretive else "DILUTIVE"
        print(f"\n{args.acquirer} acquires {args.target}")
        print(f"Premium: {args.premium:.0%} | Cash/Stock mix: {args.cash_pct:.0%} cash / {1 - args.cash_pct:.0%} stock")
        if args.debt_funded_pct > 0 or args.refinance_target_debt:
            print(f"Debt financing: {args.debt_funded_pct:.0%} of cash portion via new debt "
                  f"@ {args.interest_rate:.1%} interest, {args.tax_rate:.0%} tax rate")
            if args.refinance_target_debt:
                print(f"Target's existing debt refinanced: {result.target_debt_refinanced:,.0f}")
            print(f"New debt raised: {result.new_debt_raised:,.0f} | After-tax interest cost: {result.after_tax_interest_cost:,.0f}")
        if args.avg_price_days > 0:
            print(f"Deal priced off a {args.avg_price_days}-day average close, not spot price")
        if args.revenue_synergy_pct > 0 or args.cost_synergy_pct > 0:
            print(f"Synergies: {args.revenue_synergy_pct:.0%} revenue uplift on target NI, "
                  f"{args.cost_synergy_pct:.0%} cost synergy on combined NI (tax-effected)")
            print(f"Total synergy income: {result.synergy_income:,.0f}")
        print(f"\nAcquirer standalone EPS: {result.acquirer_standalone_eps:.4f}")
        print(f"Pro-forma EPS:           {result.pro_forma_eps:.4f}")
        print(f"\nResult: {verdict} by {abs(result.accretion_dilution_pct):.2%}")
