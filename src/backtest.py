"""
backtest.py — Performance validation for the M&A screener.

Two genuinely different things live here, and they must not be confused:

1. validate_forward_performance() — a REAL, UNBIASED backtest. Uses
   snapshots recorded by history_tracker.py at some point in the PAST,
   checks REAL price performance from that recorded date to today. No
   look-ahead bias: the ranking being checked is exactly what the model
   said on that day, and the return is what actually happened afterward.
   THIS IS THE ONLY LEGITIMATE BACKTEST IN THIS MODULE. It will correctly
   report "insufficient history" until history_tracker.py has been
   running for at least as long as the lookback window requested.

2. compute_trailing_performance() — a RETROSPECTIVE, INFORMATIONAL check
   only, NOT a backtest. It takes TODAY's composite-score ranking and
   checks how those same companies' prices performed over a PAST window.
   This has a real, structural bias: it uses today's fundamentals (P/E,
   ROE, leverage, etc.) to justify a ranking, then checks price
   performance during a period when those fundamentals may have looked
   completely different. A company that's cheap today might have been
   expensive 6 months ago — we have no way of knowing whether the model
   would have picked the SAME companies back then. This function exists
   because "how have today's top picks performed recently" is still a
   commonly-reported, informational stat — but it proves nothing about
   the model's predictive power, and the function's own output says so.

Usage:
    python -m src.backtest --check trailing --lookback-days 180
    python -m src.backtest --check forward --min-days-elapsed 30
"""

import argparse
import logging

import pandas as pd

from src.baseline_comparison import compute_naive_rank
from src.history_tracker import load_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_price_return(ticker: str, start_date: str = None, lookback_days: int = None) -> float:
    """Real % price return for a ticker, either from start_date to today,
    or over the trailing lookback_days. Exactly one of the two must be
    given. Returns None if the price history can't be fetched (delisted,
    invalid ticker, etc.) rather than raising — a single bad ticker
    shouldn't crash a whole comparison, same discipline as ingestion.py."""
    import yfinance as yf

    try:
        if start_date is not None:
            hist = yf.Ticker(ticker).history(start=start_date)
        else:
            hist = yf.Ticker(ticker).history(period=f"{lookback_days}d")

        if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < 2:
            return None

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        if start_price <= 0:
            return None
        return (end_price - start_price) / start_price
    except Exception as e:
        logger.warning(f"[{ticker}] price return fetch failed: {e}")
        return None


def compute_trailing_performance(
    scored_df: pd.DataFrame,
    lookback_days: int = 180,
    top_n: int = 10,
) -> dict:
    """RETROSPECTIVE CHECK ONLY — see module docstring. Compares trailing
    price returns of today's composite-score top-N, today's naive-P/E
    top-N, and the full universe average, over the same historical window."""
    composite_top = scored_df.sort_values("rank").head(top_n)["ticker"].tolist()

    naive_ranked = compute_naive_rank(scored_df)
    naive_top = naive_ranked.sort_values("naive_rank").head(top_n)["ticker"].tolist()

    all_tickers = scored_df["ticker"].tolist()

    returns = {}
    for ticker in set(composite_top + naive_top + all_tickers):
        returns[ticker] = fetch_price_return(ticker, lookback_days=lookback_days)

    def _group_avg(tickers):
        vals = [returns[t] for t in tickers if returns.get(t) is not None]
        return (sum(vals) / len(vals)) if vals else None

    composite_avg = _group_avg(composite_top)
    naive_avg = _group_avg(naive_top)
    universe_avg = _group_avg(all_tickers)

    return {
        "lookback_days": lookback_days,
        "composite_top_n_avg_return": composite_avg,
        "naive_top_n_avg_return": naive_avg,
        "universe_avg_return": universe_avg,
        "per_ticker_returns": returns,
        "caveat": (
            "RETROSPECTIVE CHECK, NOT A BACKTEST: uses TODAY's fundamental ranking "
            "against PAST price performance. Does not prove the model would have "
            "picked these same companies at the start of the lookback window, since "
            "fundamentals change over time. For a genuine, unbiased forward-return "
            "validation, use validate_forward_performance() once enough history has "
            "been recorded by history_tracker.py."
        ),
    }


def validate_forward_performance(
    history_path: str = "data/history/shortlist_history.csv",
    min_days_elapsed: int = 30,
) -> dict:
    """THE REAL BACKTEST. Looks at snapshots recorded by history_tracker.py
    at least min_days_elapsed ago, and checks REAL price performance from
    that recorded date to today. This is unbiased: the ranking being
    checked is exactly what the model said on that historical day, and
    the return is genuine subsequent (forward) performance — no look-ahead
    bias, unlike compute_trailing_performance() above.

    Will correctly report insufficient history until history_tracker.py
    has been running for at least min_days_elapsed."""
    history = load_history(history_path)
    if history.empty:
        return {
            "status": "insufficient_history",
            "message": (
                f"No history recorded yet at {history_path}. This backtest becomes "
                f"meaningful once history_tracker.py has been running daily for at "
                f"least {min_days_elapsed} days — run `python -m src.history_tracker` "
                f"(or the full pipeline) daily and check back later."
            ),
        }

    history["snapshot_date"] = pd.to_datetime(history["snapshot_date"])
    cutoff = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=min_days_elapsed)
    eligible = history[history["snapshot_date"] <= cutoff]

    if eligible.empty:
        oldest = history["snapshot_date"].min()
        days_tracked = (pd.Timestamp.now(tz="UTC").tz_localize(None) - oldest).days
        return {
            "status": "insufficient_history",
            "message": (
                f"History tracking started {days_tracked} day(s) ago (oldest snapshot: "
                f"{oldest.date()}), but {min_days_elapsed} days are needed for a forward "
                f"return check. Check back in {min_days_elapsed - days_tracked} more day(s)."
            ),
        }

    as_of_date = eligible["snapshot_date"].min()
    as_of_snapshot = eligible[eligible["snapshot_date"] == as_of_date]

    returns = {}
    for ticker in as_of_snapshot["ticker"]:
        returns[ticker] = fetch_price_return(ticker, start_date=as_of_date.strftime("%Y-%m-%d"))

    valid_returns = {t: r for t, r in returns.items() if r is not None}
    avg_return = (sum(valid_returns.values()) / len(valid_returns)) if valid_returns else None

    return {
        "status": "ok",
        "as_of_date": as_of_date.strftime("%Y-%m-%d"),
        "tickers_checked": list(returns.keys()),
        "per_ticker_forward_returns": returns,
        "avg_forward_return": avg_return,
        "note": (
            f"This IS a genuine forward-return check: the shortlist recorded on "
            f"{as_of_date.date()} is being validated against REAL price performance "
            f"from that date to today — no look-ahead bias."
        ),
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Backtest / performance validation for the M&A screener")
    parser.add_argument("--check", choices=["trailing", "forward"], default="trailing",
                         help="'trailing' = retrospective check on today's picks (informational only). "
                              "'forward' = genuine backtest using recorded history (needs accumulated history).")
    parser.add_argument("--lookback-days", type=int, default=180, help="For --check trailing: how many days back to look")
    parser.add_argument("--min-days-elapsed", type=int, default=30, help="For --check forward: minimum days of history needed")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--scored-path", default="data/processed/scored_universe.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.check == "forward":
        result = validate_forward_performance(min_days_elapsed=args.min_days_elapsed)
        if result["status"] == "insufficient_history":
            print(f"\n{result['message']}")
        else:
            print(f"\n=== Forward-Return Validation (genuine backtest, no look-ahead bias) ===")
            print(f"As-of date: {result['as_of_date']}")
            print(f"\nPer-ticker forward returns:")
            for t, r in result["per_ticker_forward_returns"].items():
                print(f"  {t}: {r:+.2%}" if r is not None else f"  {t}: (unavailable)")
            if result["avg_forward_return"] is not None:
                print(f"\nAverage forward return: {result['avg_forward_return']:+.2%}")
            print(f"\n{result['note']}")
    else:
        df = pd.read_csv(args.scored_path)
        result = compute_trailing_performance(df, lookback_days=args.lookback_days, top_n=args.top_n)
        print(f"\n=== Trailing Performance Check ({args.lookback_days}-day lookback) ===\n")
        print(f"Composite-score top-{args.top_n} avg return: "
              f"{result['composite_top_n_avg_return']:+.2%}" if result['composite_top_n_avg_return'] is not None else "N/A")
        print(f"Naive P/E-only top-{args.top_n} avg return:  "
              f"{result['naive_top_n_avg_return']:+.2%}" if result['naive_top_n_avg_return'] is not None else "N/A")
        print(f"Full universe avg return:            "
              f"{result['universe_avg_return']:+.2%}" if result['universe_avg_return'] is not None else "N/A")
        print(f"\n⚠️  {result['caveat']}")
