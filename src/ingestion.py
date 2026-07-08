"""
ingestion.py — Data ingestion layer for the M&A screener.

Pulls price, market cap, and financial statement data for each ticker in the
configured universe. Design goals:
  - One bad ticker should never crash the whole run (error isolation)
  - Cache raw pulls locally so repeated runs don't hammer the data source
  - Every row is tagged with a fetch timestamp for auditability

Usage:
    python -m src.ingestion
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TickerSnapshot:
    ticker: str
    sector: str | None
    fetched_at: str
    market_cap: float | None
    price: float | None
    trailing_pe: float | None
    ev_to_ebitda: float | None
    return_on_equity: float | None
    debt_to_equity: float | None
    revenue_growth: float | None
    net_margin: float | None
    asset_turnover: float | None
    total_debt: float | None
    total_equity: float | None
    fetch_ok: bool
    error: str | None = None


def load_config(config_path: str = "config/universe.yaml") -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    _validate_config(config, config_path)
    return config


def _validate_config(config: dict, config_path: str) -> None:
    """Fail fast with a clear, actionable error if the config is malformed
    — a typo'd ticker, duplicate entry, or weights that don't sum to 1.0
    should be caught here, not surfaced as a confusing failure three
    modules downstream. Added when the universe was expanded to 200
    manually-curated tickers, where these mistakes become much likelier."""
    if "universe" not in config or not config["universe"]:
        raise ValueError(f"{config_path}: 'universe' is missing or empty.")

    tickers = []
    for i, entry in enumerate(config["universe"]):
        if isinstance(entry, dict):
            ticker = entry.get("ticker")
            if not ticker:
                raise ValueError(f"{config_path}: universe entry #{i} is missing a 'ticker' field: {entry}")
            if not ticker.endswith(".NS"):
                raise ValueError(
                    f"{config_path}: ticker '{ticker}' (entry #{i}) doesn't end in '.NS' — "
                    f"yfinance needs the NSE suffix to resolve Indian tickers correctly."
                )
        else:
            ticker = entry
        tickers.append(ticker)

    duplicates = {t for t in tickers if tickers.count(t) > 1}
    if duplicates:
        raise ValueError(f"{config_path}: duplicate ticker(s) found in universe: {sorted(duplicates)}")

    if "scoring" in config and "weights" in config["scoring"]:
        weights = config["scoring"]["weights"]
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"{config_path}: scoring weights sum to {total:.4f}, not 1.0 — "
                f"check config['scoring']['weights']: {weights}"
            )


def _cache_path(ticker: str, cache_dir: str) -> Path:
    safe_name = ticker.replace(".", "_")
    return Path(cache_dir) / f"{safe_name}.json"


def _is_cache_fresh(path: Path, expiry_days: int) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=expiry_days)


def fetch_ticker(ticker: str, sector: str | None = None, retries: int = 2, backoff_seconds: float = 2.0) -> TickerSnapshot:
    """Fetch a single ticker's snapshot. Never raises — returns fetch_ok=False on failure."""
    last_error = None
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info

            market_cap = info.get("marketCap")
            price = info.get("currentPrice") or info.get("regularMarketPrice")
            trailing_pe = info.get("trailingPE")
            ev_to_ebitda = info.get("enterpriseToEbitda")
            roe = info.get("returnOnEquity")
            debt_to_equity = info.get("debtToEquity")
            revenue_growth = info.get("revenueGrowth")
            net_margin = info.get("profitMargins")

            # yfinance can return a 200 response with an almost-empty info
            # dict (Yahoo-side glitch, rate limit, or symbol lookup issue)
            # without raising an exception. If BOTH market_cap and price
            # are missing, the fetch didn't actually work — treat it as a
            # failure so it retries, instead of silently recording a
            # false-positive success with 12 null fields.
            if market_cap is None and price is None:
                raise ValueError(
                    f"yfinance returned an empty info payload for {ticker} "
                    f"(no exception, but no usable data — likely a transient "
                    f"Yahoo-side issue or an invalid/delisted symbol)"
                )

            total_debt = info.get("totalDebt")
            total_equity = None
            try:
                bs = t.balance_sheet
                if bs is not None and not bs.empty and "Common Stock Equity" in bs.index:
                    total_equity = float(bs.loc["Common Stock Equity"].iloc[0])
            except Exception:
                pass  # equity detail is a bonus field, not fetch-critical

            asset_turnover = None
            try:
                fin = t.financials
                if (
                    fin is not None and not fin.empty
                    and "Total Revenue" in fin.index
                    and total_equity
                ):
                    revenue = float(fin.loc["Total Revenue"].iloc[0])
                    asset_turnover = revenue / total_equity if total_equity else None
            except Exception:
                pass

            return TickerSnapshot(
                ticker=ticker,
                sector=sector,
                fetched_at=datetime.now(timezone.utc).isoformat(),
                market_cap=market_cap,
                price=price,
                trailing_pe=trailing_pe,
                ev_to_ebitda=ev_to_ebitda,
                return_on_equity=roe,
                debt_to_equity=debt_to_equity,
                revenue_growth=revenue_growth,
                net_margin=net_margin,
                asset_turnover=asset_turnover,
                total_debt=total_debt,
                total_equity=total_equity,
                fetch_ok=True,
            )
        except Exception as e:
            last_error = str(e)
            logger.warning(f"[{ticker}] attempt {attempt + 1} failed: {last_error}")
            if attempt < retries:
                time.sleep(backoff_seconds * (attempt + 1))

    return TickerSnapshot(
        ticker=ticker,
        sector=sector,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        market_cap=None, price=None, trailing_pe=None, ev_to_ebitda=None,
        return_on_equity=None, debt_to_equity=None, revenue_growth=None,
        net_margin=None, asset_turnover=None, total_debt=None, total_equity=None,
        fetch_ok=False, error=last_error,
    )


def fetch_universe(config: dict) -> list[TickerSnapshot]:
    cache_dir = config["data"]["cache_dir"]
    expiry_days = config["data"]["cache_expiry_days"]
    # Backward-compatible: older configs (pre-200-ticker expansion) won't
    # have this key at all — default to no delay rather than crash.
    request_delay = config["data"].get("request_delay_seconds", 0.0)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    snapshots = []
    for entry in config["universe"]:
        # Backward-compatible: supports both the new {ticker, sector} dict
        # format and the old plain-string ticker list.
        if isinstance(entry, dict):
            ticker, sector = entry["ticker"], entry.get("sector")
        else:
            ticker, sector = entry, None

        cache_file = _cache_path(ticker, cache_dir)

        if _is_cache_fresh(cache_file, expiry_days):
            logger.info(f"[{ticker}] using cached data")
            with open(cache_file, "r") as f:
                cached = json.load(f)
                cached.setdefault("sector", sector)  # backfill sector if cache predates this field
                snapshots.append(TickerSnapshot(**cached))
            continue

        logger.info(f"[{ticker}] fetching fresh data")
        snap = fetch_ticker(ticker, sector=sector)
        with open(cache_file, "w") as f:
            json.dump(asdict(snap), f, indent=2)
        snapshots.append(snap)

        if request_delay > 0:
            time.sleep(request_delay)

    ok_count = sum(1 for s in snapshots if s.fetch_ok)
    logger.info(f"Fetched {ok_count}/{len(snapshots)} tickers successfully")
    if ok_count < len(snapshots):
        failed = [s.ticker for s in snapshots if not s.fetch_ok]
        logger.warning(f"Failed tickers (excluded from downstream, not silently zeroed): {failed}")

    return snapshots


if __name__ == "__main__":
    cfg = load_config()
    results = fetch_universe(cfg)
    print(f"\nDone. {sum(1 for r in results if r.fetch_ok)}/{len(results)} tickers fetched.")
