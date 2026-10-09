"""
validate_returns.py - full-universe forward-return validation.

Complements src/backtest.py, which reports the top-10 shortlist's average
forward return with no benchmark and no dispersion. This adds:

  1. Benchmark comparison (^NSEI) so returns are EXCESS, not absolute
  2. Full universe grouped by rank decile, not just the shortlist
  3. Dispersion, bootstrap CI, and a jackknife single-name fragility check
  4. A loop over every as-of date with enough elapsed history

Prices are downloaded ONCE for the whole period and reused across every
as-of date, to avoid hammering yfinance with 200 x N requests.

Usage (from the repo root):
    python -m src.validate_returns
    python -m src.validate_returns --horizon 21 --deciles 5
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yfinance as yf

# Repo root = parent of the directory holding this file, so the snapshot glob
# resolves the same way whether you run from the root or from src/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR = os.path.join(REPO_ROOT, "snapshots")
SNAPSHOT_GLOB = os.path.join(SNAPSHOT_DIR, "**", "scored_universe.csv")
BENCHMARK = "^NSEI"


def _run_number(path):
    """snapshots/run-NN/... -> NN. Sort on this, not the path string:
    as text, run-100 sorts before run-99."""
    m = re.fullmatch(r"run-(\d+)", os.path.relpath(path, SNAPSHOT_DIR).split(os.sep)[0])
    return int(m.group(1)) if m else -1


def load_snapshots():
    """Read every archived scored_universe.csv into one tidy frame."""
    frames = []
    paths = glob.glob(SNAPSHOT_GLOB, recursive=True)
    for path in sorted(paths, key=lambda p: (_run_number(p), p)):
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"  skip {os.path.basename(os.path.dirname(path))}: {exc}")
            continue
        if not {"fetched_at", "rank", "ticker"} <= set(df.columns):
            continue
        as_of = pd.to_datetime(df["fetched_at"], errors="coerce", utc=True)
        df = df.assign(as_of=as_of.dt.tz_localize(None).dt.normalize())
        frames.append(df[["as_of", "ticker", "rank"]])

    if not frames:
        raise SystemExit(f"No snapshots matched {SNAPSHOT_GLOB}")

    out = pd.concat(frames, ignore_index=True).dropna(subset=["as_of", "rank"])
    # Manual re-runs can produce two snapshots for one calendar day.
    # Keep one observation per (date, ticker) so those don't double-count.
    # Runs are read oldest first, so this keeps the day's earliest run.
    return out.drop_duplicates(subset=["as_of", "ticker"], keep="first")


def fetch_prices(tickers, start, end):
    """Adjusted close for every ticker over the whole window, tz-naive index."""
    tickers = list(tickers)
    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=False,
    )
    if raw is None or len(raw) == 0:
        raise SystemExit("yfinance returned no price data")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in set(raw.columns.get_level_values(0)):
            px = raw["Close"].copy()
        else:
            px = raw.xs("Close", axis=1, level=-1).copy()
    else:
        px = raw[["Close"]].copy()
        px.columns = tickers[:1]

    idx = pd.to_datetime(px.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    px.index = idx
    return px.dropna(how="all")


def window_return(px, as_of, horizon_days):
    """
    Percent return from the first trading day on/after as_of, to the last
    trading day within horizon_days. Returns (values, start, end) or Nones.
    """
    idx = px.index
    on_or_after = idx[idx >= as_of]
    within = idx[idx <= as_of + pd.Timedelta(days=horizon_days)]
    if len(on_or_after) == 0 or len(within) == 0:
        return None, None, None
    start, end = on_or_after[0], within[-1]
    if end <= start:
        return None, None, None
    return (px.loc[end] / px.loc[start] - 1.0) * 100.0, start, end


def bootstrap_ci(values, n_boot=5000, seed=0):
    """Percentile bootstrap 95% CI for the mean."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return lo, hi


def jackknife(values):
    """
    Leave-one-out: which single observation moves the mean most, and what
    the mean becomes without it. Catches results carried by one lucky name.
    """
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return None, np.nan
    loo = np.array([np.delete(x, i).mean() for i in range(len(x))])
    worst = int(np.argmax(np.abs(loo - x.mean())))
    return worst, loo[worst]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int, default=30,
                    help="forward return window in calendar days (default 30)")
    ap.add_argument("--deciles", type=int, default=10,
                    help="number of rank buckets (default 10)")
    ap.add_argument("--min-days-elapsed", type=int, default=None,
                    help="skip as-of dates newer than this (default = horizon)")
    args = ap.parse_args()

    horizon = args.horizon
    n_buckets = args.deciles
    min_elapsed = args.min_days_elapsed if args.min_days_elapsed is not None else horizon

    snaps = load_snapshots()
    tickers = sorted(snaps["ticker"].dropna().unique().tolist())
    start = snaps["as_of"].min() - pd.Timedelta(days=5)
    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    print(f"{len(tickers)} tickers across {snaps['as_of'].nunique()} distinct dates")
    print(f"Downloading prices {start.date()} -> {end.date()} ...")
    px = fetch_prices(tickers, start, end)
    bench = fetch_prices([BENCHMARK], start, end)
    print(f"Got price history for {px.shape[1]} of {len(tickers)} tickers\n")

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=min_elapsed)
    dates = sorted(d for d in snaps["as_of"].unique() if d <= cutoff)
    if not dates:
        print(f"No as-of date has {min_elapsed}+ days elapsed yet. "
              f"Wait, or lower --horizon.")
        return

    rows = []
    for as_of in dates:
        rets, win_start, win_end = window_return(px, as_of, horizon)
        if rets is None:
            continue
        b_vals, _, _ = window_return(bench, as_of, horizon)
        b = float(np.asarray(b_vals, dtype=float).ravel()[0]) if b_vals is not None else np.nan

        day = snaps[snaps["as_of"] == as_of].copy()
        day["fwd"] = day["ticker"].map(rets)
        day = day.dropna(subset=["fwd"])
        if len(day) < n_buckets * 2:
            print(f"as-of {pd.Timestamp(as_of).date()}: only {len(day)} usable rows, skipping")
            continue

        day["excess"] = day["fwd"] - b
        day["bucket"] = pd.qcut(
            day["rank"].rank(method="first"),
            n_buckets, labels=False, duplicates="drop",
        ) + 1

        print(f"=== as-of {pd.Timestamp(as_of).date()}  "
              f"({win_start.date()} -> {win_end.date()}, n={len(day)}) ===")
        print(f"benchmark {BENCHMARK}: {b:+.2f}%")
        summary = day.groupby("bucket")["excess"].agg(["mean", "std", "count"])
        print(summary.round(2).to_string())

        top = day[day["bucket"] == 1]["excess"]
        bot = day[day["bucket"] == day["bucket"].max()]["excess"]
        spread = top.mean() - bot.mean()
        lo, hi = bootstrap_ci(top.values)
        wi, wm = jackknife(top.values)

        print(f"top bucket excess : {top.mean():+.2f}%   95% CI [{lo:+.2f}, {hi:+.2f}]")
        print(f"top minus bottom  : {spread:+.2f}%")
        if wi is not None:
            name = day.loc[top.index[wi], "ticker"]
            print(f"drop {name:<16s}: top bucket becomes {wm:+.2f}%")
        print()

        rows.append({"as_of": as_of, "top": top.mean(),
                     "bottom": bot.mean(), "spread": spread, "bench": b})

    if not rows:
        print("No window produced a usable result.")
        return

    agg = pd.DataFrame(rows)
    lo, hi = bootstrap_ci(agg["spread"].values)
    wins = int((agg["spread"] > 0).sum())

    print("=== Across all as-of dates ===")
    print(f"windows                  : {len(agg)}")
    print(f"mean top-bucket excess   : {agg['top'].mean():+.2f}%")
    print(f"mean bottom-bucket excess: {agg['bottom'].mean():+.2f}%")
    print(f"mean top-minus-bottom    : {agg['spread'].mean():+.2f}%  "
          f"95% CI [{lo:+.2f}, {hi:+.2f}]")
    print(f"spread positive in       : {wins}/{len(agg)} windows")
    print()
    print("NOTE: overlapping windows from daily snapshots are NOT independent.")
    print("Treat the CI as indicative only; it understates true uncertainty.")

    out_path = os.path.join(REPO_ROOT, "data", "processed", "return_validation.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    agg.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
