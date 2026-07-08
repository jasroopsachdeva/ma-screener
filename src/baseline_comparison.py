"""
baseline_comparison.py — Naive baseline comparison for the M&A screener.

Answers a question every multi-factor score should be able to answer
honestly: "does this actually do something a naive single-metric sort
wouldn't?" This is NOT a rigorous backtest (no forward returns data to
validate against) — it's a structural comparison, directionally honest
about what it can and can't claim.

Naive baseline used: rank by trailing P/E alone, within sector (lowest
P/E = best), mirroring the "just buy what's cheap" heuristic a screen-less
analyst might use. This is deliberately the simplest defensible baseline,
not a strawman — P/E is a genuinely common first-pass filter in practice.

What this module reports:
  - Rank correlation between the composite score and the naive baseline
    (high correlation would suggest the composite isn't adding much beyond
    what a single ratio already captures)
  - Top-N overlap: how many companies both approaches would shortlist
  - Specific "caught by composite, missed by naive" names — companies the
    multi-factor approach ranks highly that a P/E-only screen would have
    passed over, and vice versa — with the reason visible via the
    explainability module's output.

Usage:
    python -m src.baseline_comparison
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_naive_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by trailing P/E alone, GLOBALLY across the whole universe
    (lower P/E = better rank).

    This is deliberately NOT sector-relative, unlike the composite score.
    The composite score needs sector-relative ranking to be a fair
    per-company assessment (see scoring.py) — but this naive baseline
    exists purely to answer "what would a top-N shortlist look like under
    the simplest possible approach," and a top-N threshold only means
    something against a genuinely global rank.

    A sector-relative naive rank was tried first and is a real bug worth
    noting: with sector sizes of 7-8 here, a sector-relative rank never
    exceeds 8, so 'naive_rank <= 10' was trivially true for nearly the
    entire universe — silently turning 'naive top 10' into 'everyone,'
    which produced an internally contradictory report (100% overlap while
    also listing several 'naive only' names). Global ranking avoids that
    failure mode entirely.
    """
    result = df.copy()
    result["naive_rank"] = result["trailing_pe"].rank(method="min", na_option="bottom")
    return result


def compare_rankings(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Core comparison: rank correlation + top-N overlap + divergent names."""
    ranked = compute_naive_rank(df)

    # Spearman rank correlation, computed as Pearson correlation of the
    # rank values themselves — avoids adding a scipy dependency for a
    # single stat, and is mathematically identical to Spearman's rho.
    valid = ranked.dropna(subset=["rank", "naive_rank"])
    correlation = valid["rank"].corr(valid["naive_rank"]) if len(valid) > 1 else None

    composite_top_n = set(ranked[ranked["rank"] <= top_n]["ticker"])
    naive_top_n = set(ranked[ranked["naive_rank"] <= top_n]["ticker"])

    overlap = composite_top_n & naive_top_n
    caught_by_composite_only = composite_top_n - naive_top_n   # composite likes it, naive P/E screen would've skipped it
    caught_by_naive_only = naive_top_n - composite_top_n       # cheap on P/E alone, but composite doesn't rate it as highly

    overlap_pct = len(overlap) / top_n if top_n > 0 else 0

    return {
        "rank_correlation": correlation,
        "top_n": top_n,
        "overlap_count": len(overlap),
        "overlap_pct": overlap_pct,
        "overlap_tickers": sorted(overlap),
        "caught_by_composite_only": sorted(caught_by_composite_only),
        "caught_by_naive_only": sorted(caught_by_naive_only),
        "ranked_df": ranked,
    }


def _interpret_correlation(corr: float) -> str:
    if corr is None:
        return "insufficient data to compute a correlation"
    abs_corr = abs(corr)
    if abs_corr >= 0.85:
        return (
            f"very high ({corr:.2f}) — the composite score is closely tracking a simple P/E "
            f"sort here. Worth being honest that with this small a universe, the added factors "
            f"(quality/leverage/growth) may not be moving the ranking much yet."
        )
    if abs_corr >= 0.5:
        return (
            f"moderate ({corr:.2f}) — the composite score broadly agrees with a P/E-only view "
            f"but meaningfully reorders things based on the other factors."
        )
    return (
        f"low ({corr:.2f}) — the composite score is ranking companies quite differently from "
        f"a simple P/E sort, meaning the quality/leverage/growth factors are doing real work "
        f"in this universe."
    )


def build_comparison_report(
    scored_path: str = "data/processed/scored_universe.csv",
    output_dir: str = "data/processed",
    top_n: int = 10,
) -> dict:
    df = pd.read_csv(scored_path)
    if df.empty:
        raise ValueError(f"{scored_path} is empty — run the scoring step first.")

    required = ["ticker", "rank", "trailing_pe"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"scored_universe.csv is missing required columns {missing}")

    comparison = compare_rankings(df, top_n=top_n)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "baseline_comparison.csv"
    comparison["ranked_df"][["ticker", "sector", "rank", "naive_rank", "composite_score", "trailing_pe"]].to_csv(
        out_path, index=False
    )
    logger.info(f"Baseline comparison -> {out_path}")

    return comparison


if __name__ == "__main__":
    result = build_comparison_report()

    print(f"\n=== Composite Score vs. Naive P/E-Only Baseline (top {result['top_n']}) ===\n")
    print(f"Rank correlation: {_interpret_correlation(result['rank_correlation'])}\n")
    print(f"Top-{result['top_n']} overlap: {result['overlap_count']}/{result['top_n']} companies "
          f"({result['overlap_pct']:.0%}) appear in both shortlists\n")

    if result["caught_by_composite_only"]:
        print(f"Composite ranks highly but naive P/E-only screen would've missed: "
              f"{', '.join(result['caught_by_composite_only'])}")
    if result["caught_by_naive_only"]:
        print(f"Cheap on P/E alone but composite doesn't rate as highly: "
              f"{', '.join(result['caught_by_naive_only'])}")
    print()
