"""
scoring.py — Multi-factor scoring engine for the M&A screener.

Takes the cleaned universe and produces a composite score per company using
percentile rank within the universe (not raw z-scores — see rationale below).

Why percentile rank, not z-score:
Your universe mixes sectors (cement, IT) with structurally different normal
ranges for P/E, margins, and leverage. A z-score assumes roughly-normal,
comparably-scaled data — that assumption breaks with a small, multi-sector
sample. Percentile rank just asks "where does this company sit relative to
the others in this screen," which is an honest claim given the sample size.

Factor buckets (weights come from config/universe.yaml):
  - Valuation (lower is better):  P/E, EV/EBITDA
  - Quality   (higher is better): ROE, net margin
  - Leverage  (lower is better):  debt-to-equity ratio
  - Growth    (higher is better): revenue growth

Missing-data handling: if a company is missing one metric in a bucket (e.g.
HCLTECH's nulled ev_to_ebitda), the bucket score is computed from whatever
metrics ARE available in that bucket, not zero-filled. If an entire bucket
is missing for a company, the composite score is reweighted across the
remaining buckets rather than silently treating the missing bucket as a
zero score, which would unfairly tank that company's rank.

Usage:
    python -m src.scoring
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# metric -> (bucket, direction). direction "low" means lower raw value = better.
METRIC_MAP = {
    "trailing_pe": ("valuation", "low"),
    "ev_to_ebitda": ("valuation", "low"),
    "return_on_equity": ("quality", "high"),
    "net_margin": ("quality", "high"),
    "debt_to_equity_ratio": ("leverage", "low"),
    "revenue_growth": ("growth", "high"),
}


def load_config(config_path: str = "config/universe.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _percentile_scores(df: pd.DataFrame) -> pd.DataFrame:
    """For each metric in METRIC_MAP, compute a 0-1 percentile score where
    1.0 is always 'best' for that metric, regardless of direction. NaNs stay
    NaN — pandas rank(pct=True) naturally excludes them from the ranking,
    so a missing value doesn't get penalized as if it were the worst score.

    Ranking is done WITHIN sector when a 'sector' column is present. Ranking
    cement against IT on the same scale conflates industry norms (cement's
    naturally higher leverage, IT's naturally higher margins) with genuine
    company quality — a cheap/low-leverage cement company isn't necessarily
    a better target than an expensive/high-margin IT company, they're just
    structurally different businesses. Sector-relative ranking asks the more
    honest question: "is this a strong name relative to its own peers."
    """
    scored = df.copy()
    has_sector = "sector" in scored.columns and scored["sector"].notna().any()

    for metric, (_, direction) in METRIC_MAP.items():
        if metric not in scored.columns:
            logger.warning(f"metric '{metric}' not found in cleaned data — skipping")
            continue

        if has_sector:
            rank_pct = scored.groupby("sector")[metric].rank(pct=True, na_option="keep")
        else:
            rank_pct = scored[metric].rank(pct=True, na_option="keep")

        scored[f"{metric}_score"] = rank_pct if direction == "high" else (1 - rank_pct)

    return scored


def _bucket_scores(scored: pd.DataFrame) -> pd.DataFrame:
    """Average the metric scores within each bucket, using only the metrics
    that are actually present for that row (NaN-aware mean, not zero-fill)."""
    buckets = {}
    for metric, (bucket, _) in METRIC_MAP.items():
        buckets.setdefault(bucket, []).append(f"{metric}_score")

    for bucket, score_cols in buckets.items():
        available_cols = [c for c in score_cols if c in scored.columns]
        scored[f"{bucket}_bucket_score"] = scored[available_cols].mean(axis=1, skipna=True)
    return scored


def _composite_score(scored: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Weighted average across bucket scores, reweighting when a bucket is
    entirely missing for a given company so missing data doesn't silently
    act as a zero and unfairly tank that company's composite score."""
    bucket_cols = {b: f"{b}_bucket_score" for b in weights}

    def compute_row(row):
        available = {b: row[col] for b, col in bucket_cols.items() if pd.notna(row[col])}
        if not available:
            return None
        total_weight = sum(weights[b] for b in available)
        weighted_sum = sum(weights[b] * available[b] for b in available)
        return weighted_sum / total_weight

    scored["composite_score"] = scored.apply(compute_row, axis=1)
    scored["buckets_used"] = scored.apply(
        lambda row: sum(1 for col in bucket_cols.values() if pd.notna(row[col])), axis=1
    )
    return scored


def score_universe(
    input_path: str = "data/processed/cleaned_universe.csv",
    output_dir: str = "data/processed",
) -> pd.DataFrame:
    config = load_config()
    weights = config["scoring"]["weights"]
    top_n = config["output"]["top_n_shortlist"]

    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"{input_path} is empty — check the cleaning step ran correctly.")

    scored = _percentile_scores(df)
    scored = _bucket_scores(scored)
    scored = _composite_score(scored, weights)

    ranked = scored.sort_values("composite_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    if "sector" in ranked.columns:
        ranked["sector_rank"] = ranked.groupby("sector")["composite_score"].rank(ascending=False, method="min").astype("Int64")

    # Output every original input column (everything cleaning.py produced)
    # PLUS the new scoring-derived columns — rather than a hand-picked
    # whitelist. A curated column list has already gone stale twice
    # (comps_dupont needed total_debt/total_equity, then accretion_dilution
    # needed price/market_cap) — each fix required tracing a downstream
    # KeyError back to this list. Passing everything through structurally
    # prevents that class of bug for any future module, at the cost of a
    # slightly wider CSV.
    #
    # Metric-level score columns (e.g. trailing_pe_score) are derived from
    # METRIC_MAP rather than hardcoded, so this list can't go stale again
    # if a metric is ever added/removed from METRIC_MAP — needed by
    # explainability.py for "why did this company score this way" detail.
    metric_score_cols = [f"{metric}_score" for metric in METRIC_MAP]
    bucket_score_cols = [f"{bucket}_bucket_score" for bucket in weights]
    score_derived_cols = (
        ["rank", "sector_rank", "composite_score", "buckets_used"]
        + bucket_score_cols
        + metric_score_cols
    )
    original_cols = [c for c in df.columns if c not in score_derived_cols]
    output_cols = ["rank", "ticker"] + [c for c in original_cols if c != "ticker"] + \
                  [c for c in score_derived_cols if c not in ("rank",)]
    # De-duplicate while preserving order (rank/ticker placed first for readability)
    seen = set()
    output_cols = [c for c in output_cols if not (c in seen or seen.add(c))]
    output_cols = [c for c in output_cols if c in ranked.columns]
    result = ranked[output_cols]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "scored_universe.csv"
    result.to_csv(out_path, index=False)

    shortlist_path = Path(output_dir) / "shortlist_top_n.csv"
    result.head(top_n).to_csv(shortlist_path, index=False)

    logger.info(f"Scored {len(result)} companies -> {out_path}")
    logger.info(f"Top {top_n} shortlist -> {shortlist_path}")

    low_confidence = result[result["buckets_used"] < len(weights)]
    if len(low_confidence) > 0:
        logger.warning(
            f"{len(low_confidence)} companies scored on fewer than all "
            f"{len(weights)} buckets (reweighted, not zero-filled) — see "
            f"'buckets_used' column: {low_confidence['ticker'].tolist()}"
        )

    return result


if __name__ == "__main__":
    result = score_universe()
    print(f"\nDone. {len(result)} companies scored and ranked.")
    print("\nTop shortlist:")
    print(result.head(10).to_string(index=False))
