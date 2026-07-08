"""
explainability.py — Per-company explainability for the M&A screener.

Turns a composite score into a plain-English answer to "why did this
company score the way it did." For each company, identifies:
  - which bucket (valuation/quality/leverage/growth) contributed most to
    its composite score, and which contributed least
  - within the strongest and weakest buckets, which specific metric drove
    that bucket's score
  - a one-line, human-readable summary

This is what turns a raw ranking into something you can actually explain
in a pitch or interview, rather than just presenting a number.

Usage:
    python -m src.explainability
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Mirrors scoring.py's METRIC_MAP — kept here rather than imported to avoid
# a tight coupling that would break if scoring.py's internals change; if
# metric names ever diverge, the column-presence check below will catch it
# with a clear error rather than a silent mismatch.
METRIC_TO_BUCKET = {
    "trailing_pe": "valuation",
    "ev_to_ebitda": "valuation",
    "return_on_equity": "quality",
    "net_margin": "quality",
    "debt_to_equity_ratio": "leverage",
    "revenue_growth": "growth",
}

METRIC_LABELS = {
    "trailing_pe": "P/E ratio",
    "ev_to_ebitda": "EV/EBITDA",
    "return_on_equity": "return on equity",
    "net_margin": "net margin",
    "debt_to_equity_ratio": "leverage (debt/equity)",
    "revenue_growth": "revenue growth",
}


def load_config(config_path: str = "config/universe.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _bucket_contribution(row: pd.Series, weights: dict) -> dict:
    """Effective contribution of each bucket to the composite score,
    accounting for reweighting when a bucket was missing for this company
    (mirrors the same reweighting logic scoring.py uses for the composite
    itself, so contributions sum to the actual composite score)."""
    bucket_scores = {b: row.get(f"{b}_bucket_score") for b in weights}
    available = {b: s for b, s in bucket_scores.items() if pd.notna(s)}
    if not available:
        return {}
    total_weight = sum(weights[b] for b in available)
    return {b: (weights[b] / total_weight) * s for b, s in available.items()}


def _strongest_metric_in_bucket(row: pd.Series, bucket: str) -> tuple:
    """Within a bucket, find the metric with the highest percentile score
    (i.e. the specific number driving that bucket's strength)."""
    metrics_in_bucket = [m for m, b in METRIC_TO_BUCKET.items() if b == bucket]
    scores = {m: row.get(f"{m}_score") for m in metrics_in_bucket if pd.notna(row.get(f"{m}_score"))}
    if not scores:
        return None, None
    best_metric = max(scores, key=scores.get)
    return best_metric, scores[best_metric]


def _weakest_metric_in_bucket(row: pd.Series, bucket: str) -> tuple:
    metrics_in_bucket = [m for m, b in METRIC_TO_BUCKET.items() if b == bucket]
    scores = {m: row.get(f"{m}_score") for m in metrics_in_bucket if pd.notna(row.get(f"{m}_score"))}
    if not scores:
        return None, None
    worst_metric = min(scores, key=scores.get)
    return worst_metric, scores[worst_metric]


def explain_company(row: pd.Series, weights: dict) -> dict:
    contributions = _bucket_contribution(row, weights)
    if not contributions:
        return {
            "ticker": row["ticker"],
            "summary": f"{row['ticker']}: insufficient data across all buckets to explain this score.",
        }

    strongest_bucket = max(contributions, key=contributions.get)
    weakest_bucket = min(contributions, key=contributions.get)

    strong_metric, strong_metric_score = _strongest_metric_in_bucket(row, strongest_bucket)
    weak_metric, weak_metric_score = _weakest_metric_in_bucket(row, weakest_bucket)

    strong_metric_label = METRIC_LABELS.get(strong_metric, strong_metric) if strong_metric else None
    weak_metric_label = METRIC_LABELS.get(weak_metric, weak_metric) if weak_metric else None

    sector_note = f" (sector: {row['sector']})" if "sector" in row and pd.notna(row.get("sector")) else ""
    rank_note = f"#{int(row['rank'])} overall" if "rank" in row and pd.notna(row.get("rank")) else ""
    if "sector_rank" in row and pd.notna(row.get("sector_rank")):
        rank_note += f", #{int(row['sector_rank'])} in sector"

    summary = (
        f"{row['ticker']}{sector_note} ranked {rank_note}. "
        f"Strongest factor: {strongest_bucket} (driven primarily by {strong_metric_label})"
        + (f", scoring at the {int(strong_metric_score * 100)}th percentile within its sector on this metric" if strong_metric_score is not None else "")
        + f". Weakest factor: {weakest_bucket} (driven primarily by {weak_metric_label})."
    )

    return {
        "ticker": row["ticker"],
        "strongest_bucket": strongest_bucket,
        "strongest_bucket_contribution": round(contributions[strongest_bucket], 4),
        "strongest_metric": strong_metric,
        "strongest_metric_score": strong_metric_score,
        "weakest_bucket": weakest_bucket,
        "weakest_bucket_contribution": round(contributions[weakest_bucket], 4),
        "weakest_metric": weak_metric,
        "weakest_metric_score": weak_metric_score,
        "summary": summary,
    }


def build_explanations(
    scored_path: str = "data/processed/scored_universe.csv",
    output_dir: str = "data/processed",
    top_n: int = 10,
) -> pd.DataFrame:
    config = load_config()
    weights = config["scoring"]["weights"]

    df = pd.read_csv(scored_path)
    if df.empty:
        raise ValueError(f"{scored_path} is empty — run the scoring step first.")

    required_metric_cols = [f"{m}_score" for m in METRIC_TO_BUCKET]
    missing = [c for c in required_metric_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"scored_universe.csv is missing metric-level score columns {missing} — "
            f"re-run `python -m src.scoring` with the current version, which "
            f"passes these through for exactly this purpose."
        )

    shortlist = df[df["rank"] <= top_n].copy() if "rank" in df.columns else df.copy()

    explanations = [explain_company(row, weights) for _, row in shortlist.iterrows()]
    result = pd.DataFrame(explanations)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "explanations.csv"
    result.to_csv(out_path, index=False)
    logger.info(f"Explanations for top {len(result)} companies -> {out_path}")

    return result


if __name__ == "__main__":
    result = build_explanations()
    print(f"\nDone. Explanations generated for {len(result)} companies.\n")
    for _, row in result.iterrows():
        print(row["summary"])
        print()
