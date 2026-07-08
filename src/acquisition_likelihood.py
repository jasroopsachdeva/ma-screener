"""
acquisition_likelihood.py — M&A likelihood scoring for the screener.

This is a DELIBERATELY DIFFERENT lens from scoring.py's composite score.
The composite score answers "is this a good business" (rewards quality,
growth, low leverage, reasonable valuation). This module answers a
different, genuinely M&A-specific question: "is this a plausible
ACQUISITION TARGET" — which is not the same thing.

The classic private-equity / corp-dev "value target" thesis is: a company
that's cheap, carries low leverage (balance sheet capacity for a buyer to
use), but is UNDERPERFORMING on quality/margins relative to its own sector
peers — i.e. there's a turnaround or operational-improvement thesis for
whoever acquires it. This is precisely the kind of company the composite
score would rank LOW (since it rewards quality), which is the point: this
module is designed to surface names the main screen would overlook.

Components (built by RECOMBINING scoring.py's existing sector-relative
bucket scores, not recomputing from scratch — inherits sector-relative
discipline for free):
  - valuation_bucket_score   (reused as-is: cheap relative to sector = good)
  - leverage_bucket_score    (reused as-is: low leverage = balance sheet
                               capacity for a buyer, same direction as the
                               composite score's "good business" framing)
  - turnaround_potential     (INVERTED quality_bucket_score: weak margins/
                               ROE relative to sector = higher turnaround
                               potential — this is the one dimension that
                               runs in the OPPOSITE direction from the
                               composite score, and is what makes this a
                               genuinely different lens, not just a
                               relabeled composite score)

Companies with negative or zero total_equity are excluded, not scored —
financial distress is a different category from "undervalued but sound,"
and scoring a distressed company's weak quality as "turnaround potential"
would conflate the two.

Usage:
    python -m src.acquisition_likelihood
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"valuation": 0.40, "turnaround_potential": 0.35, "leverage": 0.25}


def load_config(config_path: str = "config/universe.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _reweighted_score(row: pd.Series, component_cols: dict, weights: dict) -> tuple:
    """Same reweighting discipline as scoring.py's _composite_score: if a
    component is missing for a company, drop it and rescale the remaining
    weights, rather than silently treating the gap as a zero."""
    available = {name: row[col] for name, col in component_cols.items() if pd.notna(row[col])}
    if not available:
        return None, 0
    total_weight = sum(weights[name] for name in available)
    weighted_sum = sum(weights[name] * available[name] for name in available)
    return weighted_sum / total_weight, len(available)


def score_acquisition_likelihood(
    df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS

    required = ["ticker", "valuation_bucket_score", "quality_bucket_score", "leverage_bucket_score", "total_equity"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"acquisition_likelihood scoring requires columns {missing} from scored_universe.csv — "
            f"re-run `python -m src.scoring` with the current version."
        )

    result = df.copy()

    # Financial distress is a different category from "undervalued but
    # sound" — a negative-equity company's weak quality score reflects
    # real distress, not an operational-improvement opportunity, so it's
    # excluded rather than scored as a high-"turnaround-potential" name.
    excluded_mask = result["total_equity"].notna() & (result["total_equity"] <= 0)
    excluded = result.loc[excluded_mask, "ticker"].tolist()
    result = result.loc[~excluded_mask].copy()

    if excluded:
        logger.info(
            f"{len(excluded)} companies excluded from acquisition-likelihood scoring "
            f"(negative/zero equity — financial distress, not a value-target thesis): {excluded[:10]}"
            f"{'...' if len(excluded) > 10 else ''}"
        )

    # Turnaround potential is the INVERSE of quality — deliberately the one
    # component that runs opposite to the composite score's direction.
    result["turnaround_potential_score"] = 1 - result["quality_bucket_score"]

    component_cols = {
        "valuation": "valuation_bucket_score",
        "turnaround_potential": "turnaround_potential_score",
        "leverage": "leverage_bucket_score",
    }

    scores_and_counts = result.apply(
        lambda row: _reweighted_score(row, component_cols, weights), axis=1
    )
    result["acquisition_likelihood_score"] = scores_and_counts.apply(lambda t: t[0])
    result["likelihood_components_used"] = scores_and_counts.apply(lambda t: t[1])

    ranked = result.dropna(subset=["acquisition_likelihood_score"]).sort_values(
        "acquisition_likelihood_score", ascending=False
    ).reset_index(drop=True)
    ranked["acquisition_likelihood_rank"] = ranked.index + 1
    if "sector" in ranked.columns:
        ranked["acquisition_likelihood_sector_rank"] = (
            ranked.groupby("sector")["acquisition_likelihood_score"]
            .rank(ascending=False, method="min").astype("Int64")
        )

    return ranked


def build_report(
    scored_path: str = "data/processed/scored_universe.csv",
    output_dir: str = "data/processed",
    top_n: int = 10,
) -> pd.DataFrame:
    df = pd.read_csv(scored_path)
    if df.empty:
        raise ValueError(f"{scored_path} is empty — run the scoring step first.")

    ranked = score_acquisition_likelihood(df)

    output_cols = [c for c in [
        "acquisition_likelihood_rank", "ticker", "sector", "acquisition_likelihood_sector_rank",
        "acquisition_likelihood_score", "likelihood_components_used",
        "valuation_bucket_score", "turnaround_potential_score", "leverage_bucket_score",
        "quality_bucket_score", "composite_score", "rank",
    ] if c in ranked.columns]
    output = ranked[output_cols]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "acquisition_likelihood.csv"
    output.to_csv(out_path, index=False)
    logger.info(f"Acquisition likelihood scoring -> {out_path}")

    return output.head(top_n) if top_n else output


if __name__ == "__main__":
    result = build_report()
    print(f"\nTop acquisition-likelihood candidates (cheap + low leverage + underperforming quality vs. sector):\n")
    display_cols = [c for c in [
        "acquisition_likelihood_rank", "ticker", "sector",
        "acquisition_likelihood_score", "quality_bucket_score", "composite_score", "rank",
    ] if c in result.columns]
    print(result[display_cols].to_string(index=False))
    print(
        "\nNote: 'rank'/'composite_score' here are the MAIN screener's quality ranking, shown "
        "for contrast — a company can score well here specifically BECAUSE it ranks weakly there."
    )
