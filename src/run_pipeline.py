"""
run_pipeline.py — Orchestrates the full M&A screener pipeline end to end.

Runs: ingestion -> cleaning -> scoring -> comps/DuPont -> explainability
-> baseline comparison, with a validation gate between stages. If a stage
produces data that fails a sanity check, the pipeline stops with a clear,
actionable error rather than silently continuing on broken data — the
same discipline built into every individual module, applied at the
pipeline level.

This is the single entry point for both manual runs and the automated
GitHub Actions workflow (see .github/workflows/run_screener.yml). A
non-zero exit code here is what GitHub Actions uses to detect failure and
trigger its built-in email notification to the repo owner — no separate
alerting service needed for a project at this scale.

Usage:
    python -m src.run_pipeline
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import ingestion, cleaning, scoring, comps_dupont, explainability, baseline_comparison, acquisition_likelihood, history_tracker


def _configure_logging(log_dir: str = "logs") -> str:
    """Log to both console (for interactive runs) and a timestamped file
    (so an automated GitHub Actions run leaves something concrete to
    inspect beyond the Actions log viewer, which can be less convenient
    to dig through after the fact — the file gets uploaded as part of the
    workflow's artifact alongside data/processed/)."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    log_path = str(Path(log_dir) / f"pipeline_run_{timestamp}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return log_path


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PipelineValidationError(Exception):
    """Raised when a pipeline stage's output fails a sanity check.
    Distinct from a plain exception so the top-level handler can report
    it clearly as 'the pipeline ran but produced suspect data', as
    opposed to 'the pipeline crashed'."""
    pass


def validate_ingestion(snapshots: list) -> None:
    total = len(snapshots)
    ok_count = sum(1 for s in snapshots if s.fetch_ok)
    if total == 0:
        raise PipelineValidationError("Ingestion returned zero tickers — check config/universe.yaml isn't empty.")
    success_rate = ok_count / total
    if success_rate < 0.5:
        raise PipelineValidationError(
            f"Only {ok_count}/{total} tickers fetched successfully ({success_rate:.0%}) — "
            f"this is low enough to suggest a systemic issue (rate limiting, network, or "
            f"a bad config) rather than a few isolated bad tickers. Stopping before "
            f"downstream stages run on a mostly-empty universe."
        )
    logger.info(f"Ingestion validation passed: {ok_count}/{total} tickers ({success_rate:.0%})")


def validate_cleaning(df: pd.DataFrame, pre_clean_count: int) -> None:
    if df.empty:
        raise PipelineValidationError("Cleaning excluded every ticker — nothing left to score.")
    retention_rate = len(df) / pre_clean_count if pre_clean_count > 0 else 0
    if retention_rate < 0.5:
        raise PipelineValidationError(
            f"Cleaning retained only {len(df)}/{pre_clean_count} tickers ({retention_rate:.0%}) — "
            f"unusually high exclusion rate, worth checking data/processed/cleaning_flags.csv "
            f"before trusting downstream results."
        )
    logger.info(f"Cleaning validation passed: {len(df)}/{pre_clean_count} tickers retained ({retention_rate:.0%})")


def validate_scoring(df: pd.DataFrame) -> None:
    if df.empty:
        raise PipelineValidationError("Scoring produced zero ranked companies.")
    if df["composite_score"].isna().all():
        raise PipelineValidationError("Every company has a null composite_score — scoring logic likely broken.")
    if not df["rank"].is_unique:
        raise PipelineValidationError("Duplicate ranks found in scored output — ranking logic produced ties incorrectly.")
    out_of_range = df[(df["composite_score"] < 0) | (df["composite_score"] > 1)]
    if len(out_of_range) > 0:
        raise PipelineValidationError(
            f"{len(out_of_range)} companies have composite_score outside [0,1]: "
            f"{out_of_range['ticker'].tolist()} — this should be mathematically impossible "
            f"given percentile-based scoring, and indicates a real bug if it happens."
        )
    logger.info(f"Scoring validation passed: {len(df)} companies ranked, all scores in valid range")


def run(config_path: str = "config/universe.yaml", top_n: int = 10) -> dict:
    stage_results = {}

    logger.info("=== Stage 1/8: Ingestion ===")
    cfg = ingestion.load_config(config_path)
    snapshots = ingestion.fetch_universe(cfg)
    validate_ingestion(snapshots)
    stage_results["ingestion"] = {"total": len(snapshots), "ok": sum(1 for s in snapshots if s.fetch_ok)}

    logger.info("=== Stage 2/8: Cleaning ===")
    cleaned_df = cleaning.clean_universe()
    validate_cleaning(cleaned_df, pre_clean_count=len(snapshots))
    stage_results["cleaning"] = {"retained": len(cleaned_df), "input": len(snapshots)}

    logger.info("=== Stage 3/8: Scoring ===")
    scored_df = scoring.score_universe()
    validate_scoring(scored_df)
    stage_results["scoring"] = {"ranked": len(scored_df)}

    logger.info("=== Stage 4/8: Comps + DuPont ===")
    comps_df = comps_dupont.build_report(top_n=top_n)
    stage_results["comps_dupont"] = {"companies": len(comps_df)}

    logger.info("=== Stage 5/8: Explainability ===")
    explanations_df = explainability.build_explanations(top_n=top_n)
    stage_results["explainability"] = {"companies": len(explanations_df)}

    logger.info("=== Stage 6/8: Baseline Comparison ===")
    comparison = baseline_comparison.build_comparison_report(top_n=top_n)
    stage_results["baseline_comparison"] = {
        "correlation": comparison["rank_correlation"],
        "overlap_pct": comparison["overlap_pct"],
    }

    logger.info("=== Stage 7/8: Acquisition Likelihood ===")
    likelihood_df = acquisition_likelihood.build_report(top_n=top_n)
    stage_results["acquisition_likelihood"] = {"companies": len(likelihood_df)}

    logger.info("=== Stage 8/8: Historical Snapshot ===")
    history_tracker.record_shortlist_snapshot(top_n=top_n)
    try:
        history_tracker.record_acquisition_likelihood_snapshot(top_n=top_n)
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Skipped acquisition-likelihood history snapshot: {e}")
    stage_results["history_tracker"] = {"recorded": True}

    logger.info("=== Pipeline completed successfully ===")
    for stage, info in stage_results.items():
        logger.info(f"  {stage}: {info}")

    return stage_results


if __name__ == "__main__":
    log_path = _configure_logging()
    logger.info(f"Logging to file: {log_path}")
    try:
        run()
        sys.exit(0)
    except PipelineValidationError as e:
        logger.error(f"PIPELINE VALIDATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"PIPELINE CRASHED: {type(e).__name__}: {e}")
        sys.exit(1)
