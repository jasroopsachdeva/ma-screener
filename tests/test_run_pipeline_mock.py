"""
Smoke tests for run_pipeline.py's validation gates. These are the actual
safety mechanism for the automated pipeline — if a stage's output looks
broken, the run should stop with a clear error rather than silently
continuing (and, in automation, silently publishing bad results).
Run with: python3 tests/test_run_pipeline_mock.py
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.run_pipeline import validate_ingestion, validate_cleaning, validate_scoring, PipelineValidationError


@dataclass
class FakeSnapshot:
    ticker: str
    fetch_ok: bool


def test_ingestion_validation_passes_on_healthy_data():
    snapshots = [FakeSnapshot(f"T{i}.NS", True) for i in range(14)] + [FakeSnapshot("BAD.NS", False)]
    validate_ingestion(snapshots)  # should not raise
    print("PASS: ingestion validation passes when success rate is high (14/15)")


def test_ingestion_validation_fails_on_mass_failure():
    """If more than half of tickers fail, that signals a systemic issue
    (rate limit, network outage) — the pipeline should stop rather than
    proceed with a mostly-empty universe."""
    snapshots = [FakeSnapshot(f"T{i}.NS", False) for i in range(10)] + [FakeSnapshot("OK.NS", True)]
    try:
        validate_ingestion(snapshots)
        assert False, "expected PipelineValidationError for mass ingestion failure"
    except PipelineValidationError as e:
        assert "1/11" in str(e)
        print("PASS: ingestion validation correctly stops the pipeline on mass failure (1/11 success)")


def test_ingestion_validation_fails_on_zero_tickers():
    try:
        validate_ingestion([])
        assert False, "expected PipelineValidationError for zero tickers"
    except PipelineValidationError as e:
        assert "zero tickers" in str(e).lower()
        print("PASS: ingestion validation correctly stops on an empty universe")


def test_cleaning_validation_fails_on_mass_exclusion():
    df = pd.DataFrame([{"ticker": "A.NS"}])  # 1 retained out of 10 pre-clean
    try:
        validate_cleaning(df, pre_clean_count=10)
        assert False, "expected PipelineValidationError for mass exclusion"
    except PipelineValidationError as e:
        assert "1/10" in str(e)
        print("PASS: cleaning validation correctly stops when retention rate is too low (1/10)")


def test_scoring_validation_catches_out_of_range_scores():
    """This should be mathematically impossible given percentile-based
    scoring, but the validation exists as a defensive backstop — if
    scoring logic ever regresses to produce an out-of-[0,1] score, this
    should catch it rather than publish a nonsensical result."""
    df = pd.DataFrame([
        {"ticker": "A.NS", "rank": 1, "composite_score": 1.5},  # impossible value
        {"ticker": "B.NS", "rank": 2, "composite_score": 0.5},
    ])
    try:
        validate_scoring(df)
        assert False, "expected PipelineValidationError for out-of-range composite_score"
    except PipelineValidationError as e:
        assert "A.NS" in str(e)
        print("PASS: scoring validation catches an out-of-[0,1] composite score as a defensive backstop")


def test_scoring_validation_catches_duplicate_ranks():
    df = pd.DataFrame([
        {"ticker": "A.NS", "rank": 1, "composite_score": 0.8},
        {"ticker": "B.NS", "rank": 1, "composite_score": 0.7},  # duplicate rank — should never happen
    ])
    try:
        validate_scoring(df)
        assert False, "expected PipelineValidationError for duplicate ranks"
    except PipelineValidationError as e:
        assert "duplicate" in str(e).lower()
        print("PASS: scoring validation catches duplicate ranks")


def test_scoring_validation_passes_on_healthy_data():
    df = pd.DataFrame([
        {"ticker": "A.NS", "rank": 1, "composite_score": 0.8},
        {"ticker": "B.NS", "rank": 2, "composite_score": 0.6},
        {"ticker": "C.NS", "rank": 3, "composite_score": 0.3},
    ])
    validate_scoring(df)  # should not raise
    print("PASS: scoring validation passes on healthy, well-formed data")


def test_file_logging_creates_a_log_file_with_content():
    """The whole point of file-based logging is having something concrete
    to inspect after an automated run fails at 3am — verify it actually
    creates a file and captures log output, not just that the function runs."""
    import tempfile
    import logging as logging_module
    from src.run_pipeline import _configure_logging

    with tempfile.TemporaryDirectory() as tmp:
        # Reset root logger handlers between test runs so repeated test
        # executions (or other tests) don't leave stale file handlers open.
        root = logging_module.getLogger()
        original_handlers = root.handlers[:]
        for h in root.handlers[:]:
            if isinstance(h, logging_module.FileHandler):
                root.removeHandler(h)
                h.close()

        log_path = _configure_logging(log_dir=tmp)
        test_logger = logging_module.getLogger("test_logger")
        test_logger.info("this is a test log line that should appear in the file")

        for h in root.handlers[:]:
            if isinstance(h, logging_module.FileHandler):
                h.flush()

        assert Path(log_path).exists(), "log file was not created"
        content = Path(log_path).read_text()
        assert "this is a test log line" in content, "log content was not written to the file"

        # cleanup: remove the handler we added so it doesn't leak into other tests
        for h in root.handlers[:]:
            if isinstance(h, logging_module.FileHandler) and str(tmp) in str(getattr(h, "baseFilename", "")):
                root.removeHandler(h)
                h.close()

    print(f"PASS: file logging creates a real log file and captures log content")


if __name__ == "__main__":
    test_ingestion_validation_passes_on_healthy_data()
    test_ingestion_validation_fails_on_mass_failure()
    test_ingestion_validation_fails_on_zero_tickers()
    test_cleaning_validation_fails_on_mass_exclusion()
    test_scoring_validation_catches_out_of_range_scores()
    test_scoring_validation_catches_duplicate_ranks()
    test_scoring_validation_passes_on_healthy_data()
    test_file_logging_creates_a_log_file_with_content()
    print("\nAll pipeline validation mock tests passed.")
