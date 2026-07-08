# M&A Screener

A multi-factor screening tool for NSE-listed companies, built to surface
acquisition candidates by combining valuation, quality, leverage, and
growth signals — with sector-relative scoring, DuPont-based quality
checks, a simplified accretion/dilution simulator, and per-company
explainability.

This is a personal project built to demonstrate applied valuation and
screening logic, not a production investment tool. See **Limitations**
below before drawing any conclusions from its output.

---

## What it does

1. **Ingests** live financial data (price, P/E, EV/EBITDA, ROE, leverage,
   growth) for a configured universe of NSE-listed companies via
   `yfinance`.
2. **Cleans** the data — excludes companies with broken/missing critical
   fields, flags and nulls statistical outliers without discarding the
   whole row, and normalizes units (e.g. debt/equity from a percentage to
   a ratio).
3. **Scores** each company on four factors — valuation, quality, leverage,
   growth — using **sector-relative percentile rank**, so cement companies
   are judged against cement peers and IT companies against IT peers,
   rather than on one shared cross-sector scale.
4. **Benchmarks** the top-ranked shortlist against sector peers (comps)
   and decomposes ROE via the DuPont identity (margin x turnover x
   leverage), so a high score can be explained rather than just reported.
5. **Simulates** accretion/dilution for a hypothetical acquirer/target
   deal, given an assumed premium and cash/stock mix.
6. **Explains** each company's ranking in plain English — which factor
   drove the score, and which metric within that factor.
7. **Compares** the composite score against a naive "cheapest P/E" sort,
   to check whether the multi-factor approach is actually doing something
   a simpler heuristic wouldn't.

## Methodology

### Why sector-relative percentile rank, not a global z-score

The universe mixes sectors (cement, IT) with structurally different
normal ranges for P/E, margins, and leverage — comparing them on one
shared scale would conflate industry norms with company quality (e.g. a
capital-intensive cement company will structurally carry more leverage
than an asset-light IT company, regardless of how well either is run).
Percentile rank *within sector* asks the more honest question: "how does
this company compare to its own peers."

### Scoring weights

Configured in `config/universe.yaml`:

| Factor    | Weight | Metrics                          | Direction     |
|-----------|--------|-----------------------------------|----------------|
| Valuation | 30%    | P/E, EV/EBITDA                    | lower is better |
| Quality   | 30%    | ROE, net margin                   | higher is better |
| Leverage  | 20%    | Debt/Equity                       | lower is better |
| Growth    | 20%    | Revenue growth                    | higher is better |

### Missing data handling

If a company is missing one metric within a factor bucket, that bucket's
score is computed from whatever metric *is* available — not zero-filled.
If an entire bucket is missing, the composite score is reweighted across
the remaining buckets rather than silently treating the gap as a zero,
which would unfairly tank that company's rank. Every company's output
includes a `buckets_used` column so you can see exactly how much of the
scoring model actually had data to work with.

### DuPont decomposition

`ROE = Net Margin x Asset Turnover x Equity Multiplier`, where Equity
Multiplier is approximated as `(Total Debt + Total Equity) / Total
Equity` — a proxy for total assets / equity, since raw total assets
aren't pulled directly from `yfinance` in this pipeline. This is flagged
as an approximation, not presented as exact; the module cross-checks its
own implied ROE against the reported ROE and flags any gap over 5
percentage points as worth a manual look.

## Limitations (read this before trusting the output)

- **Baseline comparison is not a backtest.** It shows the composite score
  ranks differently from a naive P/E sort — not that it picks
  *better-performing* companies.
- **A genuine forward-return backtest now exists** (`src/backtest.py`'s
  `validate_forward_performance()`), but it only becomes meaningful once
  `history_tracker.py` has been recording daily snapshots for at least as
  long as the lookback window requested (default 30 days) — it correctly
  reports "insufficient history" rather than fabricating a result before
  then. There's also a `compute_trailing_performance()` check that runs
  immediately, but it's explicitly a retrospective/informational stat,
  not a backtest — it checks TODAY's picks against PAST prices, which
  doesn't prove the model would have made the same picks back then.
- **DuPont's equity multiplier is a proxy**, not exact — see above.
- **Accretion/dilution is deliberately simplified**: no synergies, no tax
  adjustments, no interest cost on the cash portion of a deal. It
  demonstrates real deal mechanics, not a full banker's model.
- **Universe now spans ~200 tickers across 16 sectors** (expanded from an
  original 16-ticker cement + IT prototype). Sector sizes still vary
  (5-19 tickers depending on sector), so percentile-rank resolution is
  still somewhat sector-dependent — a sector with only 5-6 names has less
  ranking granularity than one with 19. Adding more tickers to a thin
  sector in `config/universe.yaml` is the direct fix if that becomes
  relevant.
- **Data quality depends on Yahoo Finance.** `yfinance` is a free,
  unofficial API and can be rate-limited or return incomplete data for
  specific tickers — the pipeline is built to isolate and flag these
  cases rather than silently propagate bad data, but it can't invent data
  Yahoo doesn't provide.

## Project structure

```
config/universe.yaml       # ticker universe, sectors, scoring weights — edit here, not in code
src/ingestion.py           # pulls raw data, one bad ticker never crashes the run
src/cleaning.py            # validates, flags outliers, excludes broken rows
src/scoring.py             # sector-relative multi-factor composite score
src/comps_dupont.py        # peer benchmarking + ROE decomposition
src/accretion_dilution.py  # simplified deal simulator
src/explainability.py      # plain-English per-company breakdown
src/baseline_comparison.py # composite score vs. naive P/E-only sort
src/run_pipeline.py        # orchestrates all of the above with validation gates
src/dashboard.py           # Streamlit UI
tests/                     # mock tests for every module above
.github/workflows/         # scheduled automated runs
```

## Running it

```bash
pip install -r requirements.txt

# Run the full pipeline end to end
python -m src.run_pipeline

# Or run individual stages
python -m src.ingestion
python -m src.cleaning
python -m src.scoring
python -m src.comps_dupont
python -m src.explainability
python -m src.baseline_comparison

# Accretion/dilution for a specific deal
python -m src.accretion_dilution --acquirer TCS.NS --target COFORGE.NS --premium 0.25 --cash-pct 0.5
python -m src.accretion_dilution --acquirer TCS.NS --target COFORGE.NS --sensitivity

# Dashboard
streamlit run src/dashboard.py

# Tests
python tests/test_ingestion_mock.py
python tests/test_cleaning_mock.py
python tests/test_scoring_mock.py
python tests/test_comps_dupont_mock.py
python tests/test_accretion_dilution_mock.py
python tests/test_explainability_mock.py
python tests/test_baseline_comparison_mock.py
python tests/test_run_pipeline_mock.py
```

## Automation

`.github/workflows/run_screener.yml` runs the full pipeline daily via
GitHub Actions and commits the refreshed results back to the repo. A
failed run (validation error or crash) fails the workflow, which triggers
GitHub's built-in email notification to the repo owner — see
`src/run_pipeline.py` for the validation gates between each stage.
