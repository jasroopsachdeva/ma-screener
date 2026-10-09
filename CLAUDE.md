# CLAUDE.md — ma-screener

Context for Claude Code working in this repo. Read before making changes.

## What this is

An M&A target screener over ~201 NSE-listed tickers across 16 sectors. Fetches
fundamentals, cleans them, scores each company on valuation / quality / leverage /
growth, and ranks them into a composite acquisition-likelihood list. Deployed
publicly on Render's free tier.

## Running it

```bash
python -m src.api          # the real site → http://localhost:8000
streamlit run src/dashboard.py   # separate second UI
```

`src/api.py` is a **FastAPI** app. It serves `/api/*` endpoints and mounts the
frontend in `web/` as static files at `/`, so one command serves everything.

`src/dashboard.py` is **Streamlit, not ASGI**. `uvicorn src.dashboard:app` fails
with "Attribute app not found". It is a separate UI, not the deployed site.

## Pipeline

```
ingestion.py  →  cleaning.py  →  scoring.py  →  baseline_comparison.py
   (yfinance)      (filters)      (ranks)         history_tracker.py
```

Universe and fetch settings live in `config/universe.yaml` (including
`request_delay_seconds`, a politeness delay that reduces yfinance rate-limit
failures — do not remove it).

Endpoints: `/api/hero-stats`, `/api/shortlist`, `/api/acquisition-likelihood`,
`/api/deal`, `/api/deal/heatmap`, `/api/best-targets`, `/api/summary-pdf/{ticker}`.

## Rules that are easy to break

**Scoring must stay NaN-aware. Never zero-fill a missing metric.**
Ranks use `rank(pct=True, na_option="keep")` so NaNs are excluded rather than
treated as worst. Bucket and composite scores are NaN-aware means over only the
components actually present. `buckets_used` counts non-null buckets, and rows
with `buckets_used < len(weights)` are flagged `low_confidence`. Zero-filling
would silently push incomplete rows to the top of the ranking.

**`CRITICAL_FIELDS = ["market_cap", "price"]` in `cleaning.py`.**
`trailing_pe` was deliberately demoted from this list — treating it as critical
dropped four otherwise-scoreable companies every run. Do not add it back.

**ROE is partly derived, and `asset_turnover` is revenue / equity.**
Yahoo stopped returning `returnOnEquity` for most NSE tickers in Aug 2026
(95% coverage on Jul 9, 17% from Aug 18). `cleaning.py` fills only the missing
ones as `net_margin × asset_turnover`, flagged `roe_derived`; where both exist it
is within ~1pp of reported ROE (median). Ingestion computes `asset_turnover` as
revenue / **equity**, so that product already is ROE — never multiply it by an
equity multiplier as well (`comps_dupont` did until Oct 2026, doubling ROE at
D/E = 1). Snapshots from Aug 18, 2026 until this fix shipped have ROE for ~17%
of rows and later ones ~98%, so any series across that boundary mixes two input
regimes.

**Snapshots are ordered by run number, never by path string.** As text,
`run-100` sorts before `run-99`. `api.py` and `validate_returns.py` both rely on
numeric order (`_run_number`).

**The deal model can't value a target larger than the acquirer.** It pays the
cash portion from existing reserves at no cost, so a bigger target gives
nonsense accretion (ABB buying the larger Axis Bank showed +645%).
`check_target_size()` rejects these on every single-deal path; `find_best_targets`
excludes them via `max_target_size_pct`.

**Only forward backtests are legitimate.**
`validate_forward_performance()` (`backtest.py`) and `src/validate_returns.py`
rank on what was recorded on a past date and measure price performance from that
date forward. No look-ahead. `validate_forward_performance()` reads
`data/history/shortlist_history.csv`, which is frozen at four dates (Jul 4, 6, 7
and Aug 5, 2026) and will not grow (see the .gitignore section).
`validate_returns.py` reads the committed daily snapshots, so it is the forward
test that actually accumulates.

`compute_trailing_performance()` ranks on *today's* fundamentals and then checks
*past* price performance. It is structurally biased and informational only. Never
present its output as validation.

Current honest finding, from `src/validate_returns.py` (defaults, run
2026-10-10 over snapshots dated Jul 9 – Sep 10): a benchmark-adjusted
full-universe decile backtest shows top-minus-bottom ≈ **+1.13%**, 95% CI
[+0.80, +1.45], spread positive in 50 of 64 windows. That is **still not a
statistically detectable signal**. The 64 windows are daily and overlap heavily,
leaving only ~3 independent 30-day windows (−0.75%, +2.91%, +2.39%), so the CI
is far too narrow. The scoring inputs also changed mid-sample (see ROE above).
The earlier figure (−0.97%, CI [−1.28, −0.63], 0 of 7 windows, computed Aug 5
from Jul 9–15 only) was equally inconclusive; the sign flip shows how unstable
this is. Do not retune scoring weights to make this number look better and then
call it validated.

**Acquisition likelihood uses an inverted lens** — cheap, low-leverage, weak
quality reads as a value target. Scores that look "bad" on quality are intentional.

## The .gitignore trap (this has bitten once)

`.gitignore` excludes `data/processed/*.csv`, `data/history/*.csv`,
`data/raw/*.json`. So `git add data/processed/` stages nothing,
`git diff --cached --quiet` short-circuits, `git push` exits 0 — a green CI tick
with no archive written. This silently lost a month of pipeline output.

Consequences to respect:

- The workflow's archive step writes to `snapshots/run-NN` and must use
  `git add -f`, plus an explicit failure when nothing is staged. Keep both.
- `logs/` is anchored as `/logs/` so snapshot run logs are not ignored at depth.
  A bare `logs/` pattern matches at any depth and re-breaks this.
- `src/api.py` serves whichever of `data/processed/` and the newest committed
  snapshot has the newer `fetched_at`. On Render only snapshots exist, since the
  processed CSVs are gitignored and never reach GitHub; this is what keeps the
  deploy from serving 503s on every endpoint. Do not "clean up" this fallback.
  Nothing in the API may read `data/processed/` by a CWD-relative path: it works
  locally and fails on Render (the PDF endpoint did exactly this).
- `data/history/*.csv` is matched by `.gitignore`, but an Aug 5, 2026 archive of
  both files is force-committed and tracked (40 top-10 records, 4 dates). CI's
  appends are discarded with the runner, so it only grows if someone runs the
  pipeline locally and commits it. Check `git status` for these after a local run.

## CI

`.github/workflows/run_screener.yml` — daily cron `0 3 * * *` (08:30 IST, though
GitHub queueing usually delays it to ~11:10–11:50 IST). `workflow_dispatch` is
enabled. Python 3.11, `timeout-minutes: 45`. A separate workflow runs the test
suite on every push.

The test workflow runs each `tests/test_*.py` as a plain script
(`python "$f"`), **not pytest**. Every test file inserts the repo root into
`sys.path` and calls its tests from a `__main__` block; pytest fixtures
(`tmp_path`, `monkeypatch`) don't exist there, so use `tempfile` and
`unittest.mock.patch`. A file that passes under pytest can still fail CI.

Transient yfinance rate-limit dropouts cost a handful of large-cap rows on some
runs. A run with fewer rows than usual is normal variance, not necessarily a bug.

## Deployment

Render free tier, auto-deploying `src/api.py` from GitHub on push to main.

- The public URL is stable across deploys. A push replaces what is served there.
- A failed *build* leaves the previous deploy up; a successful build that crashes
  at *runtime* serves 502s. Always run `python -m src.api` locally before pushing.
- Free tier spins down when idle; first request after that takes 30–60s.
- To see which data is live: `/api/hero-stats` → `updated` is the data's fetch
  time (UTC, from `fetched_at`), and the startup log line
  `Serving data from …/snapshots/run-NN` names the snapshot.
- Every response sends `Cache-Control: no-cache`, so browsers revalidate
  `app.js` after a deploy instead of running stale frontend code against the
  new API.

## Working agreement

- Run the test suite before proposing a push.
- Do not push directly to main when the live site is about to be shown to someone.
  Branch, verify locally, then merge.
- Flag anything that changes a research claim (scoring weights, validation method,
  what counts as a signal) rather than changing it silently.
