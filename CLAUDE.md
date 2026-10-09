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

**Only one backtest is legitimate.**
`validate_forward_performance()` reads snapshots recorded in the past and measures
real price performance from that date forward. No look-ahead. It correctly reports
"insufficient history" until enough time has passed — that is not a bug to fix.

`compute_trailing_performance()` ranks on *today's* fundamentals and then checks
*past* price performance. It is structurally biased and informational only. Never
present its output as validation.

Current honest finding, from `src/validate_returns.py`: a benchmark-adjusted
full-universe decile backtest shows **no statistically detectable signal**
(top-minus-bottom ≈ −0.97%, 95% CI [−1.28, −0.63], spread positive in 0 of 7
windows, and the windows overlap heavily). Do not retune scoring weights to make
this number look better and then call it validated.

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
- `src/api.py` falls back to the newest committed snapshot when
  `data/processed/scored_universe.csv` is absent. This is what keeps the Render
  deploy from serving 503s on every endpoint, since the processed CSVs are
  gitignored and never reach GitHub. Do not "clean up" this fallback.
- `data/history/*.csv` is still gitignored and local-only. It holds the longest
  series (top-10 records only) and does not accumulate in CI, since runners are
  ephemeral.

## CI

`.github/workflows/run_screener.yml` — daily cron `0 3 * * *` (08:30 IST, though
GitHub queueing usually delays it to ~11:10–11:50 IST). `workflow_dispatch` is
enabled. Python 3.11, `timeout-minutes: 45`. A separate workflow runs the test
suite on every push.

Transient yfinance rate-limit dropouts cost a handful of large-cap rows on some
runs. A run with fewer rows than usual is normal variance, not necessarily a bug.

## Deployment

Render free tier, auto-deploying `src/api.py` from GitHub on push to main.

- The public URL is stable across deploys. A push replaces what is served there.
- A failed *build* leaves the previous deploy up; a successful build that crashes
  at *runtime* serves 502s. Always run `python -m src.api` locally before pushing.
- Free tier spins down when idle; first request after that takes 30–60s.

## Working agreement

- Run the test suite before proposing a push.
- Do not push directly to main when the live site is about to be shown to someone.
  Branch, verify locally, then merge.
- Flag anything that changes a research claim (scoring weights, validation method,
  what counts as a signal) rather than changing it silently.
