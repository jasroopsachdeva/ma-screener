# Snapshot archive — forward validation

29 daily runs, 2026-07-08 to 2026-08-05. Downloaded from GitHub Actions
artifacts on 2026-08-05 (90-day retention would have started expiring in October).

## Known issues in this data

- Effective universe is 196, not 201. Five tickers excluded in all 29 runs:
  CROMPTON, INDIGO, TATACHEM, TMPV (missing trailing_pe), LTIM (empty payload).
  Missing trailing_pe likely means negative trailing EPS -> this rule
  systematically removes loss-making firms, which are over-represented among
  acquisition targets. Fix: demote trailing_pe from critical to optional.
- Transient batch fetch failures cost 4-12 rows in 7 of 29 runs. Identical
  failure counts across ticker groups indicate whole batches failing together.
  Affected names are large caps, so no selection bias on takeover candidacy.
- fetch_ok is always True - failed fetches are dropped before write, so the
  column cannot flag problems. Row count is the only signal.
- Run #23 took 8m03s vs ~3m45s baseline but returned complete data.
