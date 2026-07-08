"""
api.py — Backend API for the M&A screener's custom frontend.

Wraps the EXISTING, already-tested pipeline logic (accretion_dilution.py,
acquisition_likelihood.py) behind HTTP endpoints, so the custom frontend's
interactive features (deal simulator, best-targets scan) get live
computation rather than just static data — while reusing every bit of
validated logic rather than reimplementing it.

Serves the frontend directly from the same process (mounted static files),
so running this one command gives you the whole app — no separate
frontend server, no CORS configuration needed.

Usage:
    python -m src.api
    (then open http://localhost:8000 in a browser)
"""

import logging
import math
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.accretion_dilution import DealInputs, run_deal, find_optimal_terms, find_best_targets, project_multi_year_accretion
from src.acquisition_likelihood import score_acquisition_likelihood

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = "data/processed"
WEB_DIR = str(Path(__file__).resolve().parent.parent / "web")

app = FastAPI(title="M&A Screener API")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Without this, ANY unhandled exception shows the frontend a generic
    'Internal Server Error' with zero information — undiagnosable from a
    screenshot alone. This logs the full traceback server-side (visible
    in the terminal running `python -m src.api`) AND returns the actual
    error message to the frontend, so a crash is immediately actionable
    instead of a dead end."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}:\n{traceback.format_exc()}")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


import json as _json


def _clean_records(df: pd.DataFrame) -> list:
    """Serialize via pandas' own to_json rather than manual NaN-replacement.
    Manual .where(pd.notnull(df), None) doesn't reliably handle every
    pandas null variant (nullable Int64's pd.NA in particular, which
    scoring.py and acquisition_likelihood.py both produce via
    .astype('Int64') for sector_rank columns) — pandas' own JSON encoder
    already knows how to correctly serialize NaN/NA/nullable-dtype/numpy
    scalar values to valid JSON, so route through that instead of
    reimplementing it by hand."""
    return _json.loads(df.to_json(orient="records"))


def _clean_dict(d: dict) -> dict:
    """Same robust path as _clean_records, applied to a single dict result
    (e.g. a DealResult) by wrapping it in a one-row DataFrame."""
    return _json.loads(pd.DataFrame([d]).to_json(orient="records"))[0]


def _load_scored() -> pd.DataFrame:
    path = f"{DATA_DIR}/scored_universe.csv"
    if not Path(path).exists():
        raise HTTPException(503, f"No scored data found at {path} — run `python -m src.run_pipeline` first.")
    return pd.read_csv(path)


def _safe_read(path: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None


@app.get("/api/hero-stats")
def get_hero_stats():
    df = _load_scored()
    top = df.sort_values("rank").iloc[0] if "rank" in df.columns and not df.empty else None
    path = f"{DATA_DIR}/scored_universe.csv"
    try:
        updated = datetime.fromtimestamp(Path(path).stat().st_mtime).strftime("%b %d, %H:%M")
    except OSError:
        updated = "—"
    avg_score = df["composite_score"].mean() if "composite_score" in df.columns else None
    return {
        "universe": len(df),
        "sectors": int(df["sector"].nunique()) if "sector" in df.columns else 0,
        "top_pick": top["ticker"] if top is not None else "—",
        "avg_composite": round(float(avg_score), 3) if avg_score is not None and pd.notna(avg_score) else None,
        "updated": updated,
    }


@app.get("/api/sectors")
def get_sectors():
    df = _load_scored()
    return sorted(df["sector"].dropna().unique().tolist()) if "sector" in df.columns else []


@app.get("/api/tickers")
def get_tickers():
    df = _load_scored()
    return sorted(df["ticker"].tolist())


@app.get("/api/tickers/deal-eligible")
def get_deal_eligible_tickers():
    """Only tickers with every field the accretion/dilution model needs
    (price, market_cap, return_on_equity, total_equity, positive equity).
    Used to populate the Deal Simulator and Best Targets dropdowns so the
    default selection can never silently be a company that immediately
    errors — a real bug found via user testing, not a hypothetical one."""
    df = _load_scored()
    required = [c for c in ["price", "market_cap", "return_on_equity", "total_equity"] if c in df.columns]
    valid = df.dropna(subset=required)
    if "total_equity" in valid.columns:
        valid = valid[valid["total_equity"] > 0]
    return sorted(valid["ticker"].tolist())


@app.get("/api/shortlist")
def get_shortlist(sector: Optional[str] = None, top_n: int = 200):
    df = _load_scored()
    if sector:
        df = df[df["sector"] == sector]
    df = df.sort_values("rank").head(top_n)
    return _clean_records(df)


@app.get("/api/comps-dupont")
def get_comps_dupont():
    df = _safe_read(f"{DATA_DIR}/comps_dupont_report.csv")
    return _clean_records(df) if df is not None else []


@app.get("/api/explanations")
def get_explanations():
    df = _safe_read(f"{DATA_DIR}/explanations.csv")
    return _clean_records(df) if df is not None else []


@app.get("/api/baseline")
def get_baseline():
    df = _safe_read(f"{DATA_DIR}/baseline_comparison.csv")
    return _clean_records(df) if df is not None else []


@app.get("/api/acquisition-likelihood")
def get_acquisition_likelihood(sector: Optional[str] = None, top_n: int = 200):
    df = _load_scored()
    try:
        result = score_acquisition_likelihood(df)
    except ValueError as e:
        raise HTTPException(400, str(e))
    if sector:
        result = result[result["sector"] == sector]
    return _clean_records(result.head(top_n))


class DealRequest(BaseModel):
    acquirer: str
    target: str
    premium_pct: float = 0.25
    cash_pct: float = 0.5
    debt_funded_pct: float = 0.0
    interest_rate: float = 0.08
    tax_rate: float = 0.25
    revenue_synergy_pct: float = 0.0
    cost_synergy_pct: float = 0.0


def _build_deal(df: pd.DataFrame, req: DealRequest) -> DealInputs:
    if req.acquirer == req.target:
        raise HTTPException(400, "Acquirer and target must be different companies.")
    return DealInputs(
        req.acquirer, req.target, req.premium_pct, req.cash_pct,
        debt_funded_pct=req.debt_funded_pct, interest_rate=req.interest_rate, tax_rate=req.tax_rate,
        revenue_synergy_pct=req.revenue_synergy_pct, cost_synergy_pct=req.cost_synergy_pct,
    )


@app.post("/api/deal")
def post_deal(req: DealRequest):
    df = _load_scored()
    try:
        deal = _build_deal(df, req)
        result = run_deal(df, deal)
        return _clean_dict(asdict(result))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/deal/multi-year")
def post_multi_year(req: DealRequest, years: int = 3):
    df = _load_scored()
    try:
        deal = _build_deal(df, req)
        result = project_multi_year_accretion(df, deal, years=years)
        return _clean_records(result)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/deal/heatmap")
def post_heatmap(req: DealRequest):
    df = _load_scored()
    try:
        if req.acquirer == req.target:
            raise HTTPException(400, "Acquirer and target must be different companies.")
        optimal = find_optimal_terms(
            df, req.acquirer, req.target,
            debt_funded_pct=req.debt_funded_pct, interest_rate=req.interest_rate, tax_rate=req.tax_rate,
            revenue_synergy_pct=req.revenue_synergy_pct, cost_synergy_pct=req.cost_synergy_pct,
        )
        return {"best": _clean_dict(optimal["best"]), "grid": _clean_records(optimal["grid"])}
    except ValueError as e:
        raise HTTPException(400, str(e))


class BestTargetsRequest(BaseModel):
    acquirer: str
    premium_pct: float = 0.25
    cash_pct: float = 0.5
    debt_funded_pct: float = 0.0
    interest_rate: float = 0.08
    tax_rate: float = 0.25
    max_target_size_pct: float = 1.0
    sector: Optional[str] = None
    top_n: int = 20


@app.post("/api/best-targets")
def post_best_targets(req: BestTargetsRequest):
    df = _load_scored()
    try:
        result = find_best_targets(
            df, req.acquirer,
            premium_pct=req.premium_pct, cash_pct=req.cash_pct,
            debt_funded_pct=req.debt_funded_pct, interest_rate=req.interest_rate, tax_rate=req.tax_rate,
            max_target_size_pct=req.max_target_size_pct,
        )
        if req.sector:
            result = result[result["sector"] == req.sector]
        return _clean_records(result.head(req.top_n))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/summary-pdf/{ticker}")
def get_summary_pdf(ticker: str):
    from fastapi.responses import FileResponse
    from src.company_summary import generate_summary_pdf
    import tempfile

    try:
        out_path = str(Path(tempfile.gettempdir()) / f"summary_{ticker.replace('.', '_')}.pdf")
        generate_summary_pdf(ticker, out_path)
        return FileResponse(out_path, media_type="application/pdf", filename=f"summary_{ticker.replace('.', '_')}.pdf")
    except ValueError as e:
        raise HTTPException(400, str(e))


# Serve the frontend LAST — this mount catches all remaining paths, so it
# must come after every /api/* route is registered above.
if Path(WEB_DIR).exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
