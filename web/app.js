// ============================================================================
// M&A Screener — shared frontend logic across all pages
// Each page sets <body data-page="..."> to identify itself; boot() renders
// the shared nav and calls only that page's init function.
// ============================================================================

const qs = (sel, root = document) => root.querySelector(sel);
const qsa = (sel, root = document) => Array.from(root.querySelectorAll(sel));

async function fetchJSON(url, options = {}) {
    const res = await fetch(url, { headers: { "Content-Type": "application/json" }, ...options });
    if (!res.ok) {
        let detail = res.statusText;
        try {
            const body = await res.json();
            detail = body.detail || detail;
        } catch (_) { /* ignore */ }
        throw new Error(detail);
    }
    return res.json();
}

function debounce(fn, delay = 350) {
    let timer = null;
    return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), delay); };
}

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function fmtNum(v, decimals = 2) { return (v === null || v === undefined || Number.isNaN(v)) ? "&mdash;" : Number(v).toFixed(decimals); }
function fmtPct(v, decimals = 1) { return (v === null || v === undefined || Number.isNaN(v)) ? "&mdash;" : (Number(v) * 100).toFixed(decimals) + "%"; }
function fmtSignedPct(v, decimals = 2) {
    if (v === null || v === undefined || Number.isNaN(v)) return "&mdash;";
    const n = Number(v);
    return (n > 0 ? "+" : "") + n.toFixed(decimals) + "%";
}
function fmtInt(v) { return (v === null || v === undefined || Number.isNaN(v)) ? "&mdash;" : Math.round(Number(v)).toLocaleString(); }

// ---- Light-theme color helpers ----
function lerpHex(t, low, high) {
    t = Math.max(0, Math.min(1, t));
    const l = [1, 3, 5].map(i => parseInt(low.slice(i, i + 2), 16));
    const h = [1, 3, 5].map(i => parseInt(high.slice(i, i + 2), 16));
    const rgb = l.map((c, i) => Math.round(c + (h[i] - c) * t));
    return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}
function scoreCellStyle(val, vmin = 0, vmax = 1) {
    if (val === null || val === undefined || Number.isNaN(val)) return "";
    const t = vmax === vmin ? 0 : (val - vmin) / (vmax - vmin);
    const bg = lerpHex(t, "#EEEDFC", "#5B4FE8");
    const color = t > 0.6 ? "#FFFFFF" : "#1C1D2E";
    return `background-color:${bg};color:${color};`;
}
function badgeHtml(text, kind) {
    const cls = kind === "positive" ? "badge-positive" : kind === "negative" ? "badge-negative" : "badge-neutral";
    return `<span class="badge ${cls}">${escapeHtml(text)}</span>`;
}

// ============================================================================
// Shared nav — injected on every page, active link highlighted by data-page
// ============================================================================
function renderNav(activePage) {
    const placeholder = qs("#nav-placeholder");
    if (!placeholder) return;

    const isActive = (page) => activePage === page ? "active" : "";

    placeholder.innerHTML = `
        <nav id="navbar">
            <div class="nav-left">
                <a href="/shortlist.html" class="nav-wordmark">M&amp;A <span>Screener</span></a>
                <div class="nav-links">
                    <a href="/shortlist.html" class="nav-link ${isActive('shortlist')}">Shortlist</a>
                    <a href="/deal.html" class="nav-link ${isActive('deal')}">Deal Simulator</a>
                    <a href="/targets.html" class="nav-link ${isActive('targets')}">Best Targets</a>
                    <div class="nav-link tools-wrap tools-trigger ${isActive('tools')}" id="tools-trigger">
                        Tools
                        <div class="tools-menu">
                            <a href="/dupont.html" class="tool-item">
                                <div class="tool-icon">&#128202;</div>
                                <div class="tool-content">
                                    <div class="tool-title">Comps + DuPont</div>
                                    <div class="tool-desc">Peer valuation and ROE decomposition for the shortlist.</div>
                                </div>
                            </a>
                            <a href="/likelihood.html" class="tool-item">
                                <div class="tool-icon">&#127919;</div>
                                <div class="tool-content">
                                    <div class="tool-title">Acquisition Likelihood</div>
                                    <div class="tool-desc">Value / turnaround candidates — a different lens from the main shortlist.</div>
                                </div>
                            </a>
                            <a href="/baseline.html" class="tool-item">
                                <div class="tool-icon">&#9878;&#65039;</div>
                                <div class="tool-content">
                                    <div class="tool-title">vs. Naive Baseline</div>
                                    <div class="tool-desc">Does the composite score beat a simple cheapest-P/E sort?</div>
                                </div>
                            </a>
                            <div class="tools-divider"></div>
                            <a href="/summary.html" class="tool-item">
                                <div class="tool-icon">&#128196;</div>
                                <div class="tool-content">
                                    <div class="tool-title">One-Page Summary</div>
                                    <div class="tool-desc">Generate a shareable PDF for any single company.</div>
                                </div>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
    `;

    // Click-to-toggle, click-outside-to-close — replaces the old
    // hover-only behavior, which closed the moment the mouse briefly
    // left the trigger's bounding box while moving toward the menu.
    const toolsWrap = qs("#tools-trigger");
    toolsWrap.addEventListener("click", (e) => {
        // Only toggle when clicking the trigger itself, not a link inside the menu
        if (e.target.closest(".tool-item")) return;
        e.stopPropagation();
        toolsWrap.classList.toggle("open");
    });
    document.addEventListener("click", (e) => {
        if (!toolsWrap.contains(e.target)) toolsWrap.classList.remove("open");
    });
}

// ============================================================================
// Generic table renderer
// ============================================================================
function renderTable(container, columns, rows) {
    if (!rows || rows.length === 0) {
        container.innerHTML = `<div class="empty-state">No rows to display.</div>`;
        return;
    }
    const thead = columns.map(c => `<th>${escapeHtml(c.label)}</th>`).join("");
    const tbody = rows.map(row => {
        const cells = columns.map(c => {
            const val = row[c.key];
            let content, style = "", cls = "";
            switch (c.type) {
                case "mono": content = escapeHtml(val); cls = "mono"; break;
                case "pct": content = fmtPct(val, c.decimals ?? 1); break;
                case "signedPct":
                    content = fmtSignedPct(val, c.decimals ?? 2);
                    if (val !== null && val !== undefined) style = `color:${val > 0 ? "#16A34A" : "#DC2626"};font-weight:700;`;
                    break;
                case "num": content = fmtNum(val, c.decimals ?? 2); break;
                case "int": content = fmtInt(val); break;
                case "score":
                    content = fmtNum(val, c.decimals ?? 3);
                    style = scoreCellStyle(val, c.scoreMin ?? 0, c.scoreMax ?? 1);
                    cls = "cell-score";
                    break;
                case "bool":
                    content = val === true ? badgeHtml("YES", "positive") : val === false ? badgeHtml("NO", "negative") : "&mdash;";
                    break;
                case "verdict":
                    content = val === true ? badgeHtml("ACCRETIVE", "positive") : val === false ? badgeHtml("DILUTIVE", "negative") : "&mdash;";
                    break;
                default: content = escapeHtml(val ?? "—");
            }
            return `<td class="${cls}" style="${style}">${content}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
    }).join("");
    container.innerHTML = `<table><thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table>`;
}

function setupSectorPills(container, sectors, onChangeCallback) {
    container.innerHTML = "";
    const selected = new Set();
    sectors.forEach(sector => {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = sector;
        pill.addEventListener("click", () => {
            if (selected.has(sector)) { selected.delete(sector); pill.classList.remove("active"); }
            else { selected.add(sector); pill.classList.add("active"); }
            onChangeCallback(selected);
        });
        container.appendChild(pill);
    });
    return selected;
}

function bindSliderDisplay(sliderId, valueId, suffix, transform) {
    const slider = qs(`#${sliderId}`);
    const valueEl = qs(`#${valueId}`);
    slider.addEventListener("input", () => {
        const v = transform ? transform(slider.value) : slider.value;
        valueEl.textContent = v + suffix;
    });
}

// ============================================================================
// Shortlist page
// ============================================================================
let shortlistData = [];
let explanationsData = [];

function renderShortlist() {
    const topN = parseInt(qs("#shortlist-top-n").value, 10);
    const activeSectors = qsa("#shortlist-sector-pills .pill.active").map(p => p.textContent);
    let rows = activeSectors.length > 0 ? shortlistData.filter(r => activeSectors.includes(r.sector)) : shortlistData;
    rows = [...rows].sort((a, b) => a.rank - b.rank).slice(0, topN);

    renderTable(qs("#shortlist-table"), [
        { key: "rank", label: "Rank", type: "int" },
        { key: "ticker", label: "Ticker", type: "mono" },
        { key: "sector", label: "Sector", type: "text" },
        { key: "sector_rank", label: "Sector Rank", type: "int" },
        { key: "composite_score", label: "Composite", type: "score" },
        { key: "trailing_pe", label: "P/E", type: "num", decimals: 1 },
        { key: "return_on_equity", label: "ROE", type: "pct" },
        { key: "debt_to_equity_ratio", label: "D/E", type: "num" },
        { key: "revenue_growth", label: "Rev. Growth", type: "pct" },
    ], rows);

    const wrap = qs("#shortlist-explanations");
    const matches = explanationsData.filter(e => rows.map(r => r.ticker).includes(e.ticker));
    wrap.innerHTML = matches.length === 0 ? "" : `<h3 style="margin-top:24px;">Why did each company rank this way?</h3>` + matches.map(e => `
        <div class="explanation-item">
            <div class="explanation-ticker">${escapeHtml(e.ticker)}</div>
            <p class="explanation-text">${escapeHtml(e.summary)}</p>
        </div>
    `).join("");
}

async function initShortlist() {
    try {
        const [sectors, shortlist, explanations] = await Promise.all([
            fetchJSON("/api/sectors"), fetchJSON("/api/shortlist?top_n=500"), fetchJSON("/api/explanations"),
        ]);
        shortlistData = shortlist;
        explanationsData = explanations;
        setupSectorPills(qs("#shortlist-sector-pills"), sectors, renderShortlist);
        qs("#shortlist-top-n").addEventListener("input", (e) => {
            qs("#shortlist-top-n-value").textContent = e.target.value;
            renderShortlist();
        });
        renderShortlist();
    } catch (e) {
        qs("#shortlist-table").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}

// ============================================================================
// Comps + DuPont page
// ============================================================================
async function initDupont() {
    try {
        const data = await fetchJSON("/api/comps-dupont");
        renderTable(qs("#dupont-table"), [
            { key: "rank", label: "Rank", type: "int" },
            { key: "ticker", label: "Ticker", type: "mono" },
            { key: "sector", label: "Sector", type: "text" },
            { key: "valuation_label", label: "Valuation", type: "text" },
            { key: "return_on_equity", label: "ROE", type: "pct" },
            { key: "net_margin", label: "Net Margin", type: "pct" },
            { key: "asset_turnover", label: "Asset Turnover", type: "num" },
            { key: "equity_multiplier", label: "Equity Mult.", type: "num" },
            { key: "dupont_implied_roe", label: "Implied ROE", type: "pct" },
            { key: "dupont_vs_reported_roe_gap", label: "Gap", type: "pct" },
        ], data);
    } catch (e) {
        qs("#dupont-table").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}

// ============================================================================
// Deal Simulator page
// ============================================================================
function currentDealRequest() {
    return {
        acquirer: qs("#deal-acquirer").value,
        target: qs("#deal-target").value,
        premium_pct: parseInt(qs("#deal-premium").value, 10) / 100,
        cash_pct: parseInt(qs("#deal-cash").value, 10) / 100,
        debt_funded_pct: parseInt(qs("#deal-debt-pct").value, 10) / 100,
        interest_rate: parseFloat(qs("#deal-interest-rate").value) / 100,
        tax_rate: parseInt(qs("#deal-tax-rate").value, 10) / 100,
        revenue_synergy_pct: parseInt(qs("#deal-revenue-synergy").value, 10) / 100,
        cost_synergy_pct: parseInt(qs("#deal-cost-synergy").value, 10) / 100,
    };
}

function renderHeatmap(container, best, grid) {
    const premiums = [...new Set(grid.map(r => r.premium_pct))].sort((a, b) => a - b);
    const cashPcts = [...new Set(grid.map(r => r.cash_pct))].sort((a, b) => a - b);
    const values = grid.map(r => r.accretion_dilution_pct).filter(v => v !== null);
    const vmax = Math.max(...values.map(Math.abs), 1);

    const cellColor = (v) => {
        if (v === null) return "background-color:#F4F5FB;";
        const t = (v + vmax) / (2 * vmax);
        const bg = t < 0.5 ? lerpHex(t * 2, "#FEE2E2", "#FFFFFF") : lerpHex((t - 0.5) * 2, "#FFFFFF", "#DCFCE7");
        return `background-color:${bg};`;
    };

    let html = `<div class="heatmap-wrap"><table class="heatmap-table"><thead><tr><th></th>`;
    cashPcts.forEach(c => { html += `<th>${Math.round(c * 100)}% cash</th>`; });
    html += `</tr></thead><tbody>`;
    premiums.forEach(p => {
        html += `<tr><th>${Math.round(p * 100)}% prem.</th>`;
        cashPcts.forEach(c => {
            const cell = grid.find(r => r.premium_pct === p && r.cash_pct === c);
            const v = cell ? cell.accretion_dilution_pct : null;
            const isBest = best && Math.abs(best.premium_pct - p) < 1e-9 && Math.abs(best.cash_pct - c) < 1e-9;
            html += `<td class="heatmap-cell${isBest ? " best" : ""}" style="${cellColor(v)}">${v !== null ? v.toFixed(1) + "%" : "—"}</td>`;
        });
        html += `</tr>`;
    });
    html += `</tbody></table></div>`;
    container.innerHTML = html;
}

const updateDeal = debounce(async () => {
    const acquirer = qs("#deal-acquirer").value;
    const target = qs("#deal-target").value;
    const resultsEl = qs("#deal-results");
    const heatmapEl = qs("#deal-heatmap");

    if (!acquirer || !target || acquirer === target) {
        resultsEl.innerHTML = `<div class="error-state">Acquirer and target must be different companies.</div>`;
        heatmapEl.innerHTML = "";
        qs("#deal-multi-year-wrap").style.display = "none";
        return;
    }

    const req = currentDealRequest();
    try {
        const result = await fetchJSON("/api/deal", { method: "POST", body: JSON.stringify(req) });
        resultsEl.innerHTML = `
            <div class="metric-row">
                <div class="metric-card">
                    <div class="metric-label">Acquirer Standalone EPS</div>
                    <div class="metric-value">${result.acquirer_standalone_eps.toFixed(2)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Pro-Forma EPS</div>
                    <div class="metric-value">${result.pro_forma_eps.toFixed(2)}</div>
                    <div class="metric-delta ${result.accretion_dilution_pct > 0 ? "positive" : "negative"}">${fmtSignedPct(result.accretion_dilution_pct * 100, 2)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Verdict</div>
                    <div style="margin-top:6px;">${badgeHtml(result.is_accretive ? "ACCRETIVE" : "DILUTIVE", result.is_accretive ? "positive" : "negative")}</div>
                </div>
            </div>
        `;

        if (req.revenue_synergy_pct > 0 || req.cost_synergy_pct > 0) {
            const multiYear = await fetchJSON("/api/deal/multi-year?years=3", { method: "POST", body: JSON.stringify(req) });
            qs("#deal-multi-year-wrap").style.display = "block";
            renderTable(qs("#deal-multi-year-table"), [
                { key: "year", label: "Year", type: "int" },
                { key: "synergy_ramp_pct", label: "Synergy Ramp", type: "num", decimals: 0 },
                { key: "acquirer_standalone_eps", label: "Standalone EPS", type: "num" },
                { key: "pro_forma_eps", label: "Pro-Forma EPS", type: "num" },
                { key: "synergy_income", label: "Synergy Income", type: "int" },
                { key: "accretion_dilution_pct", label: "Accretion/Dilution", type: "signedPct" },
                { key: "is_accretive", label: "Verdict", type: "verdict" },
            ], multiYear);
        } else {
            qs("#deal-multi-year-wrap").style.display = "none";
        }

        const heatmapResult = await fetchJSON("/api/deal/heatmap", { method: "POST", body: JSON.stringify(req) });
        renderHeatmap(heatmapEl, heatmapResult.best, heatmapResult.grid);
    } catch (e) {
        resultsEl.innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
        heatmapEl.innerHTML = "";
    }
}, 300);

async function initDealSimulator() {
    try {
        const tickers = await fetchJSON("/api/tickers/deal-eligible");
        const options = tickers.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join("");
        qs("#deal-acquirer").innerHTML = options;
        qs("#deal-target").innerHTML = options;
        if (tickers.length > 1) qs("#deal-target").value = tickers[1];

        bindSliderDisplay("deal-premium", "deal-premium-value", "%");
        bindSliderDisplay("deal-cash", "deal-cash-value", "%");
        bindSliderDisplay("deal-debt-pct", "deal-debt-pct-value", "%");
        bindSliderDisplay("deal-interest-rate", "deal-interest-rate-value", "%", v => parseFloat(v).toFixed(1));
        bindSliderDisplay("deal-tax-rate", "deal-tax-rate-value", "%");
        bindSliderDisplay("deal-revenue-synergy", "deal-revenue-synergy-value", "%");
        bindSliderDisplay("deal-cost-synergy", "deal-cost-synergy-value", "%");

        ["deal-acquirer", "deal-target", "deal-premium", "deal-cash", "deal-debt-pct",
         "deal-interest-rate", "deal-tax-rate", "deal-revenue-synergy", "deal-cost-synergy"
        ].forEach(id => qs(`#${id}`).addEventListener("input", updateDeal));

        updateDeal();
    } catch (e) {
        qs("#deal-results").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}

// ============================================================================
// Best Targets Finder page
// ============================================================================
let bestTargetsFullResult = [];

function renderBestTargets() {
    const activeSectors = qsa("#targets-sector-pills .pill.active").map(p => p.textContent);
    let rows = activeSectors.length > 0 ? bestTargetsFullResult.filter(r => activeSectors.includes(r.sector)) : bestTargetsFullResult;
    rows = rows.slice(0, 20);

    const nAccretive = bestTargetsFullResult.filter(r => r.is_accretive).length;
    qs("#targets-caption").textContent = `${nAccretive}/${bestTargetsFullResult.length} evaluated targets would be accretive under these terms.`;

    renderTable(qs("#targets-table"), [
        { key: "target", label: "Target", type: "mono" },
        { key: "sector", label: "Sector", type: "text" },
        { key: "relative_size_pct", label: "Rel. Size", type: "pct", decimals: 1 },
        { key: "target_composite_score", label: "Composite", type: "score" },
        { key: "target_acquisition_likelihood_score", label: "Acq. Likelihood", type: "score" },
        { key: "accretion_dilution_pct", label: "Accretion/Dilution", type: "signedPct" },
        { key: "is_accretive", label: "Verdict", type: "verdict" },
    ], rows);

    const chartEl = qs("#targets-chart");
    if (rows.length === 0) { chartEl.innerHTML = ""; return; }
    const maxAbs = Math.max(...rows.map(r => Math.abs(r.accretion_dilution_pct)), 1);
    chartEl.innerHTML = rows.map(r => {
        const widthPct = (Math.abs(r.accretion_dilution_pct) / maxAbs) * 100;
        const color = r.accretion_dilution_pct > 0 ? "#16A34A" : "#DC2626";
        return `
            <div class="bar-row">
                <div class="bar-label">${escapeHtml(r.target)}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${widthPct}%;background-color:${color};"></div></div>
                <div class="bar-value">${fmtSignedPct(r.accretion_dilution_pct, 1)}</div>
            </div>
        `;
    }).join("");
}

const updateBestTargets = debounce(async () => {
    const acquirer = qs("#targets-acquirer").value;
    const tableEl = qs("#targets-table");
    if (!acquirer) return;

    const req = {
        acquirer,
        premium_pct: parseInt(qs("#targets-premium").value, 10) / 100,
        cash_pct: parseInt(qs("#targets-cash").value, 10) / 100,
        debt_funded_pct: parseInt(qs("#targets-debt-pct").value, 10) / 100,
        interest_rate: parseFloat(qs("#targets-interest-rate").value) / 100,
        tax_rate: parseInt(qs("#targets-tax-rate").value, 10) / 100,
        max_target_size_pct: parseInt(qs("#targets-max-size").value, 10) / 100,
        top_n: 500,
    };
    tableEl.innerHTML = `<div class="loading">Scanning candidate targets&hellip;</div>`;
    try {
        bestTargetsFullResult = await fetchJSON("/api/best-targets", { method: "POST", body: JSON.stringify(req) });
        renderBestTargets();
    } catch (e) {
        tableEl.innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}, 350);

async function initBestTargets() {
    try {
        const [tickers, sectors] = await Promise.all([fetchJSON("/api/tickers/deal-eligible"), fetchJSON("/api/sectors")]);
        qs("#targets-acquirer").innerHTML = tickers.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join("");

        setupSectorPills(qs("#targets-sector-pills"), sectors, renderBestTargets);

        bindSliderDisplay("targets-premium", "targets-premium-value", "%");
        bindSliderDisplay("targets-cash", "targets-cash-value", "%");
        bindSliderDisplay("targets-max-size", "targets-max-size-value", "%");
        bindSliderDisplay("targets-debt-pct", "targets-debt-pct-value", "%");
        bindSliderDisplay("targets-interest-rate", "targets-interest-rate-value", "%", v => parseFloat(v).toFixed(1));
        bindSliderDisplay("targets-tax-rate", "targets-tax-rate-value", "%");

        ["targets-acquirer", "targets-premium", "targets-cash", "targets-max-size",
         "targets-debt-pct", "targets-interest-rate", "targets-tax-rate"
        ].forEach(id => qs(`#${id}`).addEventListener("input", updateBestTargets));

        updateBestTargets();
    } catch (e) {
        qs("#targets-table").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}

// ============================================================================
// Acquisition Likelihood page
// ============================================================================
let likelihoodData = [];

function renderLikelihood() {
    const topN = parseInt(qs("#likelihood-top-n").value, 10);
    const activeSectors = qsa("#likelihood-sector-pills .pill.active").map(p => p.textContent);
    let rows = activeSectors.length > 0 ? likelihoodData.filter(r => activeSectors.includes(r.sector)) : likelihoodData;
    rows = [...rows].sort((a, b) => a.acquisition_likelihood_rank - b.acquisition_likelihood_rank).slice(0, topN);

    renderTable(qs("#likelihood-table"), [
        { key: "acquisition_likelihood_rank", label: "Rank", type: "int" },
        { key: "ticker", label: "Ticker", type: "mono" },
        { key: "sector", label: "Sector", type: "text" },
        { key: "acquisition_likelihood_score", label: "Likelihood Score", type: "score" },
        { key: "valuation_bucket_score", label: "Valuation", type: "score" },
        { key: "turnaround_potential_score", label: "Turnaround", type: "score" },
        { key: "leverage_bucket_score", label: "Leverage", type: "score" },
        { key: "composite_score", label: "Main Composite", type: "score" },
        { key: "rank", label: "Main Rank", type: "int" },
    ], rows);
}

async function initLikelihood() {
    try {
        const [sectors, data] = await Promise.all([fetchJSON("/api/sectors"), fetchJSON("/api/acquisition-likelihood?top_n=500")]);
        likelihoodData = data;
        setupSectorPills(qs("#likelihood-sector-pills"), sectors, renderLikelihood);
        qs("#likelihood-top-n").addEventListener("input", (e) => {
            qs("#likelihood-top-n-value").textContent = e.target.value;
            renderLikelihood();
        });
        renderLikelihood();
    } catch (e) {
        qs("#likelihood-table").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}

// ============================================================================
// Baseline comparison page
// ============================================================================
async function initBaseline() {
    try {
        const data = await fetchJSON("/api/baseline");
        renderTable(qs("#baseline-table"), [
            { key: "ticker", label: "Ticker", type: "mono" },
            { key: "sector", label: "Sector", type: "text" },
            { key: "rank", label: "Composite Rank", type: "int" },
            { key: "naive_rank", label: "Naive P/E Rank", type: "int" },
            { key: "composite_score", label: "Composite Score", type: "score" },
            { key: "trailing_pe", label: "P/E", type: "num", decimals: 1 },
        ], data);
    } catch (e) {
        qs("#baseline-table").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
    }
}

// ============================================================================
// One-Page Summary page
// ============================================================================
async function initSummary() {
    try {
        const tickers = await fetchJSON("/api/tickers");
        qs("#summary-ticker").innerHTML = tickers.map(t => `<option value="${escapeHtml(t)}">${escapeHtml(t)}</option>`).join("");
    } catch (e) {
        qs("#summary-status").innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
        return;
    }

    qs("#summary-generate").addEventListener("click", async () => {
        const ticker = qs("#summary-ticker").value;
        const statusEl = qs("#summary-status");
        statusEl.innerHTML = `<div class="loading">Generating PDF for ${escapeHtml(ticker)}&hellip;</div>`;
        try {
            const res = await fetch(`/api/summary-pdf/${encodeURIComponent(ticker)}`);
            if (!res.ok) {
                const body = await res.json().catch(() => ({}));
                throw new Error(body.detail || res.statusText);
            }
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            statusEl.innerHTML = `<a class="btn" href="${url}" download="summary_${escapeHtml(ticker).replace('.', '_')}.pdf">Download PDF for ${escapeHtml(ticker)}</a>`;
        } catch (e) {
            statusEl.innerHTML = `<div class="error-state">${escapeHtml(e.message)}</div>`;
        }
    });
}

// ============================================================================
// Boot — detect the current page via <body data-page="..."> and initialize
// only that page's logic, after rendering the shared nav.
// ============================================================================
document.addEventListener("DOMContentLoaded", () => {
    const page = document.body.dataset.page;
    renderNav(page);

    const initializers = {
        shortlist: initShortlist,
        dupont: initDupont,
        deal: initDealSimulator,
        targets: initBestTargets,
        likelihood: initLikelihood,
        baseline: initBaseline,
        summary: initSummary,
    };
    if (initializers[page]) initializers[page]();
});
