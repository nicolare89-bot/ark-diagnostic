// ui_panels.js — Φάσι 3 UI panels (Plotly bars + gauge + trajectory + DT 3D + stones)
//
// Περιμένει τὸ event 'ark:ready' ἀπὸ τὸ main.js, μετὰ συνδέει τὰ panels.
// window.ark = { analyze, trajectory, pyodide, stones } γίνεται διαθέσιμο.
//
// V − E + F = 2.   ε = 1.5%.

const IRREPS = ['A', 'T1', 'T2', 'G', 'H'];
const IRREP_COLORS = {
  A:  '#d4af37',  // β₁₀ Πατρικά palette
  T1: '#9aa7b4',
  T2: '#7fb069',  // β₆ Υἱικά
  G:  '#6cb4ee',  // β₄ Πνευματικά
  H:  '#ee6cae',
};
const RANKS = { A: 1, T1: 3, T2: 3, G: 8, H: 15 };
const EPSILON_STAR = 0.0150138114;
const PLOTLY_LAYOUT_BASE = {
  paper_bgcolor: 'transparent',
  plot_bgcolor: 'transparent',
  font: { family: 'Georgia, serif', color: '#e8eef5' },
  margin: { t: 10, r: 10, b: 40, l: 50 },
};

const STATE = { lastResult: null, lastTrajectory: null };

window.addEventListener('ark:ready', onReady);

function onReady() {
  console.log('[ui] ark ready, wiring panels');
  const runBtn = document.getElementById('btn-run');
  const trajBtn = document.getElementById('btn-trajectory');
  const resetBtn = document.getElementById('btn-reset');
  runBtn.disabled = false;
  trajBtn.disabled = false;
  runBtn.addEventListener('click', handleRun);
  trajBtn.addEventListener('click', handleTrajectory);
  resetBtn.addEventListener('click', handleReset);

  // Stones panel γεμίζει ἀπὸ τὸ window.ark.stones (static index)
  initStonesPanel();
  // Wu structural panel — δείχνει τὴ δομὴ τοῦ bicomplex (input-independent)
  initWuPanel();
  // Hashimoto spectrum panel — 360 eigenvalues στὸ μιγαδικὸ ἐπίπεδο
  initHashimotoPanel();
}

// ─── input parsing ────────────────────────────────────────────────

function parseNumbers(text) {
  if (!text) return [];
  const tokens = text.split(/[\s,;]+/).map((t) => t.trim()).filter(Boolean);
  const nums = [];
  for (const t of tokens) {
    const x = Number(t);
    if (Number.isFinite(x)) nums.push(x);
  }
  return nums;
}

// ─── run analyze ──────────────────────────────────────────────────

async function handleRun() {
  const text = document.getElementById('input-numbers').value;
  const strategy = document.getElementById('input-strategy').value;
  const numbers = parseNumbers(text);
  if (numbers.length === 0) {
    flashError('Δὲν ἔδωσες ἀριθμούς.');
    return;
  }
  try {
    const result = window.ark.analyze(numbers, strategy);
    STATE.lastResult = result;
    showDiagnosticsPanel();
    renderBars(result);
    renderGauge(result);
    renderVerdict(result);
    renderDT3D(result);
    renderLocalBeta10(result);
  } catch (err) {
    flashError(err?.message ?? String(err));
  }
}

async function handleTrajectory() {
  try {
    const traj = window.ark.trajectory(60);
    STATE.lastTrajectory = traj;
    showDiagnosticsPanel();
    document.getElementById('trajectory-block').hidden = false;
    renderTrajectory(traj);
  } catch (err) {
    flashError(err?.message ?? String(err));
  }
}

function handleReset() {
  document.getElementById('input-numbers').value = '';
  document.getElementById('diagnostics-panel').hidden = true;
  document.getElementById('dt-panel').hidden = true;
  document.getElementById('trajectory-block').hidden = true;
}

function showDiagnosticsPanel() {
  document.getElementById('diagnostics-panel').hidden = false;
}

function flashError(msg) {
  const banner = document.createElement('div');
  banner.className = 'error-banner';
  banner.setAttribute('role', 'alert');
  banner.innerHTML = `<strong>Σφάλμα:</strong> <span></span>`;
  banner.querySelector('span').textContent = msg;
  document.querySelector('main').prepend(banner);
  setTimeout(() => banner.remove(), 5000);
}

// ─── viz #1: Plotly bars ─────────────────────────────────────────

function renderBars(result) {
  const energies = IRREPS.map((k) => result.energies[k] ?? 0);
  const expected = IRREPS.map((k) => RANKS[k] / 30);
  const trace_measured = {
    x: IRREPS,
    y: energies,
    type: 'bar',
    name: 'measured',
    marker: { color: IRREPS.map((k) => IRREP_COLORS[k]) },
  };
  const trace_expected = {
    x: IRREPS,
    y: expected,
    type: 'scatter',
    mode: 'markers',
    name: 'isotropic',
    marker: { symbol: 'line-ew-open', size: 28, color: '#ffffff', line: { width: 2 } },
  };
  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    height: 280,
    yaxis: { title: 'energy', range: [0, Math.max(1.0, Math.max(...energies) * 1.1)], gridcolor: '#25303d' },
    xaxis: { title: 'irrep' },
    legend: { orientation: 'h', y: -0.2 },
  };
  Plotly.react('plot-bars', [trace_measured, trace_expected], layout, { displayModeBar: false, responsive: true });
}

// ─── viz #2: gauge ───────────────────────────────────────────────

function renderGauge(result) {
  const drift = result.drift_fraction || 0;
  const driftPct = drift * 100;  // ε/ε* σὲ %·ε*
  const maxAxis = Math.max(50, driftPct * 1.2);
  const trace = {
    type: 'indicator',
    mode: 'gauge+number',
    value: driftPct,
    number: { suffix: '%·ε*', font: { size: 28 } },
    gauge: {
      axis: { range: [0, maxAxis], tickcolor: '#e8eef5' },
      bar: { color: '#d4af37' },
      bgcolor: '#131a23',
      steps: [
        { range: [0, 100],    color: 'rgba(127, 176, 105, 0.25)' },  // viable
        { range: [100, 500],  color: 'rgba(212, 175, 55, 0.25)' },   // tolerant
        { range: [500, 1200], color: 'rgba(238, 108, 108, 0.20)' },  // Cheeger
        { range: [1200, maxAxis], color: 'rgba(217, 102, 102, 0.35)' },
      ],
      threshold: { line: { color: '#ffffff', width: 2 }, thickness: 0.75, value: 100 },
    },
  };
  Plotly.react('plot-gauge', [trace], { ...PLOTLY_LAYOUT_BASE, height: 280 }, { displayModeBar: false, responsive: true });
}

function renderVerdict(result) {
  const el = document.getElementById('verdict-text');
  if (!el) return;
  el.textContent = result.verdict_drift || '';
  el.dataset.state = result.within_living_imperfection ? 'ok' : 'warn';
}

// ─── viz #3: trajectory time series ──────────────────────────────

function renderTrajectory(traj) {
  const trace_drift = {
    x: traj.t, y: traj.drift_fraction,
    type: 'scatter', mode: 'lines', name: 'ε/ε*',
    line: { color: '#d4af37', width: 2 },
  };
  const trace_T1 = {
    x: traj.t, y: traj.energy_T1,
    type: 'scatter', mode: 'lines', name: 'T₁',
    line: { color: '#9aa7b4', width: 1, dash: 'dot' }, yaxis: 'y2',
  };
  const trace_H = {
    x: traj.t, y: traj.energy_H,
    type: 'scatter', mode: 'lines', name: 'H',
    line: { color: '#ee6cae', width: 1, dash: 'dot' }, yaxis: 'y2',
  };
  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    height: 320,
    xaxis: { title: 't (βῆμα)', gridcolor: '#25303d' },
    yaxis: { title: 'ε/ε*', gridcolor: '#25303d', side: 'left' },
    yaxis2: { title: 'energy', overlaying: 'y', side: 'right', range: [0, 1] },
    legend: { orientation: 'h', y: -0.2 },
    shapes: [{
      type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 1, y1: 1,
      line: { color: '#7fb069', width: 1, dash: 'dash' },
    }],
  };
  Plotly.react('plot-trajectory', [trace_drift, trace_T1, trace_H], layout, { displayModeBar: false, responsive: true });
}

// ─── viz #4: DT 3D heat-map (Plotly Scatter3d — Three.js ἔρχεται Φάσι 3.2) ─

function renderDT3D(result) {
  document.getElementById('dt-panel').hidden = false;
  const dt = window.ark.dt;  // θὰ τὸ ἐκθέσουμε στὸ main.js
  if (!dt || !Array.isArray(result.heat_beta4)) return;
  const beta4_idx = dt.orbits['β4'];
  const beta10_idx = dt.orbits['β10'];
  const beta6_idx = dt.orbits['β6'];
  const coords = dt.coords;
  const pick = (idxs) => ({
    x: idxs.map((i) => coords[i][0]),
    y: idxs.map((i) => coords[i][1]),
    z: idxs.map((i) => coords[i][2]),
  });
  const beta4_pts = pick(beta4_idx);
  const beta10_pts = pick(beta10_idx);
  const beta6_pts = pick(beta6_idx);

  const beta4_trace = {
    type: 'scatter3d', mode: 'markers',
    x: beta4_pts.x, y: beta4_pts.y, z: beta4_pts.z,
    marker: {
      size: 7,
      color: result.heat_beta4,
      colorscale: 'YlOrRd',
      cmin: 0, cmax: 1,
      showscale: true,
      colorbar: { title: 'heat', tickfont: { color: '#e8eef5' } },
      line: { color: '#0b0f14', width: 0.5 },
    },
    name: 'β₄ (30)',
  };
  const beta10_trace = {
    type: 'scatter3d', mode: 'markers',
    x: beta10_pts.x, y: beta10_pts.y, z: beta10_pts.z,
    marker: { size: 5, color: '#d4af37', opacity: 0.85 },
    name: 'β₁₀ (12)',
  };
  const beta6_trace = {
    type: 'scatter3d', mode: 'markers',
    x: beta6_pts.x, y: beta6_pts.y, z: beta6_pts.z,
    marker: { size: 4, color: '#7fb069', opacity: 0.75 },
    name: 'β₆ (20)',
  };

  // Edges ὡς line3d traces (180 ἀκμές → 1 trace μὲ None-separated coords)
  const edges = dt.edges;
  const ex = [], ey = [], ez = [];
  for (const [a, b] of edges) {
    ex.push(coords[a][0], coords[b][0], null);
    ey.push(coords[a][1], coords[b][1], null);
    ez.push(coords[a][2], coords[b][2], null);
  }
  const edge_trace = {
    type: 'scatter3d', mode: 'lines',
    x: ex, y: ey, z: ez,
    line: { color: '#25303d', width: 1 },
    hoverinfo: 'skip',
    name: 'edges (180)',
    showlegend: false,
  };

  const layout = {
    ...PLOTLY_LAYOUT_BASE,
    height: 480,
    scene: {
      xaxis: { color: '#9aa7b4', backgroundcolor: 'transparent' },
      yaxis: { color: '#9aa7b4', backgroundcolor: 'transparent' },
      zaxis: { color: '#9aa7b4', backgroundcolor: 'transparent' },
      bgcolor: 'transparent',
    },
    legend: { orientation: 'h', y: -0.05 },
  };
  Plotly.react('dt-3d', [edge_trace, beta10_trace, beta6_trace, beta4_trace], layout, { displayModeBar: false, responsive: true });
}

// ─── viz #4b: local β₁₀ neighborhood (12-bar chart) ──────────────

function renderLocalBeta10(result) {
  const data = result.local_beta10;
  if (!data || !Array.isArray(data.signals) || data.signals.length === 0) return;
  document.getElementById('beta10-panel').hidden = false;

  const n = data.signals.length;
  const labels = Array.from({ length: n }, (_, i) => `β₁₀ #${i + 1}`);
  const domIdx = data.vertex_ids.indexOf(data.dominant_vertex);
  const colors = data.signals.map((_, i) =>
    i === domIdx ? '#ffd966' : '#d4af37'  // dominant ξεχωρίζει μὲ ἀνοιχτότερο χρυσό
  );
  const lineWidths = data.signals.map((_, i) => (i === domIdx ? 2 : 0.5));

  const bar = {
    type: 'bar',
    x: labels,
    y: data.signals,
    marker: { color: colors, line: { color: '#0b0f14', width: lineWidths } },
    hovertemplate: '%{x} (DT v=%{customdata})<br>signal: %{y:.4f}<extra></extra>',
    customdata: data.vertex_ids,
  };
  const layout = {
    height: 280,
    margin: { l: 50, r: 20, t: 10, b: 60 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9aa7b4' },
    xaxis: { tickangle: -45, color: '#9aa7b4' },
    yaxis: { title: { text: getI18n('beta10.y_axis', 'local signal') }, color: '#9aa7b4' },
  };
  Plotly.react('plot-beta10', [bar], layout, { displayModeBar: false, responsive: true });

  // Summary text
  const domVal = document.getElementById('beta10-dominant-value');
  const aniVal = document.getElementById('beta10-anisotropy-value');
  if (domVal) domVal.textContent = `β₁₀ #${domIdx + 1} (DT v=${data.dominant_vertex})`;
  if (aniVal) aniVal.textContent = `${data.anisotropy_pct.toFixed(1)}%`;
}

function getI18n(key, fallback) {
  // Διαβάζει ἀπὸ τὸ πρώτο stored data-i18n element ποὺ ταιριάζει.
  const node = document.querySelector(`[data-i18n="${key}"]`);
  return node?.textContent || fallback;
}

// ─── viz #4c: Wu bicomplex structure (static, input-independent) ─

function initWuPanel() {
  const wu = window.ark.wu;
  if (!wu || !Array.isArray(wu.tot_dims) || wu.tot_dims.length === 0) return;
  document.getElementById('wu-panel').hidden = false;

  // Summary values
  const b4El = document.getElementById('wu-b4-value');
  const totEl = document.getElementById('wu-total-value');
  const total = (wu.tot_dims || []).reduce((a, b) => a + b, 0);
  if (b4El) b4El.textContent = String(wu.b4);
  if (totEl) totEl.textContent = String(total);

  // (a) Tot^n bars (filtration)
  const totBar = {
    type: 'bar',
    x: wu.tot_dims.map((_, n) => `Tot^${n}`),
    y: wu.tot_dims,
    marker: { color: '#7fb069', line: { color: '#0b0f14', width: 0.5 } },
    text: wu.tot_dims.map((v) => String(v)),
    textposition: 'auto',
    hovertemplate: '%{x}: %{y}<extra></extra>',
  };
  const totLayout = {
    height: 240,
    margin: { l: 50, r: 20, t: 10, b: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9aa7b4' },
    xaxis: { color: '#9aa7b4' },
    yaxis: { color: '#9aa7b4', type: 'log', title: { text: 'dim (log)' } },
  };
  Plotly.react('plot-wu-tot', [totBar], totLayout, { displayModeBar: false, responsive: true });

  // (b) Bidegree 3×3 heatmap
  const grid = wu.bidegree;
  if (Array.isArray(grid) && grid.length === 3) {
    const flat = grid.flat();
    const annotations = [];
    for (let p = 0; p < 3; p++) {
      for (let q = 0; q < 3; q++) {
        annotations.push({
          x: q, y: p, xref: 'x', yref: 'y',
          text: String(grid[p][q]),
          showarrow: false,
          font: { color: grid[p][q] > 300 ? '#0b0f14' : '#e8eef5', size: 14 },
        });
      }
    }
    const heat = {
      type: 'heatmap',
      z: grid,
      x: ['q=0', 'q=1', 'q=2'],
      y: ['p=0', 'p=1', 'p=2'],
      colorscale: 'YlOrRd',
      showscale: false,
      hovertemplate: 'p=%{y}, q=%{x}: %{z}<extra></extra>',
    };
    const heatLayout = {
      height: 240,
      margin: { l: 50, r: 20, t: 10, b: 40 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: { color: '#9aa7b4' },
      xaxis: { color: '#9aa7b4', side: 'bottom' },
      yaxis: { color: '#9aa7b4', autorange: 'reversed' },
      annotations,
    };
    Plotly.react('plot-wu-bideg', [heat], heatLayout, { displayModeBar: false, responsive: true });
  }
}

// ─── viz #4d: Hashimoto spectrum (static, complex plane scatter) ─

function initHashimotoPanel() {
  const h = window.ark.hashimoto;
  if (!h || !Array.isArray(h.re) || h.re.length === 0) return;
  document.getElementById('hashimoto-panel').hidden = false;

  // Summary
  const rhoEl  = document.getElementById('hashimoto-rho');
  const gapEl  = document.getElementById('hashimoto-gap');
  const realEl = document.getElementById('hashimoto-real');
  const iharaEl = document.getElementById('hashimoto-ihara');
  if (rhoEl)  rhoEl.textContent  = h.spectral_radius.toFixed(4);
  if (gapEl)  gapEl.textContent  = h.spectral_gap.toFixed(4);
  if (realEl) realEl.textContent = `${h.n_real_eigvals}/${h.re.length}`;
  if (iharaEl) {
    const okKey  = 'hashimoto.ihara_ok';
    const failKey = 'hashimoto.ihara_fail';
    iharaEl.textContent = getI18n(h.ihara_holds ? okKey : failKey,
      h.ihara_holds ? '✓ Ihara' : '✗ Ihara');
    iharaEl.style.color = h.ihara_holds ? '#7fb069' : '#d96666';
  }

  // Scatter στὸ μιγαδικὸ ἐπίπεδο: marker μέγεθος ≈ |λ| γιὰ ἔμφαση στὰ μεγαλύτερα
  const abs_eigs = h.re.map((r, i) => Math.sqrt(r * r + h.im[i] * h.im[i]));
  const trace = {
    type: 'scatter', mode: 'markers',
    x: h.re, y: h.im,
    marker: {
      size: 5,
      color: abs_eigs,
      colorscale: 'YlOrRd',
      cmin: 0, cmax: h.spectral_radius,
      showscale: true,
      colorbar: { title: '|λ|', tickfont: { color: '#e8eef5' } },
      opacity: 0.85,
      line: { color: '#0b0f14', width: 0.5 },
    },
    hovertemplate: 'λ = %{x:.3f} + %{y:.3f}i<br>|λ| = %{marker.color:.3f}<extra></extra>',
  };
  // Unit circle reference (Ihara connection: poles of ζ at |z|·sqrt(d-1) ≈ 1)
  const tArr = Array.from({ length: 100 }, (_, i) => (2 * Math.PI * i) / 99);
  const unit = {
    type: 'scatter', mode: 'lines',
    x: tArr.map(Math.cos), y: tArr.map(Math.sin),
    line: { color: '#9aa7b4', width: 1, dash: 'dot' },
    showlegend: false,
    hoverinfo: 'skip',
  };
  const layout = {
    height: 320,
    margin: { l: 60, r: 20, t: 10, b: 40 },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: { color: '#9aa7b4' },
    xaxis: { title: 'Re(λ)', color: '#9aa7b4', zerolinecolor: '#25303d', gridcolor: '#1d2630' },
    yaxis: { title: 'Im(λ)', color: '#9aa7b4', zerolinecolor: '#25303d', gridcolor: '#1d2630', scaleanchor: 'x', scaleratio: 1 },
    showlegend: false,
  };
  Plotly.react('plot-hashimoto', [unit, trace], layout, { displayModeBar: false, responsive: true });
}

// ─── viz #5: stones panel ────────────────────────────────────────

function initStonesPanel() {
  const stones = window.ark.stones?.stones || [];
  if (stones.length === 0) return;  // ἂν τὸ index εἶναι κενό, ἀφήνουμε τὸ panel κρυφό
  document.getElementById('stones-panel').hidden = false;
  const searchInput = document.getElementById('stones-search');
  const list = document.getElementById('stones-list');
  renderStones(list, stones);
  searchInput.addEventListener('input', () => {
    const q = searchInput.value.trim().toLowerCase();
    const filtered = q
      ? stones.filter((s) =>
          (s.id || '').toLowerCase().includes(q) ||
          (s.title || '').toLowerCase().includes(q) ||
          (s.statement || '').toLowerCase().includes(q))
      : stones;
    renderStones(list, filtered);
  });
}

function renderStones(list, stones) {
  list.innerHTML = '';
  if (stones.length === 0) {
    list.innerHTML = `<p class="stones-empty" data-i18n="stones.no_results">Δὲν βρέθηκαν πέτρες</p>`;
    return;
  }
  for (const s of stones.slice(0, 50)) {
    const card = document.createElement('article');
    card.className = 'stone-card';
    card.innerHTML = `
      <header>
        <span class="stone-layer">${s.layer || '·'}</span>
        <code class="stone-id">${s.id}</code>
        <span class="stone-stars">${'★'.repeat(s.stars || 0)}</span>
      </header>
      <h4 class="stone-title">${escapeHtml(s.title || '')}</h4>
      <p class="stone-statement">${escapeHtml(s.statement || '')}</p>
    `;
    list.appendChild(card);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}
