// main.js — boot logic γιὰ τὴν Κιβωτὸ PWA (Φάσι 2 + 3)
//
// Boot πρόοδος μὲ visible status + watchdog timer:
//   1. i18n      — φόρτωσι μεταφράσεων
//   2. sw        — ἐγγραφὴ service worker
//   3. pyodide  — boot Pyodide ἀπὸ τὸ CDN (v0.27.4)
//   4. packages — loadPackage numpy + scipy (~25 MB, 30+s πρώτη φορά)
//   5. sources  — fetch ark_*.py + γράψιμο στὸ Pyodide FS
//   6. state    — fetch dt.json + projectors.json + stones.json
//   7. init     — pwa_bridge.init_state (sanity V−E+F=2)
//   8. ready    — ἐκθέτει window.ark γιὰ τὰ UI panels
//
// V − E + F = 2.   ε = 1.5%.   J > 0.

const STATE = {
  lang: 'el',
  translations: {},
  pyodide: null,
  ready: false,
  bootStart: 0,
  step: '',
};

const PYODIDE_VERSION = '0.27.4';
const PYODIDE_INDEX = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
const PY_HOME = '/home/pyodide';

const PYTHON_SOURCES = [
  'ark_geometry.py',
  'ark_irreps.py',
  'ark_diagnostics.py',
  'ark_adapter.py',
  'ark_local_b10.py',
  'pwa_bridge.py',
];

const STATUS_EL = document.getElementById('status');
const STATUS_DOT = STATUS_EL?.querySelector('.dot');
const STATUS_TEXT = STATUS_EL?.querySelector('.status-text');

let _watchdog = null;

// ─── status + watchdog ───────────────────────────────────────────

function setStep(stateKey, label) {
  STATE.step = label;
  STATE.bootStart = STATE.bootStart || Date.now();
  if (STATUS_DOT) STATUS_DOT.dataset.state = stateKey;
  redrawStatus();
}

function redrawStatus() {
  if (!STATUS_TEXT) return;
  const elapsed = STATE.bootStart ? Math.round((Date.now() - STATE.bootStart) / 1000) : 0;
  STATUS_TEXT.textContent = `${STATE.step}  ·  ${elapsed}s`;
}

function startWatchdog() {
  if (_watchdog) return;
  _watchdog = setInterval(redrawStatus, 1000);
}

function stopWatchdog() {
  if (_watchdog) { clearInterval(_watchdog); _watchdog = null; }
}

function showError(err) {
  stopWatchdog();
  console.error('[ark]', err);
  const msg = err?.message ?? String(err);
  if (STATUS_DOT) STATUS_DOT.dataset.state = 'error';
  if (STATUS_TEXT) STATUS_TEXT.textContent = `Σφάλμα: ${msg}`;
  const tpl = document.getElementById('error-banner-template');
  if (tpl) {
    const node = tpl.content.firstElementChild.cloneNode(true);
    node.querySelector('.error-message').textContent = msg;
    document.querySelector('main').prepend(node);
  }
}

// ─── i18n ─────────────────────────────────────────────────────────

async function loadLang(lang) {
  const response = await fetch(`./i18n/${lang}.json`, { cache: 'no-cache' });
  if (!response.ok) throw new Error(`i18n ${lang}: HTTP ${response.status}`);
  STATE.translations = await response.json();
  STATE.lang = lang;
  document.documentElement.lang = lang;
  applyTranslations();
  updateLangButtons();
}

function applyTranslations() {
  document.querySelectorAll('[data-i18n]').forEach((el) => {
    const value = resolveKey(STATE.translations, el.getAttribute('data-i18n'));
    if (typeof value === 'string') el.textContent = value;
  });
  document.querySelectorAll('[data-i18n-attr]').forEach((el) => {
    const spec = el.getAttribute('data-i18n-attr') || '';
    const [attr, key] = spec.split('|');
    if (!attr || !key) return;
    const value = resolveKey(STATE.translations, key);
    if (typeof value === 'string') el.setAttribute(attr, value);
  });
}

function resolveKey(obj, dotted) {
  return dotted.split('.').reduce((acc, k) => (acc && acc[k] !== undefined ? acc[k] : null), obj);
}

function updateLangButtons() {
  document.querySelectorAll('.lang-switch [data-lang]').forEach((btn) => {
    btn.setAttribute('aria-pressed', String(btn.dataset.lang === STATE.lang));
  });
}

function wireLangSwitch() {
  document.querySelectorAll('.lang-switch [data-lang]').forEach((btn) => {
    btn.addEventListener('click', () => {
      const lang = btn.dataset.lang;
      if (lang && lang !== STATE.lang) loadLang(lang).catch(showError);
    });
  });
}

// ─── pyodide boot ─────────────────────────────────────────────────

async function bootPyodide() {
  if (typeof loadPyodide !== 'function') {
    throw new Error(`Pyodide loader δὲν φόρτωσε ἀπὸ τὸ CDN (v${PYODIDE_VERSION})`);
  }

  STATE.bootStart = Date.now();
  startWatchdog();

  setStep('boot', 'Φόρτωσι Pyodide…');
  console.log('[ark] step 1: loadPyodide');
  STATE.pyodide = await loadPyodide({
    indexURL: PYODIDE_INDEX,
    stdout: (msg) => console.log('[py]', msg),
    stderr: (msg) => console.warn('[py]', msg),
  });
  console.log('[ark] step 1 done');

  setStep('boot', 'Φόρτωσι numpy/scipy (~25 MB, μπορεῖ νὰ πάρει 30+s πρώτη φορά)…');
  console.log('[ark] step 2: loadPackage numpy/scipy');
  await STATE.pyodide.loadPackage(['numpy', 'scipy']);
  console.log('[ark] step 2 done');

  setStep('boot', 'Φόρτωσι Python modules…');
  console.log('[ark] step 3: fetch py sources');
  await loadPythonSources(STATE.pyodide);
  console.log('[ark] step 3 done');

  setStep('boot', 'Φόρτωσι DT + projectors…');
  console.log('[ark] step 4: fetch data');
  const [dt, projectors, stones] = await Promise.all([
    fetch('./data/dt.json').then(checkJson),
    fetch('./data/projectors.json').then(checkJson),
    fetch('./data/stones.json').then(checkJson).catch(() => ({ stones: [] })),
  ]);
  console.log('[ark] step 4 done');

  setStep('boot', 'Init pwa_bridge (sanity DT)…');
  console.log('[ark] step 5: init bridge');
  STATE.pyodide.globals.set('__dt_json', JSON.stringify(dt));
  STATE.pyodide.globals.set('__proj_json', JSON.stringify(projectors));
  // Ρητὴ ρύθμισι sys.path γιὰ νὰ βρεθοῦν τὰ modules στὸ /home/pyodide
  STATE.pyodide.runPython(`
import sys
if '${PY_HOME}' not in sys.path:
    sys.path.insert(0, '${PY_HOME}')
import pwa_bridge
__sanity = pwa_bridge.init_state(__dt_json, __proj_json)
`);
  const sanity = STATE.pyodide.globals.get('__sanity').toJs({ dict_converter: Object.fromEntries });
  console.log('[ark] step 5 done', sanity);
  if (!sanity || !sanity.ok || sanity.chi !== 2) {
    throw new Error(`Sanity ἀπέτυχε: ${JSON.stringify(sanity)}`);
  }

  STATE.stones = stones;
  STATE.dt = dt;
  STATE.ready = true;
  stopWatchdog();
  if (STATUS_DOT) STATUS_DOT.dataset.state = 'ready';
  if (STATUS_TEXT) {
    const ms = Date.now() - STATE.bootStart;
    STATUS_TEXT.textContent = `Ἕτοιμο  ·  V−E+F=2  ·  Σranks=30  ·  boot ${Math.round(ms / 100) / 10}s`;
  }

  window.ark = {
    pyodide: STATE.pyodide,
    dt,
    stones,
    analyze,
    trajectory,
  };
  console.log('[ark] dispatch ark:ready');
  window.dispatchEvent(new CustomEvent('ark:ready'));
}

async function loadPythonSources(pyodide) {
  try { pyodide.FS.mkdirTree(PY_HOME); } catch (_) { /* ἤδη ὑπάρχει */ }
  for (const name of PYTHON_SOURCES) {
    const r = await fetch(`./py/${name}`, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`fetch py/${name}: HTTP ${r.status}`);
    const src = await r.text();
    pyodide.FS.writeFile(`${PY_HOME}/${name}`, src);
  }
}

function checkJson(r) {
  if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`);
  return r.json();
}

// ─── Python bridge calls ──────────────────────────────────────────

function analyze(numbers, strategy = 'auto') {
  if (!STATE.ready) throw new Error('PWA not ready');
  STATE.pyodide.globals.set('__input_json', JSON.stringify({ numbers, strategy }));
  const result = STATE.pyodide.runPython(`
import pwa_bridge, json
json.dumps(pwa_bridge.analyze(__input_json))
`);
  return JSON.parse(result);
}

function trajectory(n_steps = 60) {
  if (!STATE.ready) throw new Error('PWA not ready');
  STATE.pyodide.globals.set('__n_steps', n_steps);
  const result = STATE.pyodide.runPython(`
import pwa_bridge, json
json.dumps(pwa_bridge.trajectory_demo(int(__n_steps)))
`);
  return JSON.parse(result);
}

// ─── service worker ───────────────────────────────────────────────

async function registerServiceWorker() {
  if (!('serviceWorker' in navigator)) return;
  try {
    const reg = await navigator.serviceWorker.register('./sw.js', { scope: './' });
    // Force update check σὲ κάθε boot ὥστε νὰ μὴ μένουμε σὲ stale SW
    reg.update().catch(() => {});
    console.log('[ark] sw scope=', reg.scope);
  } catch (err) {
    console.warn('[ark] sw failed', err);
  }
}

// ─── boot ─────────────────────────────────────────────────────────

async function boot() {
  try {
    wireLangSwitch();
    await loadLang(STATE.lang);
    registerServiceWorker();
    await bootPyodide();
  } catch (err) {
    showError(err);
  }
}

boot();
