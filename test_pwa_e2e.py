"""
test_pwa_e2e.py — Playwright e2e tests γιὰ τὴν Κιβωτὸ PWA (Φάσι 4.3)
====================================================================

Τρέχει τοπικὰ ἕναν http.server στὸ ark_web/, ἀνοίγει Chromium, καὶ
ἐλέγχει ὅτι ἡ PWA φορτώνει σωστὰ καὶ ὅτι τὰ panels ἀντιδροῦν στὸ Run.

Προϋποθέσεις (μία φορά):
    uv sync
    uv run playwright install chromium

Τρέχει:
    uv run pytest test_pwa_e2e.py -v

Σημείωσι: τὸ scipy bundle στὸ Pyodide εἶναι ~25 MB. Στὴν πρώτη
ἐκτέλεσι τὸ test μπορεῖ νὰ πάρει 60+ δευτερόλεπτα νὰ φορτώσει.

V − E + F = 2.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
ARK_WEB = HERE / 'ark_web'

# Generous timeout γιὰ τὸ scipy download τὴν πρώτη φορά.
PYODIDE_BOOT_TIMEOUT_MS = 180_000  # 3 λεπτά


def _free_port() -> int:
    """Βρίσκει ἕνα ἐλεύθερο port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture(scope='module')
def server_url():
    """Σηκώνει http.server στὸ ark_web/ καὶ τὸν σταματάει στὸ τέλος."""
    if not ARK_WEB.exists():
        pytest.skip(f'{ARK_WEB} δὲν ὑπάρχει — τρέξε build_pwa_data.py πρῶτα')

    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, '-m', 'http.server', str(port), '--bind', '127.0.0.1', '-d', str(ARK_WEB)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Περιμένουμε νὰ σηκωθεῖ
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with socket.socket() as s:
            try:
                s.connect(('127.0.0.1', port))
                break
            except OSError:
                time.sleep(0.1)
    yield f'http://127.0.0.1:{port}'
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope='module')
def browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        pytest.skip('playwright δὲν ἔχει ἐγκατασταθεῖ — uv sync καὶ uv run playwright install chromium')

    with sync_playwright() as p:
        try:
            b = p.chromium.launch(headless=True)
        except Exception as exc:
            pytest.skip(f'Chromium δὲν ξεκίνησε: {exc} (uv run playwright install chromium)')
        yield b
        b.close()


@pytest.fixture
def page(browser, server_url):
    ctx = browser.new_context()
    page = ctx.new_page()
    page.goto(server_url, wait_until='domcontentloaded')
    yield page
    ctx.close()


def _wait_ready(page) -> str:
    """Περιμένει τὸ status banner νὰ ἀνέβει σὲ data-state='ready'."""
    page.wait_for_selector(
        '#status .dot[data-state="ready"]',
        timeout=PYODIDE_BOOT_TIMEOUT_MS,
    )
    return page.text_content('#status .status-text') or ''


# ═══════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_shell_loads(page):
    """Ὁ τίτλος καὶ τὰ panels ὑπάρχουν στὸ DOM."""
    assert page.title()
    assert page.is_visible('#input-panel')
    assert page.is_visible('.lang-switch')


@pytest.mark.slow
def test_pyodide_boots(page):
    """Pyodide φορτώνει· τὸ status banner γίνεται 'Ἕτοιμο'."""
    text = _wait_ready(page)
    assert 'V−E+F=2' in text, f'sanity δὲν φάνηκε στὸ status: {text}'


@pytest.mark.slow
def test_wu_panel_renders(page):
    """Tὸ Wu structural panel φαίνεται μετὰ τὸ ready (static, χωρὶς run)."""
    _wait_ready(page)
    # b4 value πρέπει νὰ ἔχει γεμίσει (ὄχι '—')
    b4 = page.text_content('#wu-b4-value') or ''
    assert b4 and b4 != '—', f'Wu b4 value κενό: {b4!r}'
    # Tot^n bar plot πρέπει νὰ ἔχει render-αριστεῖ
    page.wait_for_selector('#plot-wu-tot .plotly', timeout=10_000)
    page.wait_for_selector('#plot-wu-bideg .plotly', timeout=10_000)


@pytest.mark.slow
def test_run_produces_diagnostics(page):
    """Μετὰ ἀπὸ Run μὲ 30 ἀριθμούς, ἐμφανίζονται bars + gauge + DT."""
    _wait_ready(page)
    numbers = ','.join(str(i / 30.0) for i in range(30))
    page.fill('#input-numbers', numbers)
    page.select_option('#input-strategy', 'direct')

    # Run δὲν εἶναι disabled
    page.wait_for_function('!document.getElementById("btn-run").disabled')
    page.click('#btn-run')

    page.wait_for_selector('#plot-bars .plotly', timeout=10_000)
    page.wait_for_selector('#plot-gauge .plotly', timeout=10_000)
    page.wait_for_selector('#dt-3d .plotly', timeout=10_000)
    page.wait_for_selector('#plot-beta10 .plotly', timeout=10_000)

    verdict = page.text_content('#verdict-text') or ''
    assert verdict, 'verdict text κενό'

    # β₁₀ panel summary πρέπει νὰ ἔχει πραγματικὲς τιμές, ὄχι '—'
    dom = page.text_content('#beta10-dominant-value') or ''
    ani = page.text_content('#beta10-anisotropy-value') or ''
    assert dom and dom != '—', f'β₁₀ dominant value κενό: {dom!r}'
    assert ani and ani != '—', f'β₁₀ anisotropy value κενό: {ani!r}'


@pytest.mark.slow
def test_trajectory_renders(page):
    """Trajectory κουμπὶ προσθέτει τὸ time-series plot."""
    _wait_ready(page)
    page.wait_for_function('!document.getElementById("btn-trajectory").disabled')
    page.click('#btn-trajectory')
    page.wait_for_selector('#plot-trajectory .plotly', timeout=15_000)


@pytest.mark.slow
def test_lang_switch(page):
    """Ἀλλαγὴ ΕΛ → EN ἐνημερώνει τὸ tagline."""
    _wait_ready(page)
    page.click('.lang-switch [data-lang="en"]')
    page.wait_for_function(
        'document.documentElement.lang === "en"',
        timeout=5_000,
    )
    title = page.text_content('h1') or ''
    assert 'Ark' in title or 'ARK' in title.upper()
