// sw.js — service worker γιὰ τὴν Κιβωτὸ PWA (Φάσι 4)
//
// Δύο cache πολιτικὲς ἀνὰ origin:
//
//   1. SHELL (same-origin: index.html, main.js, ui_panels.js, styles.css,
//      i18n, data/, py/, assets/) → network-first μὲ fallback στὸ cache.
//      Ἂν ἀνοίξεις offline, σηκώνεται κανονικά.
//
//   2. CDN (cross-origin: jsdelivr Pyodide bundle + Plotly) →
//      cache-first. Στὴν πρώτη σύνδεσι κάνει download (~25 MB scipy +
//      νὰ συνυπολογιστεῖ Plotly ~4 MB)· μετὰ ζεῖ ὁλόκληρο στὸν δίσκο.
//
// V − E + F = 2.

const CACHE_VERSION = 'v7';
const SHELL_CACHE = `ark-shell-${CACHE_VERSION}`;
const CDN_CACHE   = `ark-cdn-${CACHE_VERSION}`;

const SHELL_ASSETS = [
  './',
  './index.html',
  './main.js',
  './ui_panels.js',
  './styles.css',
  './manifest.webmanifest',
  './i18n/el.json',
  './i18n/en.json',
  './assets/icon.svg',
  './data/dt.json',
  './data/projectors.json',
  './data/stones.json',
  './data/wu_sectors.json',
  './data/hashimoto.json',
  './py/ark_geometry.py',
  './py/ark_irreps.py',
  './py/ark_diagnostics.py',
  './py/ark_adapter.py',
  './py/ark_local_b10.py',
  './py/pwa_bridge.py',
];

// Origins ποὺ θεωροῦνται «μεγάλα ἀκίνητα bundles» (cache-first).
const CDN_ORIGINS = [
  'https://cdn.jsdelivr.net',
  'https://cdn.plot.ly',
];

// ─── install ────────────────────────────────────────────────────

self.addEventListener('install', (event) => {
  event.waitUntil((async () => {
    const cache = await caches.open(SHELL_CACHE);
    // Προσπάθεια precache· ἂν κάποιο shell asset δὲν ὑπάρχει ἀκόμα
    // (π.χ. πρώτη ἐγκατάστασι), προχωρᾶμε.
    await Promise.allSettled(SHELL_ASSETS.map((url) => cache.add(url)));
    await self.skipWaiting();
  })());
});

// ─── activate ──────────────────────────────────────────────────

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    const keep = new Set([SHELL_CACHE, CDN_CACHE]);
    await Promise.all(keys.filter((k) => !keep.has(k)).map((k) => caches.delete(k)));
    await self.clients.claim();
  })());
});

// ─── fetch ────────────────────────────────────────────────────

function isCdnRequest(url) {
  return CDN_ORIGINS.some((o) => url.origin === o);
}

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  try {
    const response = await fetch(request);
    if (response.ok || response.type === 'opaque') {
      // Cross-origin χωρὶς CORS → opaque response. Τὸ ἀποθηκεύουμε ὅπως εἶναι.
      cache.put(request, response.clone()).catch(() => {});
    }
    return response;
  } catch (err) {
    // Offline + δὲν ὑπάρχει cached: ἀφήνουμε τὸν browser νὰ δείξει network error.
    throw err;
  }
}

async function networkFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  try {
    const response = await fetch(request);
    if (response.ok) cache.put(request, response.clone()).catch(() => {});
    return response;
  } catch (err) {
    const cached = await cache.match(request);
    if (cached) return cached;
    throw err;
  }
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const url = new URL(req.url);

  if (isCdnRequest(url)) {
    event.respondWith(cacheFirst(req, CDN_CACHE));
    return;
  }

  if (url.origin === self.location.origin) {
    event.respondWith(networkFirst(req, SHELL_CACHE));
    return;
  }

  // Ἄγνωστο origin: ἀφήνουμε browser default.
});

// ─── messaging (πρόοδος καθαρισμοῦ cache ἀπὸ τὴν UI) ──────────

self.addEventListener('message', (event) => {
  if (event.data?.type === 'CLEAR_CACHE') {
    event.waitUntil((async () => {
      await caches.delete(SHELL_CACHE);
      await caches.delete(CDN_CACHE);
      event.source?.postMessage({ type: 'CACHE_CLEARED' });
    })());
  }
});
