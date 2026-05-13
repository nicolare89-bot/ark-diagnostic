"""
ark_stones.py — Stone class + StoneRegistry
============================================

Φόρτωσι, εὐρετηρίασι καὶ ἐρώτησι τοῦ corpus τῶν πετρῶν τοῦ project
(YAML ἀρχεῖα), μὲ 5 ἀπαράβατες ἀρχές:

  1. AUTO-DISCOVERY  — σαρώνει patterns· ποτὲ hardcoded ὀνόματα
  2. NEWER-WINS      — ἴδια ID σὲ πολλὰ YAML → νεότερη ὑπερισχύει
  3. SCHEMA VALIDATION — ἐλλιπεῖς πέτρες σημαίνονται 'incomplete'
                       χωρὶς νὰ σπάει τὸ φόρτωμα
  4. LAZY DETAILS    — ἀρχικὰ μόνο index (id/title/layer/stars/...)·
                       πλήρης φόρτωσι on-demand
  5. PLUGIN-STYLE    — νέες ID prefixes αὐτόματα δεκτές

Τὸ schema δέχεται καὶ τὶς δύο ὀνομασίες ποὺ συνυπάρχουν στὸ corpus:
  type    ≡ layer
  rating  ≡ stars
  relations ≡ related

Στάδιο 4, Φάσι 1.5.
V − E + F = 2.   ε = 1.5%.   J > 0.
Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml


# ═══════════════════════════════════════════════════════════════════
# Α. ΣΧΗΜΑ
# ═══════════════════════════════════════════════════════════════════

DEFAULT_PATTERNS = (
    'BOOTSTRAP_STONES_v*.yaml',
    'stone_registry_LIVE*.yaml',
    'stone_registry_LIVE*.yml',
    'STONES_APPEND*.yaml',
    'STONES_APPEND*.yml',
)

REQUIRED_FIELDS = ('id', 'statement', 'layer', 'stars')   # ἀπὸ CLAUDE.md schema

# Ὀνομασίες ποὺ θεωροῦνται ἰσοδύναμες (alias → canonical).
FIELD_ALIASES = {
    'type': 'layer',
    'rating': 'stars',
    'relations': 'related',
}


_STAR_NUM_RE = re.compile(r'(\d+)')


def _parse_stars(raw) -> int | None:
    """Δέχεται int ('5'), str ('6★', '6★ ΔΙΑΜ'), int 6 — γυρίζει int ἢ None."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return raw if 1 <= raw <= 6 else None
    if isinstance(raw, str):
        m = _STAR_NUM_RE.search(raw)
        if m:
            n = int(m.group(1))
            return n if 1 <= n <= 6 else None
    return None


def _chapter_from_id(id_str: str) -> str:
    """Π32-ARK.08 → 'Π32-ARK'. Π-28-Β07 → 'Π-28-Β07' (καμμία τελεία)."""
    if '.' in id_str:
        return id_str.rsplit('.', 1)[0]
    return id_str


# ═══════════════════════════════════════════════════════════════════
# Β. Stone DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Stone:
    id: str
    chapter: str = ''
    timestamp: str | None = None
    layer: str | None = None           # 🪨/🏖️/🌊
    stars: int | None = None
    rating_raw: str | None = None
    title: str | None = None
    statement: str | None = None
    verification: dict = field(default_factory=dict)
    related: list[str] = field(default_factory=list)
    sand: str | None = None
    water: str | None = None
    primary_gate: str | None = None
    extra: dict = field(default_factory=dict)
    source_file: str = ''
    is_complete: bool = False
    missing_fields: list[str] = field(default_factory=list)
    is_full: bool = False              # False ἂν εἶναι μόνο index entry

    @classmethod
    def from_yaml_dict(cls, raw: dict, source_file: str = '') -> 'Stone':
        """Φτιάχνει Stone ἀπὸ YAML dict, normalizing aliases."""
        if not isinstance(raw, dict):
            raise TypeError(f"Stone.from_yaml_dict: ἀπαιτεῖ dict, ἔλαβε {type(raw).__name__}")

        # Normalize aliases
        d = {}
        for k, v in raw.items():
            canonical = FIELD_ALIASES.get(k, k)
            d[canonical] = v

        sid = str(d.get('id', '')).strip()
        if not sid:
            raise ValueError(f"Stone χωρὶς id (source: {source_file})")

        # Συλλογὴ verification ἀπὸ τὰ διάφορα πεδία
        verification = {}
        for vfield in ('verification', 'proof_locus', 'evidence', 'discrimination_via'):
            if vfield in d:
                verification[vfield] = d[vfield]

        related = d.get('related') or []
        if isinstance(related, str):
            related = [related]

        rating_raw = d.get('stars')
        stars = _parse_stars(rating_raw)

        # Extra: ὅ,τι δὲν εἶναι κανονικὸ ἢ verification-related
        known = {
            'id', 'timestamp', 'layer', 'stars', 'title', 'statement',
            'verification', 'proof_locus', 'evidence', 'discrimination_via',
            'related', 'sand', 'water', 'primary_gate',
        }
        extra = {k: v for k, v in d.items() if k not in known}

        # Schema validation
        present = {
            'id': bool(sid),
            'statement': bool(d.get('statement')),
            'layer': bool(d.get('layer')),
            'stars': stars is not None,
        }
        missing = [f for f, ok in present.items() if not ok]

        return cls(
            id=sid,
            chapter=_chapter_from_id(sid),
            timestamp=d.get('timestamp'),
            layer=d.get('layer'),
            stars=stars,
            rating_raw=str(rating_raw) if rating_raw is not None else None,
            title=d.get('title'),
            statement=d.get('statement'),
            verification=verification,
            related=list(related),
            sand=d.get('sand'),
            water=d.get('water'),
            primary_gate=d.get('primary_gate'),
            extra=extra,
            source_file=source_file,
            is_complete=len(missing) == 0,
            missing_fields=missing,
            is_full=True,
        )

    def to_index_entry(self) -> 'Stone':
        """Φωτεινό index entry χωρὶς statement/sand/water/verification (lazy)."""
        return Stone(
            id=self.id,
            chapter=self.chapter,
            timestamp=self.timestamp,
            layer=self.layer,
            stars=self.stars,
            rating_raw=self.rating_raw,
            title=self.title,
            statement=None,
            verification={},
            related=[],
            sand=None,
            water=None,
            primary_gate=self.primary_gate,
            extra={},
            source_file=self.source_file,
            is_complete=self.is_complete,
            missing_fields=list(self.missing_fields),
            is_full=False,
        )


# ═══════════════════════════════════════════════════════════════════
# Γ. EXECUTABLE INVARIANTS (μικρό built-in registry)
# ═══════════════════════════════════════════════════════════════════

def _check_dt_euler():
    from ark_geometry import DT_V, DT_E, DT_F, DT_CHI
    return DT_V - DT_E + DT_F == DT_CHI == 2

def _check_beta_orbit_sum():
    from ark_geometry import BETA10, BETA6, BETA4, DT_V
    return BETA10 + BETA6 + BETA4 == DT_V == 62

def _check_ih_dim_squared_sum():
    return 1**2 + 3**2 + 3**2 + 4**2 + 5**2 == 60

def _check_cheeger_critical():
    from ark_geometry import CHEEGER_CRITICAL_EDGES, BETA10
    return CHEEGER_CRITICAL_EDGES == BETA10 == 12

def _check_epsilon_exact():
    import math
    from ark_geometry import EPSILON
    expected = 4 * (4 * math.sqrt(2) - 5) / 175
    return abs(EPSILON - expected) < 1e-15


BUILTIN_INVARIANTS: dict[str, Callable[[], bool]] = {
    'core:dt-euler':           _check_dt_euler,
    'core:beta-orbit-sum':     _check_beta_orbit_sum,
    'core:ih-dim-sum':         _check_ih_dim_squared_sum,
    'core:cheeger-critical':   _check_cheeger_critical,
    'core:epsilon-exact':      _check_epsilon_exact,
}


# ═══════════════════════════════════════════════════════════════════
# Δ. ΧΡΟΝΟΣΗΜΑΝΣΗ ΓΙΑ NEWER-WINS
# ═══════════════════════════════════════════════════════════════════

# Filename timestamp: π.χ. (23h46-11-05-2026) ἢ (03:00-11-05-2026)
_FILENAME_TS_RE = re.compile(
    r'\((\d{1,2})[h:](\d{2})-(\d{2})-(\d{2})-(\d{4})\)'
)

# Internal stone timestamp: π.χ. "2026-05-11 23:46 EEST"
_STONE_TS_RE = re.compile(
    r'(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})'
)


def _parse_filename_ts(path: Path) -> tuple[int, ...] | None:
    """Δείκτης χρονικῆς σειρᾶς ἀπὸ τὸ filename (Y, M, D, h, m)."""
    m = _FILENAME_TS_RE.search(path.name)
    if not m:
        return None
    hh, mm, dd, mo, yy = (int(x) for x in m.groups())
    return (yy, mo, dd, hh, mm)


def _parse_stone_ts(ts: str | None) -> tuple[int, ...] | None:
    """Δείκτης χρονικῆς σειρᾶς ἀπὸ τὸ stone.timestamp field."""
    if not ts:
        return None
    m = _STONE_TS_RE.search(ts)
    if not m:
        return None
    yy, mo, dd, hh, mm = (int(x) for x in m.groups())
    return (yy, mo, dd, hh, mm)


def _newer(a: Stone, b: Stone) -> Stone:
    """Ἐπιστρέφει τὸν νεότερο μεταξὺ a, b — προτιμᾷ stone.timestamp,
    fallback σὲ filename timestamp, fallback σὲ filename mtime."""
    ta, tb = _parse_stone_ts(a.timestamp), _parse_stone_ts(b.timestamp)
    if ta is not None and tb is not None:
        return a if ta >= tb else b
    if ta is not None:
        return a
    if tb is not None:
        return b
    pa = Path(a.source_file) if a.source_file else None
    pb = Path(b.source_file) if b.source_file else None
    fa = _parse_filename_ts(pa) if pa else None
    fb = _parse_filename_ts(pb) if pb else None
    if fa and fb:
        return a if fa >= fb else b
    return a   # ἀδιάφορο tiebreaker


# ═══════════════════════════════════════════════════════════════════
# Ε. StoneRegistry
# ═══════════════════════════════════════════════════════════════════

class StoneRegistry:
    """Lazy YAML-backed registry τῶν πετρῶν τοῦ project."""

    def __init__(self, project_dir: str | Path, patterns=None):
        self.project_dir = Path(project_dir)
        self.patterns = tuple(patterns) if patterns is not None else DEFAULT_PATTERNS
        self._index: dict[str, Stone] | None = None
        self._full_cache: dict[str, Stone] = {}
        self._raw_by_file: dict[Path, list[dict]] = {}

    # ────────────────────────────────────────────────────────────────
    # 1. AUTO-DISCOVERY
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def auto_discover(project_dir: str | Path, patterns=None) -> list[Path]:
        """Σαρώνει τὸ project_dir γιὰ τὰ YAML patterns. Ποτὲ hardcoded ὀνόματα."""
        root = Path(project_dir)
        pats = tuple(patterns) if patterns is not None else DEFAULT_PATTERNS
        found: set[Path] = set()
        for pat in pats:
            found.update(root.glob(pat))
        return sorted(found)

    # ────────────────────────────────────────────────────────────────
    # 2. Raw loading (helper)
    # ────────────────────────────────────────────────────────────────

    def _load_file_raw(self, path: Path) -> list[dict]:
        """Φορτώνει τὶς stones ἀπὸ ἕνα YAML ἀρχεῖο. Cached."""
        if path in self._raw_by_file:
            return self._raw_by_file[path]
        try:
            with path.open('r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            self._raw_by_file[path] = []
            return []
        if not isinstance(data, dict):
            self._raw_by_file[path] = []
            return []
        stones = data.get('stones', [])
        if not isinstance(stones, list):
            self._raw_by_file[path] = []
            return []
        self._raw_by_file[path] = stones
        return stones

    # ────────────────────────────────────────────────────────────────
    # 3. LAZY INDEX
    # ────────────────────────────────────────────────────────────────

    def load_index(self, refresh: bool = False) -> dict[str, Stone]:
        """Δείκτης id → Stone (μόνο μεταδεδομένα, χωρὶς statement κλπ).
        Ἐφαρμόζει NEWER-WINS σὲ διπλὲς IDs."""
        if self._index is not None and not refresh:
            return self._index
        index: dict[str, Stone] = {}
        for path in self.auto_discover(self.project_dir, self.patterns):
            for raw in self._load_file_raw(path):
                try:
                    stone = Stone.from_yaml_dict(raw, source_file=str(path))
                except (TypeError, ValueError):
                    continue
                existing = index.get(stone.id)
                winner = _newer(stone, existing) if existing else stone
                index[stone.id] = winner.to_index_entry()
                if winner is stone:
                    self._full_cache[stone.id] = stone
        self._index = index
        return index

    # ────────────────────────────────────────────────────────────────
    # 4. ON-DEMAND FULL LOAD
    # ────────────────────────────────────────────────────────────────

    def load_full(self, stone_id: str) -> Stone | None:
        """Πλήρης φόρτωσι μίας πέτρας. Cached."""
        if stone_id in self._full_cache and self._full_cache[stone_id].is_full:
            return self._full_cache[stone_id]
        # Σιγουρευόμαστε ὅτι ξέρουμε ποῦ ζεῖ
        self.load_index()
        idx_entry = self._index.get(stone_id) if self._index else None
        if idx_entry is None:
            return None
        # Ξαναδιαβάζουμε ἀπὸ τὸ source_file ποὺ κέρδισε
        path = Path(idx_entry.source_file)
        for raw in self._load_file_raw(path):
            if str(raw.get('id', '')).strip() == stone_id:
                try:
                    full = Stone.from_yaml_dict(raw, source_file=str(path))
                    self._full_cache[stone_id] = full
                    return full
                except (TypeError, ValueError):
                    return None
        return None

    # ────────────────────────────────────────────────────────────────
    # 5. ΕΡΩΤΗΣΕΙΣ
    # ────────────────────────────────────────────────────────────────

    def find_by_id(self, stone_id: str) -> Stone | None:
        idx = self.load_index()
        return idx.get(stone_id)

    def by_layer(self, layer: str) -> list[Stone]:
        idx = self.load_index()
        return [s for s in idx.values() if s.layer == layer]

    def by_topic(self, topic: str) -> list[Stone]:
        """Filter μέσῳ chapter prefix ἢ keyword στὸ id/title."""
        idx = self.load_index()
        topic_l = topic.lower()
        out = []
        for s in idx.values():
            if s.chapter.startswith(topic):
                out.append(s); continue
            if topic_l in s.id.lower():
                out.append(s); continue
            if s.title and topic_l in s.title.lower():
                out.append(s)
        return out

    def search(self, query: str) -> list[Stone]:
        """Keyword search σὲ title καὶ statement. Πλήρης φόρτωσι κατὰ τὴν χρήσι."""
        idx = self.load_index()
        q = query.lower()
        results = []
        for sid in idx:
            full = self.load_full(sid)
            if full is None:
                continue
            in_title = full.title and q in full.title.lower()
            in_stmt = full.statement and q in full.statement.lower()
            if in_title or in_stmt:
                results.append(full)
        return results

    def related_to(self, stone: Stone | str) -> list[Stone]:
        """Σχετικὲς πέτρες μέσῳ explicit `related` field."""
        if isinstance(stone, str):
            stone = self.load_full(stone)
            if stone is None:
                return []
        else:
            if not stone.is_full:
                full = self.load_full(stone.id)
                if full is not None:
                    stone = full
        out = []
        for rel in stone.related:
            # related entries μπορεῖ νὰ εἶναι IDs ἢ ἐλεύθερο κείμενο μὲ IDs μέσα
            if isinstance(rel, str):
                # ψάχνουμε γνωστὲς IDs μέσα στὸ κείμενο
                for sid in self.load_index():
                    if sid in rel:
                        candidate = self.find_by_id(sid)
                        if candidate is not None and candidate.id != stone.id:
                            out.append(candidate)
        # μοναδικότητα διατηρώντας σειρὰ
        seen, dedup = set(), []
        for s in out:
            if s.id not in seen:
                seen.add(s.id); dedup.append(s)
        return dedup

    # ────────────────────────────────────────────────────────────────
    # 6. EXECUTABLE INVARIANTS
    # ────────────────────────────────────────────────────────────────

    def executable_invariants(self) -> dict[str, Callable[[], bool]]:
        """Built-in registry τῶν ἐκτελέσιμων ἀναλλοίωτων."""
        return dict(BUILTIN_INVARIANTS)

    def verify_all(self) -> dict[str, bool]:
        """Τρέχει ὅλα τὰ executable invariants καὶ ἐπιστρέφει id → bool."""
        out = {}
        for key, fn in self.executable_invariants().items():
            try:
                out[key] = bool(fn())
            except Exception:
                out[key] = False
        return out

    # ────────────────────────────────────────────────────────────────
    # 7. BOOKKEEPING
    # ────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.load_index())

    def __contains__(self, stone_id: str) -> bool:
        return stone_id in self.load_index()

    def __iter__(self):
        return iter(self.load_index().values())

    def stats(self) -> dict:
        """Σύντομη σύνοψι: ἀριθμοί, ἀνὰ layer, complete vs incomplete."""
        idx = self.load_index()
        by_layer: dict[str, int] = {}
        complete = 0
        for s in idx.values():
            key = s.layer or '∅'
            by_layer[key] = by_layer.get(key, 0) + 1
            if s.is_complete:
                complete += 1
        return {
            'total': len(idx),
            'complete': complete,
            'incomplete': len(idx) - complete,
            'by_layer': by_layer,
            'source_files': sorted({s.source_file for s in idx.values()}),
        }
