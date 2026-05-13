"""
test_ark_stones.py — Tests γιὰ τὸ stones registry
==================================================

Φάσι 1.5 τοῦ Σταδίου 4. Στόχος: 15+ πράσινα tests.
Κάλυψι: 5 ἀρχὲς (auto-discovery, newer-wins, schema validation,
lazy loading, plugin-style), search, related, executable invariants.
"""

import pytest
from pathlib import Path

from ark_stones import (
    Stone, StoneRegistry,
    _parse_stars, _chapter_from_id, _newer,
    _parse_filename_ts, _parse_stone_ts,
    DEFAULT_PATTERNS, REQUIRED_FIELDS, BUILTIN_INVARIANTS,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def stones_dir(tmp_path):
    """Δημιουργεῖ ἕνα tmp project dir μὲ stub YAML stones."""
    f1 = tmp_path / 'STONES_APPEND — Α (10h00-11-05-2026).yaml'
    f1.write_text("""
stones:
  - id: Π32-ARK.01
    timestamp: 2026-05-11 10:00 EEST
    type: 🪨
    rating: 6★
    title: Πρώτη πέτρα
    statement: |
      Δοκιμαστικὸ statement γιὰ Π32-ARK.01.
    relations:
      - σχέσι μὲ Π32-ARK.02

  - id: Π32-ARK.02
    timestamp: 2026-05-11 10:00 EEST
    type: 🏖️
    rating: 5★
    title: Δεύτερη πέτρα
    statement: Statement γιὰ Π32-ARK.02 μὲ φράσι «κιβωτός».

  - id: Π32-OLD.01
    timestamp: 2026-05-01 09:00 EEST
    type: 🪨
    rating: 6★
    title: Παλιὰ ἔκδοσι
    statement: Παλιὰ ἔκδοσι statement.
""", encoding='utf-8')

    f2 = tmp_path / 'STONES_APPEND — Β (12h00-11-05-2026).yaml'
    f2.write_text("""
stones:
  - id: Π32-OLD.01
    timestamp: 2026-05-11 12:00 EEST
    type: 🪨
    rating: 6★
    title: Νεότερη ἔκδοσι
    statement: Νεότερη ἔκδοσι ἴδιας πέτρας.

  - id: Π34-NEWAREA.01
    timestamp: 2026-05-11 12:00 EEST
    type: 🌊
    rating: 4★
    title: Πλήρως νέα κατηγορία
    statement: Δοκιμὴ plugin-style.

  - id: Π32-INC.01
    timestamp: 2026-05-11 12:00 EEST
    title: Ἐλλιπὴς πέτρα
    # Λείπουν: type/layer, rating/stars, statement
""", encoding='utf-8')

    # Ἀρχεῖο ποὺ ΔΕΝ ταιριάζει σὲ πάτερν
    other = tmp_path / 'something_else.yaml'
    other.write_text("stones:\n  - id: SHOULD_NOT_LOAD\n    title: x\n",
                     encoding='utf-8')

    return tmp_path


@pytest.fixture
def registry(stones_dir):
    return StoneRegistry(stones_dir)


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

class TestParseStars:
    @pytest.mark.parametrize("raw, expected", [
        ('6★', 6),
        ('6★ ΔΙΑΜ', 6),
        ('5', 5),
        (5, 5),
        (6, 6),
        ('', None),
        (None, None),
        ('xx', None),
        (7, None),       # ἐκτὸς εὔρους
        (0, None),
    ])
    def test_various(self, raw, expected):
        assert _parse_stars(raw) == expected


class TestChapterFromId:
    @pytest.mark.parametrize("sid, expected", [
        ('Π32-ARK.08', 'Π32-ARK'),
        ('Π34-NEWAREA.01', 'Π34-NEWAREA'),
        ('Π-28-Β07', 'Π-28-Β07'),    # καμμία τελεία → ὅλο τὸ id
        ('Ω.Π11.μν', 'Ω.Π11'),
    ])
    def test_extraction(self, sid, expected):
        assert _chapter_from_id(sid) == expected


# ═══════════════════════════════════════════════════════════════════
# Stone class
# ═══════════════════════════════════════════════════════════════════

class TestStoneClass:
    def test_alias_normalization(self):
        s = Stone.from_yaml_dict({
            'id': 'X.01', 'type': '🪨', 'rating': '6★',
            'statement': 'foo', 'relations': ['Y.02'],
        }, source_file='test.yaml')
        assert s.layer == '🪨'
        assert s.stars == 6
        assert s.related == ['Y.02']
        assert s.is_complete is True
        assert s.is_full is True

    def test_incomplete_marked(self):
        s = Stone.from_yaml_dict({
            'id': 'X.02', 'title': 'incomplete'
        }, source_file='test.yaml')
        assert s.is_complete is False
        # Λείπουν: statement, layer, stars
        for f in ('statement', 'layer', 'stars'):
            assert f in s.missing_fields
        # ID δὲν λείπει
        assert 'id' not in s.missing_fields

    def test_missing_id_raises(self):
        with pytest.raises(ValueError):
            Stone.from_yaml_dict({'title': 'no id here'}, source_file='x')

    def test_extra_fields_preserved(self):
        s = Stone.from_yaml_dict({
            'id': 'X.03', 'layer': '🪨', 'stars': 5, 'statement': 'x',
            'mysterious_new_field': 'σημαντικό',
        }, source_file='x')
        assert s.extra.get('mysterious_new_field') == 'σημαντικό'

    def test_to_index_entry_strips_heavy(self):
        s = Stone.from_yaml_dict({
            'id': 'X.04', 'layer': '🪨', 'stars': 6,
            'statement': 'big statement', 'sand': 's', 'water': 'w',
            'evidence': ['a', 'b'],
        }, source_file='x')
        idx = s.to_index_entry()
        assert idx.is_full is False
        assert idx.statement is None
        assert idx.sand is None
        assert idx.water is None
        assert idx.verification == {}
        # Light fields ἐπιβιώνουν
        assert idx.id == 'X.04'
        assert idx.layer == '🪨'
        assert idx.stars == 6


# ═══════════════════════════════════════════════════════════════════
# Auto-discovery
# ═══════════════════════════════════════════════════════════════════

class TestAutoDiscovery:
    def test_finds_stones_append(self, stones_dir):
        files = StoneRegistry.auto_discover(stones_dir)
        names = [p.name for p in files]
        assert any('STONES_APPEND' in n and 'Α' in n for n in names)
        assert any('STONES_APPEND' in n and 'Β' in n for n in names)

    def test_ignores_unmatched(self, stones_dir):
        files = StoneRegistry.auto_discover(stones_dir)
        names = [p.name for p in files]
        assert 'something_else.yaml' not in names

    def test_custom_patterns(self, stones_dir):
        files = StoneRegistry.auto_discover(stones_dir, patterns=['something_else.yaml'])
        assert len(files) == 1
        assert files[0].name == 'something_else.yaml'

    def test_default_patterns_constants(self):
        # Ἀσφάλεια ἀπὸ ἀνεπιθύμητες ἀλλαγές
        for needed in ('BOOTSTRAP_STONES', 'stone_registry_LIVE', 'STONES_APPEND'):
            assert any(needed in p for p in DEFAULT_PATTERNS)


# ═══════════════════════════════════════════════════════════════════
# Lazy loading
# ═══════════════════════════════════════════════════════════════════

class TestLazyLoading:
    def test_index_is_partial(self, registry):
        idx = registry.load_index()
        # Ὅλες οἱ stones ἐκτὸς τῆς "old" duplicate (newer-wins) πρέπει νὰ ὑπάρχουν
        assert 'Π32-ARK.01' in idx
        assert 'Π32-ARK.02' in idx
        assert 'Π32-OLD.01' in idx
        assert 'Π34-NEWAREA.01' in idx
        # Δὲν εἶναι full
        for sid, s in idx.items():
            assert s.is_full is False
            # Heavy fields εἶναι ἄδεια στὸ index
            assert s.statement is None or s.statement == ''

    def test_full_load_populates_statement(self, registry):
        full = registry.load_full('Π32-ARK.01')
        assert full is not None
        assert full.is_full is True
        assert full.statement is not None
        assert 'Π32-ARK.01' in full.statement

    def test_full_load_returns_none_for_missing(self, registry):
        assert registry.load_full('Π99-DOES-NOT-EXIST.01') is None

    def test_index_caches(self, registry):
        idx1 = registry.load_index()
        idx2 = registry.load_index()
        assert idx1 is idx2


# ═══════════════════════════════════════════════════════════════════
# Newer-wins
# ═══════════════════════════════════════════════════════════════════

class TestNewerWins:
    def test_newer_stone_wins(self, registry):
        full = registry.load_full('Π32-OLD.01')
        assert full is not None
        assert full.title == 'Νεότερη ἔκδοσι'
        assert 'Νεότερη' in full.statement

    def test_newer_helper_with_timestamps(self):
        a = Stone(id='X', timestamp='2026-05-01 10:00 EEST', source_file='a.yaml')
        b = Stone(id='X', timestamp='2026-05-11 10:00 EEST', source_file='b.yaml')
        assert _newer(a, b).source_file == 'b.yaml'
        assert _newer(b, a).source_file == 'b.yaml'

    def test_filename_ts_parse(self):
        p1 = Path('STONES_APPEND — Α (23h46-11-05-2026).yaml')
        p2 = Path('STONES_APPEND — Β (03h00-12-05-2026).yaml')
        t1 = _parse_filename_ts(p1)
        t2 = _parse_filename_ts(p2)
        assert t1 is not None and t2 is not None
        assert t2 > t1   # 12 Μαΐου > 11 Μαΐου

    def test_stone_ts_parse(self):
        assert _parse_stone_ts('2026-05-11 23:46 EEST') == (2026, 5, 11, 23, 46)
        assert _parse_stone_ts('') is None
        assert _parse_stone_ts(None) is None


# ═══════════════════════════════════════════════════════════════════
# Schema validation
# ═══════════════════════════════════════════════════════════════════

class TestSchemaValidation:
    def test_incomplete_loaded_not_skipped(self, registry):
        # Π32-INC.01 ἔχει μόνο id + title — δὲν πρέπει νὰ σπάει τὴ φόρτωσι
        idx = registry.load_index()
        assert 'Π32-INC.01' in idx
        assert idx['Π32-INC.01'].is_complete is False
        assert len(idx['Π32-INC.01'].missing_fields) >= 2

    def test_complete_marked_correctly(self, registry):
        full = registry.load_full('Π32-ARK.01')
        assert full.is_complete is True
        assert full.missing_fields == []

    def test_required_fields_constant(self):
        # Δίκλυδο — ἂν ἀλλάξει τὸ schema, ἔρχεται μέσα ἀπὸ ξεκάθαρη μεταβολή
        assert set(REQUIRED_FIELDS) == {'id', 'statement', 'layer', 'stars'}


# ═══════════════════════════════════════════════════════════════════
# Plugin-style
# ═══════════════════════════════════════════════════════════════════

class TestPluginStyle:
    def test_unknown_prefix_loaded(self, registry):
        idx = registry.load_index()
        assert 'Π34-NEWAREA.01' in idx
        assert idx['Π34-NEWAREA.01'].chapter == 'Π34-NEWAREA'

    def test_extra_field_preserved_through_pipeline(self, tmp_path):
        f = tmp_path / 'STONES_APPEND — Test (10h00-11-05-2026).yaml'
        f.write_text("""
stones:
  - id: Π99-FUTURE.01
    type: 🪨
    rating: 6★
    statement: future
    novel_field_2030: ἀπρόβλεπτο
""", encoding='utf-8')
        reg = StoneRegistry(tmp_path)
        full = reg.load_full('Π99-FUTURE.01')
        assert full is not None
        assert full.extra.get('novel_field_2030') == 'ἀπρόβλεπτο'


# ═══════════════════════════════════════════════════════════════════
# Queries
# ═══════════════════════════════════════════════════════════════════

class TestQueries:
    def test_find_by_id(self, registry):
        s = registry.find_by_id('Π32-ARK.01')
        assert s is not None
        assert s.title == 'Πρώτη πέτρα'

    def test_find_by_id_missing(self, registry):
        assert registry.find_by_id('NOPE') is None

    def test_by_layer_filter(self, registry):
        rocks = registry.by_layer('🪨')
        rock_ids = {s.id for s in rocks}
        assert 'Π32-ARK.01' in rock_ids
        assert 'Π32-OLD.01' in rock_ids
        assert 'Π32-ARK.02' not in rock_ids   # αὐτὴ εἶναι 🏖️

    def test_by_topic_chapter(self, registry):
        ark = registry.by_topic('Π32-ARK')
        ids = {s.id for s in ark}
        assert 'Π32-ARK.01' in ids
        assert 'Π32-ARK.02' in ids
        assert 'Π32-OLD.01' not in ids

    def test_search_in_statement(self, registry):
        results = registry.search('κιβωτός')
        ids = {s.id for s in results}
        assert 'Π32-ARK.02' in ids

    def test_search_in_title(self, registry):
        results = registry.search('Νεότερη')
        ids = {s.id for s in results}
        assert 'Π32-OLD.01' in ids


# ═══════════════════════════════════════════════════════════════════
# Related
# ═══════════════════════════════════════════════════════════════════

class TestRelatedTo:
    def test_explicit_id_in_related(self, registry):
        rels = registry.related_to('Π32-ARK.01')
        ids = {s.id for s in rels}
        assert 'Π32-ARK.02' in ids

    def test_no_self_loop(self, registry):
        rels = registry.related_to('Π32-ARK.01')
        assert all(s.id != 'Π32-ARK.01' for s in rels)


# ═══════════════════════════════════════════════════════════════════
# Executable invariants
# ═══════════════════════════════════════════════════════════════════

class TestExecutableInvariants:
    def test_registry_returns_dict(self, registry):
        inv = registry.executable_invariants()
        assert isinstance(inv, dict)
        assert len(inv) >= 5
        # Καλλιεργοῦμε αἴσθησι ὅτι ἀφορᾷ ἀναλλοίωτα τῆς Κιβωτοῦ
        keys = list(inv.keys())
        assert any('dt-euler' in k for k in keys)
        assert any('epsilon' in k for k in keys)

    def test_verify_all_passes(self, registry):
        results = registry.verify_all()
        assert all(results.values()), f"Failures: {results}"

    def test_builtin_invariants_are_callable(self):
        for k, fn in BUILTIN_INVARIANTS.items():
            assert callable(fn)


# ═══════════════════════════════════════════════════════════════════
# Bookkeeping
# ═══════════════════════════════════════════════════════════════════

class TestBookkeeping:
    def test_len_and_contains(self, registry):
        # Π32-ARK.01, Π32-ARK.02, Π32-OLD.01, Π34-NEWAREA.01, Π32-INC.01
        assert len(registry) == 5
        assert 'Π32-ARK.01' in registry
        assert 'NOPE' not in registry

    def test_iter(self, registry):
        ids = {s.id for s in registry}
        assert 'Π32-ARK.01' in ids

    def test_stats(self, registry):
        s = registry.stats()
        assert s['total'] == 5
        assert s['complete'] + s['incomplete'] == s['total']
        assert s['incomplete'] >= 1   # ἡ Π32-INC.01


# ═══════════════════════════════════════════════════════════════════
# Integration μὲ πραγματικὰ project YAML (smoke test)
# ═══════════════════════════════════════════════════════════════════

class TestProjectIntegration:
    """Φορτώνει τὰ πραγματικὰ stones τοῦ project — smoke test ὅτι
    τὸ schema τοῦ corpus ταιριάζει μὲ τὶς προσδοκίες μας."""

    def test_loads_real_stones(self):
        reg = StoneRegistry(Path(__file__).parent)
        idx = reg.load_index()
        # Πρέπει νὰ βρεθοῦν τοὐλάχιστον κάποιες
        assert len(idx) > 0
        # Καθεμία ἔχει chapter παραγωμένο
        for s in idx.values():
            assert s.chapter
