"""
ark_state.py — Save/load state σὲ JSON
========================================

Reproducibility γιὰ PAPER_UNIFIED_v3 καὶ cache γιὰ ἀκριβοὺς equivariant
projectors (ποὺ ὑπολογίζονται μέσῳ O(12³) brute-force).

Συναρτήσεις:
  save_dt(dt, path) / load_dt(path)               — γεωμετρία
  save_projectors(P, path) / load_projectors(path) — 30×30 πίνακες
  save_report(report, path) / load_report(path)   — διαγνωστικὰ ἀποτελέσματα

JSON-safe ἑρμηνεία: numpy → list/float/int, set/frozenset → sorted list,
float('inf') → 'inf'.

Στάδιο 2.7 (#3 ἀπὸ τοὺς στόχους τοῦ Σταδίου 2 στὸ CLAUDE.md).
"""

import json
import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Α. Recursive JSON serialization helper
# ═══════════════════════════════════════════════════════════════════

def _to_json(obj):
    """Recursive μετατροπὴ σὲ JSON-safe primitives.

    Χειρίζεται: dict, list/tuple, numpy scalar/array, set/frozenset,
    float('inf'), ἄλλο pass-through.
    """
    if isinstance(obj, dict):
        return {str(k): _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_json(v) for v in obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        if v == float('inf'):
            return 'inf'
        if v == float('-inf'):
            return '-inf'
        return v
    if isinstance(obj, np.ndarray):
        return _to_json(obj.tolist())
    if isinstance(obj, float):
        if obj == float('inf'):
            return 'inf'
        if obj == float('-inf'):
            return '-inf'
        return obj
    return obj


# ═══════════════════════════════════════════════════════════════════
# Β. DT geometry save/load
# ═══════════════════════════════════════════════════════════════════

def save_dt(dt_graph, path):
    """Ἀποθήκευσι DT (ἢ ὁποιουδήποτε graph dict) σὲ JSON.

    Διατηρεῖ: name, V, E, F, chi, coords, vertices, orbits, edges,
              faces, degrees, types (γιὰ 24-cell), cells.
    """
    payload = _to_json(dt_graph)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_dt(path):
    """Φόρτωσι DT/graph ἀπὸ JSON. Ἐπιστρέφει dict στὴν ἴδια μορφὴ μὲ
    τὸ build_DT() — ἀλλὰ coords ὡς np.ndarray, edges/faces ὡς set
    ἀπὸ frozenset.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Reconstruct numpy arrays καὶ frozensets
    if 'coords' in data:
        data['coords'] = np.array(data['coords'])
    if 'edges' in data:
        data['edges'] = {frozenset(e) for e in data['edges']}
    if 'faces' in data:
        data['faces'] = [frozenset(f) for f in data['faces']]
    # degrees keys μπορεῖ νὰ ἔχουν γίνει strings ἀπὸ JSON — γύρισέ τα σὲ int
    if 'degrees' in data:
        data['degrees'] = {int(k): v for k, v in data['degrees'].items()}
    return data


# ═══════════════════════════════════════════════════════════════════
# Γ. Projectors save/load
# ═══════════════════════════════════════════════════════════════════

def save_projectors(projectors, path):
    """Ἀποθήκευσι dict ἀπὸ 30×30 numpy arrays σὲ JSON.

    Format: {"A": [[...30...], ...30 rows...], "T1": [...], ...}
    """
    payload = {name: P.tolist() for name, P in projectors.items()}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_projectors(path):
    """Φόρτωσι projectors ἀπὸ JSON ὡς dict[name → np.ndarray]."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {name: np.array(P) for name, P in data.items()}


# ═══════════════════════════════════════════════════════════════════
# Δ. Diagnostic report save/load
# ═══════════════════════════════════════════════════════════════════

def save_report(report, path):
    """Ἀποθήκευσι διαγνωστικοῦ report (ἢ list reports) σὲ JSON.

    Χειρίζεται φωλιασμένα dicts, lists, numpy scalars, ∞.
    """
    payload = _to_json(report)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_report(path):
    """Φόρτωσι report ἀπὸ JSON. 'inf' strings ξανὰ σὲ float('inf')."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return _restore_inf(data)


def _restore_inf(obj):
    """Recursive ἀντικατάστασι 'inf'/'-inf' strings μὲ float('inf')."""
    if isinstance(obj, dict):
        return {k: _restore_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_inf(v) for v in obj]
    if obj == 'inf':
        return float('inf')
    if obj == '-inf':
        return float('-inf')
    return obj
