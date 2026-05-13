"""
pwa_bridge.py — Pyodide-side ἐνδιάμεσο γιὰ τὴν Κιβωτὸ PWA
==========================================================

Τὸ module αὐτὸ φορτώνεται μέσα στὸ Pyodide καὶ ἐκθέτει ἕνα μικρὸ
σύνολο functions ποὺ καλεῖ τὸ JS:

  init_state(dt_json, projectors_json)
      Δέχεται τὰ precomputed JSON (παράγονται ἀπὸ build_pwa_data.py),
      ἀνασυστήνει τὸ DT graph (μὲ frozensets γιὰ edges/faces) καὶ
      τοὺς equivariant projectors σὲ ndarray(30,30). Μὲ αὐτὰ ἀποφεύγεται
      ἡ ἐξάρτησι ἀπὸ scipy.spatial μέσα στὸ browser.

  analyze(input_json)
      Δέχεται {numbers: [...], strategy: 'auto'|...}, τρέχει adapt +
      diagnose_irrep_distribution + diagnose_epsilon_drift, καὶ
      ἐπιστρέφει JSON-safe dict μὲ ὅλα ὅσα χρειάζονται οἱ visualizations
      (bars, gauge, DT heat).

  trajectory_demo(n_steps)
      Παράγει collapse trajectory v_0,...,v_n (cumulative Gaussian)
      καὶ ἐπιστρέφει χρονοσειρὰ τῶν ε(t) + verdicts.

Στάδιο 4, Φάσι 3 (PWA bridge). numpy-only — δὲν εἰσάγει scipy.
V − E + F = 2.   ε = 1.5%.   J > 0.
Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.
"""

from __future__ import annotations

import json

import numpy as np

from ark_diagnostics import (
    diagnose_irrep_distribution,
    diagnose_epsilon_drift,
)
from ark_adapter import adapt


# Singleton state ποὺ συμπληρώνεται ἀπὸ τὸ init_state(...) ἀπὸ τὸ JS side.
_STATE: dict = {
    'dt': None,            # DT graph (μὲ frozensets ξανὰ)
    'projectors': None,    # dict[name → ndarray(30,30)]
    'beta4_coords': None,  # ndarray(30,3) — γιὰ DT 3D heat
}


def _reconstruct_dt(dt_json: dict) -> dict:
    """Ἀνακατασκευή τοῦ DT graph ἀπὸ τὸ JSON: edges/faces ξανὰ σὲ frozensets."""
    coords = np.asarray(dt_json['coords'], dtype=float)
    edges = set(frozenset(e) for e in dt_json['edges'])
    faces = [frozenset(f) for f in dt_json['faces']]
    orbits = {k: list(v) for k, v in dt_json['orbits'].items()}
    degrees = {int(k): int(v) for k, v in dt_json['degrees'].items()}
    return {
        'name': dt_json.get('name', 'DT'),
        'coords': coords,
        'vertices': list(range(int(dt_json['V']))),
        'orbits': orbits,
        'edges': edges,
        'faces': faces,
        'degrees': degrees,
        'V': int(dt_json['V']),
        'E': int(dt_json['E']),
        'F': int(dt_json['F']),
        'chi': int(dt_json['chi']),
    }


def _reconstruct_projectors(proj_json: dict) -> dict:
    return {name: np.asarray(M, dtype=float) for name, M in proj_json['matrices'].items()}


def init_state(dt_json_str: str, proj_json_str: str) -> dict:
    """Καλεῖται μία φορά ἀπὸ τὸ JS μὲ τὰ precomputed JSON.

    Ἐπιστρέφει sanity report ποὺ τὸ JS μπορεῖ νὰ ἐπιθεωρήσει.
    """
    dt_json = json.loads(dt_json_str)
    proj_json = json.loads(proj_json_str)

    dt = _reconstruct_dt(dt_json)
    projectors = _reconstruct_projectors(proj_json)

    # Sanity: V − E + F = 2
    chi = dt['V'] - dt['E'] + dt['F']
    if chi != 2:
        raise ValueError(f'DT Euler ἀναλλοίωτο ἀπέτυχε: χ={chi}')

    # Sanity: Σranks = 30
    total_rank = sum(int(round(np.trace(P).real)) for P in projectors.values())
    if total_rank != 30:
        raise ValueError(f'Σranks = {total_rank} (expected 30)')

    # β₄ coords γιὰ τὸ DT 3D heat panel: τὰ ἑνιαία διανύσματα στὰ midpoints
    beta4_idx = dt['orbits']['β4']
    beta4_coords = dt['coords'][beta4_idx]

    _STATE['dt'] = dt
    _STATE['projectors'] = projectors
    _STATE['beta4_coords'] = beta4_coords

    return {
        'ok': True,
        'V': dt['V'], 'E': dt['E'], 'F': dt['F'], 'chi': chi,
        'irreps': list(projectors.keys()),
        'total_rank': total_rank,
    }


# ─── parsing ─────────────────────────────────────────────────────

def _parse_numbers(numbers) -> np.ndarray:
    """Δέχεται list/tuple ἀριθμῶν ἢ list[list] (γιὰ 2D εἴσοδο) → ndarray."""
    arr = np.asarray(numbers, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


# ─── analyze ─────────────────────────────────────────────────────

def analyze(input_json_str: str) -> dict:
    """Κύρια διαγνωστική ροή: adapt → irrep distribution → ε drift.

    Input JSON: {'numbers': [...], 'strategy': 'auto'|'direct'|'pad_to_30'|
                 'pca_to_30'|'sliding_window'|'spectral_embed'}
    """
    if _STATE['dt'] is None:
        raise RuntimeError('init_state() δὲν ἔχει κληθεῖ ἀκόμα')

    payload = json.loads(input_json_str)
    numbers = payload.get('numbers', [])
    strategy = payload.get('strategy', 'auto')

    if not numbers:
        raise ValueError('κενὴ εἴσοδος')

    v = _parse_numbers(numbers)
    adapted = adapt(v, strategy=strategy)
    output = np.asarray(adapted['output'], dtype=float)
    if output.ndim == 1:
        output = output.reshape(1, -1)

    projectors = _STATE['projectors']
    ih = diagnose_irrep_distribution(output, projectors)
    drift = diagnose_epsilon_drift(ih)

    # Heat-map per β₄ vertex: ‖P_irrep v‖² φορτίο σὲ κάθε midpoint
    # Ὑπολογίζεται ἀπὸ τὴν μέση ἐνέργεια ἀνὰ irrep, projected back σὲ 30 vertices
    # μέσῳ τῆς διαγωνίου τῶν projectors (rank-weighted).
    diag_load = np.zeros(30, dtype=float)
    for name, P in projectors.items():
        e = float(ih['energies'].get(name, 0.0))
        diag_load += e * np.diagonal(P)
    # Κανονικοποίησι γιὰ [0,1] heat-map
    if diag_load.max() > 0:
        heat = (diag_load / diag_load.max()).tolist()
    else:
        heat = diag_load.tolist()

    return {
        'input': {
            'shape_orig': list(np.asarray(v).shape),
            'strategy_used': adapted.get('strategy_used'),
            'n_samples': output.shape[0],
        },
        'energies': {k: float(val) for k, val in ih['energies'].items()},
        'energy_std': {k: float(val) for k, val in ih.get('energy_std', {}).items()},
        'expected_isotropic': {k: float(val) for k, val in ih.get('expected_isotropic', {}).items()},
        'deviations_pct': {k: float(val) for k, val in ih.get('deviations_pct', {}).items()},
        'max_deviation_pct': float(ih.get('max_deviation_pct', 0.0)),
        'verdict_distribution': ih.get('verdict', ''),
        'drift_fraction': float(drift.get('drift_fraction', 0.0)),
        'dominant_irrep': drift.get('dominant_irrep', ''),
        'dominant_energy': float(drift.get('dominant_energy', 0.0)),
        'within_living_imperfection': bool(drift.get('within_living_imperfection', False)),
        'predicted_cycles_to_collapse': float(drift.get('predicted_cycles_to_collapse', float('inf'))) if np.isfinite(drift.get('predicted_cycles_to_collapse', float('inf'))) else None,
        'verdict_drift': drift.get('verdict', ''),
        'heat_beta4': heat,  # μῆκος 30
    }


# ─── trajectory ─────────────────────────────────────────────────

def trajectory_demo(n_steps: int = 60, step_eps: float = 0.05, seed: int = 42) -> dict:
    """Cumulative-Gaussian trajectory πάνω στὶς β₄ DT coords.

    v_0 = ι_β4 (canonical), v_{t+1} = v_t + step_eps · N(0,1).
    Διαγνώσι σὲ κάθε t, ἐπιστρέφει χρονοσειρὰ ε(t) + dominant irrep.
    """
    if _STATE['dt'] is None:
        raise RuntimeError('init_state() δὲν ἔχει κληθεῖ ἀκόμα')

    dt = _STATE['dt']
    projectors = _STATE['projectors']

    beta4_idx = dt['orbits']['β4']
    beta4_coords = dt['coords'][beta4_idx]  # (30, 3)
    base = beta4_coords.T.copy()              # (3, 30)

    rng = np.random.default_rng(seed)
    v = base.copy()
    times = []
    drifts = []
    energies_T1 = []
    energies_H = []
    verdicts = []

    for t in range(n_steps + 1):
        ih = diagnose_irrep_distribution(v, projectors)
        drift = diagnose_epsilon_drift(ih)
        times.append(t)
        drifts.append(float(drift.get('drift_fraction', 0.0)))
        energies_T1.append(float(ih['energies'].get('T1', 0.0)))
        energies_H.append(float(ih['energies'].get('H', 0.0)))
        verdicts.append(drift.get('verdict', ''))
        if t < n_steps:
            v = v + step_eps * rng.standard_normal(v.shape)

    return {
        't': times,
        'drift_fraction': drifts,
        'energy_T1': energies_T1,
        'energy_H': energies_H,
        'verdicts': verdicts,
        'n_steps': n_steps,
        'step_eps': step_eps,
    }
