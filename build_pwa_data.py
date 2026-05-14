"""
build_pwa_data.py — Precompute DT + Iₕ-equivariant projectors + stones index
============================================================================

Παράγει JSON ἀρχεῖα στὸ `ark_web/data/` ποὺ τὸ Pyodide φορτώνει στὴν PWA.
Τρέχει τοπικὰ (χρειάζεται scipy γιὰ τὸ DT construction καὶ τοὺς projectors).

Ἔξοδος:
  ark_web/data/dt.json          — coords (62,3), orbits, edges, faces
  ark_web/data/projectors.json  — equivariant projectors {A, T1, T2, G, H} σὲ (30,30)
  ark_web/data/stones.json      — index (id, title, layer, stars, statement, related, source)

Στάδιο 4, Φάσι 3 (precompute pipeline).
V − E + F = 2.   ε = 1.5%.   J > 0.
Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Windows console: force UTF-8 ὥστε νὰ τυπώνονται ε/χ/→
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from ark_geometry import build_DT, EPSILON, EPSILON_NOMINAL, PHI, BETA10, BETA6, BETA4
from ark_irreps import build_irrep_projectors_30_equivariant, RANKS_BETA4, IH_DIMS
from ark_stones import StoneRegistry
from ark_wu import build_PE, wu_bicomplex, wu_sector_decomposition


HERE = Path(__file__).resolve().parent
OUT = HERE / 'ark_web' / 'data'
PY_OUT = HERE / 'ark_web' / 'py'

# Modules ποὺ ἀντιγράφονται στὸ ark_web/py/ (φορτώνονται στὸ Pyodide FS).
# Πρέπει νὰ συμφωνοῦν μὲ τὸ PYTHON_SOURCES στὸ main.js.
PYTHON_MODULES = (
    'ark_geometry.py',
    'ark_irreps.py',
    'ark_diagnostics.py',
    'ark_adapter.py',
    'ark_local_b10.py',
    'pwa_bridge.py',
)


def dt_to_json(dt: dict) -> dict:
    """Μετατροπὴ τοῦ DT graph σὲ JSON-serializable δομή."""
    edges = sorted(sorted(e) for e in dt['edges'])
    faces = sorted(sorted(f) for f in dt['faces'])
    return {
        'name': 'DT',
        'V': dt['V'], 'E': dt['E'], 'F': dt['F'], 'chi': dt['chi'],
        'coords': dt['coords'].tolist(),
        'orbits': {k: list(v) for k, v in dt['orbits'].items()},
        'edges': edges,
        'faces': faces,
        'degrees': dict(dt['degrees']),
    }


def projectors_to_json(projectors: dict) -> dict:
    return {
        'irreps': list(projectors.keys()),
        'ranks': RANKS_BETA4,
        'dims': IH_DIMS,
        'matrices': {name: P.tolist() for name, P in projectors.items()},
    }


def stones_to_json(registry: StoneRegistry) -> dict:
    out = []
    for stone in registry:
        out.append({
            'id': stone.id,
            'chapter': stone.chapter,
            'layer': stone.layer,
            'stars': stone.stars,
            'title': stone.title,
            'statement': stone.statement,
            'related': list(stone.related),
            'source': str(Path(stone.source_file).name) if stone.source_file else None,
            'complete': stone.is_complete,
        })
    return {
        'count': len(out),
        'stats': registry.stats(),
        'stones': out,
    }


def copy_python_sources() -> None:
    PY_OUT.mkdir(parents=True, exist_ok=True)
    for name in PYTHON_MODULES:
        src = HERE / name
        if not src.exists():
            print(f'  ⚠ {name} δὲν ὑπάρχει στὸ {HERE}')
            continue
        (PY_OUT / name).write_text(src.read_text(encoding='utf-8'), encoding='utf-8')
        print(f'  → py/{name}')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print('Building DT…')
    dt = build_DT()
    assert dt['V'] - dt['E'] + dt['F'] == 2, 'Euler ἀναλλοίωτο ἀπέτυχε'
    (OUT / 'dt.json').write_text(json.dumps(dt_to_json(dt)), encoding='utf-8')
    print(f'  → dt.json: V={dt["V"]} E={dt["E"]} F={dt["F"]} χ=2')

    print('Building equivariant projectors…')
    projectors = build_irrep_projectors_30_equivariant(dt)
    for name, P in projectors.items():
        rank = int(round(np.trace(P).real))
        idem = np.allclose(P @ P, P, atol=1e-8)
        assert idem, f'projector {name} δὲν εἶναι idempotent'
        assert rank == RANKS_BETA4[name], f'rank mismatch {name}: {rank} vs {RANKS_BETA4[name]}'
        print(f'  {name}: rank={rank} ✓')
    (OUT / 'projectors.json').write_text(json.dumps(projectors_to_json(projectors)), encoding='utf-8')
    total_rank = sum(RANKS_BETA4.values())
    print(f'  → projectors.json: Σranks={total_rank} (expected 30)')

    print('Building Wu sectors (structural)…')
    try:
        pe = build_PE(dt)
        bicx = wu_bicomplex(dt, pe)
        sectors, sector_indices = wu_sector_decomposition(bicx, dt)
        # sector_indices: dict[β₁₀ vertex → list of Tot^4 basis indices]
        counts = sorted(((int(v), len(idxs)) for v, idxs in sector_indices.items()), key=lambda x: x[0])
        # Tot^n dimensions γιὰ n=0..4 (filtration)
        tot_dims = [int(len(bicx['basis'][n])) for n in range(5)]
        # Bidegree breakdown: |C^(p,q)| γιὰ p,q ∈ {0,1,2}
        bidegree = {}
        for n in range(5):
            for sigma, tau, p, q in bicx['basis'][n]:
                bidegree[(p, q)] = bidegree.get((p, q), 0) + 1
        bideg_grid = [[int(bidegree.get((p, q), 0)) for q in range(3)] for p in range(3)]
        wu_data = {
            'b4': 12,                                # ἁρμονικὴ διάστασι (Π32-WU-PROBE.01)
            'n_total_tot4': int(len(sectors)),
            'n_sectors': int(len(sector_indices)),
            'sector_vertex_ids': [v for v, _ in counts],
            'sector_counts':     [c for _, c in counts],
            'tot_dims':          tot_dims,           # [|Tot^0|, …, |Tot^4|]
            'bidegree':          bideg_grid,         # 3×3, bideg[p][q] = |C^(p,q)|
        }
        (OUT / 'wu_sectors.json').write_text(json.dumps(wu_data, ensure_ascii=False), encoding='utf-8')
        print(f'  → wu_sectors.json: Tot^n dims={tot_dims}, Σ_sectors={sum(c for _, c in counts)}')
    except Exception as exc:
        print(f'  ⚠ Wu sectors ἀπέτυχε: {exc}')
        (OUT / 'wu_sectors.json').write_text(json.dumps({
            'b4': 0, 'n_total_tot4': 0, 'n_sectors': 0,
            'sector_vertex_ids': [], 'sector_counts': [],
            'tot_dims': [], 'bidegree': [],
        }), encoding='utf-8')

    print('Building stones index…')
    try:
        registry = StoneRegistry(HERE)
        registry.load_index()
        (OUT / 'stones.json').write_text(json.dumps(stones_to_json(registry), ensure_ascii=False), encoding='utf-8')
        print(f'  → stones.json: {len(registry)} πέτρες')
    except Exception as exc:
        print(f'  ⚠ stones index ἀπέτυχε: {exc}')
        (OUT / 'stones.json').write_text(json.dumps({'count': 0, 'stones': []}), encoding='utf-8')

    print('Copying Python sources…')
    copy_python_sources()

    print(f'\nConstants γιὰ JS: ε*={EPSILON:.10f}, ε_nom={EPSILON_NOMINAL}, φ={PHI:.10f}')
    print(f'Data: {OUT}')
    print(f'Py:   {PY_OUT}')


if __name__ == '__main__':
    main()
