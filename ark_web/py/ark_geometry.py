"""
ark_geometry.py — Γεωμετρικὲς κατασκευὲς τῆς Κιβωτοῦ
=====================================================

DT (Disdyakis Triacontahedron, 3D, V=62, E=180, F=120, χ=2)
24-cell (4D αὐτοδυϊκό, V=24, E=96, F=96, cells=24, χ=0)
600-cell (4D, V=120, E=720, F=1200, cells=600, χ=0)

Σταθερὲς + builders + normalized graph Laplacian.

Στάδιο 2 (split ἀπὸ ARK_DIAGNOSTIC_v0.py).
"""

import numpy as np
from itertools import combinations, product
from scipy.spatial.distance import pdist, squareform, cdist


# ═══════════════════════════════════════════════════════════════════
# ΣΤΑΘΕΡΕΣ ΚΙΒΩΤΟΥ
# ═══════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2                # Χρυσὴ τομή
RATIO = (9 + 4*np.sqrt(2)) / 7            # Φασματικὸς λόγος
EPSILON = 4 * (4*np.sqrt(2) - 5) / 175    # Ἀκριβὲς ε* ≈ 0.0150138114
EPSILON_NOMINAL = 0.015                   # Στρογγυλεμένο display value (1.5%)

DT_V, DT_E, DT_F, DT_CHI = 62, 180, 120, 2
BETA10, BETA6, BETA4 = 12, 20, 30

CELL24_V, CELL24_E, CELL24_F, CELL24_CELLS = 24, 96, 96, 24
CELL600_V, CELL600_E = 120, 720

CHEEGER_CRITICAL_EDGES = BETA10     # 12


# ═══════════════════════════════════════════════════════════════════
# Α. ΓΕΩΜΕΤΡΙΚΕΣ ΚΑΤΑΣΚΕΥΕΣ
# ═══════════════════════════════════════════════════════════════════

def build_DT():
    """DT: V=62 (12 β₁₀ + 20 β₆ + 30 β₄), E=180, F=120, χ=2."""
    icosa_raw = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            icosa_raw.append((0, s1, s2*PHI))
            icosa_raw.append((s1, s2*PHI, 0))
            icosa_raw.append((s1*PHI, 0, s2))
    icosa = []
    seen = set()
    for v in icosa_raw:
        key = tuple(round(x, 10) for x in v)
        if key not in seen:
            seen.add(key)
            icosa.append(np.array(v))
    icosa = np.array(icosa) / np.sqrt(1 + PHI**2)
    assert len(icosa) == 12

    D = squareform(pdist(icosa))
    edge_len = np.min(D[D > 1e-6])
    icosa_edges = [(i, j) for i in range(12) for j in range(i+1, 12) if abs(D[i,j] - edge_len) < 1e-6]
    assert len(icosa_edges) == 30

    edge_set = set(frozenset(e) for e in icosa_edges)
    icosa_faces = []
    for i, j, k in combinations(range(12), 3):
        if all(frozenset({a,b}) in edge_set for a,b in [(i,j),(j,k),(i,k)]):
            icosa_faces.append((i, j, k))
    assert len(icosa_faces) == 20

    coords = list(icosa)
    beta10 = list(range(12))

    beta6 = []
    for f in icosa_faces:
        c = (icosa[f[0]] + icosa[f[1]] + icosa[f[2]]) / 3
        c = c / np.linalg.norm(c)
        coords.append(c)
        beta6.append(len(coords) - 1)

    beta4 = []
    edge_to_idx = {}
    for (i, j) in icosa_edges:
        m = (icosa[i] + icosa[j]) / 2
        m = m / np.linalg.norm(m)
        coords.append(m)
        beta4.append(len(coords) - 1)
        edge_to_idx[frozenset({i, j})] = beta4[-1]

    coords = np.array(coords)

    faces = []
    for face_local, (a, b, c) in enumerate(icosa_faces):
        Fc = beta6[face_local]
        m_ab = edge_to_idx[frozenset({a, b})]
        m_bc = edge_to_idx[frozenset({b, c})]
        m_ca = edge_to_idx[frozenset({c, a})]
        faces.extend([
            frozenset({a, m_ab, Fc}),  frozenset({m_ab, b, Fc}),
            frozenset({b, m_bc, Fc}),  frozenset({m_bc, c, Fc}),
            frozenset({c, m_ca, Fc}),  frozenset({m_ca, a, Fc}),
        ])
    assert len(faces) == 120

    edges = set()
    for tri in faces:
        verts = sorted(tri)
        for i in range(3):
            for j in range(i+1, 3):
                edges.add(frozenset({verts[i], verts[j]}))
    assert len(edges) == 180

    degrees = {v: 0 for v in range(62)}
    for e in edges:
        a, b = list(e)
        degrees[a] += 1
        degrees[b] += 1
    for v in beta10: assert degrees[v] == 10
    for v in beta6:  assert degrees[v] == 6
    for v in beta4:  assert degrees[v] == 4

    return {
        'name': 'DT',
        'coords': coords, 'vertices': list(range(62)),
        'orbits': {'β10': beta10, 'β6': beta6, 'β4': beta4},
        'edges': edges, 'faces': faces, 'degrees': degrees,
        'V': 62, 'E': 180, 'F': 120, 'chi': 2,
    }


def build_24cell():
    """24-cell: V=24, E=96, F=96 (τρίγωνα), cells=24, χ=0."""
    verts = []
    for i in range(4):
        for s in (1, -1):
            v = [0]*4; v[i] = s
            verts.append(tuple(v))
    for signs in product([0.5, -0.5], repeat=4):
        verts.append(signs)
    verts = np.array(sorted(set(verts)))
    assert verts.shape == (24, 4)

    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    edge_len = D.min()
    edges = set()
    for i in range(24):
        for j in range(i+1, 24):
            if D[i,j] < edge_len * 1.01:
                edges.add(frozenset({i, j}))
    assert len(edges) == 96

    return {
        'name': '24-cell',
        'V': 24, 'E': 96, 'F': 96, 'cells': 24,
        'chi': 0,  # 4-manifold: V−E+F−cells = 24−96+96−24 = 0
        'coords': verts,
        'edges': edges,
        'types': {'A': list(range(0, 8)), 'B': list(range(8, 24))},
    }


def build_600cell():
    """600-cell: V=120, E=720, F=1200, cells=600, χ=0."""
    verts = set()
    for i in range(4):
        for s in (1, -1):
            v = [0]*4; v[i] = s
            verts.add(tuple(v))
    for signs in product([0.5, -0.5], repeat=4):
        verts.add(signs)
    base = [0, 1/(2*PHI), 0.5, PHI/2]
    even_perms = [
        (0,1,2,3),(0,2,3,1),(0,3,1,2),
        (1,0,3,2),(1,2,0,3),(1,3,2,0),
        (2,0,1,3),(2,1,3,0),(2,3,0,1),
        (3,0,2,1),(3,1,0,2),(3,2,1,0),
    ]
    for perm in even_perms:
        permuted = [base[perm[i]] for i in range(4)]
        nz_idx = [i for i, x in enumerate(permuted) if x != 0]
        for signs in product([1, -1], repeat=len(nz_idx)):
            v = list(permuted)
            for k, idx in enumerate(nz_idx):
                v[idx] = signs[k] * abs(v[idx])
            verts.add(tuple(round(x, 10) for x in v))

    assert len(verts) == 120
    V = np.array(sorted(verts))

    D = cdist(V, V)
    np.fill_diagonal(D, np.inf)
    edge_len = D.min()
    edges = set()
    for i in range(120):
        for j in range(i+1, 120):
            if D[i,j] < edge_len * 1.01:
                edges.add(frozenset({i, j}))
    assert len(edges) == 720

    return {
        'name': '600-cell',
        'V': 120, 'E': 720, 'F': 1200, 'cells': 600,
        'chi': 0,
        'coords': V,
        'edges': edges,
    }


def graph_laplacian(graph):
    """Normalized graph Laplacian L = I - D^(-1/2) A D^(-1/2)."""
    V = graph['V']
    A = np.zeros((V, V))
    for e in graph['edges']:
        if isinstance(e, frozenset):
            i, j = list(e)
        else:
            i, j = e
        A[i, j] = A[j, i] = 1.0
    deg = A.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1)
    D_inv_half = np.diag(1.0 / np.sqrt(deg_safe))
    return np.eye(V) - D_inv_half @ A @ D_inv_half
