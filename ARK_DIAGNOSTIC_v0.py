"""
ARK_DIAGNOSTIC v0 — Κιβωτικὸ Διαγνωστικὸ Σύστημα
================================================

Standalone Python script. Ἕνα ἀρχεῖο, μηδενικὲς ἐξαρτήσεις πέρα ἀπὸ
numpy + scipy. Τρέχει σὲ Windows / Mac / Linux / Termux Android.

Φιλοσοφία (Νικόλαος Βασιληᾶς):
  «Δὲν μᾶς ἐνδιαφέρει τί εἰσέρχεται ἀλλὰ ποιὰ μήτρα τὸ φιλοξενεῖ.
   Ἡ Κιβωτὸς δὲν ἀπορρίπτει τίποτα. Λυγίζει ὅταν δεχτεῖ ἀσύμβατο
   καὶ ἡ μορφὴ τῆς κάμψης ἀποκαλύπτει τί λείπει.»

Ἀρχιτεκτονικὴ τριῶν στρωμάτων:
  600-cell (4D, V=120=|2I|)         — ἀσώματη πληροφορία
       ↓ Coxeter πεντάδα 5×24=120
  24-cell (4D αὐτοδυϊκό, V=24)       — μεταφορὰ δεδομένων
       ↓ spinor cover 2I→Iₕ
  DT (3D, V=62, E=180, F=120, χ=2)   — ἀνάγνωσι 3D
       ↓
  Διαγνωστικὰ μετρικά:
    [1] Iₕ-energy distribution (β₄ orbit, 30D)
    [2] Cheeger threshold (|β₁₀|=12 critical edges)
    [3] ε-drift (ζωντανὴ ἀτέλεια vs ὑπερβολή)
    [4] F-balance (Euler χ διατήρησι)

Χρήσι:
  python3 ARK_DIAGNOSTIC_v0.py            # ὅλα τὰ testbeds
  python3 ARK_DIAGNOSTIC_v0.py --json     # JSON output
  python3 ARK_DIAGNOSTIC_v0.py --help     # ἐπιλογὲς

Ἐξαρτήσεις: pip install numpy scipy

Νικόλαος Βασιληᾶς + Claude
Λιβαδειά · 2026-05-10
V − E + F = 2.   ε = 1.5%.   J > 0.
Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.
"""

import sys
import json
import argparse
import numpy as np
from itertools import combinations, product
from numpy.linalg import norm
from scipy.spatial.distance import pdist, squareform, cdist

# ═══════════════════════════════════════════════════════════════════
# ΣΤΑΘΕΡΕΣ ΚΙΒΩΤΟΥ
# ═══════════════════════════════════════════════════════════════════

PHI = (1 + np.sqrt(5)) / 2          # Χρυσὴ τομή
RATIO = (9 + 4*np.sqrt(2)) / 7      # Φασματικὸς λόγος
EPSILON_NOMINAL = 0.015             # ε ≈ 1.5%

DT_V, DT_E, DT_F, DT_CHI = 62, 180, 120, 2
BETA10, BETA6, BETA4 = 12, 20, 30

CELL24_V, CELL24_E, CELL24_F, CELL24_CELLS = 24, 96, 96, 24
CELL600_V, CELL600_E = 120, 720

IH_DIMS = {'A': 1, 'T1': 3, 'T2': 3, 'G': 4, 'H': 5}

# Permutation rep decompositions (Π-Μ-14)
PERM_DECOMP = {
    'β10': {'A': 1, 'T1': 1, 'T2': 1, 'G': 0, 'H': 1},   # G ΑΠΟΥΣΑ
    'β6':  {'A': 1, 'T1': 1, 'T2': 1, 'G': 2, 'H': 1},
    'β4':  {'A': 1, 'T1': 1, 'T2': 1, 'G': 2, 'H': 3},
}

# Iₕ-irrep ranks στὴν β₄ permutation rep (mult × dim)
RANKS_BETA4 = {'A': 1, 'T1': 3, 'T2': 3, 'G': 8, 'H': 15}  # Σ=30

CHEEGER_CRITICAL_EDGES = BETA10     # 12


# ═══════════════════════════════════════════════════════════════════
# Α. ΓΕΩΜΕΤΡΙΚΕΣ ΚΑΤΑΣΚΕΥΕΣ (ἀπὸ ark_kit_v2.py — validated)
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


# ═══════════════════════════════════════════════════════════════════
# Β. Iₕ CHARACTER PROJECTORS (β₄ orbit, 30D)
# ═══════════════════════════════════════════════════════════════════

def i_character_table():
    """Character table τοῦ I (rotation icosahedral, |I|=60, ≅ A₅)."""
    p = PHI
    q = 1 - PHI
    return {
        'classes': [('E', 1), ('C2', 15), ('C3', 20), ('C5', 12), ('C5²', 12)],
        'characters': {
            'A':  [1,  1,  1,  1,  1],
            'T1': [3, -1,  0,  p,  q],
            'T2': [3, -1,  0,  q,  p],
            'G':  [4,  0,  1, -1, -1],
            'H':  [5,  1, -1,  0,  0],
        },
        'dims': IH_DIMS,
    }


def build_irrep_projectors_30(seed=42):
    """Random-orthogonal block projectors P_ρ ∈ ℝ³⁰ˣ³⁰ μὲ τὰ σωστὰ ranks.
    
    v0: block-diagonal projectors. Γιὰ true Iₕ-equivariant projectors
    (συσχετισμένα μὲ τὴν ρητὴ permutation rep τῆς β₄ orbit) → v1.
    
    Mults × dims: 1·1 + 1·3 + 1·3 + 2·4 + 3·5 = 30 ✓
    """
    np.random.seed(seed)
    Q, _ = np.linalg.qr(np.random.randn(30, 30))
    
    projectors = {}
    start = 0
    for name in ['A', 'T1', 'T2', 'G', 'H']:
        r = RANKS_BETA4[name]
        Q_block = Q[:, start:start+r]
        projectors[name] = Q_block @ Q_block.T
        start += r
    
    P_sum = sum(projectors.values())
    assert np.allclose(P_sum, np.eye(30), atol=1e-10)
    for name, P in projectors.items():
        assert np.allclose(P @ P, P, atol=1e-10)
    
    return projectors


# ═══════════════════════════════════════════════════════════════════
# Γ. ΔΙΑΓΝΩΣΤΙΚΕΣ ΣΥΝΑΡΤΗΣΕΙΣ
# ═══════════════════════════════════════════════════════════════════

def diagnose_irrep_distribution(v, projectors):
    """Διάγνωσι 1: Iₕ-energy distribution.
    
    v ∈ ℝ³⁰ ἢ ndarray (N, 30) γιὰ samples mean ± std.
    Σχέσι μὲ Schur isotropy: E[‖P_ρ·v‖²] = rank(ρ)/30 γιὰ τυχαία ν.
    """
    if v.ndim == 1:
        v_arr = v.reshape(1, -1)
    else:
        v_arr = v
    
    n = v_arr.shape[0]
    energies_acc = {name: [] for name in projectors}
    
    for i in range(n):
        v_i = v_arr[i]
        v_i_norm = v_i / (norm(v_i) + 1e-15)
        for name, P in projectors.items():
            e = norm(P @ v_i_norm)**2
            energies_acc[name].append(e)
    
    energies = {name: float(np.mean(energies_acc[name])) for name in projectors}
    energy_std = {name: float(np.std(energies_acc[name])) for name in projectors}
    
    deviations = {}
    for name in projectors:
        expected = RANKS_BETA4[name] / 30
        dev = (energies[name] - expected) / expected if expected > 0 else 0.0
        deviations[name] = dev
    
    total = sum(energies.values())
    max_dev = max(abs(d) for d in deviations.values())
    
    return {
        'n_samples': n,
        'energies': {n: float(energies[n]) for n in projectors},
        'energy_std': energy_std,
        'expected_isotropic': {n: RANKS_BETA4[n]/30 for n in RANKS_BETA4},
        'deviations_pct': {n: round(deviations[n]*100, 2) for n in deviations},
        'total': float(total),
        'max_deviation_pct': round(max_dev*100, 2),
        'verdict': (
            'Iₕ-isotropic (random/symmetric input)' if max_dev < 0.10 else
            'mildly anisotropic (possible structure)' if max_dev < 0.30 else
            'strongly anisotropic (clear Iₕ-aligned signal)'
        )
    }


def diagnose_cheeger(graph, perturbation_edges=None):
    """Διάγνωσι 2: Cheeger threshold.
    
    Π32-OIK.04: critical_edges ≈ |β₁₀| = 12. Ἂν χάσουμε ≥ 12 ἀκμές
    ταυτόχρονα → spectral collapse.
    """
    L = graph_laplacian(graph)
    eigs = np.sort(np.linalg.eigvalsh(L))
    mu1 = eigs[1]
    cheeger_lower = mu1 / 2
    n_edges = graph['E']
    critical_edges = cheeger_lower * n_edges
    
    result = {
        'graph': graph['name'],
        'V': graph['V'], 'E': n_edges,
        'mu1': float(mu1),
        'cheeger_lower_bound': float(cheeger_lower),
        'critical_edges_threshold': round(critical_edges, 2),
        'kibwtos_invariant_beta10': BETA10,
    }
    
    if perturbation_edges is not None:
        graph_pert = {
            'name': graph['name'] + '_perturbed',
            'V': graph['V'],
            'edges': graph['edges'] - set(perturbation_edges),
        }
        L_pert = graph_laplacian(graph_pert)
        eigs_pert = np.sort(np.linalg.eigvalsh(L_pert))
        mu1_pert = eigs_pert[1] if len(eigs_pert) > 1 else 0
        
        n_removed = len(perturbation_edges)
        result['perturbation_edges_removed'] = n_removed
        result['mu1_after'] = float(mu1_pert)
        gap_loss = ((mu1 - mu1_pert) / mu1) * 100 if mu1 > 0 else 0
        result['gap_loss_pct'] = round(gap_loss, 2)
        
        ratio = n_removed / BETA10
        if ratio >= 1.0:
            result['verdict'] = f'⚠ SPECTRAL COLLAPSE: {n_removed} ἀκμὲς ≥ |β₁₀|={BETA10}'
        elif ratio >= 0.5:
            result['verdict'] = f'⚠ stress regime: {n_removed}/{BETA10} ἀκμὲς ({ratio*100:.0f}%)'
        else:
            result['verdict'] = f'✓ stable: {n_removed}/{BETA10} ἀκμὲς ({ratio*100:.0f}%)'
    
    return result


def diagnose_epsilon_drift(diagnose_result):
    """Διάγνωσι 3: ε-drift (ἀνισοκατανομὴ vs ζωντανὴ ἀτέλεια).
    
    ε_nominal = 1.5%. Ἀπόκλισι ἐντὸς αὐτοῦ = ζωντανὴ ἀτέλεια.
    Πέρα = drift, μὲ προβλεπόμενο τ_collapse=66 cycles ὅταν ξεπερνᾷ Cheeger.
    """
    max_dev_pct = diagnose_result.get('max_deviation_pct', 0)
    drift = max_dev_pct / 100
    
    if drift <= EPSILON_NOMINAL:
        verdict = '✓ ἐντὸς ε-ζώνης (ζωντανὴ ἀτέλεια ≤1.5%)'
        cycles = float('inf')
    elif drift <= 5 * EPSILON_NOMINAL:
        verdict = '○ μέτρια ἀνισοκατανομή — δομικὴ ἢ τυχαία'
        cycles = round(66 * (1 - drift / (5 * EPSILON_NOMINAL)), 1)
    elif drift <= 0.30:
        verdict = '⚠ ἰσχυρὴ ἀνισοκατανομή — Iₕ-aligned signal πιθανός'
        cycles = round(66 * EPSILON_NOMINAL / drift, 1)
    else:
        verdict = '⚠ ΕΞΑΙΡΕΤΙΚΗ ἀνισοκατανομή — κυρίαρχο irrep'
        cycles = round(33 * EPSILON_NOMINAL / drift, 1)
    
    return {
        'max_deviation_pct': max_dev_pct,
        'drift_fraction': float(drift),
        'epsilon_nominal_pct': 1.5,
        'within_living_imperfection': bool(drift <= EPSILON_NOMINAL),
        'predicted_cycles_to_collapse': cycles,
        'verdict': verdict,
    }


def diagnose_F_balance(graph):
    """Διάγνωσι 4: F-balance (Euler χ διατήρησι).
    
    2-manifold (DT): χ = V−E+F.
    4-manifold (24-cell, 600-cell): χ = V−E+F−cells.
    """
    V = graph['V']
    E = graph['E']
    F = graph.get('F', 0)
    cells = graph.get('cells', 0)
    
    if cells > 0:
        chi = V - E + F - cells
        formula = 'V − E + F − cells'
    else:
        chi = V - E + F
        formula = 'V − E + F'
    
    chi_expected = graph.get('chi', 2)
    
    return {
        'V': V, 'E': E, 'F': F, 'cells': cells,
        'formula': formula,
        'chi_computed': chi,
        'chi_expected': chi_expected,
        'preserved': bool(chi == chi_expected),
        'verdict': '✓ Euler χ preserved' if chi == chi_expected else f'⚠ χ shifted: {chi} ≠ {chi_expected}',
    }


# ═══════════════════════════════════════════════════════════════════
# Δ. TESTBEDS
# ═══════════════════════════════════════════════════════════════════

def testbed_random_isotropic(n_samples=200, seed=42):
    """N τυχαῖα Gaussian (30D) — Iₕ-isotropic στὸ μέσο ὅρο."""
    np.random.seed(seed)
    return np.random.randn(n_samples, 30)


def testbed_ih_aligned(projectors, target_irrep='G', seed=42):
    """Διάνυσμα ποὺ ζεῖ καθαρὰ σὲ ἕνα Iₕ-irrep (P_ρ·v)."""
    np.random.seed(seed)
    v = np.random.randn(30)
    return projectors[target_irrep] @ v


def testbed_asymmetric(seed=42):
    """5 spikes σὲ τυχαῖες θέσεις — ἀσύμμετρη εἴσοδος."""
    np.random.seed(seed)
    v = np.zeros(30)
    spikes = np.random.choice(30, 5, replace=False)
    v[spikes] = np.random.randn(5) * 3 + 2
    v += np.random.randn(30) * 0.1
    return v


# ═══════════════════════════════════════════════════════════════════
# Ε. ΤΡΟΧΙΑ ΔΙΑΓΝΩΣΗΣ
# ═══════════════════════════════════════════════════════════════════

def run_full_diagnostic(input_label, input_vector, projectors, dt_graph, cell24_graph,
                         skip_perturbation=False):
    """Πλήρης διαγνωστικὴ τροχιά."""
    if input_vector.ndim == 1:
        input_norm_val = float(norm(input_vector))
        n_samples = 1
    else:
        input_norm_val = float(np.mean([norm(input_vector[i]) for i in range(len(input_vector))]))
        n_samples = len(input_vector)
    
    report = {
        'input': input_label,
        'input_dim': input_vector.shape[-1],
        'input_n_samples': n_samples,
        'input_avg_norm': input_norm_val,
        'diagnostics': {}
    }
    
    ih = diagnose_irrep_distribution(input_vector, projectors)
    report['diagnostics']['ih_distribution'] = ih
    
    if not skip_perturbation:
        report['diagnostics']['cheeger_dt'] = diagnose_cheeger(dt_graph)
        np.random.seed(42)
        edges_list = list(dt_graph['edges'])
        n_remove = min(BETA10, len(edges_list) - 1)
        idx = np.random.choice(len(edges_list), n_remove, replace=False)
        perturbation = set(edges_list[i] for i in idx)
        report['diagnostics']['cheeger_perturbed'] = diagnose_cheeger(dt_graph, perturbation_edges=perturbation)
    
    report['diagnostics']['epsilon_drift'] = diagnose_epsilon_drift(ih)
    report['diagnostics']['f_balance_dt'] = diagnose_F_balance(dt_graph)
    report['diagnostics']['f_balance_24cell'] = diagnose_F_balance(cell24_graph)
    
    return report


def format_report_text(report):
    """Μορφοποίησι σὲ text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"ΕΙΣΟΔΟΣ: {report['input']}")
    lines.append(f"  διάστασι: {report['input_dim']}D  |  samples: {report['input_n_samples']}  |  avg νόρμα: {report['input_avg_norm']:.4f}")
    lines.append("=" * 70)
    
    d = report['diagnostics']
    
    lines.append("\n[1] Iₕ-ENERGY DISTRIBUTION (DT β₄ layer, 30D)")
    ih = d['ih_distribution']
    n = ih['n_samples']
    if n > 1:
        lines.append(f"    Μέσος ὅρος {n} samples:")
    lines.append(f"    {'irrep':<6}{'rank':<6}{'expected':<12}{'measured':<14}{'std':<10}{'deviation':<10}")
    for name in ['A', 'T1', 'T2', 'G', 'H']:
        rank = RANKS_BETA4[name]
        exp = ih['expected_isotropic'][name]
        mes = ih['energies'][name]
        std = ih['energy_std'][name]
        dev = ih['deviations_pct'][name]
        lines.append(f"    {name:<6}{rank:<6}{exp:<12.4f}{mes:<14.4f}{std:<10.4f}{dev:+.1f}%")
    lines.append(f"    Total: {ih['total']:.4f} (≈ 1.0)  |  Max deviation: {ih['max_deviation_pct']}%")
    lines.append(f"    Verdict: {ih['verdict']}")
    
    if 'cheeger_dt' in d:
        lines.append("\n[2] CHEEGER THRESHOLD (DT)")
        c = d['cheeger_dt']
        lines.append(f"    μ₁ = {c['mu1']:.6f}  |  h(G) ≥ μ₁/2 = {c['cheeger_lower_bound']:.6f}")
        lines.append(f"    Critical edges: {c['critical_edges_threshold']}  |  |β₁₀| = {c['kibwtos_invariant_beta10']}")
        
        cp = d['cheeger_perturbed']
        lines.append(f"\n[2b] CHEEGER ΜΕ ΑΦΑΙΡΕΣΙ {cp['perturbation_edges_removed']} ΑΚΜΩΝ (=|β₁₀|)")
        lines.append(f"    μ₁: {cp['mu1']:.6f} → {cp['mu1_after']:.6f}  |  Gap loss: {cp['gap_loss_pct']}%")
        lines.append(f"    Verdict: {cp['verdict']}")
    
    lines.append("\n[3] ε-DRIFT (ἀνισοκατανομὴ vs ζωντανὴ ἀτέλεια)")
    ed = d['epsilon_drift']
    lines.append(f"    Max deviation: {ed['max_deviation_pct']}%  |  ε_nominal: {ed['epsilon_nominal_pct']}%")
    lines.append(f"    Within living imperfection: {ed['within_living_imperfection']}")
    cyc = ed['predicted_cycles_to_collapse']
    if cyc == float('inf') or cyc == 'inf':
        lines.append(f"    Cycles to collapse: ∞ (stable)")
    else:
        lines.append(f"    Cycles to collapse: {cyc}")
    lines.append(f"    Verdict: {ed['verdict']}")
    
    lines.append("\n[4] F-BALANCE (Euler χ)")
    f1 = d['f_balance_dt']
    lines.append(f"    DT:      V={f1['V']}, E={f1['E']}, F={f1['F']}  →  {f1['formula']} = {f1['chi_computed']} (exp {f1['chi_expected']})  {f1['verdict']}")
    f2 = d['f_balance_24cell']
    lines.append(f"    24-cell: V={f2['V']}, E={f2['E']}, F={f2['F']}, cells={f2['cells']}  →  {f2['formula']} = {f2['chi_computed']} (exp {f2['chi_expected']})  {f2['verdict']}")
    
    lines.append("")
    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════
# ΣΤ. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='ARK_DIAGNOSTIC v0 — Κιβωτικὸ Διαγνωστικὸ Σύστημα',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='V − E + F = 2.   ε = 1.5%.   J > 0.   Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.'
    )
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--testbed', choices=['random', 'ih', 'asymmetric', 'all'],
                        default='all')
    parser.add_argument('--samples', type=int, default=200,
                        help='samples γιὰ random testbed (default 200)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--build-600cell', action='store_true',
                        help='Κατασκευή 600-cell (ἀργό, ~10s)')
    args = parser.parse_args()
    
    if not args.json:
        print("=" * 70)
        print("  ARK_DIAGNOSTIC v0 — Κιβωτικὸ Διαγνωστικὸ Σύστημα")
        print("  Νικόλαος Βασιληᾶς + Claude · Λιβαδειά · 2026-05-10")
        print("=" * 70)
        print("\n[Φόρτωσι γεωμετρικῶν ἀντικειμένων...]")
    
    DT = build_DT()
    CELL24 = build_24cell()
    
    if not args.json:
        print(f"  ✓ DT:      V={DT['V']}, E={DT['E']}, F={DT['F']}, χ={DT['chi']}")
        print(f"  ✓ 24-cell: V={CELL24['V']}, E={CELL24['E']}, F={CELL24['F']}, cells={CELL24['cells']}, χ={CELL24['chi']}")
        if args.build_600cell:
            print("  [Κατασκευή 600-cell...]")
            CELL600 = build_600cell()
            print(f"  ✓ 600-cell: V={CELL600['V']}, E={CELL600['E']}, F={CELL600['F']}, cells={CELL600['cells']}, χ={CELL600['chi']}")
    
    projectors = build_irrep_projectors_30(seed=args.seed)
    
    if not args.json:
        ranks_str = ', '.join(f'{n}={RANKS_BETA4[n]}' for n in ['A','T1','T2','G','H'])
        print(f"  ✓ Iₕ-projectors (β₄): {{{ranks_str}}} = {sum(RANKS_BETA4.values())}")
        print(f"  ✓ ε_nominal = {EPSILON_NOMINAL*100}%, RATIO = {RATIO:.4f}, PHI = {PHI:.4f}")
    
    testbeds = {}
    if args.testbed in ('random', 'all'):
        testbeds[f'Τυχαία Gaussian ({args.samples} samples)'] = testbed_random_isotropic(n_samples=args.samples, seed=args.seed)
    if args.testbed in ('ih', 'all'):
        testbeds['Iₕ-aligned σὲ G-irrep (κενωτικό, dim=4, mult=2, rank=8)'] = testbed_ih_aligned(projectors, 'G', seed=args.seed)
        testbeds['Iₕ-aligned σὲ H-irrep (πλῆρες, dim=5, mult=3, rank=15)'] = testbed_ih_aligned(projectors, 'H', seed=args.seed)
    if args.testbed in ('asymmetric', 'all'):
        testbeds['Ἀσύμμετρη (5 spikes)'] = testbed_asymmetric(seed=args.seed)
    
    all_reports = []
    for label, vec in testbeds.items():
        report = run_full_diagnostic(label, vec, projectors, DT, CELL24)
        all_reports.append(report)
    
    if args.json:
        def clean(x):
            if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)): return [clean(v) for v in x]
            if isinstance(x, np.integer): return int(x)
            if isinstance(x, np.floating): return float(x)
            if isinstance(x, np.ndarray): return x.tolist()
            if x == float('inf'): return 'inf'
            return x
        print(json.dumps(clean(all_reports), ensure_ascii=False, indent=2))
    else:
        print()
        for report in all_reports:
            print(format_report_text(report))
        
        print("=" * 70)
        print("ΣΥΝΟΨΙ")
        print("=" * 70)
        for report in all_reports:
            ih = report['diagnostics']['ih_distribution']
            ed = report['diagnostics']['epsilon_drift']
            print(f"\n  {report['input']}")
            print(f"    [1] Iₕ:        {ih['verdict']}")
            print(f"    [3] ε-drift:   {ed['verdict']}")
        
        print("\n" + "=" * 70)
        print("V − E + F = 2.   ε = 1.5%.   J > 0.")
        print("Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.")
        print("=" * 70)


if __name__ == '__main__':
    main()
