"""
ark_irreps.py — Iₕ characters & projectors
============================================

Rotation icosahedral group I = A₅, |I| = 60, irreps {A, T₁, T₂, G, H}.

Permutation rep decompositions (Π-Μ-14):
  ρ_β₁₀ = A ⊕ T₁ ⊕ T₂ ⊕ H              (G ΑΠΟΥΣΑ — G-κένωσι)
  ρ_β₆  = A ⊕ T₁ ⊕ T₂ ⊕ 2G ⊕ H
  ρ_β₄  = A ⊕ T₁ ⊕ T₂ ⊕ 2G ⊕ 3H        (rank 30)

Δύο τρόποι κατασκευῆς projectors:
  1. build_irrep_projectors_30(seed) — random-orthogonal block (fallback v0)
  2. build_irrep_projectors_30_equivariant(dt_graph) — πραγματικοὶ Iₕ-
     equivariant μέσῳ permutation rep τῆς β₄ orbit (Στάδιο 2.2)

Ὁ τύπος ποὺ χρησιμοποιεῖται γιὰ τοὺς equivariant:
  P_ρ = (dim_ρ / |I|) · Σ_{g∈I} χ_ρ(g) · π(g)
"""

import numpy as np

from ark_geometry import PHI


# ═══════════════════════════════════════════════════════════════════
# ΣΤΑΘΕΡΕΣ
# ═══════════════════════════════════════════════════════════════════

IH_DIMS = {'A': 1, 'T1': 3, 'T2': 3, 'G': 4, 'H': 5}

PERM_DECOMP = {
    'β10': {'A': 1, 'T1': 1, 'T2': 1, 'G': 0, 'H': 1},   # G ΑΠΟΥΣΑ
    'β6':  {'A': 1, 'T1': 1, 'T2': 1, 'G': 2, 'H': 1},
    'β4':  {'A': 1, 'T1': 1, 'T2': 1, 'G': 2, 'H': 3},
}

# Iₕ-irrep ranks στὴν β₄ permutation rep (mult × dim)
RANKS_BETA4 = {'A': 1, 'T1': 3, 'T2': 3, 'G': 8, 'H': 15}  # Σ=30


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

    v0 fallback: block-diagonal projectors μὲ random orthogonal basis.
    Γιὰ τὸν πραγματικὸ Iₕ-equivariant ὑπολογισμό → χρῆσι
    build_irrep_projectors_30_equivariant(dt_graph).

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
# Γ. Iₕ-EQUIVARIANT PROJECTORS (μέσῳ permutation rep τῆς β₄)
# ═══════════════════════════════════════════════════════════════════

# Conjugacy classes τῆς I, ὀρθογνώμονες μὲ τὶς γραμμὲς τοῦ character
# table. Ταξινόμησι ἀνὰ trace τῆς R ∈ SO(3): tr(R) = 1 + 2cos(θ).
#   E:    θ=0     → tr=3
#   C2:   θ=π     → tr=-1
#   C3:   θ=2π/3  → tr=0
#   C5:   θ=2π/5  → tr=1+2cos(72°) = PHI
#   C5²:  θ=4π/5  → tr=1+2cos(144°) = 1-PHI

_CLASS_INDEX = {'E': 0, 'C2': 1, 'C3': 2, 'C5': 3, 'C52': 4}


def _classify_rotation(R, tol=1e-6):
    """Ταξινομεῖ ἕναν 3×3 περιστροφὴ τῆς I σὲ μία ἀπὸ 5 conjugacy classes."""
    tr = np.trace(R)
    if abs(tr - 3) < tol:        return 'E'
    if abs(tr + 1) < tol:        return 'C2'
    if abs(tr) < tol:            return 'C3'
    if abs(tr - PHI) < tol:      return 'C5'
    if abs(tr - (1 - PHI)) < tol: return 'C52'
    raise ValueError(f"unknown class for tr(R)={tr}")


def build_icosahedral_rotations(dt_graph, tol=1e-8):
    """Ὅλες οἱ 60 περιστροφὲς ποὺ διατηροῦν τὶς β₁₀ κορυφές.

    Ἀρχή: 3 σταθερὲς linearly independent β₁₀ κορυφὲς (v₁, v₂, v₃) ὁρίζουν
    πλαίσιο. Γιὰ κάθε ordered triplet (w₁, w₂, w₃) στὶς 12 κορυφὲς ποὺ
    διατηρεῖ τὶς ζευγαριαστὲς ἀποστάσεις, τὸ R = W·V⁻¹ εἶναι μοναδικά
    ὁρισμένο. Φιλτράρισμα: ὀρθογώνιος, det=+1, διατηρεῖ τὸ σύνολο β₁₀.
    """
    coords = dt_graph['coords']
    beta10 = dt_graph['orbits']['β10']
    icosa = coords[beta10]  # (12, 3)

    # Βρὲς 3 linearly independent ἀρχικὰ
    base_idx = None
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                V = np.column_stack([icosa[i], icosa[j], icosa[k]])
                if abs(np.linalg.det(V)) > 0.1:
                    base_idx = (i, j, k)
                    break
            if base_idx: break
        if base_idx: break
    assert base_idx is not None

    i0, j0, k0 = base_idx
    v1, v2, v3 = icosa[i0], icosa[j0], icosa[k0]
    V = np.column_stack([v1, v2, v3])
    V_inv = np.linalg.inv(V)
    d12 = np.linalg.norm(v1 - v2)
    d13 = np.linalg.norm(v1 - v3)
    d23 = np.linalg.norm(v2 - v3)

    rotations = []
    for i in range(12):
        for j in range(12):
            if j == i: continue
            if abs(np.linalg.norm(icosa[i] - icosa[j]) - d12) > tol: continue
            for k in range(12):
                if k == i or k == j: continue
                w1, w2, w3 = icosa[i], icosa[j], icosa[k]
                if abs(np.linalg.norm(w1 - w3) - d13) > tol: continue
                if abs(np.linalg.norm(w2 - w3) - d23) > tol: continue
                W = np.column_stack([w1, w2, w3])
                R = W @ V_inv
                if not np.allclose(R @ R.T, np.eye(3), atol=tol): continue
                if abs(np.linalg.det(R) - 1) > tol: continue  # rotation only
                # Διατηρεῖ τὸ σύνολο τῶν 12;
                rotated = icosa @ R.T
                from scipy.spatial.distance import cdist
                D = cdist(rotated, icosa)
                if not np.all(D.min(axis=1) < tol): continue
                # Μὴν προσθέσεις διπλό
                if any(np.allclose(R, prev, atol=tol) for prev in rotations):
                    continue
                rotations.append(R)

    assert len(rotations) == 60, f"got {len(rotations)} rotations (expected 60)"
    return rotations


def beta4_permutation_rep(dt_graph, rotations=None, tol=1e-8):
    """π: I → S(30) ὡς 30×30 permutation matrices.

    Δράσι κάθε R στὶς β₄ κορυφές (midpoints τῶν icosa edges) ὁρίζει
    permutation. Ἐπιστρέφει list[60] ἀπὸ {0,1}-πίνακες 30×30.
    """
    if rotations is None:
        rotations = build_icosahedral_rotations(dt_graph)

    coords = dt_graph['coords']
    beta4 = dt_graph['orbits']['β4']
    beta4_coords = coords[beta4]  # (30, 3)

    from scipy.spatial.distance import cdist
    perms = []
    for R in rotations:
        rotated = beta4_coords @ R.T  # (30, 3)
        D = cdist(rotated, beta4_coords)  # D[j, i] = ‖R·v_j - v_i‖
        target = D.argmin(axis=1)  # target[j] = i τέτοιο ὥστε R·v_j ≈ v_i
        # Validate: minima ὀρθὰ μηδενικὰ
        assert np.all(D[np.arange(30), target] < tol)
        # Validate: permutation (ὅλα τὰ targets διακριτά)
        assert len(set(target.tolist())) == 30
        P = np.zeros((30, 30))
        for j in range(30):
            P[target[j], j] = 1
        perms.append(P)
    return perms


def build_irrep_projectors_30_equivariant(dt_graph, tol=1e-8):
    """Πραγματικοὶ Iₕ-equivariant projectors via permutation rep τῆς β₄.

    Τύπος: P_ρ = (dim_ρ / |I|) · Σ_{g∈I} χ_ρ(g) · π(g)

    Ἐγγυᾶται Schur commutation: P_ρ · π(g) = π(g) · P_ρ ∀ g ∈ I.
    Tὰ projectors P_ρ εἶναι: idempotent, ἀμοιβαίως ὀρθογώνιοι, sum=I,
    rank(P_ρ) = mult_ρ × dim_ρ = RANKS_BETA4[ρ].
    """
    rotations = build_icosahedral_rotations(dt_graph, tol=tol)
    perms = beta4_permutation_rep(dt_graph, rotations=rotations, tol=tol)
    classes = [_classify_rotation(R) for R in rotations]

    chars = i_character_table()['characters']
    G_order = 60

    projectors = {}
    for name, dim in IH_DIMS.items():
        P = np.zeros((30, 30))
        for cls, pi in zip(classes, perms):
            chi_g = chars[name][_CLASS_INDEX[cls]]
            P += chi_g * pi
        P *= dim / G_order
        projectors[name] = P

    # Sanity: idempotent + sum=I + correct ranks
    P_sum = sum(projectors.values())
    assert np.allclose(P_sum, np.eye(30), atol=1e-8), \
        f"Σ P_ρ ≠ I: max err {np.max(np.abs(P_sum - np.eye(30)))}"
    for name, P in projectors.items():
        assert np.allclose(P @ P, P, atol=1e-8), f"{name} not idempotent"
        r = int(round(np.trace(P)))
        assert r == RANKS_BETA4[name], \
            f"{name}: rank={r}, expected {RANKS_BETA4[name]}"

    return projectors
