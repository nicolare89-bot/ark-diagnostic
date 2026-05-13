"""
ark_wu.py — Wu bicomplex τοῦ ζεύγους (DT, ΠΕ)
==============================================

Wu cross-interaction cohomology (Wen-tsün Wu) στὸ ζεῦγος
  DT = Disdyakis Triacontahedron (V=62, E=180, F=120)
  ΠΕ = Πατρικὸ Εἰκοσάεδρο (V=12 = β₁₀, E=30, F=20)

Bidegree (p, q) = (dim σ στὸ DT, dim τ στὸ ΠΕ), βάσι: ζεύγη (σ, τ) μὲ
σ ∩ τ ≠ ∅. Πλήρης πίνακας διαστάσεων (Π-28-Β08):

  C^(p,q)  |  q=0   q=1   q=2
  ───────  +  ────  ────  ────
  p=0      |   12    60    60
  p=1      |  120   600   600
  p=2      |  120   600   600

Σύνολο 2772 cells. Σ(-1)^(p+q)·|C^(p,q)| = 12 = |β₁₀|.

Tot^n = ⊕_{p+q=n} C^(p,q), n=0..4. b_n = (0, 0, 0, 0, 12).
Wu concentration (Π-28-Β05): Tot ≅ ⊕_{v ∈ β₁₀} Wu_v (12 ἀνεξάρτητα ἀντίγραφα).
Π-28-Β06: Wu_v ≅ C*(openStar_DT(v)) ⊗ C*(openStar_ΠΕ(v)), Künneth → b_4 = 1
  ἀνὰ ἀντίγραφο, σύνολο 12.
Π-28-Β07: Δ_4 block-diagonal ἀκριβῶς, off-block = 0 (machine epsilon).
Π-28-Β10: 12 = V(ΠΕ) = |β₁₀| = #Wu sectors — ταυτότητα τριῶν δωδεκάδων.

Στάδιο 2.8 (#5 ἀπὸ τοὺς στόχους Σταδίου 2 στὸ CLAUDE.md).
Μεταφορὰ ἀπὸ ark_kit.py (b-ARK-a/OFFICE/ΒΟΗΘΗΤΙΚΑ/ark_kit.py).

Σημείωσι (Π32-ARK.09):
    Τὸ Wu(DT,ΠΕ) δὲν διαγιγνώσκει inputs μέσῳ DT-μόνο twisted embeddings
    — Künneth τὸ ἀπαγορεύει. Wu_v ≅ C*(openStar_DT(v)) ⊗ C*(openStar_ΠΕ(v)),
    καὶ ἡ ἁρμονικὴ συνιστῶσα ζεῖ στὸ top καὶ τῶν δύο παραγόντων· trivial ΠΕ
    συντελεστὴς → μηδενικὴ προβολή ἀναγκαστικά. Ἐπιβεβαιωμένο ἐμπειρικὰ
    σὲ Π32-WU-EMBED.03 (6 παραλλαγὲς ψ-twist DT-side: ὅλες harmonic
    fraction < 10⁻³⁰). Ἀπαιτεῖται ψ-twist στὸν ΠΕ παράγοντα γιὰ μὴ-μηδενικὴ
    προβολή — ἀνοιχτὸ Ω.ARK.γ.
"""

import numpy as np
from itertools import combinations
from numpy.linalg import eigh, matrix_rank
from scipy.spatial.distance import pdist, squareform


# ═══════════════════════════════════════════════════════════════════
# Α. Πατρικὸ Εἰκοσάεδρο
# ═══════════════════════════════════════════════════════════════════

def build_PE(dt):
    """ΠΕ ὡς simplicial complex μὲ V = β₁₀ τοῦ DT.

    V=12, E=30, F=20, χ=2, 5-regular. Edges εἶναι τὰ ζεύγη β₁₀ κορυφῶν
    σὲ ἐλάχιστη ἀπόστασι (στὶς συντεταγμένες τοῦ DT). Faces εἶναι τὰ
    τρίγωνα ποὺ σχηματίζουν.
    """
    beta10 = dt['orbits']['β10']
    coords = dt['coords'][beta10]
    D = squareform(pdist(coords))
    edge_len = np.min(D[D > 1e-6])
    edges = set()
    for i in range(12):
        for j in range(i+1, 12):
            if abs(D[i, j] - edge_len) < 1e-6:
                edges.add(frozenset({beta10[i], beta10[j]}))
    assert len(edges) == 30, f"PE edges {len(edges)}"

    faces = []
    for i, j, k in combinations(beta10, 3):
        if (frozenset({i, j}) in edges
                and frozenset({j, k}) in edges
                and frozenset({i, k}) in edges):
            faces.append(frozenset({i, j, k}))
    assert len(faces) == 20, f"PE faces {len(faces)}"

    degrees = {v: 0 for v in beta10}
    for e in edges:
        a, b = list(e)
        degrees[a] += 1
        degrees[b] += 1
    assert all(d == 5 for d in degrees.values()), "PE not 5-regular"

    return {
        'name': 'ΠΕ',
        'vertices': list(beta10),
        'edges': edges,
        'faces': faces,
        'degrees': degrees,
        'V': 12, 'E': 30, 'F': 20, 'chi': 2,
    }


# ═══════════════════════════════════════════════════════════════════
# Β. Wu bicomplex helpers
# ═══════════════════════════════════════════════════════════════════

def _simplices_by_dim(complex_dict):
    """Ἐπιστρέφει {0: [vertices ὡς frozensets], 1: [edges], 2: [faces]}."""
    return {
        0: [frozenset({v}) for v in complex_dict['vertices']],
        1: list(complex_dict['edges']),
        2: list(complex_dict['faces']),
    }


def _boundary_matrix(simplices_p, simplices_pm1):
    """Simplicial boundary ∂_p : C_p → C_{p-1} ὡς dense matrix."""
    if not simplices_p or not simplices_pm1:
        return np.zeros((len(simplices_pm1), len(simplices_p)))
    idx_pm1 = {s: i for i, s in enumerate(simplices_pm1)}
    M = np.zeros((len(simplices_pm1), len(simplices_p)))
    for j, s in enumerate(simplices_p):
        verts = sorted(s)
        for k in range(len(verts)):
            face = frozenset(verts[:k] + verts[k+1:])
            if face in idx_pm1:
                M[idx_pm1[face], j] = (-1) ** k
    return M


# ═══════════════════════════════════════════════════════════════════
# Γ. Wu bicomplex construction
# ═══════════════════════════════════════════════════════════════════

def wu_bicomplex(dt, pe):
    """Wu bicomplex τοῦ ζεύγους (DT, ΠΕ).

    Bidegree (p, q) = (dim σ, dim τ) μὲ σ ∈ DT, τ ∈ ΠΕ καὶ σ ∩ τ ≠ ∅.
    Cochain differential d(σ ⊗ τ) = d_DT σ ⊗ τ + (-1)^p σ ⊗ d_PE τ.

    Returns:
        {'basis': {n: [(σ, τ, p, q)...]} γιὰ n=0..4,
         'd':     {n: matrix Tot^n → Tot^{n+1}} γιὰ n=0..3,
         'dim':   {n: |Tot^n|}}
    """
    DT_smp = _simplices_by_dim(dt)
    PE_smp = _simplices_by_dim(pe)

    pairs_by_pq = {}
    for p in range(3):
        for q in range(3):
            lst = []
            for sigma in DT_smp[p]:
                for tau in PE_smp[q]:
                    if sigma & tau:
                        lst.append((sigma, tau))
            pairs_by_pq[(p, q)] = lst

    basis = {}
    for n in range(5):
        bn = []
        for p in range(3):
            q = n - p
            if 0 <= q <= 2:
                for sigma, tau in pairs_by_pq[(p, q)]:
                    bn.append((sigma, tau, p, q))
        basis[n] = bn

    boundary_DT = {
        1: _boundary_matrix(DT_smp[1], DT_smp[0]),
        2: _boundary_matrix(DT_smp[2], DT_smp[1]),
    }
    boundary_PE = {
        1: _boundary_matrix(PE_smp[1], PE_smp[0]),
        2: _boundary_matrix(PE_smp[2], PE_smp[1]),
    }
    DT_idx = {p: {s: i for i, s in enumerate(DT_smp[p])} for p in range(3)}
    PE_idx = {q: {s: i for i, s in enumerate(PE_smp[q])} for q in range(3)}

    def cochain_d_DT(p, sigma):
        if p >= 2:
            return []
        i_sigma = DT_idx[p][sigma]
        col = boundary_DT[p+1][i_sigma, :]
        return [(DT_smp[p+1][j], col[j]) for j in range(len(DT_smp[p+1])) if col[j] != 0]

    def cochain_d_PE(q, tau):
        if q >= 2:
            return []
        i_tau = PE_idx[q][tau]
        col = boundary_PE[q+1][i_tau, :]
        return [(PE_smp[q+1][j], col[j]) for j in range(len(PE_smp[q+1])) if col[j] != 0]

    diffs = {}
    for n in range(4):
        src = basis[n]
        tgt = basis[n+1]
        tgt_idx = {(s, t): i for i, (s, t, _, _) in enumerate(tgt)}
        D = np.zeros((len(tgt), len(src)))
        for j, (sigma, tau, p, q) in enumerate(src):
            for (sigma_new, sign) in cochain_d_DT(p, sigma):
                if sigma_new & tau and (sigma_new, tau) in tgt_idx:
                    D[tgt_idx[(sigma_new, tau)], j] += sign
            for (tau_new, sign) in cochain_d_PE(q, tau):
                if sigma & tau_new and (sigma, tau_new) in tgt_idx:
                    D[tgt_idx[(sigma, tau_new)], j] += ((-1) ** p) * sign
        diffs[n] = D

    # Πίνακας διαστάσεων ἀνὰ bidegree
    cells_per_pq = {(p, q): len(pairs_by_pq[(p, q)]) for p in range(3) for q in range(3)}

    return {
        'basis': basis,
        'd': diffs,
        'dim': {n: len(basis[n]) for n in range(5)},
        'cells_per_pq': cells_per_pq,
    }


def wu_laplacians(bicx):
    """Δ_n^Wu = d_{n-1} d_{n-1}^T + d_n^T d_n (Hodge)."""
    laps = {}
    for n in range(5):
        dim_n = bicx['dim'][n]
        L = np.zeros((dim_n, dim_n))
        if n - 1 >= 0:
            d_prev = bicx['d'][n-1]
            if d_prev.size > 0:
                L += d_prev @ d_prev.T
        if n <= 3:
            d_curr = bicx['d'][n]
            if d_curr.size > 0:
                L += d_curr.T @ d_curr
        laps[n] = L
    return laps


def wu_projector_top(bicx, laps=None, tol=1e-8):
    """P_4^Wu : Tot^4 → ker(Δ_4^Wu). Ἐπιστρέφει (P, ground_basis, eigvals)."""
    if laps is None:
        laps = wu_laplacians(bicx)
    L4 = laps[4]
    eigvals, eigvecs = eigh(L4)
    zero_mask = np.abs(eigvals) < tol
    ground_basis = eigvecs[:, zero_mask]
    P = ground_basis @ ground_basis.T
    return P, ground_basis, eigvals


def wu_sector_decomposition(bicx, dt):
    """Ἀντιστοίχισι κάθε Tot^4 basis element σὲ μία β₁₀ κορυφή.

    Π-28-Β05/Β06: γιὰ κάθε (σ, τ) μὲ p+q=4 ⇒ σ ∩ τ = {v}, μοναδικὴ v ∈ β₁₀.
    """
    beta10 = dt['orbits']['β10']
    sectors = []
    for (sigma, tau, p, q) in bicx['basis'][4]:
        common = sigma & tau
        assert len(common) == 1, f"intersection size {len(common)} ≠ 1"
        v = next(iter(common))
        assert v in beta10, f"common vertex {v} not in β10"
        sectors.append(v)
    sector_indices = {v: [i for i, s in enumerate(sectors) if s == v] for v in beta10}
    return sectors, sector_indices


# ═══════════════════════════════════════════════════════════════════
# Δ. Structural Wu diagnostic (Π-28-Β05..Β10)
# ═══════════════════════════════════════════════════════════════════

def verify_wu_structure(dt, pe, tol_block=1e-12, tol_zero=1e-8):
    """Πλήρης structural ἔλεγχος τοῦ Wu bicomplex.

    Ἐπιστρέφει dict μὲ:
      cells_per_pq         — {(p,q) → int}, ἀναμενόμενο σύνολο 2772 (Π-28-Β08)
      total_cells          — int = 2772
      euler_alternating_sum — int = 12 (Π-28-Β08, ταυτότητα μὲ |β₁₀|)
      tot_dims             — {n: |Tot^n|} γιὰ n=0..4
      d_squared_max_norm   — max_n ‖d_{n+1}·d_n‖, πρέπει ≈ 0 (bicomplex axiom)
      bettis               — [b_0, b_1, b_2, b_3, b_4], ἀναμενόμενο [0,0,0,0,12]
      top_betti_b4         — int = 12 (Π-28-Β06)
      block_diagonal_off_norm — max off-block στοιχεῖο τοῦ Δ_4, πρέπει < tol_block (Π-28-Β07)
      sector_count         — int = 12 (Π-28-Β05)
      per_sector_dim       — list 12 ἀκεραίων, ἕνας ἀνὰ β₁₀ κορυφή
      verdict              — str
    """
    bicx = wu_bicomplex(dt, pe)
    laps = wu_laplacians(bicx)
    P, ground_basis, eigvals = wu_projector_top(bicx, laps, tol=tol_zero)

    # d² = 0
    d_sq_norms = []
    for n in range(3):
        prod = bicx['d'][n+1] @ bicx['d'][n]
        d_sq_norms.append(float(np.linalg.norm(prod)))

    # Betti numbers
    bettis = []
    for n in range(5):
        dim_n = bicx['dim'][n]
        d_in = bicx['d'][n-1] if n >= 1 else None
        d_out = bicx['d'][n] if n <= 3 else None
        rk_in = int(matrix_rank(d_in, tol=tol_zero)) if d_in is not None and d_in.size > 0 else 0
        rk_out = int(matrix_rank(d_out, tol=tol_zero)) if d_out is not None and d_out.size > 0 else 0
        bettis.append(dim_n - rk_in - rk_out)

    # Sector decomposition
    sectors, sector_indices = wu_sector_decomposition(bicx, dt)
    per_sector_dim = [len(sector_indices[v]) for v in dt['orbits']['β10']]

    # Block-diagonal Δ_4 ἀκριβῶς (Π-28-Β07):
    # Στὸ Tot^4, ταξινομοῦμε τοὺς δείκτες ἀνὰ sector καὶ μετρᾶμε ‖Δ_4‖
    # off-block: στοιχεῖα (i, j) ὅπου i, j ζοῦν σὲ διαφορετικοὺς sectors.
    L4 = laps[4]
    off_max = 0.0
    for i in range(L4.shape[0]):
        for j in range(L4.shape[1]):
            if sectors[i] != sectors[j]:
                off_max = max(off_max, abs(L4[i, j]))

    # Euler ἐναλλασσόμενο σύνολο
    euler = sum(((-1) ** (p + q)) * bicx['cells_per_pq'][(p, q)]
                for p in range(3) for q in range(3))

    total = sum(bicx['cells_per_pq'].values())

    verdict_parts = []
    if total == 2772:
        verdict_parts.append('✓ 2772 cells (Π-28-Β08)')
    if euler == 12:
        verdict_parts.append('✓ Σ(-1)^(p+q)·|C^(p,q)| = 12')
    if bettis == [0, 0, 0, 0, 12]:
        verdict_parts.append('✓ Wu concentration b_4 = 12 (Π-28-Β06)')
    if off_max < tol_block:
        verdict_parts.append(f'✓ Δ_4 block-diagonal (off ≤ {off_max:.2e} < {tol_block:.0e}, Π-28-Β07)')
    if len(set(sectors)) == 12:
        verdict_parts.append('✓ 12 ἀνεξάρτητα sectors (Π-28-Β05)')

    return {
        'cells_per_pq': {f'{p},{q}': v for (p, q), v in bicx['cells_per_pq'].items()},
        'total_cells': total,
        'euler_alternating_sum': euler,
        'tot_dims': bicx['dim'],
        'd_squared_max_norm': float(max(d_sq_norms)),
        'bettis': bettis,
        'top_betti_b4': bettis[4],
        'block_diagonal_off_norm': off_max,
        'sector_count': len(set(sectors)),
        'per_sector_dim': per_sector_dim,
        'verdict': '  '.join(verdict_parts) if verdict_parts else 'Wu structure failed',
    }
