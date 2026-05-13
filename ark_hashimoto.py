"""
ark_hashimoto.py — Hashimoto non-backtracking matrix
=====================================================

Γιὰ ἕνα γράφο G μὲ |V| κορυφὲς καὶ |E| ἀκμές, ὁρίζουμε 2|E| directed
edges (κάθε ἀκμὴ (u,v) δίνει δύο directed: u→v καὶ v→u).

Ὁ Hashimoto B-matrix εἶναι 2|E| × 2|E|:

    B[(u→v), (v'→w)] = 1   ἂν v = v' ΚΑΙ u ≠ w
                       0   ἀλλιῶς

Δηλαδὴ ἕνα non-backtracking βῆμα: ἂν ἔρχομαι ἀπὸ τὸ u στὸ v, μπορῶ νὰ
πάω σὲ ὁποιοδήποτε γείτονα τοῦ v ΕΚΤΟΣ τοῦ u.

Ihara ταυτότητα (Bass 1992):
    det(I − Bx) = (1 − x²)^{|E|−|V|} · det(I − Ax + (D − I)x²)

ὅπου A = adjacency matrix, D = degree matrix.

Σπεκτρικὲς ἰδιότητες:
    ρ(B) = spectral radius — non-backtracking growth rate
    Στατιστικὲς συνδέσεις μὲ Cheeger constant καὶ mixing time.
    Γιὰ regular γράφο degree d: ρ(B) ≈ d − 1.
    Γιὰ DT (μέσος βαθμὸς ≈ 5.81), ρ(B) ≈ 4-5.

Στάδιο 2.10 (#6 ἀπὸ τοὺς στόχους Σταδίου 2 στὸ CLAUDE.md).
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# Α. Directed edges + B matrix
# ═══════════════════════════════════════════════════════════════════

def build_directed_edges(graph):
    """Ἐπιστρέφει list[(u, v)] μήκους 2|E|: γιὰ κάθε frozenset({u,v})
    στὸ graph['edges'], περιλαμβάνει καὶ (u,v) καὶ (v,u).

    Ἡ σειρὰ καθορίζει τὴν ἀρίθμησι τῶν γραμμῶν/στηλῶν τοῦ B.
    """
    directed = []
    for e in graph['edges']:
        if isinstance(e, frozenset):
            u, v = sorted(e)
        else:
            u, v = e
        directed.append((u, v))
        directed.append((v, u))
    return directed


def build_hashimoto_B(graph, directed_edges=None):
    """Hashimoto non-backtracking matrix B ∈ ℝ^{2|E| × 2|E|}.

    B[(u→v), (v'→w)] = 1 ἂν v = v' ΚΑΙ u ≠ w, ἀλλιῶς 0.
    """
    if directed_edges is None:
        directed_edges = build_directed_edges(graph)
    n_de = len(directed_edges)
    # Lookup ἀπὸ "head" κορυφή σὲ list δεικτῶν directed edges ποὺ τὴν ἔχουν tail
    out_from = {v: [] for v in graph['vertices']} if 'vertices' in graph else {}
    if not out_from:
        # Ἂν δὲν ὑπάρχει vertices key, μάζεψε ἀπὸ τὰ edges
        all_v = set()
        for u, v in directed_edges:
            all_v.add(u); all_v.add(v)
        out_from = {v: [] for v in all_v}
    for idx, (u, v) in enumerate(directed_edges):
        out_from[u].append(idx)  # idx ξεκινᾶ ἀπὸ τὸ u

    B = np.zeros((n_de, n_de))
    for j, (u_j, v_j) in enumerate(directed_edges):
        # Διαβαίνουμε ἀπὸ (u_j → v_j). Στὸ ἑπόμενο βῆμα ξεκινᾶμε ἀπὸ τὸ v_j.
        for i in out_from[v_j]:  # i ξεκινᾶ ἀπὸ τὸ v_j
            v_i, w_i = directed_edges[i]  # = (v_j, w_i)
            if w_i != u_j:  # non-backtracking
                B[i, j] = 1
    return B


# ═══════════════════════════════════════════════════════════════════
# Β. Ihara ταυτότητα
# ═══════════════════════════════════════════════════════════════════

def adjacency_matrix(graph):
    """Adjacency matrix A καὶ vertex_index map."""
    if 'vertices' in graph:
        verts = list(graph['vertices'])
    else:
        verts = sorted({v for e in graph['edges'] for v in
                        (sorted(e) if isinstance(e, frozenset) else e)})
    idx = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    A = np.zeros((n, n))
    for e in graph['edges']:
        u, v = sorted(e) if isinstance(e, frozenset) else e
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    return A, verts


def degree_matrix(graph):
    """Degree matrix D (diagonal) ταξινομημένος ὅπως ὁ A."""
    A, _ = adjacency_matrix(graph)
    return np.diag(A.sum(axis=1))


def verify_ihara_identity(graph, x=0.1, B=None):
    """Ἐλέγχει τὴν Ihara ταυτότητα σὲ συγκεκριμένο x.

    LHS: det(I − Bx)
    RHS: (1 − x²)^{|E|−|V|} · det(I − Ax + (D − I)x²)

    Ἐπιστρέφει dict μὲ lhs, rhs, abs_diff, rel_diff.
    """
    if B is None:
        B = build_hashimoto_B(graph)
    A, verts = adjacency_matrix(graph)
    D = degree_matrix(graph)
    n = A.shape[0]
    n_e = graph['E']

    lhs = float(np.linalg.det(np.eye(2 * n_e) - x * B))
    M = np.eye(n) - x * A + (D - np.eye(n)) * x * x
    rhs_det = float(np.linalg.det(M))
    rhs = ((1 - x * x) ** (n_e - n)) * rhs_det
    abs_diff = abs(lhs - rhs)
    rel_diff = abs_diff / max(abs(lhs), abs(rhs), 1e-300)
    return {
        'x': x,
        'lhs': lhs,
        'rhs': rhs,
        'abs_diff': abs_diff,
        'rel_diff': rel_diff,
    }


# ═══════════════════════════════════════════════════════════════════
# Γ. Spectrum + διαγνωστική
# ═══════════════════════════════════════════════════════════════════

def hashimoto_spectrum(graph, B=None):
    """Eigenvalues τοῦ B (general complex spectrum)."""
    if B is None:
        B = build_hashimoto_B(graph)
    return np.linalg.eigvals(B)


def diagnose_hashimoto(graph, ihara_x=0.1, ihara_tol=1e-6):
    """Πλήρης Hashimoto διάγνωσι.

    Returns:
        graph_name, V, E, n_directed_edges (= 2E)
        B_shape : (2E, 2E)
        spectral_radius : max|λ| τοῦ B
        spectral_gap    : ρ(B) − |second largest eigenvalue|
        n_real_eigvals  : πόσα eigenvalues εἶναι real (μέσα σὲ 1e-10)
        ihara          : dict ἀπὸ verify_ihara_identity
        ihara_holds    : bool ἂν rel_diff < ihara_tol
        verdict        : str
    """
    B = build_hashimoto_B(graph)
    eigvals = hashimoto_spectrum(graph, B)
    abs_eigs = np.abs(eigvals)
    sorted_abs = np.sort(abs_eigs)[::-1]
    rho = float(sorted_abs[0])
    gap = float(rho - sorted_abs[1]) if len(sorted_abs) > 1 else float(rho)
    n_real = int(np.sum(np.abs(eigvals.imag) < 1e-10))

    ihara = verify_ihara_identity(graph, x=ihara_x, B=B)
    ihara_holds = ihara['rel_diff'] < ihara_tol

    name = graph.get('name', 'graph')
    parts = []
    parts.append(f'ρ(B) = {rho:.4f}')
    parts.append(f'gap = {gap:.4f}')
    parts.append(f'real eigvals: {n_real}/{len(eigvals)}')
    if ihara_holds:
        parts.append(f"✓ Ihara at x={ihara_x} (rel_err={ihara['rel_diff']:.2e})")
    else:
        parts.append(f"⚠ Ihara off (rel_err={ihara['rel_diff']:.2e})")
    verdict = '  |  '.join(parts)

    return {
        'graph_name': name,
        'V': graph['V'],
        'E': graph['E'],
        'n_directed_edges': 2 * graph['E'],
        'B_shape': B.shape,
        'spectral_radius': rho,
        'spectral_gap': gap,
        'n_real_eigvals': n_real,
        'ihara': ihara,
        'ihara_holds': bool(ihara_holds),
        'verdict': verdict,
    }
