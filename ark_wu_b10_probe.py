"""
ark_wu_b10_probe.py — β₁₀-probe (Π32-WU-PROBE.01)
==================================================

Ὑλοποίησι τοῦ β₁₀-probe: γιὰ κάθε ζεῦγος (v, w) τῶν 12 β₁₀ κορυφῶν,

    Tr(P_w_sec · ρ_v_ground) = δ_{v,w}    ἀκριβῶς

ὅπου:
    ρ_v_ground = ground state τοῦ Δ_4 sector v (1D ἁρμονικὸ state)
    P_w_sec    = diagonal indicator projector στὸν sector w τοῦ Tot^4

Ἀφοῦ Δ_4 εἶναι block-diagonal (Π-28-Β07), κάθε ground state ζεῖ
ἀκριβῶς σὲ ἕναν sector. Ἑπομένως ἡ matrix τῶν traces εἶναι 12×12
ταυτοτικός μὲ machine-epsilon off-diagonal entries.

Σὲ ἀντίθεσι μὲ τὴν Π32-WU-EMBED.03 (ὁποιοδήποτε β₄-based embedding
εἶναι coboundary, harmonic fraction = 0), τὸ b10-probe ξεκινᾶ ΑΠΕΥΘΕΙΑΣ
ἀπὸ τὸν Tot⁴ ground state — ἑπομένως δίνει μὴ-μηδενικὸ καθαρὸ
σῆμα ἀνὰ sector.

Στάδιο 2.9 (στόχος #5 follow-up).
"""

import numpy as np
from numpy.linalg import eigh

from ark_wu import (
    build_PE, wu_bicomplex, wu_laplacians,
    wu_sector_decomposition,
)


def wu_ground_basis_per_sector(dt, pe=None, bicx=None, tol=1e-8):
    """Βρίσκει 12 ground vectors τοῦ Δ_4, ἕνα ἀνὰ β₁₀ sector.

    Ἐκμεταλλεύεται τὸ Π-28-Β07 block-diagonal Δ_4: διαγωνίζει κάθε 50×50
    sub-block ἀνεξάρτητα, παίρνει τὸ μοναδικὸ eigenvector μὲ eigenvalue ≈ 0,
    καὶ ἁπλώνει σὲ 600D μὲ μηδενικὰ ἔξω ἀπὸ τὸν sector.

    Returns:
        ground_per_sector: dict[β₁₀_vertex → ndarray (600,)]
    """
    if pe is None:
        pe = build_PE(dt)
    if bicx is None:
        bicx = wu_bicomplex(dt, pe)
    laps = wu_laplacians(bicx)
    L4 = laps[4]
    sectors, sector_indices = wu_sector_decomposition(bicx, dt)
    dim4 = L4.shape[0]

    ground_per_sector = {}
    for v, idxs in sector_indices.items():
        idxs_arr = np.array(idxs)
        # 50×50 sub-block γιὰ τὸν sector v
        block = L4[np.ix_(idxs_arr, idxs_arr)]
        eigvals, eigvecs = eigh(block)
        # Ἕνα μοναδικὸ ground state ἀνὰ sector (Π-28-Β06: b_4 = 12 ⟹ 1 ἀνὰ sector)
        zero_mask = np.abs(eigvals) < tol
        n_zero = int(np.sum(zero_mask))
        assert n_zero == 1, f"sector {v}: {n_zero} ground states (expected 1)"
        ground_local = eigvecs[:, zero_mask][:, 0]  # 50D
        # Ἁπλώστε σὲ 600D
        ground_full = np.zeros(dim4)
        ground_full[idxs_arr] = ground_local
        ground_per_sector[v] = ground_full
    return ground_per_sector


def b10_probe_matrix(dt, pe=None, bicx=None, ground_per_sector=None):
    """Ὑπολογίζει τὸν 12×12 πίνακα M[v, w] = Tr(P_w_sec · ρ_v_ground).

    Δίνει ταυτοτικό 12×12 ἂν ἡ Wu concentration ἰσχύει (Π32-WU-PROBE.01).

    Returns:
        M: ndarray (12, 12) — γραμμές=v, στῆλες=w
        beta10_order: list τῶν 12 β₁₀ vertices στὴν σειρὰ τῶν γραμμῶν/στηλῶν
    """
    if pe is None:
        pe = build_PE(dt)
    if bicx is None:
        bicx = wu_bicomplex(dt, pe)
    if ground_per_sector is None:
        ground_per_sector = wu_ground_basis_per_sector(dt, pe, bicx)

    sectors, sector_indices = wu_sector_decomposition(bicx, dt)
    beta10_order = list(dt['orbits']['β10'])

    M = np.zeros((12, 12))
    for i, v in enumerate(beta10_order):
        ground_v = ground_per_sector[v]
        for j, w in enumerate(beta10_order):
            idxs_w = np.array(sector_indices[w])
            # Tr(P_w_sec · ρ_v) = ⟨ground_v | P_w_sec | ground_v⟩
            #                  = Σ_{k ∈ idxs_w} |ground_v[k]|²
            M[i, j] = float(np.sum(ground_v[idxs_w] ** 2))
    return M, beta10_order


def verify_b10_probe(dt, pe=None, tol_off=1e-12, tol_diag=1e-9):
    """Πλήρης διαγνωστικὴ τοῦ Π32-WU-PROBE.01.

    Returns dict μὲ:
        matrix              — 12×12 πίνακας traces
        beta10_order        — σειρὰ vertices στὶς γραμμὲς/στῆλες
        diag_min            — min(diag(M)), ἀναμενόμενο ≈ 1
        diag_max            — max(diag(M))
        diag_max_dev_from_1 — max|diag − 1|, πρέπει < tol_diag
        off_max             — max|off-diagonal entry|, πρέπει < tol_off
        identity_match      — bool: εἶναι ἀκριβῶς ταυτοτικός;
        verdict             — str
    """
    if pe is None:
        pe = build_PE(dt)
    M, beta10_order = b10_probe_matrix(dt, pe)

    diag = np.diag(M)
    diag_max_dev = float(np.max(np.abs(diag - 1)))
    diag_min = float(np.min(diag))
    diag_max = float(np.max(diag))

    off_mask = ~np.eye(12, dtype=bool)
    off_max = float(np.max(np.abs(M[off_mask]))) if np.any(off_mask) else 0.0

    identity_match = (diag_max_dev < tol_diag) and (off_max < tol_off)

    if identity_match:
        verdict = (f'✓ Π32-WU-PROBE.01: 12×12 ταυτοτικός '
                   f'(diag dev ≤ {diag_max_dev:.2e}, off ≤ {off_max:.2e})')
    else:
        verdict = (f'⚠ probe failed: diag dev = {diag_max_dev:.2e}, '
                   f'off = {off_max:.2e}')

    return {
        'matrix': M,
        'beta10_order': beta10_order,
        'diag_min': diag_min,
        'diag_max': diag_max,
        'diag_max_dev_from_1': diag_max_dev,
        'off_max': off_max,
        'identity_match': bool(identity_match),
        'verdict': verdict,
    }
