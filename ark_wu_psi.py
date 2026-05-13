"""
ark_wu_psi.py — ψ-twisted Wu διάγνωσι ἀνὰ β₁₀
==============================================

ψ-twisted embedding ℝ³⁰ → ℂ⁶⁰⁰ μέσῳ ψ-coefficients ποὺ ζοῦν στὶς 12
β₁₀ κορυφές. Τὸ embedding (β₄-based) εἶναι ΠΑΝΤΑ coboundary κατὰ τὸν
Π32-WU-EMBED.03 (6★ ΔΙΑΜ): γιὰ ὁποιοδήποτε ψ, P₄·E_twist·v ≈ 0.

Ἑπομένως πρέπει νὰ διαχωριστοῦν δύο ὀρθογώνια ἐρωτήματα:
  - P₄ · v          → ποσὸ ἁρμονικότητας (= 0 — Π32-WU-EMBED.03)
  - P_v_sec · v     → ποσὸ ζωῆς στὸν sector v (μὴ-μηδέν, χρήσιμο)

ψ-ἀναθέσεις (Iₕ-equivariant only):
  'trivial'    : ψ(v) = 1+0j (control)
  'irrep_T1_x' : ψ(v) = coords[v][0] / ‖coords[v]‖ (T₁ x-component)

Παραλείπονται ρητῶς:
  (β) index mod 4 — μὴ-equivariant, ἐξαρτᾶται ἀπὸ τυχαία ἀρίθμησι
  (γ) F-functor → V₄ — προγραμματισμένο γιὰ Ω.WU-EMBED.01.γ ξεχωριστά

Στάδιο 2.9 (στόχος #5 follow-up).
"""

import numpy as np

from ark_wu import (
    build_PE, wu_bicomplex, wu_laplacians, wu_projector_top,
    wu_sector_decomposition,
)


# ═══════════════════════════════════════════════════════════════════
# Α. ψ-ἀναθέσεις στὶς 12 β₁₀ κορυφές
# ═══════════════════════════════════════════════════════════════════

def _psi_trivial(dt):
    """Trivial ψ: ψ(v) = 1+0j γιὰ κάθε β₁₀ κορυφή. Control case."""
    return {v: complex(1.0, 0.0) for v in dt['orbits']['β10']}


def _psi_irrep_T1_x(dt):
    """T₁ x-component: ψ(v) = x-συντεταγμένη τῆς v (unit-norm).

    Iₕ-equivariant by construction (οἱ x,y,z μετασχηματίζονται ὡς T₁ standard
    rep τῆς I). Ἐπιστρέφει complex (ὁ φανταστικὸς εἶναι 0 γιὰ τὸ x μόνο).
    """
    coords = dt['coords']
    return {v: complex(float(coords[v][0]), 0.0) for v in dt['orbits']['β10']}


def psi_assignments(dt):
    """Ἐπιστρέφει dict ὀνόματος → ψ-ἀνάθεσι. Ἐπεκτεῖνε ἐδῶ νέες παραλλαγές."""
    return {
        'trivial':    _psi_trivial(dt),
        'irrep_T1_x': _psi_irrep_T1_x(dt),
    }


# ═══════════════════════════════════════════════════════════════════
# Β. ψ-twisted embedding ℝ³⁰ → ℂ⁶⁰⁰
# ═══════════════════════════════════════════════════════════════════

def build_psi_twist_embedding(dt, bicx, psi_assign, pe=None):
    """E_twist : ℂ^(600 × 30) μὲ E_twist[idx, i] = ψ(β₁₀_v(σ)) ἂν β₄[i] ∈ σ.

    Iterate over τὰ 30 β₄ midpoints (στῆλες) καὶ τὰ 600 C^(2,2) ζεύγη
    (γραμμές). Γιὰ κάθε ζεῦγος (σ, τ), ἂν τὸ β₄ midpoint ζεῖ στὸν
    DT face σ (= τρίγωνο β₁₀-β₆-β₄), τότε ἡ τιμὴ εἶναι ψ τῆς β₁₀
    κορυφῆς τοῦ σ.
    """
    beta4 = dt['orbits']['β4']
    beta10_set = set(dt['orbits']['β10'])
    C22_pairs = [(sigma, tau) for (sigma, tau, _, _) in bicx['basis'][4]]
    dim4 = len(C22_pairs)
    assert dim4 == 600, f"|C^(2,2)| = {dim4}, expected 600"

    E_twist = np.zeros((dim4, 30), dtype=complex)
    for i, m in enumerate(beta4):
        for idx_pair, (sigma, tau) in enumerate(C22_pairs):
            if m in sigma:
                v_b10_list = [v for v in sigma if v in beta10_set]
                if not v_b10_list:
                    continue
                v_b10 = v_b10_list[0]
                E_twist[idx_pair, i] = psi_assign[v_b10]
    return E_twist


# ═══════════════════════════════════════════════════════════════════
# Γ. ψ-twisted διάγνωσι ἀνὰ sector
# ═══════════════════════════════════════════════════════════════════

def diagnose_wu_psi_per_beta10(input_vec, dt, psi_assign,
                                pe=None, bicx=None, P4=None,
                                E_twist=None):
    """ψ-twisted Wu διάγνωσι ἀνὰ β₁₀ sector.

    Args:
        input_vec : ndarray (30,) — διάνυσμα στὸ β₄ space
        dt        : DT graph (build_DT())
        psi_assign: dict[β₁₀_vertex → complex]
        pe, bicx, P4, E_twist: optional caches

    Returns dict μὲ:
        embedded_vec_norm : float, ‖E_twist @ input_vec‖
        total_energy      : float, Σ_v sector_energy[v]
        sector_energy     : dict[β₁₀ → float], ‖embedded[sector_idx_v]‖²
        sector_amplitude  : dict[β₁₀ → float], |Σ embedded[sector_idx_v]|
        sector_phase      : dict[β₁₀ → float], arg(Σ embedded[sector_idx_v])
        dominant_sector   : β₁₀ vertex μὲ μεγαλύτερη energy
        anisotropy        : max - min sector_energy
        harmonic_total    : float, ‖P₄ · embedded‖² (≈ 0 — Π32-WU-EMBED.03)
        harmonic_density  : dict[β₁₀ → float],
                            ‖P₄·embedded[idx_v]‖² / sector_energy[v]
    """
    if input_vec.shape != (30,):
        raise ValueError(f"input_vec shape {input_vec.shape}, expected (30,)")
    if pe is None:
        pe = build_PE(dt)
    if bicx is None:
        bicx = wu_bicomplex(dt, pe)
    if E_twist is None:
        E_twist = build_psi_twist_embedding(dt, bicx, psi_assign, pe=pe)
    if P4 is None:
        laps = wu_laplacians(bicx)
        P4, _, _ = wu_projector_top(bicx, laps)

    embedded = E_twist @ input_vec.astype(complex)  # ℂ^600
    _, sector_indices = wu_sector_decomposition(bicx, dt)
    beta10 = list(dt['orbits']['β10'])

    sector_energy = {}
    sector_amplitude = {}
    sector_phase = {}
    for v in beta10:
        idxs = np.array(sector_indices[v])
        sec = embedded[idxs]
        sector_energy[v] = float(np.sum(np.abs(sec) ** 2))
        s = np.sum(sec)
        sector_amplitude[v] = float(abs(s))
        sector_phase[v] = float(np.angle(s)) if abs(s) > 1e-15 else 0.0

    total_energy = float(sum(sector_energy.values()))

    # Harmonic projection μέσῳ P₄ (real symmetric, ἑπομένως real & imag ξεχωριστά)
    harmonic = (P4 @ embedded.real) + 1j * (P4 @ embedded.imag)
    harmonic_total = float(np.sum(np.abs(harmonic) ** 2))

    harmonic_density = {}
    for v in beta10:
        idxs = np.array(sector_indices[v])
        h = float(np.sum(np.abs(harmonic[idxs]) ** 2))
        e = sector_energy[v]
        harmonic_density[v] = (h / e) if e > 1e-15 else 0.0

    energies = list(sector_energy.values())
    dominant_v = max(sector_energy, key=sector_energy.get)
    anisotropy = float(max(energies) - min(energies))

    return {
        'embedded_vec_norm': float(np.linalg.norm(embedded)),
        'total_energy': total_energy,
        'sector_energy': sector_energy,
        'sector_amplitude': sector_amplitude,
        'sector_phase': sector_phase,
        'dominant_sector': dominant_v,
        'anisotropy': anisotropy,
        'harmonic_total': harmonic_total,
        'harmonic_density': harmonic_density,
    }
