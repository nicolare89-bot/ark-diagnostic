"""
ark_local_b10.py — Τοπικὴ διάγνωσι ἀνὰ β₁₀ κορυφή (ΟΧΙ "Wu")
=============================================================

ΠΡΟΣΟΧΗ: Αὐτὴ ἡ διάγνωσι ΔΕΝ εἶναι ἡ Wu cohomology. Εἶναι μία ἁπλὴ
τοπικὴ μέτρησι ποὺ συσχετίζει ἕνα β₄-input vector μὲ τὶς 12 β₁₀
κορυφὲς μέσῳ τῆς γεωμετρικῆς γειτονιᾶς στὸ DT.

Γιὰ κάθε β₁₀ κορυφή v, ὑπολογίζει "local signal":

    local_signal[v] = Σ_{m ∈ neighbors_β₄(v)} |input_vec[m]|²

ὅπου neighbors_β₄(v) εἶναι οἱ 5 β₄ midpoints ποὺ γειτονεύουν μὲ τὴν v
στὸ DT graph (= midpoints τῶν 5 icosa ἀκμῶν ποὺ συγκλίνουν στὴν v).

Γιατί ΟΧΙ Wu (Π32-ARK.09):
  Ἡ Wu cohomology εἶναι structural ἰδιότητα τοῦ ζεύγους (DT,ΠΕ), καὶ
  ὁποιοδήποτε β₄-based embedding προβάλλεται σὲ 0 ἁρμονικό (Π32-WU-EMBED.03).
  Ἀντίθετα, αὐτὴ ἡ τοπικὴ μέτρησι εἶναι ξεκάθαρα input-driven καὶ
  παράγει 12 μὴ-μηδενικὰ scalars ὅταν τὸ input εἶναι μὴ-μηδέν.

Στάδιο 2.11 (διαφορετικὸ ἀπὸ τοὺς 6 ἀρχικοὺς στόχους — προέκτασι).
"""

import numpy as np


def beta10_to_beta4_neighbors(dt):
    """Γιὰ κάθε β₁₀ κορυφή, βρίσκει τοὺς 5 γείτονες β₄ στὸ DT.

    Returns:
        neighbors : dict[β₁₀_vertex → list of β₄ vertex indices]
    """
    beta4_set = set(dt['orbits']['β4'])
    beta10 = dt['orbits']['β10']
    neighbors = {v: [] for v in beta10}
    for e in dt['edges']:
        a, b = sorted(e) if isinstance(e, frozenset) else e
        if a in neighbors and b in beta4_set:
            neighbors[a].append(b)
        elif b in neighbors and a in beta4_set:
            neighbors[b].append(a)
    # Κάθε β₁₀ ἔχει ἀκριβῶς 5 β₄ γείτονες (5 icosa ἀκμές)
    for v, lst in neighbors.items():
        assert len(lst) == 5, f"β₁₀ {v} has {len(lst)} β₄ neighbors"
    return neighbors


def diagnose_local_beta10_neighborhood(input_vec, dt, neighbors=None):
    """Τοπικὴ διάγνωσι ἀνὰ β₁₀ κορυφή.

    Args:
        input_vec : ndarray (30,) — διάνυσμα στὸ β₄ space
        dt        : DT graph
        neighbors : optional cache ἀπὸ beta10_to_beta4_neighbors

    Returns dict μὲ:
        local_signal       : dict[β₁₀ → float] (12 entries)
        total_signal       : float = Σ_v local_signal[v]
        global_energy      : float = ‖input_vec‖²
        dominant_beta10    : β₁₀ vertex μὲ μεγαλύτερο σῆμα
        anisotropy         : max − min local_signal
        anisotropy_pct     : 100 · anisotropy / total_signal (ἂν >0)
        coverage_redundancy : Σ_v |neighbors(v)| / 30 (= 2: κάθε β₄ ἔχει 2 β₁₀ γείτονες)
    """
    if input_vec.shape != (30,):
        raise ValueError(f"input_vec shape {input_vec.shape}, expected (30,)")
    if neighbors is None:
        neighbors = beta10_to_beta4_neighbors(dt)

    beta4 = dt['orbits']['β4']
    # Map ἀπὸ β₄ vertex index σὲ θέσι (0..29) στὸ input_vec
    b4_pos = {v: i for i, v in enumerate(beta4)}

    energies_sq = np.abs(input_vec) ** 2

    local_signal = {}
    for v_b10, neigh_list in neighbors.items():
        sig = sum(energies_sq[b4_pos[m]] for m in neigh_list)
        local_signal[v_b10] = float(sig)

    total_signal = float(sum(local_signal.values()))
    global_energy = float(np.sum(energies_sq))

    sig_values = list(local_signal.values())
    dom = max(local_signal, key=local_signal.get)
    aniso = float(max(sig_values) - min(sig_values))
    aniso_pct = (100.0 * aniso / total_signal) if total_signal > 1e-15 else 0.0

    coverage = sum(len(n) for n in neighbors.values()) / 30

    return {
        'local_signal': local_signal,
        'total_signal': total_signal,
        'global_energy': global_energy,
        'dominant_beta10': dom,
        'anisotropy': aniso,
        'anisotropy_pct': aniso_pct,
        'coverage_redundancy': float(coverage),
    }
