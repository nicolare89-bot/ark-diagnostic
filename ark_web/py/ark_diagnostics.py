"""
ark_diagnostics.py — Διαγνωστικὲς συναρτήσεις + testbeds
=========================================================

Οἱ 4 διαγνωστικές:
  [1] Iₕ-energy distribution (β₄ orbit, 30D)
  [2] Cheeger threshold (|β₁₀|=12 critical edges)
  [3] ε-drift (ζωντανὴ ἀτέλεια vs ὑπερβολή)
  [4] F-balance (Euler χ διατήρησι)

Testbeds:
  - random_isotropic (Gaussian, Iₕ-isotropic στὸ μέσο ὅρο)
  - ih_aligned (καθαρὰ σὲ ἕνα Iₕ-irrep)
  - asymmetric (5 spikes)

Στάδιο 2 (split ἀπὸ ARK_DIAGNOSTIC_v0.py).
"""

import numpy as np
from numpy.linalg import norm

from ark_geometry import BETA10, EPSILON, EPSILON_NOMINAL, graph_laplacian
from ark_irreps import RANKS_BETA4


# ═══════════════════════════════════════════════════════════════════
# Σταθερὲς ζωνῶν (multipliers τοῦ ε* = EPSILON)
# ═══════════════════════════════════════════════════════════════════

ZONE_VIABLE_K   = 1     # ε ≤ 1·ε*  : βιώσιμη ἰσορροπία
ZONE_TOLERANT_K = 5     # ε ≤ 5·ε*  : ἀνεκτικὴ παραμόρφωσι
ZONE_CHEEGER_K  = 10    # ε ≤ 10·ε* : ἐμφανὴς κάμψι (Π32-OIK Cheeger hyperinflation)
ZONE_HALF_K     = 43    # ε ≤ 43·ε* : σοβαρὴ κάμψι (half-collapse στὸ 43·ε*, ε² = 5/12)
                        # ε > 43·ε* : κατάρρευσι (T₁ μειοψηφία)

# Iₕ-isotropic floor: στὸ ε → ∞, E_T₁ → 3/30 = 10% (θερμοδυναμικὴ ἰσορροπία)
IH_ISOTROPIC_FLOOR_T1 = 3 / 30           # 0.1
IH_ISOTROPIC_FLOOR_LEAKAGE = 27 / 30     # 0.9 = 1 - 3/30


def _implied_epsilon_from_leakage(leakage):
    """Ἀντιστρέφει τὸν Schur-Gauss τύπο γιὰ T₁ base + Gaussian noise.

        leakage = 27ε² / (10 + 30ε²)
        ⇒ ε² = 10·leakage / (27 − 30·leakage)
        ⇒ ε = √(10·leakage / (27 − 30·leakage))

    Στὸ leakage = 27/30 (Iₕ-isotropic floor) ἢ πέρα → ∞.
    """
    if leakage <= 0:
        return 0.0
    if leakage >= IH_ISOTROPIC_FLOOR_LEAKAGE - 1e-12:
        return float('inf')
    return float(np.sqrt(10 * leakage / (27 - 30 * leakage)))


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
    """Διάγνωσι 3: ε-drift μὲ ἀκριβὲς threshold ε* = EPSILON, 5 ζῶνες.

    Δύο modes:
      "leakage"   — ὅταν ὑπάρχει dominant irrep (energy > 0.5):
                    leakage := 1 − E_dominant
                    epsilon_implied := √(10·leakage / (27 − 30·leakage))
                    drift := epsilon_implied (ἀντιστρέφει τὸν Schur-Gauss τύπο)
      "deviation" — γενικὴ περίπτωσι: drift := max_deviation_pct / 100.

    Ζῶνες (ε* = EPSILON ≈ 0.01501):
      ε ≤ ε*           → ζώνη βιώσιμης ἰσορροπίας
      ε ≤ 5·ε*         → ζώνη ἀνεκτικῆς παραμόρφωσης
      ε ≤ 10·ε*        → ζώνη ἐμφανοῦς κάμψης (Cheeger hyperinflation)
      ε ≤ 43·ε*        → ζώνη σοβαρῆς κάμψης (half-collapse στὸ 43·ε*)
      ε > 43·ε*        → ζώνη κατάρρευσης (T₁ μειοψηφία)

    Iₕ-isotropic floor: ε → ∞, E_T₁ → 3/30 = 10% — ἡ Κιβωτὸς ΠΟΤΕ δὲν χάνει
    ἐντελῶς τὴ μνήμη (θερμοδυναμικὴ ἰσορροπία, θερμικὸς θάνατος).

    Ἀκριβὴς Schur-Gauss τύπος (T₁ base + Iₕ-isotropic Gaussian noise):
      E_T₁(ε) = (10 + 3ε²) / (10 + 30ε²)
      leakage(ε) = 27ε² / (10 + 30ε²)
      half-collapse: leakage = E_T₁ = 50% στὸ ε² = 5/12 ⇒ ε = 43·ε*
      leakage_per_irrep(ρ, ε) ≈ rank(ρ)/27 · leakage(ε) γιὰ ρ ≠ T₁
    """
    energies = diagnose_result.get('energies', {})
    expected_iso = diagnose_result.get('expected_isotropic', {})
    max_dev_pct = diagnose_result.get('max_deviation_pct', 0)

    if energies:
        dominant = max(energies, key=energies.get)
        dominant_e = energies[dominant]
        # concentration = πόσο πάνω ἀπὸ τὴν Iₕ-isotropic τιμή
        # (στὸ random ≈ 0, στὸ καθαρὸ Iₕ-aligned ≈ 1 − expected)
        expected_dom = expected_iso.get(dominant, RANKS_BETA4.get(dominant, 0) / 30)
        concentration = dominant_e - expected_dom
    else:
        dominant = None
        dominant_e = 0.0
        concentration = 0.0

    # leakage mode: ὑπάρχει σαφὴς dominant irrep πάνω ἀπὸ Iₕ-isotropic baseline
    # (ὄχι ἁπλᾶ E > 0.5, ποὺ μπερδεύει random H-ranked μὲ Iₕ-aligned).
    # Στὸ Iₕ-isotropic floor (ε → ∞), concentration → 0 ⇒ deviation mode.
    if concentration > 0.2:
        leakage = max(0.0, 1.0 - dominant_e)
        eps_implied = _implied_epsilon_from_leakage(leakage)
        drift = eps_implied
        mode = 'leakage'
    else:
        leakage = None
        eps_implied = max_dev_pct / 100
        drift = eps_implied
        mode = 'deviation'

    eps_star = EPSILON
    if drift <= ZONE_VIABLE_K * eps_star:
        verdict = '✓ ζώνη βιώσιμης ἰσορροπίας (ε ≤ ε*)'
        cycles = float('inf')
    elif drift <= ZONE_TOLERANT_K * eps_star:
        verdict = '○ ζώνη ἀνεκτικῆς παραμόρφωσης (ε ≤ 5ε*)'
        cycles = round(66 * (1 - drift / (ZONE_TOLERANT_K * eps_star)), 1)
    elif drift <= ZONE_CHEEGER_K * eps_star:
        verdict = '⚠ ζώνη ἐμφανοῦς κάμψης — Cheeger (ε ≤ 10ε*)'
        cycles = round(66 * eps_star / drift, 1) if drift > 0 else float('inf')
    elif drift <= ZONE_HALF_K * eps_star:
        verdict = '⚠ ζώνη σοβαρῆς κάμψης (ε ≤ 43ε*, half-collapse στὸ 43ε*)'
        cycles = round(33 * eps_star / drift, 1) if drift > 0 else float('inf')
    elif drift < float('inf'):
        verdict = '⚠ ζώνη κατάρρευσης — T₁ μειοψηφία (ε > 43ε*)'
        cycles = round(11 * eps_star / drift, 1)
    else:
        verdict = '☠ Iₕ-isotropic floor — θερμοδυναμικὴ ἰσορροπία (E_T₁ → 3/30)'
        cycles = 0.0

    result = {
        'mode': mode,
        'dominant_irrep': dominant,
        'dominant_energy': float(dominant_e),
        'max_deviation_pct': max_dev_pct,
        'drift_fraction': float(drift) if drift != float('inf') else float('inf'),
        'epsilon_star': float(eps_star),
        'epsilon_star_pct': float(eps_star * 100),
        'epsilon_nominal_pct': EPSILON_NOMINAL * 100,
        'within_living_imperfection': bool(drift <= eps_star),
        'predicted_cycles_to_collapse': cycles,
        'verdict': verdict,
    }

    if mode == 'leakage':
        result['leakage_fraction'] = float(leakage)
        result['epsilon_implied'] = float(eps_implied) if eps_implied != float('inf') else float('inf')
        # Iₕ-isotropic floor flag
        if leakage >= IH_ISOTROPIC_FLOOR_LEAKAGE - 1e-6:
            result['ih_isotropic_floor'] = True
            result['floor_note'] = (
                f'E_T₁ ≈ 3/30 ({IH_ISOTROPIC_FLOOR_T1*100:.1f}%) — '
                'Iₕ-isotropic floor: θερμοδυναμικὴ ἰσορροπία (θερμικὸς θάνατος)'
            )
        else:
            result['ih_isotropic_floor'] = False

    if mode == 'leakage' and leakage is not None and leakage > 1e-9:
        ranks_others = {n: RANKS_BETA4[n] for n in RANKS_BETA4 if n != dominant}
        sum_ranks = sum(ranks_others.values())  # γιὰ T₁ dominant: 30−3 = 27
        per_irrep = []
        max_rel_err = 0.0
        for n, r in ranks_others.items():
            measured = float(energies.get(n, 0.0))
            predicted = float((r / sum_ranks) * leakage)
            rel_err = abs(measured - predicted) / predicted if predicted > 1e-9 else 0.0
            max_rel_err = max(max_rel_err, rel_err)
            per_irrep.append({
                'irrep': n,
                'rank': r,
                'measured': measured,
                'predicted_schur': predicted,
                'rel_err': float(rel_err),
            })
        schur_confirmed = max_rel_err < 0.05
        result['leakage_per_irrep'] = per_irrep
        result['leakage_max_rel_err'] = float(max_rel_err)
        result['schur_gauss_confirmed'] = bool(schur_confirmed)
        result['schur_verdict'] = (
            'Iₕ-isotropic noise → Schur-Gauss confirmed' if schur_confirmed
            else f'Schur-Gauss off (max rel_err={max_rel_err*100:.1f}%)'
        )

    return result


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


def testbed_dt_beta4_coordinates(dt_graph):
    """Οἱ x,y,z συντεταγμένες τῶν 30 β₄ midpoints ὡς 3 εἴσοδοι σὲ 30D.

    Ἐπιστρέφει shape (3, 30): ἕνα διάνυσμα ἀνὰ ἄξονα.

    Πρόβλεψι (μὲ Iₕ-equivariant projectors): ὅλη ἡ ἐνέργεια στὸν 3D
    irrep ποὺ ἀντιστοιχεῖ στὴν standard rep τῆς I (T₁ ἢ T₂ ἀνάλογα μὲ
    σύμβασι), ἀφοῦ τὰ x,y,z μετασχηματίζονται μεταξύ τους ὑπὸ τὴν δράσι
    τῆς I ⊂ SO(3). Μὲ random-orthogonal: ἀνάμικτη κατανομή.
    """
    coords = dt_graph['coords']
    beta4 = dt_graph['orbits']['β4']
    return coords[beta4].T  # (3, 30): rows = (x_axis, y_axis, z_axis)


def testbed_dt_beta4_perturbed(dt_graph, epsilons=(0.0, 0.01, 0.015, 0.05, 0.15), seed=42):
    """β₄ x,y,z + Gaussian θόρυβος σὲ διάφορα πλάτη ε.

    Γιὰ κάθε ε ∈ epsilons: vec = base + ε · N(0,1) (στοιχειακό std=ε).

    Ἐπιστρέφει list[(ε, vec)] μὲ vec.shape == (3, 30).

    Θεωρητικὴ πρόβλεψι μὲ Iₕ-equivariant projectors:
        E_T₁ = (10 + 3ε²) / (10 + 30ε²)
        διαρροή = 1 − E_T₁ = 27ε² / (10 + 30ε²)
    """
    base = testbed_dt_beta4_coordinates(dt_graph)
    rng = np.random.default_rng(seed)
    return [(eps, base + eps * rng.standard_normal(base.shape)) for eps in epsilons]


# ═══════════════════════════════════════════════════════════════════
# Συνθετικὰ datasets ποὺ ἐπιδεικνύουν κάμψι/κατάρρευσι (Στόχος 4)
# ═══════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────
# Π32-ARK.08 (🪨, 6★) — Pattern indistinguishability κρυστάλλου vs
# θερμικοῦ θανάτου (2026-05-11):
#
# Καὶ τὸ καθαρὸ T₁ state (E_T₁ = 1.0) καὶ τὸ Iₕ-isotropic floor
# (E_T₁ = 3/30 = 0.1) δίνουν max_deviation_pct ≈ 0 ὅταν θεωρηθοῦν ὡς
# ἀναμενόμενα Iₕ-isotropic distributions. Τὸ deviation mode τοῦ
# diagnose_epsilon_drift δὲν τὰ διακρίνει — καί τὰ δύο ἐπιστρέφουν
# verdict "✓ ζώνη βιώσιμης ἰσορροπίας".
#
# Διακρίνονται ΜΟΝΟ μέσῳ:
#   • ἀπολύτων τιμῶν E_ρ (1.0 στὸ T₁ vs 0.1 στὸ floor)
#   • νόρμας τοῦ διανύσματος εἰσόδου (καθαρὸ unit-norm vs ἄπειρο)
#   • ἱστορικοῦ τροχιᾶς (ἀρχὴ vs κατάληξι)
#
# Θεολογικὴ ἑρμηνεία (🌊): ὁ ζῶν Λόγος καὶ τὸ θερμοδυναμικὸ τέλος
# εἶναι ἀδιάκριτα ἀπὸ pattern μόνον. Χρειάζεται μνήμη, ἱστορία, ἢ
# "norm" γιὰ τὴν διάκρισι. Ἀντηχεῖ τὸ "Α καὶ Ω" τῆς Ἀποκαλύψεως 1:8.
#
# Καταγραφή: STONES_APPEND — Π32-ARK.08 (23h46-11-05-2026).yaml
# ─────────────────────────────────────────────────────────────────────

def testbed_dt_beta4_zone_sweep(dt_graph, multipliers=(0.5, 3.0, 7.0, 25.0, 45.0), seed=42):
    """5 σημεῖα ε = k·ε* (k ∈ multipliers) — ἕνα ἀνὰ ζώνη.

    Ἐπιστρέφει list[(zone_label, eps_target, vec)] ὅπου vec.shape == (3, 30).
    Default multipliers ἀντιστοιχοῦν σὲ:
        0.5·ε*  → βιώσιμη ἰσορροπία
        3·ε*    → ἀνεκτικὴ παραμόρφωσι
        7·ε*    → Cheeger ἐμφανὴς κάμψι
        25·ε*   → σοβαρὴ κάμψι
        45·ε*   → κατάρρευσι (T₁ μειοψηφία, πάνω ἀπὸ τὸ half-collapse 43·ε*)

    Βλ. Π32-ARK.08 παραπάνω: γιὰ πολὺ μεγάλα ε (π.χ. 100·ε*), τὸ Iₕ-
    isotropic floor δίνει false "βιώσιμη" verdict — pattern alone δὲν
    ἀρκεῖ γιὰ τὴν διάκρισι ζῶντος vs θερμικοῦ θανάτου.
    """
    base = testbed_dt_beta4_coordinates(dt_graph)
    rng = np.random.default_rng(seed)
    zone_labels = ['viable', 'tolerant', 'cheeger', 'severe', 'collapse']
    out = []
    for k, label in zip(multipliers, zone_labels):
        eps = k * EPSILON
        vec = base + eps * rng.standard_normal(base.shape)
        out.append((label, eps, vec))
    return out


def testbed_dt_beta4_anisotropic_noise(dt_graph, projectors, eps=0.10, target_irrep='H', seed=42):
    """β₄ x,y,z base + Iₕ-anisotropic noise (προβεβλημένο σὲ ἕνα μόνο irrep).

    raw_noise ~ N(0,1)^(3,30), projected_noise = P_target · raw_noise.
    Ἀποτέλεσμα: ὅλη ἡ διαρροὴ συσσωρεύεται σὲ ἕνα irrep → ΠΑΡΑΒΙΑΖΕΙ
    τὴν Schur-Gauss προφητεία rank(ρ)/27 · leakage.

    Ἐπιστρέφει vec.shape == (3, 30).
    """
    if target_irrep == 'T1':
        raise ValueError("target_irrep δὲν μπορεῖ νὰ εἶναι T1 (= base irrep)")
    base = testbed_dt_beta4_coordinates(dt_graph)
    rng = np.random.default_rng(seed)
    raw_noise = rng.standard_normal(base.shape)
    P = projectors[target_irrep]
    # P · v γιὰ κάθε row v στὸ raw_noise (rows = 3D axes)
    projected_noise = raw_noise @ P  # P symmetric, ἑπομένως v @ P = (P @ v.T).T
    return base + eps * projected_noise


def cheeger_attack_perturbation(dt_graph, n_edges_remove=12, seed=42):
    """Ἐπιστρέφει set n ἀκμῶν γιὰ ἀφαίρεσι (γιὰ χρήσι μὲ diagnose_cheeger).

    Στὸ n ≥ |β₁₀| = 12 ⇒ προβλέπεται spectral collapse (Π32-OIK.04).
    """
    rng = np.random.default_rng(seed)
    edges = list(dt_graph['edges'])
    idx = rng.choice(len(edges), n_edges_remove, replace=False)
    return {edges[i] for i in idx}


def testbed_dt_beta4_trajectory(dt_graph, n_steps=200, step_eps=0.05, seed=42):
    """Χρονικὴ ἀκολουθία v_0, v_1, ..., v_{n_steps} μὲ συσσωρευτικὸ Gaussian θόρυβο.

    v_0 = base (καθαρὸ T₁)
    v_{t+1} = v_t + step_eps · N(0,1)

    Ἀναμενόμενο cumulative std στὸ step t: ≈ step_eps · √t.
    Μὲ default step_eps=0.05, n_steps=200: cumulative ≈ 0.707 ≈ 47·ε*
    ⇒ ξεπερνᾶ τὸ half-collapse threshold 43·ε* ⇒ μετάβασι μέσῳ ζωνῶν.

    Ἐπιστρέφει list[(t, vec_t)] μὲ vec_t.shape == (3, 30).
    """
    base = testbed_dt_beta4_coordinates(dt_graph)
    rng = np.random.default_rng(seed)
    trajectory = [(0, base.copy())]
    v = base.copy()
    for t in range(1, n_steps + 1):
        v = v + step_eps * rng.standard_normal(v.shape)
        trajectory.append((t, v.copy()))
    return trajectory
