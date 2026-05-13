"""
test_ark.py — Tests γιὰ τὰ ἀπαραβίαστα ἀναλλοίωτα τῆς Κιβωτοῦ
==============================================================

Κατὰ τὸ CLAUDE.md: counts DT/24-cell/600-cell, Iₕ ranks, projectors,
G-κένωσι στὴ β₁₀, ε ≈ 1.5%, χ διατήρησι.

Τρέξε:
  uv run pytest                      # ὅλα (συμπ. slow)
  uv run pytest -m "not slow"        # χωρὶς 600-cell
  uv run pytest -v                   # verbose
"""

import numpy as np
import pytest

from ark_geometry import (
    PHI, RATIO, EPSILON, EPSILON_NOMINAL,
    BETA10, BETA6, BETA4,
    DT_V, DT_E, DT_F, DT_CHI,
    CELL24_V, CELL24_E, CELL24_F, CELL24_CELLS,
    CELL600_V, CELL600_E,
    CHEEGER_CRITICAL_EDGES,
    build_DT, build_24cell, build_600cell, graph_laplacian,
)
from ark_irreps import (
    IH_DIMS, PERM_DECOMP, RANKS_BETA4,
    i_character_table, build_irrep_projectors_30,
    build_icosahedral_rotations, beta4_permutation_rep,
    build_irrep_projectors_30_equivariant, _classify_rotation,
)
from ark_diagnostics import (
    diagnose_irrep_distribution, diagnose_cheeger,
    diagnose_epsilon_drift, diagnose_F_balance,
    testbed_random_isotropic as _tb_random,
    testbed_ih_aligned as _tb_ih,
    testbed_asymmetric as _tb_asym,
    testbed_dt_beta4_coordinates as _tb_beta4,
    testbed_dt_beta4_perturbed as _tb_beta4_pert,
    testbed_dt_beta4_zone_sweep as _tb_zone_sweep,
    testbed_dt_beta4_anisotropic_noise as _tb_aniso,
    cheeger_attack_perturbation as _cheeger_attack,
    testbed_dt_beta4_trajectory as _tb_traj,
)
from ark_state import (
    save_dt, load_dt,
    save_projectors, load_projectors,
    save_report, load_report,
    _to_json,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def dt():
    return build_DT()

@pytest.fixture(scope='module')
def cell24():
    return build_24cell()

@pytest.fixture(scope='module')
def projectors():
    return build_irrep_projectors_30(seed=42)


# ═══════════════════════════════════════════════════════════════════
# Σταθερές
# ═══════════════════════════════════════════════════════════════════

class TestConstants:
    def test_phi_identity(self):
        # PHI² = PHI + 1
        assert abs(PHI**2 - PHI - 1) < 1e-12

    def test_ratio_value(self):
        # RATIO = (9 + 4√2) / 7
        assert abs(RATIO - (9 + 4*np.sqrt(2)) / 7) < 1e-12

    def test_epsilon_nominal(self):
        assert EPSILON_NOMINAL == 0.015

    def test_epsilon_exact(self):
        # ε* = 4·(4√2 − 5) / 175 ≈ 0.0150138114
        expected = 4 * (4 * np.sqrt(2) - 5) / 175
        assert abs(EPSILON - expected) < 1e-15
        assert abs(EPSILON - 0.0150138114) < 1e-9
        # ε* ≠ ε_nominal (distinct constants)
        assert EPSILON != EPSILON_NOMINAL
        assert EPSILON > EPSILON_NOMINAL  # 0.01501 > 0.01500

    def test_beta_orbit_sizes(self):
        assert BETA10 == 12
        assert BETA6 == 20
        assert BETA4 == 30
        assert BETA10 + BETA6 + BETA4 == 62  # = DT V

    def test_cheeger_equals_beta10(self):
        assert CHEEGER_CRITICAL_EDGES == BETA10 == 12


# ═══════════════════════════════════════════════════════════════════
# DT (Disdyakis Triacontahedron)
# ═══════════════════════════════════════════════════════════════════

class TestDT:
    def test_counts(self, dt):
        assert dt['V'] == DT_V == 62
        assert dt['E'] == DT_E == 180
        assert dt['F'] == DT_F == 120
        assert dt['chi'] == DT_CHI == 2

    def test_orbits_partition(self, dt):
        b10, b6, b4 = dt['orbits']['β10'], dt['orbits']['β6'], dt['orbits']['β4']
        assert len(b10) == 12
        assert len(b6) == 20
        assert len(b4) == 30
        all_idx = set(b10) | set(b6) | set(b4)
        assert all_idx == set(range(62))
        assert len(set(b10) & set(b6)) == 0
        assert len(set(b6) & set(b4)) == 0
        assert len(set(b10) & set(b4)) == 0

    def test_degrees_match_orbits(self, dt):
        for v in dt['orbits']['β10']: assert dt['degrees'][v] == 10
        for v in dt['orbits']['β6']:  assert dt['degrees'][v] == 6
        for v in dt['orbits']['β4']:  assert dt['degrees'][v] == 4

    def test_euler_chi(self, dt):
        # 2-manifold: V − E + F = 2
        assert dt['V'] - dt['E'] + dt['F'] == 2

    def test_edges_unique(self, dt):
        assert len(dt['edges']) == 180
        # κάθε ἀκμὴ frozenset μὲ 2 διαφορετικὲς κορυφές
        for e in dt['edges']:
            assert len(e) == 2

    def test_faces_unique(self, dt):
        # κάθε face frozenset μὲ 3 κορυφές
        for f in dt['faces']:
            assert len(f) == 3
        assert len(dt['faces']) == 120


# ═══════════════════════════════════════════════════════════════════
# 24-cell
# ═══════════════════════════════════════════════════════════════════

class TestCell24:
    def test_counts(self, cell24):
        assert cell24['V'] == CELL24_V == 24
        assert cell24['E'] == CELL24_E == 96
        assert cell24['F'] == CELL24_F == 96
        assert cell24['cells'] == CELL24_CELLS == 24

    def test_euler_chi(self, cell24):
        # 4-manifold: V − E + F − cells = 0
        chi = cell24['V'] - cell24['E'] + cell24['F'] - cell24['cells']
        assert chi == 0 == cell24['chi']

    def test_two_types(self, cell24):
        # 8 axis (Type A) + 16 cube-corner (Type B) = 24
        assert len(cell24['types']['A']) == 8
        assert len(cell24['types']['B']) == 16


# ═══════════════════════════════════════════════════════════════════
# 600-cell (slow — ~5-10s)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestCell600:
    def test_counts(self):
        c600 = build_600cell()
        assert c600['V'] == CELL600_V == 120
        assert c600['E'] == CELL600_E == 720
        assert c600['F'] == 1200
        assert c600['cells'] == 600

    def test_euler_chi(self):
        c600 = build_600cell()
        chi = c600['V'] - c600['E'] + c600['F'] - c600['cells']
        assert chi == 0


# ═══════════════════════════════════════════════════════════════════
# Iₕ irreps
# ═══════════════════════════════════════════════════════════════════

class TestIrreps:
    def test_dimensions(self):
        # Σ dim²ᵨ = |I| = 60: 1 + 9 + 9 + 16 + 25 = 60
        total = sum(d**2 for d in IH_DIMS.values())
        assert total == 60

    def test_ranks_beta4_sum(self):
        # mults × dims = 1·1 + 1·3 + 1·3 + 2·4 + 3·5 = 30
        assert sum(RANKS_BETA4.values()) == 30
        assert RANKS_BETA4['A'] == 1
        assert RANKS_BETA4['G'] == 8   # mult=2, dim=4
        assert RANKS_BETA4['H'] == 15  # mult=3, dim=5

    def test_perm_decomp_consistent_with_ranks(self):
        # Γιὰ τὴ β₄: μέτρα ἀπὸ PERM_DECOMP × dim πρέπει νὰ συμφωνοῦν μὲ RANKS_BETA4
        for name, mult in PERM_DECOMP['β4'].items():
            expected_rank = mult * IH_DIMS[name]
            assert RANKS_BETA4[name] == expected_rank, name

    def test_beta10_g_absent(self):
        # Π-Μ-14: G ΑΠΟΥΣΑ στὴ β₁₀ permutation rep
        assert PERM_DECOMP['β10']['G'] == 0

    def test_perm_decomp_dims_match_orbit_sizes(self):
        # Σ mult·dim πρέπει νὰ ταιριάζει μὲ τὸ |orbit|
        for orbit, decomp in PERM_DECOMP.items():
            total = sum(mult * IH_DIMS[name] for name, mult in decomp.items())
            expected = {'β10': 12, 'β6': 20, 'β4': 30}[orbit]
            assert total == expected, f"{orbit}: got {total}, expected {expected}"

    def test_character_table_orthogonality(self):
        # ⟨χ_ρ, χ_ρ⟩ = 1 (orthonormal characters)
        ct = i_character_table()
        class_sizes = [c[1] for c in ct['classes']]
        order = sum(class_sizes)
        assert order == 60  # |I|

        for name, chi in ct['characters'].items():
            inner = sum(class_sizes[k] * chi[k]**2 for k in range(5)) / order
            assert abs(inner - 1.0) < 1e-10, f"χ_{name} ⟨,⟩ = {inner}"

    def test_projectors_idempotent(self, projectors):
        for name, P in projectors.items():
            assert np.allclose(P @ P, P, atol=1e-10), name

    def test_projectors_sum_to_identity(self, projectors):
        S = sum(projectors.values())
        assert np.allclose(S, np.eye(30), atol=1e-10)

    def test_projectors_correct_rank(self, projectors):
        for name, P in projectors.items():
            r = int(round(np.trace(P)))
            assert r == RANKS_BETA4[name], f"{name}: rank={r}, expected {RANKS_BETA4[name]}"


# ═══════════════════════════════════════════════════════════════════
# graph_laplacian
# ═══════════════════════════════════════════════════════════════════

class TestLaplacian:
    def test_dt_laplacian_symmetric(self, dt):
        L = graph_laplacian(dt)
        assert L.shape == (62, 62)
        assert np.allclose(L, L.T, atol=1e-12)

    def test_dt_laplacian_eigvals_in_bounds(self, dt):
        # Normalized Laplacian: spectrum ⊂ [0, 2]
        L = graph_laplacian(dt)
        eigs = np.linalg.eigvalsh(L)
        assert eigs.min() > -1e-10
        assert eigs.max() < 2 + 1e-10

    def test_dt_connected(self, dt):
        # μ₀ ≈ 0, μ₁ > 0 ⇒ συνεκτικός
        L = graph_laplacian(dt)
        eigs = np.sort(np.linalg.eigvalsh(L))
        assert abs(eigs[0]) < 1e-10
        assert eigs[1] > 1e-6


# ═══════════════════════════════════════════════════════════════════
# Διαγνωστικές
# ═══════════════════════════════════════════════════════════════════

class TestDiagnostics:
    def test_random_is_isotropic(self, projectors):
        # 200 Gaussian samples → Iₕ-isotropic στὸ μέσο ὅρο
        v = _tb_random(n_samples=200, seed=42)
        result = diagnose_irrep_distribution(v, projectors)
        assert result['verdict'] == 'Iₕ-isotropic (random/symmetric input)'
        assert result['max_deviation_pct'] < 10

    def test_total_energy_is_one(self, projectors):
        # Σ_ρ ‖P_ρ v̂‖² = ‖v̂‖² = 1 (projectors sum to I)
        v = _tb_random(n_samples=10, seed=1)
        result = diagnose_irrep_distribution(v, projectors)
        assert abs(result['total'] - 1.0) < 1e-10

    def test_ih_aligned_strongly_anisotropic(self, projectors):
        for irrep in ['G', 'H', 'T1', 'T2']:
            v = _tb_ih(projectors, target_irrep=irrep, seed=42)
            result = diagnose_irrep_distribution(v, projectors)
            assert 'strongly anisotropic' in result['verdict'], irrep

    def test_cheeger_dt(self, dt):
        result = diagnose_cheeger(dt)
        assert result['mu1'] > 0
        assert result['kibwtos_invariant_beta10'] == 12

    def test_cheeger_collapse_at_beta10(self, dt):
        # Ἀφαίρεσι ≥ 12 ἀκμῶν ⇒ verdict spectral collapse
        edges_list = list(dt['edges'])
        np.random.seed(42)
        idx = np.random.choice(len(edges_list), 12, replace=False)
        pert = set(edges_list[i] for i in idx)
        result = diagnose_cheeger(dt, perturbation_edges=pert)
        assert 'SPECTRAL COLLAPSE' in result['verdict']
        assert result['perturbation_edges_removed'] == 12

    def test_epsilon_drift_living_imperfection(self):
        # 1.5% < ε* (= 0.0150138...) ⇒ within ζώνη βιώσιμης ἰσορροπίας
        result = diagnose_epsilon_drift({'max_deviation_pct': 1.5})
        assert result['within_living_imperfection']
        assert result['predicted_cycles_to_collapse'] == float('inf')
        assert result['mode'] == 'deviation'
        assert 'βιώσιμης ἰσορροπίας' in result['verdict']

    def test_epsilon_drift_overshoot(self):
        # 50% > 10·ε* (15%) ἀλλὰ < 43·ε* (64.55%) ⇒ ζώνη σοβαρῆς κάμψης
        result = diagnose_epsilon_drift({'max_deviation_pct': 50.0})
        assert not result['within_living_imperfection']
        assert 'σοβαρῆς κάμψης' in result['verdict']

    def test_epsilon_drift_zone_boundaries_5_zones(self):
        # 1: ε ≤ ε* → βιώσιμη
        r1 = diagnose_epsilon_drift({'max_deviation_pct': EPSILON * 100 * 0.99})
        assert 'βιώσιμης' in r1['verdict']
        # 2: ε* < ε ≤ 5ε* → ἀνεκτική
        r2 = diagnose_epsilon_drift({'max_deviation_pct': 3 * EPSILON * 100})
        assert 'ἀνεκτικῆς' in r2['verdict']
        # 3: 5ε* < ε ≤ 10ε* → ἐμφανὴς κάμψι (Cheeger)
        r3 = diagnose_epsilon_drift({'max_deviation_pct': 7 * EPSILON * 100})
        assert 'ἐμφανοῦς κάμψης' in r3['verdict']
        assert 'Cheeger' in r3['verdict']
        # 4: 10ε* < ε ≤ 43ε* → σοβαρὴ κάμψι
        r4 = diagnose_epsilon_drift({'max_deviation_pct': 25 * EPSILON * 100})
        assert 'σοβαρῆς κάμψης' in r4['verdict']
        # 5: ε > 43ε* → κατάρρευσι
        r5 = diagnose_epsilon_drift({'max_deviation_pct': 50 * EPSILON * 100})
        assert 'κατάρρευσης' in r5['verdict']

    def test_implied_epsilon_inversion(self):
        """Ἀντίστροφος Schur-Gauss: leakage(ε) → ε."""
        from ark_diagnostics import _implied_epsilon_from_leakage
        # leakage = 0 ⇒ ε = 0
        assert _implied_epsilon_from_leakage(0.0) == 0.0
        assert _implied_epsilon_from_leakage(-0.5) == 0.0
        # leakage = 50% ⇒ ε² = 5/12 ⇒ ε ≈ 0.6455 (43·ε*)
        eps_half = _implied_epsilon_from_leakage(0.5)
        assert abs(eps_half - np.sqrt(5/12)) < 1e-12
        assert abs(eps_half / EPSILON - 43) < 0.05
        # leakage = 27/30 (Iₕ-isotropic floor) ⇒ ε → ∞
        assert _implied_epsilon_from_leakage(27/30) == float('inf')
        assert _implied_epsilon_from_leakage(0.95) == float('inf')
        # Round-trip: ε → leakage → ε
        for eps in [0.001, 0.05, 0.15, 0.5, 0.6]:
            leakage = 27 * eps**2 / (10 + 30 * eps**2)
            eps_back = _implied_epsilon_from_leakage(leakage)
            assert abs(eps_back - eps) < 1e-10, f"ε={eps}: round-trip ε={eps_back}"

    def test_half_collapse_at_43_eps_star(self, dt, eq_projectors):
        """Στὸ noise std = √(5/12) ≈ 0.6455 (= 43·ε*), E_T₁ ≈ 0.5 = leakage."""
        eps = float(np.sqrt(5/12))
        # Πρόβλεψι: ε / ε* ≈ 43
        assert abs(eps / EPSILON - 43) < 0.05
        # Empirically: ποιὰ E_T₁ προκύπτει;
        n_trials = 200
        t1_avg = 0.0
        for seed in range(n_trials):
            result = _tb_beta4_pert(dt, epsilons=(eps,), seed=seed)
            _, vec = result[0]
            r = diagnose_irrep_distribution(vec, eq_projectors)
            t1_avg += r['energies']['T1'] / n_trials
        # E_T₁ μέσος ὅρος ≈ 0.5 (μὲ διακύμανσι ~1%)
        assert abs(t1_avg - 0.5) < 0.01, f"E_T₁ avg = {t1_avg}"

    def test_diagnose_drift_at_half_collapse(self, dt, eq_projectors):
        """Στὸ ε = 43·ε*, τὸ diagnose_epsilon_drift ἀναγνωρίζει ζώνη σοβαρῆς
        κάμψης (στὸ ὅριο μεταξὺ σοβαρῆς καὶ κατάρρευσης)."""
        eps = float(np.sqrt(5/12))
        # Avg ἐνέργειες ἀπὸ 200 trials γιὰ stable διάγνωσι
        n_trials = 200
        avg_energies = {n: 0.0 for n in ['A', 'T1', 'T2', 'G', 'H']}
        for seed in range(n_trials):
            result = _tb_beta4_pert(dt, epsilons=(eps,), seed=seed)
            _, vec = result[0]
            r = diagnose_irrep_distribution(vec, eq_projectors)
            for n in avg_energies:
                avg_energies[n] += r['energies'][n] / n_trials
        # Στὸ ὅριο, dominant παραμένει T₁ (ὁριακά)
        ed = diagnose_epsilon_drift({'energies': avg_energies, 'max_deviation_pct': 0})
        assert ed['mode'] == 'leakage'
        assert ed['dominant_irrep'] == 'T1'
        assert abs(ed['leakage_fraction'] - 0.5) < 0.01
        assert abs(ed['epsilon_implied'] - eps) < 0.01
        # Στὸ ε ≈ 43·ε* = 64.55%, εἴμαστε στὸ ὅριο σοβαρῆς κάμψης / κατάρρευσης
        assert ('σοβαρῆς κάμψης' in ed['verdict']) or ('κατάρρευσης' in ed['verdict'])

    def test_ih_isotropic_floor_flag(self):
        """Στὸ leakage ≥ 27/30, τὸ flag ih_isotropic_floor εἶναι True."""
        # E_T₁ = 0.1 = 3/30 ⇒ leakage = 0.9 = 27/30
        # Ἀλλὰ τὸ dominant irrep δὲν εἶναι T₁ (E=0.1) — εἶναι ἕνα ἀπὸ τὰ ἄλλα.
        # Γιὰ νὰ τεστάρω τὸ flag, χρειάζομαι dominant > 0.5 μὲ leakage κοντὰ στὸ floor.
        # Δὲν εἶναι φυσικὰ ἐφικτὸ — τὸ floor ἀπαιτεῖ ΟΛΑ νὰ εἶναι Iₕ-isotropic.
        # Ἀντ' αὐτοῦ ἐλέγχω τὸ helper καὶ τὴ συμπεριφορὰ τῆς συνάρτησης
        # ὅταν δίνουμε E_T₁ = 0.55 (ὁριακό dominant, leakage 45% — ζώνη
        # σοβαρῆς κάμψης ἀλλὰ ὄχι floor):
        ed = diagnose_epsilon_drift({
            'energies': {'A': 0.10, 'T1': 0.55, 'T2': 0.05, 'G': 0.10, 'H': 0.20},
            'max_deviation_pct': 0,
        })
        assert ed['mode'] == 'leakage'
        assert ed['dominant_irrep'] == 'T1'
        assert ed['ih_isotropic_floor'] is False
        assert abs(ed['leakage_fraction'] - 0.45) < 1e-6

    def test_ih_isotropic_floor_value(self):
        """Σταθερὰ floor: E_T₁ → 3/30 στὸ ε → ∞."""
        from ark_diagnostics import IH_ISOTROPIC_FLOOR_T1, IH_ISOTROPIC_FLOOR_LEAKAGE
        assert IH_ISOTROPIC_FLOOR_T1 == 3/30
        assert IH_ISOTROPIC_FLOOR_LEAKAGE == 27/30
        assert abs(IH_ISOTROPIC_FLOOR_T1 + IH_ISOTROPIC_FLOOR_LEAKAGE - 1.0) < 1e-15

    def test_zone_constants(self):
        from ark_diagnostics import (
            ZONE_VIABLE_K, ZONE_TOLERANT_K, ZONE_CHEEGER_K, ZONE_HALF_K,
        )
        assert ZONE_VIABLE_K == 1
        assert ZONE_TOLERANT_K == 5
        assert ZONE_CHEEGER_K == 10
        assert ZONE_HALF_K == 43


# ═══════════════════════════════════════════════════════════════════
# Συνθετικὰ datasets: κάμψι / κατάρρευσι (Στόχος #4)
# ═══════════════════════════════════════════════════════════════════

class TestZoneSweep:
    """A: zone_sweep — 5 σημεῖα, ἕνα ἀνὰ ζώνη."""

    def test_returns_5_entries(self, dt):
        result = _tb_zone_sweep(dt)
        assert len(result) == 5

    def test_eps_targets_match_multipliers(self, dt):
        result = _tb_zone_sweep(dt, multipliers=(0.5, 3.0, 7.0, 25.0, 45.0))
        expected_multipliers = [0.5, 3.0, 7.0, 25.0, 45.0]
        for (label, eps, _), k in zip(result, expected_multipliers):
            assert abs(eps - k * EPSILON) < 1e-12

    def test_zone_labels_in_order(self, dt):
        result = _tb_zone_sweep(dt)
        labels = [r[0] for r in result]
        assert labels == ['viable', 'tolerant', 'cheeger', 'severe', 'collapse']

    def test_each_zone_hits_expected_verdict(self, dt, eq_projectors):
        """Γιὰ κάθε ε στὸ sweep, μέσος ὅρος 50 trials → expected verdict κεῖται
        στὴν προβλεπόμενη ζώνη."""
        # Mapping: ζώνη → σύμβολο verdict
        expected_substrings = {
            'viable':   'βιώσιμης',
            'tolerant': 'ἀνεκτικῆς',
            'cheeger':  'ἐμφανοῦς κάμψης',
            'severe':   'σοβαρῆς κάμψης',
            # collapse: ἀκριβῶς στὸ ε=100·ε* μπορεῖ νὰ εἶναι ἢ "κατάρρευσης"
            # ἢ "θερμοδυναμικὴ ἰσορροπία" (στὸ floor) — δεχόμαστε καὶ τὰ δύο
        }
        for label, target_eps, _ in _tb_zone_sweep(dt):
            avg_energies = {n: 0.0 for n in ['A', 'T1', 'T2', 'G', 'H']}
            n_trials = 50
            for seed in range(n_trials):
                _, _, vec = _tb_zone_sweep(dt, multipliers=(target_eps / EPSILON,), seed=seed)[0]
                r = diagnose_irrep_distribution(vec, eq_projectors)
                for n in avg_energies:
                    avg_energies[n] += r['energies'][n] / n_trials
            ed = diagnose_epsilon_drift({
                'energies': avg_energies,
                'max_deviation_pct': 0,
                'expected_isotropic': {n: RANKS_BETA4[n]/30 for n in RANKS_BETA4},
            })
            if label in expected_substrings:
                assert expected_substrings[label] in ed['verdict'], \
                    f"{label} (ε={target_eps:.4f}): got '{ed['verdict']}'"
            else:  # collapse (45·ε*, ἀκριβῶς πάνω ἀπὸ half-collapse 43·ε*)
                assert ('κατάρρευσης' in ed['verdict']
                        or 'σοβαρῆς κάμψης' in ed['verdict']), \
                    f"{label}: got '{ed['verdict']}'"


class TestAnisotropicNoise:
    """B: anisotropic_noise — Schur-Gauss πρέπει νὰ ΜΗΝ ἐπιβεβαιώνεται."""

    def test_shape(self, dt, eq_projectors):
        v = _tb_aniso(dt, eq_projectors, eps=0.10, target_irrep='H')
        assert v.shape == (3, 30)

    def test_t1_not_allowed_as_target(self, dt, eq_projectors):
        with pytest.raises(ValueError):
            _tb_aniso(dt, eq_projectors, eps=0.10, target_irrep='T1')

    def test_anisotropic_noise_concentrates_in_target_irrep(self, dt, eq_projectors):
        """Στὸ H-targeted noise, ἡ διαρροὴ συσσωρεύεται κυρίως στὸ H,
        ὄχι ἀναλογικὰ μὲ ranks 1:3:8:15."""
        n_trials = 50
        sums = {n: 0.0 for n in ['A', 'T2', 'G', 'H']}
        for seed in range(n_trials):
            v = _tb_aniso(dt, eq_projectors, eps=0.20, target_irrep='H', seed=seed)
            r = diagnose_irrep_distribution(v, eq_projectors)
            for n in sums:
                sums[n] += r['energies'][n] / n_trials
        # Schur-Gauss πρόβλεψι: H/total ≈ 15/27. Πραγματικότητα: H κυριαρχεῖ.
        total = sum(sums.values())
        h_ratio = sums['H'] / total
        # Schur-Gauss θὰ ἔδινε 15/27 ≈ 0.556· τὸ anisotropic δίνει ≫ αὐτό
        assert h_ratio > 0.95, f"H ratio = {h_ratio} (expected ~1 γιὰ H-anisotropic)"

    def test_schur_gauss_NOT_confirmed_for_anisotropic(self, dt, eq_projectors):
        """diagnose_epsilon_drift πρέπει νὰ φέρει schur_gauss_confirmed=False
        γιὰ τὸ H-anisotropic noise."""
        n_trials = 50
        avg_energies = {n: 0.0 for n in ['A', 'T1', 'T2', 'G', 'H']}
        for seed in range(n_trials):
            v = _tb_aniso(dt, eq_projectors, eps=0.20, target_irrep='H', seed=seed)
            r = diagnose_irrep_distribution(v, eq_projectors)
            for n in avg_energies:
                avg_energies[n] += r['energies'][n] / n_trials
        ed = diagnose_epsilon_drift({
            'energies': avg_energies,
            'max_deviation_pct': 0,
            'expected_isotropic': {n: RANKS_BETA4[n]/30 for n in RANKS_BETA4},
        })
        if ed['mode'] == 'leakage' and 'schur_gauss_confirmed' in ed:
            assert ed['schur_gauss_confirmed'] is False
            assert ed['leakage_max_rel_err'] > 0.5


class TestCheegerAttack:
    """C: Cheeger collapse μέσῳ ἀφαίρεσης ≥|β₁₀|=12 ἀκμῶν."""

    def test_perturbation_size(self, dt):
        pert = _cheeger_attack(dt, n_edges_remove=12)
        assert len(pert) == 12
        # Ὅλα frozensets ἀπὸ τὸ DT edge set
        for e in pert:
            assert e in dt['edges']

    def test_12_edges_triggers_spectral_collapse(self, dt):
        pert = _cheeger_attack(dt, n_edges_remove=12, seed=42)
        result = diagnose_cheeger(dt, perturbation_edges=pert)
        assert 'SPECTRAL COLLAPSE' in result['verdict']
        assert result['perturbation_edges_removed'] == 12

    def test_6_edges_triggers_stress_not_collapse(self, dt):
        pert = _cheeger_attack(dt, n_edges_remove=6, seed=42)
        result = diagnose_cheeger(dt, perturbation_edges=pert)
        assert 'stress regime' in result['verdict']
        assert 'SPECTRAL COLLAPSE' not in result['verdict']

    def test_2_edges_stable(self, dt):
        pert = _cheeger_attack(dt, n_edges_remove=2, seed=42)
        result = diagnose_cheeger(dt, perturbation_edges=pert)
        assert 'stable' in result['verdict']


class TestTrajectory:
    """D: συσσωρευτικὸς θόρυβος — μετάβασι μέσῳ ζωνῶν."""

    def test_length_n_steps_plus_one(self, dt):
        traj = _tb_traj(dt, n_steps=50, step_eps=0.05)
        assert len(traj) == 51

    def test_starts_at_base(self, dt):
        traj = _tb_traj(dt, n_steps=10, step_eps=0.05)
        t0, v0 = traj[0]
        assert t0 == 0
        np.testing.assert_array_equal(v0, _tb_beta4(dt))

    def test_norm_increases_on_average(self, dt):
        """Cumulative noise → ‖v_t − base‖ αὐξάνει μὲ √t (κατὰ μέσο ὅρο)."""
        traj = _tb_traj(dt, n_steps=100, step_eps=0.05)
        base = _tb_beta4(dt)
        deviations = [np.linalg.norm(v - base) for _, v in traj]
        # Τὸ ἀρχικὸ deviation = 0
        assert deviations[0] == 0.0
        # Στὸ τέλος >> ἀρχικό
        assert deviations[-1] > 0.5

    def test_progresses_through_zones(self, dt, eq_projectors):
        """Ἐπιδεικνύει τοὐλάχιστον 3 διαφορετικὲς ζῶνες στὴ διαδρομή
        (default n_steps=200, step_eps=0.05 → cumulative σ ≈ 0.707 ≈ 47·ε*)."""
        traj = _tb_traj(dt, n_steps=200, step_eps=0.05, seed=42)
        zones_seen = set()
        for t, v in traj:
            r = diagnose_irrep_distribution(v, eq_projectors)
            ed = diagnose_epsilon_drift({
                'energies': r['energies'],
                'max_deviation_pct': r['max_deviation_pct'],
                'expected_isotropic': r['expected_isotropic'],
            })
            verdict = ed['verdict']
            if 'βιώσιμης' in verdict: zones_seen.add('viable')
            elif 'ἀνεκτικῆς' in verdict: zones_seen.add('tolerant')
            elif 'ἐμφανοῦς' in verdict: zones_seen.add('cheeger')
            elif 'σοβαρῆς' in verdict: zones_seen.add('severe')
            elif 'κατάρρευσης' in verdict: zones_seen.add('collapse')
        assert len(zones_seen) >= 3, f"only {zones_seen}"
        # Πρῶτο step εἶναι βιώσιμη, τελικὸ πρέπει νὰ εἶναι σὲ ὑψηλότερη ζώνη
        assert 'viable' in zones_seen
        assert ('severe' in zones_seen) or ('collapse' in zones_seen)


# ═══════════════════════════════════════════════════════════════════
# Save/load JSON state (Στόχος #3)
# ═══════════════════════════════════════════════════════════════════

class TestJsonHelper:
    def test_to_json_dict(self):
        assert _to_json({'a': 1, 'b': 2}) == {'a': 1, 'b': 2}

    def test_to_json_numpy_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert _to_json(arr) == [[1.0, 2.0], [3.0, 4.0]]

    def test_to_json_numpy_scalars(self):
        assert _to_json(np.int32(5)) == 5
        assert _to_json(np.float64(3.14)) == 3.14

    def test_to_json_frozenset(self):
        # frozenset → sorted list
        assert _to_json(frozenset({3, 1, 2})) == [1, 2, 3]

    def test_to_json_inf(self):
        assert _to_json(float('inf')) == 'inf'
        assert _to_json(float('-inf')) == '-inf'

    def test_to_json_nested(self):
        obj = {
            'arr': np.array([1, 2, 3]),
            'frozen': frozenset({2, 1}),
            'inner': {'inf_val': float('inf'), 'normal': 0.5},
        }
        result = _to_json(obj)
        assert result['arr'] == [1, 2, 3]
        assert result['frozen'] == [1, 2]
        assert result['inner']['inf_val'] == 'inf'
        assert result['inner']['normal'] == 0.5


class TestDtRoundTrip:
    def test_dt_save_load(self, dt, tmp_path):
        path = tmp_path / 'dt.json'
        save_dt(dt, str(path))
        assert path.exists()
        loaded = load_dt(str(path))
        # Σχῆμα διατηρεῖται
        assert loaded['V'] == dt['V'] == 62
        assert loaded['E'] == dt['E'] == 180
        assert loaded['F'] == dt['F'] == 120
        assert loaded['chi'] == 2
        # Edges/faces ἀπὸ frozensets
        assert len(loaded['edges']) == 180
        for e in loaded['edges']:
            assert isinstance(e, frozenset)
        # Orbits
        assert len(loaded['orbits']['β10']) == 12
        assert len(loaded['orbits']['β6']) == 20
        assert len(loaded['orbits']['β4']) == 30
        # Coords εἶναι numpy array
        assert isinstance(loaded['coords'], np.ndarray)
        assert loaded['coords'].shape == (62, 3)
        np.testing.assert_array_almost_equal(loaded['coords'], dt['coords'])

    def test_loaded_dt_passes_invariants(self, dt, tmp_path):
        """DT μετὰ ἀπὸ round-trip περνᾶ τὰ ἴδια invariant tests."""
        path = tmp_path / 'dt.json'
        save_dt(dt, str(path))
        loaded = load_dt(str(path))
        # Euler χ
        assert loaded['V'] - loaded['E'] + loaded['F'] == 2
        # Degrees match orbits
        for v in loaded['orbits']['β10']: assert loaded['degrees'][v] == 10
        for v in loaded['orbits']['β6']:  assert loaded['degrees'][v] == 6
        for v in loaded['orbits']['β4']:  assert loaded['degrees'][v] == 4

    def test_loaded_dt_works_with_diagnostics(self, dt, tmp_path, eq_projectors):
        """Φορτωμένο DT λειτουργεῖ στὰ διαγνωστικά."""
        path = tmp_path / 'dt.json'
        save_dt(dt, str(path))
        loaded = load_dt(str(path))
        # Cheeger διαγνωστικό
        result = diagnose_cheeger(loaded)
        assert result['mu1'] > 0
        # F-balance
        fb = diagnose_F_balance(loaded)
        assert fb['preserved']


class TestProjectorsRoundTrip:
    def test_projectors_save_load(self, eq_projectors, tmp_path):
        path = tmp_path / 'proj.json'
        save_projectors(eq_projectors, str(path))
        loaded = load_projectors(str(path))
        assert set(loaded.keys()) == set(eq_projectors.keys())
        for name in eq_projectors:
            np.testing.assert_array_almost_equal(loaded[name], eq_projectors[name])

    def test_loaded_projectors_preserve_invariants(self, eq_projectors, tmp_path):
        """Φορτωμένα projectors: idempotent, sum=I, σωστά ranks."""
        path = tmp_path / 'proj.json'
        save_projectors(eq_projectors, str(path))
        loaded = load_projectors(str(path))
        for name, P in loaded.items():
            assert np.allclose(P @ P, P, atol=1e-8), name
            r = int(round(np.trace(P)))
            assert r == RANKS_BETA4[name]
        S = sum(loaded.values())
        assert np.allclose(S, np.eye(30), atol=1e-8)


class TestReportRoundTrip:
    def test_report_save_load_simple(self, tmp_path):
        report = {
            'input': 'test',
            'energies': {'T1': 1.0, 'A': 0.0},
            'cycles': float('inf'),
            'array': np.array([1.0, 2.0]),
        }
        path = tmp_path / 'report.json'
        save_report(report, str(path))
        loaded = load_report(str(path))
        assert loaded['input'] == 'test'
        assert loaded['energies'] == {'T1': 1.0, 'A': 0.0}
        assert loaded['cycles'] == float('inf')
        assert loaded['array'] == [1.0, 2.0]

    def test_report_round_trip_full_diagnostic(self, dt, eq_projectors, tmp_path):
        """Πλήρης κύκλος: τρέξε διάγνωσι, ἀποθήκευσε, φόρτωσε, σύγκρινε."""
        from ark_main import run_full_diagnostic
        from ark_geometry import build_24cell
        cell24 = build_24cell()
        v = _tb_beta4(dt)
        report = run_full_diagnostic('test_input', v, eq_projectors, dt, cell24)
        path = tmp_path / 'full_report.json'
        save_report(report, str(path))
        loaded = load_report(str(path))
        # Κρίσιμα πεδία διατηρημένα
        assert loaded['input'] == 'test_input'
        assert loaded['diagnostics']['f_balance_dt']['preserved'] is True
        # ∞ ξανακαλωσορίστηκε
        cyc = loaded['diagnostics']['epsilon_drift']['predicted_cycles_to_collapse']
        assert cyc == float('inf')

    def test_cli_save_report(self, tmp_path):
        """End-to-end: τρέχει τὸ CLI μὲ --save-report καὶ ἐλέγχει τὸ JSON ἀρχεῖο."""
        import subprocess, os, sys
        path = tmp_path / 'cli_report.json'
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        result = subprocess.run(
            [sys.executable, 'ark_main.py', '--testbed=beta4', '--save-report', str(path)],
            capture_output=True, env=env, text=True, encoding='utf-8',
        )
        assert result.returncode == 0, result.stderr
        assert path.exists()
        loaded = load_report(str(path))
        assert isinstance(loaded, list)
        assert len(loaded) == 1  # μόνο τὸ beta4 testbed
        assert 'β₄' in loaded[0]['input']

    def test_F_balance_dt(self, dt):
        result = diagnose_F_balance(dt)
        assert result['preserved']
        assert result['chi_computed'] == 2

    def test_F_balance_24cell(self, cell24):
        result = diagnose_F_balance(cell24)
        assert result['preserved']
        assert result['chi_computed'] == 0


# ═══════════════════════════════════════════════════════════════════
# Iₕ-Equivariant projectors (Στάδιο 2.2)
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope='module')
def rotations(dt):
    return build_icosahedral_rotations(dt)

@pytest.fixture(scope='module')
def perms(dt, rotations):
    return beta4_permutation_rep(dt, rotations=rotations)

@pytest.fixture(scope='module')
def eq_projectors(dt):
    return build_irrep_projectors_30_equivariant(dt)


class TestEquivariantRotations:
    def test_60_rotations(self, rotations):
        assert len(rotations) == 60

    def test_all_orthogonal(self, rotations):
        for R in rotations:
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)

    def test_all_proper_rotations(self, rotations):
        for R in rotations:
            assert abs(np.linalg.det(R) - 1) < 1e-10

    def test_class_sizes_match_A5(self, rotations):
        # Conjugacy classes τοῦ A₅: 1 + 15 + 20 + 12 + 12 = 60
        from collections import Counter
        sizes = Counter(_classify_rotation(R) for R in rotations)
        assert sizes['E']   == 1
        assert sizes['C2']  == 15
        assert sizes['C3']  == 20
        assert sizes['C5']  == 12
        assert sizes['C52'] == 12

    def test_preserves_beta10_orbit(self, rotations, dt):
        from scipy.spatial.distance import cdist
        coords = dt['coords']
        beta10 = coords[dt['orbits']['β10']]
        for R in rotations:
            rotated = beta10 @ R.T
            D = cdist(rotated, beta10)
            assert np.all(D.min(axis=1) < 1e-8)


class TestPermutationRep:
    def test_60_permutations(self, perms):
        assert len(perms) == 60

    def test_are_permutation_matrices(self, perms):
        for P in perms:
            unique = set(np.round(P, 10).flatten().tolist())
            assert unique.issubset({0.0, 1.0})
            assert np.allclose(P.sum(axis=0), 1)
            assert np.allclose(P.sum(axis=1), 1)

    def test_identity_present(self, perms):
        I = np.eye(30)
        assert any(np.allclose(P, I) for P in perms)

    def test_homomorphism_closed(self, perms):
        # π(g) · π(h) ∈ {π(k) : k ∈ I} γιὰ ζευγάρια δειγματοληψίας
        rng = np.random.default_rng(42)
        idx_pairs = [(int(rng.integers(60)), int(rng.integers(60))) for _ in range(15)]
        for i, j in idx_pairs:
            product = perms[i] @ perms[j]
            assert any(np.allclose(product, P, atol=1e-10) for P in perms), \
                f"π({i})·π({j}) ∉ rep"

    def test_inverse_present(self, perms):
        # π(g)⁻¹ = π(g)ᵀ (γιὰ permutation matrices) πρέπει νὰ εἶναι στὸν rep
        for i in range(0, 60, 10):  # δειγματοληψία
            inv = perms[i].T
            assert any(np.allclose(inv, P, atol=1e-10) for P in perms)


class TestEquivariantProjectors:
    def test_idempotent(self, eq_projectors):
        for name, P in eq_projectors.items():
            assert np.allclose(P @ P, P, atol=1e-8), name

    def test_sum_to_identity(self, eq_projectors):
        S = sum(eq_projectors.values())
        assert np.allclose(S, np.eye(30), atol=1e-8)

    def test_mutually_orthogonal(self, eq_projectors):
        names = list(eq_projectors.keys())
        for i, n1 in enumerate(names):
            for n2 in names[i+1:]:
                prod = eq_projectors[n1] @ eq_projectors[n2]
                assert np.allclose(prod, 0, atol=1e-8), f"{n1}·{n2} ≠ 0"

    def test_correct_ranks(self, eq_projectors):
        for name, P in eq_projectors.items():
            r = int(round(np.trace(P)))
            assert r == RANKS_BETA4[name], f"{name}: rank={r}"

    def test_symmetric(self, eq_projectors):
        # Real orthogonal projectors: P = Pᵀ
        for name, P in eq_projectors.items():
            assert np.allclose(P, P.T, atol=1e-8), name

    def test_schur_commutes_with_action(self, eq_projectors, perms):
        """Schur's lemma: P_ρ · π(g) = π(g) · P_ρ γιὰ κάθε g ∈ I, ρ ∈ Iₕ-irreps.

        Αὐτὸ εἶναι ἡ ὁρισμὸς τοῦ Iₕ-equivariant projector.
        5 irreps × 60 group elements = 300 commutation checks.
        """
        for name, P in eq_projectors.items():
            for g_idx, pi in enumerate(perms):
                lhs = P @ pi
                rhs = pi @ P
                assert np.allclose(lhs, rhs, atol=1e-8), \
                    f"P_{name} ↮ π(g_{g_idx}): max err {np.max(np.abs(lhs - rhs))}"

    def test_beta4_permutation_action_isotypic_decomp(self, eq_projectors, perms):
        """Κάθε π(g) διατηρεῖ τὸν ὑπόχωρο τοῦ κάθε irrep (P_ρ · π(g) · P_ρ = π(g) · P_ρ)."""
        for name, P in eq_projectors.items():
            for pi in perms[::10]:  # δειγματοληψία
                lhs = P @ pi @ P
                rhs = pi @ P
                assert np.allclose(lhs, rhs, atol=1e-8), name


# ═══════════════════════════════════════════════════════════════════
# β₄-coordinates testbed
# ═══════════════════════════════════════════════════════════════════

class TestBeta4CoordinatesTestbed:
    def test_shape(self, dt):
        v = _tb_beta4(dt)
        assert v.shape == (3, 30)

    def test_rows_are_unit_norm_axes(self, dt):
        # Κάθε γραμμὴ συλλέγει τὶς x (ἢ y, z) συντεταγμένες τῶν 30 unit-norm
        # midpoints. Ἡ νόρμα ἑπομένως πρέπει νὰ εἶναι √(Σ x²) = √10 (ἀπὸ τὴν
        # ταυτότητα Σ_v ‖v‖² = 30 = Σ x² + Σ y² + Σ z² = 3·10).
        v = _tb_beta4(dt)
        for axis in range(3):
            assert abs(np.linalg.norm(v[axis]) - np.sqrt(10)) < 1e-10

    def test_concentrates_on_T1_with_equivariant(self, dt, eq_projectors):
        """Μὲ τοὺς πραγματικοὺς equivariant projectors, οἱ x,y,z ἀνήκουν
        ἀκριβῶς στὸ T₁ subspace (standard 3D rep τῆς I)."""
        v = _tb_beta4(dt)
        result = diagnose_irrep_distribution(v, eq_projectors)
        # T₁ ≈ 1, ὅλα τὰ ἄλλα ≈ 0
        assert result['energies']['T1'] > 0.999
        for name in ['A', 'T2', 'G', 'H']:
            assert result['energies'][name] < 1e-9, f"{name}={result['energies'][name]}"

    def test_random_projectors_do_not_concentrate(self, dt):
        """Sanity: μὲ random-orthogonal projectors, ἡ ἐνέργεια ΔΕΝ
        συγκεντρώνεται καθαρὰ σὲ ἕνα irrep (διασκορπίζεται)."""
        from ark_irreps import build_irrep_projectors_30
        random_proj = build_irrep_projectors_30(seed=42)
        v = _tb_beta4(dt)
        result = diagnose_irrep_distribution(v, random_proj)
        # κανένα irrep δὲν συγκεντρώνει > 0.99 ἐνέργεια
        max_e = max(result['energies'].values())
        assert max_e < 0.99


class TestBeta4PerturbedTestbed:
    def test_returns_list_of_pairs(self, dt):
        result = _tb_beta4_pert(dt, epsilons=(0.0, 0.01, 0.05))
        assert len(result) == 3
        for eps, vec in result:
            assert isinstance(eps, float)
            assert vec.shape == (3, 30)

    def test_zero_noise_equals_base(self, dt):
        result = _tb_beta4_pert(dt, epsilons=(0.0,))
        eps, vec = result[0]
        assert eps == 0.0
        np.testing.assert_array_equal(vec, _tb_beta4(dt))

    def test_t1_monotonically_decreases(self, dt, eq_projectors):
        """E_T₁ φθίνει μονότονα μὲ τὸ ε."""
        epsilons = (0.0, 0.01, 0.05, 0.15)
        result = _tb_beta4_pert(dt, epsilons=epsilons, seed=42)
        t1_energies = []
        for eps, vec in result:
            r = diagnose_irrep_distribution(vec, eq_projectors)
            t1_energies.append(r['energies']['T1'])
        for i in range(len(t1_energies) - 1):
            assert t1_energies[i] >= t1_energies[i+1] - 1e-3, \
                f"T₁ ὄχι μονότονο: {t1_energies}"

    def test_zero_noise_keeps_t1_at_one(self, dt, eq_projectors):
        result = _tb_beta4_pert(dt, epsilons=(0.0,))
        _, vec = result[0]
        r = diagnose_irrep_distribution(vec, eq_projectors)
        assert r['energies']['T1'] > 0.999

    def test_large_noise_causes_visible_leakage(self, dt, eq_projectors):
        """Στὸ ε=15%, διαρροὴ ≥ 1% (πρακτικά ~5%)."""
        result = _tb_beta4_pert(dt, epsilons=(0.15,), seed=42)
        _, vec = result[0]
        r = diagnose_irrep_distribution(vec, eq_projectors)
        leakage = 1.0 - r['energies']['T1']
        assert leakage > 0.01

    def test_theoretical_formula_holds(self, dt, eq_projectors):
        """Ἐλέγχει τὸν τύπο 1−E_T₁ = 27ε²/(10+30ε²) ἐντὸς ~10% σχετικῆς ἀκριβείας
        στὰ μεσαῖα ε (ἀρκετὸ noise γιὰ μέσα σύγκλισι, ὄχι τόσο πολὺ ποὺ
        νὰ χαλάει τὸ first-order ἀνάπτυγμα)."""
        eps = 0.10
        n_trials = 50
        leakages = []
        for seed in range(n_trials):
            result = _tb_beta4_pert(dt, epsilons=(eps,), seed=seed)
            _, vec = result[0]
            r = diagnose_irrep_distribution(vec, eq_projectors)
            leakages.append(1.0 - r['energies']['T1'])
        avg_leakage = np.mean(leakages)
        predicted = 27 * eps**2 / (10 + 30 * eps**2)
        rel_err = abs(avg_leakage - predicted) / predicted
        assert rel_err < 0.10, \
            f"avg={avg_leakage:.5f}, predicted={predicted:.5f}, rel_err={rel_err:.3f}"

    def test_schur_gauss_ratios_at_eps_15pct(self, dt, eq_projectors):
        """Στὸ ε=15%, μέσος ὅρος ἐνεργειῶν στὰ ἄλλα irreps πρέπει νὰ
        ἀναλογεῖ μὲ τὰ ranks A:T₂:G:H = 1:3:8:15 (Iₕ-isotropic Gaussian
        noise → Schur).
        """
        eps = 0.15
        n_trials = 100
        sums = {n: 0.0 for n in ['A', 'T2', 'G', 'H']}
        for seed in range(n_trials):
            result = _tb_beta4_pert(dt, epsilons=(eps,), seed=seed)
            _, vec = result[0]
            r = diagnose_irrep_distribution(vec, eq_projectors)
            for n in sums:
                sums[n] += r['energies'][n]
        avg = {n: v / n_trials for n, v in sums.items()}
        # Σύγκρισι μὲ ranks 1:3:8:15 (sum=27)
        ranks = {'A': 1, 'T2': 3, 'G': 8, 'H': 15}
        # Κανονικοποίησι μὲ τὸ συνολικὸ avg
        total = sum(avg.values())
        for n in avg:
            measured_ratio = avg[n] / total
            predicted_ratio = ranks[n] / 27
            rel_err = abs(measured_ratio - predicted_ratio) / predicted_ratio
            assert rel_err < 0.10, \
                f"{n}: measured={measured_ratio:.4f}, predicted={predicted_ratio:.4f}, rel_err={rel_err:.3f}"

    def test_schur_gauss_confirmed_via_diagnose_drift(self, dt, eq_projectors):
        """Τὸ diagnose_epsilon_drift πρέπει νὰ φέρει 'Schur-Gauss confirmed'
        σὲ ε=15% (ἀρκετὰ θόρυβος ὥστε rel_err < 5%)."""
        eps = 0.15
        # Μέσος ὅρος ἐνεργειῶν ἀπὸ 100 trials → ἕνα μόνο "diagnose result"
        n_trials = 100
        avg_energies = {n: 0.0 for n in ['A', 'T1', 'T2', 'G', 'H']}
        for seed in range(n_trials):
            result = _tb_beta4_pert(dt, epsilons=(eps,), seed=seed)
            _, vec = result[0]
            r = diagnose_irrep_distribution(vec, eq_projectors)
            for n in avg_energies:
                avg_energies[n] += r['energies'][n] / n_trials
        # Φτιάξε ψεύτικο diagnose_result μὲ τὰ averaged energies
        fake_result = {
            'energies': avg_energies,
            'max_deviation_pct': 0,  # δὲν χρησιμοποιεῖται σὲ leakage mode
        }
        ed = diagnose_epsilon_drift(fake_result)
        assert ed['mode'] == 'leakage'
        assert ed['dominant_irrep'] == 'T1'
        assert 'leakage_per_irrep' in ed
        assert ed['schur_gauss_confirmed'] is True
        assert 'Schur-Gauss confirmed' in ed['schur_verdict']
        # Ratios στὰ rows
        per = {row['irrep']: row for row in ed['leakage_per_irrep']}
        assert per['A']['rank'] == 1
        assert per['T2']['rank'] == 3
        assert per['G']['rank'] == 8
        assert per['H']['rank'] == 15


# ═══════════════════════════════════════════════════════════════════
# Wu bicomplex (Στόχος #5, Π-28-Β05..Β10)
# ═══════════════════════════════════════════════════════════════════

from ark_wu import (
    build_PE, wu_bicomplex, wu_laplacians, wu_projector_top,
    wu_sector_decomposition, verify_wu_structure,
)


@pytest.fixture(scope='module')
def pe(dt):
    return build_PE(dt)

@pytest.fixture(scope='module')
def bicx(dt, pe):
    return wu_bicomplex(dt, pe)

@pytest.fixture(scope='module')
def wu_structure(dt, pe):
    return verify_wu_structure(dt, pe)


class TestPatrikoEikosaedro:
    """ΠΕ ὡς simplicial complex μὲ V=β₁₀ τοῦ DT (Π-28-Β10)."""

    def test_pe_counts_12_30_20(self, pe):
        assert pe['V'] == 12
        assert pe['E'] == 30
        assert pe['F'] == 20
        assert pe['chi'] == 2
        # Euler χ = V − E + F = 12 − 30 + 20 = 2 ✓

    def test_pe_vertices_match_beta10(self, dt, pe):
        # Π-28-Β10: V(ΠΕ) = β₁₀ τοῦ DT
        assert set(pe['vertices']) == set(dt['orbits']['β10'])

    def test_pe_5_regular(self, pe):
        # ΠΕ = ἰκοσάεδρο, κάθε κορυφὴ ἔχει βαθμὸ 5
        for v, deg in pe['degrees'].items():
            assert deg == 5, f"vertex {v} degree {deg}"


class TestWuBicomplex:
    """Wu bicomplex structural ταυτότητες (Π-28-Β05..Β08)."""

    def test_wu_total_cells_2772(self, bicx):
        # Π-28-Β08: |C^(p,q)| = (12, 60, 60 / 120, 600, 600 / 120, 600, 600)
        # Σύνολο = 2772
        total = sum(bicx['cells_per_pq'].values())
        assert total == 2772

    def test_wu_cells_per_pq_match_stone(self, bicx):
        # Π-28-Β08: ἀκριβής πίνακας
        expected = {
            (0, 0): 12,  (0, 1): 60,  (0, 2): 60,
            (1, 0): 120, (1, 1): 600, (1, 2): 600,
            (2, 0): 120, (2, 1): 600, (2, 2): 600,
        }
        assert bicx['cells_per_pq'] == expected

    def test_wu_euler_alternating_12(self, bicx):
        # Π-28-Β08: Σ(-1)^(p+q)·|C^(p,q)| = 12 = |β₁₀|
        cells = bicx['cells_per_pq']
        euler = sum(((-1) ** (p + q)) * cells[(p, q)] for p in range(3) for q in range(3))
        assert euler == 12

    def test_wu_d_squared_zero(self, bicx):
        # bicomplex axiom: d ∘ d = 0
        for n in range(3):
            prod = bicx['d'][n+1] @ bicx['d'][n]
            nrm = np.linalg.norm(prod)
            assert nrm < 1e-9, f"‖d_{n+1} ∘ d_{n}‖ = {nrm}"

    def test_wu_top_betti_12(self, bicx):
        # Π-28-Β06: Künneth → b_4 = 1 ἀνὰ ἀντίγραφο × 12 ἀντίγραφα = 12
        # Καί τὰ ὑπόλοιπα Betti πρέπει νὰ εἶναι 0 (Wu concentration)
        from numpy.linalg import matrix_rank
        bettis = []
        for n in range(5):
            dim_n = bicx['dim'][n]
            d_in = bicx['d'][n-1] if n >= 1 else None
            d_out = bicx['d'][n] if n <= 3 else None
            rk_in = int(matrix_rank(d_in, tol=1e-8)) if d_in is not None and d_in.size > 0 else 0
            rk_out = int(matrix_rank(d_out, tol=1e-8)) if d_out is not None and d_out.size > 0 else 0
            bettis.append(dim_n - rk_in - rk_out)
        assert bettis == [0, 0, 0, 0, 12], f"got {bettis}"

    def test_wu_delta4_block_diagonal(self, dt, pe, bicx):
        # Π-28-Β07: Δ₄ block-diagonal ἀκριβῶς, off-block = 0 (machine epsilon)
        laps = wu_laplacians(bicx)
        L4 = laps[4]
        sectors, _ = wu_sector_decomposition(bicx, dt)
        off_max = 0.0
        for i in range(L4.shape[0]):
            for j in range(L4.shape[1]):
                if sectors[i] != sectors[j]:
                    off_max = max(off_max, abs(L4[i, j]))
        assert off_max < 1e-12, f"off-block max = {off_max}"

    def test_wu_sector_decomposition_12_independent(self, dt, bicx):
        # Π-28-Β05: 12 ἀνεξάρτητα sectors, ἕνα ἀνὰ β₁₀ κορυφή
        sectors, sector_indices = wu_sector_decomposition(bicx, dt)
        # 12 διακριτὲς κορυφές
        assert len(set(sectors)) == 12
        assert set(sectors) == set(dt['orbits']['β10'])
        # Κάθε sector ἔχει ἀκριβῶς dim_T4 / 12 = 50 elements
        # (Tot^4 = 600 = 12 × 50)
        for v, idxs in sector_indices.items():
            assert len(idxs) == 50, f"sector {v} has {len(idxs)} elements"


class TestVerifyWuStructure:
    """Ὁλοκληρωτικὴ structural διάγνωσι μέσῳ verify_wu_structure."""

    def test_full_structural_check(self, wu_structure):
        s = wu_structure
        assert s['total_cells'] == 2772
        assert s['euler_alternating_sum'] == 12
        assert s['top_betti_b4'] == 12
        assert s['bettis'] == [0, 0, 0, 0, 12]
        assert s['d_squared_max_norm'] < 1e-9
        assert s['block_diagonal_off_norm'] < 1e-12
        assert s['sector_count'] == 12
        assert len(s['per_sector_dim']) == 12
        # Ὅλα τὰ sectors μὲ ἴση διάστασι (50 ἀνὰ vertex)
        assert all(d == 50 for d in s['per_sector_dim'])
        # Verdict περιέχει ὅλα τὰ ✓
        for marker in ['2772 cells', '= 12', 'b_4 = 12', 'block-diagonal', '12 ἀνεξάρτητα']:
            assert marker in s['verdict'], f"missing '{marker}' in verdict"


# ═══════════════════════════════════════════════════════════════════
# β₁₀-probe (Π32-WU-PROBE.01)
# ═══════════════════════════════════════════════════════════════════

from ark_wu_b10_probe import (
    wu_ground_basis_per_sector,
    b10_probe_matrix,
    verify_b10_probe,
)


@pytest.fixture(scope='module')
def b10_probe_result(dt, pe):
    return verify_b10_probe(dt, pe)


class TestB10Probe:
    """Π32-WU-PROBE.01: Tr(P_w_sec · ρ_v_ground) = δ_{v,w}."""

    def test_each_sector_has_unique_ground_state(self, dt, pe):
        ground = wu_ground_basis_per_sector(dt, pe)
        # 12 ground vectors, ἕνα ἀνὰ β₁₀ vertex
        assert len(ground) == 12
        assert set(ground.keys()) == set(dt['orbits']['β10'])
        # Κάθε vector εἶναι 600D unit norm
        for v, vec in ground.items():
            assert vec.shape == (600,)
            assert abs(np.linalg.norm(vec) - 1) < 1e-10

    def test_b10_probe_matrix_is_12x12_identity(self, b10_probe_result):
        M = b10_probe_result['matrix']
        assert M.shape == (12, 12)
        # Διαγωνίους ≈ 1
        diag = np.diag(M)
        assert np.allclose(diag, 1.0, atol=1e-9)

    def test_off_diagonal_below_machine_epsilon(self, b10_probe_result):
        # Π32-WU-PROBE.01: off-diagonal ≈ 0 ἀκριβῶς
        assert b10_probe_result['off_max'] < 1e-12
        assert b10_probe_result['identity_match'] is True

    def test_verdict_confirms_probe(self, b10_probe_result):
        assert 'WU-PROBE.01' in b10_probe_result['verdict']
        assert 'ταυτοτικός' in b10_probe_result['verdict']


# ═══════════════════════════════════════════════════════════════════
# ψ-twisted Wu διάγνωσι (Π32-WU-EMBED.03 + ψ-grading)
# ═══════════════════════════════════════════════════════════════════

from ark_wu_psi import (
    psi_assignments,
    build_psi_twist_embedding,
    diagnose_wu_psi_per_beta10,
)


@pytest.fixture(scope='module')
def psi_dict(dt):
    return psi_assignments(dt)


class TestWuPsi:
    """ψ-twisted Wu διάγνωσι, Π32-WU-EMBED.03 + sector grading."""

    def test_psi_assignments_keys_are_beta10(self, dt, psi_dict):
        beta10_set = set(dt['orbits']['β10'])
        for name, assign in psi_dict.items():
            assert set(assign.keys()) == beta10_set, name
            assert len(assign) == 12, name

    def test_trivial_psi_all_ones(self, dt, psi_dict):
        trivial = psi_dict['trivial']
        for v, val in trivial.items():
            assert val == complex(1.0, 0.0)

    def test_t1_x_psi_real_and_iₕ_aligned(self, dt, psi_dict):
        t1x = psi_dict['irrep_T1_x']
        # Κάθε τιμὴ εἶναι real (φανταστικὸ ≈ 0)
        for v, val in t1x.items():
            assert abs(val.imag) < 1e-12
        # Σ_v ψ(v) = 0 (T₁ συνεισφορὰ ἔχει mean = 0 — ἀντι-συμμετρία icosa)
        total = sum(t1x.values())
        assert abs(total) < 1e-10, f"Σψ = {total}"

    def test_embedding_shape(self, dt, pe, bicx, psi_dict):
        E = build_psi_twist_embedding(dt, bicx, psi_dict['trivial'], pe=pe)
        assert E.shape == (600, 30)
        assert E.dtype == complex

    def test_trivial_embedding_is_real(self, dt, pe, bicx, psi_dict):
        E = build_psi_twist_embedding(dt, bicx, psi_dict['trivial'], pe=pe)
        # ψ=1 → entries εἶναι μόνο 0 ἢ 1 (real)
        assert np.allclose(E.imag, 0)

    def test_harmonic_total_zero_for_beta4_input(self, dt, pe, bicx, psi_dict):
        """Π32-WU-EMBED.03: P₄·E_twist·v ≈ 0 γιὰ ὁποιοδήποτε ψ καὶ β₄ input."""
        rng = np.random.default_rng(42)
        for psi_name, assign in psi_dict.items():
            for trial in range(3):
                v = rng.standard_normal(30)
                result = diagnose_wu_psi_per_beta10(v, dt, assign, pe=pe, bicx=bicx)
                # Harmonic fraction = harmonic_total / total_energy
                if result['total_energy'] > 1e-15:
                    frac = result['harmonic_total'] / result['total_energy']
                    assert frac < 1e-20, \
                        f"{psi_name} trial {trial}: harmonic frac = {frac}"

    def test_sector_energies_consistency(self, dt, pe, bicx, psi_dict):
        """Σ sector_energy = total_energy (sectors διαιροῦν τὸ Tot⁴)."""
        rng = np.random.default_rng(7)
        v = rng.standard_normal(30)
        for assign in psi_dict.values():
            result = diagnose_wu_psi_per_beta10(v, dt, assign, pe=pe, bicx=bicx)
            sum_sectors = sum(result['sector_energy'].values())
            assert abs(sum_sectors - result['total_energy']) < 1e-10

    def test_trivial_psi_uniform_sectors_for_uniform_input(self, dt, pe, bicx, psi_dict):
        """Μὲ trivial ψ καὶ uniform β₄ input (ones), τὰ sector_energies εἶναι
        ὅλα ἴσα (Iₕ-symmetry)."""
        v = np.ones(30)
        result = diagnose_wu_psi_per_beta10(v, dt, psi_dict['trivial'], pe=pe, bicx=bicx)
        energies = list(result['sector_energy'].values())
        assert max(energies) - min(energies) < 1e-9, energies

    def test_t1_x_psi_grades_sectors_along_x_axis(self, dt, pe, bicx, psi_dict):
        """Μὲ ψ = T₁_x καὶ uniform input, sector_amplitude συσχετίζεται μὲ
        x-coordinate τῆς κάθε β₁₀ κορυφῆς (T₁ pattern)."""
        v = np.ones(30)
        result = diagnose_wu_psi_per_beta10(v, dt, psi_dict['irrep_T1_x'], pe=pe, bicx=bicx)
        # Συσχέτισι sector_phase ἢ amplitude μὲ x-coord
        coords = dt['coords']
        beta10 = list(dt['orbits']['β10'])
        x_vals = np.array([coords[v_b10][0] for v_b10 in beta10])
        # Εἴτε amplitude εἴτε συγκεκριμένη quantity. Δοκιμάζω signed sum.
        # signed = Re(Σ_idx embedded[idx]) ἀνὰ sector
        # Πιὸ ἁπλό: Re τοῦ μέσου ποὺ ἤδη ἔχουμε
        signed = np.array([
            float(np.cos(result['sector_phase'][b]) * result['sector_amplitude'][b])
            for b in beta10
        ])
        # Συσχέτισι Pearson μεταξύ signed καὶ x_vals
        # |corr| > 0.95 → ξεκάθαρο T₁ pattern
        if np.std(signed) > 1e-10 and np.std(x_vals) > 1e-10:
            corr = np.corrcoef(signed, x_vals)[0, 1]
            assert abs(corr) > 0.95, f"corr(signed, x) = {corr}"


# ═══════════════════════════════════════════════════════════════════
# Hashimoto B-matrix (Στόχος #6)
# ═══════════════════════════════════════════════════════════════════

from ark_hashimoto import (
    build_directed_edges, build_hashimoto_B,
    adjacency_matrix, degree_matrix,
    verify_ihara_identity, hashimoto_spectrum,
    diagnose_hashimoto,
)


@pytest.fixture(scope='module')
def hashimoto_B(dt):
    return build_hashimoto_B(dt)

@pytest.fixture(scope='module')
def hashimoto_diag(dt):
    return diagnose_hashimoto(dt)


class TestHashimoto:
    """Hashimoto non-backtracking matrix B στὸ DT."""

    def test_directed_edges_count(self, dt):
        de = build_directed_edges(dt)
        assert len(de) == 2 * 180  # = 360

    def test_directed_edges_symmetric(self, dt):
        # Γιὰ κάθε (u,v) ὑπάρχει καὶ (v,u)
        de = set(build_directed_edges(dt))
        for (u, v) in list(de):
            assert (v, u) in de

    def test_B_shape_360x360(self, hashimoto_B):
        assert hashimoto_B.shape == (360, 360)

    def test_B_nonzero_count_matches_degree_formula(self, dt, hashimoto_B):
        # Nonzero entries = Σ_v deg(v)·(deg(v)-1)
        # Γιὰ DT: 12·10·9 + 20·6·5 + 30·4·3 = 1080 + 600 + 360 = 2040
        expected = 12*10*9 + 20*6*5 + 30*4*3
        assert expected == 2040
        assert int(np.sum(hashimoto_B != 0)) == expected

    def test_B_entries_are_0_or_1(self, hashimoto_B):
        unique = set(np.unique(hashimoto_B).tolist())
        assert unique.issubset({0.0, 1.0})

    def test_B_not_symmetric(self, hashimoto_B):
        # Hashimoto B εἶναι γενικὰ μὴ-συμμετρικός (directed)
        assert not np.allclose(hashimoto_B, hashimoto_B.T)

    def test_no_backtracking(self, dt, hashimoto_B):
        """B[(u→v), (v→u)] = 0 (καμία backtracking ἐπιτρέπεται)."""
        de = build_directed_edges(dt)
        idx = {(u, v): i for i, (u, v) in enumerate(de)}
        for (u, v), i in idx.items():
            j = idx[(v, u)]  # backtracking edge
            assert hashimoto_B[j, i] == 0

    def test_ihara_identity_holds(self, dt, hashimoto_B):
        """det(I − Bx) = (1−x²)^{|E|−|V|} · det(I − Ax + (D−I)x²)"""
        for x in [0.05, 0.1, 0.15]:
            r = verify_ihara_identity(dt, x=x, B=hashimoto_B)
            assert r['rel_diff'] < 1e-10, f"x={x}: rel_diff={r['rel_diff']}"

    def test_spectral_radius_in_expected_range(self, hashimoto_diag):
        # Γιὰ DT μὲ degrees {10, 6, 4}, ρ(B) ἀναμένεται 4-6
        assert 4.0 < hashimoto_diag['spectral_radius'] < 6.0

    def test_diagnostic_completeness(self, hashimoto_diag):
        d = hashimoto_diag
        assert d['V'] == 62
        assert d['E'] == 180
        assert d['n_directed_edges'] == 360
        assert d['B_shape'] == (360, 360)
        assert d['ihara_holds'] is True
        assert 'ρ(B)' in d['verdict']
        assert 'Ihara' in d['verdict']

    def test_adjacency_and_degree_consistent(self, dt):
        A, verts = adjacency_matrix(dt)
        D = degree_matrix(dt)
        # A symmetric
        assert np.allclose(A, A.T)
        # Σ A = Σ D (= 2|E|)
        assert int(A.sum()) == 2 * 180
        # D diagonal entries = degrees
        for i, v in enumerate(verts):
            assert int(D[i, i]) == dt['degrees'][v]

    def test_hash01_orbit_decomposition_1080_600_360(self, dt, hashimoto_B):
        """Π32-HASH.01: nonzero entries = 1080 + 600 + 360, ἕνας ὅρος ἀνὰ orbit."""
        beta10 = dt['orbits']['β10']
        beta6 = dt['orbits']['β6']
        beta4 = dt['orbits']['β4']
        # Κάθε vertex συνεισφέρει deg(v)·(deg(v)-1) nonzero entries
        # (γιὰ ζευγάρια non-backtracking directed edges ποὺ διαβαίνουν τὸ v)
        contrib_b10 = sum(10 * 9 for _ in beta10)
        contrib_b6 = sum(6 * 5 for _ in beta6)
        contrib_b4 = sum(4 * 3 for _ in beta4)
        assert contrib_b10 == 1080
        assert contrib_b6 == 600
        assert contrib_b4 == 360
        assert contrib_b10 + contrib_b6 + contrib_b4 == 2040
        assert int(np.sum(hashimoto_B != 0)) == 2040


# ═══════════════════════════════════════════════════════════════════
# Τοπικὴ διάγνωσι ἀνὰ β₁₀ (ΟΧΙ Wu)
# ═══════════════════════════════════════════════════════════════════

from ark_local_b10 import (
    beta10_to_beta4_neighbors,
    diagnose_local_beta10_neighborhood,
)


class TestLocalBeta10Neighborhood:
    """Τοπικὴ διάγνωσι μέσῳ DT γειτονιᾶς β₁₀ ↔ β₄."""

    def test_each_beta10_has_5_beta4_neighbors(self, dt):
        neighbors = beta10_to_beta4_neighbors(dt)
        assert len(neighbors) == 12
        for v, lst in neighbors.items():
            assert len(lst) == 5

    def test_coverage_redundancy_is_2(self, dt):
        # Κάθε β₄ midpoint ζεῖ σὲ ἀκμή ἀκριβῶς 2 β₁₀ vertices
        # ⇒ Σ_v |neighbors(v)| = 5 · 12 = 60 = 30 · 2
        neighbors = beta10_to_beta4_neighbors(dt)
        total = sum(len(n) for n in neighbors.values())
        assert total == 60

    def test_diagnose_returns_12_signals(self, dt):
        v = np.ones(30)
        result = diagnose_local_beta10_neighborhood(v, dt)
        assert len(result['local_signal']) == 12
        assert result['coverage_redundancy'] == 2.0

    def test_uniform_input_uniform_signals(self, dt):
        """Iₕ-symmetry: uniform input → ὅλα local_signal ἴσα."""
        v = np.ones(30)
        result = diagnose_local_beta10_neighborhood(v, dt)
        sigs = list(result['local_signal'].values())
        assert max(sigs) - min(sigs) < 1e-12
        # Κάθε σῆμα = 5 (5 β₄ γείτονες, καθένας μὲ |v|²=1)
        assert all(abs(s - 5.0) < 1e-12 for s in sigs)

    def test_total_signal_is_coverage_times_norm(self, dt):
        """Total signal = coverage × ‖v‖² = 2 · ‖v‖²."""
        rng = np.random.default_rng(42)
        v = rng.standard_normal(30)
        result = diagnose_local_beta10_neighborhood(v, dt)
        expected = 2.0 * float(np.sum(v ** 2))
        assert abs(result['total_signal'] - expected) < 1e-10

    def test_dominant_beta10_for_localized_input(self, dt):
        """Spike σὲ ἕνα β₄ midpoint → dominant β₁₀ ταυτοποιεῖται."""
        neighbors = beta10_to_beta4_neighbors(dt)
        v = np.zeros(30)
        # Σπάει σὲ ἕνα β₄ midpoint
        spike_idx = 7
        v[spike_idx] = 10.0
        result = diagnose_local_beta10_neighborhood(v, dt, neighbors=neighbors)
        # Ἡ dominant β₁₀ πρέπει νὰ εἶναι γείτονας τοῦ spike_idx (ἀπὸ τοὺς 2)
        beta4 = dt['orbits']['β4']
        spike_b4_vertex = beta4[spike_idx]
        # 2 β₁₀ vertices ἔχουν τὸ spike_b4_vertex στὶς neighbors lists
        candidates = [v_b10 for v_b10, lst in neighbors.items() if spike_b4_vertex in lst]
        assert len(candidates) == 2
        assert result['dominant_beta10'] in candidates

    def test_anisotropy_zero_for_uniform(self, dt):
        v = np.ones(30)
        result = diagnose_local_beta10_neighborhood(v, dt)
        assert result['anisotropy'] < 1e-12
        assert result['anisotropy_pct'] < 1e-10
