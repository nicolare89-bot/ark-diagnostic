"""
ark_main.py — Orchestration, CLI, formatting
==============================================

Πῶς τρέχει:
  uv run python ark_main.py            # ὅλα τὰ testbeds
  uv run python ark_main.py --json     # JSON output
  uv run python ark_main.py --help     # ἐπιλογὲς

Στάδιο 2 (split ἀπὸ ARK_DIAGNOSTIC_v0.py).
V − E + F = 2.   ε = 1.5%.   J > 0.
Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.
"""

import sys
import json
import argparse
import numpy as np
from numpy.linalg import norm

# UTF-8 stdout γιὰ Windows console (πολυτονικὰ Ἑλληνικὰ).
# Ἀπαραίτητο γιὰ τὸ frozen .exe καὶ γιὰ direct python run χωρὶς
# PYTHONIOENCODING=utf-8 env var.
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass

from ark_geometry import (
    PHI, RATIO, EPSILON, EPSILON_NOMINAL, BETA10,
    build_DT, build_24cell, build_600cell,
)
from ark_irreps import (
    RANKS_BETA4,
    build_irrep_projectors_30,
    build_irrep_projectors_30_equivariant,
)
from ark_diagnostics import (
    diagnose_irrep_distribution, diagnose_cheeger,
    diagnose_epsilon_drift, diagnose_F_balance,
    testbed_random_isotropic, testbed_ih_aligned, testbed_asymmetric,
    testbed_dt_beta4_coordinates,
)
from ark_state import save_report


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

    lines.append("\n[3] ε-DRIFT (ἀκριβὲς ε* + 5 ζῶνες)")
    ed = d['epsilon_drift']
    drift_str = '∞' if ed['drift_fraction'] == float('inf') else f"{ed['drift_fraction']*100:.4f}%"
    lines.append(f"    Mode: {ed['mode']}  |  ε* = {ed['epsilon_star_pct']:.6f}%  |  drift (ε) = {drift_str}")
    if ed['mode'] == 'leakage':
        lines.append(f"    Dominant irrep: {ed['dominant_irrep']} (E={ed['dominant_energy']:.6f})")
        eps_imp = ed.get('epsilon_implied', 0)
        eps_imp_str = '∞' if eps_imp == float('inf') else f"{eps_imp*100:.4f}%"
        lines.append(f"    Leakage: {ed['leakage_fraction']*100:.4f}%  |  ε_implied = {eps_imp_str}")
        if ed.get('ih_isotropic_floor'):
            lines.append(f"    {ed['floor_note']}")
    lines.append(f"    Within living imperfection: {ed['within_living_imperfection']}")
    cyc = ed['predicted_cycles_to_collapse']
    if cyc == float('inf') or cyc == 'inf':
        lines.append(f"    Cycles to collapse: ∞ (stable)")
    else:
        lines.append(f"    Cycles to collapse: {cyc}")
    lines.append(f"    Verdict: {ed['verdict']}")
    if 'leakage_per_irrep' in ed:
        lines.append(f"    {'irrep':<6}{'rank':<6}{'measured':<14}{'Schur pred.':<14}{'rel.err':<10}")
        for row in ed['leakage_per_irrep']:
            lines.append(f"    {row['irrep']:<6}{row['rank']:<6}{row['measured']:<14.6f}{row['predicted_schur']:<14.6f}{row['rel_err']*100:.2f}%")
        lines.append(f"    {ed['schur_verdict']}")

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
        description='ARK_DIAGNOSTIC — Κιβωτικὸ Διαγνωστικὸ Σύστημα',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='V − E + F = 2.   ε = 1.5%.   J > 0.   Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.'
    )
    parser.add_argument('--json', action='store_true', help='JSON output')
    parser.add_argument('--testbed', choices=['random', 'ih', 'asymmetric', 'beta4', 'all'],
                        default='all')
    parser.add_argument('--samples', type=int, default=200,
                        help='samples γιὰ random testbed (default 200)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--build-600cell', action='store_true',
                        help='Κατασκευή 600-cell (ἀργό, ~10s)')
    parser.add_argument('--equivariant', action='store_true',
                        help='Iₕ-equivariant projectors μέσῳ permutation rep '
                             '(default: random-orthogonal block, συμβατότητα v0)')
    parser.add_argument('--save-report', type=str, default=None,
                        metavar='PATH',
                        help='Ἀποθήκευσι ὅλων τῶν reports σὲ JSON ἀρχεῖο')
    args = parser.parse_args()

    if not args.json:
        print("=" * 70)
        print("  ARK_DIAGNOSTIC — Κιβωτικὸ Διαγνωστικὸ Σύστημα (modular)")
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

    if args.equivariant:
        projectors = build_irrep_projectors_30_equivariant(DT)
        proj_method = 'equivariant (Schur-commuting)'
    else:
        projectors = build_irrep_projectors_30(seed=args.seed)
        proj_method = 'random-orthogonal block (v0 fallback)'

    if not args.json:
        ranks_str = ', '.join(f'{n}={RANKS_BETA4[n]}' for n in ['A','T1','T2','G','H'])
        print(f"  ✓ Iₕ-projectors (β₄, {proj_method}): {{{ranks_str}}} = {sum(RANKS_BETA4.values())}")
        print(f"  ✓ ε* = {EPSILON*100:.6f}% (display: {EPSILON_NOMINAL*100}%)  |  RATIO = {RATIO:.4f}  |  PHI = {PHI:.4f}")

    testbeds = {}
    if args.testbed in ('random', 'all'):
        testbeds[f'Τυχαία Gaussian ({args.samples} samples)'] = testbed_random_isotropic(n_samples=args.samples, seed=args.seed)
    if args.testbed in ('ih', 'all'):
        testbeds['Iₕ-aligned σὲ G-irrep (κενωτικό, dim=4, mult=2, rank=8)'] = testbed_ih_aligned(projectors, 'G', seed=args.seed)
        testbeds['Iₕ-aligned σὲ H-irrep (πλῆρες, dim=5, mult=3, rank=15)'] = testbed_ih_aligned(projectors, 'H', seed=args.seed)
    if args.testbed in ('asymmetric', 'all'):
        testbeds['Ἀσύμμετρη (5 spikes)'] = testbed_asymmetric(seed=args.seed)
    if args.testbed in ('beta4', 'all'):
        testbeds['DT β₄ coordinates (x,y,z τῶν 30 midpoints)'] = testbed_dt_beta4_coordinates(DT)

    all_reports = []
    for label, vec in testbeds.items():
        report = run_full_diagnostic(label, vec, projectors, DT, CELL24)
        all_reports.append(report)

    if args.save_report:
        save_report(all_reports, args.save_report)
        if not args.json:
            print(f"\n  ✓ Reports saved → {args.save_report}")

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
