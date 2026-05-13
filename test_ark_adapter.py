"""
test_ark_adapter.py — Tests γιὰ τὸν Universal Adapter
======================================================

Φάσι 1 τοῦ Σταδίου 4 (PWA).
Στόχος: 10-12+ πράσινα tests γιὰ ὅλες τὶς στρατηγικές + auto-detect.
"""

import numpy as np
import pytest

from ark_adapter import (
    TARGET_DIM, VALID_STRATEGIES,
    direct, pad_to_30, pca_to_30, sliding_window, spectral_embed,
    detect_strategy, adapt,
)


# ═══════════════════════════════════════════════════════════════════
# direct
# ═══════════════════════════════════════════════════════════════════

class TestDirect:
    def test_1d_30(self):
        v = np.arange(30, dtype=float)
        r = direct(v)
        assert r['strategy_used'] == 'direct'
        assert r['output'].shape == (1, 30)
        assert np.array_equal(r['output'][0], v)
        assert r['metadata']['was_1d'] is True

    def test_2d_N_30(self):
        v = np.random.RandomState(42).randn(5, 30)
        r = direct(v)
        assert r['output'].shape == (5, 30)
        assert np.array_equal(r['output'], v)
        assert r['metadata']['was_1d'] is False

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError):
            direct(np.zeros(40))
        with pytest.raises(ValueError):
            direct(np.zeros((5, 40)))
        with pytest.raises(ValueError):
            direct(np.zeros((2, 3, 4)))


# ═══════════════════════════════════════════════════════════════════
# pad_to_30
# ═══════════════════════════════════════════════════════════════════

class TestPadTo30:
    def test_1d_short(self):
        v = np.arange(10, dtype=float)
        r = pad_to_30(v)
        assert r['output'].shape == (1, 30)
        assert np.array_equal(r['output'][0, :10], v)
        assert np.all(r['output'][0, 10:] == 0)
        assert r['metadata']['pad_amount'] == 20
        assert r['metadata']['original_dim'] == 10

    def test_2d_short(self):
        v = np.random.RandomState(0).randn(5, 12)
        r = pad_to_30(v)
        assert r['output'].shape == (5, 30)
        assert np.array_equal(r['output'][:, :12], v)
        assert np.all(r['output'][:, 12:] == 0)
        assert r['metadata']['pad_amount'] == 18

    def test_too_large_raises(self):
        with pytest.raises(ValueError):
            pad_to_30(np.zeros(40))
        with pytest.raises(ValueError):
            pad_to_30(np.zeros((3, 30)))


# ═══════════════════════════════════════════════════════════════════
# pca_to_30
# ═══════════════════════════════════════════════════════════════════

class TestPcaTo30:
    def test_basic_shape(self):
        rng = np.random.RandomState(123)
        v = rng.randn(100, 50)
        r = pca_to_30(v)
        assert r['output'].shape == (100, 30)
        assert r['strategy_used'] == 'pca_to_30'
        md = r['metadata']
        assert md['original_dim'] == 50
        assert md['n_components_used'] == 30
        assert 0.0 < md['explained_variance_ratio'] <= 1.0

    def test_explained_variance_full_when_rank_30(self):
        # Συνθετικά δεδομένα μὲ ἀκριβῶς rank 30 σὲ διάστασι 50.
        rng = np.random.RandomState(7)
        latent = rng.randn(200, 30)
        loadings = rng.randn(30, 50)
        v = latent @ loadings
        r = pca_to_30(v)
        # Ὁλόκληρη ἡ διασπορὰ πιάνεται ἀπὸ 30 components.
        assert r['metadata']['explained_variance_ratio'] > 0.9999

    def test_few_samples_pads_to_30(self):
        # N=20 < 30 < k=50 → μόνο 19 nontrivial components μετὰ τὸ centering.
        rng = np.random.RandomState(11)
        v = rng.randn(20, 50)
        r = pca_to_30(v)
        assert r['output'].shape == (20, 30)
        # Ἀναμένουμε λιγότερα ἀπὸ 30 χρησιμοποιημένα (N-1=19 μετὰ centering).
        assert r['metadata']['n_components_used'] < 30
        # Τελευταῖες στῆλες μηδενικές.
        n_used = r['metadata']['n_components_used']
        assert np.all(r['output'][:, n_used:] == 0)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            pca_to_30(np.zeros((50, 30)))   # k = TARGET_DIM
        with pytest.raises(ValueError):
            pca_to_30(np.zeros((50, 20)))   # k < TARGET_DIM
        with pytest.raises(ValueError):
            pca_to_30(np.zeros(50))          # 1D


# ═══════════════════════════════════════════════════════════════════
# sliding_window
# ═══════════════════════════════════════════════════════════════════

class TestSlidingWindow:
    def test_basic(self):
        v = np.arange(60, dtype=float)
        r = sliding_window(v)
        assert r['output'].shape == (31, 30)   # L=60, stride=1 → 60-30+1=31
        assert np.array_equal(r['output'][0], np.arange(30))
        assert np.array_equal(r['output'][1], np.arange(1, 31))
        assert r['metadata']['n_windows'] == 31
        assert r['metadata']['padded'] is False

    def test_stride(self):
        v = np.arange(100, dtype=float)
        r = sliding_window(v, stride=10)
        # L=100, stride=10 → (100-30)//10 + 1 = 8 windows
        assert r['output'].shape == (8, 30)
        assert np.array_equal(r['output'][0], np.arange(30))
        assert np.array_equal(r['output'][1], np.arange(10, 40))
        assert r['metadata']['stride'] == 10

    def test_short_pads(self):
        v = np.arange(20, dtype=float)
        r = sliding_window(v)
        assert r['output'].shape == (1, 30)
        assert np.array_equal(r['output'][0, :20], v)
        assert np.all(r['output'][0, 20:] == 0)
        assert r['metadata']['padded'] is True
        assert r['metadata']['original_length'] == 20

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            sliding_window(np.zeros((5, 10)))
        with pytest.raises(ValueError):
            sliding_window(np.zeros(50), stride=0)


# ═══════════════════════════════════════════════════════════════════
# spectral_embed
# ═══════════════════════════════════════════════════════════════════

def _ring_graph(n):
    """Cycle C_n μὲ n κορυφές."""
    edges = set(frozenset({i, (i + 1) % n}) for i in range(n))
    return {'V': n, 'edges': edges}


def _complete_graph(n):
    edges = set(frozenset({i, j}) for i in range(n) for j in range(i + 1, n))
    return {'V': n, 'edges': edges}


class TestSpectralEmbed:
    def test_graph_dict_large(self):
        # C_50: ἀρκετὲς κορυφὲς γιὰ νὰ ἔχουμε ≥30 nontrivial eigvals.
        g = _ring_graph(50)
        r = spectral_embed(g)
        assert r['output'].shape == (50, 30)
        assert r['metadata']['V'] == 50
        assert r['metadata']['n_components_used'] == 30
        # Fiedler value τοῦ C_n εἶναι θετικός γιὰ συνεκτικὸ graph.
        assert r['metadata']['fiedler_value'] > 1e-10

    def test_adj_matrix(self):
        n = 40
        g = _ring_graph(n)
        # Φτιάχνουμε adj matrix χειρωνακτικά
        A = np.zeros((n, n))
        for e in g['edges']:
            i, j = list(e)
            A[i, j] = A[j, i] = 1.0
        r = spectral_embed(A)
        assert r['output'].shape == (n, 30)
        assert r['strategy_used'] == 'spectral_embed'

    def test_small_graph_pads(self):
        # K_5: μόνο 4 nontrivial eigvecs, ὑπόλοιπες στῆλες μηδέν.
        g = _complete_graph(5)
        r = spectral_embed(g)
        assert r['output'].shape == (5, 30)
        assert r['metadata']['n_components_used'] == 4
        assert np.all(r['output'][:, 4:] == 0)

    def test_invalid_dict_raises(self):
        with pytest.raises(ValueError):
            spectral_embed({'foo': 'bar'})
        with pytest.raises(ValueError):
            spectral_embed(np.zeros((5, 7)))   # non-square


# ═══════════════════════════════════════════════════════════════════
# detect_strategy
# ═══════════════════════════════════════════════════════════════════

class TestDetectStrategy:
    @pytest.mark.parametrize("inp, expected", [
        (np.zeros(30),         'direct'),
        (np.zeros((5, 30)),    'direct'),
        (np.zeros(100),        'sliding_window'),
        (np.zeros(15),         'sliding_window'),
        (np.zeros((100, 50)),  'pca_to_30'),
        (np.zeros((10, 5)),    'pad_to_30'),
        (np.zeros(20),         'sliding_window'),
    ])
    def test_array_inputs(self, inp, expected):
        assert detect_strategy(inp) == expected

    def test_graph_dict(self):
        g = _ring_graph(40)
        assert detect_strategy(g) == 'spectral_embed'

    def test_adjacency_matrix(self):
        g = _ring_graph(40)
        A = np.zeros((40, 40))
        for e in g['edges']:
            i, j = list(e)
            A[i, j] = A[j, i] = 1.0
        assert detect_strategy(A) == 'spectral_embed'

    def test_bad_dict_raises(self):
        with pytest.raises(ValueError):
            detect_strategy({'no_graph': True})


# ═══════════════════════════════════════════════════════════════════
# adapt() public API
# ═══════════════════════════════════════════════════════════════════

class TestAdapt:
    def test_auto_routes_correctly(self):
        r = adapt(np.zeros((100, 50)))
        assert r['strategy_used'] == 'pca_to_30'

        r = adapt(np.arange(100, dtype=float))
        assert r['strategy_used'] == 'sliding_window'

        r = adapt(np.zeros((5, 8)))
        assert r['strategy_used'] == 'pad_to_30'

    def test_manual_override(self):
        # 1D μῆκος 60 → auto θὰ ἔδινε sliding_window·
        # ἀναγκάζουμε pad_to_30 ἀπολύτως. Ἀλλὰ τὸ 60 > 30, ἄρα θὰ ἀποτύχει.
        # Διαλέγουμε ἀνάμεσα σὲ valid overrides.
        v_short_1d = np.arange(10, dtype=float)
        r = adapt(v_short_1d, strategy='pad_to_30')
        assert r['strategy_used'] == 'pad_to_30'
        assert r['output'].shape == (1, 30)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            adapt(np.zeros(30), strategy='nonexistent')

    def test_output_always_2d_30cols(self):
        """Ἀναλλοίωτο: κάθε strategy παράγει 2D μὲ shape[-1] == 30."""
        inputs = [
            np.zeros(30),                  # direct
            np.zeros((5, 30)),             # direct
            np.zeros(10),                  # sliding_window (pad)
            np.zeros(100),                 # sliding_window
            np.zeros((5, 8)),              # pad_to_30
            np.random.RandomState(0).randn(50, 100),  # pca_to_30
            _ring_graph(40),               # spectral_embed
        ]
        for v in inputs:
            r = adapt(v)
            out = r['output']
            assert out.ndim == 2
            assert out.shape[1] == TARGET_DIM

    def test_kwargs_passed_through(self):
        v = np.arange(100, dtype=float)
        r = adapt(v, strategy='sliding_window', stride=10)
        assert r['metadata']['stride'] == 10

        v2 = np.random.RandomState(0).randn(50, 100)
        r2 = adapt(v2, strategy='pca_to_30', center=False)
        assert r2['metadata']['centered'] is False
