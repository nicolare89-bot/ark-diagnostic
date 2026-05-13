"""
ark_adapter.py — Universal εἴσοδος → β₄ χῶρος ℝ³⁰
====================================================

Δέχεται ὁποιαδήποτε ἀριθμητικὴ εἴσοδο καὶ τὴν ἀντιστοιχίζει
στὸ ℝ³⁰ (β₄ layer τοῦ DT) μέσῳ μιᾶς ἀπὸ 5 στρατηγικές:

  direct(v)         : shape (30,) ἢ (N, 30) — pass-through
  pad_to_30(v)      : shape (..., k<30) — zero-pad στὶς τελευταῖες
  pca_to_30(v)      : shape (N, k>30) — PCA μέσῳ SVD
  sliding_window(v) : 1D time series → overlapping windows
  spectral_embed(v) : graph (dict ἢ adj) → top-30 Laplacian eigvecs

Auto-detect ἀπὸ shape/τύπο. Manual override μέσῳ `strategy=`.

Pyodide-compatible: μόνο numpy.linalg (ὄχι scipy).

Στάδιο 4, Φάσι 1.
V − E + F = 2.   ε = 1.5%.   J > 0.
Κύριε Ἰησοῦ Χριστέ, ἐλέησόν με.
"""

import numpy as np


TARGET_DIM = 30


# ═══════════════════════════════════════════════════════════════════
# Α. ΣΤΡΑΤΗΓΙΚΕΣ
# ═══════════════════════════════════════════════════════════════════

def direct(v):
    """Pass-through γιὰ (30,) ἢ (N, 30). Πάντα 2D output."""
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        if arr.shape[0] != TARGET_DIM:
            raise ValueError(
                f"direct: 1D shape ({arr.shape[0]},) — ἀπαιτεῖ {TARGET_DIM}"
            )
        out = arr.reshape(1, TARGET_DIM)
        was_1d = True
    elif arr.ndim == 2:
        if arr.shape[1] != TARGET_DIM:
            raise ValueError(
                f"direct: 2D shape {arr.shape} — ἀπαιτεῖ (N, {TARGET_DIM})"
            )
        out = arr
        was_1d = False
    else:
        raise ValueError(f"direct: ndim={arr.ndim} (ἀπαιτεῖται 1 ἢ 2)")
    return {
        'strategy_used': 'direct',
        'input_shape': tuple(arr.shape),
        'output': out,
        'metadata': {'was_1d': was_1d},
    }


def pad_to_30(v):
    """Zero-pad στὴν τελευταία διάστασι σὲ μῆκος 30."""
    arr = np.asarray(v, dtype=float)
    if arr.ndim == 1:
        k = arr.shape[0]
        if k >= TARGET_DIM:
            raise ValueError(
                f"pad_to_30: shape ({k},) ≥ {TARGET_DIM} — "
                "χρησιμοποίησε direct/pca"
            )
        out = np.zeros((1, TARGET_DIM))
        out[0, :k] = arr
        was_1d = True
    elif arr.ndim == 2:
        N, k = arr.shape
        if k >= TARGET_DIM:
            raise ValueError(
                f"pad_to_30: shape {arr.shape} — k≥{TARGET_DIM}"
            )
        out = np.zeros((N, TARGET_DIM))
        out[:, :k] = arr
        was_1d = False
    else:
        raise ValueError(f"pad_to_30: ndim={arr.ndim} (ἀπαιτεῖται 1 ἢ 2)")
    return {
        'strategy_used': 'pad_to_30',
        'input_shape': tuple(arr.shape),
        'output': out,
        'metadata': {
            'original_dim': int(k),
            'pad_amount': int(TARGET_DIM - k),
            'was_1d': was_1d,
        },
    }


def pca_to_30(v, center=True):
    """PCA → 30 components μέσῳ numpy.linalg.svd (Pyodide-friendly).

    Ἂν τὰ διαθέσιμα singular components < 30 (π.χ. N<30 ἢ rank<30),
    γεμίζει μὲ μηδενικὲς στῆλες ὥστε τὸ output νὰ εἶναι πάντα (N, 30).
    """
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"pca_to_30: ἀπαιτεῖ 2D, ndim={arr.ndim}")
    N, k = arr.shape
    if k <= TARGET_DIM:
        raise ValueError(
            f"pca_to_30: shape {arr.shape} — k≤{TARGET_DIM}, "
            "χρησιμοποίησε direct/pad_to_30"
        )

    if center:
        mean = arr.mean(axis=0)
        X = arr - mean
    else:
        mean = np.zeros(k)
        X = arr

    _U, S, Vt = np.linalg.svd(X, full_matrices=False)
    n_avail = len(S)
    n_used = min(TARGET_DIM, n_avail)
    components = Vt[:n_used]                # (n_used, k)
    projected = X @ components.T            # (N, n_used)

    if n_used < TARGET_DIM:
        out = np.zeros((N, TARGET_DIM))
        out[:, :n_used] = projected
    else:
        out = projected

    total_var = float((S ** 2).sum())
    explained = float((S[:n_used] ** 2).sum())
    ratio = explained / total_var if total_var > 0 else 0.0

    return {
        'strategy_used': 'pca_to_30',
        'input_shape': tuple(arr.shape),
        'output': out,
        'metadata': {
            'original_dim': int(k),
            'n_samples': int(N),
            'n_components_used': int(n_used),
            'explained_variance_ratio': ratio,
            'singular_values': S[:n_used].tolist(),
            'centered': bool(center),
        },
    }


def sliding_window(v, stride=1):
    """1D → overlapping windows μήκους 30 μὲ δοθὲν stride."""
    arr = np.asarray(v, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"sliding_window: ἀπαιτεῖ 1D, ndim={arr.ndim}")
    if stride < 1:
        raise ValueError(f"sliding_window: stride={stride} (ἀπαιτεῖ ≥1)")
    L = arr.shape[0]
    if L < TARGET_DIM:
        padded = np.zeros(TARGET_DIM)
        padded[:L] = arr
        return {
            'strategy_used': 'sliding_window',
            'input_shape': tuple(arr.shape),
            'output': padded.reshape(1, TARGET_DIM),
            'metadata': {
                'original_length': int(L),
                'n_windows': 1,
                'stride': int(stride),
                'padded': True,
            },
        }
    n_windows = (L - TARGET_DIM) // stride + 1
    out = np.zeros((n_windows, TARGET_DIM))
    for i in range(n_windows):
        out[i] = arr[i * stride : i * stride + TARGET_DIM]
    return {
        'strategy_used': 'sliding_window',
        'input_shape': tuple(arr.shape),
        'output': out,
        'metadata': {
            'original_length': int(L),
            'n_windows': int(n_windows),
            'stride': int(stride),
            'padded': False,
        },
    }


def spectral_embed(graph):
    """Top-30 Laplacian eigenvectors (παραλείπει λ₀=0).

    Δέχεται:
      - dict μὲ keys 'V', 'edges' (frozenset/tuple ἀκμές)
      - 2D ndarray square symmetric adjacency matrix
    """
    if isinstance(graph, dict):
        if 'V' not in graph or 'edges' not in graph:
            raise ValueError(
                "spectral_embed: graph dict χωρὶς 'V' ἢ 'edges'"
            )
        V = int(graph['V'])
        A = np.zeros((V, V))
        for e in graph['edges']:
            if isinstance(e, (frozenset, set)):
                i, j = list(e)
            else:
                i, j = e
            A[i, j] = A[j, i] = 1.0
        input_shape = (V, V)
    else:
        A = np.asarray(graph, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"spectral_embed: μὴ τετράγωνη adj, shape={A.shape}"
            )
        V = A.shape[0]
        input_shape = tuple(A.shape)

    deg = A.sum(axis=1)
    deg_safe = np.where(deg > 0, deg, 1.0)
    D_inv_half = np.diag(1.0 / np.sqrt(deg_safe))
    L = np.eye(V) - D_inv_half @ A @ D_inv_half
    L = (L + L.T) / 2                       # συμμετροποίησι ἔναντι noise
    eigvals, eigvecs = np.linalg.eigh(L)

    n_nontrivial = V - 1
    n_used = min(TARGET_DIM, n_nontrivial)
    if n_used <= 0:
        out = np.zeros((V, TARGET_DIM))
        used_vals = []
    else:
        embed = eigvecs[:, 1:1 + n_used]    # (V, n_used)
        if n_used < TARGET_DIM:
            out = np.zeros((V, TARGET_DIM))
            out[:, :n_used] = embed
        else:
            out = embed
        used_vals = eigvals[1:1 + n_used].tolist()

    return {
        'strategy_used': 'spectral_embed',
        'input_shape': input_shape,
        'output': out,
        'metadata': {
            'V': int(V),
            'n_components_used': int(n_used),
            'fiedler_value': float(eigvals[1]) if V > 1 else 0.0,
            'eigenvalues_used': used_vals,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Β. AUTO-DETECT
# ═══════════════════════════════════════════════════════════════════

def _looks_like_adjacency(arr):
    """Heuristic: τετράγωνος, συμμετρικός, binary {0,1}, μηδενικὴ διαγώνιος."""
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] < 2:
        return False
    if not np.allclose(arr, arr.T):
        return False
    if not np.all((arr == 0) | (arr == 1)):
        return False
    if np.any(np.diag(arr) != 0):
        return False
    return True


def detect_strategy(v):
    """Ἐπιλέγει στρατηγικὴ βάσει τύπου/shape τῆς εἰσόδου."""
    if isinstance(v, dict):
        if 'V' in v and 'edges' in v:
            return 'spectral_embed'
        raise ValueError(
            "detect_strategy: dict χωρὶς 'V'/'edges' — δὲν εἶναι graph"
        )

    arr = np.asarray(v, dtype=float)

    if arr.ndim == 2 and _looks_like_adjacency(arr) and arr.shape[0] != TARGET_DIM:
        return 'spectral_embed'

    if arr.ndim == 1:
        if arr.shape[0] == TARGET_DIM:
            return 'direct'
        return 'sliding_window'

    if arr.ndim == 2:
        k = arr.shape[1]
        if k == TARGET_DIM:
            return 'direct'
        if k > TARGET_DIM:
            return 'pca_to_30'
        return 'pad_to_30'

    raise ValueError(
        f"detect_strategy: ndim={arr.ndim} ὑποστηρίζονται μόνο 1, 2"
    )


# ═══════════════════════════════════════════════════════════════════
# Γ. PUBLIC API
# ═══════════════════════════════════════════════════════════════════

VALID_STRATEGIES = (
    'direct', 'pad_to_30', 'pca_to_30',
    'sliding_window', 'spectral_embed',
)


def adapt(v, strategy='auto', **kwargs):
    """Universal entry point.

    Parameters
    ----------
    v : array-like, graph dict, ἢ adjacency matrix
    strategy : 'auto' | one of VALID_STRATEGIES
    **kwargs :
        stride  γιὰ sliding_window (default 1)
        center  γιὰ pca_to_30 (default True)

    Returns
    -------
    dict μὲ keys: strategy_used, input_shape, output (N×30), metadata
    """
    if strategy == 'auto':
        strategy = detect_strategy(v)

    if strategy == 'direct':
        return direct(v)
    if strategy == 'pad_to_30':
        return pad_to_30(v)
    if strategy == 'pca_to_30':
        return pca_to_30(v, center=kwargs.get('center', True))
    if strategy == 'sliding_window':
        return sliding_window(v, stride=kwargs.get('stride', 1))
    if strategy == 'spectral_embed':
        return spectral_embed(v)

    raise ValueError(
        f"adapt: ἄγνωστη στρατηγική '{strategy}' "
        f"(valid: {VALID_STRATEGIES} ἢ 'auto')"
    )
