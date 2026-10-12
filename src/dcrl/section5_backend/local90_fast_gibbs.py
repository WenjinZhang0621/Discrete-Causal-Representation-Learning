"""Optional local vectorized Gibbs kernel; never modifies the frozen package.

Rows are conditionally independent given the fixed smoothed state counts. Each
row uses its own uniformly random coordinate permutation and one Bernoulli draw
per coordinate. Interleaving updates across rows therefore preserves the scalar
transition kernel, but consumes random numbers in a different order: this is
distributional equivalence, NOT a bitwise replay of the original implementation.
No simulation truth enters this module.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

import numpy as np
from scipy.special import expit

# The public estimator adds final orientation to run(); PSAEM and the
# optimizer are unchanged from the cluster validation source.
FROZEN_ESTIMATOR_SHA256 = "117838b66f2950ea315f6d99183c8158a032d1c757651148e9548095937a9670"


def coordinate_orders(n: int, k: int, rng=None) -> np.ndarray:
    """Independent uniform Fisher-Yates permutations, shape (n, k)."""
    rng = np.random if rng is None else rng
    order = np.broadcast_to(np.arange(k), (n, k)).copy()
    rows = np.arange(n)
    integers = rng.integers if hasattr(rng, "integers") else rng.randint
    for j in range(k - 1, 0, -1):
        chosen = integers(0, j + 1, size=n)
        saved = order[:, j].copy()
        order[:, j] = order[rows, chosen]
        order[rows, chosen] = saved
    return order


def conditional_probabilities(X, z0, z1, log_prior_ratio, gamma_hat, distribution):
    """Per-row P(A_k=1 | A_-k, X), matching original exponential-family code.

    z0/z1 are current linear predictors with the selected coordinate fixed to
    zero/one. gamma_hat is a variance for Lognormal. Original numerical floors
    and normalizers are retained; float summation order can differ slightly.
    """
    X, z0, z1 = (np.asarray(v, dtype=float) for v in (X, z0, z1))
    if distribution == "Poisson":
        eta0 = np.log(np.maximum(z0, 1e-150))
        eta1 = np.log(np.maximum(z1, 1e-150))
        ll0 = np.sum(eta0 * X - np.exp(eta0), axis=1)
        ll1 = np.sum(eta1 * X - np.exp(eta1), axis=1)
    elif distribution == "Bernoulli":
        # Identical expression to the frozen scalar code, rather than replacing
        # log1p(exp()) with a different numerical stabilization in this pilot.
        ll0 = np.sum(z0 * X - np.log1p(np.exp(z0)), axis=1)
        ll1 = np.sum(z1 * X - np.log1p(np.exp(z1)), axis=1)
    elif distribution == "Lognormal":
        variance = np.maximum(np.asarray(gamma_hat, dtype=float).reshape(1, -1), 1e-100)
        eta_first = 1 / (2 * variance)
        normalizer_first = np.maximum(eta_first, 1e-100)
        tx = np.log(X)
        fixed = -eta_first * tx**2
        eta0, eta1 = z0 / variance, z1 / variance
        norm0 = eta0**2 / (4 * normalizer_first) + np.log(1 / (2 * normalizer_first)) / 2
        norm1 = eta1**2 / (4 * normalizer_first) + np.log(1 / (2 * normalizer_first)) / 2
        ll0 = np.sum(fixed + eta0 * tx - norm0, axis=1)
        ll1 = np.sum(fixed + eta1 * tx - norm1, axis=1)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    return expit(np.asarray(log_prior_ratio) + ll1 - ll0)


def _prior_lookup(counts_smooth, kappa: float, k: int):
    if not np.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    if k <= 20:
        counts = np.zeros(1 << k, dtype=float)
        for state, value in counts_smooth.items():
            counts[int(state)] = float(value)
        logs = np.log(counts + kappa)
        return lambda states: logs[states]
    # Sparse fallback avoids allocating exponentially large arrays.
    return lambda states: np.log(np.fromiter(
        (float(counts_smooth.get(int(s), 0.0)) + kappa for s in states),
        dtype=float, count=len(states)))


def gibbs_sweep(X, A_cur, B_hat, gamma_hat, counts_smooth, kappa, distribution,
                *, rng=None, orders=None, uniforms=None):
    """Mutate A_cur for one random-scan sweep; return (A_cur, state_indices).

    Optional orders/uniforms make deterministic transition checks possible.
    Normal fitting generates orders internally and draws Bernoulli bits. The
    prior counts remain fixed throughout the sweep, exactly as in frozen PSAEM.
    Supports up to 63 binary coordinates (all current study graphs have <=13).
    """
    rng = np.random if rng is None else rng
    A_cur = np.asarray(A_cur)
    X, B_hat = np.asarray(X), np.asarray(B_hat)
    n, k = A_cur.shape
    if not 0 < k <= 63:
        raise ValueError("accelerated Gibbs supports 1 <= K <= 63")
    if X.ndim != 2 or X.shape[0] != n or B_hat.shape != (X.shape[1], k + 1):
        raise ValueError("incompatible X, A_cur, B_hat shapes")
    if not np.all((A_cur == 0) | (A_cur == 1)):
        raise ValueError("A_cur must be binary")
    weights = np.left_shift(np.int64(1), np.arange(k - 1, -1, -1, dtype=np.int64))
    idx_cur = np.sum(A_cur.astype(np.int64) * weights, axis=1, dtype=np.int64)
    orders = coordinate_orders(n, k, rng) if orders is None else np.asarray(orders)
    if orders.shape != (n, k) or not np.all(np.sort(orders, axis=1) == np.arange(k)):
        raise ValueError("orders must contain a coordinate permutation per row")
    if uniforms is not None:
        uniforms = np.asarray(uniforms)
        if uniforms.shape != (n, k) or not np.all((uniforms >= 0) & (uniforms < 1)):
            raise ValueError("uniforms must have shape (N,K) and values in [0,1)")
    rows = np.arange(n)
    z = A_cur @ B_hat[:, 1:].T + B_hat[:, 0]
    logs = _prior_lookup(counts_smooth, kappa, k)
    for pos in range(k):
        coord = orders[:, pos]
        mask = weights[coord]
        col = B_hat[:, coord + 1].T
        old_bit = A_cur[rows, coord].copy()
        z0 = z - old_bit[:, None] * col
        z1 = z + (1 - old_bit)[:, None] * col
        idx0, idx1 = idx_cur & ~mask, idx_cur | mask
        probability = conditional_probabilities(
            X, z0, z1, logs(idx1) - logs(idx0), gamma_hat, distribution)
        new_bit = (uniforms[:, pos] < probability).astype(int) if uniforms is not None else rng.binomial(1, probability)
        changed = new_bit != old_bit
        A_cur[rows, coord] = new_bit
        # Keep unchanged predictors unchanged, as the scalar code does.
        chosen_z = np.where(new_bit[:, None] == 1, z1, z0)
        z[changed] = chosen_z[changed]
        idx_cur = np.where(new_bit == 1, idx1, idx0)
    return A_cur, idx_cur


def make_fast_psaem(base_class):
    """Return PSAEM with only its three Gibbs row loops replaced.

    The base class can be a subclass overriding objective helpers. The frozen
    PSAEM is located on its MRO, then its full file hash is checked before
    extracting and transforming source. This function never edits that file.
    """
    method = None
    for parent in base_class.__mro__:
        candidate = parent.__dict__.get("PSAEM")
        if candidate is None:
            continue
        source_file = inspect.getsourcefile(candidate)
        if source_file and Path(source_file).is_file():
            actual_hash = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
            if actual_hash == FROZEN_ESTIMATOR_SHA256:
                method = candidate
                break
    if method is None:
        raise RuntimeError("No hash-verified frozen PSAEM found in class MRO")
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class ReplaceLoops(ast.NodeTransformer):
        count = 0

        def visit_For(self, node):
            if (isinstance(node.target, ast.Name) and node.target.id == "i"
                    and ast.unparse(node.iter) == "range(self.N)"):
                self.count += 1
                replacement = ast.parse(
                    "A_cur, idx_cur = _local90_gibbs_sweep(self.X, A_cur, self.B_hat, "
                    "self.gamma_hat, counts_smooth, self.kappa, self.distribution)"
                ).body[0]
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

    transform = ReplaceLoops()
    tree = ast.fix_missing_locations(transform.visit(tree))
    if transform.count != 3:
        raise RuntimeError(f"Expected exactly three Gibbs loops, got {transform.count}")
    namespace = dict(method.__globals__)
    namespace["_local90_gibbs_sweep"] = gibbs_sweep
    exec(compile(tree, str(Path(__file__).resolve()) + ":generated_PSAEM", "exec"), namespace)
    generated = namespace["PSAEM"]
    generated.__doc__ = __doc__
    generated._frozen_source_sha256 = actual_hash
    generated._replaced_gibbs_loops = transform.count
    return generated


def accelerated_gibbs_class(base_class):
    """Compose with any objective-accelerating subclass of frozen DAGEstimator."""
    transformed = make_fast_psaem(base_class)

    def fast_psaem(self, ite):
        reset = getattr(self, "_reset_fast_objective", None)
        if reset is not None:
            reset()
        return transformed(self, ite)

    return type("VectorizedGibbs" + base_class.__name__, (base_class,), {
        "PSAEM": fast_psaem,
        "frozen_estimator_sha256": FROZEN_ESTIMATOR_SHA256,
        "gibbs_kernel": "row-vectorized random-scan; distributionally equivalent, RNG differs",
    })
