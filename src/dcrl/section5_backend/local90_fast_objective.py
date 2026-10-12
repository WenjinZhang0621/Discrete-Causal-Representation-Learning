"""Exact sufficient-statistic SAEM objectives for the frozen DCRL estimator.

This experimental subclass changes objective evaluation only. It inherits the
original Gibbs sampler, RNG calls, gain, support threshold, solver settings,
parameter bounds, and stopping rule. The frozen source is never edited.

The original objective is a weighted sum of all sampled complete-data negative
log likelihoods, initialized at zero. For Bernoulli/Poisson, counts and response
sums by binary latent state suffice; Lognormal needs quadratic statistics.
The probability-law smoother is separate and retains its original initialization.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from dcrl.estimator import DAGEstimator


@dataclass
class ItemStatistics:
    counts: np.ndarray | None = None
    response: np.ndarray | None = None
    gram: np.ndarray | None = None
    cross: np.ndarray | None = None
    sum_squares: float = 0.0
    mass: float = 0.0

    def blend(self, fresh: "ItemStatistics", gain: float) -> "ItemStatistics":
        def combine(old, new):
            return None if new is None else (1.0-gain)*old + gain*new
        return ItemStatistics(
            counts=combine(self.counts, fresh.counts),
            response=combine(self.response, fresh.response),
            gram=combine(self.gram, fresh.gram),
            cross=combine(self.cross, fresh.cross),
            sum_squares=(1.0-gain)*self.sum_squares+gain*fresh.sum_squares,
            mass=(1.0-gain)*self.mass+gain*fresh.mass,
        )

    def zero_like(self) -> "ItemStatistics":
        def zero(value):
            return None if value is None else np.zeros_like(value)
        return ItemStatistics(zero(self.counts), zero(self.response),
                              zero(self.gram), zero(self.cross), 0.0, 0.0)


def augmented_states(K: int) -> np.ndarray:
    if not 1 <= K <= 16:
        raise ValueError("This exact state-statistic accelerator supports 1 <= K <= 16")
    ids = np.arange(1 << K, dtype=np.int64)
    bits = ((ids[:, None] >> np.arange(K-1, -1, -1)) & 1).astype(float)
    return np.column_stack((np.ones(len(ids)), bits))


def sample_statistics(Xj, A_sample_long, distribution: str, K: int, C: int) -> ItemStatistics:
    # The original F_1_SAEM makes the same float32 copy. Binary design entries
    # are exactly representable; all reductions below use float64.
    samples = np.array(A_sample_long, dtype=np.float32)
    y = np.asarray(Xj, dtype=float).reshape(-1)
    if samples.shape != (len(y), K+1, C):
        raise ValueError("Unexpected latent sample dimensions")
    if not np.all(samples[:, 0, :] == 1) or not np.isin(samples[:, 1:, :], [0, 1]).all():
        raise ValueError("Expected intercept plus binary latent samples")
    if distribution == "Lognormal":
        y = np.log(y)
        gram = np.zeros((K+1, K+1))
        cross = np.zeros(K+1)
        for c in range(C):
            design = samples[:, :, c].astype(float)
            gram += design.T @ design / C
            cross += design.T @ y / C
        return ItemStatistics(gram=gram, cross=cross,
                              sum_squares=float(y @ y), mass=float(len(y)))
    if distribution not in {"Bernoulli", "Poisson"}:
        raise ValueError(f"Unknown distribution {distribution}")
    counts = np.zeros(1 << K)
    response = np.zeros(1 << K)
    powers = 1 << np.arange(K-1, -1, -1, dtype=np.int64)
    for c in range(C):
        state = samples[:, 1:, c].astype(np.int64) @ powers
        counts += np.bincount(state, minlength=len(counts)) / C
        response += np.bincount(state, weights=y, minlength=len(counts)) / C
    return ItemStatistics(counts=counts, response=response, mass=float(len(y)))


def likelihood_objective(stats: ItemStatistics, distribution: str, states: np.ndarray):
    """Return original negative complete-data log likelihood without penalty."""
    if distribution == "Lognormal":
        def objective(x):
            beta = np.asarray(x[:-1], dtype=float)
            variance = max(float(x[-1]), 1e-100)  # original ftn_h clipping
            residual = stats.sum_squares - 2.0 * (beta @ stats.cross) + beta @ stats.gram @ beta
            return float(residual/(2.0*variance) + 0.5*stats.mass*np.log(variance))
        return objective
    positive = stats.counts > 0
    design = states[positive]
    counts = stats.counts[positive]
    response = stats.response[positive]
    if distribution == "Bernoulli":
        def objective(x):
            linear = design @ np.asarray(x, dtype=float)
            return float(np.sum(counts*np.log1p(np.exp(linear))-response*linear))
        return objective
    if distribution == "Poisson":
        def objective(x):
            means = design @ np.asarray(x, dtype=float)
            # Preserve original log(0)/0*log(0) behavior at the solver boundary.
            eta = np.log(means)
            return float(np.sum(counts*np.exp(eta)-response*eta))
        return objective
    raise ValueError(f"Unknown distribution {distribution}")


def _unused_original_closure(_x):
    raise RuntimeError("The accumulated-statistic objective should replace this placeholder")


class FastObjectiveEstimator(DAGEstimator):
    """Drop-in estimator with exact accumulated objectives, without history scans."""

    def _reset_fast_objective(self):
        self._fast_states = augmented_states(self.K)
        self._fast_stats = {}
        self._fast_pending = None
        self._fast_last_call = -1

    def PSAEM(self, ite):
        self._reset_fast_objective()
        return super().PSAEM(ite)

    def F_1_SAEM(self, Xj, A_sample_long):
        if self._fast_pending is not None:
            raise RuntimeError("An item statistic was not consumed by the optimizer")
        self._fast_pending = sample_statistics(Xj, A_sample_long, self.distribution, self.K, self.C)
        # Frozen PSAEM still builds its small closure chain, but no sample arrays
        # are captured and _minimize below never evaluates that chain.
        return _unused_original_closure

    def _minimize(self, objective, x0, *, bounds, method, options, item, iteration):
        call = iteration*self.J + item
        if call != self._fast_last_call+1 or self._fast_pending is None:
            raise RuntimeError("Unexpected order of SAEM item optimization")
        fresh = self._fast_pending
        self._fast_pending = None
        self._fast_last_call = call
        old = self._fast_stats.get(item, fresh.zero_like())
        gain = self.gain_numerator/(iteration+self.gain_offset)
        cumulative = old.blend(fresh, gain)
        self._fast_stats[item] = cumulative
        likelihood = likelihood_objective(cumulative, self.distribution, self._fast_states)

        def cumulative_objective(x):
            return likelihood(x) + self.ftn_pen(x)

        return super()._minimize(cumulative_objective, x0, bounds=bounds, method=method,
                                 options=options, item=item, iteration=iteration)
