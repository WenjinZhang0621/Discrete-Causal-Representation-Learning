"""TIMSS graph discovery using the published multinomial resampling step."""
from numbers import Integral

import numpy as np
from causallearn.search.ScoreBased.GES import ges


def normalize_probabilities(p, K):
    p = np.asarray(p, dtype=float).reshape(-1)
    if not isinstance(K, Integral) or K < 1 or p.size != 2**K:
        raise ValueError("Require one probability for each of 2**K binary states")
    if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
        raise ValueError("Probabilities must be finite, nonnegative, and have positive sum")
    return p / p.sum()


def sample_latents(p, N, K=7, fn_ratio=0.2, seed=1):
    """Draw floor(fn_ratio*N) rows, with at least one row, using a local RNG.

    N remains the number of original students. Only sampled configurations
    are decoded; counts are never truncated or padded with all-zero records.
    """
    p = normalize_probabilities(p, K)
    if not isinstance(N, Integral) or N < 1 or not np.isfinite(fn_ratio) or fn_ratio <= 0:
        raise ValueError("Require a positive integer N and finite fn_ratio > 0")
    sample_size = max(1, int(np.floor(fn_ratio * N)))
    counts = np.random.RandomState(seed).multinomial(sample_size, p)
    samples = np.empty((sample_size, K), dtype=np.uint8)
    start = 0
    for state in np.flatnonzero(counts):
        count = int(counts[state])
        pattern = [(int(state) >> shift) & 1 for shift in range(K-1, -1, -1)]
        samples[start:start+count] = pattern
        start += count
    return samples, counts


def learn_graph(p, N, K=7, fn_ratio=0.2, seed=1):
    samples, counts = sample_latents(p, N, K, fn_ratio, seed)
    record = ges(samples, score_func="local_score_BDeu")
    return record, samples, counts
