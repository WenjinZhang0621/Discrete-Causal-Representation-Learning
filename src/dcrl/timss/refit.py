"""Vectorized fixed-Q lognormal EM used for the revised TIMSS analysis.

The item bounds and missing-cell treatment follow timss/em.py. Weighted
least squares and a profiled variance update solve its item M-step directly.
Reported likelihoods include the Gaussian normalizing constant on log times.
"""
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import logsumexp


def initialize(q, seed):
    rng = np.random.RandomState(seed)
    J, K = q.shape
    p = rng.dirichlet(5 * np.ones(2**K))
    g = 2 + .5 * rng.rand(J, 1)
    c = 4 + .5 * rng.rand(J, 1)
    beta = np.zeros((J, K+1))
    beta[:, 0] = g[:, 0]
    for j in range(J):
        active = np.flatnonzero(q[j])
        if len(active):
            beta[j, 1+active] = rng.dirichlet(5 * np.ones(len(active))) * (c[j, 0]-g[j, 0])
    return dict(p=p, beta=beta, gamma=np.full(J, .5))


def posterior(y, observed, design, p, beta, gamma):
    means = design @ beta.T
    lp = y @ (means/gamma).T - .5 * observed @ (means**2/gamma + np.log(gamma)).T
    lp -= .5 * (y*y/gamma).sum(axis=1)[:, None]
    lp += np.log(np.maximum(p, 1e-300))[None, :]
    norms = logsumexp(lp, axis=1)
    return np.exp(lp-norms[:, None]), float(norms.sum())


def fit(x, q, seed=48, max_iter=2000, tol=1e-7, initial=None):
    x, q = np.asarray(x, float), np.asarray(q)
    if x.ndim != 2 or q.ndim != 2 or q.shape[0] != x.shape[1] or not np.isin(q, [0, 1]).all():
        raise ValueError('Require a data matrix and a matching binary Q matrix')
    if max_iter < 1 or not np.isfinite(tol) or tol <= 0:
        raise ValueError('Require max_iter >= 1 and tol > 0')
    N, J = x.shape
    K = q.shape[1]
    observed = (np.isfinite(x) & (x > 0)).astype(float)
    if not N or (observed.sum(axis=0) == 0).any():
        raise ValueError('Every item must have at least one observed positive response')
    y = np.zeros_like(x)
    mask = observed.astype(bool)
    y[mask] = np.log(x[mask])
    states = ((np.arange(2**K)[:, None] >> np.arange(K-1, -1, -1)) & 1).astype(float)
    design = np.column_stack([np.ones(len(states)), states])
    indices = [np.r_[0, 1+np.flatnonzero(row)] for row in q]
    designs = [design[:, index] for index in indices]
    initial = initialize(q, seed) if initial is None else initial
    p, beta, gamma = [np.asarray(initial[key], float).copy().squeeze() for key in ['p', 'beta', 'gamma']]
    phi, ll = posterior(y, observed, design, p, beta, gamma)
    trace = []
    constant = .5 * observed.sum() * np.log(2*np.pi)
    for iteration in range(max_iter):
        old = ll
        weights, targets = observed.T @ phi, y.T @ phi
        yy, n = (y*y).sum(axis=0), observed.sum(axis=0)
        beta = np.zeros((J, K+1))
        gamma = np.empty(J)
        for j, (index, d) in enumerate(zip(indices, designs)):
            w = weights[j]
            root = np.sqrt(np.maximum(w, 1e-300))
            a, b = d * root[:, None], targets[j]/root
            lower, upper = np.r_[-2., np.full(len(index)-1, -3.)], np.full(len(index), 4.)
            coef = np.linalg.lstsq(a, b, rcond=None)[0]
            if np.any(coef < lower) or np.any(coef > upper):
                solution = lsq_linear(a, b, bounds=(lower, upper), method='bvls', tol=1e-12, max_iter=100)
                if not solution.success:
                    raise RuntimeError(solution.message)
                coef = solution.x
            beta[j, index] = coef
            mean = d @ coef
            rss = yy[j] - 2*mean @ targets[j] + (mean*mean) @ w
            gamma[j] = np.clip(rss/n[j], 1e-12, 2.)
        p = phi.mean(axis=0)
        phi, ll = posterior(y, observed, design, p, beta, gamma)
        change = ll-old
        trace.append([iteration+1, ll-constant, change])
        if iteration > 0 and abs(change) <= tol:
            break
    return dict(p=p, beta=beta, gamma=gamma, loglik=ll-constant, seed=seed,
                iterations=iteration+1, converged=bool(abs(change) <= tol), trace=np.array(trace))
