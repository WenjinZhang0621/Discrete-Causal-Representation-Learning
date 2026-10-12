"""Experimental SLSQP Jacobian for the unchanged accumulated SAEM objective.

This is NOT a bitwise-equivalent optimizer implementation. Bounds, the scalar
objective, penalty, gain, support threshold, and SLSQP options are unchanged;
supplying a Jacobian changes its numerical path and possibly its attained local
optimum. Do not mix this backend into an already-running experiment.

TLP is nondifferentiable at 0 and +/-tau. Away from those points its derivative
is analytic. When a default SLSQP finite-difference step crosses a corner, this
module uses that bound-aware one-sided difference for the penalty only. Fixed
coordinates receive zero derivatives. This is a convention at corners, not a
claim that a classical derivative or a global convergence guarantee exists.

Near the original Poisson zero-mean and Lognormal zero-variance boundaries, the
entire objective Jacobian falls back to bound-aware finite differences. The
scalar objective is never clipped or repaired. In particular, original NaN at
Poisson mean zero is retained and remains an optimizer failure possibility.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

import numpy as np
from scipy.special import expit

from dcrl.estimator import DAGEstimator
from .local90_fast_gibbs import FROZEN_ESTIMATOR_SHA256
from .local90_fast_objective import FastObjectiveEstimator, likelihood_objective

DEFAULT_EPS = float(np.sqrt(np.finfo(float).eps))


def bound_difference_steps(x, bounds, eps=DEFAULT_EPS):
    """SLSQP-style absolute forward steps, reversed/shrunk to stay in bounds."""
    x = np.asarray(x, dtype=float)
    lower, upper = np.asarray(bounds, dtype=float).T
    if x.shape != lower.shape or np.any(lower > upper):
        raise ValueError('Invalid derivative bounds')
    if not np.isfinite(eps) or eps <= 0:
        raise ValueError('eps must be finite and positive')
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError('Jacobian point outside bounds')
    h = np.full_like(x, eps)
    unchanged = x + h == x
    h[unchanged] = eps * np.where(x[unchanged] >= 0, 1., -1.) * np.maximum(1., np.abs(x[unchanged]))
    lo, hi = x - lower, upper - x
    violated = (x + h < lower) | (x + h > upper)
    fitting = np.abs(h) <= np.maximum(lo, hi)
    h[violated & fitting] *= -1
    forward = (hi >= lo) & ~fitting
    backward = (hi < lo) & ~fitting
    h[forward] = hi[forward]
    h[backward] = -lo[backward]
    h[lower == upper] = 0.
    # Use the actual floating-point displacement, just like numdiff.
    return (x + h) - x


def bounded_numeric_jac(objective, x, bounds, eps=DEFAULT_EPS):
    x = np.asarray(x, dtype=float)
    steps = bound_difference_steps(x, bounds, eps)
    base = objective(x)
    out = np.zeros_like(x)
    for index in np.flatnonzero(steps):
        perturbed = x.copy()
        perturbed[index] += steps[index]
        out[index] = (objective(perturbed) - base) / steps[index]
    return out


def tlp_jac(x, tau, pen, bounds, distribution, eps=DEFAULT_EPS):
    """Return (gradient, corner-coordinate count), excluding intercept/variance."""
    x = np.asarray(x, dtype=float)
    result = np.zeros_like(x)
    if tau == 0 or pen == 0:
        return result, 0
    if tau < 0 or pen < 0:
        raise ValueError('tau and pen must be nonnegative')
    slopes = np.arange(1, len(x) - (distribution == 'Lognormal'))
    steps = bound_difference_steps(x, bounds, eps)
    a = x[slopes]
    h = steps[slopes]
    # At or across a corner, reproduce the finite-step penalty derivative.
    near = np.minimum(np.abs(a), np.abs(np.abs(a) - tau)) <= np.abs(h)
    ordinary = (~near) & (h != 0)
    result[slopes[ordinary]] = pen * np.sign(a[ordinary]) * (np.abs(a[ordinary]) < tau)
    finite = near & (h != 0)
    result[slopes[finite]] = pen * (np.minimum(np.abs(a[finite] + h[finite]), tau)
                                  - np.minimum(np.abs(a[finite]), tau)) / h[finite]
    return result, int(np.sum(finite))


class ObjectiveJacobian:
    """Gradient of one item's cumulative likelihood plus unchanged TLP."""
    def __init__(self, stats, distribution, states, objective, bounds, tau, pen, eps=DEFAULT_EPS):
        self.stats, self.distribution, self.objective = stats, distribution, objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.tau, self.pen, self.eps = tau, pen, eps
        self.info = dict(backend='analytic_likelihood_with_corner_fallback',
                         analytic_calls=0, numeric_fallback_calls=0,
                         penalty_corner_coordinates=0,
                         fixed_coordinates=int(np.sum(self.bounds[:, 0] == self.bounds[:, 1])),
                         finite_difference_eps=float(eps), bitwise_equivalent=False)
        if distribution != 'Lognormal':
            positive = stats.counts > 0
            self.design = states[positive]
            self.counts = stats.counts[positive]
            self.response = stats.response[positive]

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        family, stats = self.distribution, self.stats
        if family == 'Lognormal':
            singular = not np.isfinite(x[-1]) or x[-1] <= 32 * self.eps
        elif family == 'Poisson':
            means = self.design @ x
            singular = np.any(~np.isfinite(means)) or np.any(means <= 32 * self.eps)
        elif family == 'Bernoulli':
            singular = False
        else:
            raise ValueError(f'Unknown distribution {family}')
        if singular:
            self.info['numeric_fallback_calls'] += 1
            return bounded_numeric_jac(self.objective, x, self.bounds, self.eps)
        if family == 'Bernoulli':
            linear = self.design @ x
            gradient = self.design.T @ (self.counts * expit(linear) - self.response)
        elif family == 'Poisson':
            gradient = self.design.T @ (self.counts - self.response / means)
        else:
            beta, variance = x[:-1], float(x[-1])
            sse = stats.sum_squares - 2 * (beta @ stats.cross) + beta @ stats.gram @ beta
            # Symmetric expression also remains correct for tiny Gram asymmetry.
            beta_grad = ((stats.gram + stats.gram.T) @ beta - 2 * stats.cross) / (2 * variance)
            variance_grad = -sse / (2 * variance**2) + stats.mass / (2 * variance)
            gradient = np.r_[beta_grad, variance_grad]
        penalty_grad, corners = tlp_jac(x, self.tau, self.pen, self.bounds, family, self.eps)
        gradient = gradient + penalty_grad
        gradient[self.bounds[:, 0] == self.bounds[:, 1]] = 0.
        self.info['analytic_calls'] += 1
        self.info['penalty_corner_coordinates'] += corners
        return gradient


def _make_recorded_jac_minimize():
    """Inject jac into the hash-verified original diagnostic/solver wrapper."""
    method = DAGEstimator._minimize
    source = Path(inspect.getsourcefile(method))
    if hashlib.sha256(source.read_bytes()).hexdigest() != FROZEN_ESTIMATOR_SHA256:
        raise RuntimeError('Frozen estimator source changed')
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Inject(ast.NodeTransformer):
        solver_calls = 0
        records = 0

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'minimize':
                self.solver_calls += 1
                if any(k.arg == 'jac' for k in node.keywords):
                    raise RuntimeError('Original minimize unexpectedly already has jac')
                node.keywords.append(ast.keyword(arg='jac', value=ast.parse('self._active_analytic_jac', mode='eval').body))
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if ast.unparse(node) == 'self.optimizer_diagnostics.append(record)':
                self.records += 1
                addition = ast.parse("record['gradient_diagnostics'] = dict(self._active_analytic_jac.info)").body[0]
                return [ast.copy_location(addition, node), node]
            return self.generic_visit(node)

    inject = Inject()
    tree = ast.fix_missing_locations(inject.visit(tree))
    if (inject.solver_calls, inject.records) != (1, 2):
        raise RuntimeError(f'Unexpected diagnostic wrapper structure: {inject.solver_calls}, {inject.records}')
    namespace = dict(method.__globals__)
    exec(compile(tree, str(Path(__file__).resolve()) + ':generated_minimize', 'exec'), namespace)
    return namespace['_minimize']


_recorded_jac_minimize = _make_recorded_jac_minimize()


class AnalyticObjectiveEstimator(FastObjectiveEstimator):
    """Opt-in numerical backend; combine with accelerated_gibbs_class if needed."""
    def _minimize(self, objective, x0, *, bounds, method, options, item, iteration):
        if method != 'SLSQP':
            raise ValueError('This experimental Jacobian backend supports SLSQP only')
        call = iteration * self.J + item
        if call != self._fast_last_call + 1 or self._fast_pending is None:
            raise RuntimeError('Unexpected order of SAEM item optimization')
        fresh = self._fast_pending
        self._fast_pending = None
        self._fast_last_call = call
        old = self._fast_stats.get(item, fresh.zero_like())
        gain = self.gain_numerator / (iteration + self.gain_offset)
        cumulative = old.blend(fresh, gain)
        self._fast_stats[item] = cumulative
        likelihood = likelihood_objective(cumulative, self.distribution, self._fast_states)

        def cumulative_objective(x):
            return likelihood(x) + self.ftn_pen(x)

        self._active_analytic_jac = ObjectiveJacobian(cumulative, self.distribution,
            self._fast_states, cumulative_objective, bounds, self.tau, self.pen,
            eps=options.get('eps', DEFAULT_EPS))
        return _recorded_jac_minimize(self, cumulative_objective, x0, bounds=bounds,
            method=method, options=options, item=item, iteration=iteration)
