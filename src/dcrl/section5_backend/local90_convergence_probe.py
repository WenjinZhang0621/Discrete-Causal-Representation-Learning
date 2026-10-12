"""Opt-in convergence observations/stopping for the local accelerated PSAEM.

Default mode is observational: it leaves the original stopping decision, gain,
sampling, objective, optimizer, and parameter updates unchanged. No truth or SHD
is accepted by this interface. Explicit StopRule thresholds are experimental,
not calibrated convergence guarantees.

The original `tol` compares squared Frobenius parameter change: B alone for
Bernoulli/Poisson, B plus gamma for Lognormal. Defaults are 0.5 for Bernoulli and
0.05 otherwise. The optional rule below uses separate RMS changes instead;
these thresholds must not be equated with the original squared-sum thresholds.
Small adjacent p-TV can be caused by a tiny gain or a stuck Gibbs chain. TV/gain
is available as an optional additional check, but neither is a mixing or global
optimum certificate. Consecutive stochastic changes can also fluctuate at a
stationary fit; reaching max_iter does not certify convergence.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import inspect
import math
from pathlib import Path
import textwrap

import numpy as np

from .local90_fast_gibbs import FROZEN_ESTIMATOR_SHA256, gibbs_sweep


@dataclass(frozen=True)
class StopRule:
    """Require all checks for 5 or 10 consecutive rounds, after >=50 rounds.

    p_tv_tol and b_rms_tol intentionally have no default: the caller must state
    its experimental thresholds. gamma uses b_rms_tol unless separately set.
    Inner optimizer success is required by default, so stationary values from
    failed inner solves cannot alone trigger the optional convergence stop.
    """
    p_tv_tol: float
    b_rms_tol: float
    gamma_rms_tol: float | None = None
    p_tv_per_gain_tol: float | None = None
    min_iterations: int = 50
    consecutive: int = 5
    require_optimizer_success: bool = True

    def __post_init__(self):
        if not isinstance(self.min_iterations, int) or self.min_iterations < 50:
            raise ValueError("min_iterations must be an integer >= 50")
        if self.consecutive not in (5, 10):
            raise ValueError("consecutive must be 5 or 10")
        for field in ("p_tv_tol", "b_rms_tol", "gamma_rms_tol", "p_tv_per_gain_tol"):
            value = getattr(self, field)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{field} must be finite and nonnegative")


def normalized_counts(counts):
    """Return a separate normalized mass dictionary; do not modify counts."""
    values = {int(key): float(value) for key, value in counts.items()}
    mass = sum(values.values())
    if not np.isfinite(mass) or mass <= 0 or any(not np.isfinite(v) or v < 0 for v in values.values()):
        return None
    return {key: value / mass for key, value in values.items()}


def count_law_tv(previous, current):
    """TV between two independently normalized count laws, including new states."""
    old, new = normalized_counts(previous), normalized_counts(current)
    if old is None or new is None:
        return float("nan")
    return .5 * sum(abs(old.get(key, 0.) - new.get(key, 0.)) for key in old.keys() | new.keys())


def _finite(value):
    value = float(value)
    return value if np.isfinite(value) else None


def _setup_probe(self, counts_smooth):
    self.convergence_records = []
    self.convergence_probe_metadata = {
        "mode": "observe" if self._convergence_stop_rule is None else "optional_stop",
        "stop_rule": None if self._convergence_stop_rule is None else asdict(self._convergence_stop_rule),
        "frozen_estimator_sha256": FROZEN_ESTIMATOR_SHA256,
        "original_tol": self.tol,
        "original_tol_scale": "squared Frobenius B change + squared Frobenius gamma change" if self.distribution == "Lognormal" else "squared Frobenius B change",
        "truth_used_for_stopping": False,
        "global_optimum_certified": False,
    }
    self._probe_previous_counts = dict(counts_smooth)
    self._probe_previous_B = self.B_hat.copy()
    self._probe_previous_gamma = self.gamma_hat.copy()
    self._probe_streak = 0


def _meets_rule(record, rule, family):
    required = (record["p_update_tv"], record["B_delta_rms"])
    if any(value is None for value in required):
        return False
    # Strict comparison follows the requested '< threshold' rule.
    qualifies = record["p_update_tv"] < rule.p_tv_tol and record["B_delta_rms"] < rule.b_rms_tol
    if family == "Lognormal":
        limit = rule.b_rms_tol if rule.gamma_rms_tol is None else rule.gamma_rms_tol
        qualifies = qualifies and record["gamma_delta_rms"] is not None and record["gamma_delta_rms"] < limit
    if rule.p_tv_per_gain_tol is not None:
        qualifies = qualifies and record["p_update_tv_per_gain"] is not None and record["p_update_tv_per_gain"] < rule.p_tv_per_gain_tol
    if rule.require_optimizer_success:
        qualifies = qualifies and record["optimizer_failures_this_iteration"] == 0
    return bool(qualifies)


def _observe_update(self, t, err, step, counts_smooth):
    # Each law is separately normalized, even when accumulated mass drifts.
    tv = count_law_tv(self._probe_previous_counts, counts_smooth)
    delta_B = self.B_hat - self._probe_previous_B
    delta_gamma = self.gamma_hat - self._probe_previous_gamma
    B_sq = float(np.sum(delta_B**2))
    gamma_sq = float(np.sum(delta_gamma**2))
    rule = self._convergence_stop_rule
    current_optimizer = self.optimizer_diagnostics[-self.J:]
    record = {
        "iteration": int(t), "gain": _finite(step),
        "p_update_tv": _finite(tv),
        "p_update_tv_per_gain": _finite(tv / step) if np.isfinite(step) and step > 0 else None,
        "B_delta_fro_sq": _finite(B_sq),
        "B_delta_rms": _finite(np.sqrt(B_sq / delta_B.size)),
        "gamma_delta_fro_sq": _finite(gamma_sq),
        "gamma_delta_rms": _finite(np.sqrt(gamma_sq / delta_gamma.size)),
        "original_parameter_change_sq": _finite(err),
        "p_count_mass": _finite(sum(counts_smooth.values())),
        "optimizer_failures_this_iteration": sum(not r["success"] for r in current_optimizer),
        "meets_optional_thresholds": None,
        "consecutive_qualifying_iterations": None,
    }
    if rule is not None:
        record["meets_optional_thresholds"] = _meets_rule(record, rule, self.distribution)
        self._probe_streak = self._probe_streak + 1 if record["meets_optional_thresholds"] else 0
        record["consecutive_qualifying_iterations"] = self._probe_streak
    self.convergence_records.append(record)
    self._probe_previous_counts = dict(counts_smooth)
    self._probe_previous_B = self.B_hat.copy()
    self._probe_previous_gamma = self.gamma_hat.copy()


def optional_stop_decision(record, rule: StopRule, max_iter: int):
    """Return (continue, reason); hard iteration cap takes precedence."""
    t = record["iteration"]
    if t >= max_iter:
        return False, "max_iter"
    if record["original_parameter_change_sq"] is None:
        return False, "nonfinite_parameter_change"
    ready = t >= rule.min_iterations and record["consecutive_qualifying_iterations"] >= rule.consecutive
    return (False, "convergence_probe_window") if ready else (True, None)


def _make_probed_psaem(base_class):
    method = None
    for parent in base_class.__mro__:
        candidate = parent.__dict__.get("PSAEM")
        if candidate is None:
            continue
        location = inspect.getsourcefile(candidate)
        if location and Path(location).is_file():
            if hashlib.sha256(Path(location).read_bytes()).hexdigest() == FROZEN_ESTIMATOR_SHA256:
                method = candidate
                break
    if method is None:
        raise RuntimeError("No hash-verified frozen PSAEM in class MRO")
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))

    class Transform(ast.NodeTransformer):
        loops = 0
        setup = 0
        observations = 0

        def visit_For(self, node):
            if isinstance(node.target, ast.Name) and node.target.id == "i" and ast.unparse(node.iter) == "range(self.N)":
                self.loops += 1
                replacement = ast.parse("A_cur, idx_cur = _probe_gibbs(self.X, A_cur, self.B_hat, self.gamma_hat, counts_smooth, self.kappa, self.distribution)").body[0]
                return ast.copy_location(replacement, node)
            return self.generic_visit(node)

        def visit_Assign(self, node):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if names == ["options"]:
                self.setup += 1
                hook = ast.parse("_probe_setup(self, counts_smooth)").body[0]
                return [ast.copy_location(hook, node), node]
            if names == ["iter_indicator"] and ast.unparse(node.value) == "self._continue(err, t)":
                self.observations += 1
                hook = ast.parse("_probe_observe(self, t, err, step, counts_smooth)").body[0]
                return [ast.copy_location(hook, node), node]
            return self.generic_visit(node)

    transform = Transform()
    tree = ast.fix_missing_locations(transform.visit(tree))
    if (transform.loops, transform.setup, transform.observations) != (3, 1, 3):
        raise RuntimeError(f"Unexpected AST edits: {(transform.loops, transform.setup, transform.observations)}")
    namespace = dict(method.__globals__)
    namespace.update(_probe_gibbs=gibbs_sweep, _probe_setup=_setup_probe, _probe_observe=_observe_update)
    exec(compile(tree, str(Path(__file__).resolve()) + ":generated_PSAEM", "exec"), namespace)
    return namespace["PSAEM"]


def convergence_probe_class(base_class=None, *, stop_rule: StopRule | None = None):
    """Factory for public fitting API: fitting.DAGEstimator = returned class.

    base_class may be FastObjectiveEstimator or its vectorized Gibbs subclass.
    With stop_rule=None the accelerated stochastic trajectory and original
    stopping decisions are unchanged; only diagnostics gain extra fields.
    With a rule the original tol/min_outer_iter no longer decide stopping, but
    max_iter remains a hard cap. The class does not enlarge that budget.
    """
    if base_class is None:
        from .local90_fast_objective import FastObjectiveEstimator
        base_class = FastObjectiveEstimator
    if stop_rule is not None and not isinstance(stop_rule, StopRule):
        raise TypeError("stop_rule must be StopRule or None")
    transformed = _make_probed_psaem(base_class)

    class ProbedEstimator(base_class):
        _convergence_stop_rule = stop_rule

        def PSAEM(self, ite):
            reset = getattr(self, "_reset_fast_objective", None)
            if reset is not None:
                reset()
            self._probe_stop_reason = None
            return transformed(self, ite)

        def _continue(self, err, t):
            if self._convergence_stop_rule is None:
                return super()._continue(err, t)
            continuing, reason = optional_stop_decision(
                self.convergence_records[-1], self._convergence_stop_rule, self.max_iter)
            self._probe_stop_reason = reason
            return continuing

        def _record_iteration(self, t, err, step, started, continuing, A_new):
            callback = self.iteration_callback
            self.iteration_callback = None
            try:
                super()._record_iteration(t, err, step, started, continuing, A_new)
            finally:
                self.iteration_callback = callback
            record = self.iteration_diagnostics[-1]
            record["convergence_probe"] = dict(self.convergence_records[-1])
            if self._convergence_stop_rule is not None:
                record["stopping_reason"] = self._probe_stop_reason
                self.stopping_reason = self._probe_stop_reason
            if callback is not None:
                callback(dict(record))

    ProbedEstimator.__name__ = "ConvergenceProbe" + base_class.__name__
    return ProbedEstimator
