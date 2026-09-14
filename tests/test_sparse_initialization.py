"""Validate sparse initialization against the audited public implementation."""
import contextlib
import io
from pathlib import Path
import random
import subprocess
import sys
import types

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from dcrl.config import ExperimentConfig
from dcrl.data_generator import GenerateData
from dcrl.estimator import DAGEstimator
from dcrl.initialization import initialize_parameters, _empirical_latent_probabilities

AUDITED_COMMIT = '190f198f9134655cca7c517905cdf61a994da6a8'

def original_module(filename):
    source = subprocess.check_output(
        ['git', '-C', str(REPO), 'show', f'{AUDITED_COMMIT}:src/dcrl/{filename}.py'], text=True)
    module = types.ModuleType(f'dcrl.reference_{filename}')
    module.__package__ = 'dcrl'
    exec(compile(source, f'reference_{filename}.py', 'exec'), module.__dict__)
    return module


@pytest.mark.parametrize('family', ['Lognormal', 'Bernoulli', 'Poisson'])
def test_same_initial_estimates_and_dense_compatibility(family):
    random.seed(9301)
    np.random.seed(9301)
    config = ExperimentConfig(N=300, K=4, distribution=family).finalize()
    generator = GenerateData.from_config(config)
    generator.generate_latent_data()
    X = generator.generate_data()
    args = (X, config.J, config.K, family, config.epsilon)
    with threadpool_limits(limits=1):
        reference = original_module('initialization').initialize_parameters(*args)
        dense = initialize_parameters(*args)
        sparse = initialize_parameters(*args, sparse=True)
    for old, new in zip(reference, dense):
        np.testing.assert_array_equal(old, new)
    for old, new in zip(reference[1:], sparse[1:]):
        np.testing.assert_array_equal(old, new)
    reconstructed = np.zeros_like(reference[0])
    for state, probability in sparse[0].items():
        reconstructed[state, 0] = probability
    np.testing.assert_array_equal(reference[0], reconstructed)
    assert len(sparse[0]) <= config.N


def test_sparse_histogram_does_not_allocate_state_space(monkeypatch):
    A = np.zeros((3, 80))
    A[0, 0] = A[1, 0] = 1
    A[2, -1] = 1
    def fail_allocation(*args, **kwargs):
        raise AssertionError('Sparse counting must not allocate a dense vector')
    monkeypatch.setattr(np, 'zeros', fail_allocation)
    assert _empirical_latent_probabilities(A, 80, sparse=True) == {1 << 79: 2/3, 1: 1/3}


def test_seesaw_initial_estimates_and_sparse_law_match_original():
    from dcrl.seesaw.init import init_seesaw
    rng = np.random.RandomState(8421)
    X = rng.binomial(1, rng.uniform(.1, .9, size=36), size=(300, 36))
    with threadpool_limits(limits=1):
        reference = original_module('seesaw/init').init_seesaw(K=4, X=X)
        dense = init_seesaw(K=4, X=X)
        sparse = init_seesaw(K=4, X=X, sparse=True)
    for old, new in zip(reference, dense):
        np.testing.assert_array_equal(old, new)
    for old, new in zip(reference[1:], sparse[1:]):
        np.testing.assert_array_equal(old, new)
    restored = np.zeros_like(reference[0])
    for state, probability in sparse[0].items():
        restored[state, 0] = probability
    np.testing.assert_array_equal(restored, reference[0])


@pytest.mark.parametrize('family', ['Lognormal', 'Bernoulli', 'Poisson'])
def test_saem_result_unchanged_with_sparse_initial_law(family):
    config = ExperimentConfig(N=100, K=4, distribution=family, max_iter=3).finalize()
    kwargs = {k: getattr(config, k) for k in (
        'N', 'J', 'K', 'Q_type', 'DAG_type', 'distribution', 'algorithm',
        'tau', 'pen', 'max_iter', 'tol', 'C', 'epsilon', 'Q_N', 'kappa', 'L_final')}
    kwargs.update(vars(config.bounds))
    reference = original_module('estimator')
    reference.initialize_parameters = original_module('initialization').initialize_parameters
    results = []
    for cls in (reference.DAGEstimator, DAGEstimator):
        random.seed(9302)
        np.random.seed(9302)
        obj = cls(**kwargs)
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()):
            results.append(obj.PSAEM(0))
    np.testing.assert_allclose(results[0][0], results[1][0].to_dense(), rtol=1e-14, atol=1e-16)
    for old, new in zip(results[0][1:], results[1][1:]):
        np.testing.assert_array_equal(old, new)
    # The actual public PSAEM initializer selects sparse mode.
    assert isinstance(obj.init(0)[0], dict)
