"""Distribution equivalence, large-state sampling, and main-pipeline coverage."""
from types import SimpleNamespace
import numpy as np
import pytest

from test_sparse_initialization import original_module
from dcrl.estimator import DAGEstimator
from dcrl.evaluation import sample_latent_states_from_p, recover_full_graph_from_estimates
from dcrl.latent_law import SparseLatentLaw


@pytest.mark.parametrize('counts', [{}, {1: 0.0}, {0: 2., 3: 5.5}, dict.fromkeys(range(8), 2.)])
@pytest.mark.parametrize('smoothing', [0., 0.7])
def test_finalization_matches_public_probability_law(counts, smoothing):
    old = SimpleNamespace(K=3, L_final=smoothing)
    original_module('estimator').DAGEstimator._saem_finalize_p(old, counts)
    new = SimpleNamespace(K=3, L_final=smoothing, dense_p_output=False)
    DAGEstimator._saem_finalize_p(new, counts)
    np.testing.assert_allclose(new.p_hat.to_dense(), old.p_hat, rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose([new.p_hat.probability(s) for s in range(8)], old.p_hat[:, 0])
    new.dense_p_output = True
    DAGEstimator._saem_finalize_p(new, counts)
    np.testing.assert_allclose(new.p_hat, old.p_hat, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize('smoothing', [0., 1.5])
def test_direct_samples_follow_original_law(smoothing):
    law = SparseLatentLaw.from_counts(4, {0: 2., 9: 1., 15: 4.}, smoothing)
    np.random.seed(2417)
    samples = sample_latent_states_from_p(law, 4, 200000)
    encoded = (samples @ np.array([8, 4, 2, 1])).astype(int)
    frequencies = np.bincount(encoded, minlength=16) / len(samples)
    np.testing.assert_allclose(frequencies, law.to_dense()[:, 0], atol=.004, rtol=0)
    assert law.sample(0).shape == (0, 4)


def test_large_state_sparse_export_and_sampling(tmp_path, monkeypatch):
    law = SparseLatentLaw.from_counts(80, {1 << 79: 2., 1: 1.}, .5)
    def no_dense(*args, **kwargs):
        raise AssertionError('No dense state enumeration is allowed')
    monkeypatch.setattr(SparseLatentLaw, 'to_dense', no_dense)
    filename = tmp_path / 'law.npz'
    law.save_npz(filename)
    restored = SparseLatentLaw.load_npz(filename)
    assert restored == law
    assert filename.stat().st_size < 4096
    np.random.seed(4182)
    samples = sample_latent_states_from_p(restored, 80, 1000)
    assert samples.shape == (1000, 80)
    assert np.isin(samples, [0, 1]).all()
    assert restored.stored_state_count == 2
    assert (samples[:, 0] == 1).mean() > .5


def test_psaem_gibbs_preserves_more_than_64_state_bits(monkeypatch):
    import dcrl.estimator as estimator_module
    obj = object.__new__(DAGEstimator)
    obj.N, obj.J, obj.K, obj.C = 1, 1, 80, 1
    obj.distribution = 'Poisson'
    obj.max_iter, obj.tol, obj.tau, obj.kappa = 1, .05, .1, 1e-200
    obj.L_final, obj.dense_p_output = 0., False
    obj.X = np.ones((1, 1))
    A = np.zeros((1, 80)); A[0, [0, -1]] = 1
    B = np.zeros((1, 81)); B[0, 0] = 1
    obj.init = lambda ite: ({}, B, np.ones((1, 1)), A)
    monkeypatch.setattr(estimator_module, 'minimize', lambda fun, x0, **kwargs: SimpleNamespace(x=x0))
    monkeypatch.setattr(SparseLatentLaw, 'to_dense', lambda self: pytest.fail('Dense export called'))
    obj.PSAEM(0)
    assert obj.p_hat.masses == {(1 << 79) | 1: 1.}
    np.testing.assert_array_equal(obj.p_hat.sample(3), np.repeat(A, 3, axis=0))


def test_existing_ges_pipeline_accepts_sparse_law():
    np.random.seed(4511)
    law = SparseLatentLaw.from_counts(3, {0: 3., 3: 1., 5: 2., 7: 4.})
    graph = recover_full_graph_from_estimates(law, np.ones((9, 4)), 3, 9, 200)
    assert graph.shape == (12, 12)
