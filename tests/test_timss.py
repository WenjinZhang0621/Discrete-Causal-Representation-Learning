"""TIMSS resampling parity, reproducibility, and observed-data fit checks."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from scipy.special import logsumexp
from causallearn.search.ScoreBased.GES import ges

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from dcrl.timss.graph import learn_graph, sample_latents
from dcrl.timss.refit import fit


def checkpoint():
    return json.loads((REPO/'data/TIMSS/timss_refit.json').read_text())


@pytest.mark.parametrize('ratio', [.05, .1, .2, .3, .5, .75, 1., 2., np.log(620)])
def test_multinomial_sample_matches_published_algorithm(ratio):
    p = np.asarray(checkpoint()['p_est'])
    samples, counts = sample_latents(p, 620, fn_ratio=ratio, seed=1)
    expected = np.random.RandomState(1).multinomial(max(1, int(np.floor(ratio*620))), p/p.sum())
    states = ((np.arange(128)[:, None] >> np.arange(6, -1, -1)) & 1)
    np.testing.assert_array_equal(counts, expected)
    np.testing.assert_array_equal(samples, np.repeat(states, expected, axis=0))


def test_sampling_is_reproducible_and_does_not_modify_global_rng():
    p = checkpoint()['p_est']
    np.random.seed(782)
    state = np.random.get_state()
    a, _ = sample_latents(p, 620, seed=1)
    b, _ = sample_latents(p, 620, seed=1)
    after = np.random.get_state()
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]


def test_no_zero_padding_and_integer_rounding():
    p = np.zeros(128)
    p[127] = 1
    sample, counts = sample_latents(p, 621, fn_ratio=.5, seed=1)
    assert sample.shape == (310, 7)
    assert counts.sum() == 310
    assert (sample == 1).all()


@pytest.mark.parametrize('ratio', [0, -1, float('nan'), float('inf')])
def test_invalid_ratio_is_rejected(ratio):
    with pytest.raises(ValueError):
        sample_latents(np.ones(128), 620, fn_ratio=ratio)


def test_ges_uses_actual_sample_and_reproduces_half_n_graph():
    p = checkpoint()['p_est']
    record, samples, counts = learn_graph(p, 620, fn_ratio=.5, seed=1)
    expected = ges(samples, score_func='local_score_BDeu')['G'].graph
    np.testing.assert_array_equal(record['G'].graph, expected)
    assert np.count_nonzero(expected)//2 == 10
    assert counts.sum() == 310


def test_vectorized_fit_likelihood_matches_direct_observed_cell_calculation():
    rng = np.random.RandomState(19)
    q = np.vstack([np.eye(3, dtype=int)]*3)
    x = np.exp(rng.normal(2, .4, size=(40, 9)))
    x[::3, 1] = np.nan
    result = fit(x, q, seed=7, max_iter=20)
    states = ((np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1)
    means = np.column_stack([np.ones(8), states]) @ result['beta'].T
    direct = 0.
    for row in x:
        obs = np.isfinite(row)
        y = np.log(row[obs])
        gamma = result['gamma'][obs]
        terms = -.5*((y-means[:, obs])**2/gamma + np.log(2*np.pi*gamma)).sum(axis=1)
        direct += logsumexp(terms + np.log(np.maximum(result['p'], 1e-300)))
    assert result['loglik'] == pytest.approx(direct, abs=1e-8)
    assert np.all(np.diff(result['trace'][:, 1]) >= -1e-8)
    assert np.all(result['beta'][:, 1:][q == 0] == 0)


def test_default_saved_fit_cli_repeats_exactly_with_plot(tmp_path):
    for name in ['first', 'second']:
        subprocess.run([sys.executable, str(REPO/'experiments/run_timss_from_saved_p.py'),
                        '--results_dir', str(tmp_path/name)],
                       check=True, capture_output=True, text=True)
    for name in ['latent_counts.npy', 'latent_samples_from_phat.npy', 'latent_graph_estimated.npy']:
        np.testing.assert_array_equal(np.load(tmp_path/'first'/name), np.load(tmp_path/'second'/name))
    summary = json.loads((tmp_path/'first/summary.json').read_text())
    assert summary['sampling'] == 'multinomial'
    assert summary['N'] == 620 and summary['f_N'] == 124
    assert summary['fn_ratio'] == .2 and summary['graph_seed'] == 1
    assert summary['graph_edges'] == 7
    assert (tmp_path/'first/latent_graph_estimated.png').read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_default_graph_matches_corrected_comparison():
    record, samples, _ = learn_graph(checkpoint()['p_est'], 620)
    expected = np.array([[0,1,0,0,0,0,0],[-1,0,1,0,0,0,0],
                         [0,-1,0,1,0,1,0],[0,0,-1,0,0,0,-1],
                         [0,0,0,0,0,-1,0],[0,0,-1,0,-1,0,-1],
                         [0,0,0,-1,0,-1,0]])
    np.testing.assert_array_equal(record['G'].graph, expected)
    assert samples.shape == (124, 7)
