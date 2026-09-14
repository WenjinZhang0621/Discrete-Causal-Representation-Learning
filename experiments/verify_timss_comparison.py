"""Audit saved TIMSS likelihoods, BIC values and multinomial graph discovery.

This checks the fitted artifacts; it does not rerun parameter optimization.
"""
import json
import os
from pathlib import Path
import sys

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ.setdefault(key, '1')
import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from dcrl.timss.graph import learn_graph
from dcrl.timss.output import file_hash


def main():
    folder = ROOT/'data/TIMSS'
    artifact = json.loads((folder/'latent_law_comparison.json').read_text())
    for name, expected in artifact['input_sha256'].items():
        assert file_hash(folder/name) == expected, f'Input file changed: {name}'
    x = np.genfromtxt(folder/'time.csv', delimiter=',')
    x = np.clip(x[~np.isnan(x).all(axis=1)], np.e, np.e**6)
    q = np.genfromtxt(folder/'Q.csv', delimiter=',')[:, 1:]
    N, J = x.shape
    K = q.shape[1]
    assert (N, J, K) == (artifact['N'], artifact['J'], artifact['K'])
    measurement_parameters = 2*J + int(q.sum())
    states = ((np.arange(2**K)[:, None] >> np.arange(K-1, -1, -1)) & 1)
    design = np.column_stack([np.ones(len(states)), states])
    observed = np.isfinite(x)
    y = np.zeros_like(x)
    y[observed] = np.log(x[observed])
    reference = json.loads((folder/'timss_refit.json').read_text())['p_est']
    for row in artifact['records']:
        fitted = row['fit']
        p, beta, gamma = [np.asarray(fitted[key]) for key in ['p', 'beta', 'gamma']]
        np.testing.assert_allclose(p.sum(), 1., atol=1e-12)
        assert (p >= 0).all() and (gamma > 0).all()
        assert np.all(beta[:, 1:][q == 0] == 0)
        means = design @ beta.T
        terms = y @ (means/gamma).T - .5*observed @ (means**2/gamma + np.log(2*np.pi*gamma)).T
        terms -= .5*(y*y/gamma).sum(axis=1)[:, None]
        loglik = float(logsumexp(terms + np.log(np.maximum(p, 1e-300))[None, :], axis=1).sum())
        d_latent = {'independent': K, 'ho1': 2*K, 'ho2': 2*K+1, 'saturated': 2**K-1}.get(row['key'])
        if 'ratio' in row:
            dag = np.asarray(fitted['dag'])
            d_latent = int(sum(2**int(n) for n in dag.sum(axis=0)))
            product = np.ones(len(p))
            for k in range(K):
                parents = np.flatnonzero(dag[:, k])
                codes = states[:, parents] @ (2**np.arange(len(parents)-1, -1, -1))
                totals = np.bincount(codes, weights=p)
                ones = np.bincount(codes, weights=p*states[:, k])
                theta = ones[codes]/totals[codes]
                product *= np.where(states[:, k], theta, 1-theta)
            np.testing.assert_allclose(product, p, rtol=1e-8, atol=1e-12)
            ratio = np.log(N) if row['ratio'] == 'logN' else float(row['ratio'])
            record, samples, _ = learn_graph(reference, N, K, ratio, row['graph_seed'])
            np.testing.assert_array_equal(record['G'].graph, row['cpdag'])
            assert len(samples) == row['sample_size']
        assert d_latent == row['latent_parameters'] == fitted['d_latent']
        d = measurement_parameters + d_latent
        assert d == row['parameters']
        bic = -2*loglik + d*np.log(N)
        np.testing.assert_allclose(loglik, row['loglik'], rtol=0, atol=1e-6)
        np.testing.assert_allclose(bic, row['bic'], rtol=0, atol=2e-6)
        print(f"{row['key']:15s} loglik={loglik:.5f} d={d:3d} BIC={bic:.5f}")
    print('Verified all 13 saved scores and all nine multinomial graphs.')


if __name__ == '__main__':
    main()
