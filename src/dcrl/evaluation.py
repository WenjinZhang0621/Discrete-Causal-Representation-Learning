import numpy as np
from causallearn.search.ScoreBased.GES import ges

from .utils import binary


def stitch_full_A(ZZ, ZX, n_x):
    n_z = ZZ.shape[0]
    assert ZX.shape == (n_z, n_x)
    A = np.block([
        [ZZ, ZX],
        [np.zeros((n_x, n_z), dtype=int), np.zeros((n_x, n_x), dtype=int)],
    ])
    np.fill_diagonal(A, 0)
    return A


def shd_cpdag(M1, M2):
    n = M1.shape[0]

    def code(M, i, j):
        a, b = M[i, j], M[j, i]
        if a == 0 and b == 0:
            return 0
        if a == -1 and b == -1:
            return 2
        if b == 1 and a == -1:
            return 1
        if a == 1 and b == -1:
            return -1
        return 2

    shd = 0
    for i in range(n):
        for j in range(i + 1, n):
            c1 = code(M1, i, j)
            c2 = code(M2, i, j)
            if c1 != c2:
                shd += 1
    return shd


def sample_latent_states_from_p(p_hat, K, sample_size):
    sam = np.zeros((sample_size, K))
    qwe = p_hat.flatten()
    counts = np.random.multinomial(sample_size, qwe)
    A_src = binary(np.arange(2 ** K), K)
    n = 0
    for a in range(2 ** K):
        cnt = int(counts[a])
        if cnt > 0:
            sam[n:n + cnt, :K] = np.tile(A_src[a, :K], (cnt, 1))
            n += cnt
    return sam


def recover_full_graph_from_estimates(p_hat, B_hat, K, J, sample_size):
    sam = sample_latent_states_from_p(p_hat, K, sample_size)
    record = ges(sam, score_func="local_score_BDeu")
    cpdag_ZZ_est = record["G"].graph
    ZX_est = (B_hat[:, 1:] != 0).astype(int).T
    return stitch_full_A(cpdag_ZZ_est, ZX_est, n_x=J)


def compute_shd_triplet(p_hat, B_hat, K, J, Q_N, truth_graph, Q_truth):
    A_truth = stitch_full_A(truth_graph.graph, Q_truth.T, n_x=J)
    A_est_1 = recover_full_graph_from_estimates(p_hat, B_hat, K, J, Q_N)
    A_est_2 = recover_full_graph_from_estimates(p_hat, B_hat, K, J, 2 * Q_N)
    A_est_3 = recover_full_graph_from_estimates(p_hat, B_hat, K, J, 3 * Q_N)
    shd_val = shd_cpdag(A_truth, A_est_1)
    shd2_val = shd_cpdag(A_truth, A_est_2)
    shd3_val = shd_cpdag(A_truth, A_est_3)
    return shd_val, shd2_val, shd3_val
