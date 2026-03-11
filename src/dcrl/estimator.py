import numpy as np
from collections import Counter
from scipy.optimize import minimize
from scipy.special import expit

from .data_generator import GenerateData
from .evaluation import compute_shd_triplet
from .initialization import initialize_parameters
from .utils import binary, TLP, thres, initialize_function


class DAGEstimator(GenerateData):
    def __init__(
        self,
        N,
        J,
        K,
        Q_type,
        DAG_type,
        distribution,
        algorithm,
        upb,
        lob,
        lob2,
        upb2,
        lob3,
        upb3,
        lob4,
        upb4,
        lob5,
        upb5,
        tau,
        pen,
        max_iter,
        tol,
        C,
        epsilon,
        Q_N,
        kappa,
        L_final,
    ):
        super().__init__(N, J, K, Q_type, DAG_type, distribution, lob, upb, lob2, upb2, lob3, upb3, lob4, upb4, lob5, upb5)
        self.tau = tau
        self.pen = pen
        self.max_iter = max_iter
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.p_hat = None
        self.B_hat = None
        self.gamma_hat = None
        self.A_hat = None
        self.Q_N = Q_N
        self.algorithm = algorithm
        self.kappa = kappa
        self.L_final = L_final

    def ftn_T(self, X):
        if self.distribution == "Poisson":
            return np.atleast_2d(np.array(X))
        if self.distribution == "Lognormal":
            T_1 = -(np.log(X)) ** 2
            T_2 = np.log(X)
            return np.vstack((T_1, T_2))
        if self.distribution == "Bernoulli":
            return np.atleast_2d(np.array(X))
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def ftn_h(self, Y):
        if self.distribution == "Poisson":
            return np.atleast_2d(np.log(Y[0, :]))
        if self.distribution == "Lognormal":
            Y_1 = Y[0, :]
            Y_2 = np.maximum(Y[1, :], 1e-100)
            return np.vstack((1 / (2 * Y_2), Y_1 / Y_2))
        if self.distribution == "Bernoulli":
            return np.atleast_2d(Y[0, :])
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def ftn_A(self, eta):
        if self.distribution == "Poisson":
            return np.exp(eta)
        if self.distribution == "Lognormal":
            eta_1 = np.maximum(eta[:, 0], 1e-100)
            eta_2 = eta[:, 1]
            return (eta_2 ** 2 / (4 * eta_1) + np.log(1 / (2 * eta_1)) / 2).reshape(len(eta_1), 1)
        if self.distribution == "Bernoulli":
            return np.log1p(np.exp(eta))
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def objective(self, phi, j):
        if self.distribution == "Poisson":
            def obj(x):
                beta_long = x.reshape(self.K + 1, 1)
                tmp = 0
                penalty = self.pen * TLP(x, self.tau)
                for a in range(2 ** self.K):
                    eta = np.log(np.dot((np.insert(binary(a, self.K).flatten(), 0, 1)).reshape(1, 1 + self.K), beta_long))
                    tmp += np.dot(eta.T, self.ftn_T(self.X[:, j])) @ phi[:, a] - np.sum(self.ftn_A(eta.T)) * np.sum(phi[:, a])
                return penalty - tmp
            return obj

        if self.distribution == "Lognormal":
            def obj(x):
                beta_long = x[:-1].reshape(self.K + 1, 1)
                gamma = np.array(x[-1]).reshape(1, 1)
                tmp = 0
                penalty = self.pen * TLP(x, self.tau)
                for a in range(2 ** self.K):
                    eta = self.ftn_h(np.vstack((np.dot((np.insert(binary(a, self.K).flatten(), 0, 1)).reshape(1, 1 + self.K), beta_long), gamma)))
                    tmp += np.dot(eta.T, self.ftn_T(self.X[:, j])) @ phi[:, a] - np.sum(self.ftn_A(eta.T)) * np.sum(phi[:, a])
                return penalty - tmp
            return obj

        if self.distribution == "Bernoulli":
            def obj(x):
                beta_long = x.reshape(self.K + 1, 1)
                tmp = 0
                penalty = self.pen * TLP(x, self.tau)
                for a in range(2 ** self.K):
                    eta = np.dot((np.insert(binary(a, self.K).flatten(), 0, 1)).reshape(1, 1 + self.K), beta_long)
                    tmp += np.dot(eta.T, self.ftn_T(self.X[:, j])) @ phi[:, a] - np.sum(self.ftn_A(eta.T)) * np.sum(phi[:, a])
                return penalty - tmp
            return obj

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def ftn_pen(self, beta_gamma):
        if self.distribution == "Poisson":
            return self.pen * TLP(beta_gamma[1:], self.tau)
        if self.distribution == "Lognormal":
            return self.pen * TLP(beta_gamma[1:-1], self.tau)
        if self.distribution == "Bernoulli":
            return self.pen * TLP(beta_gamma[1:], self.tau)
        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def F_1_SAEM(self, Xj, A_sample_long):
        A_sample_long = np.array(A_sample_long, dtype=np.float32)

        if self.distribution == "Poisson":
            def obj(x):
                beta_long = x.reshape(self.K + 1, 1)
                tmp = 0
                for c in range(self.C):
                    A_beta = np.dot(A_sample_long[:, :, c], beta_long)
                    eta = self.ftn_h(A_beta.T)
                    tmp += np.sum(eta * self.ftn_T(Xj)) - np.sum(self.ftn_A(eta.T))
                return -(tmp / self.C)
            return obj

        if self.distribution == "Lognormal":
            def obj(x):
                beta_long = x[:-1].reshape(self.K + 1, 1)
                gamma = x[-1].reshape(1, 1)
                tmp = 0
                for c in range(self.C):
                    A_beta = np.dot(A_sample_long[:, :, c], beta_long)
                    eta = self.ftn_h(np.vstack((A_beta.T, np.tile(gamma, (1, self.N)))))
                    tmp += np.sum(eta * self.ftn_T(Xj)) - np.sum(self.ftn_A(eta.T))
                return -(tmp / self.C)
            return obj

        if self.distribution == "Bernoulli":
            def obj(x):
                beta_long = x.reshape(self.K + 1, 1)
                tmp = 0
                for c in range(self.C):
                    A_beta = np.dot(A_sample_long[:, :, c], beta_long)
                    eta = self.ftn_h(A_beta.T)
                    tmp += np.sum(eta * self.ftn_T(Xj)) - np.sum(self.ftn_A(eta.T))
                return -(tmp / self.C)
            return obj

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def _saem_finalize_p(self, counts_smooth):
        U_T = (1 << self.K) - len(counts_smooth)
        nu_dense = np.zeros(1 << self.K, dtype=float)

        if counts_smooth:
            idxs = np.fromiter(counts_smooth.keys(), dtype=np.int64)
            vals = np.fromiter((counts_smooth[k] for k in idxs), dtype=float)
            nu_dense[idxs] = vals

        if self.L_final > 0.0 and U_T > 0:
            nu_dense += (self.L_final / U_T)

        total = nu_dense.sum()
        if total > 0:
            nu_dense /= total
        else:
            nu_dense.fill(1.0 / (1 << self.K))

        self.p_hat = nu_dense.reshape(-1, 1)

    def PSAEM(self, ite):
        self.p_hat, self.B_hat, self.gamma_hat, self.A_hat = self.init(ite)
        pow2 = (1 << np.arange(self.K - 1, -1, -1))
        err = 1
        t = 0
        loglik = 0
        rows_idx0 = (self.A_hat * pow2).sum(axis=1).astype(np.int64)
        counts_smooth = Counter()
        for u, c in zip(*np.unique(rows_idx0, return_counts=True)):
            counts_smooth[int(u)] = float(c)

        options = {"disp": False, "maxiter": self.max_iter}

        if self.distribution == "Poisson":
            lb = np.zeros(self.K + 1)
            ub = np.concatenate(([2], 3 * np.ones(self.K)))
            B_update = np.zeros((self.J, self.K + 1))
            iter_indicator = True
            A_new = self.A_hat.copy()
            A_sample_long = np.zeros((self.N, self.K + 1, self.C))
            f_old_1 = [initialize_function() for _ in range(self.J)]

            while iter_indicator:
                A_cur = A_new.copy()
                counts_hat = Counter()
                ones_col = np.ones((self.N, 1), dtype=int)

                for c in range(self.C):
                    idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64)
                    for i in range(self.N):
                        z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), self.B_hat.T).astype(float)
                        T_i = self.ftn_T(self.X[i, :]).T
                        for k in np.random.permutation(self.K):
                            mask = int(pow2[k])
                            col = self.B_hat[:, k + 1].astype(float)
                            if A_cur[i, k] == 1:
                                z0, z1 = z_i - col, z_i
                                idx1, idx0 = idx_cur[i] | mask, idx_cur[i] & ~mask
                            else:
                                z0, z1 = z_i, z_i + col
                                idx1, idx0 = idx_cur[i] | mask, idx_cur[i] & ~mask

                            log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + self.kappa) - np.log(counts_smooth.get(idx0, 0.0) + self.kappa)
                            eta1 = self.ftn_h(np.maximum(z1.reshape(1, -1), 1e-150))
                            eta2 = self.ftn_h(np.maximum(z0.reshape(1, -1), 1e-150))
                            eta1T = eta1.T
                            eta2T = eta2.T
                            dots1 = np.sum(eta1T * T_i, axis=1)
                            dots0 = np.sum(eta2T * T_i, axis=1)
                            A1 = self.ftn_A(eta1T)[:, 0]
                            A0 = self.ftn_A(eta2T)[:, 0]
                            pa1 = float(np.sum(dots1 - A1))
                            pa0 = float(np.sum(dots0 - A0))
                            prob1 = expit(log_prior_ratio + (pa1 - pa0))
                            new_bit = np.random.binomial(1, prob1)
                            if new_bit != A_cur[i, k]:
                                A_cur[i, k] = new_bit
                                z_i = z1 if new_bit == 1 else z0
                                idx_cur[i] = idx1 if new_bit == 1 else idx0

                    A_sample_long[:, :, c] = np.hstack((ones_col, A_cur))
                    uniq, cnts = np.unique(idx_cur, return_counts=True)
                    for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                        counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)

                c0, t0 = 0.5, 10.0
                step = c0 / (t + t0)
                if self.C > 1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx] /= float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx] *= (1.0 - step)
                for k_idx, v in counts_hat.items():
                    counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

                A_new = A_cur.copy()
                for j in range(self.J):
                    Xj = self.X[:, j].copy()
                    f_loglik = self.F_1_SAEM(Xj, A_sample_long)

                    def update_f_old_1(x, f_old=f_old_1[j], f_new=f_loglik):
                        return (1 - step) * f_old(x) + step * f_new(x)

                    f_old_1[j] = update_f_old_1

                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma) + self.ftn_pen(beta_gamma)

                    opt_result = minimize(f_j, self.B_hat[j, :].flatten(), bounds=list(zip(lb, ub)), method="SLSQP", options=options)
                    B_update[j, :] = opt_result.x

                err = np.linalg.norm(self.B_hat - np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], self.tau)), axis=1), "fro") ** 2
                self.B_hat = np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], self.tau)), axis=1)
                t += 1
                print(f"SAEM Iteration {t}, Err {err:.5f}")
                iter_indicator = (abs(err) > self.tol and t < self.max_iter)

            self._saem_finalize_p(counts_smooth)
            return self.p_hat, self.B_hat, self.A_hat, t, loglik

        if self.distribution == "Lognormal":
            lb = np.concatenate(([-2], np.zeros(self.K + 1)))
            ub = np.concatenate((4 * np.ones(self.K + 1), [2]))
            B_update = np.zeros((self.J, self.K + 1))
            gamma_update = np.zeros((self.J, 1))
            iter_indicator = True
            A_new = self.A_hat.copy()
            A_sample_long = np.zeros((self.N, self.K + 1, self.C))
            f_old_1 = [initialize_function() for _ in range(self.J)]

            while iter_indicator:
                A_cur = A_new.copy()
                counts_hat = Counter()
                ones_col = np.ones((self.N, 1), dtype=int)

                for c in range(self.C):
                    idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64)
                    for i in range(self.N):
                        z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), self.B_hat.T).astype(float)
                        T_i = self.ftn_T(self.X[i, :]).T
                        for k in np.random.permutation(self.K):
                            mask = int(pow2[k])
                            col = self.B_hat[:, k + 1].astype(float)
                            if A_cur[i, k] == 1:
                                z0, z1 = z_i - col, z_i
                                idx1, idx0 = idx_cur[i] | mask, idx_cur[i] & ~mask
                            else:
                                z0, z1 = z_i, z_i + col
                                idx1, idx0 = idx_cur[i] | mask, idx_cur[i] & ~mask

                            log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + self.kappa) - np.log(counts_smooth.get(idx0, 0.0) + self.kappa)
                            eta1 = self.ftn_h(np.vstack((z1.reshape(1, -1), self.gamma_hat.reshape(1, -1))))
                            eta2 = self.ftn_h(np.vstack((z0.reshape(1, -1), self.gamma_hat.reshape(1, -1))))
                            eta1T = eta1.T
                            eta2T = eta2.T
                            dots1 = np.sum(eta1T * T_i, axis=1)
                            dots0 = np.sum(eta2T * T_i, axis=1)
                            A1 = self.ftn_A(eta1T)[:, 0]
                            A0 = self.ftn_A(eta2T)[:, 0]
                            pa1 = float(np.sum(dots1 - A1))
                            pa0 = float(np.sum(dots0 - A0))
                            prob1 = expit(log_prior_ratio + (pa1 - pa0))
                            new_bit = np.random.binomial(1, prob1)
                            if new_bit != A_cur[i, k]:
                                A_cur[i, k] = new_bit
                                z_i = z1 if new_bit == 1 else z0
                                idx_cur[i] = idx1 if new_bit == 1 else idx0

                    A_sample_long[:, :, c] = np.hstack((ones_col, A_cur))
                    uniq, cnts = np.unique(idx_cur, return_counts=True)
                    for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                        counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)

                c0, t0 = 0.5, 10.0
                step = c0 / (t + t0)
                if self.C > 1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx] /= float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx] *= (1.0 - step)
                for k_idx, v in counts_hat.items():
                    counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

                A_new = A_cur.copy()
                for j in range(self.J):
                    Xj = self.X[:, j]
                    f_loglik = self.F_1_SAEM(Xj, A_sample_long)

                    def update_f_old_1(x, f_old=f_old_1[j], f_new=f_loglik):
                        return (1 - step) * f_old(x) + step * f_new(x)

                    f_old_1[j] = update_f_old_1

                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma) + self.ftn_pen(beta_gamma)

                    opt_result = minimize(
                        f_j,
                        np.concatenate((self.B_hat[j, :].flatten(), np.array(self.gamma_hat[j]))),
                        bounds=list(zip(lb, ub)),
                        method="SLSQP",
                        options=options,
                    )
                    B_update[j, :] = opt_result.x[:-1]
                    gamma_update[j] = opt_result.x[-1]

                err = (
                    np.linalg.norm(self.B_hat - np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], self.tau)), axis=1), "fro") ** 2
                    + np.linalg.norm(self.gamma_hat - gamma_update, "fro") ** 2
                )
                self.B_hat = np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], self.tau)), axis=1)
                self.gamma_hat = gamma_update.copy()
                t += 1
                print(f"SAEM Iteration {t}, Err {err:.5f}")
                iter_indicator = (abs(err) > self.tol and t < self.max_iter)

            self._saem_finalize_p(counts_smooth)
            return self.p_hat, self.B_hat, self.gamma_hat, self.A_hat, t, loglik

        if self.distribution == "Bernoulli":
            lb = np.concatenate(([-5], -2.5 * np.ones(self.K)))
            ub = 5 * np.ones(self.K + 1)
            B_update = np.zeros((self.J, self.K + 1))
            iter_indicator = True
            A_new = self.A_hat.copy()
            A_sample_long = np.zeros((self.N, self.K + 1, self.C))
            f_old_1 = [initialize_function() for _ in range(self.J)]

            while iter_indicator:
                A_cur = A_new.copy()
                counts_hat = Counter()
                ones_col = np.ones((self.N, 1), dtype=int)

                for c in range(self.C):
                    idx_cur = (A_cur * pow2).sum(axis=1).astype(np.int64)
                    for i in range(self.N):
                        z_i = np.dot(np.insert(A_cur[i], 0, 1, axis=0), self.B_hat.T).astype(float)
                        T_i = self.ftn_T(self.X[i, :]).T
                        for k in np.random.permutation(self.K):
                            mask = int(pow2[k])
                            col = self.B_hat[:, k + 1].astype(float)
                            if A_cur[i, k] == 1:
                                z0, z1 = z_i - col, z_i
                                idx1, idx0 = idx_cur[i] | mask, idx_cur[i] & ~mask
                            else:
                                z0, z1 = z_i, z_i + col
                                idx1, idx0 = idx_cur[i] | mask, idx_cur[i] & ~mask

                            log_prior_ratio = np.log(counts_smooth.get(idx1, 0.0) + self.kappa) - np.log(counts_smooth.get(idx0, 0.0) + self.kappa)
                            eta1 = self.ftn_h(z1.reshape(1, -1))
                            eta2 = self.ftn_h(z0.reshape(1, -1))
                            eta1T = eta1.T
                            eta2T = eta2.T
                            dots1 = np.sum(eta1T * T_i, axis=1)
                            dots0 = np.sum(eta2T * T_i, axis=1)
                            A1 = self.ftn_A(eta1T)[:, 0]
                            A0 = self.ftn_A(eta2T)[:, 0]
                            pa1 = float(np.sum(dots1 - A1))
                            pa0 = float(np.sum(dots0 - A0))
                            prob1 = expit(log_prior_ratio + (pa1 - pa0))
                            new_bit = np.random.binomial(1, prob1)
                            if new_bit != A_cur[i, k]:
                                A_cur[i, k] = new_bit
                                z_i = z1 if new_bit == 1 else z0
                                idx_cur[i] = idx1 if new_bit == 1 else idx0

                    A_sample_long[:, :, c] = np.hstack((ones_col, A_cur))
                    uniq, cnts = np.unique(idx_cur, return_counts=True)
                    for k_idx, v in zip(uniq.tolist(), cnts.tolist()):
                        counts_hat[k_idx] = counts_hat.get(int(k_idx), 0.0) + float(v)

                c0, t0 = 0.1, 10
                step = c0 / (t + t0)
                if self.C > 1:
                    for k_idx in list(counts_hat.keys()):
                        counts_hat[k_idx] /= float(self.C)
                if counts_smooth:
                    for k_idx in list(counts_smooth.keys()):
                        counts_smooth[k_idx] *= (1.0 - step)
                for k_idx, v in counts_hat.items():
                    counts_smooth[k_idx] = counts_smooth.get(k_idx, 0.0) + step * float(v)

                A_new = A_cur.copy()
                for j in range(self.J):
                    Xj = self.X[:, j].copy()
                    f_loglik = self.F_1_SAEM(Xj, A_sample_long)

                    def update_f_old_1(x, f_old=f_old_1[j], f_new=f_loglik):
                        return (1 - step) * f_old(x) + step * f_new(x)

                    f_old_1[j] = update_f_old_1

                    def f_j(beta_gamma):
                        return f_old_1[j](beta_gamma) + self.ftn_pen(beta_gamma)

                    opt_result = minimize(f_j, self.B_hat[j, :].flatten(), bounds=list(zip(lb, ub)), method="SLSQP", options=options)
                    B_update[j, :] = opt_result.x

                err = np.linalg.norm(self.B_hat - np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], self.tau)), axis=1), "fro") ** 2
                self.B_hat = np.concatenate((B_update[:, [0]], thres(B_update[:, 1:], self.tau)), axis=1)
                t += 1
                print(f"SAEM Iteration {t}, Err {err:.5f}")
                iter_indicator = (abs(err) > self.tol and t < self.max_iter)

            self._saem_finalize_p(counts_smooth)
            return self.p_hat, self.B_hat, self.A_hat, t, loglik

        raise ValueError(f"Unsupported distribution: {self.distribution}")

    def init(self, ite):
        self.generate_latent_data()
        self.generate_data()
        return initialize_parameters(self.X, self.J, self.K, self.distribution, self.epsilon)

    def estimate(self, ite, truth_graph):
        if self.distribution == "Poisson":
            if self.algorithm == "PSAEM":
                self.p_hat, self.B_hat, self.A_hat, _, _ = self.PSAEM(ite)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        elif self.distribution == "Lognormal":
            if self.algorithm == "PSAEM":
                self.p_hat, self.B_hat, self.gamma_hat, self.A_hat, _, _ = self.PSAEM(ite)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")

        elif self.distribution == "Bernoulli":
            if self.algorithm == "PSAEM":
                self.p_hat, self.B_hat, self.A_hat, _, _ = self.PSAEM(ite)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
        else:
            raise ValueError(f"Unsupported distribution: {self.distribution}")

        return compute_shd_triplet(
            p_hat=self.p_hat,
            B_hat=self.B_hat,
            K=self.K,
            J=self.J,
            Q_N=self.Q_N,
            truth_graph=truth_graph,
            Q_truth=self.Q,
        )


DAG_Estimator = DAGEstimator
