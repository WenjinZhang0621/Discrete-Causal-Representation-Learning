import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcrl.utils import binary


def infer_K_from_p(p_est: np.ndarray) -> int:
    m = int(np.asarray(p_est).size)
    k = int(round(np.log2(m)))
    if 2 ** k != m:
        raise ValueError(
            f"Length of p_est is {m}, which is not a power of 2, so K cannot be inferred."
        )
    return k


def sample_latents_from_p(p_est: np.ndarray, N: int, K: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    p = np.asarray(p_est, dtype=float).reshape(-1)
    p = np.clip(p, 0.0, None)
    if p.sum() <= 0:
        raise ValueError("p_est has nonpositive total mass.")
    p = p / p.sum()

    if p.size != 2 ** K:
        raise ValueError(f"p_est has length {p.size}, but K={K} implies length {2**K}.")

    counts = rng.multinomial(N, p)
    A_src = binary(np.arange(2 ** K), K)

    sam = np.zeros((N, K), dtype=np.uint8)
    n = 0
    for a in range(2 ** K):
        cnt = int(counts[a])
        if cnt > 0:
            sam[n:n + cnt, :K] = np.tile(A_src[a, :K], (cnt, 1))
            n += cnt

    return sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mat_path",
        type=str,
        default="data/TIMSS/timss_matlab_estimates.mat",
        help="Path to the MATLAB estimates file produced by matlab/timss/main_data.m",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/timss",
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=620,
        help="Number of latent samples drawn from p_est for causal-learn",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=7,
        help="Number of latent variables. If omitted, infer from beta_est or p_est.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for latent sampling",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    data = scipy.io.loadmat(args.mat_path)

    if "p_est" not in data:
        raise KeyError(f"'p_est' not found in {args.mat_path}")
    if "beta_est" not in data:
        raise KeyError(f"'beta_est' not found in {args.mat_path}")
    if "gamma_est" not in data:
        raise KeyError(f"'gamma_est' not found in {args.mat_path}")

    p_est = np.asarray(data["p_est"], dtype=float)
    beta_est = np.asarray(data["beta_est"], dtype=float)
    gamma_est = np.asarray(data["gamma_est"], dtype=float)

    if args.K is None:
        if beta_est.ndim == 2 and beta_est.shape[1] >= 2:
            K = int(beta_est.shape[1] - 1)
            pK = infer_K_from_p(p_est)
            if K != pK:
                raise ValueError(
                    f"Inferred K mismatch: beta_est implies K={K}, but p_est implies K={pK}."
                )
        else:
            K = infer_K_from_p(p_est)
    else:
        K = int(args.K)
        pK = infer_K_from_p(p_est)
        if K != pK:
            raise ValueError(f"--K={K}, but p_est implies K={pK}.")
        if beta_est.ndim == 2 and beta_est.shape[1] != K + 1:
            raise ValueError(
                f"--K={K}, but beta_est has shape {beta_est.shape}, expected second dim {K+1}."
            )

    p_vec = p_est.reshape(-1, 1)

    np.save(os.path.join(args.results_dir, "p_est.npy"), p_vec)
    np.save(os.path.join(args.results_dir, "beta_est.npy"), beta_est)
    np.save(os.path.join(args.results_dir, "gamma_est.npy"), gamma_est)

    np.save(os.path.join(args.results_dir, "p_hat.npy"), p_vec)
    np.save(os.path.join(args.results_dir, "B_hat.npy"), beta_est)
    np.save(os.path.join(args.results_dir, "gamma_hat.npy"), gamma_est)

    sam = sample_latents_from_p(p_vec, N=args.N, K=K, seed=args.seed)
    np.save(os.path.join(args.results_dir, "latent_samples_from_pest.npy"), sam)

    record_est = ges(sam, score_func="local_score_BDeu")
    G_hat = record_est["G"].graph
    np.save(os.path.join(args.results_dir, "latent_graph_estimated.npy"), G_hat)

    labels = [f"Z{i+1}" for i in range(K)]
    pyd_est = GraphUtils.to_pydot(record_est["G"], labels=labels)
    pyd_est.write_png(os.path.join(args.results_dir, "latent_graph_estimated.png"))

    best_cc = None
    if "best_cc" in data:
        try:
            best_cc = int(np.asarray(data["best_cc"]).reshape(-1)[0])
        except Exception:
            best_cc = None

    loglik = None
    if "loglik" in data:
        try:
            loglik_arr = np.asarray(data["loglik"], dtype=float).reshape(-1)
            if best_cc is not None and 1 <= best_cc <= loglik_arr.size:
                loglik = float(loglik_arr[best_cc - 1])  # MATLAB is 1-based
            else:
                loglik = float(np.max(loglik_arr))
        except Exception:
            loglik = None

    summary = {
        "mat_path": args.mat_path,
        "results_dir": args.results_dir,
        "N_sampled_for_graph": int(args.N),
        "K": int(K),
        "J": int(beta_est.shape[0]),
        "seed": int(args.seed),
        "best_cc": best_cc,
        "selected_loglik": loglik,
        "p_est_shape": list(p_vec.shape),
        "beta_est_shape": list(beta_est.shape),
        "gamma_est_shape": list(gamma_est.shape),
    }

    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Loaded MATLAB estimates from:", args.mat_path)
    print("Saved p_est to:", os.path.join(args.results_dir, "p_est.npy"))
    print("Saved beta_est to:", os.path.join(args.results_dir, "beta_est.npy"))
    print("Saved gamma_est to:", os.path.join(args.results_dir, "gamma_est.npy"))
    print("Saved latent samples to:", os.path.join(args.results_dir, "latent_samples_from_pest.npy"))
    print("Saved estimated graph PNG to:", os.path.join(args.results_dir, "latent_graph_estimated.png"))
    print("Saved estimated graph matrix to:", os.path.join(args.results_dir, "latent_graph_estimated.npy"))
    print("Saved summary to:", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
