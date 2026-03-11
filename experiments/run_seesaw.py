import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.GraphUtils import GraphUtils

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dcrl.utils import binary
from dcrl.seesaw.dataset import make_dataset
from dcrl.seesaw.psaem import psaem_seesaw
from dcrl.seesaw.plotting import save_effect_heatmaps, save_probability_heatmaps


def sample_latents_from_p(p_hat: np.ndarray, N: int, K: int) -> np.ndarray:
    sam = np.zeros((N, K), dtype=np.uint8)
    qwe = p_hat.flatten().astype(float)
    qwe = qwe / qwe.sum()

    counts = np.random.multinomial(N, qwe)
    n = 0
    A_src = binary(np.arange(2**K), K)

    for a in range(2**K):
        cnt = int(counts[a])
        if cnt > 0:
            sam[n : n + cnt, :K] = np.tile(A_src[a, :K], (cnt, 1))
            n += cnt
    return sam


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--N", type=int, default=10000)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--orig_hw", type=int, default=256)
    parser.add_argument("--mask_hw", type=int, default=96)
    parser.add_argument("--pool_hw", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bank_size", type=int, default=2000)
    parser.add_argument("--jitter", type=float, default=0.001)
    parser.add_argument("--ball_thresh", type=int, default=80)

    parser.add_argument("--noisy_z3", action="store_true")
    parser.add_argument("--p_up_if11", type=float, default=0.8)
    parser.add_argument("--p_up_else", type=float, default=0.2)
    parser.add_argument("--p_ball4_if_up", type=float, default=0.99)
    parser.add_argument("--p_ball4_if_down", type=float, default=0.0)
    parser.add_argument("--pooling", type=str, default="min", choices=["min", "max"])

    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--L_final", type=float, default=0.0)
    parser.add_argument("--pen", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)

    parser.add_argument("--results_dir", type=str, default="results/seesaw")

    args = parser.parse_args()

    np.random.seed(args.seed)

    if args.pen is None:
        args.pen = args.N ** (1 / 4)
    if args.tau is None:
        args.tau = args.N ** (-7 / 32)

    os.makedirs(args.results_dir, exist_ok=True)

    X_gray256, X_balls96, Y, Z = make_dataset(
        N=args.N,
        orig_hw=args.orig_hw,
        mask_hw=args.mask_hw,
        pool_hw=args.pool_hw,
        seed=args.seed,
        bank_size=args.bank_size,
        jitter=args.jitter,
        ball_thresh=args.ball_thresh,
        noisy_z3=args.noisy_z3,
        p_up_if11=args.p_up_if11,
        p_up_else=args.p_up_else,
        p_ball4_if_up=args.p_ball4_if_up,
        p_ball4_if_down=args.p_ball4_if_down,
        pooling=args.pooling,
    )

    dataset_path = os.path.join(args.results_dir, "seesaw_dataset.npz")
    np.savez_compressed(
        dataset_path,
        X_gray256=X_gray256,
        X_balls96=X_balls96,
        Y=Y,
        Z=Z,
        orig_hw=np.array([args.orig_hw], dtype=np.int32),
        mask_hw=np.array([args.mask_hw], dtype=np.int32),
        pool_hw=np.array([args.pool_hw], dtype=np.int32),
    )

    p_hat, B_hat, A_hat, t, loglik = psaem_seesaw(
        X=Y.astype(float),
        K=args.K,
        C=args.C,
        tol=args.tol,
        max_iter=args.max_iter,
        pen=args.pen,
        tau=args.tau,
        epsilon=args.epsilon,
        kappa=args.kappa,
        L_final=args.L_final,
        verbose=True,
    )

    np.save(os.path.join(args.results_dir, "p_hat.npy"), p_hat)
    np.save(os.path.join(args.results_dir, "B_hat.npy"), B_hat)
    np.save(os.path.join(args.results_dir, "A_hat_init.npy"), A_hat)

    save_effect_heatmaps(
        B_hat,
        output_dir=os.path.join(args.results_dir, "heatmaps"),
        grid_hw=args.pool_hw,
    )
    save_probability_heatmaps(
        B_hat,
        output_dir=os.path.join(args.results_dir, "probability_maps"),
        grid_hw=args.pool_hw,
    )

    sam = sample_latents_from_p(p_hat, N=args.N, K=args.K)
    np.save(os.path.join(args.results_dir, "latent_samples_from_phat.npy"), sam)

    record_est = ges(sam, score_func="local_score_BDeu")
    G_hat = record_est["G"].graph
    np.save(os.path.join(args.results_dir, "latent_graph_estimated.npy"), G_hat)

    pyd_est = GraphUtils.to_pydot(record_est["G"])
    pyd_est.write_png(os.path.join(args.results_dir, "latent_graph_estimated.png"))

    summary = {
        "N": args.N,
        "K": args.K,
        "J": int(Y.shape[1]),
        "iterations": int(t),
        "loglik": float(loglik),
        "dataset_path": dataset_path,
        "results_dir": args.results_dir,
    }
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved dataset to:", dataset_path)
    print("Saved p_hat to:", os.path.join(args.results_dir, "p_hat.npy"))
    print("Saved B_hat to:", os.path.join(args.results_dir, "B_hat.npy"))
    print("Saved estimated graph to:", os.path.join(args.results_dir, "latent_graph_estimated.png"))
    print("Saved summary to:", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
