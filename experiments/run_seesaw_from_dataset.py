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
from dcrl.seesaw.psaem import psaem_seesaw
from dcrl.seesaw.plotting import save_effect_heatmaps


def sample_latents_from_p(p_hat: np.ndarray, N: int, K: int) -> np.ndarray:
    sam = np.zeros((N, K), dtype=np.uint8)
    counts = np.random.multinomial(N, p_hat.flatten())
    A_src = binary(np.arange(2**K), K)

    n = 0
    for a in range(2**K):
        cnt = int(counts[a])
        if cnt > 0:
            sam[n:n + cnt, :K] = np.tile(A_src[a, :K], (cnt, 1))
            n += cnt
    return sam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/seesaw/seesaw_dataset.npz",
        help="Path to an existing seesaw .npz dataset.",
    )
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)

    parser.add_argument("--C", type=int, default=1)
    parser.add_argument("--tol", type=float, default=0.1)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--kappa", type=float, default=0.1)
    parser.add_argument("--L_final", type=float, default=0.0)
    parser.add_argument("--pen", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)

    parser.add_argument("--results_dir", type=str, default="results/seesaw_from_dataset")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.results_dir, exist_ok=True)

    dataset_path = ROOT / args.dataset_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = np.load(dataset_path)

    if "Y" not in data:
        raise KeyError(
            f"'Y' not found in {dataset_path}. Available keys: {list(data.keys())}"
        )

    Y = data["Y"]
    N = int(Y.shape[0])
    J = int(Y.shape[1])

    if "pool_hw" in data:
        pool_hw = int(np.asarray(data["pool_hw"]).reshape(-1)[0])
    else:
        rootJ = int(round(np.sqrt(J)))
        pool_hw = rootJ if rootJ * rootJ == J else None

    if args.pen is None:
        args.pen = N ** (1 / 4)
    if args.tau is None:
        args.tau = N ** (-7 / 32)

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

    if pool_hw is not None:
        save_effect_heatmaps(
            B_hat,
            output_dir=os.path.join(args.results_dir, "heatmaps"),
            grid_hw=pool_hw,
        )

    np.random.seed(args.seed)
    sam = sample_latents_from_p(p_hat, N=N, K=args.K)
    np.save(os.path.join(args.results_dir, "latent_samples_from_phat.npy"), sam)

    record_est = ges(sam, score_func="local_score_BDeu")
    G_hat = record_est["G"].graph
    np.save(os.path.join(args.results_dir, "latent_graph_estimated.npy"), G_hat)

    pyd_est = GraphUtils.to_pydot(record_est["G"])
    pyd_est.write_png(os.path.join(args.results_dir, "latent_graph_estimated.png"))

    summary = {
        "N": N,
        "K": args.K,
        "J": J,
        "iterations": int(t),
        "loglik": float(loglik),
        "dataset_path": str(dataset_path),
        "results_dir": args.results_dir,
        "used_existing_dataset": True,
    }
    if pool_hw is not None:
        summary["pool_hw"] = int(pool_hw)

    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Loaded dataset from:", dataset_path)
    print("Saved p_hat to:", os.path.join(args.results_dir, "p_hat.npy"))
    print("Saved B_hat to:", os.path.join(args.results_dir, "B_hat.npy"))
    print("Saved estimated graph to:", os.path.join(args.results_dir, "latent_graph_estimated.png"))
    print("Saved summary to:", os.path.join(args.results_dir, "summary.json"))


if __name__ == "__main__":
    main()
