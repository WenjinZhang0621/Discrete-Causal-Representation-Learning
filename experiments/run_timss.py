import argparse
import json
import os
import sys
from pathlib import Path

# Set before NumPy is imported; explicit user thread settings are respected.
for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(key, "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcrl.timss.refit import fit
from dcrl.timss.output import file_hash, save_graph


def main():
    parser = argparse.ArgumentParser(description="Fit raw TIMSS data, then use multinomial latent resampling.")
    parser.add_argument("--x_path", default=str(ROOT / "data/TIMSS/time.csv"))
    parser.add_argument("--q_path", default=str(ROOT / "data/TIMSS/Q.csv"))
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--seed", type=int, default=48, help="Measurement initialization seed")
    parser.add_argument("--graph_seed", type=int, default=1)
    parser.add_argument("--fn_ratio", type=float, default=0.2)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=1e-7)
    parser.add_argument("--results_dir", default="results/timss")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()
    x = np.genfromtxt(args.x_path, delimiter=",")
    remove = np.isnan(x).all(axis=1)
    x = np.clip(x[~remove], np.e, np.e**6)
    q = np.genfromtxt(args.q_path, delimiter=",")[:, 1:]
    if q.shape[1] != args.K:
        raise ValueError("K differs from the supplied Q matrix")
    result = fit(x, q, seed=args.seed, max_iter=args.max_iter, tol=args.tol)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    for key, name in [("p", "p_hat.npy"), ("beta", "B_hat.npy"), ("gamma", "gamma_hat.npy")]:
        value = result[key]
        np.save(out / name, value.reshape(-1, 1) if key in ["p", "gamma"] else value)
    np.savez_compressed(out / "fit.npz", **result)
    metadata = dict(fit_source="raw student data", seed=args.seed, graph_seed=args.graph_seed,
                    J=x.shape[1], iterations=int(result["iterations"]), converged=result["converged"],
                    loglik=float(result["loglik"]), likelihood="Gaussian on clipped log times, including normalizing constant",
                    max_iter=args.max_iter, tol=args.tol, x_path=args.x_path, q_path=args.q_path,
                    x_sha256=file_hash(args.x_path), q_sha256=file_hash(args.q_path),
                    n_removed_all_nan_rows=int(remove.sum()), clip_lower=float(np.e), clip_upper=float(np.e**6))
    summary = save_graph(result["p"], len(x), args.K, args.fn_ratio, args.graph_seed,
                         out, metadata=metadata, plot=not args.no_plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
