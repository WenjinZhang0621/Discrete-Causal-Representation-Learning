import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dcrl.timss.output import file_hash, save_graph


def main():
    parser = argparse.ArgumentParser(description="Reproduce TIMSS using multinomial latent resampling.")
    parser.add_argument("--p_path", default=str(ROOT / "data/TIMSS/timss_refit.json"))
    parser.add_argument("--K", type=int, default=7)
    parser.add_argument("--N", type=int, default=620, help="Number of original students")
    parser.add_argument("--fn_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1, help="Graph-sampling seed")
    parser.add_argument("--results_dir", default="results/timss_saved_p")
    parser.add_argument("--no_plot", action="store_true")
    args = parser.parse_args()
    path = Path(args.p_path)
    if path.suffix == ".json":
        saved = json.loads(path.read_text())
        p = np.asarray(saved["p_est"], dtype=float)
        if saved.get("N", args.N) != args.N or saved.get("K", args.K) != args.K:
            raise ValueError("Saved fit N/K differ from the requested N/K")
    elif path.suffix == ".npy":
        p = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as saved:
            key = "p_est" if "p_est" in saved else "p"
            p = np.asarray(saved[key], dtype=float)
    summary = save_graph(p, args.N, args.K, args.fn_ratio, args.seed, args.results_dir,
                         metadata={"p_path": str(path), "p_file_sha256": file_hash(path),
                                   "fit_source": "saved latent distribution"}, plot=not args.no_plot)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
