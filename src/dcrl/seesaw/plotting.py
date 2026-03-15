import os
import math

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from scipy.special import expit


def save_effect_heatmaps(
    B_hat: np.ndarray,
    output_dir: str,
    grid_hw: int | None = None,
    prefix: str = "effect",
):
    os.makedirs(output_dir, exist_ok=True)

    J, K1 = B_hat.shape
    if grid_hw is None:
        grid_hw = int(round(math.sqrt(J)))
    if grid_hw * grid_hw != J:
        raise ValueError(
            f"J={J} is not a perfect square, so it cannot be reshaped to a square heatmap."
        )

    heatmaps = []
    for k in range(K1):
        lv = np.zeros(K1, dtype=float)
        lv[k] = 1.0
        heatmaps.append(expit(B_hat @ lv))

    vals = np.concatenate(heatmaps)
    vmin, vmax = float(vals.min()), float(vals.max())

    for k, hm in enumerate(heatmaps):
        plt.figure(figsize=(4, 4))
        plt.imshow(
            hm.reshape(grid_hw, grid_hw),
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar()

        if k == 0:
            title = "Intercept"
            fname = "intercept.png"
        else:
            title = f"Effect of Z{k}"
            fname = f"{prefix}_Z{k}.png"

        plt.title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=160)
        plt.close()
