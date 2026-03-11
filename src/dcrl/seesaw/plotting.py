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
    include_intercept: bool = False,
    prefix: str = "effect",
):
    """
    Save heatmaps for the intercept and/or latent effects.

    B_hat has shape (J, K+1).
    Column 0 is the intercept and columns 1..K are latent effects.
    """
    os.makedirs(output_dir, exist_ok=True)
    J, K1 = B_hat.shape

    if grid_hw is None:
        grid_hw = int(round(math.sqrt(J)))
    if grid_hw * grid_hw != J:
        raise ValueError(f"J={J} is not a perfect square, so it cannot be reshaped to a square heatmap.")

    start_col = 0 if include_intercept else 1
    for col in range(start_col, K1):
        hm = B_hat[:, col].reshape(grid_hw, grid_hw)

        plt.figure(figsize=(4, 4))
        plt.imshow(hm, cmap="gray", interpolation="nearest")
        plt.colorbar()
        title = "Intercept" if col == 0 else f"Effect of Z{col}"
        plt.title(title)
        plt.tight_layout()
        fname = "intercept.png" if col == 0 else f"{prefix}_Z{col}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=160)
        plt.close()


def save_probability_heatmaps(
    B_hat: np.ndarray,
    output_dir: str,
    grid_hw: int | None = None,
    prefix: str = "probability",
):
    """
    Save sigmoid(intercept + one latent effect) heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    J, K1 = B_hat.shape
    K = K1 - 1

    if grid_hw is None:
        grid_hw = int(round(math.sqrt(J)))
    if grid_hw * grid_hw != J:
        raise ValueError(f"J={J} is not a perfect square, so it cannot be reshaped to a square heatmap.")

    vals = []
    maps = []
    for k in range(K):
        eta = B_hat[:, 0] + B_hat[:, k + 1]
        prob = expit(eta)
        maps.append(prob.reshape(grid_hw, grid_hw))
        vals.append(prob)

    vals = np.concatenate(vals)
    vmin, vmax = float(vals.min()), float(vals.max())

    for k, hm in enumerate(maps, start=1):
        plt.figure(figsize=(4, 4))
        plt.imshow(hm, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f"P(Y=1 | Z{k}=1)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{prefix}_Z{k}.png"), dpi=160)
        plt.close()
