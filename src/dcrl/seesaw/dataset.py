import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.transforms import Affine2D
from PIL import Image

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


def minpool_to_hw(bin_img: np.ndarray, out_hw: int) -> np.ndarray:
    H, W = bin_img.shape
    if H != W:
        raise ValueError("Input image must be square.")
    H2 = (H // out_hw) * out_hw
    x = bin_img[:H2, :H2]
    bh = H2 // out_hw
    bw = H2 // out_hw
    return (
        x.reshape(out_hw, bh, out_hw, bw)
        .transpose(0, 2, 1, 3)
        .min(axis=(2, 3))
        .astype(np.uint8)
    )


def maxpool_to_hw(bin_img: np.ndarray, out_hw: int) -> np.ndarray:
    H, W = bin_img.shape
    if H != W:
        raise ValueError("Input image must be square.")
    H2 = (H // out_hw) * out_hw
    x = bin_img[:H2, :H2]
    bh = H2 // out_hw
    bw = H2 // out_hw
    return (
        x.reshape(out_hw, bh, out_hw, bw)
        .transpose(0, 2, 1, 3)
        .max(axis=(2, 3))
        .astype(np.uint8)
    )


def render_seesaw_rgb(
    z1: int,
    z2: int,
    z3: int,
    z4: int,
    seed: int,
    render_hw: int = 256,
    jitter: float = 0.004,
    angle_deg: float = 25.0,
    rise_lift: float = 0.04,
    ball_r: float = 0.055,
    plank_w: float = 0.82,
    plank_h: float = 0.040,
    structure_gray: float = 0.72,
    tray_x_center: float = 0.74,
    tray_total_w: float = 0.66,
    tray_gap: float = 0.26,
    arm_h: float = 0.024,
    tray_ball_sep: float = 0.38,
    tray_clear: float = 0.010,
    tray_y_fixed: float = 0.6,
    ball4_r: float = 0.050,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    def jxy() -> float:
        return float(rng.uniform(-jitter, jitter)) if jitter > 0 else 0.0

    dpi = 100
    fig = plt.figure(figsize=(render_hw / dpi, render_hw / dpi), dpi=dpi)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none"))

    g = (structure_gray, structure_gray, structure_gray)

    pivot = np.array([0.50 + jxy(), 0.30 + jxy()], dtype=float)

    ax.add_patch(
        Polygon(
            [[pivot[0] - 0.04, 0.10], [pivot[0] + 0.04, 0.10], [pivot[0], pivot[1] - 0.02]],
            closed=True,
            facecolor=g,
            edgecolor="none",
            antialiased=False,
        )
    )

    plank_center = pivot + np.array([0.00 + jxy(), 0.07 + jxy()], dtype=float)
    clearance = 0.008

    def left_ball_center_for_angle(ang_deg: float) -> np.ndarray:
        theta = np.deg2rad(ang_deg)
        u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        n = np.array([-np.sin(theta), np.cos(theta)], dtype=float)
        left_end = plank_center - u * (plank_w / 2 - 0.05)
        center = left_end + n * (plank_h / 2 + ball_r + clearance)
        return center

    ang = -angle_deg if z3 == 1 else angle_deg
    theta = np.deg2rad(ang)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    n = np.array([-np.sin(theta), np.cos(theta)], dtype=float)

    left_end = plank_center - u * (plank_w / 2 - 0.05)

    plank = Rectangle(
        (plank_center[0] - plank_w / 2, plank_center[1] - plank_h / 2),
        plank_w,
        plank_h,
        facecolor=g,
        edgecolor="none",
        antialiased=False,
    )
    plank.set_transform(
        Affine2D().rotate_deg_around(plank_center[0], plank_center[1], ang) + ax.transData
    )
    ax.add_patch(plank)

    left_ball_center = left_end + n * (plank_h / 2 + ball_r + clearance)

    ang_down = +angle_deg
    ball4_center = left_ball_center_for_angle(ang_down)

    delta = np.array([jxy(), jxy()], dtype=float)
    left_ball_center = left_ball_center + delta
    ball4_center = ball4_center + delta

    if z3 == 1:
        left_ball_center = left_ball_center + np.array([0.0, rise_lift], dtype=float)

    left_ball_center[0] = np.clip(left_ball_center[0], ball_r + 0.02, 1.0 - ball_r - 0.02)
    left_ball_center[1] = np.clip(left_ball_center[1], ball_r + 0.02, 1.0 - ball_r - 0.02)
    ball4_center[0] = np.clip(ball4_center[0], ball4_r + 0.02, 1.0 - ball4_r - 0.02)
    ball4_center[1] = np.clip(ball4_center[1], ball4_r + 0.02, 1.0 - ball4_r - 0.02)

    if z4 == 1:
        ax.add_patch(
            Circle(
                (float(ball4_center[0]), float(ball4_center[1])),
                radius=ball4_r,
                facecolor="black",
                edgecolor="black",
                linewidth=0.0,
                antialiased=False,
            )
        )

    ax.add_patch(
        Circle(
            (float(left_ball_center[0]), float(left_ball_center[1])),
            radius=ball_r,
            facecolor="black",
            edgecolor="black",
            linewidth=0.0,
            antialiased=False,
        )
    )

    tray_x = float(np.clip(tray_x_center, 0.20, 0.92))
    tray_y = float(np.clip(tray_y_fixed, 0.15, 0.88))

    stem_w = 0.018
    stem_h = 0.35
    ax.add_patch(
        Rectangle(
            (tray_x - stem_w / 2, tray_y - stem_h),
            stem_w,
            stem_h,
            facecolor=(0.85, 0.85, 0.85),
            edgecolor="none",
            antialiased=False,
        )
    )

    arm_w = (tray_total_w - tray_gap) / 2
    left_arm_x = tray_x - tray_total_w / 2
    right_arm_x = tray_x + tray_gap / 2

    ax.add_patch(Rectangle((left_arm_x, tray_y), arm_w, arm_h, facecolor=g, edgecolor="none", antialiased=False))
    ax.add_patch(Rectangle((right_arm_x, tray_y), arm_w, arm_h, facecolor=g, edgecolor="none", antialiased=False))

    x1 = tray_x - tray_ball_sep / 2
    x2 = tray_x + tray_ball_sep / 2
    x1 = float(np.clip(x1, ball_r + 0.02, 1.0 - ball_r - 0.02))
    x2 = float(np.clip(x2, ball_r + 0.02, 1.0 - ball_r - 0.02))
    ball_y = tray_y + arm_h + ball_r + tray_clear
    ball_y = float(np.clip(ball_y, ball_r + 0.02, 1.0 - ball_r - 0.02))

    if z1 == 1:
        ax.add_patch(
            Circle((x1, ball_y), radius=ball_r, facecolor="black", edgecolor="black", linewidth=0.0, antialiased=False)
        )
    if z2 == 1:
        ax.add_patch(
            Circle((x2, ball_y), radius=ball_r, facecolor="black", edgecolor="black", linewidth=0.0, antialiased=False)
        )

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return rgb


def make_dataset(
    N: int = 10_000,
    orig_hw: int = 256,
    mask_hw: int = 96,
    pool_hw: int = 12,
    seed: int = 123,
    bank_size: int = 500,
    jitter: float = 0.004,
    ball_thresh: int = 80,
    noisy_z3: bool = True,
    p_up_if11: float = 0.75,
    p_up_else: float = 0.25,
    p_ball4_if_up: float = 0.95,
    p_ball4_if_down: float = 0.02,
    pooling: str = "min",
):
    rng = np.random.default_rng(seed)

    z1 = rng.integers(0, 2, size=N, dtype=np.uint8)
    z2 = rng.integers(0, 2, size=N, dtype=np.uint8)

    if not noisy_z3:
        z3 = (z1 & z2).astype(np.uint8)
    else:
        p = np.where((z1 == 1) & (z2 == 1), p_up_if11, p_up_else)
        z3 = (rng.random(N) < p).astype(np.uint8)

    p4 = np.where(z3 == 1, p_ball4_if_up, p_ball4_if_down)
    z4 = (rng.random(N) < p4).astype(np.uint8)

    Z = np.stack([z1, z2, z3, z4], axis=1)

    Jin_orig = orig_hw * orig_hw
    Jin_mask = mask_hw * mask_hw
    Jout = pool_hw * pool_hw

    X_gray256 = np.empty((N, Jin_orig), dtype=np.uint8)
    X_balls96 = np.empty((N, Jin_mask), dtype=np.uint8)
    Y = np.empty((N, Jout), dtype=np.uint8)

    bank_keys = []
    for a in (0, 1):
        for b in (0, 1):
            cs = [int(a & b)] if not noisy_z3 else [0, 1]
            for c in cs:
                if c == 0:
                    bank_keys.append((a, b, 0, 0))
                else:
                    bank_keys.append((a, b, 1, 0))
                    bank_keys.append((a, b, 1, 1))
    bank_keys = list({tuple(x) for x in bank_keys})

    pool_fn = minpool_to_hw if pooling.lower() == "min" else maxpool_to_hw
    banks = {}

    for a, b, c, d_bank in bank_keys:
        bank_gray = np.empty((bank_size, Jin_orig), dtype=np.uint8)
        bank_balls = np.empty((bank_size, Jin_mask), dtype=np.uint8)
        bank_y = np.empty((bank_size, Jout), dtype=np.uint8)

        for t in range(bank_size):
            rgb = render_seesaw_rgb(
                a,
                b,
                c,
                d_bank,
                seed=int(rng.integers(0, 10**9)),
                render_hw=orig_hw,
                jitter=jitter,
                angle_deg=25.0,
                rise_lift=0.04,
                ball_r=0.055,
                ball4_r=0.050,
                tray_ball_sep=0.38,
                tray_total_w=0.66,
                tray_gap=0.26,
                tray_x_center=0.74,
                tray_y_fixed=0.6,
            )

            gray_orig = np.array(Image.fromarray(rgb, mode="RGB").convert("L"), dtype=np.uint8)

            balls_hi = (gray_orig > ball_thresh).astype(np.uint8)
            balls96 = np.array(
                Image.fromarray(balls_hi * 255).resize((mask_hw, mask_hw), RESAMPLE_NEAREST),
                dtype=np.uint8,
            ) // 255

            pooled = pool_fn(balls96, out_hw=pool_hw)

            bank_gray[t] = gray_orig.reshape(-1)
            bank_balls[t] = balls96.reshape(-1)
            bank_y[t] = pooled.reshape(-1)

        banks[(a, b, c, d_bank)] = (bank_gray, bank_balls, bank_y)

    filled = np.zeros(N, dtype=bool)

    for a in (0, 1):
        for b in (0, 1):
            idx0 = np.where((z1 == a) & (z2 == b) & (z3 == 0))[0]
            if idx0.size > 0:
                bank_gray, bank_balls, bank_y = banks[(a, b, 0, 0)]
                choices0 = rng.integers(0, bank_size, size=idx0.size)
                X_gray256[idx0] = bank_gray[choices0]
                X_balls96[idx0] = bank_balls[choices0]
                Y[idx0] = bank_y[choices0]
                filled[idx0] = True

            for d in (0, 1):
                idx1 = np.where((z1 == a) & (z2 == b) & (z3 == 1) & (z4 == d))[0]
                if idx1.size == 0:
                    continue
                bank_gray, bank_balls, bank_y = banks[(a, b, 1, d)]
                choices1 = rng.integers(0, bank_size, size=idx1.size)
                X_gray256[idx1] = bank_gray[choices1]
                X_balls96[idx1] = bank_balls[choices1]
                Y[idx1] = bank_y[choices1]
                filled[idx1] = True

    if not filled.all():
        bad = np.where(~filled)[0]
        raise RuntimeError(f"Unfilled rows exist: {bad.size}. First few: {bad[:10].tolist()}")

    return X_gray256, X_balls96, Y, Z
