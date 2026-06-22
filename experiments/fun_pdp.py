"""Fun #2 (the detailed one): WHAT DOES THE MODEL LOVE?

Permutation importance (which physics inputs the model leans on) + partial dependence
(the SHAPE of what it rewards). A preview of the eventual NLA explanation, model-free.

    uv run --with matplotlib python experiments/fun_pdp.py
"""
import datetime as dt

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from whyplus.model.data import FULL_START, RAW_COLS, load_statcast  # noqa: E402
from whyplus.model.features import INPUT_COLS, add_fastball_context, mirror_lhp  # noqa: E402
from xrv import MLP  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIRRORED = {"vx0", "ax", "release_pos_x", "spin_axis"}


def load_model(path="artifacts/xrv_model.pt"):
    ck = torch.load(path, weights_only=True)
    m = MLP(len(ck["input_cols"])).to(DEVICE)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, np.array(ck["scaler_mean"], "float32"), np.array(ck["scaler_scale"], "float32")


@torch.no_grad()
def predict_std(m, Xs, bs=500_000):
    return np.concatenate([m(torch.from_numpy(Xs[i:i+bs]).to(DEVICE)).cpu().numpy()
                           for i in range(0, len(Xs), bs)])


df = add_fastball_context(mirror_lhp(load_statcast(FULL_START, dt.date.today().isoformat())))
df = df[df[INPUT_COLS].notna().all(axis=1)].reset_index(drop=True)
m, mean, scale = load_model()
Xr = df[INPUT_COLS].to_numpy("float32")
rng = np.random.default_rng(0)
idx = rng.choice(len(Xr), 150_000, replace=False)
Xrs = Xr[idx]
Xstd = ((Xrs - mean) / scale).astype("float32")
base = predict_std(m, Xstd)

# --- permutation importance (mean |Δ prediction| when a feature is shuffled) ---
imp = []
for j, c in enumerate(INPUT_COLS):
    Xp = Xstd.copy(); Xp[:, j] = Xstd[rng.permutation(len(Xp)), j]
    imp.append((c, float(np.mean(np.abs(predict_std(m, Xp) - base)))))
imp.sort(key=lambda t: -t[1])
print("=== permutation importance (mean |Δ predicted stuff| when shuffled) ===")
for c, v in imp:
    tag = " [pitch]" if c in RAW_COLS else " [fb-context]"
    print(f"  {c:24s} {v:.4f}{tag}")

# --- partial dependence for the 12 raw pitch-physics inputs ---
fig, axes = plt.subplots(3, 4, figsize=(17, 10))
for ax, c in zip(axes.flat, RAW_COLS):
    j = INPUT_COLS.index(c)
    lo, hi = np.percentile(Xrs[:, j], [2, 98])
    grid = np.linspace(lo, hi, 22)
    pdp = []
    for v in grid:
        Xt = Xrs.copy(); Xt[:, j] = v
        pdp.append(predict_std(m, ((Xt - mean) / scale).astype("float32")).mean())
    ax.plot(grid, pdp, lw=2.2, color="#3b6fb0")
    ax.set_title(c + (" (RHP-mirrored)" if c in MIRRORED else ""), fontsize=10)
    ax.grid(alpha=0.25)
fig.suptitle("What the model loves — partial dependence of stuff on each raw physics input\n"
             "(y = predicted stuff value, higher = nastier; rug = 2nd–98th pctile range)",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("figures/what_the_model_loves.png", dpi=120)
print("\nsaved figures/what_the_model_loves.png")
